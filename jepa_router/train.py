"""Train + evaluate routers. Usage:

  python -m jepa_router.train --source synthetic --embedder tfidf              # offline smoke test
  python -m jepa_router.train --source routerbench --path routerbench_0shot.pkl \
         --tasks grade-school-math --embedder sbert:all-MiniLM-L6-v2 --epochs 60

Compared routers (all see the same frozen query embedding, same head size):
  random-mix   : randomly send fraction t of queries to the best model, rest to cheapest
  raw-head     : MLP on query embedding trained with BCE only      (BERT-classifier-style baseline)
  jepa-joint   : JEPA latent-prediction loss + BCE head, trained together
  jepa-2stage  : pretrain with JEPA loss only, freeze, then fit a linear head
  oracle       : upper bound (cheapest model that is correct)
  routellm-bert: (--routellm) RouteLLM's released BERT router, strong = best model, weak = cheapest
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from .data import load_routerbench, make_synthetic, split, inspect_routerbench
from .embed import get_embedder
from .model import JEPARouter, RawHeadRouter, jepa_loss, variance_loss
from .routing import route_curve, auc_on_grid, cost_at_frac, baselines


def target_view_index(score, cost):
    """Which model's response is the JEPA target for each query: best score, ties -> cheapest."""
    key = score * 1e6 - cost / (cost.max() + 1e-12)
    return key.argmax(1)


def routellm_curve(prompts, ckpt, score, cost, strong, weak):
    """RouteLLM BERTRouter (github.com/lm-sys/RouteLLM): P(strong wins) = 1 - softmax[-2:].sum().
    Send to the strong model iff it beats a threshold; sweep thresholds over its quantiles."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ckpt)
    m = AutoModelForSequenceClassification.from_pretrained(ckpt).to(dev).eval()
    w = []
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            b = tok(prompts[i : i + 32], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            w.append(1 - torch.softmax(m(**b).logits, -1)[:, -2:].sum(-1).cpu().numpy())
    w, idx = np.concatenate(w), np.arange(len(prompts))
    pts = []
    for t in np.append(np.quantile(w, np.linspace(0, 1, 41)), np.inf):
        ch = np.where(w >= t, strong, weak)
        pts.append((cost[idx, ch].mean(), score[idx, ch].mean()))
    return np.array(pts)


def build_tensors(train, test, embedder_spec):
    emb = get_embedder(embedder_spec, fit_texts=train.prompts + [r for rs in train.responses for r in rs])
    Xtr, Xte = emb.encode(train.prompts), emb.encode(test.prompts)
    sel = target_view_index(train.score, train.cost)
    resp_tr = [train.responses[m][i] for i, m in enumerate(sel)]
    Ytr = emb.encode(resp_tr)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return t(Xtr), t(Ytr), t(Xte), t(train.score), emb


def fit(model, Xtr, Ytr, Str, mode, epochs, lr=1e-3, bs=128, beta=1.0, gamma=1.0, seed=0, log=print):
    """mode: raw | joint | jepa_only"""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = {"jepa": 0.0, "bce": 0.0, "var": 0.0}
        for i in range(0, n, bs):
            b = perm[i : i + bs]
            out = model(Xtr[b], Ytr[b] if mode != "raw" else None)
            loss = 0.0
            if mode in ("raw", "joint"):
                bce = F.binary_cross_entropy_with_logits(out["logits"], Str[b])
                loss = loss + beta * bce
                tot["bce"] += bce.item() * len(b)
            if mode in ("joint", "jepa_only"):
                jl, vl = jepa_loss(out), variance_loss(out["z"], 1.0)
                loss = loss + jl + gamma * vl
                tot["jepa"] += jl.item() * len(b)
                tot["var"] += vl.item() * len(b)
            opt.zero_grad(); loss.backward(); opt.step()
            if hasattr(model, "update_target") and mode != "raw":
                model.update_target()
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            log(f"  [{mode}] ep {ep:3d} " + " ".join(f"{k}={v / n:.4f}" for k, v in tot.items() if v))
    return model


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.sigmoid(model(X)["logits"]).numpy()


def fit_linear_head_on_frozen(model, Xtr, Str, epochs=100, lr=1e-2):
    """Stage 2 of jepa-2stage: freeze encoder, train only the head with BCE."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    with torch.no_grad():
        Z = model.embed(Xtr)
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(model.head(Z), Str)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "routerbench"])
    ap.add_argument("--path")
    ap.add_argument("--tasks", nargs="*", default=["grade-school-math"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--max_n", type=int, default=None)
    ap.add_argument("--embedder", default="tfidf")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--d_lat", type=int, default=64)
    ap.add_argument("--holdout_task", default=None, help="leave-one-task-out OOD test")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train_frac", type=float, default=1.0, help="per-seed random subset of train (data-scarcity curve)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--routellm", default=None, help="HF checkpoint, e.g. routellm/bert_gpt4_augmented")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    if a.inspect:
        return inspect_routerbench(a.path)
    os.makedirs(a.out, exist_ok=True)

    data = make_synthetic() if a.source == "synthetic" else load_routerbench(a.path, a.tasks or None, a.models, a.max_n)
    print(f"{len(data)} queries, models={data.models}, mean score per model={np.round(data.score.mean(0), 3).tolist()}")
    tr, te = split(data, holdout_task=a.holdout_task)
    Xall, Yall, Xte, Sall, _ = build_tensors(tr, te, a.embedder)
    Xtr = Xall
    M, d_in = len(data.models), Xtr.shape[1]

    single, rand, oracle, best_m = baselines(te.score, te.cost)
    c_lo, c_hi = min(c for c, _ in single.values()), max(c for c, _ in single.values())
    best_q = te.score.mean(0)[best_m]
    res = {"random-mix": {"auc": auc_on_grid(rand, c_lo, c_hi), "cost@95%": float(cost_at_frac(rand, best_q))},
           "oracle": {"cost": float(oracle[0]), "quality": float(oracle[1])},
           "best-single": {"model": data.models[best_m], "cost": float(single[best_m][0]), "quality": float(best_q)}}
    curves = {}
    if a.routellm:
        curve = routellm_curve(te.prompts, a.routellm, te.score, te.cost, best_m, int(np.argmin(te.cost.mean(0))))
        curves["routellm-bert"] = curve.tolist()
        res["routellm-bert"] = {"auc": auc_on_grid(curve, c_lo, c_hi), "cost@95%": float(cost_at_frac(curve, best_q))}

    for name in ["raw-head", "jepa-joint", "jepa-2stage"]:
        aucs, c95 = [], []
        for s in range(a.seeds):
            # same subset for every router at a given seed -> paired comparison
            idx = torch.from_numpy(np.random.default_rng(s).permutation(len(Xall))[: max(1, int(a.train_frac * len(Xall)))])
            Xtr, Ytr, Str = Xall[idx], Yall[idx], Sall[idx]
            if name == "raw-head":
                m = fit(RawHeadRouter(d_in, M, a.d_lat), Xtr, Ytr, Str, "raw", a.epochs, seed=s)
            elif name == "jepa-joint":
                m = fit(JEPARouter(d_in, M, a.d_lat), Xtr, Ytr, Str, "joint", a.epochs, seed=s)
            else:
                m = fit(JEPARouter(d_in, M, a.d_lat), Xtr, Ytr, Str, "jepa_only", a.epochs, seed=s)
                m = fit_linear_head_on_frozen(m, Xtr, Str)
            curve = route_curve(predict(m, Xte), te.score, te.cost)
            curves[f"{name}_s{s}"] = curve.tolist()
            aucs.append(auc_on_grid(curve, c_lo, c_hi)); c95.append(cost_at_frac(curve, best_q))
        res[name] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "aucs": aucs,
                     "cost@95%_mean": float(np.nanmean(c95)),  # only over seeds that reach 95%:
                     "cost@95%_reached": f"{int(np.sum(~np.isnan(c95)))}/{a.seeds}"}

    json.dump({"results": res, "curves": curves}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print("\n=== results (held-out queries) ===")
    for k, v in res.items():
        print(f"{k:12s} {json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})}")


if __name__ == "__main__":
    main()
