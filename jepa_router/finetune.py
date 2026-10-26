"""End-to-end fine-tuned text encoder (train.py keeps it frozen). GPU job. Usage:

  python -m jepa_router.finetune --path routerbench_0shot.pkl --tasks grade-school-math mbpp --holdout_task mbpp

Compared routers (same encoder init, same head):
  ft-raw  : encoder + head, BCE only          (RouteLLM-style fine-tuned BERT classifier)
  ft-jepa : encoder + head, BCE + JEPA loss   (predict the EMA target encoder's latent of the best response)
"""
from __future__ import annotations
import argparse, copy, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .data import load_routerbench, make_synthetic, split
from .model import mlp, jepa_loss, variance_loss
from .routing import route_curve, auc_on_grid, cost_at_frac, baselines
from .train import target_view_index


class TextRouter(nn.Module):
    def __init__(self, name, n_models, d_lat=64, ema=0.996):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.proj = mlp(self.enc.config.hidden_size, 256, d_lat)
        self.head = nn.Linear(d_lat, n_models)
        self.pred = mlp(d_lat, 256, d_lat)
        self.tgt = copy.deepcopy(nn.ModuleList([self.enc, self.proj]))  # EMA target, applied to responses
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.ema = ema

    @staticmethod
    def _z(enc, proj, b):
        h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1)
        return proj((h * m).sum(1) / m.sum(1))

    def forward(self, bq, br=None):
        z = self._z(self.enc, self.proj, bq)
        out = {"z": z, "logits": self.head(z)}
        if br is not None:
            with torch.no_grad():
                out["zt"] = self._z(self.tgt[0], self.tgt[1], br)
            out["pred"] = self.pred(z)
        return out

    @torch.no_grad()
    def update_target(self):
        for pt, pc in zip(self.tgt.parameters(), list(self.enc.parameters()) + list(self.proj.parameters())):
            pt.mul_(self.ema).add_(pc.detach(), alpha=1 - self.ema)


def fit_predict(mode, tr, te, a, seed, dev):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(a.encoder)
    T = lambda xs: tok(list(xs), padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to(dev)
    m = TextRouter(a.encoder, len(tr.models)).to(dev).train()
    heads = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": m.enc.parameters(), "lr": a.lr}, {"params": heads, "lr": 1e-3}], weight_decay=1e-4)
    S = torch.tensor(tr.score, dtype=torch.float32, device=dev)
    sel = target_view_index(tr.score, tr.cost)
    resp = [tr.responses[k][i] for i, k in enumerate(sel)]
    rng = np.random.default_rng(seed)
    for ep in range(a.epochs):
        for b in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // a.bs)):
            out = m(T(tr.prompts[i] for i in b), T(resp[i] for i in b) if mode == "jepa" else None)
            loss = F.binary_cross_entropy_with_logits(out["logits"], S[torch.as_tensor(b, device=dev)])
            if mode == "jepa":
                loss = loss + jepa_loss(out) + variance_loss(out["z"])
            opt.zero_grad(); loss.backward(); opt.step()
            if mode == "jepa":
                m.update_target()
        print(f"  [ft-{mode} s{seed}] ep {ep} loss={loss.item():.4f}", flush=True)
    m.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(m(T(te.prompts[i : i + 64]))["logits"]).cpu().numpy()
                               for i in range(0, len(te), 64)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="routerbench", choices=["synthetic", "routerbench"])
    ap.add_argument("--path")
    ap.add_argument("--tasks", nargs="*", default=["grade-school-math"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--max_n", type=int, default=None)
    ap.add_argument("--holdout_task", default=None)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = make_synthetic(a.max_n or 4000) if a.source == "synthetic" else load_routerbench(a.path, a.tasks or None, a.models, a.max_n)
    print(f"{len(data)} queries, device={dev}")
    tr, te = split(data, holdout_task=a.holdout_task)
    single, rand, oracle, best_m = baselines(te.score, te.cost)
    c_lo, c_hi = min(c for c, _ in single.values()), max(c for c, _ in single.values())
    best_q = te.score.mean(0)[best_m]
    res = {"random-mix": {"auc": auc_on_grid(rand, c_lo, c_hi)},
           "oracle": {"cost": float(oracle[0]), "quality": float(oracle[1])},
           "best-single": {"model": data.models[best_m], "cost": float(single[best_m][0]), "quality": float(best_q)}}
    for mode in ["raw", "jepa"]:
        aucs, c95 = [], []
        for s in range(a.seeds):
            curve = route_curve(fit_predict(mode, tr, te, a, s, dev), te.score, te.cost)
            aucs.append(auc_on_grid(curve, c_lo, c_hi)); c95.append(cost_at_frac(curve, best_q))
        res[f"ft-{mode}"] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "aucs": aucs,
                             "cost@95%_mean": float(np.nanmean(c95)),
                             "cost@95%_reached": f"{int(np.sum(~np.isnan(c95)))}/{a.seeds}"}

    json.dump({"results": res}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print("\n=== results (held-out queries) ===")
    for k, v in res.items():
        print(f"{k:12s} {json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})}")


if __name__ == "__main__":
    main()
