"""Routing rule + evaluation.

Rule: pick argmax over models of  P(success | query) - lam * normalized_cost.
Sweeping lam traces a cost-vs-quality curve. We compare curves, not single points.

Metrics (all on held-out queries):
  - curve: list of (avg_cost, avg_score)
  - auc: area under the quality-vs-cost curve on a shared cost grid (higher = better)
  - cost_at_frac: cheapest avg cost that reaches X% of the best single model's quality
"""
from __future__ import annotations
import numpy as np


def route_curve(p_success, score, cost, lams=None):
    """p_success[N,M] predicted; score/cost[N,M] ground truth. Returns array [K,2] of (cost, quality)."""
    cn = cost / (cost.max() + 1e-12)
    lams = np.concatenate([[0], np.geomspace(1e-3, 10, 60)]) if lams is None else lams
    pts = []
    idx = np.arange(len(score))
    for lam in lams:
        choice = (p_success - lam * cn).argmax(1)
        pts.append((cost[idx, choice].mean(), score[idx, choice].mean()))
    return np.array(pts)


def pareto(points):
    pts = points[np.argsort(points[:, 0])]
    best, out = -1, []
    for c, q in pts:
        if q > best:
            out.append((c, q))
            best = q
    return np.array(out)


def auc_on_grid(curve, c_lo, c_hi, n=100):
    """Mean quality of the (linearly interpolated) Pareto frontier over a shared cost range."""
    p = pareto(curve)
    grid = np.linspace(c_lo, c_hi, n)
    q = np.interp(grid, p[:, 0], p[:, 1])  # clamps outside the curve's own range
    return float(np.trapezoid(q, grid) / (c_hi - c_lo))


def cost_at_frac(curve, best_quality, frac=0.95):
    p = pareto(curve)
    ok = p[p[:, 1] >= frac * best_quality]
    return ok[0, 0] if len(ok) else np.nan


def baselines(score, cost):
    """Reference curves: each single model, random mix of cheapest and best, oracle."""
    N, M = score.shape
    single = {m: (cost[:, m].mean(), score[:, m].mean()) for m in range(M)}
    best_m = int(np.argmax(score.mean(0)))
    cheap_m = int(np.argmin(cost.mean(0)))
    ts = np.linspace(0, 1, 21)
    rand = np.array([[(1 - t) * cost[:, cheap_m].mean() + t * cost[:, best_m].mean(),
                      (1 - t) * score[:, cheap_m].mean() + t * score[:, best_m].mean()] for t in ts])
    # oracle: cheapest model among those that succeed (else cheapest overall)
    big = 1e9
    masked = np.where(score >= 0.5, cost, big)
    ch = masked.argmin(1)
    none = (masked.min(1) >= big)
    ch[none] = cost[none].argmin(1)
    idx = np.arange(N)
    oracle = (cost[idx, ch].mean(), score[idx, ch].mean())
    return single, rand, oracle, best_m
