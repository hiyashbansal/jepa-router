"""JEPA-style router model.

Two views of the same underlying 'problem':
    x = the query                       -> context encoder   -> z_x
    y = a good solution to the query    -> target encoder    -> z_y   (EMA copy, no gradient)
Predictor tries to map z_x to z_y, in latent space (never in token space).

Intuition: to predict what a good solution "looks like", z_x must encode what kind of
problem it is and how hard it is. That is exactly what a router needs.

A small head on z_x predicts, per candidate LLM, P(model solves the query).
Collapse guard: variance term keeps z_x dimensions from shrinking to a constant (VICReg-style).
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(d_in, d_hid, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Linear(d_hid, d_out))


class JEPARouter(nn.Module):
    def __init__(self, d_in, n_models, d_lat=64, d_hid=256, ema=0.99):
        super().__init__()
        self.ctx = mlp(d_in, d_hid, d_lat)  # query -> latent
        self.tgt = copy.deepcopy(self.ctx)  # same encoder, EMA weights, applied to the response view
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.pred = mlp(d_lat, d_hid, d_lat)
        self.head = nn.Linear(d_lat, n_models)
        self.ema = ema

    @torch.no_grad()
    def update_target(self):
        # as in I-JEPA: target encoder = exponential moving average of the context encoder
        for pt, pc in zip(self.tgt.parameters(), self.ctx.parameters()):
            pt.mul_(self.ema).add_(pc.detach(), alpha=1 - self.ema)

    def embed(self, x):
        return self.ctx(x)

    def forward(self, xq, yr=None):
        z = self.ctx(xq)
        out = {"z": z, "logits": self.head(z)}
        if yr is not None:
            with torch.no_grad():
                zt = self.tgt(yr)
            out["pred"] = self.pred(z)
            out["zt"] = zt
        return out


def jepa_loss(out):
    p = F.normalize(out["pred"], dim=-1)
    t = F.normalize(out["zt"], dim=-1)
    return (2 - 2 * (p * t).sum(-1)).mean()  # cosine distance in latent space


def variance_loss(z, gamma=1.0):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(gamma - std).mean()


class RawHeadRouter(nn.Module):
    """Baseline: same capacity, NO JEPA objective. MLP on frozen embedding -> P(success) per model.
    This is the 'BERT-classifier-style' router (RouteLLM uses a BERT classifier on the query)."""

    def __init__(self, d_in, n_models, d_lat=64, d_hid=256):
        super().__init__()
        self.enc = mlp(d_in, d_hid, d_lat)
        self.head = nn.Linear(d_lat, n_models)

    def embed(self, x):
        return self.enc(x)

    def forward(self, xq, yr=None):
        z = self.enc(xq)
        return {"z": z, "logits": self.head(z)}
