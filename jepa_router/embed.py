"""Frozen text embedders. The JEPA sits ON TOP of these (v0 keeps things cheap).

  tfidf            : offline, sklearn TF-IDF + SVD (smoke tests only)
  sbert:<name>     : sentence-transformers, e.g. sbert:all-MiniLM-L6-v2 (default real choice)
  bert:<name>      : mean-pooled HF encoder, e.g. bert:bert-base-uncased (closest to RouteLLM's BERT)
"""
from __future__ import annotations
import numpy as np


class Embedder:
    def fit(self, texts):  # only tfidf needs this
        return self

    def encode(self, texts) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    def __init__(self, dim=64):
        self.dim = dim

    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(min(self.dim, X.shape[1] - 1), random_state=0).fit(X)
        return self

    def encode(self, texts):
        Z = self.svd.transform(self.vec.transform(texts))
        return (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


class SBertEmbedder(Embedder):
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(name)

    def encode(self, texts):
        return self.m.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)


class HFMeanPoolEmbedder(Embedder):
    def __init__(self, name):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.tok, self.m = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m.to(self.dev)

    def encode(self, texts):
        import torch

        out = []
        texts = list(texts)
        for i in range(0, len(texts), 32):
            b = self.tok(texts[i : i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.dev)
            with torch.no_grad():
                h = self.m(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1)
            e = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def get_embedder(spec: str, fit_texts=None) -> Embedder:
    kind, _, name = spec.partition(":")
    if kind == "tfidf":
        return TfidfEmbedder().fit(fit_texts)
    if kind == "sbert":
        return SBertEmbedder(name or "all-MiniLM-L6-v2")
    if kind == "bert":
        return HFMeanPoolEmbedder(name or "bert-base-uncased")
    raise ValueError(spec)
