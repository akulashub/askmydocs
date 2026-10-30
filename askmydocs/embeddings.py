"""Embedding backends behind one interface.

- sentence-transformers: local neural model (default BAAI/bge-small-en-v1.5)
- openai: hosted embeddings
- hashing: deterministic, dependency-free, offline. Signed feature hashing of word
  unigrams/bigrams and character trigrams with sublinear TF. Not semantic, but
  stable across machines, which makes it ideal for unit tests and keyless CI.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-v1-{dim}"

    def _features(self, text: str) -> dict[str, float]:
        text = text.lower()
        words = _WORD.findall(text)
        feats: dict[str, float] = {}
        for w in words:
            feats["w:" + w] = feats.get("w:" + w, 0) + 1.0
        for a, b in zip(words, words[1:]):
            k = f"b:{a}_{b}"
            feats[k] = feats.get(k, 0) + 0.5
        for w in words:
            padded = f"#{w}#"
            for i in range(len(padded) - 2):
                k = "c:" + padded[i:i + 3]
                feats[k] = feats.get(k, 0) + 0.2
        return feats

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for feat, tf in self._features(text).items():
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += (1.0 if (h >> 63) & 1 else -1.0) * (1.0 + math.log(tf + 1e-9) if tf > 1 else tf)
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class SentenceTransformerEmbedder:
    def __init__(self, model: str, query_instruction: str = "", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # pip install .[models]

        self.model = SentenceTransformer(model)
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.name = f"st:{model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 200).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([self.query_instruction + text], normalize_embeddings=True)[0].tolist()


class OpenAIEmbedder:
    def __init__(self, model: str, batch_size: int = 64):
        from openai import OpenAI  # pip install .[openai]

        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size
        self.name = f"openai:{model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + self.batch_size])
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedder(cfg: dict) -> Embedder:
    backend = cfg["backend"]
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg["model"], cfg.get("query_instruction", ""),
                                           cfg.get("batch_size", 32))
    if backend == "openai":
        return OpenAIEmbedder(cfg["model"])
    raise ValueError(f"Unknown embeddings backend: {backend}")
