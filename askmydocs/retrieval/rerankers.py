"""Second-stage re-ranking.

A bi-encoder embeds query and chunk independently; a cross-encoder reads them
*together* and scores the pair, which is far more precise but too slow to run over
the whole corpus. So we retrieve ~20 candidates cheaply, then re-rank to the top 5.
"""

from __future__ import annotations

import math
import os
from typing import Protocol

from .retrievers import Candidate
from .text import tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]: ...


def _finish(cands: list[Candidate], scores: list[float], top_n: int, key: str) -> list[Candidate]:
    for c, s in zip(cands, scores):
        c.scores[key] = float(s)
        c.score = float(s)
    return sorted(cands, key=lambda c: c.score, reverse=True)[:top_n]


class CrossEncoderReranker:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder  # pip install .[models]

        self.model = CrossEncoder(model, max_length=512)
        self.name = f"cross-encoder:{model}"

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        if not candidates:
            return []
        raw = [float(x) for x in self.model.predict([(query, c.chunk.embed_text) for c in candidates])]
        # Depending on the sentence-transformers version / model config, predict() returns
        # raw logits or sigmoid probabilities. Normalise to probabilities so the relevance
        # gate (guardrails.min_rerank_score) has one meaning across versions.
        if any(x < 0 or x > 1 for x in raw):
            raw = [1 / (1 + math.exp(-x)) for x in raw]
        return _finish(candidates, raw, top_n, "rerank")


class CohereReranker:
    def __init__(self, model: str):
        import cohere  # pip install .[cohere]

        self.client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        self.model = model
        self.name = f"cohere:{model}"

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        if not candidates:
            return []
        resp = self.client.rerank(model=self.model, query=query,
                                  documents=[c.chunk.embed_text for c in candidates], top_n=len(candidates))
        scores = [0.0] * len(candidates)
        for r in resp.results:
            scores[r.index] = r.relevance_score
        return _finish(candidates, scores, top_n, "rerank")


class LexicalReranker:
    """Offline stand-in for a cross-encoder: scores each (query, chunk) pair jointly
    on IDF-weighted query-term coverage, exact phrase/identifier matches, and section
    title match. Output is in [0, 1], so a fixed relevance gate is meaningful."""

    name = "lexical-v1"

    def __init__(self, idf: dict[str, float] | None = None):
        self.idf = idf or {}

    def _w(self, term: str) -> float:
        return self.idf.get(term, 1.0)

    def score(self, query: str, cand: Candidate) -> float:
        q_terms = set(tokenize(query))
        if not q_terms:
            return 0.0
        body = set(tokenize(cand.chunk.text))
        head = set(tokenize(f"{cand.chunk.title} {cand.chunk.section}"))
        total = sum(self._w(t) for t in q_terms)
        coverage = sum(self._w(t) for t in q_terms if t in body) / total
        head_cov = sum(self._w(t) for t in q_terms if t in head) / total
        q_low, t_low = query.lower(), cand.chunk.text.lower()
        exact = [w for w in q_low.replace("?", " ").split() if ("_" in w or "." in w) and len(w) > 3]
        exact_hit = (sum(w.strip("`'\".,") in t_low for w in exact) / len(exact)) if exact else 0.0
        prior = 1 / (1 + math.exp(-cand.scores.get("rrf", 0) * 60 + 1))   # keep a bit of first-stage signal
        return 0.6 * coverage + 0.2 * head_cov + 0.1 * exact_hit + 0.1 * prior

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        return _finish(candidates, [self.score(query, c) for c in candidates], top_n, "rerank")


class NoopReranker:
    name = "none"

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        return candidates[:top_n]


def build_reranker(cfg: dict, idf: dict[str, float] | None = None) -> Reranker:
    backend = cfg["backend"]
    if backend == "cross-encoder":
        return CrossEncoderReranker(cfg["model"])
    if backend == "cohere":
        return CohereReranker(cfg.get("cohere_model", "rerank-english-v3.0"))
    if backend == "lexical":
        return LexicalReranker(idf)
    if backend == "none":
        return NoopReranker()
    raise ValueError(f"Unknown reranker backend: {backend}")
