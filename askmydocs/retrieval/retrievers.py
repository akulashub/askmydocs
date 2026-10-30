"""Vector, BM25 and hybrid (Reciprocal Rank Fusion) retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from ..embeddings import Embedder
from ..ingest.chunker import Chunk
from ..store import ChromaVectorStore
from .text import tokenize


@dataclass
class Candidate:
    chunk: Chunk
    score: float                                   # score from the stage that produced it
    scores: dict[str, float] = field(default_factory=dict)   # per-stage scores for debugging
    ranks: dict[str, int] = field(default_factory=dict)


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks], k1=k1, b=b)

    def search(self, query: str, k: int) -> list[Candidate]:
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [Candidate(self.chunks[i], float(scores[i]), {"bm25": float(scores[i])}, {"bm25": r})
                for r, i in enumerate(order, start=1) if scores[i] > 0]


class VectorRetriever:
    def __init__(self, store: ChromaVectorStore, embedder: Embedder, chunks_by_id: dict[str, Chunk]):
        self.store = store
        self.embedder = embedder
        self.chunks_by_id = chunks_by_id

    def search(self, query: str, k: int) -> list[Candidate]:
        hits = self.store.query(self.embedder.embed_query(query), k)
        return [Candidate(self.chunks_by_id[cid], s, {"vector": s}, {"vector": r})
                for r, (cid, s) in enumerate(hits, start=1) if cid in self.chunks_by_id]


def reciprocal_rank_fusion(result_lists: dict[str, list[Candidate]], k: int = 60,
                           weights: dict[str, float] | None = None) -> list[Candidate]:
    """RRF: score(d) = sum_i w_i / (k + rank_i(d)). Rank-based, so it needs no score
    calibration between BM25 (unbounded) and cosine similarity (bounded)."""
    weights = weights or {}
    fused: dict[str, Candidate] = {}
    for name, results in result_lists.items():
        w = weights.get(name, 1.0)
        for rank, cand in enumerate(results, start=1):
            cid = cand.chunk.chunk_id
            if cid not in fused:
                fused[cid] = Candidate(cand.chunk, 0.0)
            fused[cid].score += w / (k + rank)
            fused[cid].scores.update(cand.scores)
            fused[cid].ranks[name] = rank
    out = sorted(fused.values(), key=lambda c: c.score, reverse=True)
    for c in out:
        c.scores["rrf"] = c.score
    return out


class HybridRetriever:
    def __init__(self, vector: VectorRetriever | None, bm25: BM25Retriever | None, cfg: dict):
        self.vector = vector
        self.bm25 = bm25
        self.cfg = cfg

    def search(self, query: str) -> list[Candidate]:
        mode = self.cfg.get("mode", "hybrid")
        lists: dict[str, list[Candidate]] = {}
        if mode in ("vector", "hybrid") and self.vector:
            lists["vector"] = self.vector.search(query, self.cfg["vector_k"])
        if mode in ("bm25", "hybrid") and self.bm25:
            lists["bm25"] = self.bm25.search(query, self.cfg["bm25_k"])
        if len(lists) == 1:
            return next(iter(lists.values()))[: self.cfg["fused_k"]]
        return reciprocal_rank_fusion(lists, k=self.cfg.get("rrf_k", 60))[: self.cfg["fused_k"]]
