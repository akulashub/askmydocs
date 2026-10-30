"""Persistence: Chroma for vectors, a JSONL chunk store as the source of truth
(BM25 index, citation rendering, and eval all read from it), plus a manifest that
records exactly how the index was built so stale indexes are detected."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .ingest.chunker import Chunk

MANIFEST = "manifest.json"
CHUNKS = "chunks.jsonl"


@dataclass
class IndexPaths:
    root: Path

    @property
    def chroma(self) -> Path:
        return self.root / "chroma"

    @property
    def chunks(self) -> Path:
        return self.root / CHUNKS

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST


def write_chunks(paths: IndexPaths, chunks: list[Chunk]) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    with paths.chunks.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def read_chunks(paths: IndexPaths) -> list[Chunk]:
    if not paths.chunks.exists():
        raise FileNotFoundError(f"No index at {paths.root}. Run `askmydocs ingest` first.")
    with paths.chunks.open(encoding="utf-8") as fh:
        return [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]


def write_manifest(paths: IndexPaths, manifest: dict) -> None:
    paths.manifest.write_text(json.dumps(manifest, indent=2))


def read_manifest(paths: IndexPaths) -> dict:
    return json.loads(paths.manifest.read_text()) if paths.manifest.exists() else {}


class ChromaVectorStore:
    """Thin wrapper; embeddings are computed by our Embedder so the same vectors are
    used regardless of Chroma's own default embedding function."""

    def __init__(self, paths: IndexPaths, collection: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.client = chromadb.PersistentClient(
            path=str(paths.chroma), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection_name = collection

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

    @property
    def collection(self):
        return self.client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def add(self, chunks: list[Chunk], vectors: list[list[float]], batch: int = 256) -> None:
        col = self.collection
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            col.add(
                ids=[c.chunk_id for c in part],
                embeddings=vectors[i:i + batch],
                documents=[c.text for c in part],
                metadatas=[{"doc_id": c.doc_id, "section": c.section, "anchor": c.anchor,
                            "line_start": c.line_start, "line_end": c.line_end} for c in part],
            )

    def query(self, vector: list[float], k: int, where: dict | None = None) -> list[tuple[str, float]]:
        res = self.collection.query(query_embeddings=[vector], n_results=k, where=where,
                                    include=["distances"])
        ids, dists = res["ids"][0], res["distances"][0]
        return [(i, 1.0 - d) for i, d in zip(ids, dists)]  # cosine similarity

    def count(self) -> int:
        return self.collection.count()
