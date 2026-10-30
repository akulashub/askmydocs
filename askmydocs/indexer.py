"""Ingestion: load -> chunk -> embed -> persist (Chroma + chunk store + manifest)."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from pathlib import Path

from .config import Settings
from .embeddings import build_embedder
from .ingest.chunker import Chunk, chunk_corpus
from .ingest.loaders import load_corpus
from .retrieval.text import tokenize
from .store import ChromaVectorStore, IndexPaths, read_chunks, read_manifest, write_chunks, write_manifest
from .tokens import tokenizer_name


def index_paths(settings: Settings) -> IndexPaths:
    """One index per embedding model, so switching profiles never mixes vectors."""
    slug = re.sub(r"[^\w.-]+", "_", f"{settings.embeddings['backend']}-{settings.embeddings['model']}")
    return IndexPaths(settings.resolve_path(settings.index["dir"]) / slug)


def corpus_fingerprint(roots: list[Path], include: list[str]) -> str:
    h = hashlib.sha256()
    for root in roots:
        for pattern in include:
            for p in sorted(root.glob(pattern)):
                if p.is_file():
                    h.update(p.relative_to(root).as_posix().encode())
                    h.update(p.read_bytes())
    return h.hexdigest()[:16]


class IdfTable(dict):
    """term -> IDF. A term never seen in the corpus is the *most* informative kind of
    term (e.g. "sentry" in a question about these docs), so lookups for unseen terms
    return the maximum IDF instead of a neutral default. Without this, out-of-scope
    questions look well covered and the system answers them."""

    def __init__(self, data: dict[str, float]):
        super().__init__(data)
        self.unseen = max(data.values()) if data else 1.0

    def get(self, key, default=None):  # noqa: D401 - dict API
        return super().get(key, self.unseen)


def pipeline_fingerprint() -> str:
    """Hash of the code that determines chunk boundaries and vectors. Changing the
    chunker or tokenizer logic must invalidate the index, not just changing config."""
    pkg = Path(__file__).resolve().parent
    files = sorted((pkg / "ingest").glob("*.py")) + [pkg / "tokens.py", pkg / "embeddings.py"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def compute_idf(chunks: list[Chunk]) -> IdfTable:
    df: Counter[str] = Counter()
    for c in chunks:
        df.update(set(tokenize(c.embed_text)))
    n = len(chunks)
    return IdfTable({t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()})


def build_index(settings: Settings, log=print) -> dict:
    t0 = time.time()
    roots = [settings.resolve_path(p) for p in settings.corpus["paths"]]
    docs = load_corpus(roots, settings.corpus["include"])
    if not docs:
        raise SystemExit(f"No documents found under {roots}")
    ch_cfg = settings.chunking
    chunks = chunk_corpus(docs, target_tokens=ch_cfg["target_tokens"], max_tokens=ch_cfg["max_tokens"],
                          min_tokens=ch_cfg["min_tokens"], overlap_tokens=ch_cfg["overlap_tokens"],
                          section_break_tokens=ch_cfg.get("section_break_tokens", 450))
    log(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")

    paths = index_paths(settings)
    embedder = build_embedder(settings.embeddings)
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    store = ChromaVectorStore(paths, settings.index["collection"])
    store.reset()
    store.add(chunks, vectors)
    write_chunks(paths, chunks)
    sizes = sorted(c.token_count for c in chunks)
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedder": embedder.name,
        "tokenizer": tokenizer_name(),
        "chunking": dict(ch_cfg),
        "corpus_fingerprint": corpus_fingerprint(roots, settings.corpus["include"]),
        "pipeline_fingerprint": pipeline_fingerprint(),
        "documents": len(docs),
        "chunks": len(chunks),
        "chunk_tokens": {"min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]},
        "seconds": round(time.time() - t0, 2),
    }
    write_manifest(paths, manifest)
    log(f"Indexed {store.count()} vectors with {embedder.name} in {manifest['seconds']}s -> {paths.root}")
    return manifest


def ensure_index(settings: Settings, log=print) -> IndexPaths:
    """Build the index if missing or stale: corpus content, chunking config, ingestion
    code or tokenizer changed."""
    paths = index_paths(settings)
    manifest = read_manifest(paths)
    roots = [settings.resolve_path(p) for p in settings.corpus["paths"]]
    stale = (not manifest
             or manifest.get("corpus_fingerprint") != corpus_fingerprint(roots, settings.corpus["include"])
             or manifest.get("chunking") != dict(settings.chunking)
             or manifest.get("pipeline_fingerprint") != pipeline_fingerprint()
             or manifest.get("tokenizer") != tokenizer_name())
    if stale:
        log("Index missing or stale; rebuilding...")
        build_index(settings, log)
    return paths


__all__ = ["build_index", "ensure_index", "index_paths", "compute_idf", "read_chunks"]
