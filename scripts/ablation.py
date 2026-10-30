"""Retrieval ablation on the golden set: does each Phase-2 component earn its place?

  python scripts/ablation.py --profile offline
  python scripts/ablation.py --profile models      # bge + cross-encoder (downloads models)

Writes eval/reports/ablation-<profile>.md. Only retrieval is evaluated (no LLM calls),
so this is cheap enough to run on every retrieval change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from askmydocs.config import ROOT, load_settings  # noqa: E402
from askmydocs.pipeline import RAGPipeline  # noqa: E402

VARIANTS = [
    ("BM25 only", {"retrieval": {"mode": "bm25"}, "reranker": {"backend": "none"}}),
    ("Vector only", {"retrieval": {"mode": "vector"}, "reranker": {"backend": "none"}}),
    ("Hybrid (RRF)", {"retrieval": {"mode": "hybrid"}, "reranker": {"backend": "none"}}),
    ("Vector + rerank", {"retrieval": {"mode": "vector"}}),
    ("Hybrid + rerank (shipped)", {"retrieval": {"mode": "hybrid"}}),
]


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="offline")
    args = ap.parse_args()
    golden = [json.loads(x) for x in (ROOT / "eval" / "golden.jsonl").read_text().splitlines() if x.strip()]
    golden = [g for g in golden if g["answerable"]]
    rows = []
    for name, override in VARIANTS:
        settings = load_settings(profile=args.profile, overrides=override)
        rag = RAGPipeline(settings, log=lambda *a: None)
        k = settings.retrieval["rerank_top_n"]
        hits1 = hitsk = rr = 0.0
        by_tag: dict[str, list[float]] = {}
        for g in golden:
            ev = [(_norm(s["evidence"]), s["doc_id"]) for s in g["sources"]]
            ctx = rag.retrieve_only(g["question"])[:k]
            rank = next((i for i, c in enumerate(ctx, 1)
                         if any(e in _norm(c.chunk.text) and d == c.chunk.doc_id for e, d in ev)), None)
            hits1 += rank == 1
            hitsk += rank is not None
            rr += 1 / rank if rank else 0
            for t in g["tags"]:
                by_tag.setdefault(t, []).append(1.0 if rank else 0.0)
        n = len(golden)
        rows.append((name, hits1 / n, hitsk / n, rr / n,
                     sum(by_tag.get("exact-term", [0])) / max(1, len(by_tag.get("exact-term", []))),
                     sum(by_tag.get("paraphrase", [0])) / max(1, len(by_tag.get("paraphrase", [])))))
        print(f"{name:28s} hit@1={rows[-1][1]:.3f} recall@{k}={rows[-1][2]:.3f} mrr={rows[-1][3]:.3f}", file=sys.stderr)

    md = [f"## Retrieval ablation — `{args.profile}` profile ({len(golden)} answerable questions)", "",
          f"| variant | hit@1 | recall@{k} | MRR@{k} | recall: exact-term qs | recall: paraphrase qs |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} |" for r in rows]
    out = ROOT / "eval" / "reports" / f"ablation-{args.profile}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
