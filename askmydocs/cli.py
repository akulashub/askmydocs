"""Command line interface.

  askmydocs ingest                       build / rebuild the index
  askmydocs ask "How do I ...?"          answer with citations
  askmydocs retrieve "query"             inspect hybrid + re-ranked retrieval
  askmydocs eval                         run the golden-set evaluation (CI gate)
  askmydocs prompts --lock               refresh config/prompts.lock after a version bump
Add --profile offline to run without model downloads or API keys.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

from .config import load_settings


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(" ".join(text.split()), width=100, initial_indent=indent, subsequent_indent=indent)


def cmd_ingest(args) -> int:
    from .indexer import build_index

    manifest = build_index(load_settings(profile=args.profile))
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_ask(args) -> int:
    from .pipeline import RAGPipeline

    rag = RAGPipeline(load_settings(profile=args.profile))
    ans = rag.ask(args.question)
    if args.json:
        print(json.dumps(ans.to_dict(), indent=2, default=str))
        return 0
    print(f"\nQ: {ans.question}\n")
    print(_wrap(ans.answer, ""))
    if ans.refused:
        print(f"\n  (refused: {ans.refusal_reason})")
    if ans.sources:
        print("\nSources:")
        for s in ans.sources:
            print(f"  [{s.n}] {s.citation}")
            print(_wrap(f'"{s.paragraph[:400]}{"..." if len(s.paragraph) > 400 else ""}"', "      "))
    print(f"\n  trace: {ans.trace['prompt']} | {ans.trace['llm']} | {ans.trace['reranker']} | "
          f"supported={ans.supported_fraction:.0%} | {ans.trace['timings_ms']}")
    return 0


def cmd_retrieve(args) -> int:
    from .pipeline import RAGPipeline

    rag = RAGPipeline(load_settings(profile=args.profile))
    for i, c in enumerate(rag.retrieve_only(args.query), start=1):
        sc = " ".join(f"{k}={v:.3f}" for k, v in c.scores.items())
        rk = " ".join(f"{k}#{v}" for k, v in c.ranks.items())
        print(f"{i}. {c.chunk.citation()}  [{sc}] [{rk}]")
        print(_wrap(c.chunk.text[:240] + "..."))
    return 0


def cmd_eval(args) -> int:
    from .eval.runner import main as eval_main

    argv = ["--profile", args.profile] if args.profile else []
    argv += args.rest
    return eval_main(argv)


def cmd_prompts(args) -> int:
    from .prompts import LOCK_FILE, lock_snapshot

    snap = lock_snapshot()
    if args.lock:
        LOCK_FILE.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {LOCK_FILE}")
    print(json.dumps(snap, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="askmydocs", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default=None, help="settings profile: default | offline | openai")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").set_defaults(fn=cmd_ingest)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_ask)
    r = sub.add_parser("retrieve")
    r.add_argument("query")
    r.set_defaults(fn=cmd_retrieve)
    e = sub.add_parser("eval", help="options: see `python -m askmydocs.eval.runner --help`")
    e.set_defaults(fn=cmd_eval)
    pr = sub.add_parser("prompts")
    pr.add_argument("--lock", action="store_true")
    pr.set_defaults(fn=cmd_prompts)
    args, unknown = p.parse_known_args(argv)
    if unknown and args.cmd != "eval":
        p.error(f"unrecognized arguments: {' '.join(unknown)}")
    args.rest = unknown
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
