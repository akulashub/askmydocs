"""Offline evaluation over the golden set, used locally and as the CI quality gate.

  python -m askmydocs.eval.runner --profile offline
  python -m askmydocs.eval.runner --profile default --judge ragas
  python -m askmydocs.eval.runner --profile offline --update-baseline

Exit code 0 = all gates pass, 1 = a threshold or regression gate failed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from ..config import ROOT, load_settings
from ..pipeline import RAGPipeline
from .judges import RagasJudge, build_judge

GOLDEN = ROOT / "eval" / "golden.jsonl"
THRESHOLDS = ROOT / "eval" / "thresholds.yaml"
BASELINES = ROOT / "eval" / "baselines"
REPORTS = ROOT / "eval" / "reports"


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def _f1(pred: str, ref: str) -> float:
    tok = lambda s: re.findall(r"[a-z0-9_]+", s.lower())  # noqa: E731
    p, r = Counter(tok(pred)), Counter(tok(ref))
    common = sum((p & r).values())
    if not common:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return round(sum(xs) / len(xs), 4) if xs else float("nan")


def load_golden(path: Path = GOLDEN, tags: list[str] | None = None, ids: list[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if tags:
        rows = [r for r in rows if set(tags) & set(r["tags"])]
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    return rows


def evaluate_item(rag: RAGPipeline, item: dict, judge) -> dict:
    t0 = time.perf_counter()
    ans = rag.ask(item["question"])
    latency = time.perf_counter() - t0
    context = [rag.by_id[cid] for cid in ans.context_chunk_ids]
    candidates = [rag.by_id[cid] for cid in ans.trace.get("candidates", [])]
    rec = {
        "id": item["id"], "question": item["question"], "answerable": item["answerable"],
        "tags": item["tags"], "refused": ans.refused, "refusal_reason": ans.refusal_reason,
        "answer": ans.answer, "raw_answer": ans.raw_answer,
        "raw_supported_fraction": ans.supported_fraction if ans.sentences else None,
        "context": [c["citation"] for c in ans.retrieved],
        "sources": [s.citation for s in ans.sources],
        "latency_s": round(latency, 3),
        "top_rerank_score": ans.retrieved[0]["scores"].get("rerank") if ans.retrieved else None,
    }
    if item["answerable"]:
        ev = [(_norm(s["evidence"]), s["doc_id"]) for s in item["sources"]]

        def hit(chunk) -> bool:
            return any(e in _norm(chunk.text) and d == chunk.doc_id for e, d in ev)

        rank = next((i for i, c in enumerate(context, start=1) if hit(c)), None)
        rec.update({
            "hit@candidates": any(hit(c) for c in candidates),
            "hit@context": rank is not None,
            "rr@context": 1.0 / rank if rank else 0.0,
        })
        if not ans.refused:
            gold_docs = {d for _, d in ev}
            cited = [rag.by_id[s.chunk_id] for s in ans.sources]
            rec["citation_doc_correct"] = any(c.doc_id in gold_docs for c in cited)
            rec["citation_paragraph_correct"] = any(
                e in _norm(s.paragraph) and d == rag.by_id[s.chunk_id].doc_id
                for s in ans.sources for e, d in ev)
            rec["answer_f1"] = round(_f1(ans.answer, item["reference_answer"]), 4)
            if judge is not None and not isinstance(judge, RagasJudge):
                rec["faithfulness"] = judge.faithfulness(item["question"], ans.answer, context)
    rec["_ragas_row"] = {"question": item["question"], "answer": ans.answer,
                         "contexts": [c.text for c in context], "reference": item["reference_answer"]}
    return rec


def aggregate(records: list[dict]) -> dict:
    ans = [r for r in records if r["answerable"]]
    unans = [r for r in records if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    lat = sorted(r["latency_s"] for r in records)
    hard = [r for r in unans if "hard-negative" in r["tags"]]
    m = {
        "n_items": len(records), "n_answerable": len(ans), "n_unanswerable": len(unans),
        "retrieval_hit@candidates": _mean(r["hit@candidates"] for r in ans),
        "context_recall@k": _mean(r["hit@context"] for r in ans),
        "mrr@k": _mean(r["rr@context"] for r in ans),
        "answer_rate": _mean(not r["refused"] for r in ans),
        "refusal_accuracy": _mean(r["refused"] for r in unans),
        "hard_negative_refusal": _mean(r["refused"] for r in hard),
        "faithfulness": _mean(r.get("faithfulness") for r in answered),
        "judge_coverage": _mean(
            r.get("faithfulness") is not None and not math.isnan(r["faithfulness"]) for r in answered),
        "citation_doc_accuracy": _mean(r.get("citation_doc_correct") for r in answered),
        "citation_paragraph_accuracy": _mean(r.get("citation_paragraph_correct") for r in answered),
        "answer_f1": _mean(r.get("answer_f1") for r in answered),
        "raw_supported_fraction": _mean(r.get("raw_supported_fraction") for r in records),
        "latency_p50_s": round(statistics.median(lat), 3) if lat else None,
        "latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 3) if lat else None,
    }
    for key in ("ragas_faithfulness", "ragas_context_recall", "ragas_factual_correctness"):
        vals = [r.get(key) for r in answered]
        if any(v is not None for v in vals):
            m[key] = _mean(vals)
    return m


def check_gates(metrics: dict, profile: str, baseline: dict | None, judged: bool = True) -> list[str]:
    cfg = yaml.safe_load(THRESHOLDS.read_text())
    failures = []
    # A judge that silently errors (bad key, rate limit, parser failure) yields NaN
    # scores; never let that turn into a skipped faithfulness gate.
    if judged and metrics.get("judge_coverage", 1.0) < cfg.get("min_judge_coverage", 0.9):
        failures.append(f"judge_coverage={metrics.get('judge_coverage')}: faithfulness judge failed on too many items")
    for name, minimum in (cfg.get("profiles", {}).get(profile) or {}).items():
        val = metrics.get(name)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue  # metric not computed in this run (e.g. no judge)
        if name.startswith("latency"):
            if val > minimum:
                failures.append(f"{name}={val} exceeds max {minimum}")
        elif val < minimum:
            failures.append(f"{name}={val:.3f} below threshold {minimum}")
    if baseline:
        tol = cfg.get("max_regression", 0.03)
        for name in cfg.get("regression_metrics", []):
            old, new = baseline.get(name), metrics.get(name)
            if old is None or new is None or any(isinstance(x, float) and math.isnan(x) for x in (old, new)):
                continue
            if new < old - tol:
                failures.append(f"{name} regressed {old:.3f} -> {new:.3f} (tolerance {tol})")
    return failures


def render_markdown(metrics: dict, failures: list[str], records: list[dict], meta: dict) -> str:
    lines = [f"## RAG eval: `{meta['profile']}` profile — {'PASS ✅' if not failures else 'FAIL ❌'}", "",
             f"Prompt `{meta['prompt']}` · LLM `{meta['llm']}` · reranker `{meta['reranker']}` · "
             f"embedder `{meta['embedder']}` · judge `{meta['judge']}`", "",
             "| metric | value |", "|---|---|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")
    if failures:
        lines += ["", "### Gate failures", *[f"- {f}" for f in failures]]
    misses = [r for r in records if (r["answerable"] and (r["refused"] or not r.get("hit@context")))
              or (not r["answerable"] and not r["refused"])]
    if misses:
        lines += ["", f"### Items to inspect ({len(misses)})", "", "| id | question | issue |", "|---|---|---|"]
        for r in misses[:40]:
            issue = ("answered an unanswerable question" if not r["answerable"]
                     else "refused" + (f" ({r['refusal_reason']})" if r["refused"] else "")
                     if r["refused"] else "evidence not in context")
            lines.append(f"| {r['id']} | {r['question'][:70]} | {issue} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default=os.environ.get("ASKMYDOCS_PROFILE", "default"))
    p.add_argument("--judge", default="auto", choices=["auto", "ragas", "llm", "nli", "lexical", "none"])
    p.add_argument("--golden", default=str(GOLDEN))
    p.add_argument("--tags", nargs="*")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--update-baseline", action="store_true", help="write this run's metrics as the new baseline")
    p.add_argument("--no-gate", action="store_true", help="report only; always exit 0")
    p.add_argument("--allow-uncalibrated", action="store_true",
                   help="if this profile has no committed baseline yet, report but don't fail (CI bootstrap)")
    args = p.parse_args(argv)

    settings = load_settings(profile=args.profile)
    rag = RAGPipeline(settings, log=lambda *a: print(*a, file=sys.stderr))
    judge = None if args.judge == "none" else build_judge(args.judge, settings, rag.idf)
    items = load_golden(Path(args.golden), args.tags, args.ids)[: args.limit]
    print(f"Evaluating {len(items)} items | profile={args.profile} | judge={getattr(judge, 'name', 'none')}",
          file=sys.stderr)

    records = []
    for i, item in enumerate(items, start=1):
        records.append(evaluate_item(rag, item, judge))
        if i % 10 == 0:
            print(f"  {i}/{len(items)}", file=sys.stderr)

    if isinstance(judge, RagasJudge):
        idx = [i for i, r in enumerate(records) if r["answerable"] and not r["refused"]]
        scores = judge.evaluate([records[i]["_ragas_row"] for i in idx]) if idx else []
        for i, s in zip(idx, scores):
            records[i].update({f"ragas_{k}": v for k, v in s.items()})
            records[i]["faithfulness"] = s.get("faithfulness")
    for r in records:
        r.pop("_ragas_row", None)

    metrics = aggregate(records)
    baseline_path = BASELINES / f"{args.profile}.json"
    baseline = json.loads(baseline_path.read_text())["metrics"] if baseline_path.exists() else None
    failures = [] if args.update_baseline else check_gates(metrics, args.profile, baseline, judge is not None)

    meta = {"profile": args.profile, "prompt": rag.prompt.ref, "llm": rag.llm.name, "reranker": rag.reranker.name,
            "embedder": rag.manifest.get("embedder"), "judge": getattr(judge, "name", "none"),
            "golden": Path(args.golden).name, "commit": os.environ.get("GITHUB_SHA", "local"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = {"meta": meta, "metrics": metrics, "failures": failures, "records": records}
    (REPORTS / f"{stamp}-{args.profile}.json").write_text(json.dumps(report, indent=2, default=str))
    (REPORTS / f"latest-{args.profile}.json").write_text(json.dumps(report, indent=2, default=str))
    md = render_markdown(metrics, failures, records, meta)
    (REPORTS / f"latest-{args.profile}.md").write_text(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(md)
    print(md)

    if args.update_baseline:
        BASELINES.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps({"meta": meta, "metrics": metrics}, indent=2) + "\n")
        print(f"Baseline updated: {baseline_path}")
        return 0
    if failures and args.allow_uncalibrated and baseline is None:
        warn = (f"\n> ⚠️ Profile `{args.profile}` has no baseline yet, so its gates are not enforced. "
                f"Run the **calibrate** workflow and commit `eval/baselines/{args.profile}.json`.\n")
        print(warn, file=sys.stderr)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
                fh.write(warn)
        return 0
    if failures and not args.no_gate:
        print("EVAL GATE FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
