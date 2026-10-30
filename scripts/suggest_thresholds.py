"""Suggest gate thresholds from a baseline: observed value minus a safety margin.

  python scripts/suggest_thresholds.py models default --margin 0.05

Prints a YAML snippet to paste under `profiles:` in eval/thresholds.yaml.
Thresholds catch big drops; the regression check (max_regression vs the baseline)
catches small ones, so a margin of ~0.05 below observed is a sensible floor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATED = ["retrieval_hit@candidates", "context_recall@k", "mrr@k", "answer_rate",
         "refusal_accuracy", "faithfulness", "citation_doc_accuracy"]


def suggest(metrics: dict, margin: float) -> dict:
    out = {}
    for name in GATED:
        v = metrics.get(name)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out[name] = max(0.0, math.floor((v - margin) * 100) / 100)
    p95 = metrics.get("latency_p95_s")
    if p95:
        out["latency_p95_s"] = round(max(1.0, math.ceil(p95 * 2)), 1)  # 2x headroom for runner noise
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="+")
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()
    lines = ["```yaml", "# paste under `profiles:` in eval/thresholds.yaml"]
    for prof in args.profiles:
        path = ROOT / "eval" / "baselines" / f"{prof}.json"
        if not path.exists():
            print(f"# no baseline for {prof} (did its eval run?)", file=sys.stderr)
            continue
        data = json.loads(path.read_text())
        lines.append(f"  {prof}:   # from {data['meta'].get('commit', 'local')[:8]}, judge {data['meta'].get('judge')}")
        lines += [f"    {k}: {v}" for k, v in suggest(data["metrics"], args.margin).items()]
    lines.append("```")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
