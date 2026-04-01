#!/usr/bin/env python3
"""Compute aggregated metrics across all cross-benchmark evaluation results.

Loads 6 judged JSONL files (3 systems x 2 benchmarks), computes per-system
per-benchmark aggregate metrics (Judge EM, EM, F1, Containment), and generates
a summary JSON + markdown comparison table.

Usage:
    python scripts/compute-cross-benchmark-metrics.py
    python scripts/compute-cross-benchmark-metrics.py --include-2wiki
"""

import argparse
import json
import math
import os
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROSS_DIR = os.path.join(PROJECT_ROOT, "data", "cross_benchmark")

# System -> benchmark -> JSONL file mapping
EVAL_FILES = {
    "hotpotqa": {
        "hybrid": "hybrid_hotpotqa_1000q_judged.jsonl",
        "logicrag": "logicrag_hotpotqa_1000q_judged.jsonl",
        "linearrag": "linearrag_hotpotqa_1000q_judged.jsonl",
    },
    "musique": {
        "hybrid": "hybrid_musique_1000q_judged.jsonl",
        "logicrag": "logicrag_musique_1000q_judged.jsonl",
        "linearrag": "linearrag_musique_1000q_judged.jsonl",
    },
}

# 2Wiki results from existing evaluations (for complete 3x3 grid)
TWIKI_FILES = {
    "hybrid": os.path.join(PROJECT_ROOT, "data", "eval_1000q", "hybrid",
                           "hybrid_eval_1000_judged.jsonl"),
    "logicrag": os.path.join(PROJECT_ROOT, "data", "eval_1000q", "logicrag",
                             "logicrag_1000q_traced_judged.jsonl"),
    "linearrag": os.path.join(PROJECT_ROOT, "data", "eval_1000q", "linearrag",
                              "linearrag_1000q_judged.jsonl"),
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def load_judged(path: str) -> list[dict]:
    """Load JSONL records from judged file."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def compute_aggregates(records: list[dict]) -> dict:
    """Compute aggregate metrics from a list of judged records."""
    n = len(records)
    if n == 0:
        return {"n": 0}

    judge_correct = sum(1 for r in records if r.get("judge_correct", False))
    em_sum = sum(r.get("em", 0.0) for r in records)
    f1_sum = sum(r.get("f1", 0.0) for r in records)
    cont_sum = sum(r.get("containment", 0.0) for r in records)

    lo, hi = wilson_ci(judge_correct, n)

    return {
        "n": n,
        "judge_em": round(judge_correct / n * 100, 1),
        "judge_em_count": judge_correct,
        "judge_em_ci_lo": round(lo * 100, 1),
        "judge_em_ci_hi": round(hi * 100, 1),
        "em": round(em_sum / n * 100, 1),
        "f1": round(f1_sum / n * 100, 1),
        "containment": round(cont_sum / n * 100, 1),
    }


def compute_per_type(records: list[dict]) -> dict:
    """Compute aggregates per question_type."""
    by_type = defaultdict(list)
    for r in records:
        qtype = r.get("question_type", "unknown")
        by_type[qtype].append(r)
    return {qtype: compute_aggregates(recs) for qtype, recs in sorted(by_type.items())}


def main():
    parser = argparse.ArgumentParser(description="Cross-benchmark metrics aggregation")
    parser.add_argument("--include-2wiki", action="store_true",
                        help="Include 2WikiMultiHopQA results for full 3x3 grid")
    parser.add_argument("--output", default=os.path.join(CROSS_DIR, "metrics_summary.json"))
    args = parser.parse_args()

    results = {}
    systems = ["hybrid", "logicrag", "linearrag"]
    benchmarks = list(EVAL_FILES.keys())

    if args.include_2wiki:
        benchmarks.append("2wiki")

    # Load and compute for each benchmark x system
    for bench in benchmarks:
        results[bench] = {}
        for sys_name in systems:
            if bench == "2wiki":
                path = TWIKI_FILES.get(sys_name, "")
            else:
                fname = EVAL_FILES[bench][sys_name]
                path = os.path.join(CROSS_DIR, bench, fname)

            records = load_judged(path)
            if not records:
                print(f"  WARN: No records for {sys_name}/{bench} at {path}")
                results[bench][sys_name] = {"n": 0, "path": path}
                continue

            agg = compute_aggregates(records)
            agg["per_type"] = compute_per_type(records)
            agg["path"] = path
            results[bench][sys_name] = agg
            print(f"  {sys_name}/{bench}: n={agg['n']} Judge EM={agg['judge_em']}% "
                  f"({agg['judge_em_ci_lo']}-{agg['judge_em_ci_hi']}%) "
                  f"EM={agg['em']}% F1={agg['f1']}%")

    # Save JSON summary
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {args.output}")

    # Generate markdown table
    print("\n" + "=" * 80)
    print("## Cross-Benchmark Comparison (Judge EM %)")
    print()
    header = "| System    |"
    sep = "|-----------|"
    for bench in benchmarks:
        label = "2Wiki" if bench == "2wiki" else bench.capitalize()
        header += f" {label:>18s} |"
        sep += "-------------------:|"
    print(header)
    print(sep)

    for sys_name in systems:
        row = f"| {sys_name.capitalize():<9s} |"
        for bench in benchmarks:
            d = results[bench].get(sys_name, {})
            if d.get("n", 0) == 0:
                row += f" {'(no data)':>18s} |"
            else:
                row += f" {d['judge_em']:5.1f}% ({d['judge_em_ci_lo']:.1f}-{d['judge_em_ci_hi']:.1f}) |"
        print(row)

    # LinearRAG published footnote
    print()
    print("LinearRAG published (GPT-4o-mini): 2Wiki 63.7%, HotpotQA 66.5%, MuSiQue 37.0% GPT-Acc")
    print("Note: HotpotQA evaluated in closed-corpus setting (~10K passages)")


if __name__ == "__main__":
    main()
