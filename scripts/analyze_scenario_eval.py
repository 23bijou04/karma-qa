#!/usr/bin/env python3
"""Analyze scenario evaluation results: Hybrid vs LogicRAG-forced.

Reads two JSONL files and prints per-scenario metrics, head-to-head
comparison, and method distribution tables.

Usage:
    python scripts/analyze_scenario_eval.py \\
        data/eval_1000q/scenario_eval_hybrid.jsonl \\
        data/eval_1000q/scenario_eval_logicrag.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict


def load_jsonl(path: str) -> dict[str, dict]:
    """Load JSONL file into dict keyed by question_id."""
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["question_id"]] = rec
    return records


def print_per_scenario(hybrid: dict, logicrag: dict, shared_ids: set):
    """Print per-scenario performance table."""
    by_scenario = defaultdict(lambda: {
        "h_em": 0.0, "h_f1": 0.0, "l_em": 0.0, "l_f1": 0.0, "count": 0,
    })
    for qid in shared_ids:
        h, lg = hybrid[qid], logicrag[qid]
        sc = h["scenario"]
        by_scenario[sc]["count"] += 1
        by_scenario[sc]["h_em"] += h.get("em", 0)
        by_scenario[sc]["h_f1"] += h.get("f1", 0)
        by_scenario[sc]["l_em"] += lg.get("em", 0)
        by_scenario[sc]["l_f1"] += lg.get("f1", 0)

    print("## Per-Scenario Performance\n")
    print("| Scenario | N | Hybrid EM | Hybrid F1 | LogicRAG EM | LogicRAG F1 | Delta EM |")
    print("|----------|---|-----------|-----------|-------------|-------------|----------|")

    totals = {"h_em": 0, "h_f1": 0, "l_em": 0, "l_f1": 0, "count": 0}
    for sc in sorted(by_scenario.keys()):
        s = by_scenario[sc]
        n = s["count"]
        h_em = s["h_em"] / n * 100
        h_f1 = s["h_f1"] / n * 100
        l_em = s["l_em"] / n * 100
        l_f1 = s["l_f1"] / n * 100
        delta = h_em - l_em
        sign = "+" if delta >= 0 else ""
        print(f"| {sc}        | {n} | {h_em:5.1f}%    | {h_f1:5.1f}%    | "
              f"{l_em:7.1f}%      | {l_f1:7.1f}%      | {sign}{delta:5.1f}%   |")
        for k in totals:
            totals[k] += s[k]

    n = totals["count"]
    if n:
        h_em = totals["h_em"] / n * 100
        h_f1 = totals["h_f1"] / n * 100
        l_em = totals["l_em"] / n * 100
        l_f1 = totals["l_f1"] / n * 100
        delta = h_em - l_em
        sign = "+" if delta >= 0 else ""
        print(f"| **ALL**  | {n} | {h_em:5.1f}%    | {h_f1:5.1f}%    | "
              f"{l_em:7.1f}%      | {l_f1:7.1f}%      | {sign}{delta:5.1f}%   |")
    print()


def print_head_to_head(hybrid: dict, logicrag: dict, shared_ids: set):
    """Print head-to-head comparison table."""
    both_correct = hybrid_only = logicrag_only = both_wrong = 0
    for qid in shared_ids:
        h_ok = hybrid[qid].get("em", 0) > 0
        l_ok = logicrag[qid].get("em", 0) > 0
        if h_ok and l_ok:
            both_correct += 1
        elif h_ok:
            hybrid_only += 1
        elif l_ok:
            logicrag_only += 1
        else:
            both_wrong += 1

    n = len(shared_ids)
    print("## Head-to-Head (EM)\n")
    print("| Outcome       | Count | %     |")
    print("|---------------|-------|-------|")
    for label, count in [("Both correct", both_correct), ("Hybrid only", hybrid_only),
                         ("LogicRAG only", logicrag_only), ("Both wrong", both_wrong)]:
        pct = count / n * 100 if n else 0
        print(f"| {label:<13} | {count:>5} | {pct:5.1f} |")
    print()


def print_method_distribution(records: dict, shared_ids: set, mode_label: str):
    """Print method distribution per scenario."""
    by_scenario = defaultdict(lambda: defaultdict(int))
    for qid in shared_ids:
        rec = records[qid]
        by_scenario[rec["scenario"]][rec.get("method", "unknown")] += 1

    # Collect all methods
    all_methods = sorted({
        m for sc in by_scenario.values() for m in sc
    })

    print(f"## Method Distribution ({mode_label})\n")
    header = "| Scenario | " + " | ".join(f"{m[:15]}" for m in all_methods) + " | Total |"
    sep = "|----------|" + "|".join("-" * (max(len(m[:15]), 5) + 2) for m in all_methods) + "|-------|"
    print(header)
    print(sep)

    for sc in sorted(by_scenario.keys()):
        counts = by_scenario[sc]
        total = sum(counts.values())
        cols = " | ".join(f"{counts.get(m, 0):>{max(len(m[:15]), 5)}}" for m in all_methods)
        print(f"| {sc}        | {cols} | {total:>5} |")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze scenario evaluation results")
    parser.add_argument("hybrid_file", help="Hybrid JSONL results")
    parser.add_argument("logicrag_file", help="LogicRAG JSONL results")
    args = parser.parse_args()

    hybrid = load_jsonl(args.hybrid_file)
    logicrag = load_jsonl(args.logicrag_file)

    shared_ids = set(hybrid.keys()) & set(logicrag.keys())
    h_only = set(hybrid.keys()) - shared_ids
    l_only = set(logicrag.keys()) - shared_ids

    print(f"Hybrid: {len(hybrid)} questions, LogicRAG: {len(logicrag)} questions")
    print(f"Shared: {len(shared_ids)}")
    if h_only:
        print(f"WARNING: {len(h_only)} questions only in hybrid")
    if l_only:
        print(f"WARNING: {len(l_only)} questions only in logicrag")
    print()

    if not shared_ids:
        print("No shared questions to compare.")
        sys.exit(1)

    print_per_scenario(hybrid, logicrag, shared_ids)
    print_head_to_head(hybrid, logicrag, shared_ids)
    print_method_distribution(hybrid, shared_ids, "Hybrid")
    print_method_distribution(logicrag, shared_ids, "LogicRAG-forced")


if __name__ == "__main__":
    main()
