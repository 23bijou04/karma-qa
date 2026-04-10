#!/usr/bin/env python3
"""Compare baseline vs revised hybrid eval with McNemar test and per-role breakdown.

Reads two judged JSONL files and computes:
- McNemar significance test (paired, 2x2 contingency)
- Per-type accuracy comparison (4 question types)
- Per-method accuracy breakdown (revised system)
- Net improvement / regression count

Usage:
    python3 scripts/compare-evals-statistical.py \
        data/eval_1000q/hybrid_eval_346_baseline_judged.jsonl \
        data/eval_1000q/hybrid_eval_346_revised_judged.jsonl
"""

import json
import math
import sys
from collections import defaultdict


def load(path):
    """Load judged JSONL into {question_id: record} dict."""
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                results[d["question_id"]] = d
    return results


def mcnemar(baseline, revised):
    """Compute McNemar test on paired judge_correct outcomes."""
    shared = set(baseline) & set(revised)
    b, c = 0, 0  # b=base_correct+rev_wrong, c=base_wrong+rev_correct
    regressions, improvements = [], []
    for qid in shared:
        bc = baseline[qid].get("judge_correct", False)
        rc = revised[qid].get("judge_correct", False)
        if bc and not rc:
            b += 1
            regressions.append(qid)
        elif not bc and rc:
            c += 1
            improvements.append(qid)
    # McNemar with continuity correction (standard for 2x2 chi-squared)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    # Chi-squared(df=1) survival function for proper p-value
    p_val = math.erfc(math.sqrt(chi2) / math.sqrt(2)) if chi2 > 0 else 1.0
    sig = f"p={p_val:.4f} ({'significant' if p_val < 0.05 else 'not significant'})"
    return {
        "shared": len(shared),
        "improved": c,
        "regressed": b,
        "net": c - b,
        "chi2": round(chi2, 2),
        "p_value": round(p_val, 4),
        "significance": sig,
        "regression_rate": f"{b / (b + c) * 100:.1f}%" if (b + c) else "N/A",
        "improvements": improvements,
        "regressions": regressions,
    }


def per_group(results, key):
    """Compute accuracy per group (e.g., per question_type or method)."""
    groups = defaultdict(lambda: {"correct": 0, "total": 0})
    for d in results.values():
        g = d.get(key, "unknown")
        groups[g]["total"] += 1
        if d.get("judge_correct"):
            groups[g]["correct"] += 1
    return {
        k: {**v, "pct": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0}
        for k, v in groups.items()
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/compare-evals-statistical.py "
              "<baseline.jsonl> <revised.jsonl>")
        sys.exit(1)

    base = load(sys.argv[1])
    rev = load(sys.argv[2])

    print(f"Baseline: {len(base)} questions from {sys.argv[1]}")
    print(f"Revised:  {len(rev)} questions from {sys.argv[2]}")

    # McNemar test
    print("\n=== McNemar Test ===")
    mc = mcnemar(base, rev)
    for k in ["shared", "improved", "regressed", "net", "chi2",
              "significance", "regression_rate"]:
        print(f"  {k}: {mc[k]}")

    # Per-method accuracy (revised)
    print("\n=== Per-Method Accuracy (Revised) ===")
    for m, s in sorted(per_group(rev, "method").items()):
        print(f"  {m}: {s['correct']}/{s['total']} = {s['pct']}%")

    # Per-route-source accuracy (revised)
    print("\n=== Per-Route-Source Accuracy (Revised) ===")
    for m, s in sorted(per_group(rev, "route_source").items()):
        print(f"  {m}: {s['correct']}/{s['total']} = {s['pct']}%")

    # Per-type comparison
    print("\n=== Per-Type Comparison ===")
    bt = per_group(base, "question_type")
    rt = per_group(rev, "question_type")
    for t in sorted(set(bt) | set(rt)):
        bp = bt.get(t, {}).get("pct", 0)
        rp = rt.get(t, {}).get("pct", 0)
        bn = bt.get(t, {}).get("total", 0)
        rn = rt.get(t, {}).get("total", 0)
        print(f"  {t}: {bp}% (n={bn}) -> {rp}% (n={rn}) (delta: {rp - bp:+.1f})")

    # Overall accuracy
    print("\n=== Overall ===")
    base_correct = sum(1 for d in base.values() if d.get("judge_correct"))
    rev_correct = sum(1 for d in rev.values() if d.get("judge_correct"))
    print(f"  Baseline: {base_correct}/{len(base)} = "
          f"{base_correct / len(base) * 100:.1f}%")
    print(f"  Revised:  {rev_correct}/{len(rev)} = "
          f"{rev_correct / len(rev) * 100:.1f}%")

    # Top regressions
    if mc["regressions"]:
        print(f"\n=== Regressions ({len(mc['regressions'])}) ===")
        for qid in mc["regressions"][:10]:
            q = rev[qid].get("question", "")[:60]
            method = rev[qid].get("method", "?")
            print(f"  {qid}: [{method}] {q}...")


if __name__ == "__main__":
    main()
