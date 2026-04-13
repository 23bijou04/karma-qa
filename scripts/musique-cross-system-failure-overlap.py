#!/usr/bin/env python3
"""Phase 3: Cross-system failure overlap for MuSiQue 1000q.

Classifies each question into 8 categories by (Hybrid, LinearRAG, LogicRAG) correctness.
Computes Venn-style overlap, oracle ceiling, and fixable Hybrid failure breakdown.

Output: data/cross_benchmark/musique/ablation/cross_system_failure_overlap.json
"""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "mu", Path(__file__).resolve().parent / "musique-analysis-utils.py"
)
_mu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mu)

load_hybrid = _mu.load_hybrid
load_linearrag = _mu.load_linearrag
load_logicrag = _mu.load_logicrag
save_json = _mu.save_json

# ── 8-category classification ──────────────────────────────────────────────
CATEGORY_MAP = {
    (True, True, True): "all_correct",
    (False, False, False): "all_wrong",
    (True, False, False): "hybrid_only",
    (False, True, False): "linearrag_only",
    (False, False, True): "logicrag_only",
    (True, True, False): "hybrid_linear",
    (True, False, True): "hybrid_logic",
    (False, True, True): "linear_logic",
}
CATEGORY_ORDER = [
    "all_correct", "all_wrong", "hybrid_only", "linearrag_only",
    "logicrag_only", "hybrid_linear", "hybrid_logic", "linear_logic",
]


def main():
    hybrid = load_hybrid()
    linear = load_linearrag()
    logic = load_logicrag()

    shared = sorted(set(hybrid) & set(linear) & set(logic))
    if len(shared) < 1000:
        print(f"WARNING: only {len(shared)} shared questions (expected 1000)", file=sys.stderr)

    per_question = []
    cat_counts = defaultdict(int)
    cat_by_type = defaultdict(lambda: defaultdict(int))
    # Fixable analysis: Hybrid wrong, at least one other correct
    fixable_by_method = defaultdict(int)
    fixable_by_route = defaultdict(int)
    fixable_total = 0

    for qid in shared:
        hc = bool(hybrid[qid].get("judge_correct", False))
        lc = bool(linear[qid].get("judge_correct", False))
        rc = bool(logic[qid].get("judge_correct", False))
        cat = CATEGORY_MAP[(hc, lc, rc)]
        qtype = hybrid[qid].get("question_type", "unknown")

        cat_counts[cat] += 1
        cat_by_type[qtype][cat] += 1

        rec = {
            "question_id": qid,
            "question_type": qtype,
            "category": cat,
            "hybrid_correct": hc,
            "linearrag_correct": lc,
            "logicrag_correct": rc,
            "hybrid_method": hybrid[qid].get("method", "?"),
            "hybrid_route": hybrid[qid].get("route_source", "?"),
        }
        per_question.append(rec)

        # Fixable: Hybrid wrong but someone else got it
        if not hc and (lc or rc):
            fixable_total += 1
            fixable_by_method[rec["hybrid_method"]] += 1
            fixable_by_route[rec["hybrid_route"]] += 1

    # Oracle ceiling: any system correct
    oracle = sum(1 for q in per_question if q["hybrid_correct"] or q["linearrag_correct"]
                 or q["logicrag_correct"])
    universally_unsolvable = cat_counts["all_wrong"]

    output = {
        "summary": {
            "shared_questions": len(shared),
            "category_counts": {c: cat_counts[c] for c in CATEGORY_ORDER},
            "oracle_ceiling": oracle,
            "oracle_ceiling_pct": round(100 * oracle / max(len(shared), 1), 1),
            "fixable_hybrid_failures": fixable_total,
            "universally_unsolvable": universally_unsolvable,
        },
        "per_type": {
            t: {c: cat_by_type[t].get(c, 0) for c in CATEGORY_ORDER}
            for t in sorted(cat_by_type)
        },
        "fixable_analysis": {
            "by_method": dict(sorted(fixable_by_method.items(), key=lambda x: -x[1])),
            "by_route": dict(sorted(fixable_by_route.items(), key=lambda x: -x[1])),
        },
        "per_question": per_question,
    }

    out = save_json(output, "cross_system_failure_overlap.json")

    # ── Console summary ─────────────────────────────────────────────────────
    print("CROSS-SYSTEM FAILURE OVERLAP (MuSiQue 1000q)")
    print("=" * 55)
    print(f"{'Category':<20} {'Count':>6} {'%':>7}")
    print("-" * 35)
    for c in CATEGORY_ORDER:
        n = cat_counts[c]
        pct = 100 * n / max(len(shared), 1)
        print(f"{c:<20} {n:>6} {pct:>6.1f}%")
    print("-" * 35)
    print(f"{'Total':<20} {len(shared):>6}")
    print()
    print(f"Oracle ceiling: {oracle}/{len(shared)} ({100*oracle/max(len(shared),1):.1f}%)")
    print(f"Fixable Hybrid failures: {fixable_total}/{sum(1 for q in per_question if not q['hybrid_correct'])}")
    print(f"Universally unsolvable: {universally_unsolvable}")

    # Per-type mini-table
    print(f"\n{'Type':<12} {'all_ok':>7} {'all_bad':>8} {'H_only':>7} {'L_only':>7} "
          f"{'R_only':>7} {'H+L':>5} {'H+R':>5} {'L+R':>5}")
    print("-" * 72)
    for t in sorted(cat_by_type):
        d = cat_by_type[t]
        print(f"{t:<12} {d.get('all_correct',0):>7} {d.get('all_wrong',0):>8} "
              f"{d.get('hybrid_only',0):>7} {d.get('linearrag_only',0):>7} "
              f"{d.get('logicrag_only',0):>7} {d.get('hybrid_linear',0):>5} "
              f"{d.get('hybrid_logic',0):>5} {d.get('linear_logic',0):>5}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
