#!/usr/bin/env python3
"""Classify 1000 questions into KG coverage scenarios (A-F).

Reads hop_verification_results.json (per-hop entity_resolved + property_exists)
and produces scenario_classification_1000.json with per-question scenario labels.

Scenarios:
  A  KG_COMPLETE         — All hops have properties available
  B  HOP1_MISSING        — Seed entity missing hop-1 predicate
  C  HOP2_MISSING        — Bridge entity missing hop-2 predicate
  D  BOTH_HOPS_MISSING   — Both seed and bridge have missing predicates
  E  ENTITY_NOT_FOUND    — Entity not in DBpedia, no property gap on resolved entities
  F  ENTITY_NOT_FOUND_PLUS_PROP — Entity not found + property gap on resolved entities

Data source: hop_verification_results.json (ground truth, per-hop verification).
Supersedes the old coverage_analysis_1000.json-based classification.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

INPUT = Path("data/eval_1000q/ground-truth/hop_verification_results.json")
OUTPUT = Path("data/eval_1000q/ground-truth/scenario_classification_1000.json")

SCENARIO_LABELS = {
    "A": "KG_COMPLETE",
    "B": "HOP1_MISSING",
    "C": "HOP2_MISSING",
    "D": "BOTH_HOPS_MISSING",
    "E": "ENTITY_NOT_FOUND",
    "F": "ENTITY_NOT_FOUND_PLUS_PROP",
}


def wilson_ci(count: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = count / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def split_hops_by_role(hops: list[dict], qtype: str) -> tuple[list[dict], list[dict]]:
    """Split hops into hop-1 (seed→bridge) and hop-2 (bridge→answer) roles.

    Returns (hop1_list, hop2_list).

    Comparison questions have NO hop-2 — all hops are independent seed-property
    checks (e.g., entity1.birthDate, entity2.birthDate). Even 3-4 hop comparisons
    (dual-nationality, "who lived longer") are all hop-1 level.

    Bridge_comparison has hops[:2] as hop-1 (seed→bridge for each chain)
    and hops[2:] as hop-2 (bridge→answer for each chain). Verified for 5-6 hop
    edge cases (extra hops are additional expected values for bridge properties).
    """
    if qtype == "comparison":
        # All hops are hop-1 level — no bridge entity, no hop-2
        return hops, []
    elif qtype == "bridge_comparison":
        # First 2 hops = seed→bridge (hop-1), remaining = bridge→answer (hop-2)
        return hops[:2], hops[2:]
    else:
        # compositional / inference: hop[0] = hop-1, hop[1] = hop-2
        return [hops[0]], [hops[1]] if len(hops) > 1 else []


def classify_question(q: dict) -> dict:
    """Classify one question into scenario A-F using hop verification data."""
    hops = q["hops"]
    qtype = q["question_type"]

    result = {
        "question_id": q["question_id"],
        "question": q["question"],
        "question_type": qtype,
        "gold_answer": q["gold_answer"],
        "num_hops": len(hops),
    }

    # Check entity resolution first (E/F scenarios)
    unresolved_hops = [h for h in hops if not h["entity_resolved"]]
    if unresolved_hops:
        # Some entity not in DBpedia — check if resolved entities have property gaps
        resolved_hops = [h for h in hops if h["entity_resolved"]]
        has_prop_gap = any(not h["property_exists"] for h in resolved_hops)
        sc = "F" if has_prop_gap else "E"
        result["scenario"] = sc
        result["scenario_label"] = SCENARIO_LABELS[sc]
        result["unresolved_entities"] = [h["subject"] for h in unresolved_hops]
        result["missing_properties"] = [
            f"{h['subject']}.{h['property']}"
            for h in resolved_hops if not h["property_exists"]
        ]
        return result

    # All entities resolved — check property completeness
    if q["all_properties_exist"]:
        result["scenario"] = "A"
        result["scenario_label"] = SCENARIO_LABELS["A"]
        result["missing_properties"] = []
        return result

    # Property gap — classify by which hop role is missing
    hop1_list, hop2_list = split_hops_by_role(hops, qtype)

    hop1_ok = all(h["property_exists"] for h in hop1_list)
    hop2_ok = all(h["property_exists"] for h in hop2_list) if hop2_list else True

    if not hop1_ok and hop2_ok:
        sc = "B"
    elif hop1_ok and not hop2_ok:
        sc = "C"
    elif not hop1_ok and not hop2_ok:
        sc = "D"
    else:
        # Both ok but all_properties_exist=False — should not happen
        raise ValueError(
            f"Inconsistency: all hops ok but all_properties_exist=False "
            f"for {q['question_id']}"
        )

    result["scenario"] = sc
    result["scenario_label"] = SCENARIO_LABELS[sc]
    result["missing_properties"] = [
        f"{h['subject']}.{h['property']}"
        for h in hops if not h["property_exists"]
    ]
    return result


def build_missing_property_ranking(questions: list[dict], all_qs: list[dict]) -> dict:
    """Rank missing properties by hop role (hop-1 vs hop-2).

    Uses the original hop data (all_qs) to get hop details,
    paired with classification results (questions) for scenario filtering.
    """
    hop1: Counter = Counter()
    hop2: Counter = Counter()

    for classified, original in zip(questions, all_qs):
        if classified["scenario"] in ("E", "F"):
            continue
        hops = original["hops"]
        qtype = original["question_type"]
        hop1_list, hop2_list = split_hops_by_role(hops, qtype)

        for h in hop1_list:
            if not h["property_exists"]:
                hop1[h["property"]] += 1
        for h in hop2_list:
            if not h["property_exists"]:
                hop2[h["property"]] += 1

    return {
        "hop1": dict(hop1.most_common(20)),
        "hop2": dict(hop2.most_common(20)),
    }


def print_report(questions: list[dict], summary: dict) -> None:
    """Print human-readable summary to stdout."""
    total = summary["total"]
    by_sc = summary["by_scenario"]

    print("=" * 72)
    print("SCENARIO CLASSIFICATION — 1000 QUESTIONS (hop-verified)")
    print("=" * 72)

    print(f"\n{'Scenario':<6} {'Label':<28} {'Count':>5} {'%':>7}  {'95% CI'}")
    print("-" * 72)
    for sc in "ABCDEF":
        n = by_sc.get(sc, 0)
        lo, hi = wilson_ci(n, total)
        pct = 100 * n / total
        print(f"  {sc}    {SCENARIO_LABELS[sc]:<28} {n:>5} {pct:>6.1f}%  "
              f"[{100*lo:.1f}%, {100*hi:.1f}%]")
    print(f"{'':>36} {total:>5}")

    # Cross-tab: scenario × type
    by_sc_type = summary["by_scenario_and_type"]
    types = sorted({q["question_type"] for q in questions})
    print(f"\n{'Scenario':<6}", end="")
    for t in types:
        print(f" {t:>18}", end="")
    print(f" {'TOTAL':>8}")
    print("-" * (6 + 19 * len(types) + 9))
    for sc in "ABCDEF":
        print(f"  {sc}   ", end="")
        row_total = 0
        for t in types:
            n = by_sc_type.get(t, {}).get(sc, 0)
            row_total += n
            print(f" {n:>18}", end="")
        print(f" {row_total:>8}")
    print(f"{'TOTAL':>6}", end="")
    for t in types:
        n = sum(by_sc_type.get(t, {}).get(sc, 0) for sc in "ABCDEF")
        print(f" {n:>18}", end="")
    print(f" {total:>8}")

    # Missing properties ranking
    ranking = summary["missing_properties_ranking"]
    print("\nTop missing properties (hop-1 = seed entity):")
    for prop, cnt in list(ranking["hop1"].items())[:10]:
        print(f"  {prop:<35} {cnt:>4}")
    print("\nTop missing properties (hop-2 = bridge entity):")
    for prop, cnt in list(ranking["hop2"].items())[:10]:
        print(f"  {prop:<35} {cnt:>4}")


def main() -> None:
    with open(INPUT) as f:
        data = json.load(f)

    raw_questions = data["results"]
    classified = [classify_question(q) for q in raw_questions]

    # Build summary
    total = len(classified)
    by_scenario: Counter = Counter()
    by_scenario_type: dict[str, Counter] = defaultdict(Counter)

    for q in classified:
        sc = q["scenario"]
        by_scenario[sc] += 1
        by_scenario_type[q["question_type"]][sc] += 1

    ranking = build_missing_property_ranking(classified, raw_questions)

    summary = {
        "total": total,
        "by_scenario": dict(sorted(by_scenario.items())),
        "by_scenario_and_type": {
            t: dict(sorted(counts.items()))
            for t, counts in sorted(by_scenario_type.items())
        },
        "missing_properties_ranking": ranking,
        "data_source": "hop_verification_results.json",
        "note": "Recomputed 2026-03-31 from hop-verified ground truth",
    }

    # Validate: A must equal all_properties_exist count
    a_count = by_scenario["A"]
    hop_verified_count = sum(1 for q in raw_questions if q["all_properties_exist"])
    assert a_count == hop_verified_count, (
        f"A={a_count} != all_properties_exist={hop_verified_count}"
    )
    assert sum(by_scenario.values()) == total, "Scenarios not exhaustive"

    output = {"summary": summary, "questions": classified}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {OUTPUT} ({total} questions)")
    print_report(classified, summary)


if __name__ == "__main__":
    main()
