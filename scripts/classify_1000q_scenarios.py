#!/usr/bin/env python3
"""Classify 1000 questions into KG coverage scenarios (A-F).

Reads coverage_analysis_1000.json and produces scenario_classification_1000.json
with per-question scenario labels and aggregate statistics.

Scenarios:
  A  KG_COMPLETE         — All entities exist, all properties available
  B  HOP1_MISSING        — Seed entity missing hop-1 predicate
  C  HOP2_MISSING        — Bridge entity missing hop-2 predicate
  D  BOTH_HOPS_MISSING   — Both seed and bridge have missing predicates
  E  ENTITY_NOT_FOUND    — Entity not in DBpedia, no property gap on existing
  F  ENTITY_NOT_FOUND_PLUS_PROP — Entity not found + property gap on existing
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

INPUT = Path("data/eval_1000q/coverage_analysis_1000.json")
OUTPUT = Path("data/eval_1000q/scenario_classification_1000.json")

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


def entity_summary(e: dict) -> dict:
    """Build per-entity summary for output."""
    return {
        "name": e["name"],
        "exists": e["exists"],
        "properties_found": e["properties_found"],
        "properties_missing": e["properties_missing"],
    }


def has_prop_gap(entity: dict) -> bool:
    return entity["exists"] and len(entity["properties_missing"]) > 0


def classify_chain(seed: dict, bridge: dict | None) -> str:
    """Classify a single seed→bridge chain. Returns 'complete', 'hop1', 'hop2', 'both'."""
    s_ok = len(seed["properties_missing"]) == 0
    if bridge is None:
        return "complete" if s_ok else "hop1"
    b_ok = len(bridge["properties_missing"]) == 0
    if s_ok and b_ok:
        return "complete"
    if not s_ok and b_ok:
        return "hop1"
    if s_ok and not b_ok:
        return "hop2"
    return "both"


def chain_to_scenario(status: str) -> str:
    """Map chain status to scenario code."""
    return {"complete": "A", "hop1": "B", "hop2": "C", "both": "D"}[status]


def worst_chain(s1: str, s2: str) -> str:
    """Return the worst chain status between two chains.
    Ordering: complete < hop1 < hop2 < both."""
    order = {"complete": 0, "hop1": 1, "hop2": 2, "both": 3}
    return s1 if order[s1] >= order[s2] else s2


def missing_props_str(entities: list[dict], role: str) -> str:
    """Build 'Entity.prop' summary string for missing properties."""
    parts = []
    for e in entities:
        for p in e["properties_missing"]:
            parts.append(f"{e['name']}.{p}")
    return "; ".join(parts) if parts else ""


def classify_question(q: dict) -> dict:
    """Classify one question and return enriched record."""
    entities = q["entities"]
    qtype = q["question_type"]

    # Split into seeds / bridges
    if qtype == "comparison":
        seeds, bridges = entities[0:2], []
    elif qtype == "bridge_comparison":
        seeds, bridges = entities[0:2], entities[2:4]
    else:  # compositional, inference
        seeds = [entities[0]]
        bridges = [entities[1]] if len(entities) > 1 else []

    result = {
        "question_id": q["question_id"],
        "question": q["question"],
        "question_type": qtype,
        "gold_answer": q["gold_answer"],
        "seed_entities": [entity_summary(e) for e in seeds],
        "bridge_entities": [entity_summary(e) for e in bridges],
    }

    # A: answerable
    if q["is_answerable"]:
        result["scenario"] = "A"
        result["scenario_label"] = SCENARIO_LABELS["A"]
        result["missing_properties_summary"] = ""
        return result

    # E/F: entity not found
    any_missing = not q["all_entities_exist"]
    if any_missing:
        existing_have_gap = any(has_prop_gap(e) for e in entities)
        sc = "F" if existing_have_gap else "E"
        result["scenario"] = sc
        result["scenario_label"] = SCENARIO_LABELS[sc]
        missing_entities = [e["name"] for e in entities if not e["exists"]]
        prop_parts = missing_props_str(
            [e for e in entities if e["exists"]], "existing"
        )
        parts = [f"not_found: {', '.join(missing_entities)}"]
        if prop_parts:
            parts.append(prop_parts)
        result["missing_properties_summary"] = "; ".join(parts)
        return result

    # Property-gap classification (all entities exist)
    if qtype == "comparison":
        # 1-hop only: seeds are the only entities, no bridges
        result["scenario"] = "B"
        result["scenario_label"] = SCENARIO_LABELS["B"]
        result["missing_properties_summary"] = missing_props_str(seeds, "seed")
        return result

    if qtype == "bridge_comparison":
        chain1 = classify_chain(seeds[0], bridges[0])
        chain2 = classify_chain(seeds[1], bridges[1])
        # Use aggregate seed/bridge completeness (not worst-chain) so that
        # cross-chain failures (e.g. seed₁ missing + bridge₂ missing) classify as D.
        s_ok = all(len(e["properties_missing"]) == 0 for e in seeds)
        b_ok = all(len(e["properties_missing"]) == 0 for e in bridges)
        if not s_ok and b_ok:
            sc = "B"
        elif s_ok and not b_ok:
            sc = "C"
        elif not s_ok and not b_ok:
            sc = "D"
        else:
            # Both complete but not answerable — should not happen
            sc = "A"
        result["scenario"] = sc
        result["scenario_label"] = SCENARIO_LABELS[sc]
        result["chain_detail"] = {
            "chain1": {"seed": seeds[0]["name"], "bridge": bridges[0]["name"],
                       "status": chain1},
            "chain2": {"seed": seeds[1]["name"], "bridge": bridges[1]["name"],
                       "status": chain2},
        }
        result["missing_properties_summary"] = missing_props_str(
            seeds + bridges, "all"
        )
        return result

    # compositional / inference
    seed = seeds[0]
    bridge = bridges[0] if bridges else None
    chain = classify_chain(seed, bridge)
    sc = chain_to_scenario(chain)
    result["scenario"] = sc
    result["scenario_label"] = SCENARIO_LABELS[sc]
    all_ents = seeds + bridges
    result["missing_properties_summary"] = missing_props_str(all_ents, "all")
    return result


def build_missing_property_ranking(questions: list[dict]) -> dict:
    """Rank missing properties by hop (seed=hop1, bridge=hop2)."""
    hop1: Counter = Counter()
    hop2: Counter = Counter()
    for q in questions:
        if q["scenario"] in ("E", "F"):
            continue
        for e in q["seed_entities"]:
            for p in e["properties_missing"]:
                hop1[p] += 1
        for e in q["bridge_entities"]:
            for p in e["properties_missing"]:
                hop2[p] += 1
    return {
        "hop1": dict(hop1.most_common(20)),
        "hop2": dict(hop2.most_common(20)),
    }


def print_report(questions: list[dict], summary: dict) -> None:
    """Print human-readable summary to stdout."""
    total = summary["total"]
    by_sc = summary["by_scenario"]
    by_sc_type = summary["by_scenario_and_type"]
    ranking = summary["missing_properties_ranking"]

    print("=" * 72)
    print("SCENARIO CLASSIFICATION — 1000 QUESTIONS")
    print("=" * 72)

    # Overall table
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
    # Totals row
    print(f"{'TOTAL':>6}", end="")
    for t in types:
        n = sum(by_sc_type.get(t, {}).get(sc, 0) for sc in "ABCDEF")
        print(f" {n:>18}", end="")
    print(f" {total:>8}")

    # Missing properties ranking
    print("\nTop missing properties (hop-1 = seed entity):")
    for prop, cnt in list(ranking["hop1"].items())[:10]:
        print(f"  {prop:<35} {cnt:>4}")

    print("\nTop missing properties (hop-2 = bridge entity):")
    for prop, cnt in list(ranking["hop2"].items())[:10]:
        print(f"  {prop:<35} {cnt:>4}")

    # Interpretation
    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    a_pct = 100 * by_sc["A"] / total
    b_pct = 100 * by_sc.get("B", 0) / total
    c_pct = 100 * by_sc.get("C", 0) / total
    ef_pct = 100 * (by_sc.get("E", 0) + by_sc.get("F", 0)) / total
    print(f"  A ({a_pct:.1f}%): Pure SPARQL can answer these — no Wikipedia needed.")
    print(f"  B ({b_pct:.1f}%): Hop-1 predicate missing — need Wikipedia infobox/text"
          " for seed entity.")
    print(f"  C ({c_pct:.1f}%): Hop-2 predicate missing — SPARQL finds bridge entity,"
          " but need Wikipedia for bridge properties.")
    print(f"  D ({100*by_sc.get('D',0)/total:.1f}%): Both hops broken — need Wikipedia"
          " for both seed and bridge entities.")
    print(f"  E+F ({ef_pct:.1f}%): Entity not in DBpedia — need text-only pipeline.")


def main() -> None:
    with open(INPUT) as f:
        data = json.load(f)

    raw_questions = data["questions"]
    classified = [classify_question(q) for q in raw_questions]

    # Build summary
    total = len(classified)
    by_scenario: Counter = Counter()
    by_scenario_type: dict[str, Counter] = defaultdict(Counter)

    for q in classified:
        sc = q["scenario"]
        by_scenario[sc] += 1
        by_scenario_type[q["question_type"]][sc] += 1

    ranking = build_missing_property_ranking(classified)

    summary = {
        "total": total,
        "by_scenario": dict(sorted(by_scenario.items())),
        "by_scenario_and_type": {
            t: dict(sorted(counts.items()))
            for t, counts in sorted(by_scenario_type.items())
        },
        "missing_properties_ranking": ranking,
    }

    # Validate
    assert sum(by_scenario.values()) == total, "Scenarios not exhaustive"
    assert by_scenario["A"] == data["statistics"]["overall"]["answerable_count"], (
        f"A={by_scenario['A']} != answerable={data['statistics']['overall']['answerable_count']}"
    )

    output = {"summary": summary, "questions": classified}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {OUTPUT} ({total} questions)")
    print_report(classified, summary)


if __name__ == "__main__":
    main()
