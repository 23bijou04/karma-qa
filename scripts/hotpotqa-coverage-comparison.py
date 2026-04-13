#!/usr/bin/env python3
"""HotpotQA vs 2WikiMultiHopQA coverage comparison.

Loads existing HotpotQA grounding results (already computed 2026-04-07) and
produces a side-by-side coverage table comparing both benchmarks.

No API calls — reads pre-computed data only.

Usage:
    python scripts/hotpotqa-coverage-comparison.py
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

HOTPOT_GROUNDING = BASE / "data/cross_benchmark/hotpotqa/dbpedia_grounding_results.json"
OUTPUT_PATH = BASE / "data/eval_1000q/ablation/hotpotqa_coverage_comparison.json"

# ── 2WikiMultiHopQA ground-truth numbers (from hop_verification_results.json) ──
# Source: scripts/verify_hop_coverage.py on data/eval_1000q/hop_verification_results.json
# Methodology: hop-verified predicate coverage — each hop's specific predicate exists on entity
WIKI2_NUMBERS = {
    "benchmark": "2WikiMultiHopQA",
    "sample_size": 1000,
    "entity_coverage": {
        "count": 936,
        "pct": 93.6,
        "method": "DBpedia Spotlight + SPARQL label fallback",
    },
    "property_coverage": {
        "count": 368,
        "pct": 36.8,
        "method": "hop-verified: specific predicate exists on entity per hop",
        "note": "funnel: 1000 → 936 (entity) → 368 (predicate per hop)",
    },
    "answer_grounding": {
        "pct": None,
        "note": "not computed for 2WikiMHQA subset (KG-answerable proxy = 11.9%)",
    },
}

# ── Methodology notes ────────────────────────────────────────────────────────
METHODOLOGY_NOTE = (
    "HotpotQA 'property coverage' uses pair-connectivity (any DBpedia property connecting "
    "entity pair, 1-hop or 2-hop). 2WikiMHQA uses hop-verified predicate coverage (specific "
    "predicate per hop from [S,P,O] evidence triples). HotpotQA lacks hop-annotated [S,P,O] "
    "triples — only Wikipedia title pairs are provided — so exact methodology match is impossible."
)


def load_hotpot_grounding() -> dict:
    with open(HOTPOT_GROUNDING) as f:
        return json.load(f)


def build_comparison(hotpot: dict) -> dict:
    s = hotpot["summary"]
    total = s["total_questions"]
    entity_count = s["entity_coverage"]["count"]
    entity_pct = s["entity_coverage"]["pct"]

    conn_count = s["pair_connectivity"]["count"]
    conn_pct = s["pair_connectivity"]["pct"]
    conn_direct = s["pair_connectivity"]["direct"]
    conn_twohop = s["pair_connectivity"]["two_hop"]

    # value-only answer grounding (corrected: 303/692 = 43.8%, length-ratio guard applied)
    # Memory note (2026-04-07): denominator 692 from tier-3 fix (length-ratio guard >= 0.4)
    # naive denominator = total - not_applicable - entity_answer = 1000 - 63 - 225 = 712
    # but length-ratio guard filters 20 additional non-matches → 692 applicable questions
    value_grounded = s["answer_grounding"]["value_grounded"]
    value_pct = round(value_grounded / 692 * 100, 1)

    hotpot_numbers = {
        "benchmark": "HotpotQA",
        "sample_size": total,
        "entity_coverage": {
            "count": entity_count,
            "pct": entity_pct,
            "method": "Wikipedia title → DBpedia URI (title-to-resource mapping)",
        },
        "property_coverage": {
            "count": conn_count,
            "pct": conn_pct,
            "method": "pair-connectivity: any DBpedia property connecting entity pair (1-hop or 2-hop)",
            "breakdown": {
                "direct_1hop": conn_direct,
                "two_hop": conn_twohop,
                "comparison_questions": 0,
                "note": "comparison questions (189) excluded: entities share property type, not connected",
            },
        },
        "answer_grounding": {
            "value_grounded": value_grounded,
            "applicable_denominator": 692,
            "pct": value_pct,
            "method": "value-only substring match with length-ratio guard (>=0.4)",
            "note": "entity answers (225) and not-applicable (63) excluded from denominator",
        },
        "by_type": hotpot["by_type"],
    }

    # Normalized property coverage: both on /1000 denominator for fair delta
    hotpot_prop_per_1000 = round(conn_count / total * 100, 1)

    comparison = {
        "2wikimhqa": WIKI2_NUMBERS,
        "hotpotqa": hotpot_numbers,
        "methodology_note": METHODOLOGY_NOTE,
        "table": {
            "entity_coverage_pct": {
                "2wikimhqa": 93.6,
                "hotpotqa": 98.0,
                "delta": round(98.0 - 93.6, 1),
            },
            "property_coverage_pct": {
                "2wikimhqa": 36.8,
                "hotpotqa": conn_pct,
                "delta": round(conn_pct - 36.8, 1),
                "denominator_note": (
                    f"raw denominators differ: HotpotQA {conn_count}/{entity_count} "
                    f"(entity-covered), 2WikiMHQA 368/1000 (total)"
                ),
                "note": "different methodologies — see methodology_note",
            },
            "answer_grounding_pct": {
                "2wikimhqa": None,
                "hotpotqa": value_pct,
            },
        },
        "normalized_comparison": {
            "note": "Both normalized to /1000 total questions for fair delta computation",
            "property_coverage_per_1000": {
                "2wikimhqa": 36.8,
                "hotpotqa": hotpot_prop_per_1000,
                "delta": round(hotpot_prop_per_1000 - 36.8, 1),
            },
            "bridge_only": {
                "hotpotqa_bridge_count": conn_count,
                "hotpotqa_bridge_total": hotpot_numbers["by_type"]["bridge"]["total"],
                "hotpotqa_bridge_pct": round(
                    conn_count / hotpot_numbers["by_type"]["bridge"]["total"] * 100, 1
                ),
                "note": "HotpotQA bridge-only (excluding 189 comparison questions at 0%)",
            },
        },
    }
    return comparison


def print_table(comparison: dict) -> None:
    t = comparison["table"]
    print("\n" + "=" * 60)
    print(f"{'Metric':<30} {'2WikiMHQA':>12} {'HotpotQA':>12}")
    print("-" * 60)

    ec = t["entity_coverage_pct"]
    print(f"{'Entity coverage':<30} {ec['2wikimhqa']:>11.1f}% {ec['hotpotqa']:>11.1f}%")

    pc = t["property_coverage_pct"]
    wiki_pc = f"{pc['2wikimhqa']:.1f}%" if pc["2wikimhqa"] is not None else "N/A"
    print(f"{'Property/connectivity coverage':<30} {wiki_pc:>12} {pc['hotpotqa']:>11.1f}%")

    ag = t["answer_grounding_pct"]
    wiki_ag = f"{ag['2wikimhqa']:.1f}%" if ag["2wikimhqa"] is not None else "N/A"
    print(f"{'Answer grounding (value-only)':<30} {wiki_ag:>12} {ag['hotpotqa']:>11.1f}%")

    print("=" * 60)

    nc = comparison.get("normalized_comparison", {})
    np_ = nc.get("property_coverage_per_1000", {})
    bo = nc.get("bridge_only", {})
    print(f"\n{'Normalized (/1000)':<30} {np_.get('2wikimhqa', 'N/A'):>11}% {np_.get('hotpotqa', 'N/A'):>11}%")
    print(f"{'Bridge-only (HotpotQA)':<30} {'—':>12} {bo.get('hotpotqa_bridge_pct', 'N/A'):>11}%")

    print("\nNotes:")
    print("  - Property/connectivity: different methodologies (see JSON for details)")
    print("  - 2WikiMHQA uses hop-verified [S,P,O] predicate coverage")
    print("  - HotpotQA uses any-property pair-connectivity (no [S,P,O] triples available)")
    print(f"  - Raw delta: {pc['delta']:+.1f}pp (different denominators)")
    print(f"  - Normalized delta (/1000): {np_.get('delta', 'N/A'):+.1f}pp")
    print(f"  - Entity delta: {ec['delta']:+.1f}pp (methodology artifact)")


def main() -> None:
    hotpot = load_hotpot_grounding()
    comparison = build_comparison(hotpot)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

    print_table(comparison)


if __name__ == "__main__":
    main()
