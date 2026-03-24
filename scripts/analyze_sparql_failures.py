"""
Quantitative analysis of SPARQL success/failure patterns in scenario_eval_hybrid_judged.jsonl.

Usage:
    python scripts/analyze_sparql_failures.py
"""

import json
from collections import Counter, defaultdict

JUDGED = "data/eval_1000q/scenario_eval_hybrid_judged.jsonl"


def load(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify(r):
    """Return ('sparql_ok'|'sparql_failed'|'no_sparql', rescue_method)."""
    method = r.get("method", "")
    fallback = r.get("sparql_fallback", False)
    if method == "sparql":
        return "sparql_ok", None
    elif fallback:
        return "sparql_failed", method
    else:
        return "no_sparql", method


def infer_failure_reason(r):
    """
    Infer why SPARQL failed given only the fields available in the JSONL.
    (No SPARQL query text is logged, so we reason from route_source + scenario.)
    """
    route = r.get("route_source", "")
    scenario = r.get("scenario_label", "")
    qtype = r.get("question_type", "")

    if scenario == "HOP2_MISSING":
        return "hop2_property_not_in_KG"
    if scenario in ("BOTH_HOPS_MISSING", "ENTITY_NOT_FOUND_PLUS_PROP"):
        return "entity_or_property_absent"
    if scenario == "ENTITY_NOT_FOUND":
        return "entity_not_in_KG"
    if scenario == "KG_COMPLETE" and qtype == "bridge_comparison":
        return "comparison_SPARQL_not_implemented"
    if scenario == "KG_COMPLETE" and "comparison" in qtype:
        return "comparison_SPARQL_not_implemented"
    if scenario == "KG_COMPLETE":
        return "KG_complete_but_SPARQL_failed (unexpected)"
    return "unknown"


def sep(title, width=72):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def table(rows, headers, col_widths):
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def main():
    records = load(JUDGED)
    total = len(records)

    sparql_ok = [r for r in records if classify(r)[0] == "sparql_ok"]
    sparql_failed = [r for r in records if classify(r)[0] == "sparql_failed"]
    no_sparql = [r for r in records if classify(r)[0] == "no_sparql"]

    # ------------------------------------------------------------------ #
    sep("1. TOP-LEVEL BREAKDOWN")
    # ------------------------------------------------------------------ #
    print(f"  Total questions evaluated : {total}")
    print(f"  SPARQL succeeded (method=sparql)        : {len(sparql_ok):3d}  ({100*len(sparql_ok)/total:.0f}%)")
    print(f"  SPARQL attempted + failed (rescued)     : {len(sparql_failed):3d}  ({100*len(sparql_failed)/total:.0f}%)")
    print(f"  No SPARQL attempted                     : {len(no_sparql):3d}  ({100*len(no_sparql)/total:.0f}%)")
    print()
    total_sparql_attempted = len(sparql_ok) + len(sparql_failed)
    print(f"  Total SPARQL attempts : {total_sparql_attempted}")
    print(f"  SPARQL end-to-end success rate : {len(sparql_ok)}/{total_sparql_attempted} = "
          f"{100*len(sparql_ok)/total_sparql_attempted:.1f}%")

    # ------------------------------------------------------------------ #
    sep("2. SPARQL RESULT BY ROUTE SOURCE")
    # ------------------------------------------------------------------ #
    route_counts = defaultdict(lambda: {"sparql_ok": 0, "sparql_failed": 0, "no_sparql": 0})
    for r in records:
        cls, _ = classify(r)
        route_counts[r.get("route_source", "")][cls] += 1

    table(
        [(rs, d["sparql_ok"], d["sparql_failed"], d["no_sparql"], d["sparql_ok"] + d["sparql_failed"])
         for rs, d in sorted(route_counts.items(), key=lambda x: -(x[1]["sparql_ok"] + x[1]["sparql_failed"]))],
        ["route_source", "sparql_ok", "sparql_failed", "no_sparql", "attempted"],
        [40, 10, 13, 10, 10],
    )

    # ------------------------------------------------------------------ #
    sep("3. SPARQL FAILURE REASONS (inferred)")
    # ------------------------------------------------------------------ #
    reason_counts = Counter(infer_failure_reason(r) for r in sparql_failed)
    table(
        [(reason, cnt, f"{100*cnt/len(sparql_failed):.0f}%")
         for reason, cnt in reason_counts.most_common()],
        ["failure_reason", "count", "pct"],
        [45, 7, 5],
    )

    # ------------------------------------------------------------------ #
    sep("4. RESCUE METHOD FOR FAILED SPARQL")
    # ------------------------------------------------------------------ #
    rescue_counts = Counter(r["method"] for r in sparql_failed)
    for method, cnt in rescue_counts.most_common():
        correct_after_rescue = sum(1 for r in sparql_failed
                                   if r["method"] == method and r.get("judge_correct"))
        print(f"  rescued by {method}: {cnt}  |  correct after rescue: "
              f"{correct_after_rescue}/{cnt} = {100*correct_after_rescue/cnt:.0f}%")

    # ------------------------------------------------------------------ #
    sep("5. SCENARIO BREAKDOWN FOR SPARQL FAILURES")
    # ------------------------------------------------------------------ #
    scenario_counts = Counter((r["scenario_label"], r["question_type"]) for r in sparql_failed)
    table(
        [(sc, qt, cnt) for (sc, qt), cnt in scenario_counts.most_common()],
        ["scenario", "question_type", "count"],
        [30, 22, 7],
    )

    # ------------------------------------------------------------------ #
    sep("6. ACCURACY: sparql_ok vs sparql_failed vs no_sparql")
    # ------------------------------------------------------------------ #
    def acc(group):
        if not group:
            return "n/a"
        c = sum(1 for r in group if r.get("judge_correct"))
        return f"{c}/{len(group)} = {100*c/len(group):.0f}%"

    print(f"  sparql_ok       : {acc(sparql_ok)}")
    print(f"  sparql_failed   : {acc(sparql_failed)}")
    print(f"  no_sparql       : {acc(no_sparql)}")
    print(f"  overall         : {acc(records)}")

    # ------------------------------------------------------------------ #
    sep("7. THE 21 FAILED-SPARQL QUESTIONS IN DETAIL")
    # ------------------------------------------------------------------ #
    for i, r in enumerate(sparql_failed, 1):
        reason = infer_failure_reason(r)
        rescued_ok = "YES" if r.get("judge_correct") else "NO"
        print(f"\n  [{i:02d}] {r['question']}")
        print(f"       route    : {r['route_source']}")
        print(f"       scenario : {r['scenario_label']}  /  type: {r['question_type']}")
        print(f"       reason   : {reason}")
        print(f"       rescue   : {r['method']}  |  correct={rescued_ok}")
        print(f"       gold     : {r['gold_answer']}")
        print(f"       pred     : {r['predicted_answer']}")

    # ------------------------------------------------------------------ #
    sep("8. THE 7 SUCCESSFUL SPARQL QUERIES — WHAT MADE THEM SUCCEED")
    # ------------------------------------------------------------------ #
    print()
    for i, r in enumerate(sparql_ok, 1):
        print(f"  [{i}] {r['question']}")
        print(f"       route    : {r['route_source']}")
        print(f"       scenario : {r['scenario_label']}  /  type: {r['question_type']}")
        print(f"       gold     : {r['gold_answer']}")
        print(f"       pred     : {r['predicted_answer']}")
        print(f"       correct  : {r.get('judge_correct')}")
        print()

    # ------------------------------------------------------------------ #
    sep("9. COMPARISON: bridge_comparison routing SPARQL vs LogicRAG")
    # ------------------------------------------------------------------ #
    bc_all = [r for r in records if r.get("question_type") == "bridge_comparison"]
    bc_sparql_attempted = [r for r in bc_all if r.get("sparql_fallback") or r["method"] == "sparql"]
    bc_sparql_ok = [r for r in bc_all if r["method"] == "sparql"]
    bc_logicrag = [r for r in bc_all if r["method"] == "logicrag_decomposed"]

    print(f"  bridge_comparison total        : {len(bc_all)}")
    print(f"  SPARQL attempted (incl. failed): {len(bc_sparql_attempted)}")
    print(f"  SPARQL actually succeeded      : {len(bc_sparql_ok)}")
    print(f"  rescued by LogicRAG            : {len(bc_logicrag)}")
    print(f"  LogicRAG correct               : {sum(1 for r in bc_logicrag if r.get('judge_correct'))}/{len(bc_logicrag)}")

    # ------------------------------------------------------------------ #
    sep("10. SUMMARY: SPARQL ROUTING PROBLEMS")
    # ------------------------------------------------------------------ #
    print("""
  Problem 1 — bridge_comparison always routes to SPARQL (ask_verified),
    but the SPARQL query for date comparison across two films is never
    generated as a single query (requires two sub-queries + comparison).
    All 9 bridge_comparison SPARQL attempts fail and fall to LogicRAG.
    Fix: route bridge_comparison directly to DECOMPOSE, skip SPARQL ASK.

  Problem 2 — direct_property route fires for compositional (2-hop) questions
    (e.g., "award of director of film X") in HOP2_MISSING and BOTH_HOPS_MISSING
    scenarios. Property is not in KG so SPARQL returns empty → fallback.
    Fix: tighten direct_property routing to require property present in schema.

  Problem 3 — bridge_hop2+ask_verified fires even when hop2 property is absent
    (scenario=HOP2_MISSING). ASK verifies the hop1 triple but the hop2 query
    still fails. LogicRAG rescues most of these correctly.
    Fix: also ASK-verify the hop2 property before committing to SPARQL.

  Problem 4 — ENTITY_NOT_FOUND scenarios: routing still fires rules (direct
    property) when the entity isn't in DBpedia at all.  Text solver rescues.
    Fix: if Spotlight entity linking returns no URI, skip SPARQL entirely.

  Bright spot: the 7 successful SPARQL queries share:
    - Scenario KG_COMPLETE (property actually present)
    - Single-hop or well-formed 2-hop with ask_verified
    - Simple factual properties (birthPlace, deathDate, relative)
""")


if __name__ == "__main__":
    main()
