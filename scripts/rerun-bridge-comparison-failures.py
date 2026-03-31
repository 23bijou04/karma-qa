#!/usr/bin/env python3
"""Rerun the 23 bridge_comparison+ask_verified failures after the ASK fix.

Compares old vs new results to measure the fix impact.
Output: JSONL file + summary stats.
"""

import json
import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
_parent = os.path.dirname(PROJECT_ROOT)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import importlib.util

from dbpedia_qa.dbpedia import (
    DBpediaEntityLinker,
    DBpediaSPARQLClient,
    SchemaDiscovery,
)
from dbpedia_qa.models import extract_final_answer
from hybrid_qa import hybrid_answer
from llm_provider import create_llm_client

# Load metrics
_spec = importlib.util.spec_from_file_location(
    "metrics", os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
compute_em = _metrics.compute_em
compute_f1 = _metrics.compute_f1
compute_containment = _metrics.compute_containment

# --- Config ---
COVERAGE_FILE = os.path.join(
    PROJECT_ROOT, "data/eval_1000q/ground-truth/coverage_analysis_1000.json"
)
JUDGED_FILE = os.path.join(
    PROJECT_ROOT, "data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl"
)
OUTPUT_FILE = os.path.join(
    PROJECT_ROOT, "data/eval_1000q/hybrid/bridge_comparison_rerun.jsonl"
)


def find_affected_questions():
    """Extract question IDs that failed via bridge_comparison+ask_verified."""
    affected = {}
    with open(JUDGED_FILE) as f:
        for line in f:
            rec = json.loads(line)
            route = rec.get("route_source", "")
            em = rec.get("em", 0)
            judge_em = rec.get("judge_em", 0)
            if "bridge_comparison" in route and "ask_verified" in route:
                if em == 0 and judge_em == 0:
                    affected[rec["question_id"]] = rec
    return affected


def load_questions(question_ids):
    """Load full question records from coverage file."""
    questions = []
    with open(COVERAGE_FILE) as f:
        data = json.load(f)
    for item in data["questions"]:
        qid = item.get("question_id", item.get("_id", ""))
        if qid in question_ids:
            # Normalize field names for hybrid_answer
            item["_id"] = qid
            questions.append(item)
    return questions


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    model = "llama-3.3-70b-versatile"
    llm_client = create_llm_client("groq", api_key)
    kg_client = DBpediaSPARQLClient()
    linker = DBpediaEntityLinker()
    schema_disc = SchemaDiscovery(kg_client)

    # Find affected questions
    affected = find_affected_questions()
    print(f"Found {len(affected)} affected questions")

    questions = load_questions(set(affected.keys()))
    print(f"Loaded {len(questions)} questions from coverage file")

    # Track results
    results = []
    improved = 0
    same = 0
    regressed = 0

    for i, q in enumerate(questions, 1):
        qid = q["_id"]
        question = q["question"]
        gold = q.get("gold_answer", q.get("answer", ""))
        old_rec = affected[qid]

        print(f"\n[{i}/{len(questions)}] {question[:80]}")
        print(f"  Gold: {gold}")
        print(f"  Old predicted: {old_rec.get('predicted_answer', '')}")
        print(f"  Old route: {old_rec.get('route_source', '')}")

        try:
            result = hybrid_answer(
                question, api_key,
                model=model,
                llm_client=llm_client,
                kg_client=kg_client,
                linker=linker,
                schema_disc=schema_disc,
            )

            raw_answer = result.answer if result else ""
            predicted = extract_final_answer(raw_answer)
            route_source = result.route_source if result else ""
            method = result.method if result else ""

            em = compute_em(predicted, gold)
            f1 = compute_f1(predicted, gold)
            containment = compute_containment(predicted, gold)

            rec = {
                "question_id": qid,
                "question": question,
                "gold_answer": gold,
                "predicted_answer": predicted,
                "raw_answer": raw_answer[:500],
                "route_source": route_source,
                "method": method,
                "em": em,
                "f1": f1,
                "containment": containment,
                "old_route": old_rec.get("route_source", ""),
                "old_predicted": old_rec.get("predicted_answer", ""),
            }
            results.append(rec)

            # Compare with old
            if em > 0 or containment > 0.5:
                improved += 1
                status = "IMPROVED"
            else:
                same += 1
                status = "SAME (still wrong)"

            print(f"  New predicted: {predicted}")
            print(f"  New route: {route_source}")
            print(f"  EM={em} F1={f1:.2f} Containment={containment:.2f} → {status}")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "question_id": qid,
                "question": question,
                "gold_answer": gold,
                "error": str(e),
                "em": 0, "f1": 0, "containment": 0,
            })
            same += 1

        # Rate limit
        time.sleep(2)

    # Write results
    with open(OUTPUT_FILE, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    # Summary
    print("\n" + "=" * 60)
    print(f"RERUN SUMMARY ({len(results)} questions)")
    print(f"  Improved:  {improved} ({improved/len(results)*100:.1f}%)")
    print(f"  Same:      {same} ({same/len(results)*100:.1f}%)")
    print(f"  Regressed: {regressed} ({regressed/len(results)*100:.1f}%)")

    new_routes = {}
    for r in results:
        rs = r.get("route_source", "error")
        new_routes[rs] = new_routes.get(rs, 0) + 1
    print(f"\nNew route distribution:")
    for route, count in sorted(new_routes.items(), key=lambda x: -x[1]):
        print(f"  {route}: {count}")

    avg_em = sum(r["em"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    print(f"\nAvg EM: {avg_em:.3f}  Avg F1: {avg_f1:.3f}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
