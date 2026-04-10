#!/usr/bin/env python3
"""Convert LinearRAG predictions.json to project eval formats.

Produces:
- JSONL for judge_hybrid_eval.py (question_id, question, gold_answer, predicted_answer, question_type)
- JSON for unified_comparison.py (id, pred_answer, gold_answer, question_type)
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Convert LinearRAG predictions.json to project eval formats"
    )
    parser.add_argument("input", help="LinearRAG predictions.json path")
    parser.add_argument(
        "--jsonl-out",
        default="data/eval_1000q/linearrag/linearrag_1000q.jsonl",
    )
    parser.add_argument(
        "--json-out",
        default="data/eval_1000q/linearrag_predictions_1000.json",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        preds = json.load(f)

    # JSONL for judge script
    os.makedirs(os.path.dirname(args.jsonl_out), exist_ok=True)
    with open(args.jsonl_out, "w") as f:
        for p in preds:
            rec = {
                "question_id": p["id"],
                "question": p["question"],
                "question_type": p.get("question_type", ""),
                "gold_answer": p.get("gold_answer", ""),
                "predicted_answer": p.get("pred_answer", ""),
                "method": "linearrag",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # JSON for unified_comparison (already has correct field names)
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(preds)} records:")
    print(f"  JSONL: {args.jsonl_out}")
    print(f"  JSON:  {args.json_out}")

    # Sanity checks
    empty = sum(1 for p in preds if not p.get("pred_answer", "").strip())
    print(f"  Empty pred_answer: {empty}/{len(preds)}")

    types = {}
    for p in preds:
        t = p.get("question_type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"  Question types: {types}")


if __name__ == "__main__":
    main()
