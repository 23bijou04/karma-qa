#!/usr/bin/env python3
"""LLM judge for hybrid evaluation results.

Reads a JSONL file from hybrid_batch_eval.py and adds a 'judge_correct'
boolean field using LLM semantic equivalence check.
Resume-safe: skips already-judged question IDs.

Usage:
    python3 scripts/judge_hybrid_eval.py data/eval_1000q/hybrid_eval_1000.jsonl
    python3 scripts/judge_hybrid_eval.py data/eval.jsonl --provider github --model gpt-4o
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from llm_provider import create_llm_client  # noqa: E402

JUDGE_PROMPT = """Judge if the predicted answer is semantically equivalent to the gold answer.
Question: {question}
Gold: {gold}
Predicted: {pred}
Rules:
- "yes" if they refer to the same entity/fact even with different specificity
- "yes" if gold is contained in predicted as the same location/entity (e.g. "Munich" vs "Nymphenburg Palace, Munich, Germany" = yes)
- "yes" if predicted adds a title/honorific to the same person (e.g. "Henriette of Nassau" vs "Princess Henriette of Nassau" = yes)
- "yes" for equivalent date formats (e.g. "July 16, 2013" vs "2013-07-16" = yes)
- "no" if the predicted answer is wrong, empty, or a different entity
Answer ONLY "yes" or "no":"""


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation/whitespace for containment check."""
    return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()


def judge(client, question, gold, pred, model="llama-3.3-70b-versatile"):
    """Return True if LLM judge deems prediction semantically correct."""
    if not pred or pred.strip() == "":
        return False
    # Fast path: if gold is contained in predicted (or vice versa), it's correct
    g, p = _normalize(gold), _normalize(pred)
    if g and p and (g in p or p in g):
        return True
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=pred),
        }],
    )
    return resp.choices[0].message.content.strip().lower().startswith("yes")


def main():
    parser = argparse.ArgumentParser(description="LLM judge for hybrid evaluation results")
    parser.add_argument("input_file", help="Input JSONL file from hybrid_batch_eval.py")
    parser.add_argument("--provider", default="groq", choices=["groq", "github", "ollama"],
                        help="LLM provider (default: groq)")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="LLM model name")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: provider-specific env var)")
    args = parser.parse_args()

    input_path = args.input_file
    output_path = input_path.replace(".jsonl", "_judged.jsonl")

    try:
        client = create_llm_client(args.provider, api_key=args.api_key)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load already-judged IDs for resume (crash-resilient)
    done = set()
    done_correct = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        done.add(d["question_id"])
                        if d.get("judge_correct"):
                            done_correct += 1
                    except (json.JSONDecodeError, KeyError):
                        pass  # skip truncated lines from mid-write crash
        print(f"Resume: {len(done)} already judged in {output_path}")

    with open(input_path) as f:
        records = [json.loads(line.strip()) for line in f if line.strip()]

    remaining = [r for r in records if r["question_id"] not in done]
    print(f"Total: {len(records)}, remaining: {len(remaining)}")

    if not remaining:
        print("All questions already judged.")
        return

    correct_count = done_correct

    with open(output_path, "a") as out:
        for i, rec in enumerate(remaining):
            try:
                rec["judge_correct"] = judge(
                    client, rec["question"], rec["gold_answer"],
                    rec["predicted_answer"], model=args.model,
                )
            except Exception as e:
                print(f"  Error judging {rec['question_id']}: {e}")
                rec["judge_correct"] = False

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

            if rec["judge_correct"]:
                correct_count += 1

            if (i + 1) % 20 == 0:
                total_so_far = len(done) + i + 1
                print(f"  Judged {total_so_far}/{len(records)} "
                      f"({correct_count}/{total_so_far} correct)")

            time.sleep(0.5)  # rate limit courtesy

    # Final summary
    total = len(records)
    with open(output_path) as f:
        all_judged = [json.loads(line) for line in f if line.strip()]
    correct = sum(1 for r in all_judged if r.get("judge_correct"))
    print(f"\nJudge EM: {correct}/{total} = {correct / total * 100:.1f}%")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
