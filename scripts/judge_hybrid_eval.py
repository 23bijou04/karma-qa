#!/usr/bin/env python3
"""LLM judge for hybrid evaluation results.

Reads a JSONL file from hybrid_batch_eval.py and adds a 'judge_correct'
boolean field using Groq LLM semantic equivalence check.
Resume-safe: skips already-judged question IDs.

Usage:
    GROQ_API_KEY=$KEY python3 scripts/judge_hybrid_eval.py \
        data/eval_1000q/hybrid_eval_346_revised.jsonl
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from groq import Groq  # noqa: E402

JUDGE_PROMPT = """Judge if predicted answer is semantically equivalent to gold answer.
Question: {question}
Gold: {gold}
Predicted: {pred}
Rules: "yes" if same entity/fact (even if more specific). "no" if wrong or empty.
Answer ONLY "yes" or "no":"""


def judge(client, question, gold, pred, model="llama-3.3-70b-versatile"):
    """Return True if LLM judge deems prediction semantically correct."""
    if not pred or pred.strip() == "":
        return False
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
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/judge_hybrid_eval.py <input.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = input_path.replace(".jsonl", "_judged.jsonl")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: set GROQ_API_KEY environment variable")
        sys.exit(1)

    client = Groq(api_key=api_key)

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
                    rec["predicted_answer"],
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
