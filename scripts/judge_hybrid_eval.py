#!/usr/bin/env python3
"""LLM judge for hybrid evaluation results.

Reads a JSONL file from hybrid_batch_eval.py and adds a 'judge_correct'
boolean field using LLM semantic equivalence check.
Resume-safe: skips already-judged question IDs.

Note: default model (gemma3:12b via Ollama) has not been formally validated
as a judge. For thesis-grade results, validate against human annotations
on a 100-question sample, or use a stronger model (--provider groq --model
llama-3.3-70b-versatile).

Usage:
    python3 scripts/judge_hybrid_eval.py data/eval_1000q/hybrid/hybrid_eval_1000.jsonl
    python3 scripts/judge_hybrid_eval.py data/eval.jsonl --provider groq --model llama-3.3-70b-versatile
    python3 scripts/judge_hybrid_eval.py data/eval.jsonl --provider github --model gpt-4o
"""

import argparse
import json
import os
import re
import string
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from llm_provider import create_llm_client  # noqa: E402

JUDGE_SYSTEM = (
    "You are a strict evaluation judge for a QA benchmark. "
    "Determine if two answers refer to the exact same entity or fact. "
    "Precision matters more than recall — when in doubt, answer no."
)

JUDGE_PROMPT = """Determine if the predicted answer is semantically equivalent to the gold answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Matching rules (answer YES):
- Same entity/fact with different surface forms (e.g. "July 16, 2013" vs "2013-07-16")
- Gold contained in a more specific predicted answer (e.g. "Munich" vs "Nymphenburg Palace, Munich")
- Added title/honorific/disambiguator (e.g. "Henry III" vs "Henry III, Holy Roman Emperor")
- Full formal name of same person (e.g. "Giovanni Mario" vs "Giovanni Mario de Candia")
- Administrative region containing the exact gold location (e.g. gold "Theni" vs pred "Theni district")
- Unambiguous well-known city with or without country (e.g. gold "Florence, Italy" vs pred "Florence")
- Nationality adjective matching country name (e.g. gold "Italy" vs pred "Italian")

Rejection rules (answer NO):
- Wrong, empty, or completely different entity
- Predicted is broader/vaguer than gold (e.g. gold "Taipei Medical University Hospital" vs pred "Taipei")
- Year only when gold has full date (e.g. gold "December 25, 1757" vs pred "1757")
- Related but distinct entities (e.g. gold "Hollywood" vs pred "Los Angeles")
- Same first name but different person (e.g. gold "Paris" vs pred "Paris Hilton")
- Predicted lists multiple answers and gold is NOT clearly among them
- Completely different location even if in same region (e.g. gold "Theni" vs pred "Mallingapuram")

First, explain in ONE sentence why the answers match or don't.
Then write your verdict on a new line: VERDICT: yes or VERDICT: no"""


def _normalize(s: str) -> str:
    """Normalize for containment check — aligned with evaluation/metrics.py."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return " ".join(s.split())


def _word_set(s: str) -> set[str]:
    """Split normalized string into word set."""
    return set(s.split())


def _containment_precheck(gold: str, pred: str) -> bool | None:
    """Fast-path containment check: gold words subset of pred words.

    Returns True if gold is clearly contained in pred (word-level).
    Returns None if the LLM should decide.
    Only checks gold-in-pred direction to avoid false positives from
    vague/broader predictions (e.g. "Taipei" matching "Taipei Medical University").
    """
    g, p = _normalize(gold), _normalize(pred)
    if not g or not p:
        return None
    g_words = _word_set(g)
    # Require at least 2 gold words to avoid single-word substring traps
    # (e.g. "Paris" matching "Paris Hilton", "Jordan" matching "Michael Jordan")
    if len(g_words) < 2:
        return None
    return g_words.issubset(_word_set(p)) or None


def _parse_verdict(response: str) -> bool:
    """Extract VERDICT from LLM response. Falls back to last line."""
    text = response.strip().lower()
    if not text:
        return False
    # Search for "verdict:" anywhere in the response (last occurrence wins)
    if "verdict:" in text:
        remainder = text.rsplit("verdict:", 1)[1].strip()
        word = remainder.split()[0].rstrip(string.punctuation) if remainder else ""
        return word == "yes"
    # Fallback: check if last non-empty line is "yes"
    lines = text.splitlines()
    if not lines or not lines[-1].strip():
        return False
    word = lines[-1].strip().split()[0].rstrip(string.punctuation)
    return word == "yes"


def judge(client, question, gold, pred, model="gemma3:12b", aliases=None):
    """Return True if LLM judge deems prediction semantically correct."""
    if not pred or pred.strip() == "":
        return False
    # Alias pre-check: if benchmark declares aliases, exact match is authoritative
    # (e.g. MuSiQue provides answer_aliases; 2WikiMultiHop/HotPotQA have none)
    pred_norm = _normalize(pred)
    for candidate in [gold] + (aliases or []):
        if candidate and _normalize(candidate) == pred_norm:
            return True
    # Fast path: word-level containment (gold-in-pred only, min 2 words)
    precheck = _containment_precheck(gold, pred)
    if precheck is True:
        return True
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=100,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_PROMPT.format(
                question=question, gold=gold, pred=pred,
            )},
        ],
    )
    return _parse_verdict(resp.choices[0].message.content or "")


def main():
    parser = argparse.ArgumentParser(description="LLM judge for hybrid evaluation results")
    parser.add_argument("input_file", help="Input JSONL file from hybrid_batch_eval.py")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "groq", "github"],
                        help="LLM provider (default: ollama)")
    parser.add_argument("--model", default="gemma3:12b",
                        help="LLM model name (default: gemma3:12b for ollama)")
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

    records = []
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipping corrupt line {line_num} in {input_path}")
                continue

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
                    aliases=rec.get("gold_aliases", []),
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

            if args.provider != "ollama":
                time.sleep(0.5)  # rate limit courtesy for remote APIs

    # Final summary
    total = len(records)
    with open(output_path) as f:
        all_judged = [json.loads(line) for line in f if line.strip()]
    correct = sum(1 for r in all_judged if r.get("judge_correct"))
    print(f"\nJudge EM: {correct}/{total} = {correct / total * 100:.1f}%")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
