#!/usr/bin/env python3
"""Judge QA evaluation results using Claude API (Anthropic).

Reads a JSONL file, judges each record for semantic equivalence between
gold_answer and predicted_answer, writes to *_judged_claude_sonnet.jsonl.

Resume-safe: skips already-judged question IDs.

Usage:
    python3 scripts/judge-with-claude.py <input.jsonl> \
        --start 0 --end 100 \
        --api-key $ANTHROPIC_API_KEY
"""

import argparse
import json
import os
import re
import string
import sys
import time

# Auto-load .env from project root (needed for subagent environments)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

JUDGE_MODEL = "claude-sonnet-4-20250514"
JUDGE_NAME = "claude-sonnet-4"

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
    """Normalize for containment check."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return " ".join(s.split())


def _word_set(s: str) -> set[str]:
    return set(s.split())


def _containment_precheck(gold: str, pred: str) -> bool | None:
    """Fast-path: gold words subset of pred words (min 2 gold words)."""
    g, p = _normalize(gold), _normalize(pred)
    if not g or not p:
        return None
    g_words = _word_set(g)
    if len(g_words) < 2:
        return None
    return g_words.issubset(_word_set(p)) or None


def _parse_verdict(response: str) -> bool:
    """Extract VERDICT from LLM response."""
    text = response.strip().lower()
    if not text:
        return False
    if "verdict:" in text:
        remainder = text.rsplit("verdict:", 1)[1].strip()
        word = remainder.split()[0].rstrip(string.punctuation) if remainder else ""
        return word == "yes"
    lines = text.splitlines()
    if not lines or not lines[-1].strip():
        return False
    word = lines[-1].strip().split()[0].rstrip(string.punctuation)
    return word == "yes"


def judge_with_claude(client, question, gold, pred, aliases=None):
    """Return True if Claude judge deems prediction semantically correct."""
    if not pred or pred.strip() == "":
        return False
    # Alias exact match pre-check
    pred_norm = _normalize(pred)
    for candidate in [gold] + (aliases or []):
        if candidate and _normalize(candidate) == pred_norm:
            return True
    # Fast path: word-level containment
    precheck = _containment_precheck(gold, pred)
    if precheck is True:
        return True
    # Call Claude API
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=150,
        temperature=0,
        system=JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=pred),
        }],
    )
    return _parse_verdict(resp.content[0].text if resp.content else "")


def main():
    parser = argparse.ArgumentParser(description="Judge QA results with Claude Sonnet")
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--api-key", default=None, help="Anthropic API key")
    parser.add_argument("--output-suffix", default="_judged_claude_sonnet",
                        help="Suffix for output file (default: _judged_claude_sonnet)")
    args = parser.parse_args()

    # Import anthropic SDK
    try:
        import anthropic
    except ImportError:
        print("Error: pip install anthropic")
        sys.exit(1)

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY or pass --api-key")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    input_path = args.input_file
    # Output path: replace .jsonl with suffix.jsonl
    base = input_path.replace(".jsonl", "")
    output_path = f"{base}{args.output_suffix}.jsonl"

    # Load all records
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Slice to assigned range
    end = args.end if args.end is not None else len(records)
    subset = records[args.start:end]
    print(f"Processing {len(subset)} records [{args.start}:{end}] from {input_path}")
    print(f"Output: {output_path}")
    print(f"Judge model: {JUDGE_NAME}")

    # Load already-judged IDs for resume
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        done.add(d["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"Resume: {len(done)} already judged")

    remaining = [r for r in subset if r["question_id"] not in done]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All done.")
        return

    correct = 0
    errors = 0

    with open(output_path, "a") as out:
        for i, rec in enumerate(remaining):
            try:
                rec["judge_correct_claude"] = judge_with_claude(
                    client, rec["question"], rec["gold_answer"],
                    rec.get("predicted_answer", ""),
                    aliases=rec.get("gold_aliases", []),
                )
                rec["judge_model"] = JUDGE_NAME
            except Exception as e:
                print(f"  Error {rec.get('question_id', '?')}: {e}")
                rec["judge_correct_claude"] = False
                rec["judge_model"] = JUDGE_NAME
                errors += 1
                # Back off on rate limit errors
                if "rate" in str(e).lower() or "429" in str(e):
                    time.sleep(5)

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

            if rec["judge_correct_claude"]:
                correct += 1

            if (i + 1) % 25 == 0:
                total_so_far = i + 1
                print(f"  [{args.start}:{end}] {total_so_far}/{len(remaining)} "
                      f"({correct}/{total_so_far} correct, {errors} errors)")

            # Rate limit: ~50 req/min for Sonnet
            time.sleep(0.3)

    total = len(remaining)
    print(f"\nDone [{args.start}:{end}]: {correct}/{total} = {correct / total * 100:.1f}%")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
