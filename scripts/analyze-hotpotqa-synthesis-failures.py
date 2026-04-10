"""
Analyze synthesis failures in HotPotQA hybrid evaluation.
Synthesis failure: judge_correct=False AND gold answer appears in a sub_question_log answer.
Excludes judge errors (em=1.0 but judge_correct=False).
"""

import json
from pathlib import Path

ROOT = Path("/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa")
INPUT = ROOT / "data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged.jsonl"
OUTPUT_IDS = ROOT / "data/cross_benchmark/hotpotqa/synthesis_failure_19_ids.json"

# ── helpers ──────────────────────────────────────────────────────────────────

def gold_in_subq(record):
    """Return (found: bool, matches: list) where gold appears in any sub-Q answer."""
    gold = record.get("gold_answer", "").lower().strip()
    aliases = [a.lower().strip() for a in record.get("gold_aliases", [])]
    all_gold = [g for g in [gold] + aliases if g]

    matches = []
    for sq in record.get("sub_question_log", []):
        sq_ans = str(sq.get("answer", "")).lower().strip()
        for g in all_gold:
            if g in sq_ans:
                matches.append({
                    "sub_question": sq.get("sub_question", ""),
                    "answer": sq.get("answer", ""),
                    "matched_gold": g,
                })
    return len(matches) > 0, matches


def is_judge_error(record):
    """em=1 means predicted==gold; if judge_correct=False that's a judge error."""
    return record.get("em", 0) == 1.0


def categorize(record, matches):
    """Classify WHY synthesis failed. Returns (category, explanation)."""
    gold = record.get("gold_answer", "").lower().strip()
    pred = record.get("predicted_answer", "").lower().strip()
    sub_logs = record.get("sub_question_log", [])

    # wrong_hop: the matched gold appears in a sub-Q answer that is the
    # INTERMEDIATE result, and the LLM returned that intermediate instead of
    # continuing to the final hop.
    # Heuristic: predicted answer == a sub-Q answer AND that sub-Q answer
    # contains the gold but is not itself the final answer.
    for sq in sub_logs:
        sq_ans = str(sq.get("answer", "")).lower().strip()
        if gold in sq_ans and pred in sq_ans and pred != gold:
            return "wrong_hop", (
                f"Predicted '{record['predicted_answer']}' is part of the "
                f"sub-Q answer '{sq['answer']}' which contains gold "
                f"'{record['gold_answer']}', but wrong item selected from list."
            )

    # list_selection: sub-Q returned a list; synthesizer picked wrong item
    for sq in sub_logs:
        sq_ans = str(sq.get("answer", "")).lower().strip()
        if gold in sq_ans and ("," in sq_ans or " and " in sq_ans or " or " in sq_ans):
            # Is the predicted answer one of the other list items?
            if pred not in sq_ans and len(sq_ans) > len(gold) + 10:
                return "list_selection", (
                    f"Sub-Q answer is a list containing gold "
                    f"'{record['gold_answer']}' but predicted is "
                    f"'{record['predicted_answer']}' — wrong item selected."
                )
            if pred in sq_ans and pred != gold:
                return "list_selection", (
                    f"Sub-Q answer is a list; synthesizer picked "
                    f"'{record['predicted_answer']}' instead of "
                    f"'{record['gold_answer']}'."
                )

    # over_generalization: pred contains gold or gold contains pred (partial match)
    if gold in pred or pred in gold:
        return "over_generalization", (
            f"Gold '{record['gold_answer']}' and predicted "
            f"'{record['predicted_answer']}' are partial matches — "
            "synthesizer generalized or truncated the answer."
        )

    # wrong_hop: predicted is clearly an intermediate entity, not the final answer
    # Check if pred equals exactly a sub-Q answer
    for sq in sub_logs:
        sq_ans = str(sq.get("answer", "")).lower().strip()
        if pred == sq_ans or pred in sq_ans:
            return "wrong_hop", (
                f"Predicted '{record['predicted_answer']}' matches sub-Q answer "
                f"'{sq['answer']}' (intermediate hop), not the final answer "
                f"'{record['gold_answer']}'."
            )

    return "other", (
        f"Gold '{record['gold_answer']}' in sub-Qs but predicted "
        f"'{record['predicted_answer']}'. Does not fit standard categories."
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    records = []
    with open(INPUT) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Total records: {len(records)}")
    failures = [r for r in records if r.get("judge_correct") is False]
    print(f"Total judge_correct=False: {len(failures)}")

    # Find synthesis failures, exclude judge errors (em=1 but judge wrong)
    synthesis = []
    judge_errors = []
    for r in failures:
        if not r.get("sub_question_log"):
            continue
        found, matches = gold_in_subq(r)
        if not found:
            continue
        if is_judge_error(r):
            judge_errors.append(r)
            continue
        synthesis.append((r, matches))

    print(f"Judge errors excluded (em=1 but judge_correct=False): {len(judge_errors)}")
    print(f"True synthesis failures: {len(synthesis)}")
    print()

    # Categorize
    category_counts = {"list_selection": 0, "over_generalization": 0,
                       "wrong_hop": 0, "other": 0}
    ids = []

    sep = "=" * 80

    for i, (r, matches) in enumerate(synthesis):
        cat, explanation = categorize(r, matches)
        category_counts[cat] += 1
        ids.append(r["question_id"])

        print(sep)
        print(f"CASE {i+1:02d}  [{cat.upper()}]  ID: {r['question_id']}")
        print(f"Question  : {r['question']}")
        print(f"Gold      : {r['gold_answer']}")
        if r.get("gold_aliases"):
            print(f"Aliases   : {r['gold_aliases']}")
        print(f"Predicted : {r['predicted_answer']}")
        print(f"Route     : {r['route_source']}")
        print()
        print("Sub-questions:")
        for j, sq in enumerate(r.get("sub_question_log", []), 1):
            print(f"  [{j}] Q: {sq.get('sub_question','')}")
            print(f"      A: {sq.get('answer','')}")
        print()
        raw = r.get("raw_answer", "")
        if raw and raw != r.get("predicted_answer", ""):
            # truncate long raw answers
            display_raw = raw if len(raw) <= 600 else raw[:600] + "...[truncated]"
            print(f"Raw answer:\n{display_raw}")
        print()
        print(f"Why failed: {explanation}")
        print()

    # Summary
    print(sep)
    print("CATEGORY SUMMARY")
    print(sep)
    total = sum(category_counts.values())
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"  {cat:<25} {count:2d} / {total}  ({pct:.0f}%)")
    print()

    # Representative examples
    print("REPRESENTATIVE EXAMPLES PER CATEGORY")
    print()
    shown = set()
    for cat in ["list_selection", "over_generalization", "wrong_hop", "other"]:
        for i, (r, matches) in enumerate(synthesis):
            c, _ = categorize(r, matches)
            if c == cat and cat not in shown:
                shown.add(cat)
                print(f"  [{cat.upper()}] Q: {r['question']}")
                print(f"    Gold: {r['gold_answer']}  |  Pred: {r['predicted_answer']}")
                print()

    # Save IDs
    with open(OUTPUT_IDS, "w") as f:
        json.dump(ids, f, indent=2)
    print(f"Saved {len(ids)} question IDs to: {OUTPUT_IDS}")

    return ids, category_counts


if __name__ == "__main__":
    main()
