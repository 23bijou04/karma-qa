#!/usr/bin/env python3
"""Fix 3 confirmed data bugs in cross-benchmark evaluation JSONL files.

Fix #8  — Recompute containment for HotPotQA Hybrid rerun records (0.0 → 1.0 for 40 records)
Fix #3  — Backfill question_type in LinearRAG MuSiQue from question_id prefix
Fix #15 — Patch 6 alias-match records in LinearRAG MuSiQue where judge_correct=False but alias matches
"""

import importlib.util
import json
import os
import re
import string
from collections import Counter

# ---------------------------------------------------------------------------
# Bootstrap: load evaluation/metrics.py via importlib (same pattern as other scripts)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "metrics", os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
)
_metrics_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics_mod)
compute_containment = _metrics_mod.compute_containment
compute_em = _metrics_mod.compute_em
compute_f1 = _metrics_mod.compute_f1


# ---------------------------------------------------------------------------
# Normalize helper (matches evaluation/metrics.py normalize_answer logic)
# ---------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ===========================================================================
# Fix #8 — Recompute containment for HotPotQA Hybrid file
# ===========================================================================
def fix_hotpotqa_containment() -> None:
    path = os.path.join(
        PROJECT_ROOT,
        "data", "cross_benchmark", "hotpotqa",
        "hybrid_hotpotqa_1000q_judged.jsonl",
    )
    print("\n=== Fix #8: HotPotQA Hybrid — recompute containment ===")
    records = load_jsonl(path)

    old_sum = sum(r.get("containment", 0.0) for r in records)
    changed = 0

    for r in records:
        pred = r.get("predicted_answer") or ""
        gold = r.get("gold_answer") or ""
        aliases = r.get("gold_aliases") or []
        candidates = [gold] + [a for a in aliases if a]
        new_cont = max((compute_containment(pred, c) for c in candidates), default=0.0)
        old_cont = r.get("containment", 0.0)
        if new_cont != old_cont:
            r["containment"] = new_cont
            changed += 1

    new_sum = sum(r.get("containment", 0.0) for r in records)
    new_mean = new_sum / len(records)

    print(f"  Records changed: {changed} (expected 40)")
    print(f"  Containment 1.0 count: {sum(1 for r in records if r.get('containment') == 1.0)}")
    print(f"  Old sum: {old_sum:.0f}  →  New sum: {new_sum:.0f}  (delta +{new_sum - old_sum:.0f})")
    print(f"  Old mean: {old_sum/len(records):.4f}  →  New mean: {new_mean:.4f}")
    print(f"  Expected: 639/1000 = 0.6390, got: {new_sum:.0f}/1000")

    write_jsonl(path, records)
    print(f"  Written: {path}")


# ===========================================================================
# Fix #3 — Backfill question_type in LinearRAG MuSiQue
# ===========================================================================
def fix_musique_question_type(records: list[dict]) -> list[dict]:
    print("\n=== Fix #3: MuSiQue LinearRAG — backfill question_type ===")
    empty_before = sum(1 for r in records if not r.get("question_type"))
    print(f"  Empty question_type before: {empty_before}/1000")

    for r in records:
        qid = r.get("question_id", "")
        prefix = qid.split("__")[0] if "__" in qid else ""
        r["question_type"] = prefix

    dist = Counter(r["question_type"] for r in records)
    print("  Distribution after backfill:")
    for qt, cnt in sorted(dist.items()):
        print(f"    {qt}: {cnt}")

    expected = {"2hop": 518, "3hop1": 243, "3hop2": 73, "4hop1": 108, "4hop2": 27, "4hop3": 31}
    mismatches = [(k, v, expected.get(k)) for k, v in dist.items() if expected.get(k) != v]
    if mismatches:
        print("  WARNING — distribution mismatch:")
        for qt, got, exp in mismatches:
            print(f"    {qt}: got {got}, expected {exp}")
    else:
        print("  Distribution matches expected.")

    return records


# ===========================================================================
# Fix #15 — Patch alias-match records where judge_correct=False but alias matches
# ===========================================================================
KNOWN_IDS = {
    "2hop__95970_152907",
    "3hop1__106845_5188_86687",
    "2hop__860241_766393",
    "2hop__15650_77173",
    "2hop__143650_152023",
    "3hop1__9285_5188_86687",
}


def fix_musique_alias_patch(records: list[dict]) -> list[dict]:
    print("\n=== Fix #15: MuSiQue LinearRAG — patch alias-match records ===")
    jc_before = sum(1 for r in records if r.get("judge_correct"))
    print(f"  judge_correct before: {jc_before}/1000")

    flipped = []
    for r in records:
        if r.get("judge_correct"):
            continue
        pred = r.get("predicted_answer") or ""
        aliases = r.get("gold_aliases") or []
        if not aliases:
            continue
        norm_pred = normalize_answer(pred)
        if any(normalize_answer(a) == norm_pred for a in aliases if a):
            old_val = r["judge_correct"]
            r["judge_correct"] = True
            flipped.append({
                "question_id": r.get("question_id"),
                "predicted_answer": pred,
                "gold_answer": r.get("gold_answer"),
                "gold_aliases": aliases,
                "was": old_val,
            })

    jc_after = sum(1 for r in records if r.get("judge_correct"))
    print(f"  Flipped records ({len(flipped)}):")
    for f in flipped:
        print(f"    {f['question_id']} | pred={repr(f['predicted_answer'])} | gold={repr(f['gold_answer'])} | aliases={f['gold_aliases']}")

    # Verify against known IDs
    flipped_ids = {f["question_id"] for f in flipped}
    missing_known = KNOWN_IDS - flipped_ids
    extra_unknown = flipped_ids - KNOWN_IDS
    if missing_known:
        print(f"  WARNING — known IDs not flipped: {missing_known}")
    if extra_unknown:
        print(f"  INFO — additional records also flipped (not in known list): {extra_unknown}")

    print(f"  judge_correct after:  {jc_after}/1000  (expected 516)")
    return records


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    # Fix #8: independent file — process first
    fix_hotpotqa_containment()

    # Fix #3 + #15: same MuSiQue LinearRAG file — load once, apply both, write once
    musique_path = os.path.join(
        PROJECT_ROOT,
        "data", "cross_benchmark", "musique",
        "linearrag_musique_1000q_judged.jsonl",
    )
    print("\n--- Loading MuSiQue LinearRAG file ---")
    records = load_jsonl(musique_path)
    print(f"  Loaded {len(records)} records")

    records = fix_musique_question_type(records)
    records = fix_musique_alias_patch(records)

    write_jsonl(musique_path, records)
    print(f"\n  Written: {musique_path}")
    print("\nAll fixes complete.")


if __name__ == "__main__":
    main()
