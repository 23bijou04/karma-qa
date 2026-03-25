#!/usr/bin/env python3
"""Detect text-triple contradictions in the 2WikiMultiHopQA benchmark.

Phase 1: Spouse/parent evidence contradictions (text-based keyword scan).
Phase 2: Wikidata deprecated/deleted claims check (SPARQL-based).

Usage:
    python scripts/detect_benchmark_errors.py --phase 1 --verbose
    python scripts/detect_benchmark_errors.py --phase 2
    python scripts/detect_benchmark_errors.py --phase all
    python scripts/detect_benchmark_errors.py --output data/eval_1000q/benchmark_error_flags.json
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import date

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
_parent = os.path.dirname(PROJECT_ROOT)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROXIMITY_WINDOW = 200  # characters around entity name for keyword proximity check

# Strong unambiguous contradiction keywords — use proximity check only
HIGH_KEYWORDS = [
    "mistress", "concubine",
    "refused to marry", "never wed", "never married",
    "unmarried", "not married",
]
# Ambiguous keywords — require the keyword AND a relevant entity name in the same sentence
HIGH_SENTENCE_KEYWORDS = ["lover"]
MEDIUM_KEYWORDS = ["illegitimate", "companion", "partner"]
FALSE_POSITIVE_PHRASES = ["consort", "morganatic", "common-law wife", "common-law husband"]
PARENT_CONTRADICTION_KEYWORDS = ["adopted", "stepfather", "stepmother", "foster father", "foster mother"]

STOPWORDS = {"of", "the", "de", "von", "van", "le", "la", "du", "and", "in", "a", "an"}


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Unicode-normalize and lowercase a string (strips diacritics)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def title_matches_entity(title: str, entity_name: str) -> bool:
    """Return True if context paragraph title likely refers to entity_name.

    Uses deterministic matching only (no fuzzy):
    1. Substring match after Unicode normalization
    2. Word-overlap fallback: >=2 significant shared words
    """
    t = _normalize(title)
    e = _normalize(entity_name)

    if not t or not e:
        return False

    # Substring check
    if e in t or t in e:
        return True

    # Word overlap fallback
    t_words = {w for w in t.split() if w not in STOPWORDS and len(w) > 2}
    e_words = {w for w in e.split() if w not in STOPWORDS and len(w) > 2}
    shared = t_words & e_words
    return len(shared) >= 2


# ---------------------------------------------------------------------------
# Contradiction scanning helpers
# ---------------------------------------------------------------------------

def _has_false_positive(text_lower: str) -> bool:
    return any(fp in text_lower for fp in FALSE_POSITIVE_PHRASES)


# Matches 'Mistress [CapitalWord]' — honorific/title usage (e.g., "Mistress Page")
_HONORIFIC_MISTRESS_RE = re.compile(r"\bMistress\s+[A-Z]")
# Matches 'his/her/their mistress' — possessive: describes someone else's mistress
_POSSESSIVE_MISTRESS_RE = re.compile(r"\b(his|her|their)\s+\w*\s*mistress\b", re.IGNORECASE)


def _is_benign_mistress_use(original_text: str, pos_in_lower: int) -> bool:
    """Return True when 'mistress' at given position is NOT a spouse-contradiction keyword.

    Handles two benign patterns:
    - Honorific: 'Mistress Page' (capital M, followed by proper noun) — theatrical/archaic title
    - Possessive: 'his/her mistress Joan' — describes a third person, not the entity
    """
    # Check honorific pattern in original-case text (same byte offset)
    window = original_text[max(0, pos_in_lower - 5):pos_in_lower + 30]
    if _HONORIFIC_MISTRESS_RE.search(window):
        return True
    # Check possessive pattern: 'his/her mistress [name]' — mistress of someone else
    pre_window = original_text[max(0, pos_in_lower - 30):pos_in_lower + 10]
    if _POSSESSIVE_MISTRESS_RE.search(pre_window):
        return True
    return False


def _keyword_near_entity(
    text: str, text_lower: str, keyword: str, entity_variants: list[str]
) -> bool:
    """Return True if keyword appears within PROXIMITY_WINDOW chars of any entity variant.

    For 'mistress' specifically, also checks benign-use patterns (honorific, possessive).
    """
    idx = 0
    while True:
        pos = text_lower.find(keyword, idx)
        if pos == -1:
            break
        # Skip benign 'mistress' uses before proximity check
        if keyword == "mistress" and _is_benign_mistress_use(text, pos):
            idx = pos + 1
            continue
        window_start = max(0, pos - PROXIMITY_WINDOW)
        window_end = min(len(text_lower), pos + len(keyword) + PROXIMITY_WINDOW)
        window = text_lower[window_start:window_end]
        if any(_normalize(ev) in window for ev in entity_variants):
            return True
        idx = pos + 1
    return False


def _scan_paragraph(text: str, entity_name: str, spouse_name: str) -> tuple[str, list[str], str]:
    """Scan paragraph text for contradiction keywords.

    Returns (confidence, matched_keywords, snippet) or ('NONE', [], '').
    """
    text_lower = text.lower()
    entity_variants = [entity_name, spouse_name]

    if _has_false_positive(text_lower):
        return "NONE", [], ""

    found_high = []
    for kw in HIGH_KEYWORDS:
        if kw in text_lower and _keyword_near_entity(text, text_lower, kw, entity_variants):
            found_high.append(kw)

    # HIGH_SENTENCE_KEYWORDS: require keyword in same sentence as an entity name
    for kw in HIGH_SENTENCE_KEYWORDS:
        if kw in text_lower:
            for sentence in text.split("."):
                sent_lower = sentence.lower()
                if kw in sent_lower and any(_normalize(ev) in sent_lower for ev in entity_variants):
                    found_high.append(kw)
                    break

    if found_high:
        # Extract a snippet around the first matched keyword
        pos = text_lower.find(found_high[0])
        snippet_start = max(0, pos - 80)
        snippet_end = min(len(text), pos + len(found_high[0]) + 120)
        snippet = "..." + text[snippet_start:snippet_end].strip() + "..."
        return "HIGH", found_high, snippet

    found_medium = []
    for kw in MEDIUM_KEYWORDS:
        if kw in text_lower:
            # MEDIUM: keyword must be in a sentence that also mentions one entity
            for sentence in text.split("."):
                sent_lower = sentence.lower()
                if kw in sent_lower and any(_normalize(ev) in sent_lower for ev in entity_variants):
                    found_medium.append(kw)
                    break

    if found_medium:
        pos = text_lower.find(found_medium[0])
        snippet_start = max(0, pos - 80)
        snippet_end = min(len(text), pos + len(found_medium[0]) + 120)
        snippet = "..." + text[snippet_start:snippet_end].strip() + "..."
        return "MEDIUM", found_medium, snippet

    return "NONE", [], ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dev_full_path: str, coverage_path: str) -> tuple[list[str], dict]:
    """Load 1000q question IDs and build dev_full index keyed by question ID."""
    with open(coverage_path) as f:
        cov = json.load(f)
    question_ids = [q["question_id"] for q in cov["questions"]]

    with open(dev_full_path) as f:
        dev_full = json.load(f)
    index = {q["_id"]: q for q in dev_full}
    return question_ids, index


# ---------------------------------------------------------------------------
# Phase 1: Spouse contradictions
# ---------------------------------------------------------------------------

def find_spouse_contradictions(
    question_ids: list[str], index: dict, limit: int | None, verbose: bool
) -> tuple[list[dict], int]:
    """Scan spouse-evidence questions for text-triple contradictions."""
    flags = []
    scanned = 0

    for qid in question_ids:
        q = index.get(qid)
        if q is None:
            continue

        spouse_triples = [e for e in q.get("evidences", []) if len(e) >= 3 and e[1] == "spouse"]
        if not spouse_triples:
            continue

        scanned += 1
        if limit and scanned > limit:
            break

        for triple in spouse_triples:
            entity_name, _, spouse_name = triple[0], triple[1], triple[2]

            for title, sentences in q["context"]:
                if not (title_matches_entity(title, entity_name) or title_matches_entity(title, spouse_name)):
                    continue

                paragraph_text = " ".join(sentences)
                confidence, keywords, snippet = _scan_paragraph(paragraph_text, entity_name, spouse_name)

                if confidence == "NONE":
                    continue

                if verbose:
                    print(f"  [{confidence}] {qid[:8]}... | {entity_name} — {spouse_name}")
                    print(f"    Para title: {title!r}")
                    print(f"    Keywords: {keywords}")
                    print(f"    Snippet: {snippet[:120]}")

                flags.append({
                    "question_id": qid,
                    "question": q["question"],
                    "confidence": confidence,
                    "evidence_triple": triple,
                    "contradiction_keywords": keywords,
                    "context_title": title,
                    "context_snippet": snippet,
                    "gold_answer": q["answer"],
                })
                break  # One flag per (question, triple) — highest confidence found

    return flags, scanned


# ---------------------------------------------------------------------------
# Phase 1b: Parent (father/mother) adoption/step-parent check
# ---------------------------------------------------------------------------

def find_parent_contradictions(
    question_ids: list[str], index: dict, limit: int | None, verbose: bool
) -> tuple[list[dict], int]:
    """Light-touch scan for adopted/step-parent contradictions in father/mother triples."""
    flags = []
    scanned = 0

    for qid in question_ids:
        q = index.get(qid)
        if q is None:
            continue

        parent_triples = [
            e for e in q.get("evidences", [])
            if len(e) >= 3 and e[1] in ("father", "mother")
        ]
        if not parent_triples:
            continue

        scanned += 1
        if limit and scanned > limit:
            break

        for triple in parent_triples:
            child_name, relation, parent_name = triple[0], triple[1], triple[2]

            for title, sentences in q["context"]:
                if not (title_matches_entity(title, child_name) or title_matches_entity(title, parent_name)):
                    continue

                paragraph_text = " ".join(sentences).lower()
                found = [kw for kw in PARENT_CONTRADICTION_KEYWORDS if kw in paragraph_text]
                if not found:
                    continue

                if verbose:
                    print(f"  [MEDIUM/parent] {qid[:8]}... | {child_name} {relation} {parent_name}")
                    print(f"    Keywords: {found}")

                flags.append({
                    "question_id": qid,
                    "question": q["question"],
                    "confidence": "MEDIUM",
                    "evidence_triple": triple,
                    "contradiction_keywords": found,
                    "context_title": title,
                    "context_snippet": "",
                    "gold_answer": q["answer"],
                })
                break

    return flags, scanned


# ---------------------------------------------------------------------------
# Phase 2: Wikidata claim status check
# ---------------------------------------------------------------------------

def _build_entity_uri_map(coverage_path: str) -> dict[str, str]:
    """Build {entity_name: dbpedia_uri} from coverage analysis (for QID resolution)."""
    with open(coverage_path) as f:
        cov = json.load(f)
    uri_map: dict[str, str] = {}
    for q in cov["questions"]:
        for ent in q.get("entities", []):
            name = ent.get("name", "")
            uri = ent.get("uri", "")
            if name and uri:
                uri_map[name] = uri
    return uri_map


def run_phase2(
    question_ids: list[str],
    index: dict,
    coverage_path: str,
    phase1_flags: list[dict],
    cache_path: str,
    checkpoint_path: str,
    verbose: bool,
) -> list[dict]:
    """Run Phase 2: Wikidata deprecated/deleted claims scan.

    Resolves evidence entity names to QIDs, queries Wikidata for claim
    status across all 27 evidence properties, and cross-references with
    Phase 1 text flags to produce a merged error taxonomy.
    """
    from scripts.wikidata_claim_checker import (
        EVIDENCE_PROP_TO_PID,
        check_claim_status,
        classify_errors,
        resolve_entity_qids,
    )
    from scripts.wikidata_sparql_utils import DATE_PROPS, is_date_like

    uri_map = _build_entity_uri_map(coverage_path)

    # Collect all unique entity names from evidence triples
    all_names: set[str] = set()
    for qid in question_ids:
        q = index.get(qid)
        if q is None:
            continue
        for ev in q.get("evidences", []):
            if len(ev) >= 3:
                all_names.add(ev[0])  # subject
                all_names.add(ev[2])  # object/value

    print(f"  Resolving QIDs for {len(all_names)} unique entity names...")
    qid_cache = resolve_entity_qids(list(all_names), uri_map, cache_path)
    resolved = sum(1 for n in all_names if n in qid_cache)
    print(f"  Resolved: {resolved}/{len(all_names)} ({100*resolved//max(len(all_names),1)}%)")

    # Build triples_by_pid: {PID: [(subj_qid, obj_val_or_qid, question_id, is_date), ...]}
    triples_by_pid: dict[str, list[tuple]] = {}

    for qid in question_ids:
        q = index.get(qid)
        if q is None:
            continue
        for ev in q.get("evidences", []):
            if len(ev) < 3:
                continue
            subj_name, prop, obj_name = ev[0], ev[1], ev[2]
            pid_entry = EVIDENCE_PROP_TO_PID.get(prop)
            if pid_entry is None:
                continue
            subj_qid = qid_cache.get(subj_name)
            if not subj_qid:
                continue
            pids = pid_entry if isinstance(pid_entry, list) else [pid_entry]
            is_date = pids[0] in DATE_PROPS

            if is_date:
                obj_val = obj_name  # keep raw date string
            else:
                obj_val = qid_cache.get(obj_name, "")
                if not obj_val or is_date_like(obj_name):
                    continue  # unresolved entity — skip

            for pid in pids:
                triples_by_pid.setdefault(pid, []).append(
                    (subj_qid, obj_val, qid, is_date)
                )

    total_triples = sum(len(v) for v in triples_by_pid.values())
    print(f"  Checking {total_triples} evidence triples across {len(triples_by_pid)} properties...")

    claim_results = check_claim_status(triples_by_pid, checkpoint_path)

    deprecated = sum(1 for s in claim_results.values() if s == "DEPRECATED")
    not_found = sum(1 for s in claim_results.values() if s == "NOT_FOUND")
    print(f"  Results: {deprecated} DEPRECATED, {not_found} NOT_FOUND")

    merged = classify_errors(phase1_flags, claim_results)
    if verbose:
        for entry in merged:
            cat = entry.get("error_category", "?")
            print(f"  [{cat}] {entry.get('question_id', '')[:12]}... {entry.get('question', '')[:60]}")

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect text-triple contradictions in 2WikiMultiHopQA")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all",
                        help="Which detection phase to run (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max questions to process per phase (for testing)")
    parser.add_argument("--output", default="data/eval_1000q/benchmark_error_flags.json",
                        help="Output JSON path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dev_full_path = os.path.join(PROJECT_ROOT, "data", "dev_full.json")
    coverage_path = os.path.join(PROJECT_ROOT, "data", "eval_1000q", "coverage_analysis_1000.json")
    output_path = os.path.join(PROJECT_ROOT, args.output) if not os.path.isabs(args.output) else args.output
    cache_path = os.path.join(PROJECT_ROOT, "data", "eval_1000q", "entity_qid_cache.json")
    checkpoint_path = os.path.join(PROJECT_ROOT, "data", "eval_1000q", "wikidata_scan_checkpoint.json")

    print("Loading data...")
    question_ids, index = load_data(dev_full_path, coverage_path)

    all_flags = []
    summary_lines = []

    if args.phase in ("1", "all"):
        print("\n=== Phase 1: Spouse contradiction scan ===")
        spouse_flags, spouse_scanned = find_spouse_contradictions(
            question_ids, index, args.limit, args.verbose
        )
        high = [f for f in spouse_flags if f["confidence"] == "HIGH"]
        medium = [f for f in spouse_flags if f["confidence"] == "MEDIUM"]
        print(f"Scanned {spouse_scanned} spouse questions")
        print(f"Flags: {len(high)} HIGH, {len(medium)} MEDIUM")
        all_flags.extend(spouse_flags)
        summary_lines.append(f"Spouse scan: {spouse_scanned} scanned, {len(high)} HIGH, {len(medium)} MEDIUM")

    if args.phase in ("all",):
        print("\n=== Phase 1b: Parent contradiction scan ===")
        parent_flags, parent_scanned = find_parent_contradictions(
            question_ids, index, args.limit, args.verbose
        )
        print(f"Scanned {parent_scanned} parent questions")
        print(f"Flags: {len(parent_flags)} MEDIUM")
        all_flags.extend(parent_flags)
        summary_lines.append(f"Parent scan: {parent_scanned} scanned, {len(parent_flags)} MEDIUM")

    # Deduplicate Phase 1: one flag per question (highest confidence wins)
    seen: dict[str, dict] = {}
    for flag in all_flags:
        qid = flag["question_id"]
        if qid not in seen or flag["confidence"] == "HIGH":
            seen[qid] = flag
    phase1_flags = list(seen.values())

    merged_flags = phase1_flags  # default: Phase 1 only

    if args.phase in ("2", "all"):
        # For --phase 2 without prior Phase 1 run, load existing Phase 1 output
        if args.phase == "2" and not phase1_flags:
            try:
                with open(output_path) as f:
                    existing = json.load(f)
                phase1_flags = existing.get("flags", [])
                print(f"Loaded {len(phase1_flags)} Phase 1 flags from {output_path}")
            except FileNotFoundError:
                phase1_flags = []

        print("\n=== Phase 2: Wikidata deprecated/deleted claims ===")
        merged_flags = run_phase2(
            question_ids, index, coverage_path,
            phase1_flags, cache_path, checkpoint_path, args.verbose,
        )
        summary_lines.append(f"Phase 2 Wikidata scan: {len(merged_flags)} total flagged questions")

    categories = {}
    for f in merged_flags:
        cat = f.get("error_category")
        if cat:
            categories[cat] = categories.get(cat, 0) + 1

    output = {
        "scan_date": date.today().isoformat(),
        "method": "text-triple-contradiction + wikidata-claim-status"
        if args.phase in ("2", "all") else "text-triple-contradiction",
        "scope": "spouse + parent evidence + all 27 evidence properties, 1000q subset",
        "total_flags": len(merged_flags),
        "high_confidence": sum(1 for f in merged_flags if f.get("confidence") == "HIGH"),
        "medium_confidence": sum(1 for f in merged_flags if f.get("confidence") == "MEDIUM"),
        "error_categories": categories,
        "flags": merged_flags,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Summary ===")
    for line in summary_lines:
        print(f"  {line}")
    print(f"  Total unique questions flagged: {len(merged_flags)}")
    if categories:
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count}")
    print(f"  Output written to: {output_path}")

    # Show HIGH-confidence flags (Phase 1 only)
    high_flags = [f for f in merged_flags if f.get("confidence") == "HIGH"]
    if high_flags:
        print(f"\n--- HIGH confidence flags ({len(high_flags)}) ---")
        for f in high_flags:
            print(f"  [{f['question_id'][:12]}] {f.get('question', '')}")
            print(f"    Triple: {f.get('evidence_triple', '')}")
            print(f"    Keywords: {f.get('contradiction_keywords', '')}")
            print(f"    Para: {f.get('context_title', '')!r}")
            print()


if __name__ == "__main__":
    main()
