#!/usr/bin/env python3
"""Corrected ensemble entity-linking test: mention-only gold + strict URI matching.

Fixes from the original test_ensemble_el_live.py:
  1. Gold entities filtered to MENTION-ONLY (name appears in question text)
  2. Strict URI matching (no label-substring fallback)
  3. Context-augmented WikiSearch queries for disambiguation
  4. Scaled to 100 questions (25 per type, seed=42)

Usage:
    source venv/bin/activate
    python scripts/test_ensemble_el_corrected.py [--limit N]
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import requests
import spacy

# ── project root on sys.path ──────────────────────────────────────────────────
_root = Path(__file__).resolve().parent.parent
_parent_dir = str(_root.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from dbpedia_qa.dbpedia.entity_linker import (  # noqa: E402
    _MENTION_BLOCKLIST,
    _STRIP_WORDS,
    DBpediaEntityLinker,
)

# ── constants ─────────────────────────────────────────────────────────────────
COVERAGE_PATH = _root / "data" / "eval_1000q" / "coverage_analysis_1000.json"
REPORT_PATH = _root / "docs" / "ensemble-el-corrected-results.md"
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "DBpediaQA/1.0 (research; ensemble-el-corrected)"}

SPOTLIGHT_DELAY = 1.0
WIKI_DELAY = 0.5

_nlp = spacy.load("en_core_web_sm")


# ─────────────────────────────────────────────────────────────────────────────
# Mention-only gold filtering (key fix)
# ─────────────────────────────────────────────────────────────────────────────

def get_mention_entities(question_text: str, gold_entities: list[dict]) -> list[dict]:
    """Filter gold entities to those whose name appears in the question text."""
    q_lower = question_text.lower()
    mentions = []
    for e in gold_entities:
        name_lower = e["name"].lower()
        # Strip disambiguation suffix: "Thomas Jefferson (film)" → "Thomas Jefferson"
        base_name = re.sub(r'\s*\([^)]*\)\s*$', '', name_lower).strip()
        if base_name in q_lower or name_lower in q_lower:
            mentions.append(e)
    return mentions


def get_bridge_entities(question_text: str, gold_entities: list[dict]) -> list[dict]:
    """Return gold entities NOT mentioned in the question text (bridge entities)."""
    mention_uris = {e["uri"] for e in get_mention_entities(question_text, gold_entities)}
    return [e for e in gold_entities if e["uri"] not in mention_uris]


# ─────────────────────────────────────────────────────────────────────────────
# Strict URI matching (no label fallback)
# ─────────────────────────────────────────────────────────────────────────────

def norm_uri(uri: str) -> str:
    return unquote(uri).lower().replace(" ", "_").replace("-", "_").rstrip("/")


def uri_matches(predicted_uri: str, gold_uri: str) -> bool:
    """Strict URI match with normalization only."""
    return norm_uri(predicted_uri) == norm_uri(gold_uri)


# ─────────────────────────────────────────────────────────────────────────────
# Question selection — 100 questions, 25 per type
# ─────────────────────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def select_questions(questions: list[dict], per_type: int = 25) -> list[dict]:
    """Return questions stratified by type (per_type each, seed=42)."""
    rng = random.Random(42)
    by_type: dict[str, list[dict]] = {}
    for q in questions:
        qt = q.get("question_type", "unknown")
        by_type.setdefault(qt, []).append(q)

    selected = []
    for qt in sorted(by_type.keys()):
        pool = list(by_type[qt])
        rng.shuffle(pool)
        for q in pool[:per_type]:
            selected.append({**q, "_type_bucket": qt})

    rng.shuffle(selected)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Mention extraction (mirrors entity_linker._extract_mentions)
# ─────────────────────────────────────────────────────────────────────────────

def extract_mentions(question: str) -> list[str]:
    """Extract entity mentions using spaCy NER + noun chunks."""
    doc = _nlp(question)
    mentions: list[str] = []

    # Quoted strings
    mentions.extend(re.findall(r'"([^"]+)"', question))

    # Noun chunks (strip leading stop words)
    for chunk in doc.noun_chunks:
        words = chunk.text.split()
        while words and words[0].lower() in _STRIP_WORDS:
            words.pop(0)
        text = " ".join(words).strip()
        if text:
            mentions.append(text)

    # Named entities
    for ent in doc.ents:
        if ent.label_ not in ("CARDINAL", "ORDINAL", "QUANTITY", "PERCENT"):
            if " or " in ent.text:
                for part in ent.text.split(" or "):
                    part = part.strip()
                    if part:
                        mentions.append(part)
            else:
                mentions.append(ent.text)

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for m in mentions:
        key = m.lower().strip()
        if key not in seen and len(key) > 1 and key not in _MENTION_BLOCKLIST:
            seen.add(key)
            unique.append(m)

    # Remove substrings
    final: list[str] = []
    for m in unique:
        if not any(
            m.lower() != o.lower() and m.lower() in o.lower()
            for o in unique
        ):
            final.append(m)
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Context-augmented WikiSearch
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_SUFFIXES = {
    "film": "film", "movie": "film", "album": "album", "song": "song",
    "band": "band", "book": "book", "novel": "novel", "series": "TV series",
    "show": "TV series",
}


def wiki_search_with_context(
    mention: str, question: str, limit: int = 5
) -> list[dict]:
    """Search Wikipedia with question context for disambiguation."""
    query = mention
    q_lower = question.lower()
    for keyword, suffix in _TYPE_SUFFIXES.items():
        if keyword in q_lower and suffix not in mention.lower():
            query = f"{mention} {suffix}"
            break

    try:
        resp = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srnamespace": 0,
                "format": "json",
            },
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("query", {}).get("search", [])
    except Exception as exc:
        print(f"  WikiSearch error for '{query}': {exc}")
        return []


def title_to_dbpedia_uri(title: str) -> str:
    return "http://dbpedia.org/resource/" + title.replace(" ", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Scoring (strict URI only)
# ─────────────────────────────────────────────────────────────────────────────

def score_approach(
    found_entities: list[dict],
    gold_entities: list[dict],
) -> dict:
    """Score found entities against gold using strict URI match."""
    found_count = 0
    details = []
    for gold in gold_entities:
        matched = None
        for fe in found_entities:
            if uri_matches(fe["uri"], gold["uri"]):
                matched = fe
                break
        found_count += 1 if matched else 0
        details.append({
            "gold_name": gold["name"],
            "gold_uri": gold["uri"],
            "found": matched is not None,
            "found_uri": matched["uri"] if matched else None,
            "found_source": matched.get("source", "?") if matched else None,
        })
    total = len(gold_entities)
    return {
        "found": found_count,
        "total": total,
        "recall": found_count / total if total else 1.0,
        "details": details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Approach A: Entity Linker (Spotlight + bif:contains + type validation)
# ─────────────────────────────────────────────────────────────────────────────

_linker: DBpediaEntityLinker | None = None


def get_linker() -> DBpediaEntityLinker:
    global _linker
    if _linker is None:
        _linker = DBpediaEntityLinker(min_confidence=0.5, min_support=20)
    return _linker


def approach_spotlight(question: str) -> tuple[list[dict], float, str]:
    """Run entity linker (Spotlight + bif:contains + type validation)."""
    t0 = time.perf_counter()
    status = "ok"
    try:
        linker = get_linker()
        entities_raw = linker.link_entities(question, use_cache=False)
        elapsed = (time.perf_counter() - t0) * 1000
        entities = [
            {"uri": e.uri, "label": e.label, "source": "spotlight"}
            for e in entities_raw
        ]
        time.sleep(SPOTLIGHT_DELAY)
        return entities, elapsed, status
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        status = f"error: {exc}"
        time.sleep(SPOTLIGHT_DELAY)
        return [], elapsed, status


# ─────────────────────────────────────────────────────────────────────────────
# Approach B: WikiSearch-only (with context augmentation)
# ─────────────────────────────────────────────────────────────────────────────

def approach_wiki_search(question: str) -> tuple[list[dict], float, str]:
    """Run WikiSearch-only EL with context augmentation."""
    t0 = time.perf_counter()
    status = "ok"
    mentions = extract_mentions(question)
    entities: list[dict] = []
    seen_uris: set[str] = set()

    for mention in mentions:
        results = wiki_search_with_context(mention, question, limit=5)
        time.sleep(WIKI_DELAY)
        if results:
            top1 = results[0]
            uri = title_to_dbpedia_uri(top1["title"])
            if uri not in seen_uris:
                entities.append({
                    "uri": uri,
                    "label": top1["title"],
                    "source": "wikisearch",
                    "mention": mention,
                })
                seen_uris.add(uri)

    elapsed = (time.perf_counter() - t0) * 1000
    return entities, elapsed, status


# ─────────────────────────────────────────────────────────────────────────────
# Approach C: Ensemble (Spotlight + context-augmented WikiSearch for unlinked)
# ─────────────────────────────────────────────────────────────────────────────

def _find_unlinked_mentions(
    mentions: list[str], spotlight_entities: list[dict]
) -> list[str]:
    """Find mentions not covered by Spotlight results."""
    surface_forms = [e["label"] for e in spotlight_entities]
    unlinked = []
    for mention in mentions:
        m_lower = mention.lower().strip()
        m_words = set(m_lower.split())
        matched = False
        for sf in surface_forms:
            sf_lower = sf.lower().strip()
            sf_words = set(sf_lower.split())
            intersection = m_words & sf_words
            union = m_words | sf_words
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= 0.5 or m_lower in sf_lower:
                matched = True
                break
        if not matched:
            unlinked.append(mention)
    return unlinked


def approach_ensemble(
    question: str,
    spotlight_entities: list[dict],
    spotlight_elapsed_ms: float,
) -> tuple[list[dict], float, str]:
    """Ensemble: Spotlight + context-augmented WikiSearch for unlinked mentions."""
    t0 = time.perf_counter()
    status = "ok"

    mentions = extract_mentions(question)
    unlinked = _find_unlinked_mentions(mentions, spotlight_entities)

    ensemble = list(spotlight_entities)
    seen_uris = {e["uri"] for e in ensemble}

    for mention in unlinked:
        results = wiki_search_with_context(mention, question, limit=5)
        time.sleep(WIKI_DELAY)
        if results:
            top1 = results[0]
            uri = title_to_dbpedia_uri(top1["title"])
            if uri not in seen_uris:
                ensemble.append({
                    "uri": uri,
                    "label": top1["title"],
                    "source": "ensemble_wikisearch",
                    "mention": mention,
                })
                seen_uris.add(uri)

    wiki_elapsed = (time.perf_counter() - t0) * 1000
    total_elapsed = spotlight_elapsed_ms + wiki_elapsed
    return ensemble, total_elapsed, status


# ─────────────────────────────────────────────────────────────────────────────
# Per-question runner
# ─────────────────────────────────────────────────────────────────────────────

def _short_uri(uri: str) -> str:
    return uri.replace("http://dbpedia.org/resource/", "dbr:")


def run_question(q: dict) -> dict:
    question = q["question"]
    all_gold = [
        {"name": e["name"], "uri": e["uri"]}
        for e in q.get("entities", [])
        if e.get("exists") and e.get("uri")
    ]
    mention_gold = get_mention_entities(question, all_gold)
    bridge_gold = get_bridge_entities(question, all_gold)
    qtype = q.get("question_type", "?")

    print(f"\n{'─' * 70}")
    print(f"[{qtype.upper()}] {question}")
    print(f"  Mention gold: {[g['name'] for g in mention_gold]}")
    print(f"  Bridge gold (excluded): {[g['name'] for g in bridge_gold]}")

    # A: Spotlight
    sp_entities, sp_elapsed, sp_status = approach_spotlight(question)
    sp_score = score_approach(sp_entities, mention_gold)
    print(f"  A) EntityLinker [{sp_status}] ({sp_elapsed:.0f}ms): "
          f"{[_short_uri(e['uri']) for e in sp_entities] or '(none)'}")
    print(f"     Mention recall: {sp_score['found']}/{sp_score['total']}")

    # B: WikiSearch
    ws_entities, ws_elapsed, ws_status = approach_wiki_search(question)
    ws_score = score_approach(ws_entities, mention_gold)
    print(f"  B) WikiSearch [{ws_status}] ({ws_elapsed:.0f}ms): "
          f"{[_short_uri(e['uri']) for e in ws_entities] or '(none)'}")
    print(f"     Mention recall: {ws_score['found']}/{ws_score['total']}")

    # C: Ensemble
    ens_entities, ens_elapsed, ens_status = approach_ensemble(
        question, sp_entities, sp_elapsed
    )
    ens_score = score_approach(ens_entities, mention_gold)
    print(f"  C) Ensemble [{ens_status}] ({ens_elapsed:.0f}ms): "
          f"{[_short_uri(e['uri']) for e in ens_entities] or '(none)'}")
    print(f"     Mention recall: {ens_score['found']}/{ens_score['total']}")

    # Classify each mention gold entity by comparing SP vs actual ENS results.
    # Using ENS (not WS-only) because ensemble only searches UNLINKED mentions,
    # so WS-only may find things the ensemble never actually searches for.
    ensemble_helps = []   # ENS finds what SP misses
    ensemble_hurts = []   # SP finds but ENS doesn't (shouldn't happen: ENS ⊇ SP)
    both_missed = []

    for d_sp, d_ens in zip(sp_score["details"], ens_score["details"]):
        name = d_sp["gold_name"]
        if not d_sp["found"] and d_ens["found"]:
            ensemble_helps.append({
                "name": name,
                "gold_uri": d_sp["gold_uri"],
                "sp_uri": None,
                "ws_uri": d_ens["found_uri"],
            })
        elif d_sp["found"] and not d_ens["found"]:
            ensemble_hurts.append({
                "name": name,
                "gold_uri": d_sp["gold_uri"],
                "sp_uri": d_sp["found_uri"],
                "ws_uri": None,
            })
        elif not d_sp["found"] and not d_ens["found"]:
            # Check what URIs each returned (wrong ones)
            sp_wrong = None
            for fe in sp_entities:
                if name.lower() in fe["label"].lower() or fe["label"].lower() in name.lower():
                    sp_wrong = fe["uri"]
                    break
            ens_wrong = None
            for fe in ens_entities:
                m = fe.get("mention", fe.get("label", ""))
                if name.lower() in m.lower() or m.lower() in name.lower():
                    ens_wrong = fe["uri"]
                    break
            both_missed.append({
                "name": name,
                "gold_uri": d_sp["gold_uri"],
                "sp_wrong_uri": sp_wrong,
                "ws_wrong_uri": ens_wrong,
            })

    return {
        "question_id": q["question_id"],
        "question": question,
        "question_type": qtype,
        "mention_gold": mention_gold,
        "bridge_gold": bridge_gold,
        "spotlight": {
            "entities": sp_entities,
            "score": sp_score,
            "elapsed_ms": sp_elapsed,
            "status": sp_status,
        },
        "wikisearch": {
            "entities": ws_entities,
            "score": ws_score,
            "elapsed_ms": ws_elapsed,
            "status": ws_status,
        },
        "ensemble": {
            "entities": ens_entities,
            "score": ens_score,
            "elapsed_ms": ens_elapsed,
            "status": ens_status,
        },
        "ensemble_helps": ensemble_helps,
        "ensemble_hurts": ensemble_hurts,
        "both_missed": both_missed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    total_mention = sum(len(r["mention_gold"]) for r in results)
    total_bridge = sum(len(r["bridge_gold"]) for r in results)
    sp_found = sum(r["spotlight"]["score"]["found"] for r in results)
    ws_found = sum(r["wikisearch"]["score"]["found"] for r in results)
    ens_found = sum(r["ensemble"]["score"]["found"] for r in results)

    helps = sum(len(r["ensemble_helps"]) for r in results)
    hurts = sum(len(r["ensemble_hurts"]) for r in results)
    missed = sum(len(r["both_missed"]) for r in results)

    return {
        "n_questions": len(results),
        "total_mention_entities": total_mention,
        "total_bridge_entities": total_bridge,
        "spotlight_found": sp_found,
        "wikisearch_found": ws_found,
        "ensemble_found": ens_found,
        "spotlight_recall": sp_found / total_mention if total_mention else 0,
        "wikisearch_recall": ws_found / total_mention if total_mention else 0,
        "ensemble_recall": ens_found / total_mention if total_mention else 0,
        "ensemble_helps": helps,
        "ensemble_hurts": hurts,
        "both_missed": missed,
    }


def type_stats(results: list[dict]) -> dict[str, dict]:
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r)
    return {qt: aggregate(subset) for qt, subset in sorted(by_type.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(results: list[dict], agg: dict, type_agg: dict[str, dict]) -> None:
    lines = []
    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    lines += [
        "# Corrected Ensemble Entity Linking Test Results",
        "",
        f"**Generated**: {now}",
        f"**Questions tested**: {agg['n_questions']}",
        f"**Mention entities evaluated**: {agg['total_mention_entities']} "
        f"(bridge excluded: {agg['total_bridge_entities']})",
        "**Matching**: Strict URI only (no label-substring fallback)",
        "",
        "## Methodology Fixes",
        "",
        "1. **Mention-only gold**: Only entities whose name appears in the question text",
        "2. **Strict URI matching**: Normalized URI comparison, no label-substring fallback",
        "3. **Context-augmented WikiSearch**: Type hints from question appended to queries",
        "4. **100 questions**: 25 per type (stratified, seed=42)",
        "",
    ]

    # Executive summary
    lines += [
        "## Executive Summary",
        "",
        "| Approach | Mention Recall | Found/Total |",
        "|----------|---------------|-------------|",
        f"| Entity Linker (Spotlight+bif:contains) | {agg['spotlight_recall']*100:.1f}% "
        f"| {agg['spotlight_found']}/{agg['total_mention_entities']} |",
        f"| WikiSearch-only | {agg['wikisearch_recall']*100:.1f}% "
        f"| {agg['wikisearch_found']}/{agg['total_mention_entities']} |",
        f"| Ensemble | {agg['ensemble_recall']*100:.1f}% "
        f"| {agg['ensemble_found']}/{agg['total_mention_entities']} |",
        "",
        f"Ensemble gain over Spotlight: "
        f"+{(agg['ensemble_recall'] - agg['spotlight_recall'])*100:.1f}pp",
        "",
        "| Category | Count |",
        "|----------|-------|",
        f"| Ensemble helps (ENS finds what EL misses) | {agg['ensemble_helps']} |",
        f"| Ensemble neutral (EL finds, ENS doesn't) | {agg['ensemble_hurts']} |",
        f"| Both missed | {agg['both_missed']} |",
        "",
    ]

    # By question type
    lines += ["## By Question Type", ""]
    lines += [
        "| Type | N | Mention Ents | EL Recall | WS Recall | ENS Recall | Gain |",
        "|------|---|-------------|-----------|-----------|------------|------|",
    ]
    for qt, s in type_agg.items():
        gain = (s["ensemble_recall"] - s["spotlight_recall"]) * 100
        lines.append(
            f"| {qt} | {s['n_questions']} | {s['total_mention_entities']} "
            f"| {s['spotlight_recall']*100:.1f}% "
            f"| {s['wikisearch_recall']*100:.1f}% "
            f"| {s['ensemble_recall']*100:.1f}% "
            f"| {gain:+.1f}pp |"
        )
    lines.append("")

    # Where ensemble helps
    helps = [
        (r, h)
        for r in results
        for h in r["ensemble_helps"]
    ]
    if helps:
        lines += [
            "## Where Ensemble Helps (ENS finds what Entity Linker misses)",
            "",
            "| Question (truncated) | Entity | Gold URI | WS URI |",
            "|---------------------|--------|----------|--------|",
        ]
        for r, h in helps:
            q_short = r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"]
            lines.append(
                f"| {q_short} | {h['name']} "
                f"| `{_short_uri(h['gold_uri'])}` "
                f"| `{_short_uri(h['ws_uri'])}` |"
            )
        lines.append("")

    # Both missed
    missed = [
        (r, m)
        for r in results
        for m in r["both_missed"]
    ]
    if missed:
        lines += [
            "## Both Missed (mention entity not found by either)",
            "",
            "| Question (truncated) | Entity | Gold URI | SP wrong | WS wrong |",
            "|---------------------|--------|----------|----------|----------|",
        ]
        for r, m in missed:
            q_short = r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"]
            sp_w = _short_uri(m["sp_wrong_uri"]) if m["sp_wrong_uri"] else "—"
            ws_w = _short_uri(m["ws_wrong_uri"]) if m["ws_wrong_uri"] else "—"
            lines.append(
                f"| {q_short} | {m['name']} "
                f"| `{_short_uri(m['gold_uri'])}` "
                f"| `{sp_w}` | `{ws_w}` |"
            )
        lines.append("")

    # Bridge entity summary
    total_bridge = agg["total_bridge_entities"]
    total_all = agg["total_mention_entities"] + total_bridge
    lines += [
        "## Bridge Entities (excluded from eval)",
        "",
        f"Total bridge entities: {total_bridge} "
        f"({total_bridge/total_all*100:.0f}% of all gold entities)" if total_all else "",
        "",
    ]

    # Recommendation
    gain = (agg["ensemble_recall"] - agg["spotlight_recall"]) * 100
    lines += ["## Recommendation", ""]
    if agg["ensemble_helps"] >= 3:
        lines += [
            "**PROCEED** with ensemble implementation.",
            f"Ensemble delivers +{gain:.1f}pp recall and fixes "
            f"{agg['ensemble_helps']} entities that the entity linker misses.",
        ]
    elif gain >= 1:
        lines += [
            "**PROCEED WITH CAVEATS** — marginal improvement.",
            f"+{gain:.1f}pp gain with {agg['ensemble_helps']} fixes.",
        ]
    else:
        lines += [
            "**HOLD** — ensemble adds minimal improvement.",
            f"Only +{gain:.1f}pp gain; entity linker already covers most mention entities.",
        ]
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected ensemble EL test")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max questions per type (default: 25 × 4 = 100)")
    args = parser.parse_args()
    per_type = max(1, args.limit // 4) if args.limit is not None else 25

    print("=" * 70)
    print("Corrected Ensemble Entity Linking Test")
    print("Mention-only gold | Strict URI matching | Context-augmented WikiSearch")
    print("=" * 70)

    questions = load_questions(COVERAGE_PATH)
    selected = select_questions(questions, per_type=per_type)

    type_counts = {}
    for q in selected:
        qt = q.get("question_type", "?")
        type_counts[qt] = type_counts.get(qt, 0) + 1
    type_summary = ", ".join(f"{n} {t}" for t, n in sorted(type_counts.items()))

    print(f"\nSelected {len(selected)} questions: {type_summary}")
    print(f"Estimated time: ~{len(selected) * 5 // 60 + 1}-{len(selected) * 8 // 60 + 1} min\n")

    results = []
    for i, q in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] Processing...")
        try:
            r = run_question(q)
            results.append(r)
        except Exception as exc:
            print(f"  ERROR on question {q['question_id']}: {exc}")
            import traceback
            traceback.print_exc()

    # Aggregate
    agg = aggregate(results)
    t_agg = type_stats(results)

    # Print summary
    print("\n" + "=" * 70)
    print("MENTION-ONLY ENTITY RECALL (strict URI matching)")
    print("=" * 70)
    total = agg["total_mention_entities"]
    print(f"\n{'Approach':<20} {'Recall':<12} {'Found/Total':<15}")
    print("-" * 50)
    print(f"{'EntityLinker':<20} {agg['spotlight_recall']*100:.1f}%{'':<6} "
          f"{agg['spotlight_found']}/{total}")
    print(f"{'WikiSearch':<20} {agg['wikisearch_recall']*100:.1f}%{'':<6} "
          f"{agg['wikisearch_found']}/{total}")
    print(f"{'Ensemble':<20} {agg['ensemble_recall']*100:.1f}%{'':<6} "
          f"{agg['ensemble_found']}/{total}")

    print(f"\nEnsemble gain over EntityLinker: "
          f"+{(agg['ensemble_recall'] - agg['spotlight_recall'])*100:.1f}pp")
    print(f"Bridge entities excluded: {agg['total_bridge_entities']}")

    print(f"\nEnsemble helps (ENS finds what EL misses): {agg['ensemble_helps']}")
    print(f"Ensemble neutral (EL finds, ENS doesn't): {agg['ensemble_hurts']}")
    print(f"Both missed: {agg['both_missed']}")

    # By type
    print(f"\n{'Type':<20} {'EL Recall':<12} {'WS Recall':<12} {'ENS Recall':<12} {'Gain':<8}")
    print("-" * 65)
    for qt, s in t_agg.items():
        gain = (s["ensemble_recall"] - s["spotlight_recall"]) * 100
        print(f"{qt:<20} {s['spotlight_recall']*100:.1f}%{'':<6} "
              f"{s['wikisearch_recall']*100:.1f}%{'':<6} "
              f"{s['ensemble_recall']*100:.1f}%{'':<6} {gain:+.1f}pp")

    write_report(results, agg, t_agg)

    # Decision gate — based on actual ENS vs EL delta
    actual_gain = agg["ensemble_found"] - agg["spotlight_found"]
    gain_pp = (agg["ensemble_recall"] - agg["spotlight_recall"]) * 100
    if actual_gain >= 3:
        print(f"\n>>> DECISION: PROCEED — ensemble fixes {actual_gain} entities (+{gain_pp:.1f}pp)")
    else:
        print(f"\n>>> DECISION: HOLD — only {actual_gain} fixes, +{gain_pp:.1f}pp")


if __name__ == "__main__":
    main()
