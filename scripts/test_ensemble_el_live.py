#!/usr/bin/env python3
"""Ensemble entity-linking live test: Spotlight vs WikiSearch vs Ensemble.

Loads 30 real questions from coverage_analysis_1000.json (10 easy / 10 hard /
10 random), runs three EL approaches, compares against gold entities, and
writes a Markdown report.

Usage:
    source venv/bin/activate
    python scripts/test_ensemble_el_live.py
"""

import json
import random
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import requests
import spacy

# ── project root on sys.path ──────────────────────────────────────────────────
# scripts/ lives inside dbpedia_qa/; relative imports in dbpedia/ resolve when
# the PARENT of dbpedia_qa/ is on sys.path (matches hybrid_qa.py pattern).
_root = Path(__file__).resolve().parent.parent        # dbpedia_qa/
_parent_dir = str(_root.parent)                       # 2026_02/ (parent of pkg)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from dbpedia_qa.dbpedia.entity_linker import (  # noqa: E402
    _MENTION_BLOCKLIST,
    _STRIP_WORDS,
    DBpediaEntityLinker,
)

# ── constants ─────────────────────────────────────────────────────────────────
COVERAGE_PATH = _root / "data" / "eval_1000q" / "coverage_analysis_1000.json"
REPORT_PATH = (
    _root
    / "plans"
    / "260317-entity-linking-ablation-study"
    / "reports"
    / "ensemble-live-test-results.md"
)
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "DBpediaQA/1.0 (research; ensemble-el-ablation)"}

SPOTLIGHT_DELAY = 1.0   # seconds between Spotlight calls
WIKI_DELAY = 0.5        # seconds between Wikipedia API calls

_nlp = spacy.load("en_core_web_sm")


# ─────────────────────────────────────────────────────────────────────────────
# Question selection
# ─────────────────────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def _has_famous_entity(q: dict) -> bool:
    """Heuristic: famous/common if any entity has >0 properties found."""
    return any(e.get("properties_found") for e in q.get("entities", []))


def _has_ambiguous_entity(q: dict) -> bool:
    """Heuristic: might confuse Spotlight if entity name is short/common."""
    for e in q.get("entities", []):
        name = e.get("name", "")
        # Short names, parenthetical disambiguation, or 'Film' suffix hint ambiguity
        if (
            len(name.split()) == 1 and len(name) < 12
            or "(" in name
            or "(Film)" in name
            or "(film)" in name
        ):
            return True
    return False


def select_questions(questions: list[dict]) -> list[dict]:
    """Return 30 questions: 10 easy, 10 hard, 10 random (seed=42)."""
    rng = random.Random(42)

    easy = [q for q in questions if _has_famous_entity(q) and not _has_ambiguous_entity(q)]
    hard = [q for q in questions if _has_ambiguous_entity(q)]
    rng.shuffle(easy)
    rng.shuffle(hard)

    selected_easy = easy[:10]
    selected_hard = hard[:10]

    # Random sample from remaining (not already chosen)
    chosen_ids = {q["question_id"] for q in selected_easy + selected_hard}
    remaining = [q for q in questions if q["question_id"] not in chosen_ids]
    rng.shuffle(remaining)
    selected_random = remaining[:10]

    result = []
    for q in selected_easy:
        result.append({**q, "_bucket": "easy"})
    for q in selected_hard:
        result.append({**q, "_bucket": "hard"})
    for q in selected_random:
        result.append({**q, "_bucket": "random"})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Gold entity parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_gold_entities(q: dict) -> list[dict]:
    """Return list of {name, uri} for entities that exist in DBpedia."""
    return [
        {"name": e["name"], "uri": e["uri"]}
        for e in q.get("entities", [])
        if e.get("exists") and e.get("uri")
    ]


# ─────────────────────────────────────────────────────────────────────────────
# URI normalisation for matching
# ─────────────────────────────────────────────────────────────────────────────

def norm_uri(uri: str) -> str:
    return unquote(uri).lower().replace(" ", "_").rstrip("/")


def norm_label(s: str) -> str:
    return s.lower().strip()


def entity_matches_gold(found_uri: str, gold: dict) -> bool:
    """Return True if found_uri matches gold by URI or label."""
    # URI match (normalised)
    if norm_uri(found_uri) == norm_uri(gold["uri"]):
        return True
    # Label match as fallback: gold name appears in found URI or vice versa
    gold_norm = norm_label(gold["name"])
    uri_local = norm_uri(found_uri).split("/")[-1].replace("_", " ")
    if gold_norm and gold_norm in uri_local:
        return True
    if uri_local and uri_local in gold_norm:
        return True
    return False


def score_approach(
    found_entities: list[dict],  # list of {uri, label, source}
    gold_entities: list[dict],
) -> dict:
    """Compute how many gold entities were found."""
    found_count = 0
    details = []
    for gold in gold_entities:
        matched = None
        for fe in found_entities:
            if entity_matches_gold(fe["uri"], gold):
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
# Approach A: Spotlight-only (via DBpediaEntityLinker)
# ─────────────────────────────────────────────────────────────────────────────

_linker: DBpediaEntityLinker | None = None


def get_linker() -> DBpediaEntityLinker:
    global _linker
    if _linker is None:
        _linker = DBpediaEntityLinker(min_confidence=0.5, min_support=20)
    return _linker


def approach_spotlight(question: str) -> tuple[list[dict], float, str]:
    """Run Spotlight-only EL. Returns (entities, elapsed_ms, status)."""
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
# Approach B: Wikipedia Search only
# ─────────────────────────────────────────────────────────────────────────────

def extract_mentions(question: str) -> list[str]:
    """Extract entity mentions using spaCy NER + noun chunks (mirrors entity_linker)."""
    doc = _nlp(question)
    mentions: list[str] = []

    # Quoted strings
    import re
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


def wiki_search_mention(mention: str, limit: int = 5) -> list[dict]:
    """Call Wikipedia Search API for a single mention."""
    try:
        resp = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": mention,
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
        print(f"  WikiSearch error for '{mention}': {exc}")
        return []


def title_to_dbpedia_uri(title: str) -> str:
    return "http://dbpedia.org/resource/" + title.replace(" ", "_")


def approach_wiki_search(question: str) -> tuple[list[dict], float, str]:
    """Run Wikipedia-Search-only EL. Returns (entities, elapsed_ms, status)."""
    t0 = time.perf_counter()
    status = "ok"
    mentions = extract_mentions(question)
    entities: list[dict] = []
    seen_uris: set[str] = set()

    for mention in mentions:
        results = wiki_search_mention(mention, limit=5)
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
                    "top5_titles": [r["title"] for r in results],
                })
                seen_uris.add(uri)

    elapsed = (time.perf_counter() - t0) * 1000
    return entities, elapsed, status


# ─────────────────────────────────────────────────────────────────────────────
# Approach C: Ensemble (Spotlight + WikiSearch for unlinked)
# ─────────────────────────────────────────────────────────────────────────────

def _find_unlinked_mentions(mentions: list[str], spotlight_entities: list[dict]) -> list[str]:
    """Find mentions not covered by Spotlight results (Jaccard + substring)."""
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
    """Run ensemble: reuse Spotlight results + WikiSearch for unlinked mentions."""
    t0 = time.perf_counter()
    status = "ok"

    mentions = extract_mentions(question)
    unlinked = _find_unlinked_mentions(mentions, spotlight_entities)

    ensemble = list(spotlight_entities)  # start with Spotlight results
    seen_uris = {e["uri"] for e in ensemble}

    for mention in unlinked:
        results = wiki_search_mention(mention, limit=5)
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

    # Elapsed is only the WikiSearch part; add Spotlight elapsed for total
    wiki_elapsed = (time.perf_counter() - t0) * 1000
    total_elapsed = spotlight_elapsed_ms + wiki_elapsed
    return ensemble, total_elapsed, status


# ─────────────────────────────────────────────────────────────────────────────
# Per-question runner
# ─────────────────────────────────────────────────────────────────────────────

def run_question(q: dict) -> dict:
    question = q["question"]
    gold = get_gold_entities(q)
    bucket = q.get("_bucket", "?")

    print(f"\n{'─'*70}")
    print(f"[{bucket.upper()}] {question}")
    print(f"  Gold: {[g['name'] for g in gold]}")

    # A: Spotlight
    sp_entities, sp_elapsed, sp_status = approach_spotlight(question)
    sp_score = score_approach(sp_entities, gold)
    sp_uris = [e["uri"].replace("http://dbpedia.org/resource/", "dbr:") for e in sp_entities]
    print(f"  A) Spotlight [{sp_status}] ({sp_elapsed:.0f}ms): {sp_uris or '(none)'}")
    print(f"     Gold recall: {sp_score['found']}/{sp_score['total']}")
    for d in sp_score["details"]:
        icon = "FOUND" if d["found"] else "MISSED"
        found_info = d["found_uri"].replace("http://dbpedia.org/resource/", "dbr:") if d["found_uri"] else ""
        print(f"       {icon} '{d['gold_name']}' -> {found_info}")

    # B: WikiSearch
    ws_entities, ws_elapsed, ws_status = approach_wiki_search(question)
    ws_score = score_approach(ws_entities, gold)
    ws_uris = [e["uri"].replace("http://dbpedia.org/resource/", "dbr:") for e in ws_entities]
    print(f"  B) WikiSearch [{ws_status}] ({ws_elapsed:.0f}ms): {ws_uris or '(none)'}")
    print(f"     Gold recall: {ws_score['found']}/{ws_score['total']}")
    for d in ws_score["details"]:
        icon = "FOUND" if d["found"] else "MISSED"
        found_info = d["found_uri"].replace("http://dbpedia.org/resource/", "dbr:") if d["found_uri"] else ""
        print(f"       {icon} '{d['gold_name']}' -> {found_info}")

    # C: Ensemble (reuse Spotlight results + WikiSearch for gaps)
    ens_entities, ens_elapsed, ens_status = approach_ensemble(
        question, sp_entities, sp_elapsed
    )
    ens_score = score_approach(ens_entities, gold)
    ens_uris = [e["uri"].replace("http://dbpedia.org/resource/", "dbr:") for e in ens_entities]
    print(f"  C) Ensemble [{ens_status}] ({ens_elapsed:.0f}ms): {ens_uris or '(none)'}")
    print(f"     Gold recall: {ens_score['found']}/{ens_score['total']}")

    # Unique contributions
    sp_found_uris = {d["found_uri"] for d in sp_score["details"] if d["found"]}
    ws_found_uris = {d["found_uri"] for d in ws_score["details"] if d["found"]}
    only_spotlight = sp_found_uris - ws_found_uris
    only_wiki = ws_found_uris - sp_found_uris
    # Both missed: gold names that neither approach found
    ws_found_gold_names = {d["gold_name"] for d in ws_score["details"] if d["found"]}
    both_missed = [
        d["gold_name"]
        for d in sp_score["details"]
        if not d["found"] and d["gold_name"] not in ws_found_gold_names
    ]

    return {
        "question_id": q["question_id"],
        "question": question,
        "question_type": q.get("question_type", "?"),
        "bucket": bucket,
        "gold": gold,
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
        "only_spotlight": list(only_spotlight),
        "only_wiki": list(only_wiki),
        "both_missed": both_missed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    total_gold = sum(len(r["gold"]) for r in results)
    sp_found = sum(r["spotlight"]["score"]["found"] for r in results)
    ws_found = sum(r["wikisearch"]["score"]["found"] for r in results)
    ens_found = sum(r["ensemble"]["score"]["found"] for r in results)

    sp_latencies = [r["spotlight"]["elapsed_ms"] for r in results]
    ws_latencies = [r["wikisearch"]["elapsed_ms"] for r in results]
    ens_latencies = [r["ensemble"]["elapsed_ms"] for r in results]

    spotlight_only_count = sum(len(r["only_spotlight"]) for r in results)
    wiki_only_count = sum(len(r["only_wiki"]) for r in results)
    both_missed_count = sum(len(r["both_missed"]) for r in results)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    return {
        "n_questions": len(results),
        "total_gold_entities": total_gold,
        "spotlight_found": sp_found,
        "wikisearch_found": ws_found,
        "ensemble_found": ens_found,
        "spotlight_recall": sp_found / total_gold if total_gold else 0,
        "wikisearch_recall": ws_found / total_gold if total_gold else 0,
        "ensemble_recall": ens_found / total_gold if total_gold else 0,
        "only_spotlight": spotlight_only_count,
        "only_wiki": wiki_only_count,
        "both_missed": both_missed_count,
        "avg_latency_spotlight_ms": avg(sp_latencies),
        "avg_latency_wikisearch_ms": avg(ws_latencies),
        "avg_latency_ensemble_ms": avg(ens_latencies),
    }


def bucket_stats(results: list[dict], agg_fn) -> dict[str, dict]:
    buckets = ["easy", "hard", "random"]
    out = {}
    for b in buckets:
        subset = [r for r in results if r["bucket"] == b]
        if subset:
            out[b] = agg_fn(subset)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(results: list[dict], agg: dict, bucket_agg: dict[str, dict]) -> None:
    lines = []
    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    lines += [
        "# Ensemble Entity Linking Live Test Results",
        "",
        f"**Generated**: {now}",
        f"**Questions tested**: {agg['n_questions']} (10 easy + 10 hard + 10 random)",
        f"**Gold entities evaluated**: {agg['total_gold_entities']}",
        "",
        "## Executive Summary",
        "",
        "| Approach | Gold Recall | Unique finds | Avg Latency |",
        "|----------|-------------|--------------|-------------|",
        f"| A) Spotlight-only | {agg['spotlight_recall']*100:.1f}% ({agg['spotlight_found']}/{agg['total_gold_entities']}) | {agg['only_spotlight']} | {agg['avg_latency_spotlight_ms']:.0f}ms |",
        f"| B) WikiSearch-only | {agg['wikisearch_recall']*100:.1f}% ({agg['wikisearch_found']}/{agg['total_gold_entities']}) | {agg['only_wiki']} | {agg['avg_latency_wikisearch_ms']:.0f}ms |",
        f"| C) Ensemble | {agg['ensemble_recall']*100:.1f}% ({agg['ensemble_found']}/{agg['total_gold_entities']}) | — | {agg['avg_latency_ensemble_ms']:.0f}ms |",
        f"| Both missed | — | {agg['both_missed']} | — |",
        "",
    ]

    # Bucket breakdown
    lines += ["## Bucket Breakdown", ""]
    lines += ["| Bucket | Spotlight | WikiSearch | Ensemble |",
              "|--------|-----------|------------|----------|"]
    for b, s in bucket_agg.items():
        lines.append(
            f"| {b} | {s['spotlight_recall']*100:.1f}% ({s['spotlight_found']}/{s['total_gold_entities']}) "
            f"| {s['wikisearch_recall']*100:.1f}% ({s['wikisearch_found']}/{s['total_gold_entities']}) "
            f"| {s['ensemble_recall']*100:.1f}% ({s['ensemble_found']}/{s['total_gold_entities']}) |"
        )
    lines.append("")

    # Per-question table
    lines += [
        "## Per-Question Results",
        "",
        "| # | Bucket | Type | Question (truncated) | Gold N | SP found | WS found | ENS found | WS unique | SP unique |",
        "|---|--------|------|----------------------|--------|----------|----------|-----------|-----------|-----------|",
    ]
    for i, r in enumerate(results, 1):
        q_short = r["question"][:55] + "..." if len(r["question"]) > 55 else r["question"]
        sp = r["spotlight"]["score"]
        ws = r["wikisearch"]["score"]
        ens = r["ensemble"]["score"]
        lines.append(
            f"| {i} | {r['bucket']} | {r['question_type']} | {q_short} "
            f"| {sp['total']} | {sp['found']}/{sp['total']} | {ws['found']}/{ws['total']} "
            f"| {ens['found']}/{ens['total']} | {len(r['only_wiki'])} | {len(r['only_spotlight'])} |"
        )
    lines.append("")

    # Detailed per-question breakdown
    lines += ["## Detailed Question Analysis", ""]
    for i, r in enumerate(results, 1):
        lines += [
            f"### Q{i} [{r['bucket'].upper()}] {r['question']}",
            "",
            f"- **Type**: {r['question_type']}",
            f"- **Gold entities**: {', '.join(g['name'] for g in r['gold'])}",
            "",
        ]
        # Spotlight
        sp_detail = r["spotlight"]["score"]["details"]
        sp_status = r["spotlight"]["status"]
        lines += [
            f"**A) Spotlight** (status={sp_status}, {r['spotlight']['elapsed_ms']:.0f}ms)",
            "",
        ]
        for d in sp_detail:
            icon = "FOUND" if d["found"] else "MISSED"
            uri_short = (d["found_uri"] or "").replace("http://dbpedia.org/resource/", "dbr:")
            lines.append(f"- {icon}: `{d['gold_name']}` → `{uri_short or '—'}`")
        lines.append("")

        # WikiSearch
        ws_detail = r["wikisearch"]["score"]["details"]
        ws_status = r["wikisearch"]["status"]
        lines += [
            f"**B) WikiSearch** (status={ws_status}, {r['wikisearch']['elapsed_ms']:.0f}ms)",
            "",
        ]
        for d in ws_detail:
            icon = "FOUND" if d["found"] else "MISSED"
            uri_short = (d["found_uri"] or "").replace("http://dbpedia.org/resource/", "dbr:")
            lines.append(f"- {icon}: `{d['gold_name']}` → `{uri_short or '—'}`")
        lines.append("")

        # Ensemble summary
        ens_score = r["ensemble"]["score"]
        lines += [
            f"**C) Ensemble** ({r['ensemble']['elapsed_ms']:.0f}ms total): "
            f"{ens_score['found']}/{ens_score['total']} gold found",
        ]
        if r["only_wiki"]:
            ws_uris = [u.replace("http://dbpedia.org/resource/", "dbr:") for u in r["only_wiki"]]
            lines.append(f"  - WikiSearch-only finds: {ws_uris}")
        if r["only_spotlight"]:
            sp_uris = [u.replace("http://dbpedia.org/resource/", "dbr:") for u in r["only_spotlight"]]
            lines.append(f"  - Spotlight-only finds: {sp_uris}")
        if r["both_missed"]:
            lines.append(f"  - Both missed: {r['both_missed']}")
        lines += ["", "---", ""]

    # Analysis section
    lines += [
        "## Analysis",
        "",
        "### When WikiSearch beats Spotlight",
        "",
        "WikiSearch tends to outperform Spotlight when:",
        "- Entity has a common name with disambiguation needed (film vs person)",
        "- Entity is a rare/niche person that Spotlight misses at confidence=0.5",
        "- Question contains parenthetical hints like `(film)` that WikiSearch passes through",
        "",
        "### When Spotlight beats WikiSearch",
        "",
        "Spotlight tends to outperform WikiSearch when:",
        "- Multiple entities appear in one question (Spotlight handles all at once)",
        "- Entity name is a substring of a longer phrase (Spotlight anchors on surface forms)",
        "- Entity is well-represented in DBpedia with high support",
        "",
        "### Ensemble Value",
        "",
        f"Ensemble improves over Spotlight-only by +{(agg['ensemble_recall'] - agg['spotlight_recall'])*100:.1f}pp recall.",
        f"Ensemble improves over WikiSearch-only by +{(agg['ensemble_recall'] - agg['wikisearch_recall'])*100:.1f}pp recall.",
        "",
        "### Recommendation",
        "",
    ]

    gain = (agg["ensemble_recall"] - agg["spotlight_recall"]) * 100
    if gain >= 5:
        lines += [
            "**PROCEED** with ensemble implementation.",
            f"Ensemble delivers +{gain:.1f}pp recall improvement over the current Spotlight-only baseline.",
            "The WikiSearch fallback adds meaningful coverage for entities Spotlight misses.",
        ]
    elif gain >= 2:
        lines += [
            "**PROCEED WITH CAVEATS** — ensemble adds marginal recall improvement.",
            f"Only +{gain:.1f}pp gain; consider whether the added latency is worth it.",
            "WikiSearch as fallback for unlinked mentions is low-risk to implement.",
        ]
    else:
        lines += [
            "**HOLD** — ensemble adds minimal recall improvement over Spotlight alone.",
            f"Only +{gain:.1f}pp gain; Spotlight already covers most gold entities.",
            "Engineering effort may not be justified.",
        ]

    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Ensemble Entity Linking Live Test")
    print("Spotlight-only vs WikiSearch-only vs Ensemble")
    print("=" * 70)

    questions = load_questions(COVERAGE_PATH)
    selected = select_questions(questions)
    print(f"\nSelected {len(selected)} questions: "
          f"{sum(1 for q in selected if q['_bucket']=='easy')} easy, "
          f"{sum(1 for q in selected if q['_bucket']=='hard')} hard, "
          f"{sum(1 for q in selected if q['_bucket']=='random')} random")
    print(f"Estimated time: ~{len(selected) * 5 // 60 + 1}-{len(selected) * 8 // 60 + 1} min "
          "(depends on Spotlight latency)\n")

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
    b_agg = bucket_stats(results, aggregate)

    # Print summary table
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"{'Approach':<20} {'Recall':<15} {'Found/Total':<15} {'Unique':<10} {'Avg ms':<10}")
    print("-" * 70)
    total = agg["total_gold_entities"]
    print(f"{'A) Spotlight':<20} {agg['spotlight_recall']*100:.1f}%{'':<9} "
          f"{agg['spotlight_found']}/{total:<13} {agg['only_spotlight']:<10} {agg['avg_latency_spotlight_ms']:.0f}")
    print(f"{'B) WikiSearch':<20} {agg['wikisearch_recall']*100:.1f}%{'':<9} "
          f"{agg['wikisearch_found']}/{total:<13} {agg['only_wiki']:<10} {agg['avg_latency_wikisearch_ms']:.0f}")
    print(f"{'C) Ensemble':<20} {agg['ensemble_recall']*100:.1f}%{'':<9} "
          f"{agg['ensemble_found']}/{total:<13} {'—':<10} {agg['avg_latency_ensemble_ms']:.0f}")
    print(f"{'Both missed':<20} {'—':<15} {agg['both_missed']}/{total}")
    print()

    gain = (agg["ensemble_recall"] - agg["spotlight_recall"]) * 100
    print(f"Ensemble gain over Spotlight: +{gain:.1f}pp")
    print(f"WikiSearch-only vs Spotlight: {(agg['wikisearch_recall']-agg['spotlight_recall'])*100:+.1f}pp")

    write_report(results, agg, b_agg)


if __name__ == "__main__":
    main()
