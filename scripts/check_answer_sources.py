#!/usr/bin/env python3
"""Check whether gold answers are reachable via DBpedia properties or Wikipedia text.

Two-phase analysis:
  Phase 1 (offline): Check DBpedia property coverage + benchmark context paragraphs.
    For each hop value, first checks the entity's own paragraph, then falls back to
    other paragraphs. Distinguishes own-entity vs different-entity paragraph matches.
  Phase 2 (API): For hops not in DBpedia, fetch the entity's actual Wikipedia page via
    our pipeline's retrieval method and check if the value appears there. Detects
    disambiguation pages. This tests whether our pipeline can actually reach the info.

Output: data/eval_1000q/answer_source_analysis_1000.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation_study_v3.wiki_utils import get_wikipedia_text  # noqa: E402


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles for fuzzy matching."""
    t = text.lower().strip()
    t = re.sub(r"\b(the|a|an)\b", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def value_in_text(value: str, text: str) -> bool:
    """Check if evidence value appears in text (fuzzy)."""
    if not value or not text:
        return False
    if value.lower() in text.lower():
        return True
    nv = normalize(value)
    if not nv:
        return False
    nt = normalize(text)
    return nv in nt


def _is_own_paragraph(entity: str, title: str) -> bool:
    """Check if a paragraph title belongs to this entity.

    Handles disambiguated titles like "Thomas Jefferson (film)" or
    "Henry I of France" matching entity "Thomas Jefferson" / "Henry I".
    Uses prefix matching: entity name must be a prefix of the title.
    """
    ne = normalize(entity)
    nt = normalize(title)
    return nt == ne or nt.startswith(ne + " ")


def find_value_in_context(value: str, entity: str, context_paragraphs: list) -> tuple[str | None, bool]:
    """Search benchmark context paragraphs for the value.

    First checks the entity's own paragraph (exact or disambiguated match),
    then falls back to other paragraphs.
    Returns (paragraph_title, is_own_paragraph) or (None, False).
    """
    # First pass: entity's own paragraph
    for title, sentences in context_paragraphs:
        if _is_own_paragraph(entity, title):
            text = " ".join(sentences)
            if value_in_text(value, text):
                return title, True
    # Second pass: any other paragraph
    for title, sentences in context_paragraphs:
        if not _is_own_paragraph(entity, title):
            text = " ".join(sentences)
            if value_in_text(value, text):
                return title, False
    return None, False


def load_data(eval_path: str, dev_path: str):
    """Load coverage analysis and dev_full, return matched records."""
    with open(eval_path) as f:
        coverage = json.load(f)
    with open(dev_path) as f:
        dev_full = json.load(f)

    dev_by_id = {q["_id"]: q for q in dev_full}

    matched = []
    for cq in coverage["questions"]:
        qid = cq["question_id"]
        if qid in dev_by_id:
            matched.append({"coverage": cq, "dev": dev_by_id[qid]})
    return matched


def check_hop_dbpedia(hop_entity: str, hop_property: str, coverage_entities: list) -> bool:
    """Check if a hop's property is found in DBpedia coverage data."""
    ne = normalize(hop_entity)
    np_ = normalize(hop_property)
    for ent in coverage_entities:
        if normalize(ent["name"]) == ne:
            if any(normalize(pf) == np_ for pf in ent.get("properties_found", [])):
                return True
    return False


def phase1_check(matched_data: list) -> list:
    """Phase 1: Offline check — DBpedia coverage + benchmark context (all paragraphs)."""
    results = []
    for item in matched_data:
        cq = item["coverage"]
        dq = item["dev"]
        evidences = dq.get("evidences", [])
        context = dq.get("context", [])

        hops = []
        for entity, prop, value in evidences:
            dbpedia_found = check_hop_dbpedia(entity, prop, cq["entities"])

            # Search context paragraphs: entity's own first, then others
            found_in_title, is_own = find_value_in_context(value, entity, context)
            in_context = found_in_title is not None

            hops.append({
                "entity": entity,
                "property": prop,
                "value": value,
                "dbpedia_property": dbpedia_found,
                "in_benchmark_context": in_context,
                "in_own_paragraph": is_own,
                "context_paragraph_title": found_in_title,
                "needs_wiki_check": not dbpedia_found,
            })

        results.append({
            "question_id": cq["question_id"],
            "question": cq["question"],
            "question_type": cq["question_type"],
            "gold_answer": cq["gold_answer"],
            "hops": hops,
        })
    return results


def _is_disambiguation_page(text: str) -> bool:
    """Detect if fetched Wikipedia text is a disambiguation page."""
    first_500 = text[:500].lower()
    markers = ["may refer to", "can refer to", "commonly refers to", "(disambiguation)"]
    return any(m in first_500 for m in markers)


def phase2_wiki_check(results: list, delay: float = 0.5) -> list:
    """Phase 2: For hops without DBpedia property, check actual Wikipedia pages.

    This tests whether our pipeline's Wikipedia retrieval can find the value,
    as opposed to the curated benchmark context. Loads from cache file if present,
    only fetches entities not already cached.
    """
    candidates = []
    for r in results:
        for i, hop in enumerate(r["hops"]):
            if hop["needs_wiki_check"]:
                candidates.append((r["question_id"], i, hop))

    # Deduplicate by entity name to avoid fetching the same page multiple times
    wiki_cache: dict[str, str | None] = {}
    disambiguation_set: set[str] = set()
    unique_entities = {hop["entity"] for _, _, hop in candidates}

    # Load existing cache if available
    pages_path = Path(__file__).resolve().parent.parent / "data/eval_1000q/wiki_pages_phase2.json"
    if pages_path.exists():
        with open(pages_path) as f:
            cached = json.load(f)
        for entity in unique_entities:
            if entity in cached and cached[entity]["text"]:
                wiki_cache[entity] = cached[entity]["text"]
            elif entity in cached and cached[entity]["disambiguation"]:
                disambiguation_set.add(entity)
                wiki_cache[entity] = None
        cached_count = sum(1 for e in unique_entities if e in cached and
                           (cached[e]["text"] or cached[e]["disambiguation"]))
        print(f"Phase 2: loaded {cached_count} entities from cache")

    to_fetch = sorted(unique_entities - set(wiki_cache.keys()))
    print(f"Phase 2: {len(candidates)} hops need Wikipedia check "
          f"({len(unique_entities)} unique entities, {len(to_fetch)} to fetch)...")

    fetched = 0
    for entity in to_fetch:
        wiki_text = get_wikipedia_text(entity, max_chars=8000)
        if wiki_text and _is_disambiguation_page(wiki_text):
            disambiguation_set.add(entity)
            wiki_cache[entity] = None  # treat as not found
        else:
            wiki_cache[entity] = wiki_text
        fetched += 1
        if fetched % 50 == 0:
            print(f"  ... fetched {fetched}/{len(to_fetch)} Wikipedia pages")
        time.sleep(delay)

    # Persist fetched pages for further analysis
    pages_out = {}
    for entity in sorted(wiki_cache):
        pages_out[entity] = {
            "text": wiki_cache[entity],
            "disambiguation": entity in disambiguation_set,
        }
    with open(pages_path, "w") as f:
        json.dump(pages_out, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(pages_out)} Wikipedia pages to {pages_path}")

    for _, _, hop in candidates:
        wiki_text = wiki_cache.get(hop["entity"])
        is_disambig = hop["entity"] in disambiguation_set
        hop["wiki_fetched"] = wiki_text is not None
        hop["wiki_disambiguation"] = is_disambig
        hop["in_wikipedia"] = value_in_text(hop["value"], wiki_text) if wiki_text else False

    # Fill in hops that didn't need checking
    for r in results:
        for hop in r["hops"]:
            if "in_wikipedia" not in hop:
                hop["in_wikipedia"] = None
                hop["wiki_fetched"] = None
                hop["wiki_disambiguation"] = None
    return results


def classify_results(results: list) -> list:
    """Classify each hop and question's reachability.

    A hop is reachable if: DBpedia has the property, OR the value is found in the
    entity's own Wikipedia page (via API or benchmark's own-entity paragraph).
    Values only found in a *different* entity's paragraph are NOT reachable by our
    pipeline (which retrieves text for the subject entity).
    """
    for r in results:
        unreachable_hops = []
        for hop in r["hops"]:
            if hop["dbpedia_property"]:
                reachable = True
            elif hop.get("in_wikipedia"):
                reachable = True  # Phase 2 confirmed via live API
            elif hop.get("in_wikipedia") is None and not hop["dbpedia_property"]:
                # Phase 2 not run — use entity-scoped paragraph as proxy
                reachable = hop.get("in_own_paragraph", False)
            else:
                reachable = False
            hop["reachable"] = reachable
            if not reachable:
                unreachable_hops.append(hop)

        r["all_hops_reachable"] = len(unreachable_hops) == 0
        r["unreachable_hops"] = len(unreachable_hops)
        r["unreachable_details"] = [
            f"{h['entity']}.{h['property']}={h['value']}" for h in unreachable_hops
        ]
    return results


def compute_statistics(results: list) -> dict:
    """Compute summary statistics."""
    total = len(results)
    reachable = sum(1 for r in results if r["all_hops_reachable"])
    unreachable = total - reachable

    by_type = {}
    for r in results:
        qt = r["question_type"]
        if qt not in by_type:
            by_type[qt] = {"total": 0, "reachable": 0, "unreachable": 0}
        by_type[qt]["total"] += 1
        if r["all_hops_reachable"]:
            by_type[qt]["reachable"] += 1
        else:
            by_type[qt]["unreachable"] += 1

    # Where each hop's value was found (5 categories)
    hop_value_location = {
        "dbpedia_property": 0,
        "wikipedia_api": 0,
        "own_paragraph": 0,
        "other_paragraph": 0,
        "not_found": 0,
    }
    total_hops = 0
    for r in results:
        for h in r["hops"]:
            total_hops += 1
            if h["dbpedia_property"]:
                hop_value_location["dbpedia_property"] += 1
            elif h.get("in_wikipedia"):
                hop_value_location["wikipedia_api"] += 1
            elif h["in_benchmark_context"] and h.get("in_own_paragraph"):
                hop_value_location["own_paragraph"] += 1
            elif h["in_benchmark_context"] and not h.get("in_own_paragraph"):
                hop_value_location["other_paragraph"] += 1
            else:
                hop_value_location["not_found"] += 1

    # Non-DBpedia hop counts (for clarity)
    non_dbpedia_hops = total_hops - hop_value_location["dbpedia_property"]
    unique_entities_needing_check = len({
        h["entity"] for r in results for h in r["hops"] if h["needs_wiki_check"]
    })

    # Unreachable properties breakdown
    unreachable_props = {}
    for r in results:
        for h in r["hops"]:
            if not h["reachable"]:
                prop = h["property"]
                unreachable_props[prop] = unreachable_props.get(prop, 0) + 1
    unreachable_props = dict(sorted(unreachable_props.items(), key=lambda x: -x[1]))

    # Wikipedia fetch stats (per unique entity that needed checking)
    wiki_fetch_stats = {"fetched": 0, "not_found": 0, "disambiguation": 0, "not_checked": 0}
    entity_wiki_status: dict[str, dict] = {}
    for r in results:
        for h in r["hops"]:
            entity = h["entity"]
            if not h["needs_wiki_check"] or entity in entity_wiki_status:
                continue
            entity_wiki_status[entity] = h
    for h in entity_wiki_status.values():
        if h.get("wiki_fetched") is True:
            wiki_fetch_stats["fetched"] += 1
        elif h.get("wiki_disambiguation"):
            wiki_fetch_stats["disambiguation"] += 1
        elif h.get("wiki_fetched") is False:
            wiki_fetch_stats["not_found"] += 1
        else:
            wiki_fetch_stats["not_checked"] += 1

    return {
        "total_questions": total,
        "reachable": reachable,
        "unreachable": unreachable,
        "reachable_pct": round(100 * reachable / total, 1),
        "by_type": by_type,
        "total_hops": total_hops,
        "non_dbpedia_hops": non_dbpedia_hops,
        "unique_entities_needing_check": unique_entities_needing_check,
        "hop_value_location": hop_value_location,
        "wiki_fetch_stats": wiki_fetch_stats,
        "top_unreachable_properties": dict(list(unreachable_props.items())[:20]),
    }


def main():
    parser = argparse.ArgumentParser(description="Check gold answer reachability")
    parser.add_argument("--skip-wiki", action="store_true",
                        help="Skip Phase 2 Wikipedia API checks (use benchmark context as proxy)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between Wikipedia API calls (seconds)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    eval_path = base / "data/eval_1000q/coverage_analysis_1000.json"
    dev_path = base / "data/dev_full.json"
    out_path = base / "data/eval_1000q/answer_source_analysis_1000.json"

    print("Loading data...")
    matched = load_data(str(eval_path), str(dev_path))
    print(f"Matched {len(matched)} questions between coverage analysis and dev_full")

    print("Phase 1: Offline check (DBpedia coverage + benchmark context)...")
    results = phase1_check(matched)

    needs_wiki = sum(1 for r in results for h in r["hops"] if h["needs_wiki_check"])
    in_ctx = sum(1 for r in results for h in r["hops"]
                 if h["needs_wiki_check"] and h["in_benchmark_context"])
    print(f"Phase 1 complete: {needs_wiki} hops without DBpedia property")
    print(f"  Of these, {in_ctx} have value in benchmark context (Wikipedia text exists)")
    print(f"  {needs_wiki - in_ctx} have value NOT in any benchmark context paragraph")

    if not args.skip_wiki and needs_wiki > 0:
        results = phase2_wiki_check(results, delay=args.delay)
    elif args.skip_wiki:
        print("Skipping Phase 2 (--skip-wiki), using benchmark context as proxy")

    results = classify_results(results)
    stats = compute_statistics(results)

    output = {"statistics": stats, "questions": results}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"ANSWER SOURCE ANALYSIS — {stats['total_questions']} questions")
    print(f"{'='*60}")
    print(f"Reachable (all hops):   {stats['reachable']} ({stats['reachable_pct']}%)")
    print(f"Unreachable (≥1 hop):   {stats['unreachable']}")
    print(f"\nHops: {stats['total_hops']} total, "
          f"{stats['non_dbpedia_hops']} non-DBpedia "
          f"({stats['unique_entities_needing_check']} unique entities)")
    print("\nBy question type:")
    for qt, v in stats["by_type"].items():
        pct = round(100 * v["reachable"] / v["total"], 1)
        print(f"  {qt:25s}: {v['reachable']}/{v['total']} reachable ({pct}%)")
    print(f"\nHop value location ({stats['total_hops']} total hops):")
    for src, cnt in stats["hop_value_location"].items():
        pct = round(100 * cnt / stats["total_hops"], 1)
        print(f"  {src:25s}: {cnt} ({pct}%)")
    if stats.get("wiki_fetch_stats"):
        ws = stats["wiki_fetch_stats"]
        parts = [f"{ws['fetched']} fetched"]
        if ws["disambiguation"]:
            parts.append(f"{ws['disambiguation']} disambiguation")
        parts.append(f"{ws['not_found']} not found")
        parts.append(f"{ws['not_checked']} skipped")
        print(f"\nWikipedia API fetch (unique entities): {', '.join(parts)}")
    if stats["top_unreachable_properties"]:
        print("\nTop unreachable properties:")
        for prop, cnt in list(stats["top_unreachable_properties"].items())[:10]:
            print(f"  {prop:30s}: {cnt}")


if __name__ == "__main__":
    main()
