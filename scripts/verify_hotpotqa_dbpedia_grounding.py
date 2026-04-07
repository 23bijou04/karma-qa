#!/usr/bin/env python3
"""HotpotQA DBpedia grounding verification — 3-tier funnel.

Measures how much of HotpotQA 1000q is grounded in DBpedia:
  Tier 1: Wikipedia title → DBpedia URI resolution (entity coverage)
  Tier 2: Entity pair connectivity via any DBpedia property (1-hop + 2-hop)
  Tier 3: Gold answer findable in DBpedia property values

Unlike 2WikiMultiHopQA (which provides [S,P,O] evidence triples), HotpotQA
only provides Wikipedia titles as supporting facts. So Tier 2 checks ANY
connection (not specific predicates), and Tier 3 uses value matching.

Usage:
    python scripts/verify_hotpotqa_dbpedia_grounding.py [--resume] [--limit N]
"""

import argparse

# ── Metrics import (for normalize_answer) ────────────────────────────────
import importlib.util
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, unquote

from SPARQLWrapper import JSON, SPARQLWrapper

BASE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("metrics", BASE / "evaluation/metrics.py")
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
normalize_answer = _metrics.normalize_answer

# ── Paths ────────────────────────────────────────────────────────────────
DATASET_PATH = BASE / "baselines/LogicRAG/dataset/hotpotqa.json"
OUTPUT_DIR = BASE / "data/cross_benchmark/hotpotqa"
OUTPUT_PATH = OUTPUT_DIR / "dbpedia_grounding_results.json"
CHECKPOINT_PATH = OUTPUT_DIR / "grounding_checkpoint.json"

# ── SPARQL setup (copied from verify_hop_coverage.py) ────────────────────
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
PREFIXES = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
"""

sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.setReturnFormat(JSON)
sparql.setTimeout(30)
sparql.addCustomHttpHeader("User-Agent", "DBpediaQA-HotpotGrounding/1.0")

_last_query_time = 0.0


def _rate_limit():
    global _last_query_time
    now = time.time()
    elapsed = now - _last_query_time
    if elapsed < 0.2:
        time.sleep(0.2 - elapsed)
    _last_query_time = time.time()


def run_sparql(query: str, retries: int = 3, timeout: int = 30) -> dict | None:
    """Execute SPARQL query with rate limiting and retries."""
    full = f"{PREFIXES}\n{query}"
    sparql.setTimeout(timeout)
    for attempt in range(retries):
        _rate_limit()
        try:
            sparql.setQuery(full)
            return sparql.query().convert()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    SPARQL error (retry in {wait}s): {e}")
                time.sleep(wait)
            else:
                return None


# ── Helpers ──────────────────────────────────────────────────────────────

def _escape(s: str) -> str:
    """Escape string for SPARQL literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def load_hotpotqa() -> list[dict]:
    """Load HotpotQA dataset and extract per-question metadata."""
    with open(DATASET_PATH) as f:
        data = json.load(f)
    return data


def extract_titles(question: dict) -> list[str]:
    """Extract unique Wikipedia titles from supporting_facts."""
    # supporting_facts = [[title, sent_idx], ...]
    seen = set()
    titles = []
    for title, _ in question["supporting_facts"]:
        if title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


# ── Tier 1: Entity Coverage ─────────────────────────────────────────────

# Cache: title → URI or None
_uri_cache: dict[str, str | None] = {}


def resolve_title_to_uri(title: str) -> str | None:
    """Resolve Wikipedia title → DBpedia URI via 2-layer fallback."""
    if title in _uri_cache:
        return _uri_cache[title]

    # Layer 1: Direct URI construction + ASK verify
    encoded = title.replace(" ", "_")
    uri = f"http://dbpedia.org/resource/{quote(encoded, safe='_,()·')}"
    result = run_sparql(f"ASK {{ <{uri}> ?p ?o }}")
    if result and result.get("boolean"):
        _uri_cache[title] = uri
        return uri

    # Layer 2: SPARQL label search (exact rdfs:label@en)
    escaped = _escape(title)
    result = run_sparql(
        f'SELECT ?s WHERE {{ ?s rdfs:label "{escaped}"@en }} LIMIT 1'
    )
    if result:
        bindings = result.get("results", {}).get("bindings", [])
        if bindings:
            found_uri = bindings[0]["s"]["value"]
            _uri_cache[title] = found_uri
            return found_uri

    _uri_cache[title] = None
    return None


def batch_resolve_titles(all_titles: list[str], checkpoint_data: dict | None = None):
    """Resolve all unique titles, with checkpoint support."""
    global _uri_cache
    if checkpoint_data and "uri_cache" in checkpoint_data:
        _uri_cache = checkpoint_data["uri_cache"]
        print(f"  Restored {len(_uri_cache)} cached URIs from checkpoint")

    remaining = [t for t in all_titles if t not in _uri_cache]
    print(f"  {len(remaining)} titles to resolve ({len(_uri_cache)} cached)")

    for i, title in enumerate(remaining, 1):
        resolve_title_to_uri(title)
        if i % 100 == 0:
            resolved = sum(1 for v in _uri_cache.values() if v is not None)
            print(f"    {i}/{len(remaining)} resolved ({resolved} found so far)")

    resolved = sum(1 for v in _uri_cache.values() if v is not None)
    print(f"  URI resolution complete: {resolved}/{len(all_titles)} titles resolved")


# ── Tier 2: Entity Pair Connectivity ─────────────────────────────────────

# Properties to exclude from connectivity checks (noise)
NOISE_FILTERS = """
  FILTER(?p != <http://dbpedia.org/ontology/wikiPageWikiLink>)
  FILTER(?p != <http://www.w3.org/2002/07/owl#sameAs>)
  FILTER(?p != <http://www.w3.org/ns/prov#wasDerivedFrom>)
  FILTER(?p != <http://xmlns.com/foaf/0.1/name>)
  FILTER(!STRSTARTS(STR(?p), "http://www.w3.org/"))
"""


def check_direct_connection(uri1: str, uri2: str) -> tuple[bool, list[str]]:
    """Check 1-hop connection between two entities. Returns (connected, properties)."""
    query = f"""
SELECT DISTINCT ?p WHERE {{
  {{ <{uri1}> ?p <{uri2}> }} UNION {{ <{uri2}> ?p <{uri1}> }}
  {NOISE_FILTERS}
}} LIMIT 10
"""
    result = run_sparql(query)
    if not result:
        return False, []
    bindings = result.get("results", {}).get("bindings", [])
    props = [b["p"]["value"] for b in bindings]
    return len(props) > 0, props


def check_2hop_connection(uri1: str, uri2: str) -> tuple[bool, str]:
    """Check 2-hop connection (both directions) with 15s timeout."""
    query = f"""
ASK {{
  {{ <{uri1}> ?p1 ?mid . ?mid ?p2 <{uri2}> }}
  UNION
  {{ <{uri2}> ?p1 ?mid . ?mid ?p2 <{uri1}> }}
  FILTER(?p1 != <http://dbpedia.org/ontology/wikiPageWikiLink>)
  FILTER(?p2 != <http://dbpedia.org/ontology/wikiPageWikiLink>)
  FILTER(!STRSTARTS(STR(?p1), "http://www.w3.org/"))
  FILTER(!STRSTARTS(STR(?p2), "http://www.w3.org/"))
}}
"""
    result = run_sparql(query, retries=1, timeout=15)
    if result is None:
        return False, "timeout"
    return result.get("boolean", False), "2hop"


# ── Tier 3: Answer Grounding ────────────────────────────────────────────

def retrieve_entity_values(uri: str) -> list[str]:
    """Fetch all dbo:/dbp: property values for an entity."""
    query = f"""
SELECT ?p ?o ?label WHERE {{
  <{uri}> ?p ?o .
  FILTER(STRSTARTS(STR(?p), "http://dbpedia.org/ontology/") ||
         STRSTARTS(STR(?p), "http://dbpedia.org/property/"))
  OPTIONAL {{ ?o rdfs:label ?label . FILTER(LANG(?label) = "en") }}
}} LIMIT 100
"""
    result = run_sparql(query)
    if not result:
        return []

    values = []
    for b in result.get("results", {}).get("bindings", []):
        if "label" in b:
            values.append(b["label"]["value"])
        else:
            val = b["o"]["value"]
            if val.startswith("http://dbpedia.org/resource/"):
                val = unquote(val.split("/resource/")[-1]).replace("_", " ")
            values.append(val)
    return values


def answer_grounded(gold_answer: str, actual_values: list[str]) -> tuple[bool, str | None]:
    """Check if gold answer matches any DBpedia value. Returns (matched, matched_value).

    Uses length-ratio guard (>=0.4) on substring containment to prevent
    short answers (e.g. "8") from matching inside long values (e.g. "658798").
    """
    norm_gold = normalize_answer(gold_answer)
    if not norm_gold:
        return False, None
    for val in actual_values:
        norm_val = normalize_answer(val)
        if not norm_val:
            continue
        # Exact match
        if norm_gold == norm_val:
            return True, val
        # Substring containment with length-ratio guard
        # Ratio = len(shorter) / len(longer) must be >= 0.4
        shorter, longer = sorted([norm_gold, norm_val], key=len)
        ratio = len(shorter) / len(longer)
        if ratio >= 0.4 and (norm_gold in norm_val or norm_val in norm_gold):
            return True, val
        # Year extraction: "1932-08-16" → "1932"
        if re.match(r"^\d{4}-\d{2}-\d{2}", val):
            year = val[:4]
            if year == gold_answer.strip():
                return True, val
    return False, None


# ── Per-question verification ────────────────────────────────────────────

def verify_question(question: dict) -> dict:
    """Run all 3 tiers for a single question."""
    qid = question["_id"]
    qtype = question["type"]
    gold = question["answer"]
    titles = extract_titles(question)

    result = {
        "question_id": qid,
        "question": question["question"],
        "type": qtype,
        "gold_answer": gold,
        "titles": titles,
        # Tier 1
        "uris": [],
        "all_resolved": False,
        # Tier 2
        "pair_connected": False,
        "connection_type": "none",
        "connecting_properties": [],
        # Tier 3
        "answer_grounded": False,
        "answer_category": "value_lookup",  # entity_answer | not_applicable | value_lookup
        "grounding_source": "none",
        "matched_value": None,
    }

    # ── Tier 1: resolve titles to URIs ──
    uris = [_uri_cache.get(t) for t in titles]
    result["uris"] = [u if u else None for u in uris]
    result["all_resolved"] = all(u is not None for u in uris)

    if not result["all_resolved"]:
        return result

    # ── Tier 2: pair connectivity (same check for bridge + comparison) ──
    uri1, uri2 = uris[0], uris[1]
    connected, props = check_direct_connection(uri1, uri2)
    if connected:
        result["pair_connected"] = True
        result["connection_type"] = "direct"
        result["connecting_properties"] = props
    else:
        hop2_connected, hop2_status = check_2hop_connection(uri1, uri2)
        if hop2_connected:
            result["pair_connected"] = True
            result["connection_type"] = "2hop"
        elif hop2_status == "timeout":
            result["connection_type"] = "timeout"
        else:
            result["connection_type"] = "none"

    # ── Tier 3: answer grounding ──
    gold_lower = gold.strip().lower()

    # Sub-category: yes/no → not applicable
    if gold_lower in ("yes", "no"):
        result["answer_category"] = "not_applicable"
        return result

    # Sub-category: answer IS a supporting_facts entity title
    for title in titles:
        if normalize_answer(gold) == normalize_answer(title):
            result["answer_category"] = "entity_answer"
            # Check if URI exists (it does, since Tier 1 passed)
            result["answer_grounded"] = True
            result["grounding_source"] = "entity_title"
            result["matched_value"] = title
            return result

    # Sub-category: value lookup on both entities
    result["answer_category"] = "value_lookup"
    for idx, uri in enumerate(uris):
        values = retrieve_entity_values(uri)
        matched, matched_val = answer_grounded(gold, values)
        if matched:
            result["answer_grounded"] = True
            result["grounding_source"] = f"entity{idx + 1}"
            result["matched_value"] = matched_val
            return result

    return result


# ── Statistics & comparison ──────────────────────────────────────────────

def print_statistics(results: list[dict]):
    """Print funnel summary, by-type breakdown, and cross-benchmark comparison."""
    total = len(results)
    print(f"\n{'=' * 72}")
    print(f"HOTPOTQA DBPEDIA GROUNDING VERIFICATION ({total} questions)")
    print(f"{'=' * 72}")

    # Tier 1
    t1 = sum(1 for r in results if r["all_resolved"])
    print("\nTier 1 — Entity Coverage:")
    print(f"  All titles resolved:  {t1}/{total} ({100 * t1 / total:.1f}%)")

    # Tier 2 (only for Tier-1-passing)
    t1_results = [r for r in results if r["all_resolved"]]
    t2 = sum(1 for r in t1_results if r["pair_connected"])
    direct = sum(1 for r in t1_results if r["connection_type"] == "direct")
    hop2 = sum(1 for r in t1_results if r["connection_type"] == "2hop")
    timeout = sum(1 for r in t1_results if r["connection_type"] == "timeout")
    none_ = sum(1 for r in t1_results if r["connection_type"] == "none")
    print(f"\nTier 2 — Pair Connectivity (of {len(t1_results)} Tier-1 passes):")
    t1_pct = f"{100 * t2 / len(t1_results):.1f}%" if t1_results else "N/A"
    print(f"  Connected (any):      {t2}/{len(t1_results)} ({t1_pct})")
    print(f"    Direct (1-hop):     {direct}")
    print(f"    Indirect (2-hop):   {hop2}")
    print(f"    Timeout (unknown):  {timeout}")
    print(f"    Not connected:      {none_}")

    # Tier 3 (all Tier-1-passing, independent of Tier 2)
    not_applicable = sum(1 for r in t1_results if r["answer_category"] == "not_applicable")
    entity_answer = sum(1 for r in t1_results if r["answer_category"] == "entity_answer"
                        and r["answer_grounded"])
    value_grounded = sum(1 for r in t1_results if r["answer_category"] == "value_lookup"
                         and r["answer_grounded"])
    total_grounded = entity_answer + value_grounded
    applicable = len(t1_results) - not_applicable
    value_applicable = applicable - entity_answer
    print(f"\nTier 3 — Answer Grounding (of {len(t1_results)} Tier-1 passes):")
    print(f"  Not applicable (yes/no): {not_applicable}")
    print(f"  Entity-as-answer:        {entity_answer} (URI existence only)")
    print(f"  Value grounded:          {value_grounded}")
    grnd_pct = f"{100 * total_grounded / applicable:.1f}%" if applicable else "N/A"
    val_pct = (f"{100 * value_grounded / value_applicable:.1f}%"
               if value_applicable else "N/A")
    print(f"  Total grounded:          {total_grounded}/{applicable} "
          f"({grnd_pct} of applicable)")
    print(f"  Value-only grounded:     {value_grounded}/{value_applicable} "
          f"({val_pct} excl. entity-as-answer)")

    # By type breakdown
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)

    print("\nBy question type:")
    for qtype in sorted(by_type.keys()):
        items = by_type[qtype]
        n = len(items)
        e = sum(1 for r in items if r["all_resolved"])
        c = sum(1 for r in items if r["pair_connected"])
        g = sum(1 for r in items if r["answer_grounded"])
        print(f"  {qtype} ({n}q): entity={100 * e / n:.1f}% "
              f"conn={100 * c / n:.1f}% grounded={100 * g / n:.1f}%")

    # Cross-benchmark comparison
    print(f"\n{'=' * 72}")
    print("CROSS-BENCHMARK COMPARISON")
    print(f"{'=' * 72}")
    print(f"{'Metric':<26} {'2WikiMultiHop':>14} {'HotpotQA':>14}")
    print(f"{'-' * 56}")
    print(f"{'Entity coverage':<26} {'93.6% (936)':>14} "
          f"{f'{100 * t1 / total:.1f}% ({t1})':>14}")
    print(f"{'Property / Connectivity':<26} {'36.8% (368)':>14} "
          f"{f'{100 * t2 / len(t1_results):.1f}% ({t2})':>14}  [*]")
    val_pct_cmp = f"{100 * value_grounded / value_applicable:.1f}%" if value_applicable else "N/A"
    print(f"{'Answer grounding [**]':<26} {'11.9% (119)':>14} "
          f"{f'{val_pct_cmp} ({value_grounded})':>14}")
    print(f"{'Hybrid Judge EM':<26} {'89.4% (894)':>14} {'86.1% (861)':>14}")
    print(f"{'-' * 56}")
    print("[*]  2Wiki checks specific predicates per hop;")
    print("     HotpotQA checks ANY connection between entity pair.")
    print("[**] Value-only grounding (excl. entity-as-answer).")


# ── Checkpoint / save ────────────────────────────────────────────────────

def save_checkpoint(results: list[dict], completed_ids: set[str]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({
            "completed_ids": list(completed_ids),
            "uri_cache": _uri_cache,
            "results": results,
        }, f)


def load_checkpoint() -> tuple[set[str], list[dict], dict]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        return set(data["completed_ids"]), data["results"], data
    return set(), [], {}


def save_results(results: list[dict]):
    """Save full results JSON."""
    total = len(results)
    t1_results = [r for r in results if r["all_resolved"]]
    not_app = sum(1 for r in t1_results if r["answer_category"] == "not_applicable")
    applicable = len(t1_results) - not_app

    summary = {
        "total_questions": total,
        "entity_coverage": {
            "count": sum(1 for r in results if r["all_resolved"]),
            "pct": round(100 * sum(1 for r in results if r["all_resolved"]) / total, 1),
        },
        "pair_connectivity": {
            "count": sum(1 for r in t1_results if r["pair_connected"]),
            "pct": round(100 * sum(1 for r in t1_results if r["pair_connected"])
                         / len(t1_results), 1) if t1_results else 0,
            "direct": sum(1 for r in t1_results if r["connection_type"] == "direct"),
            "two_hop": sum(1 for r in t1_results if r["connection_type"] == "2hop"),
            "timeout": sum(1 for r in t1_results if r["connection_type"] == "timeout"),
        },
        "answer_grounding": {
            "entity_answer": sum(1 for r in t1_results
                                 if r["answer_category"] == "entity_answer"
                                 and r["answer_grounded"]),
            "value_grounded": sum(1 for r in t1_results
                                  if r["answer_category"] == "value_lookup"
                                  and r["answer_grounded"]),
            "not_applicable": not_app,
            "total_grounded": sum(1 for r in results if r["answer_grounded"]),
            "pct_of_applicable": round(
                100 * sum(1 for r in results if r["answer_grounded"]) / applicable, 1
            ) if applicable else 0,
        },
    }

    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)

    type_summary = {}
    for qtype, items in by_type.items():
        n = len(items)
        type_summary[qtype] = {
            "total": n,
            "entity_coverage": sum(1 for r in items if r["all_resolved"]),
            "pair_connected": sum(1 for r in items if r["pair_connected"]),
            "answer_grounded": sum(1 for r in items if r["answer_grounded"]),
        }

    output = {
        "summary": summary,
        "by_type": type_summary,
        "per_question": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HotpotQA DBpedia grounding verification (3-tier funnel)"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")
    args = parser.parse_args()

    # Load dataset
    print("Loading HotpotQA dataset...")
    data = load_hotpotqa()
    print(f"  {len(data)} questions loaded")

    # Resume or fresh start
    completed_ids, results, ckpt_data = (
        load_checkpoint() if args.resume else (set(), [], {})
    )
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} questions already done")

    # Determine which questions to process
    remaining = [q for q in data if q["_id"] not in completed_ids]
    if args.limit > 0:
        remaining = remaining[: args.limit]

    # Extract unique titles only for questions we'll process
    # (full run resolves all 1988; --limit resolves only what's needed)
    target_questions = remaining if args.limit > 0 else data
    all_titles = []
    seen = set()
    for q in target_questions:
        for t in extract_titles(q):
            if t not in seen:
                seen.add(t)
                all_titles.append(t)
    print(f"  {len(all_titles)} unique Wikipedia titles to resolve")

    # Tier 1: batch resolve titles
    print("\n── Tier 1: Entity Resolution ──")
    batch_resolve_titles(all_titles, ckpt_data if args.resume else None)
    print(f"\n── Processing {len(remaining)} questions (Tiers 1+2+3) ──")

    start_time = time.time()
    for i, q in enumerate(remaining, 1):
        titles = extract_titles(q)
        uris_str = ", ".join(
            ("✓" if _uri_cache.get(t) else "✗") for t in titles
        )
        print(f"[{i}/{len(remaining)}] {q['type']}: {q['question'][:55]}... [{uris_str}]")

        result = verify_question(q)
        results.append(result)
        completed_ids.add(q["_id"])

        # Status line
        conn = result["connection_type"] if result["all_resolved"] else "n/a"
        grnd = result["answer_category"] if result["answer_grounded"] else "-"
        print(f"  → T1:{'✓' if result['all_resolved'] else '✗'} "
              f"T2:{conn} T3:{grnd}")

        # Checkpoint every 50 questions
        if i % 50 == 0:
            save_checkpoint(results, completed_ids)
            elapsed = time.time() - start_time
            rate = i / elapsed * 60
            print(f"\n  --- Checkpoint: {len(completed_ids)} done, "
                  f"{rate:.1f} q/min ---\n")

    # Save and report
    save_results(results)
    print_statistics(results)

    # Cleanup checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
