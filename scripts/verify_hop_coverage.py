#!/usr/bin/env python3
"""Hop-by-hop KG coverage verification for 1000q eval subset.

Verifies each evidence triple [S, P, O] from the benchmark against live DBpedia:
  1. Entity resolution: S → DBpedia URI
  2. Property existence: does URI have a predicate for P?
  3. Value match: does the predicate value match/contain O?

Outputs detailed per-question, per-hop results + summary statistics.
Checkpoints every 50 questions for resume support.

Usage:
    python scripts/verify_hop_coverage.py [--resume] [--limit N]
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote, unquote

from SPARQLWrapper import JSON, SPARQLWrapper

BASE = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE / "data/eval_1000q/hop_verification_checkpoint.json"
OUTPUT_PATH = BASE / "data/eval_1000q/ground-truth/hop_verification_results.json"

# ── SPARQL setup ──────────────────────────────────────────────────────────
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
sparql.addCustomHttpHeader("User-Agent", "DBpediaQA-HopVerify/1.0")

# Rate limiting — be nice to DBpedia (5 req/sec)
_last_query_time = 0.0


def _rate_limit():
    global _last_query_time
    now = time.time()
    elapsed = now - _last_query_time
    if elapsed < 0.2:
        time.sleep(0.2 - elapsed)
    _last_query_time = time.time()


def run_sparql(query: str, retries: int = 3) -> dict | None:
    """Execute SPARQL query with rate limiting and retries."""
    full = f"{PREFIXES}\n{query}"
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


# ── Property mapping: benchmark names → DBpedia predicates ────────────────
# Merged from verify_entity_property_coverage.py + coverage_analysis.py
PROPERTY_MAP = {
    "director": ["dbo:director", "dbp:director"],
    "date of birth": ["dbo:birthDate", "dbp:birthDate"],
    "date of death": ["dbo:deathDate", "dbp:deathDate"],
    "place of birth": ["dbo:birthPlace", "dbp:birthPlace"],
    "place of death": ["dbo:deathPlace", "dbp:deathPlace"],
    "father": ["dbp:father", "dbo:parent"],
    "mother": ["dbp:mother", "dbo:parent"],
    "spouse": ["dbo:spouse", "dbp:spouse"],
    "child": ["dbo:child", "dbp:children"],
    "sibling": ["dbp:sibling", "dbp:siblings", "dbo:relative"],
    "country of citizenship": [
        "dbo:citizenship", "dbp:citizenship",
        "dbo:nationality", "dbp:nationality",
    ],
    "country of origin": ["dbo:country", "dbp:country"],
    "country": ["dbo:country", "dbp:country"],
    "publication date": [
        "dbo:releaseDate", "dbp:releaseDate",
        "dbo:publicationDate", "dbp:published",
    ],
    "inception": ["dbo:foundingDate", "dbp:founded", "dbo:formationDate"],
    "composer": ["dbo:musicComposer", "dbp:composer", "dbp:music"],
    "performer": ["dbo:artist", "dbp:artist", "dbo:musicalArtist"],
    "educated at": ["dbo:almaMater", "dbp:almaMater", "dbp:education"],
    "place of burial": ["dbp:placeOfBurial", "dbp:burialPlace", "dbo:restingPlace"],
    "employer": ["dbo:employer", "dbp:employer"],
    "award received": ["dbo:award", "dbp:awards"],
    "cause of death": ["dbo:deathCause", "dbp:deathCause"],
    "founded by": ["dbo:founder", "dbp:founder"],
    "producer": ["dbo:producer", "dbp:producer"],
    "publisher": ["dbo:publisher", "dbp:publisher"],
    "editor": ["dbp:editor", "dbo:editor"],
    "place of detention": ["dbp:placeOfDetention", "dbp:imprisoned"],
}


def _escape(s: str) -> str:
    """Escape string for SPARQL literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# ── Entity resolution ─────────────────────────────────────────────────────
# Cache: entity_name (lowercase) → URI or None
_entity_cache: dict[str, str | None] = {}


def _load_preresolved_uris():
    """Load URIs already resolved in coverage_analysis_1000.json."""
    path = BASE / "data/eval_1000q/ground-truth/coverage_analysis_1000.json"
    with open(path) as f:
        data = json.load(f)
    for q in data["questions"]:
        for e in q["entities"]:
            if e.get("uri"):
                _entity_cache[e["name"].lower()] = e["uri"]
    print(f"  Pre-loaded {len(_entity_cache)} entity URIs from coverage data")


def resolve_entity(name: str) -> str | None:
    """Resolve entity name → DBpedia URI. Returns URI or None."""
    key = name.lower()
    if key in _entity_cache:
        return _entity_cache[key]

    # Strategy 1: Direct URI construction
    encoded = name.replace(" ", "_")
    uri = f"http://dbpedia.org/resource/{quote(encoded, safe='_,()·')}"
    result = run_sparql(f'ASK {{ <{uri}> ?p ?o }}')
    if result and result.get("boolean"):
        _entity_cache[key] = uri
        return uri

    # Strategy 2: SPARQL label search (exact match)
    escaped = _escape(name)
    result = run_sparql(
        f'SELECT ?s WHERE {{ ?s rdfs:label "{escaped}"@en }} LIMIT 1'
    )
    if result:
        bindings = result.get("results", {}).get("bindings", [])
        if bindings:
            found_uri = bindings[0]["s"]["value"]
            _entity_cache[key] = found_uri
            return found_uri

    # Not found
    _entity_cache[key] = None
    return None


# ── Property verification ─────────────────────────────────────────────────

def check_property_exists(uri: str, prop_name: str) -> tuple[bool, str | None]:
    """Check if entity has property. Returns (found, matching_predicate)."""
    predicates = PROPERTY_MAP.get(prop_name, [])
    if not predicates:
        # Fallback: convert to camelCase and try dbo:/dbp:
        camel = prop_name.replace(" ", "")
        camel = camel[0].lower() + camel[1:]
        predicates = [f"dbo:{camel}", f"dbp:{camel}"]

    for pred in predicates:
        result = run_sparql(f"ASK {{ <{uri}> {pred} ?o }}")
        if result and result.get("boolean"):
            return True, pred

    return False, None


def get_property_value(uri: str, pred: str) -> list[str]:
    """Get all values for a property. Returns list of value strings."""
    result = run_sparql(f"""
        SELECT ?o ?label WHERE {{
            <{uri}> {pred} ?o .
            OPTIONAL {{ ?o rdfs:label ?label . FILTER(LANG(?label) = "en") }}
        }} LIMIT 20
    """)
    if not result:
        return []

    values = []
    for b in result.get("results", {}).get("bindings", []):
        # Prefer label over raw URI/value
        if "label" in b:
            values.append(b["label"]["value"])
        else:
            val = b["o"]["value"]
            # Extract readable name from URI
            if val.startswith("http://dbpedia.org/resource/"):
                val = unquote(val.split("/resource/")[-1]).replace("_", " ")
            values.append(val)
    return values


def value_matches(actual_values: list[str], expected: str) -> bool:
    """Check if any actual value matches the expected evidence value."""
    exp_lower = expected.lower().strip()
    for val in actual_values:
        val_lower = val.lower().strip()
        # Exact match
        if val_lower == exp_lower:
            return True
        # Containment (either direction)
        if exp_lower in val_lower or val_lower in exp_lower:
            return True
    return False


# ── Main verification loop ────────────────────────────────────────────────

def verify_single_hop(subject_name: str, prop_name: str, expected_obj: str) -> dict:
    """Verify one evidence triple [S, P, O] against live DBpedia."""
    result = {
        "subject": subject_name,
        "property": prop_name,
        "expected_object": expected_obj,
        "entity_resolved": False,
        "entity_uri": None,
        "property_exists": False,
        "property_predicate": None,
        "value_match": False,
        "actual_values": [],
    }

    # Step 1: Resolve entity
    uri = resolve_entity(subject_name)
    if not uri:
        return result
    result["entity_resolved"] = True
    result["entity_uri"] = uri

    # Step 2: Check property existence
    found, pred = check_property_exists(uri, prop_name)
    if not found:
        return result
    result["property_exists"] = True
    result["property_predicate"] = pred

    # Step 3: Check value match
    values = get_property_value(uri, pred)
    result["actual_values"] = values[:5]  # Keep top 5 for debugging
    result["value_match"] = value_matches(values, expected_obj)

    return result


def verify_question(question: dict) -> dict:
    """Verify all evidence hops for a single question."""
    hops = []
    all_entities_resolved = True
    all_properties_exist = True
    all_values_match = True
    chain_connected = True

    for i, evidence in enumerate(question["evidences"]):
        s, p, o = evidence
        hop_result = verify_single_hop(s, p, o)
        hop_result["hop_index"] = i + 1
        hops.append(hop_result)

        if not hop_result["entity_resolved"]:
            all_entities_resolved = False
            all_properties_exist = False
            all_values_match = False
            chain_connected = False
        elif not hop_result["property_exists"]:
            all_properties_exist = False
            all_values_match = False
            chain_connected = False
        elif not hop_result["value_match"]:
            all_values_match = False
            chain_connected = False

    return {
        "question_id": question["_id"],
        "question": question["question"],
        "question_type": question["type"],
        "gold_answer": question["answer"],
        "num_hops": len(question["evidences"]),
        "hops": hops,
        "all_entities_resolved": all_entities_resolved,
        "all_properties_exist": all_properties_exist,
        "all_values_match": all_values_match,
        "chain_connected": chain_connected,
    }


def print_statistics(results: list[dict]):
    """Print summary statistics from verification results."""
    total = len(results)
    print(f"\n{'=' * 72}")
    print(f"HOP-BY-HOP VERIFICATION SUMMARY ({total} questions)")
    print(f"{'=' * 72}")

    # Overall stats
    ent_ok = sum(1 for r in results if r["all_entities_resolved"])
    prop_ok = sum(1 for r in results if r["all_properties_exist"])
    val_ok = sum(1 for r in results if r["all_values_match"])
    chain_ok = sum(1 for r in results if r["chain_connected"])

    print(f"\nOverall (all hops must pass):")
    print(f"  All entities resolved:  {ent_ok}/{total} ({100*ent_ok/total:.1f}%)")
    print(f"  All properties exist:   {prop_ok}/{total} ({100*prop_ok/total:.1f}%)")
    print(f"  All values match:       {val_ok}/{total} ({100*val_ok/total:.1f}%)")
    print(f"  Full chain traversable: {chain_ok}/{total} ({100*chain_ok/total:.1f}%)")

    # Per-question-type breakdown
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    print(f"\nBy question type:")
    for qtype in sorted(by_type.keys()):
        items = by_type[qtype]
        n = len(items)
        e = sum(1 for r in items if r["all_entities_resolved"])
        p = sum(1 for r in items if r["all_properties_exist"])
        v = sum(1 for r in items if r["all_values_match"])
        c = sum(1 for r in items if r["chain_connected"])
        print(f"  {qtype} ({n}q): entity={100*e/n:.1f}% "
              f"prop={100*p/n:.1f}% val={100*v/n:.1f}% chain={100*c/n:.1f}%")

    # Per-hop statistics (across all questions)
    hop_stats = defaultdict(lambda: {"total": 0, "ent": 0, "prop": 0, "val": 0})
    for r in results:
        for h in r["hops"]:
            idx = h["hop_index"]
            hop_stats[idx]["total"] += 1
            if h["entity_resolved"]:
                hop_stats[idx]["ent"] += 1
            if h["property_exists"]:
                hop_stats[idx]["prop"] += 1
            if h["value_match"]:
                hop_stats[idx]["val"] += 1

    print(f"\nPer-hop statistics:")
    for idx in sorted(hop_stats.keys()):
        s = hop_stats[idx]
        n = s["total"]
        print(f"  Hop {idx} ({n} triples): entity={100*s['ent']/n:.1f}% "
              f"prop={100*s['prop']/n:.1f}% val={100*s['val']/n:.1f}%")

    # Property-level failure analysis
    prop_total = Counter()
    prop_missing = Counter()
    for r in results:
        for h in r["hops"]:
            prop_total[h["property"]] += 1
            if not h["property_exists"] and h["entity_resolved"]:
                prop_missing[h["property"]] += 1

    print(f"\nProperty existence (entity resolved but property missing):")
    for prop in sorted(prop_total.keys(), key=lambda p: -prop_missing.get(p, 0)):
        t = prop_total[prop]
        m = prop_missing.get(prop, 0)
        found = t - m
        pct = 100 * found / t if t else 0
        print(f"  {prop}: {found}/{t} ({pct:.0f}%) found")


def save_checkpoint(results: list[dict], completed_ids: set[str]):
    """Save checkpoint for resume."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({
            "completed_ids": list(completed_ids),
            "results": results,
        }, f)


def load_checkpoint() -> tuple[set[str], list[dict]]:
    """Load checkpoint if it exists."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        return set(data["completed_ids"]), data["results"]
    return set(), []


def main():
    parser = argparse.ArgumentParser(description="Hop-by-hop KG coverage verification")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions")
    parser.add_argument("--no-values", action="store_true",
                        help="Skip value matching (faster, entity+property only)")
    args = parser.parse_args()

    # Load data
    print("Loading benchmark data...")
    with open(BASE / "data/dev_full.json") as f:
        all_questions = json.load(f)
    with open(BASE / "data/eval_1000q/ground-truth/coverage_analysis_1000.json") as f:
        cov = json.load(f)
    ids_1k = {q["question_id"] for q in cov["questions"]}

    # Filter to 1000q subset
    questions = [q for q in all_questions if q["_id"] in ids_1k]
    print(f"  {len(questions)} questions in 1000q subset")
    print(f"  {sum(len(q['evidences']) for q in questions)} evidence triples to verify")

    # Load pre-resolved entity URIs
    _load_preresolved_uris()

    # Resume from checkpoint if requested
    completed_ids, results = (load_checkpoint() if args.resume else (set(), []))
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} already done")

    # Filter out already-completed questions
    remaining = [q for q in questions if q["_id"] not in completed_ids]
    if args.limit > 0:
        remaining = remaining[:args.limit]
    print(f"  {len(remaining)} questions to process\n")

    # Main verification loop
    start_time = time.time()
    for i, q in enumerate(remaining, 1):
        qid = q["_id"]
        qtype = q["type"]
        n_hops = len(q["evidences"])
        print(f"[{i}/{len(remaining)}] {qtype} ({n_hops} hops): "
              f"{q['question'][:60]}...")

        result = verify_question(q)

        # Quick status for each hop
        for h in result["hops"]:
            status = "✓" if h["value_match"] else (
                "prop✗" if h["entity_resolved"] and not h["property_exists"] else (
                    "val✗" if h["property_exists"] else "ent✗"
                )
            )
            print(f"  Hop {h['hop_index']}: [{status}] {h['subject']}.{h['property']}"
                  f" → {h['expected_object'][:30]}")

        results.append(result)
        completed_ids.add(qid)

        # Checkpoint every 50 questions
        if i % 50 == 0:
            save_checkpoint(results, completed_ids)
            elapsed = time.time() - start_time
            rate = i / elapsed * 60
            print(f"\n  --- Checkpoint: {len(completed_ids)} done, "
                  f"{rate:.1f} q/min ---\n")

    # Final save
    print(f"\nSaving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"results": results}, f, indent=2)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # Print statistics
    print_statistics(results)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
