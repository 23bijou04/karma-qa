"""
verify_kg_scenarios_live.py

Verify Layer 5 (KG Coverage Scenarios) against live DBpedia SPARQL.
Samples 5 questions per scenario (A-F, seed=42) and checks entity/property
existence claims against the real endpoint.
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
DELAY = 0.2  # seconds between queries
SAMPLE_N = 5
SEED = 42

PROPERTY_MAP = {
    "director":              ["dbo:director", "dbp:director"],
    "place of birth":        ["dbo:birthPlace", "dbp:birthPlace"],
    "date of birth":         ["dbo:birthDate", "dbp:birthDate"],
    "date of death":         ["dbo:deathDate", "dbp:deathDate"],
    "father":                ["dbp:father", "dbo:parent"],
    "mother":                ["dbp:mother", "dbo:parent"],
    "spouse":                ["dbo:spouse", "dbp:spouse"],
    "country of citizenship":["dbo:citizenship", "dbp:citizenship",
                              "dbo:nationality", "dbp:nationality"],
    "country of origin":     ["dbo:country", "dbp:country"],
    "publication date":      ["dbo:releaseDate", "dbp:releaseDate",
                              "dbo:publicationDate"],
    "composer":              ["dbo:musicComposer", "dbp:composer"],
    "performer":             ["dbo:artist", "dbp:artist"],
    "cause of death":        ["dbo:deathCause", "dbp:deathCause"],
    "educated at":           ["dbo:almaMater", "dbp:almaMater"],
    "country":               ["dbo:country", "dbp:country"],
}

PREFIX = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""

# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------
sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.setReturnFormat(JSON)
sparql.setTimeout(15)
sparql.addCustomHttpHeader("User-Agent", "DBpedia-QA-Verifier/1.0")


def ask(query: str) -> bool:
    """Run a SPARQL ASK query, return True/False. Returns None on error."""
    sparql.setQuery(PREFIX + query)
    try:
        result = sparql.query().convert()
        time.sleep(DELAY)
        return bool(result.get("boolean", False))
    except Exception as e:
        print(f"    [SPARQL ERROR] {e}", file=sys.stderr)
        time.sleep(DELAY)
        return None


def entity_exists(uri: str) -> bool:
    q = f"ASK {{ <{uri}> ?p ?o }}"
    return ask(q)


def property_exists(entity_uri: str, prop_label: str) -> bool:
    """Returns True if ANY known predicate for prop_label holds for entity_uri."""
    predicates = PROPERTY_MAP.get(prop_label.lower(), [])
    if not predicates:
        return None  # unknown property, can't verify

    parts = []
    for pred in predicates:
        parts.append(f"{{ <{entity_uri}> {pred} ?o }}")
    union = " UNION ".join(parts)
    q = f"ASK {{ {union} }}"
    return ask(q)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent / "data" / "eval_1000q"

print("Loading scenario_classification_1000.json ...")
with open(BASE / "scenario_classification_1000.json") as f:
    sc_data = json.load(f)

print("Loading coverage_analysis_1000.json ...")
with open(BASE / "coverage_analysis_1000.json") as f:
    ca_data = json.load(f)

sc_questions = sc_data["questions"]          # list[dict], has scenario field
ca_questions = ca_data["questions"]          # list[dict], has uri

# Build lookup: question_id -> coverage record (for URIs)
ca_by_id = {q["question_id"]: q for q in ca_questions}

# ---------------------------------------------------------------------------
# Verify report counts
# ---------------------------------------------------------------------------
scenario_counts = defaultdict(int)
for q in sc_questions:
    scenario_counts[q["scenario"]] += 1

REPORTED = {
    "A": 346, "B": 304, "C": 158, "D": 128, "E": 39, "F": 25
}

print("\n=== Scenario Count Verification ===")
count_ok = True
for scenario, reported in sorted(REPORTED.items()):
    actual = scenario_counts[scenario]
    match = "OK" if actual == reported else "MISMATCH"
    if actual != reported:
        count_ok = False
    print(f"  Scenario {scenario}: actual={actual:3d}  reported={reported:3d}  [{match}]")
print(f"  Total: {len(sc_questions)}")
print(f"  All counts match: {count_ok}")

# ---------------------------------------------------------------------------
# Sample 5 per scenario
# ---------------------------------------------------------------------------
rng = random.Random(SEED)

by_scenario = defaultdict(list)
for q in sc_questions:
    by_scenario[q["scenario"]].append(q)

samples = {}
for scenario in sorted(REPORTED.keys()):
    pool = by_scenario[scenario]
    n = min(SAMPLE_N, len(pool))
    samples[scenario] = rng.sample(pool, n)

# ---------------------------------------------------------------------------
# Live verification
# ---------------------------------------------------------------------------
SCENARIO_LABELS = {
    "A": "KG_COMPLETE",
    "B": "HOP1_MISSING",
    "C": "HOP2_MISSING",
    "D": "BOTH_HOPS_MISSING",
    "E": "ENTITY_NOT_FOUND",
    "F": "ENTITY_NOT_FOUND + PROP",
}

results = {}   # scenario -> list of per-question result dicts
mismatches = []

for scenario in sorted(REPORTED.keys()):
    label = SCENARIO_LABELS[scenario]
    print(f"\n{'='*60}")
    print(f"Scenario {scenario}: {label}")
    print(f"{'='*60}")

    scenario_results = []
    for q in samples[scenario]:
        qid = q["question_id"]
        question = q["question"]
        ca_q = ca_by_id.get(qid)

        print(f"\n  Q: {question[:80]}")
        print(f"     ID: {qid}")

        if ca_q is None:
            print("     [SKIP] Not found in coverage_analysis")
            scenario_results.append({"qid": qid, "status": "skip", "checks": []})
            continue

        entities = ca_q.get("entities", [])
        checks = []

        for ent in entities:
            ent_name = ent.get("name", "?")
            ent_uri = ent.get("uri", "")
            claimed_exists = ent.get("exists", False)

            print(f"\n     Entity: {ent_name}")
            print(f"       URI: {ent_uri}")

            if not ent_uri:
                print("       [SKIP] No URI available")
                checks.append({"entity": ent_name, "status": "no_uri"})
                continue

            # --- entity existence check ---
            live_exists = entity_exists(ent_uri)
            ent_match = (live_exists == claimed_exists) if live_exists is not None else None

            status_str = "OK" if ent_match else ("MISMATCH" if ent_match is False else "ERROR")
            print(f"       Exists claimed={claimed_exists}  live={live_exists}  [{status_str}]")

            if ent_match is False:
                mismatches.append({
                    "type": "entity_existence",
                    "qid": qid,
                    "entity": ent_name,
                    "uri": ent_uri,
                    "claimed": claimed_exists,
                    "live": live_exists,
                })

            check = {
                "entity": ent_name,
                "uri": ent_uri,
                "entity_claimed_exists": claimed_exists,
                "entity_live_exists": live_exists,
                "entity_match": ent_match,
                "property_checks": [],
            }

            # --- property checks (only if entity exists in at least one source) ---
            if claimed_exists and ent_uri:
                for prop in ent.get("properties_found", []):
                    live = property_exists(ent_uri, prop)
                    match = (live == True) if live is not None else None
                    status_str = "OK" if match else ("MISMATCH" if match is False else "ERROR")
                    print(f"       Prop FOUND '{prop}': live={live}  [{status_str}]")
                    check["property_checks"].append({
                        "prop": prop, "claimed": "found", "live": live, "match": match
                    })
                    if match is False:
                        mismatches.append({
                            "type": "property_found_but_live_missing",
                            "qid": qid, "entity": ent_name, "prop": prop,
                            "uri": ent_uri, "claimed": "found", "live": live,
                        })

                for prop in ent.get("properties_missing", []):
                    live = property_exists(ent_uri, prop)
                    match = (live == False) if live is not None else None
                    status_str = "OK" if match else ("MISMATCH" if match is False else "ERROR")
                    print(f"       Prop MISSING '{prop}': live={live}  [{status_str}]")
                    check["property_checks"].append({
                        "prop": prop, "claimed": "missing", "live": live, "match": match
                    })
                    if match is False:
                        mismatches.append({
                            "type": "property_missing_but_live_found",
                            "qid": qid, "entity": ent_name, "prop": prop,
                            "uri": ent_uri, "claimed": "missing", "live": live,
                        })

            checks.append(check)

        # Summarize per-question: confirmed if all non-error checks match
        all_checks_flat = []
        for c in checks:
            if c.get("status") in ("no_uri", "skip"):
                continue
            if c.get("entity_match") is not None:
                all_checks_flat.append(c["entity_match"])
            for pc in c.get("property_checks", []):
                if pc["match"] is not None:
                    all_checks_flat.append(pc["match"])

        confirmed = all(all_checks_flat) if all_checks_flat else None
        scenario_results.append({
            "qid": qid,
            "question": question,
            "confirmed": confirmed,
            "checks": checks,
        })
        print(f"\n     => Question confirmed={confirmed}")

    results[scenario] = scenario_results

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n\n" + "="*60)
print("SUMMARY")
print("="*60)

total_confirmed = 0
total_questions = 0
total_errors = 0

for scenario in sorted(REPORTED.keys()):
    label = SCENARIO_LABELS[scenario]
    qs = results[scenario]
    n_confirmed = sum(1 for q in qs if q.get("confirmed") is True)
    n_contradict = sum(1 for q in qs if q.get("confirmed") is False)
    n_error = sum(1 for q in qs if q.get("confirmed") is None)
    n = len(qs)
    print(f"\n  Scenario {scenario} ({label}):")
    print(f"    Sampled: {n}  Confirmed: {n_confirmed}  Contradicted: {n_contradict}  Errors/Unknown: {n_error}")
    total_confirmed += n_confirmed
    total_questions += n
    total_errors += n_error

overall_rate = total_confirmed / (total_questions - total_errors) * 100 if (total_questions - total_errors) > 0 else 0
print(f"\n  Overall confirmation rate: {total_confirmed}/{total_questions - total_errors} = {overall_rate:.1f}%")
print(f"  (Excluding {total_errors} questions with SPARQL errors/unknowns)")

print(f"\n  Total mismatches found: {len(mismatches)}")
if mismatches:
    print("\n  --- Mismatch Details ---")
    for m in mismatches:
        print(f"    [{m['type']}]")
        print(f"      Q: {m['qid']}")
        print(f"      Entity: {m.get('entity')}  URI: {m.get('uri')}")
        if "prop" in m:
            print(f"      Property: {m['prop']}  claimed={m['claimed']}  live={m['live']}")
        else:
            print(f"      Claimed exists={m['claimed']}  live={m['live']}")

print("\nDone.")
