"""Exact correction of property coverage in coverage_analysis_1000.json.

Re-checks ALL (entity_uri, property_name) pairs marked missing for properties
that had >0% found rate in the sampled verification. Skips properties confirmed
0% in sampling (director, father, publication date, etc.) to save SPARQL calls.

Outputs corrected JSON alongside the original.
"""

import json
import time
from pathlib import Path

from SPARQLWrapper import SPARQLWrapper, JSON

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
RATE_LIMIT_S = 0.25  # seconds between SPARQL calls

INPUT = Path("data/eval_1000q/coverage_analysis_1000.json")
OUTPUT = Path("data/eval_1000q/coverage_analysis_1000_corrected.json")

# Only re-check properties with >0% found rate from sampled verification.
# Maps property_name -> list of DBpedia predicates to try (expanded mapping).
RECHECK_PROPERTIES = {
    "country of origin": [
        "dbo:country", "dbp:country", "dbo:nationality",
        "dbp:origin", "dbp:countryOfOrigin",
    ],
    "performer": [
        "dbo:artist", "dbp:artist", "dbo:musicalArtist", "dbp:performer",
    ],
    "place of burial": [
        "dbp:placeOfBurial", "dbp:burialPlace", "dbo:restingPlace", "dbp:restingPlace",
    ],
    "inception": [
        "dbo:foundingDate", "dbp:founded", "dbo:formationDate", "dbp:established",
    ],
    "award received": [
        "dbo:award", "dbp:awards", "dbp:award",
    ],
    "country": [
        "dbo:country", "dbp:country", "dbo:location",
    ],
    "educated at": [
        "dbo:almaMater", "dbp:almaMater", "dbp:education", "dbo:education",
    ],
    "country of citizenship": [
        "dbo:citizenship", "dbp:citizenship", "dbo:nationality",
        "dbp:nationality", "dbo:stateOfOrigin", "dbp:allegiance",
    ],
    "composer": [
        "dbo:musicComposer", "dbp:composer", "dbp:music", "dbo:composer",
    ],
}


# ---------------------------------------------------------------------------
# SPARQL helper
# ---------------------------------------------------------------------------
def _expand_prefix(pred: str) -> str:
    """Expand dbo:/dbp: shorthand to full URI."""
    return (pred
            .replace("dbo:", "http://dbpedia.org/ontology/")
            .replace("dbp:", "http://dbpedia.org/property/"))


def ask_predicate_exists(sparql: SPARQLWrapper, entity_uri: str, pred_short: str) -> bool:
    """Return True if <entity_uri> <pred> ?o triple exists."""
    pred_uri = _expand_prefix(pred_short)
    query = f"ASK {{ <{entity_uri}> <{pred_uri}> ?o . }}"
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
        time.sleep(RATE_LIMIT_S)
        return bool(result.get("boolean", False))
    except Exception:
        time.sleep(RATE_LIMIT_S)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading {INPUT} ...")
    with open(INPUT) as f:
        data = json.load(f)

    questions = data["questions"]
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setTimeout(30)

    # Collect all (entity_uri, prop_name) pairs to recheck
    pairs_to_check = []  # (q_idx, ent_idx, prop_name, entity_uri)
    for qi, q in enumerate(questions):
        for ei, ent in enumerate(q.get("entities", [])):
            uri = ent.get("uri", "")
            if not uri:
                continue
            for prop in ent.get("properties_missing", []):
                if prop in RECHECK_PROPERTIES:
                    pairs_to_check.append((qi, ei, prop, uri))

    print(f"Pairs to recheck: {len(pairs_to_check)}")
    print(f"Properties being rechecked: {sorted(RECHECK_PROPERTIES.keys())}")

    # Check each pair
    found_count = 0
    for i, (qi, ei, prop_name, uri) in enumerate(pairs_to_check):
        predicates = RECHECK_PROPERTIES[prop_name]
        found = False
        for pred in predicates:
            if ask_predicate_exists(sparql, uri, pred):
                found = True
                break

        if found:
            found_count += 1
            ent = questions[qi]["entities"][ei]
            # Move from properties_missing to properties_found
            ent["properties_missing"].remove(prop_name)
            ent["properties_found"].append(prop_name)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Progress: {i+1}/{len(pairs_to_check)} "
                  f"({found_count} found so far)")

    print(f"\nTotal newly found: {found_count}/{len(pairs_to_check)}")

    # Recalculate all_properties_exist and is_answerable for each question
    original_prop_true = sum(1 for q in questions if q["all_properties_exist"])
    original_answerable = sum(1 for q in questions if q["is_answerable"])

    for q in questions:
        # all_properties_exist = True only if NO entity has any missing property
        all_props = True
        for ent in q.get("entities", []):
            if ent.get("properties_missing"):
                all_props = False
                break
        q["all_properties_exist"] = all_props
        # is_answerable = all entities exist AND all properties exist
        q["is_answerable"] = q["all_entities_exist"] and all_props
        # Update failure_reason
        if q["is_answerable"]:
            q["failure_reason"] = None
        elif not q["all_entities_exist"]:
            pass  # keep existing entity failure reason
        elif not all_props:
            # Find first missing property
            for ent in q.get("entities", []):
                if ent.get("properties_missing"):
                    q["failure_reason"] = (
                        f"Property missing: {ent['name']}.{ent['properties_missing'][0]}"
                    )
                    break

    new_prop_true = sum(1 for q in questions if q["all_properties_exist"])
    new_answerable = sum(1 for q in questions if q["is_answerable"])

    total = len(questions)
    ent_true = sum(1 for q in questions if q["all_entities_exist"])

    print(f"\n{'='*60}")
    print(f"EXACT CORRECTED COVERAGE")
    print(f"{'='*60}")
    print(f"Entity coverage:   {ent_true}/{total} = {100*ent_true/total:.1f}% (unchanged)")
    print(f"Property coverage: {original_prop_true} → {new_prop_true}/{total} "
          f"= {100*new_prop_true/total:.1f}% (was {100*original_prop_true/total:.1f}%)")
    print(f"Answerable:        {original_answerable} → {new_answerable}/{total} "
          f"= {100*new_answerable/total:.1f}% (was {100*original_answerable/total:.1f}%)")
    print(f"\nDelta: +{new_prop_true - original_prop_true} property-complete, "
          f"+{new_answerable - original_answerable} answerable")

    # Update statistics in the data
    data["statistics"]["overall"]["property_coverage"] = 100 * new_prop_true / total
    data["statistics"]["overall"]["answerable"] = 100 * new_answerable / total
    data["statistics"]["overall"]["answerable_count"] = new_answerable

    # Save corrected JSON
    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
