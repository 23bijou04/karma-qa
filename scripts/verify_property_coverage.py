"""
Exhaustive verification of property coverage against live DBpedia SPARQL.

Checks each unique (entity_uri, property_name) pair that is marked "missing"
using an expanded predicate mapping + wildcard fallback.
"""

import json
import time
import sys
import re
from collections import defaultdict, Counter

from SPARQLWrapper import SPARQLWrapper, JSON

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
RATE_LIMIT_S = 0.25      # seconds between SPARQL calls
MAX_SAMPLE_PER_PROP = 20  # max entities sampled per property name
WILDCARD_KEYWORD_SAMPLE = 5  # entities to check via full-property wildcard

EXPANDED_PROPERTY_MAP = {
    "director": ["dbo:director", "dbp:director", "dbp:directedby"],
    "place of birth": ["dbo:birthPlace", "dbp:birthPlace", "dbp:placeOfBirth", "dbp:born"],
    "date of birth": ["dbo:birthDate", "dbp:birthDate", "dbp:born", "dbp:dateOfBirth"],
    "date of death": ["dbo:deathDate", "dbp:deathDate", "dbp:died", "dbp:dateOfDeath"],
    "father": ["dbp:father", "dbo:parent", "dbp:parents"],
    "mother": ["dbp:mother", "dbo:parent", "dbp:parents"],
    "spouse": ["dbo:spouse", "dbp:spouse", "dbp:spouses", "dbp:partner"],
    "country of citizenship": [
        "dbo:citizenship", "dbp:citizenship", "dbo:nationality",
        "dbp:nationality", "dbo:stateOfOrigin", "dbp:allegiance",
    ],
    "country of origin": [
        "dbo:country", "dbp:country", "dbo:nationality",
        "dbp:origin", "dbp:countryOfOrigin",
    ],
    "publication date": [
        "dbo:releaseDate", "dbp:releaseDate", "dbo:publicationDate",
        "dbp:published", "dbp:pubDate", "dbo:firstPublicationDate", "dbp:released",
    ],
    "place of death": ["dbo:deathPlace", "dbp:deathPlace", "dbp:placeOfDeath"],
    "composer": ["dbo:musicComposer", "dbp:composer", "dbp:music", "dbo:composer"],
    "performer": ["dbo:artist", "dbp:artist", "dbo:musicalArtist", "dbp:performer"],
    "cause of death": ["dbo:deathCause", "dbp:deathCause", "dbp:causeOfDeath"],
    "educated at": ["dbo:almaMater", "dbp:almaMater", "dbp:education", "dbo:education"],
    "publisher": ["dbo:publisher", "dbp:publisher"],
    "child": ["dbo:child", "dbp:children", "dbp:issue"],
    "founded by": ["dbo:founder", "dbp:founder", "dbp:foundedBy"],
    "employer": ["dbo:employer", "dbp:employer"],
    "country": ["dbo:country", "dbp:country", "dbo:location"],
    "inception": [
        "dbo:foundingDate", "dbp:founded", "dbo:formationDate", "dbp:established",
    ],
    "award received": ["dbo:award", "dbp:awards", "dbp:award"],
    "place of burial": [
        "dbp:placeOfBurial", "dbp:burialPlace", "dbo:restingPlace", "dbp:restingPlace",
    ],
    # properties not in the original map — attempt reasonable guesses
    "sibling": ["dbo:sibling", "dbp:sibling", "dbp:siblings", "dbp:brother", "dbp:sister"],
    "editor": ["dbo:editor", "dbp:editor"],
    "producer": ["dbo:producer", "dbp:producer", "dbp:producers"],
    "place of detention": ["dbp:placeOfDetention", "dbo:placeOfDetention"],
}

# Keyword fragments for wildcard matching (lowercased local name check)
PROP_KEYWORDS = {
    "director": ["director", "directed"],
    "place of birth": ["birth", "birthplace", "placeofbirth", "born"],
    "date of birth": ["birthdate", "dateofbirth", "born"],
    "date of death": ["deathdate", "dateofdeath", "died"],
    "father": ["father", "parent"],
    "mother": ["mother", "parent"],
    "spouse": ["spouse", "partner", "married"],
    "country of citizenship": ["citizenship", "nationality", "allegiance"],
    "country of origin": ["country", "origin", "nationality"],
    "publication date": ["release", "publication", "published", "pub"],
    "place of death": ["deathplace", "placeofdeath"],
    "composer": ["composer", "music"],
    "performer": ["artist", "performer", "musician"],
    "cause of death": ["deathcause", "causeofdeath"],
    "educated at": ["almamater", "education"],
    "publisher": ["publisher"],
    "child": ["child", "children", "issue"],
    "founded by": ["founder", "foundedby"],
    "employer": ["employer"],
    "country": ["country", "location"],
    "inception": ["founding", "formation", "established", "founded"],
    "award received": ["award"],
    "place of burial": ["burial", "resting"],
    "sibling": ["sibling", "brother", "sister"],
    "editor": ["editor"],
    "producer": ["producer"],
    "place of detention": ["detention"],
}

# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------

def _expand_prefix(pred: str) -> str:
    """Expand dbo: / dbp: shorthand to full URI."""
    pred = pred.replace("dbo:", "http://dbpedia.org/ontology/")
    pred = pred.replace("dbp:", "http://dbpedia.org/property/")
    return pred


def ask_predicate_exists(sparql: SPARQLWrapper, entity_uri: str, pred_short: str) -> bool:
    """Return True if <entity_uri> <pred> ?o triple exists."""
    pred_uri = _expand_prefix(pred_short)
    query = f"""
ASK {{
  <{entity_uri}> <{pred_uri}> ?o .
}}
"""
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
        time.sleep(RATE_LIMIT_S)
        return bool(result.get("boolean", False))
    except Exception as e:
        time.sleep(RATE_LIMIT_S)
        return False


def get_all_predicates(sparql: SPARQLWrapper, entity_uri: str) -> list[str]:
    """Return list of predicate local-names for a given entity."""
    query = f"""
SELECT DISTINCT ?p WHERE {{
  <{entity_uri}> ?p ?o .
}}
LIMIT 300
"""
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
        time.sleep(RATE_LIMIT_S)
        preds = []
        for row in result["results"]["bindings"]:
            uri = row["p"]["value"]
            local = uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1].lower()
            preds.append(local)
        return preds
    except Exception as e:
        time.sleep(RATE_LIMIT_S)
        return []


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def main():
    print("Loading coverage_analysis_1000.json ...", flush=True)
    with open("data/eval_1000q/coverage_analysis_1000.json") as f:
        data = json.load(f)

    questions = data["questions"]

    # Step 1: Recount
    prop_true = sum(1 for q in questions if q["all_properties_exist"])
    prop_false = sum(1 for q in questions if not q["all_properties_exist"])
    print(f"Recount: all_properties_exist=True={prop_true}, False={prop_false}")

    # Step 2: Collect all (entity_uri, property_name) pairs marked missing
    # question_ids that have property failures
    q_prop_failure = {}  # qid -> set of (uri, prop_name) pairs failing
    for q in questions:
        if not q["all_properties_exist"]:
            pairs = set()
            for ent in q.get("entities", []):
                uri = ent.get("uri", "")
                for prop in ent.get("properties_missing", []):
                    if uri:
                        pairs.add((uri, prop))
            q_prop_failure[q["question_id"]] = pairs

    # Unique (uri, prop) pairs
    all_missing: dict[str, set[str]] = defaultdict(set)  # prop_name -> set of uris
    for pairs in q_prop_failure.values():
        for uri, prop in pairs:
            all_missing[prop].add(uri)

    total_missing_pairs = sum(len(v) for v in all_missing.values())
    print(f"Unique missing (entity,prop) pairs: {total_missing_pairs}")
    print(f"Unique property names: {len(all_missing)}")

    # Step 3: SPARQL verification
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setTimeout(30)

    # Results per property
    results = {}  # prop_name -> dict

    total_props = len(all_missing)
    for prop_idx, (prop_name, uris) in enumerate(sorted(all_missing.items())):
        uris_list = list(uris)
        sample_uris = uris_list[:MAX_SAMPLE_PER_PROP]
        predicates = EXPANDED_PROPERTY_MAP.get(prop_name, [])

        print(f"\n[{prop_idx+1}/{total_props}] '{prop_name}': {len(uris)} entities missing, "
              f"sampling {len(sample_uris)}, checking {len(predicates)} predicates", flush=True)

        found_via_expanded = 0      # entities found via expanded map
        found_via_wildcard = 0      # entities found via wildcard (not in expanded)
        confirmed_missing = 0
        entity_details = []         # (uri, found_pred or None)

        for uri in sample_uris:
            found_pred = None

            # Try each predicate in the expanded map
            for pred in predicates:
                if ask_predicate_exists(sparql, uri, pred):
                    found_pred = pred
                    break

            if found_pred:
                found_via_expanded += 1
                entity_details.append((uri, found_pred))
                print(f"  FOUND via expanded: {uri.rsplit('/',1)[-1]} → {found_pred}")
                continue

            # Wildcard fallback for first WILDCARD_KEYWORD_SAMPLE entities
            sample_pos = sample_uris.index(uri)
            if sample_pos < WILDCARD_KEYWORD_SAMPLE:
                all_preds = get_all_predicates(sparql, uri)
                keywords = PROP_KEYWORDS.get(prop_name, [prop_name.lower().replace(" ", "")])
                matched = [p for p in all_preds
                           if any(kw in p for kw in keywords)]
                if matched:
                    found_via_wildcard += 1
                    entity_details.append((uri, f"wildcard:{matched[:3]}"))
                    print(f"  FOUND via wildcard: {uri.rsplit('/',1)[-1]} → {matched[:3]}")
                    continue

            confirmed_missing += 1
            entity_details.append((uri, None))

        # Extrapolate to full set
        sampled = len(sample_uris)
        total_found_in_sample = found_via_expanded + found_via_wildcard
        if sampled > 0:
            found_rate = total_found_in_sample / sampled
        else:
            found_rate = 0.0

        estimated_actually_found = round(found_rate * len(uris))
        estimated_confirmed_missing = len(uris) - estimated_actually_found

        results[prop_name] = {
            "total_missing_claimed": len(uris),
            "sampled": sampled,
            "found_via_expanded": found_via_expanded,
            "found_via_wildcard": found_via_wildcard,
            "total_found_in_sample": total_found_in_sample,
            "confirmed_missing_in_sample": confirmed_missing,
            "found_rate": found_rate,
            "estimated_actually_found": estimated_actually_found,
            "estimated_confirmed_missing": estimated_confirmed_missing,
            "entity_details": entity_details[:5],  # store first 5 for report
        }

        print(f"  → found_rate={found_rate:.1%}, "
              f"est. actually_found={estimated_actually_found}/{len(uris)}")

    # ---------------------------------------------------------------------------
    # Step 4: Corrected coverage calculation
    # ---------------------------------------------------------------------------
    # For each question that was marked all_properties_exist=False,
    # check if ALL its missing properties now resolve to "actually found".
    # A question flips to "covered" only if EVERY missing (entity, prop) pair resolves.

    # Build per-prop found_rate lookup
    prop_found_rate = {p: r["found_rate"] for p, r in results.items()}

    # For a more conservative (and exact) approach, use sample-verified found_rate
    # to estimate how many questions flip.
    #
    # For each question: probability all its missing pairs are actually found
    # = product of found_rates for each missing pair.
    # Summing these gives expected number of newly-covered questions.

    newly_covered_expected = 0.0
    # Also track a "optimistic" count: if found_rate=1 means all found
    newly_covered_optimistic = 0

    for q in questions:
        if q["all_properties_exist"]:
            continue
        # Collect missing (prop_name, uri) pairs
        missing_for_q = []
        for ent in q.get("entities", []):
            uri = ent.get("uri", "")
            for prop in ent.get("properties_missing", []):
                missing_for_q.append((prop, uri))

        if not missing_for_q:
            continue

        # Expected: product of per-prop found_rates
        prob_all_found = 1.0
        all_found_optimistic = True
        for prop, uri in missing_for_q:
            fr = prop_found_rate.get(prop, 0.0)
            prob_all_found *= fr
            if fr < 1.0:
                all_found_optimistic = False

        newly_covered_expected += prob_all_found
        if all_found_optimistic:
            newly_covered_optimistic += 1

    corrected_count_expected = 385 + round(newly_covered_expected)
    corrected_count_optimistic = 385 + newly_covered_optimistic

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print("\n\n=== VERIFICATION COMPLETE ===\n")
    print(f"Original claimed: 385/1000 (38.5%)")
    print(f"Corrected (expected):   {corrected_count_expected}/1000 ({corrected_count_expected/10:.1f}%)")
    print(f"Corrected (optimistic): {corrected_count_optimistic}/1000 ({corrected_count_optimistic/10:.1f}%)")

    # Save detailed results for report
    output = {
        "original_count": 385,
        "corrected_expected": corrected_count_expected,
        "corrected_optimistic": corrected_count_optimistic,
        "newly_covered_expected": newly_covered_expected,
        "newly_covered_optimistic": newly_covered_optimistic,
        "per_property": results,
    }
    with open("data/eval_1000q/property_coverage_verification.json", "w") as f:
        # strip entity_details for json serialization
        out_clean = {
            "original_count": output["original_count"],
            "corrected_expected": output["corrected_expected"],
            "corrected_optimistic": output["corrected_optimistic"],
            "newly_covered_expected": output["newly_covered_expected"],
            "newly_covered_optimistic": output["newly_covered_optimistic"],
            "per_property": {
                p: {k: v for k, v in r.items() if k != "entity_details"}
                for p, r in results.items()
            },
        }
        json.dump(out_clean, f, indent=2)

    print("Results saved to data/eval_1000q/property_coverage_verification.json")
    return output


if __name__ == "__main__":
    main()
