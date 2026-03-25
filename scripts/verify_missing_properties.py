"""
Verify "missing" properties against live DBpedia SPARQL using expanded predicate mappings.
Recomputes corrected property coverage for the 1000-question eval set.
"""

import json
import time
import sys
from SPARQLWrapper import SPARQLWrapper, JSON

DATA_PATH = "/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa/data/eval_1000q/coverage_analysis_1000.json"
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
RATE_LIMIT = 0.2  # seconds between calls

EXPANDED = {
    "country of origin": ["dbo:country", "dbp:country", "dbp:origin", "dbp:countryOfOrigin"],
    "place of burial": ["dbp:placeOfBurial", "dbp:burialPlace", "dbo:restingPlace", "dbp:restingPlace"],
    "inception": ["dbo:foundingDate", "dbp:founded", "dbo:formationDate", "dbp:established"],
    "performer": ["dbo:artist", "dbp:artist", "dbo:musicalArtist", "dbp:performer"],
    "award received": ["dbo:award", "dbp:awards", "dbp:award"],
    "country": ["dbo:country", "dbp:country", "dbo:location", "dbp:location"],
    "educated at": ["dbo:almaMater", "dbp:almaMater", "dbp:education", "dbo:education"],
    "composer": ["dbo:musicComposer", "dbp:composer", "dbp:music", "dbo:composer"],
    "place of birth": ["dbo:birthPlace", "dbp:birthPlace", "dbp:placeOfBirth", "dbp:born"],
}

PREFIXES = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX dbr: <http://dbpedia.org/resource/>
"""


def ask_property_exists(sparql: SPARQLWrapper, uri: str, predicates: list[str]) -> bool:
    """Check if any of the given predicates exist for the entity URI."""
    # Build UNION of ASK patterns
    patterns = "\n  UNION\n  ".join(
        f"{{ <{uri}> {pred} ?v }}" for pred in predicates
    )
    query = f"{PREFIXES}\nASK {{\n  {patterns}\n}}"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        result = sparql.query().convert()
        return result.get("boolean", False)
    except Exception as e:
        print(f"  SPARQL error for {uri}: {e}", file=sys.stderr)
        return False


def main():
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.addCustomHttpHeader("User-Agent", "DBpediaQA-Coverage-Verifier/1.0")

    with open(DATA_PATH) as f:
        data = json.load(f)

    questions = data["questions"]

    # Collect all unique (uri, property) pairs that need verification
    to_check: set[tuple[str, str]] = set()
    for q in questions:
        for ent in q.get("entities", []):
            if not ent.get("exists"):
                continue
            uri = ent.get("uri", "")
            if not uri:
                continue
            for prop in ent.get("properties_missing", []):
                if prop in EXPANDED:
                    to_check.add((uri, prop))

    total_pairs = len(to_check)
    print(f"Pairs to verify: {total_pairs}")

    # Run ASK queries
    verified_found: set[tuple[str, str]] = set()
    per_property_stats: dict[str, dict] = {p: {"checked": 0, "found": 0} for p in EXPANDED}

    for i, (uri, prop) in enumerate(sorted(to_check), 1):
        predicates = EXPANDED[prop]
        found = ask_property_exists(sparql, uri, predicates)
        per_property_stats[prop]["checked"] += 1
        if found:
            per_property_stats[prop]["found"] += 1
            verified_found.add((uri, prop))
        if i % 20 == 0 or i == total_pairs:
            print(f"  [{i}/{total_pairs}] checked — found so far: {len(verified_found)}")
        time.sleep(RATE_LIMIT)

    print("\nPer-property results:")
    for prop, stats in per_property_stats.items():
        checked = stats["checked"]
        found = stats["found"]
        rate = (found / checked * 100) if checked > 0 else 0
        print(f"  {prop:25s}: {found:3d}/{checked:3d} found ({rate:.0f}%)")

    # Recompute coverage
    original_all_exist = sum(1 for q in questions if q.get("all_properties_exist"))

    corrected_all_exist = 0
    for q in questions:
        if q.get("all_properties_exist"):
            corrected_all_exist += 1
            continue
        # Check if all missing properties (for entities that exist) are now resolved
        all_resolved = True
        for ent in q.get("entities", []):
            if not ent.get("exists"):
                continue
            uri = ent.get("uri", "")
            for prop in ent.get("properties_missing", []):
                # A property is resolved if it was found in verification,
                # OR it was not in our check list (not in EXPANDED — those stay missing)
                if prop not in EXPANDED:
                    all_resolved = False
                    break
                if (uri, prop) not in verified_found:
                    all_resolved = False
                    break
            if not all_resolved:
                break
        if all_resolved:
            corrected_all_exist += 1

    print(f"\nOriginal property coverage : {original_all_exist}/1000 ({original_all_exist/10:.1f}%)")
    print(f"Corrected property coverage: {corrected_all_exist}/1000 ({corrected_all_exist/10:.1f}%)")
    print(f"Net gain from verification : +{corrected_all_exist - original_all_exist} questions")

    # Save verification results for reference
    out_path = "/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa/data/eval_1000q/property_verification_results.json"
    output = {
        "verified_found_pairs": sorted(list(verified_found)),
        "per_property_stats": per_property_stats,
        "original_property_coverage": original_all_exist,
        "corrected_property_coverage": corrected_all_exist,
        "total_pairs_checked": total_pairs,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
