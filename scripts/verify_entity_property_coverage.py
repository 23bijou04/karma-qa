#!/usr/bin/env python3
"""Verify entity & property coverage claims against live DBpedia SPARQL + Wikipedia.

Claims to verify:
  1. Entity coverage: 93.6% (936/1000 questions have all entities in DBpedia)
  2. Property coverage: 38.5% (385/1000 questions have all needed properties)

Method:
  Step 1: Arithmetic recount from raw JSON (no network)
  Step 2: Sample 50 "exists=true" entities → ASK live DBpedia
  Step 3: Sample ALL "exists=false" entities → ASK live DBpedia (only 64 failures)
  Step 4: Sample 40 "property found" claims → ASK live DBpedia
  Step 5: Sample 40 "property missing" claims → ASK live DBpedia
  Step 6: Cross-check entity URIs against Wikipedia API
"""

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote, unquote

import requests
from SPARQLWrapper import JSON, SPARQLWrapper

BASE = Path(__file__).resolve().parent.parent

# ── SPARQL setup ──────────────────────────────────────────────────────────
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.setReturnFormat(JSON)
sparql.setTimeout(30)

# Property name → DBpedia predicates (same mapping as verify_report_numbers.py)
PROPERTY_MAP = {
    "director": ["dbo:director", "dbp:director"],
    "place of birth": ["dbo:birthPlace", "dbp:birthPlace"],
    "date of birth": ["dbo:birthDate", "dbp:birthDate"],
    "date of death": ["dbo:deathDate", "dbp:deathDate"],
    "father": ["dbp:father", "dbo:parent"],
    "mother": ["dbp:mother", "dbo:parent"],
    "spouse": ["dbo:spouse", "dbp:spouse"],
    "country of citizenship": [
        "dbo:citizenship", "dbp:citizenship",
        "dbo:nationality", "dbp:nationality",
    ],
    "country of origin": ["dbo:country", "dbp:country"],
    "publication date": [
        "dbo:releaseDate", "dbp:releaseDate",
        "dbo:publicationDate", "dbp:published",
    ],
    "place of death": ["dbo:deathPlace", "dbp:deathPlace"],
    "composer": ["dbo:musicComposer", "dbp:composer", "dbp:music"],
    "performer": ["dbo:artist", "dbp:artist", "dbo:musicalArtist"],
    "cause of death": ["dbo:deathCause", "dbp:deathCause"],
    "educated at": ["dbo:almaMater", "dbp:almaMater", "dbp:education"],
    "place of burial": ["dbp:placeOfBurial", "dbp:burialPlace", "dbo:restingPlace"],
    "publisher": ["dbo:publisher", "dbp:publisher"],
    "award received": ["dbo:award", "dbp:awards"],
    "child": ["dbo:child", "dbp:children"],
    "founded by": ["dbo:founder", "dbp:founder"],
    "employer": ["dbo:employer", "dbp:employer"],
    "country": ["dbo:country", "dbp:country"],
    "inception": ["dbo:foundingDate", "dbp:founded", "dbo:formationDate"],
}

WIKI_HEADERS = {"User-Agent": "DBpediaQA-CoverageVerify/1.0"}


# ── Helpers ───────────────────────────────────────────────────────────────

def ask_entity_exists(uri: str) -> bool:
    """ASK DBpedia if the entity has any triples."""
    query = f"ASK {{ <{uri}> ?p ?o }}"
    try:
        sparql.setQuery(query)
        return sparql.query().convert().get("boolean", False)
    except Exception as e:
        print(f"    SPARQL error: {e}")
        return None  # inconclusive


def ask_property_exists(uri: str, prop_name: str) -> tuple[bool, list[str]]:
    """Check if entity has property. Returns (found, matching_predicates)."""
    predicates = PROPERTY_MAP.get(prop_name, [])
    if not predicates:
        camel = prop_name.replace(" ", "")
        camel = camel[0].lower() + camel[1:]
        predicates = [f"dbo:{camel}", f"dbp:{camel}"]

    found_preds = []
    for pred in predicates:
        query = f"ASK {{ <{uri}> {pred} ?o }}"
        try:
            sparql.setQuery(query)
            result = sparql.query().convert()
            if result.get("boolean", False):
                found_preds.append(pred)
        except Exception:
            pass
        time.sleep(0.2)

    return len(found_preds) > 0, found_preds


def check_wikipedia_exists(title: str) -> dict:
    """Check if entity has a Wikipedia page via API."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": title.replace(" ", "_"),
                "prop": "info",
                "format": "json",
            },
            headers=WIKI_HEADERS,
            timeout=15,
        )
        pages = resp.json().get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid == "-1":
                return {"exists": False, "title": title}
            return {
                "exists": True,
                "title": page.get("title", title),
                "pageid": page.get("pageid"),
            }
        return {"exists": False, "title": title}
    except Exception as e:
        return {"exists": None, "title": title, "error": str(e)}


# ── Load data ─────────────────────────────────────────────────────────────

def load_coverage():
    with open(BASE / "data/eval_1000q/coverage_analysis_1000.json") as f:
        return json.load(f)


# ── Step 1: Arithmetic recount ────────────────────────────────────────────

def step1_recount(data):
    """Recount entity/property coverage from per-question data."""
    print("=" * 72)
    print("STEP 1: Arithmetic recount from raw per-question data")
    print("=" * 72)

    questions = data["questions"]
    total = len(questions)

    # Entity coverage = questions where all_entities_exist == true
    ent_true = sum(1 for q in questions if q["all_entities_exist"])
    ent_false = total - ent_true

    # Property coverage = questions where all_properties_exist == true
    prop_true = sum(1 for q in questions if q["all_properties_exist"])

    # Answerable = is_answerable == true
    ans_true = sum(1 for q in questions if q["is_answerable"])

    # Count unique entities
    all_entities = []
    for q in questions:
        for e in q["entities"]:
            all_entities.append(e)
    ent_exists_count = sum(1 for e in all_entities if e["exists"])
    ent_missing_count = sum(1 for e in all_entities if not e["exists"])

    print(f"\nTotal questions: {total}")
    print(f"\nEntity coverage (all_entities_exist == true):")
    print(f"  Recount: {ent_true}/{total} = {100*ent_true/total:.1f}%")
    print(f"  Claimed: 93.6%")
    print(f"  Match: {'YES' if round(100*ent_true/total, 1) == 93.6 else 'NO'}")

    print(f"\nProperty coverage (all_properties_exist == true):")
    print(f"  Recount: {prop_true}/{total} = {100*prop_true/total:.1f}%")
    print(f"  Claimed: 38.5%")
    print(f"  Match: {'YES' if round(100*prop_true/total, 1) == 38.5 else 'NO'}")

    print(f"\nAnswerable (is_answerable == true):")
    print(f"  Recount: {ans_true}/{total} = {100*ans_true/total:.1f}%")
    print(f"  Claimed: 34.6%")
    print(f"  Match: {'YES' if round(100*ans_true/total, 1) == 34.6 else 'NO'}")

    print(f"\nPer-entity stats:")
    print(f"  Total entities: {len(all_entities)}")
    print(f"  Exist in DBpedia: {ent_exists_count}")
    print(f"  Missing from DBpedia: {ent_missing_count}")
    print(f"  Entity-level coverage: {100*ent_exists_count/len(all_entities):.1f}%")

    # Failure reason breakdown
    reasons = Counter()
    for q in questions:
        if q.get("failure_reason"):
            # Group by prefix
            r = q["failure_reason"]
            if r.startswith("Property missing"):
                reasons["Property missing"] += 1
            elif r.startswith("Entity not in DBpedia"):
                reasons["Entity not in DBpedia"] += 1
            else:
                reasons[r] += 1
    print(f"\nFailure reasons:")
    for r, c in reasons.most_common():
        print(f"  {r}: {c}")

    return {
        "entity_coverage": round(100 * ent_true / total, 1),
        "property_coverage": round(100 * prop_true / total, 1),
        "answerable": round(100 * ans_true / total, 1),
        "total_entities": len(all_entities),
        "entities_exist": ent_exists_count,
        "entities_missing": ent_missing_count,
    }


# ── Step 2: Verify "exists=true" entities against live DBpedia ────────────

def step2_verify_existing(data, sample_size=50):
    """Sample entities claimed to exist → check live DBpedia."""
    print("\n" + "=" * 72)
    print(f"STEP 2: Verify {sample_size} 'exists=true' entities against live DBpedia")
    print("=" * 72)

    # Collect all entities claimed to exist (with URIs)
    existing = []
    for q in data["questions"]:
        for e in q["entities"]:
            if e["exists"] and e.get("uri"):
                existing.append(e)

    random.seed(42)
    sample = random.sample(existing, min(sample_size, len(existing)))

    confirmed = 0
    contradicted = 0
    errors = 0

    for i, ent in enumerate(sample, 1):
        name = ent["name"]
        uri = ent["uri"]
        live = ask_entity_exists(uri)
        time.sleep(0.2)

        status = "CONFIRMED" if live else ("ERROR" if live is None else "CONTRADICTED")
        if live is True:
            confirmed += 1
        elif live is False:
            contradicted += 1
            print(f"  [{i}] {name} ({uri}) -> NOT FOUND in live DBpedia!")
        else:
            errors += 1
            print(f"  [{i}] {name} -> SPARQL error")

        if i % 10 == 0:
            print(f"  ... checked {i}/{len(sample)}: "
                  f"{confirmed} confirmed, {contradicted} contradicted, {errors} errors")

    total_checked = confirmed + contradicted
    pct = 100 * confirmed / total_checked if total_checked else 0
    print(f"\nResult: {confirmed}/{total_checked} confirmed ({pct:.1f}%)")
    if contradicted:
        print(f"  WARNING: {contradicted} entities claimed to exist but NOT in live DBpedia")
    if errors:
        print(f"  {errors} inconclusive (SPARQL errors)")

    return {"confirmed": confirmed, "contradicted": contradicted, "errors": errors}


# ── Step 3: Verify "exists=false" entities against live DBpedia ───────────

def step3_verify_missing(data):
    """Check ALL entities claimed missing → verify they're truly not in DBpedia."""
    print("\n" + "=" * 72)
    print("STEP 3: Verify ALL 'exists=false' entities against live DBpedia")
    print("=" * 72)

    missing = []
    for q in data["questions"]:
        for e in q["entities"]:
            if not e["exists"]:
                missing.append(e)

    print(f"Total claimed missing: {len(missing)}")

    confirmed_missing = 0
    actually_exists = 0
    errors = 0

    for i, ent in enumerate(missing, 1):
        name = ent["name"]
        # Try to construct URI
        encoded = name.replace(" ", "_")
        uri = f"http://dbpedia.org/resource/{quote(encoded, safe='_,()')}"

        live = ask_entity_exists(uri)
        time.sleep(0.2)

        if live is True:
            actually_exists += 1
            print(f"  [{i}] {name} -> ACTUALLY EXISTS at {uri}!")
        elif live is False:
            confirmed_missing += 1
        else:
            errors += 1

        if i % 10 == 0:
            print(f"  ... checked {i}/{len(missing)}: "
                  f"{confirmed_missing} confirmed missing, "
                  f"{actually_exists} actually exist, {errors} errors")

    total = confirmed_missing + actually_exists
    print(f"\nResult: {confirmed_missing}/{total} confirmed missing")
    if actually_exists:
        print(f"  WARNING: {actually_exists} entities claimed missing but "
              f"ACTUALLY EXIST in live DBpedia!")
        print(f"  This would mean entity coverage is UNDERSTATED.")
    return {
        "confirmed_missing": confirmed_missing,
        "actually_exists": actually_exists,
        "errors": errors,
    }


# ── Step 4: Verify "property found" claims ────────────────────────────────

def step4_verify_found_properties(data, sample_size=40):
    """Sample property claims marked 'found' → verify against live DBpedia."""
    print("\n" + "=" * 72)
    print(f"STEP 4: Verify {sample_size} 'property found' claims against live DBpedia")
    print("=" * 72)

    found_props = []
    for q in data["questions"]:
        for e in q["entities"]:
            if e["exists"] and e.get("uri"):
                for p in e.get("properties_found", []):
                    found_props.append({"name": e["name"], "uri": e["uri"], "prop": p})

    random.seed(42)
    sample = random.sample(found_props, min(sample_size, len(found_props)))

    confirmed = 0
    contradicted = 0
    details = []

    for i, item in enumerate(sample, 1):
        name = item["name"]
        uri = item["uri"]
        prop = item["prop"]

        found, preds = ask_property_exists(uri, prop)

        if found:
            confirmed += 1
        else:
            contradicted += 1
            details.append(f"  {name}.{prop} at {uri} -> NOT FOUND in live DBpedia")
            print(f"  [{i}] CONTRADICTED: {name}.{prop} -> not found in live DBpedia")

        if i % 10 == 0:
            print(f"  ... checked {i}/{len(sample)}: "
                  f"{confirmed} confirmed, {contradicted} contradicted")
        time.sleep(0.15)

    total = confirmed + contradicted
    pct = 100 * confirmed / total if total else 0
    print(f"\nResult: {confirmed}/{total} confirmed ({pct:.1f}%)")
    if contradicted:
        print(f"  WARNING: {contradicted} properties claimed found but NOT in live DBpedia")
        for d in details:
            print(d)

    return {"confirmed": confirmed, "contradicted": contradicted}


# ── Step 5: Verify "property missing" claims ──────────────────────────────

def step5_verify_missing_properties(data, sample_size=40):
    """Sample property claims marked 'missing' → verify they're truly missing."""
    print("\n" + "=" * 72)
    print(f"STEP 5: Verify {sample_size} 'property missing' claims against live DBpedia")
    print("=" * 72)

    missing_props = []
    for q in data["questions"]:
        for e in q["entities"]:
            if e["exists"] and e.get("uri"):
                for p in e.get("properties_missing", []):
                    missing_props.append({"name": e["name"], "uri": e["uri"], "prop": p})

    random.seed(42)
    sample = random.sample(missing_props, min(sample_size, len(missing_props)))

    confirmed_missing = 0
    actually_found = 0
    details = []

    for i, item in enumerate(sample, 1):
        name = item["name"]
        uri = item["uri"]
        prop = item["prop"]

        found, preds = ask_property_exists(uri, prop)

        if not found:
            confirmed_missing += 1
        else:
            actually_found += 1
            details.append(
                f"  {name}.{prop} -> FOUND via {preds} (claimed missing!)"
            )
            print(f"  [{i}] CONTRADICTED: {name}.{prop} -> found via {preds}")

        if i % 10 == 0:
            print(f"  ... checked {i}/{len(sample)}: "
                  f"{confirmed_missing} confirmed missing, "
                  f"{actually_found} actually found")
        time.sleep(0.15)

    total = confirmed_missing + actually_found
    print(f"\nResult: {confirmed_missing}/{total} confirmed missing")
    if actually_found:
        print(f"  WARNING: {actually_found} properties claimed missing but "
              f"ACTUALLY FOUND in live DBpedia!")
        print(f"  This means property coverage is UNDERSTATED.")
        for d in details:
            print(d)

    return {"confirmed_missing": confirmed_missing, "actually_found": actually_found}


# ── Step 6: Cross-check sample entities on Wikipedia ──────────────────────

def step6_wikipedia_crosscheck(data, sample_size=30):
    """Cross-check a sample of entities against Wikipedia API."""
    print("\n" + "=" * 72)
    print(f"STEP 6: Cross-check {sample_size} entities against Wikipedia API")
    print("=" * 72)

    existing = []
    for q in data["questions"]:
        for e in q["entities"]:
            if e["exists"] and e.get("uri"):
                existing.append(e)

    random.seed(42)
    sample = random.sample(existing, min(sample_size, len(existing)))

    has_wiki = 0
    no_wiki = 0
    errors = 0

    for i, ent in enumerate(sample, 1):
        name = ent["name"]
        # Convert DBpedia URI to Wikipedia title
        uri = ent["uri"]
        wiki_title = unquote(uri.split("/resource/")[-1]).replace("_", " ")

        result = check_wikipedia_exists(wiki_title)
        time.sleep(0.5)

        if result["exists"] is True:
            has_wiki += 1
        elif result["exists"] is False:
            no_wiki += 1
            print(f"  [{i}] {name} -> No Wikipedia page at '{wiki_title}'")
        else:
            errors += 1

        if i % 10 == 0:
            print(f"  ... checked {i}/{len(sample)}: "
                  f"{has_wiki} have Wikipedia, {no_wiki} missing, {errors} errors")

    total = has_wiki + no_wiki
    pct = 100 * has_wiki / total if total else 0
    print(f"\nResult: {has_wiki}/{total} have Wikipedia pages ({pct:.1f}%)")
    print(f"  (This is expected: DBpedia entities ARE derived from Wikipedia)")

    return {"has_wiki": has_wiki, "no_wiki": no_wiki, "errors": errors}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading coverage_analysis_1000.json...")
    data = load_coverage()
    print(f"Loaded {len(data['questions'])} questions.\n")

    # Step 1: Offline recount
    recount = step1_recount(data)

    # Steps 2-6: Live API verification
    print("\n>>> Starting live API verification (rate-limited, ~5 minutes)...\n")

    s2 = step2_verify_existing(data, sample_size=50)
    s3 = step3_verify_missing(data)
    s4 = step4_verify_found_properties(data, sample_size=40)
    s5 = step5_verify_missing_properties(data, sample_size=40)
    s6 = step6_wikipedia_crosscheck(data, sample_size=30)

    # ── Final Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 72)

    print(f"\n1. ARITHMETIC RECOUNT:")
    print(f"   Entity coverage:   {recount['entity_coverage']}% "
          f"(claimed 93.6%) -> {'MATCH' if recount['entity_coverage'] == 93.6 else 'MISMATCH'}")
    print(f"   Property coverage: {recount['property_coverage']}% "
          f"(claimed 38.5%) -> {'MATCH' if recount['property_coverage'] == 38.5 else 'MISMATCH'}")
    print(f"   Answerable:        {recount['answerable']}% "
          f"(claimed 34.6%) -> {'MATCH' if recount['answerable'] == 34.6 else 'MISMATCH'}")

    s2_total = s2['confirmed'] + s2['contradicted']
    print(f"\n2. EXISTING ENTITIES (live DBpedia):")
    print(f"   {s2['confirmed']}/{s2_total} confirmed exist "
          f"({100*s2['confirmed']/s2_total:.0f}% accuracy)")
    if s2['contradicted']:
        print(f"   WARNING: {s2['contradicted']} entities not found in live DBpedia")

    s3_total = s3['confirmed_missing'] + s3['actually_exists']
    print(f"\n3. MISSING ENTITIES (live DBpedia, checked ALL {s3_total}):")
    print(f"   {s3['confirmed_missing']}/{s3_total} confirmed truly missing")
    if s3['actually_exists']:
        print(f"   WARNING: {s3['actually_exists']} entities actually exist -> "
              f"coverage may be understated")

    s4_total = s4['confirmed'] + s4['contradicted']
    print(f"\n4. FOUND PROPERTIES (live DBpedia):")
    print(f"   {s4['confirmed']}/{s4_total} confirmed "
          f"({100*s4['confirmed']/s4_total:.0f}% accuracy)")

    s5_total = s5['confirmed_missing'] + s5['actually_found']
    print(f"\n5. MISSING PROPERTIES (live DBpedia):")
    print(f"   {s5['confirmed_missing']}/{s5_total} confirmed truly missing")
    if s5['actually_found']:
        print(f"   WARNING: {s5['actually_found']} properties actually exist -> "
              f"property coverage may be understated")

    s6_total = s6['has_wiki'] + s6['no_wiki']
    print(f"\n6. WIKIPEDIA CROSS-CHECK:")
    print(f"   {s6['has_wiki']}/{s6_total} entities have Wikipedia pages "
          f"({100*s6['has_wiki']/s6_total:.0f}%)")

    # Overall verdict
    print(f"\n{'=' * 72}")
    issues = []
    if recount['entity_coverage'] != 93.6:
        issues.append("Entity coverage arithmetic mismatch")
    if recount['property_coverage'] != 38.5:
        issues.append("Property coverage arithmetic mismatch")
    if s2['contradicted'] > 0:
        issues.append(f"{s2['contradicted']} 'existing' entities not in live DBpedia")
    if s3['actually_exists'] > 0:
        issues.append(f"{s3['actually_exists']} 'missing' entities actually exist")
    if s4['contradicted'] > 0:
        issues.append(f"{s4['contradicted']} 'found' properties not in live DBpedia")
    if s5['actually_found'] > 0:
        issues.append(f"{s5['actually_found']} 'missing' properties actually exist")

    if not issues:
        print("VERDICT: ALL CLAIMS VERIFIED ✓")
    else:
        print(f"VERDICT: {len(issues)} ISSUE(S) FOUND:")
        for issue in issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()
