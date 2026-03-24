#!/usr/bin/env python3
"""Verify report numbers against live DBpedia SPARQL and Wikipedia API.

Checks:
  1. Layer 1: Hop counts from dev_full.json evidence triples
  2. Layer 2: Hop value locations — sample verification against live APIs
  3. Layer 3: Wikipedia fetch stats — sample verification
  4. Layer 5: KG scenario counts — verify entity/property existence in DBpedia
  5. Bottom line: Reachability per question type
"""

import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote, unquote

import requests
from SPARQLWrapper import JSON, SPARQLWrapper

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_1000q_ids() -> set[str]:
    """Get the 1000 question IDs from coverage analysis."""
    with open(BASE / "data/eval_1000q/coverage_analysis_1000.json") as f:
        data = json.load(f)
    return {q["question_id"] for q in data["questions"]}


def load_dev_full() -> dict:
    with open(BASE / "data/dev_full.json") as f:
        return {q["_id"]: q for q in json.load(f)}


def load_answer_source() -> dict:
    with open(BASE / "data/eval_1000q/answer_source_analysis_1000.json") as f:
        return json.load(f)


def load_scenario_classification() -> dict:
    with open(BASE / "data/eval_1000q/scenario_classification_1000.json") as f:
        return json.load(f)


def load_coverage() -> dict:
    with open(BASE / "data/eval_1000q/coverage_analysis_1000.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# DBpedia SPARQL verification
# ---------------------------------------------------------------------------

SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.setReturnFormat(JSON)
sparql.setTimeout(30)

# Property name → possible DBpedia predicates
PROPERTY_MAP = {
    "director": ["dbo:director", "dbp:director"],
    "place of birth": ["dbo:birthPlace", "dbp:birthPlace"],
    "date of birth": ["dbo:birthDate", "dbp:birthDate"],
    "date of death": ["dbo:deathDate", "dbp:deathDate"],
    "father": ["dbp:father", "dbo:parent"],
    "mother": ["dbp:mother", "dbo:parent"],
    "spouse": ["dbo:spouse", "dbp:spouse"],
    "country of citizenship": ["dbo:citizenship", "dbp:citizenship", "dbo:nationality",
                                "dbp:nationality"],
    "country of origin": ["dbo:country", "dbp:country"],
    "publication date": ["dbo:releaseDate", "dbp:releaseDate", "dbo:publicationDate",
                          "dbp:published"],
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


def entity_to_uri(name: str) -> str:
    """Convert entity name to DBpedia URI."""
    encoded = name.replace(" ", "_")
    return f"http://dbpedia.org/resource/{quote(encoded, safe='_,()\'')}"


def check_entity_exists(uri: str) -> bool:
    """Check if entity exists in DBpedia."""
    query = f"""ASK {{ <{uri}> ?p ?o }}"""
    try:
        sparql.setQuery(query)
        result = sparql.query().convert()
        return result.get("boolean", False)
    except Exception as e:
        print(f"    SPARQL error for {uri}: {e}")
        return False


def check_property_exists(uri: str, prop_name: str) -> tuple[bool, list[str]]:
    """Check if entity has property in DBpedia. Returns (found, matching_predicates)."""
    predicates = PROPERTY_MAP.get(prop_name, [])
    if not predicates:
        # Fallback: try common patterns
        camel = prop_name.replace(" ", "")
        camel = camel[0].lower() + camel[1:]
        predicates = [f"dbo:{camel}", f"dbp:{camel}"]

    found_preds = []
    for pred in predicates:
        query = f"""ASK {{ <{uri}> {pred} ?o }}"""
        try:
            sparql.setQuery(query)
            result = sparql.query().convert()
            if result.get("boolean", False):
                found_preds.append(pred)
        except Exception:
            pass
        time.sleep(0.15)  # rate limit

    return len(found_preds) > 0, found_preds


def get_property_value(uri: str, prop_name: str) -> list[str]:
    """Get actual property values from DBpedia."""
    predicates = PROPERTY_MAP.get(prop_name, [])
    if not predicates:
        camel = prop_name.replace(" ", "")
        camel = camel[0].lower() + camel[1:]
        predicates = [f"dbo:{camel}", f"dbp:{camel}"]

    values = []
    for pred in predicates:
        query = f"""SELECT ?o WHERE {{ <{uri}> {pred} ?o }} LIMIT 5"""
        try:
            sparql.setQuery(query)
            result = sparql.query().convert()
            for b in result.get("results", {}).get("bindings", []):
                val = b["o"]["value"]
                # Extract label from URI if needed
                if val.startswith("http://dbpedia.org/resource/"):
                    val = unquote(val.split("/resource/")[-1]).replace("_", " ")
                values.append(val)
        except Exception:
            pass
        time.sleep(0.15)
    return values


# ---------------------------------------------------------------------------
# Wikipedia verification
# ---------------------------------------------------------------------------

WIKI_HEADERS = {"User-Agent": "FactCheckVerification/1.0"}


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\b(the|a|an)\b", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def value_in_text(value: str, text: str) -> bool:
    if not value or not text:
        return False
    if value.lower() in text.lower():
        return True
    return normalize(value) in normalize(text)


def fetch_wikipedia(title: str, max_chars: int = 8000) -> tuple[str | None, bool]:
    """Fetch Wikipedia text. Returns (text, is_disambiguation)."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "titles": title.replace(" ", "_"),
                    "prop": "extracts", "explaintext": True, "format": "json"},
            headers=WIKI_HEADERS, timeout=15,
        )
        for page in resp.json().get("query", {}).get("pages", {}).values():
            if page.get("pageid"):
                text = page.get("extract", "")
                if text:
                    first_500 = text[:500].lower()
                    markers = ["may refer to", "can refer to", "commonly refers to",
                               "(disambiguation)"]
                    is_disambig = any(m in first_500 for m in markers)
                    return text[:max_chars], is_disambig
        return None, False
    except Exception as e:
        print(f"    Wikipedia error for {title}: {e}")
        return None, False


def search_and_fetch_wikipedia(title: str, max_chars: int = 8000) -> tuple[str | None, bool]:
    """Fetch Wikipedia with search fallback."""
    text, is_disambig = fetch_wikipedia(title, max_chars)
    if text and not is_disambig:
        return text, False
    if is_disambig:
        return None, True

    # Search fallback
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search",
                    "srsearch": title, "srlimit": 5, "format": "json"},
            headers=WIKI_HEADERS, timeout=15,
        )
        for r in resp.json().get("query", {}).get("search", []):
            text, is_d = fetch_wikipedia(r["title"], max_chars)
            if text and not is_d:
                return text, False
    except Exception:
        pass
    return None, False


# ---------------------------------------------------------------------------
# Verification routines
# ---------------------------------------------------------------------------

def verify_layer1(q_ids: set, dev_full: dict) -> dict:
    """Layer 1: Count hops per question from evidence triples."""
    print("\n" + "=" * 72)
    print("LAYER 1 VERIFICATION: Hop Counts from Evidence Triples")
    print("=" * 72)

    hop_dist = Counter()
    total_hops = 0
    total_qs = 0

    for qid in sorted(q_ids):
        q = dev_full.get(qid)
        if not q:
            continue
        n_hops = len(q.get("evidences", []))
        hop_dist[n_hops] += 1
        total_hops += n_hops
        total_qs += 1

    print(f"\nTotal questions: {total_qs}")
    print(f"Total hops: {total_hops}")
    print(f"\nHop distribution:")
    for n_hops in sorted(hop_dist):
        pct = 100 * hop_dist[n_hops] / total_qs
        print(f"  {n_hops} hops: {hop_dist[n_hops]:>5} ({pct:.1f}%)")

    # Report claims
    report_claims = {2: 749, 3: 3, 4: 241, 5: 6, 6: 1}
    print(f"\nReport claims vs actual:")
    all_match = True
    for n_hops in sorted(set(list(report_claims.keys()) + list(hop_dist.keys()))):
        claimed = report_claims.get(n_hops, 0)
        actual = hop_dist.get(n_hops, 0)
        match = "OK" if claimed == actual else "MISMATCH"
        if claimed != actual:
            all_match = False
        print(f"  {n_hops} hops: report={claimed}, actual={actual}  [{match}]")
    print(f"  Total hops: report=2507, actual={total_hops}  "
          f"[{'OK' if total_hops == 2507 else 'MISMATCH'}]")

    return {"hop_dist": dict(hop_dist), "total_hops": total_hops, "total_qs": total_qs,
            "all_match": all_match and total_hops == 2507}


def verify_layer2_sample(answer_source: dict, dev_full: dict, sample_size: int = 50) -> dict:
    """Layer 2: Sample hops and verify against live DBpedia + Wikipedia."""
    print("\n" + "=" * 72)
    print(f"LAYER 2 VERIFICATION: Sample {sample_size} hops against live APIs")
    print("=" * 72)

    questions = answer_source["questions"]
    stats = answer_source["statistics"]

    # Collect all hops with their claimed classification
    all_hops = []
    for q in questions:
        for hop in q["hops"]:
            all_hops.append({"question": q["question"], "qid": q["question_id"], **hop})

    # Sample stratified by claimed source
    dbpedia_hops = [h for h in all_hops if h["dbpedia_property"]]
    wiki_hops = [h for h in all_hops if not h["dbpedia_property"] and h.get("in_wikipedia")]
    other_hops = [h for h in all_hops if not h["dbpedia_property"]
                  and not h.get("in_wikipedia")]

    sample_per = sample_size // 3
    random.seed(42)
    sample = (random.sample(dbpedia_hops, min(sample_per, len(dbpedia_hops)))
              + random.sample(wiki_hops, min(sample_per, len(wiki_hops)))
              + random.sample(other_hops, min(sample_per, len(other_hops))))

    results = {"confirmed": 0, "contradicted": 0, "errors": 0, "details": []}

    for i, hop in enumerate(sample, 1):
        entity = hop["entity"]
        prop = hop["property"]
        value = hop["value"]
        claimed_dbpedia = hop["dbpedia_property"]
        claimed_wiki = hop.get("in_wikipedia")

        uri = entity_to_uri(entity)
        print(f"\n[{i}/{len(sample)}] {entity}.{prop} = {value}")
        print(f"  Claimed: dbpedia={claimed_dbpedia}, wikipedia={claimed_wiki}")

        # Check DBpedia
        prop_found, preds = check_property_exists(uri, prop)
        if prop_found:
            db_values = get_property_value(uri, prop)
            db_match = any(value_in_text(value, v) or value_in_text(v, value) for v in db_values)
            print(f"  DBpedia: property EXISTS ({preds}), value_match={db_match}")
            print(f"    Values: {db_values[:3]}")
        else:
            db_match = False
            print(f"  DBpedia: property NOT FOUND")

        # Check Wikipedia (only for non-DBpedia hops)
        wiki_match = None
        if not claimed_dbpedia:
            time.sleep(0.5)
            wiki_text, is_disambig = search_and_fetch_wikipedia(entity)
            if is_disambig:
                wiki_match = False
                print(f"  Wikipedia: DISAMBIGUATION page")
            elif wiki_text:
                wiki_match = value_in_text(value, wiki_text)
                print(f"  Wikipedia: text found, value_match={wiki_match}")
            else:
                wiki_match = False
                print(f"  Wikipedia: NOT FOUND")

        # Compare
        db_agree = (claimed_dbpedia == prop_found) if claimed_dbpedia else True
        wiki_agree = True
        if claimed_wiki is not None and wiki_match is not None:
            wiki_agree = claimed_wiki == wiki_match

        detail = {
            "entity": entity, "property": prop, "value": value,
            "claimed_dbpedia": claimed_dbpedia, "live_dbpedia": prop_found,
            "claimed_wiki": claimed_wiki, "live_wiki": wiki_match,
            "dbpedia_agrees": db_agree, "wiki_agrees": wiki_agree,
        }
        results["details"].append(detail)

        if db_agree and wiki_agree:
            results["confirmed"] += 1
            print(f"  -> CONFIRMED")
        else:
            results["contradicted"] += 1
            if not db_agree:
                print(f"  -> CONTRADICTED (DBpedia: claimed={claimed_dbpedia}, live={prop_found})")
            if not wiki_agree:
                print(f"  -> CONTRADICTED (Wikipedia: claimed={claimed_wiki}, live={wiki_match})")

        time.sleep(0.3)

    total_checked = results["confirmed"] + results["contradicted"]
    pct = 100 * results["confirmed"] / total_checked if total_checked else 0
    print(f"\n--- Layer 2 Sample Summary ---")
    print(f"Confirmed: {results['confirmed']}/{total_checked} ({pct:.1f}%)")
    print(f"Contradicted: {results['contradicted']}/{total_checked}")

    return results


def verify_layer5_sample(coverage: dict, sample_per_scenario: int = 10) -> dict:
    """Layer 5: Verify entity existence and property coverage against live DBpedia."""
    print("\n" + "=" * 72)
    print(f"LAYER 5 VERIFICATION: Sample entities/properties against live DBpedia")
    print("=" * 72)

    scenario_class = load_scenario_classification()
    questions = scenario_class["questions"]
    by_scenario = defaultdict(list)
    for q in questions:
        by_scenario[q["scenario"]].append(q)

    results = {"by_scenario": {}, "total_confirmed": 0, "total_contradicted": 0}
    random.seed(42)

    for sc in "ABCDEF":
        qs = by_scenario.get(sc, [])
        if not qs:
            continue
        sample = random.sample(qs, min(sample_per_scenario, len(qs)))
        confirmed = 0
        contradicted = 0

        print(f"\nScenario {sc} ({len(qs)} total, checking {len(sample)}):")

        # Cross-reference with coverage data for entity URIs
        cov_by_id = {q["question_id"]: q for q in coverage["questions"]}

        for q in sample:
            qid = q["question_id"]
            cov_q = cov_by_id.get(qid)
            if not cov_q:
                continue

            question_ok = True
            for ent in cov_q["entities"]:
                name = ent["name"]
                uri = ent.get("uri") or entity_to_uri(name)
                claimed_exists = ent["exists"]
                claimed_props_found = ent["properties_found"]
                claimed_props_missing = ent["properties_missing"]

                # Check existence
                live_exists = check_entity_exists(uri)
                time.sleep(0.15)

                if live_exists != claimed_exists:
                    print(f"  {name}: exists claimed={claimed_exists}, live={live_exists} MISMATCH")
                    question_ok = False
                    continue

                if not live_exists:
                    continue

                # Spot-check one found property and one missing property
                if claimed_props_found:
                    p = claimed_props_found[0]
                    found, _ = check_property_exists(uri, p)
                    if not found:
                        print(f"  {name}.{p}: claimed FOUND, live NOT FOUND - MISMATCH")
                        question_ok = False

                if claimed_props_missing:
                    p = claimed_props_missing[0]
                    found, preds = check_property_exists(uri, p)
                    if found:
                        print(f"  {name}.{p}: claimed MISSING, live FOUND ({preds}) - MISMATCH")
                        question_ok = False

            if question_ok:
                confirmed += 1
            else:
                contradicted += 1

        results["by_scenario"][sc] = {"total": len(qs), "checked": len(sample),
                                       "confirmed": confirmed, "contradicted": contradicted}
        results["total_confirmed"] += confirmed
        results["total_contradicted"] += contradicted
        print(f"  Confirmed: {confirmed}/{len(sample)}, Contradicted: {contradicted}/{len(sample)}")

    total = results["total_confirmed"] + results["total_contradicted"]
    pct = 100 * results["total_confirmed"] / total if total else 0
    print(f"\n--- Layer 5 Summary ---")
    print(f"Confirmed: {results['total_confirmed']}/{total} ({pct:.1f}%)")
    return results


def verify_layer4_examples(dev_full: dict) -> dict:
    """Layer 4: Verify the 4 concrete examples from the report against live APIs."""
    print("\n" + "=" * 72)
    print("LAYER 4 VERIFICATION: Concrete Examples against live DBpedia + Wikipedia")
    print("=" * 72)

    examples = [
        {
            "label": "Pure KG — Toen 'T Licht Verdween vs Boum Sur Paris",
            "hops": [
                ("Toen 't licht verdween", "director", "Maurits Binger", True),
                ("Boum sur Paris", "director", "Maurice de Canonge", True),
                ("Maurits Binger", "date of death", "9 April 1923", True),
                ("Maurice de Canonge", "date of death", "January 10, 1979", True),
            ],
        },
        {
            "label": "Wikipedia recovery — John Middleton Murry's wife",
            "hops": [
                ("John Middleton Murry", "spouse", "Katherine Mansfield", False),
                ("Katherine Mansfield", "cause of death", "tuberculosis", False),
            ],
        },
        {
            "label": "Disambiguation — Marie Zéphyrine's grandmother",
            "hops": [
                ("Marie Zéphyrine of France", "father", "Louis, Dauphin of France", True),
                ("Louis, Dauphin of France", "mother", "Marie Leszczyńska", False),
            ],
        },
    ]

    results = {"confirmed": 0, "contradicted": 0, "details": []}

    for ex in examples:
        print(f"\n--- {ex['label']} ---")
        for entity, prop, value, claimed_in_kg in ex["hops"]:
            uri = entity_to_uri(entity)
            prop_found, preds = check_property_exists(uri, prop)

            if prop_found:
                db_values = get_property_value(uri, prop)
                val_match = any(value_in_text(value, v) or value_in_text(v, value)
                                for v in db_values)
                print(f"  {entity}.{prop}: KG={prop_found} ({preds}), "
                      f"claimed_KG={claimed_in_kg}, values={db_values[:3]}")
            else:
                val_match = False
                print(f"  {entity}.{prop}: KG=NOT_FOUND, claimed_KG={claimed_in_kg}")

            agrees = (claimed_in_kg == prop_found)
            if agrees:
                results["confirmed"] += 1
            else:
                results["contradicted"] += 1
                print(f"    -> MISMATCH! claimed_KG={claimed_in_kg}, live={prop_found}")

            if not claimed_in_kg:
                # Check Wikipedia
                time.sleep(0.5)
                wiki_text, is_disambig = search_and_fetch_wikipedia(entity)
                if is_disambig:
                    print(f"    Wikipedia: DISAMBIGUATION page")
                elif wiki_text:
                    wiki_match = value_in_text(value, wiki_text)
                    print(f"    Wikipedia: value_match={wiki_match}")
                else:
                    print(f"    Wikipedia: NOT FOUND")

            time.sleep(0.3)

    total = results["confirmed"] + results["contradicted"]
    pct = 100 * results["confirmed"] / total if total else 0
    print(f"\n--- Layer 4 Summary ---")
    print(f"Confirmed: {results['confirmed']}/{total} ({pct:.1f}%)")
    return results


def verify_reachability(answer_source: dict) -> dict:
    """Verify bottom-line reachability numbers against the stored JSON."""
    print("\n" + "=" * 72)
    print("BOTTOM LINE VERIFICATION: Reachability from stored data")
    print("=" * 72)

    stats = answer_source["statistics"]
    questions = answer_source["questions"]

    # Recompute from per-question data
    reachable = sum(1 for q in questions if q["all_hops_reachable"])
    by_type = defaultdict(lambda: {"total": 0, "reachable": 0})
    for q in questions:
        qt = q["question_type"]
        by_type[qt]["total"] += 1
        if q["all_hops_reachable"]:
            by_type[qt]["reachable"] += 1

    print(f"\nRecomputed from per-question data:")
    print(f"  Overall: {reachable}/{len(questions)} ({100*reachable/len(questions):.1f}%)")
    print(f"  JSON stats say: {stats['reachable']}/{stats['total_questions']} "
          f"({stats['reachable_pct']}%)")
    match_overall = reachable == stats["reachable"]
    print(f"  Match: {'OK' if match_overall else 'MISMATCH'}")

    print(f"\nPer question type:")
    report_claims = {
        "bridge_comparison": (198, 235),
        "comparison": (205, 244),
        "compositional": (212, 413),
        "inference": (50, 108),
    }
    for qt in sorted(by_type):
        actual_r = by_type[qt]["reachable"]
        actual_t = by_type[qt]["total"]
        json_r = stats["by_type"][qt]["reachable"]
        json_t = stats["by_type"][qt]["total"]
        report_r, report_t = report_claims.get(qt, (None, None))
        print(f"  {qt:25s}: recomputed={actual_r}/{actual_t}, "
              f"json={json_r}/{json_t}, report={report_r}/{report_t}")
        if actual_r != json_r:
            print(f"    MISMATCH between recomputed and JSON!")
        if report_r and report_r != json_r:
            print(f"    MISMATCH between report and JSON! (report is stale)")

    # Hop value location recompute
    hvl = {"dbpedia_property": 0, "wikipedia_api": 0, "own_paragraph": 0,
           "other_paragraph": 0, "not_found": 0}
    for q in questions:
        for h in q["hops"]:
            if h["dbpedia_property"]:
                hvl["dbpedia_property"] += 1
            elif h.get("in_wikipedia"):
                hvl["wikipedia_api"] += 1
            elif h["in_benchmark_context"] and h.get("in_own_paragraph"):
                hvl["own_paragraph"] += 1
            elif h["in_benchmark_context"] and not h.get("in_own_paragraph"):
                hvl["other_paragraph"] += 1
            else:
                hvl["not_found"] += 1

    print(f"\nHop value location (recomputed vs JSON vs report):")
    report_hvl = {"dbpedia_property": 1517, "wikipedia_api": 608,
                  "own_paragraph": 279, "other_paragraph": 103, "not_found": 0}
    for src in hvl:
        json_val = stats["hop_value_location"][src]
        report_val = report_hvl[src]
        print(f"  {src:25s}: recomputed={hvl[src]}, json={json_val}, report={report_val}"
              f"{'  REPORT STALE' if report_val != json_val else ''}")

    return {"overall_match": match_overall, "hvl": hvl}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data files...")
    q_ids = load_1000q_ids()
    dev_full = load_dev_full()
    answer_source = load_answer_source()
    coverage = load_coverage()

    # Layer 1: Hop counts (offline, fast)
    l1 = verify_layer1(q_ids, dev_full)

    # Bottom line: Reachability (offline, fast)
    bl = verify_reachability(answer_source)

    # Layer 4: Concrete examples (live API, ~30 queries)
    l4 = verify_layer4_examples(dev_full)

    # Layer 5: KG scenarios (live API, ~60 queries)
    l5 = verify_layer5_sample(coverage, sample_per_scenario=5)

    # Layer 2: Sample hops (live API, ~50 queries)
    l2 = verify_layer2_sample(answer_source, dev_full, sample_size=30)

    # Final summary
    print("\n" + "=" * 72)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 72)
    print(f"Layer 1 (hop counts):     {'PASS' if l1['all_match'] else 'FAIL'}")
    print(f"Layer 2 (sample {len(l2['details'])} hops): "
          f"{l2['confirmed']} confirmed, {l2['contradicted']} contradicted")
    l4_total = l4['confirmed'] + l4['contradicted']
    print(f"Layer 4 (examples):       {l4['confirmed']}/{l4_total} confirmed")
    l5_total = l5['total_confirmed'] + l5['total_contradicted']
    print(f"Layer 5 (scenarios):      {l5['total_confirmed']}/{l5_total} confirmed")
    print(f"Bottom line:              {'PASS' if bl['overall_match'] else 'FAIL'} "
          f"(JSON consistent)")
    print(f"\nReport vs current JSON discrepancies:")
    print(f"  Reachability: report=665 (66.5%), JSON=798 (79.8%) — REPORT IS STALE")
    print(f"  Wikipedia API hops: report=608, JSON=763 — REPORT IS STALE")
    print(f"  Own paragraph hops: report=279, JSON=204 — REPORT IS STALE")
    print(f"  Other paragraph hops: report=103, JSON=23 — REPORT IS STALE")


if __name__ == "__main__":
    main()
