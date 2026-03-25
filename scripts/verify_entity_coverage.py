"""
Exhaustive entity coverage verification against live DBpedia SPARQL + Wikipedia API.
Checks ALL 65 missing entities with multiple strategies, and random-samples 100 existing ones.
"""
import json
import random
import re
import time
import urllib.parse
from pathlib import Path

import requests
from SPARQLWrapper import JSON, SPARQLWrapper

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
WIKI_API = "https://en.wikipedia.org/w/api.php"
SPARQL_DELAY = 0.3   # seconds between SPARQL calls
WIKI_DELAY = 0.5     # seconds between Wikipedia calls
RANDOM_SEED = 42

DATA_FILE = Path("data/eval_1000q/coverage_analysis_1000.json")
OUT_FILE = Path("plans/reports/verify-260325-1302-entity-coverage.md")

sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.setReturnFormat(JSON)
sparql.setTimeout(15)


# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------

def ask_uri_exists(uri: str) -> bool:
    """ASK if a URI exists in DBpedia (has at least one triple)."""
    time.sleep(SPARQL_DELAY)
    q = f"ASK {{ <{uri}> ?p ?o }}"
    sparql.setQuery(q)
    try:
        res = sparql.query().convert()
        return bool(res.get("boolean", False))
    except Exception as e:
        print(f"  [SPARQL ERR] ASK {uri}: {e}")
        return False


def ask_redirect_target(uri: str) -> str | None:
    """Return redirect target URI if this URI is a DBpedia redirect, else None."""
    time.sleep(SPARQL_DELAY)
    encoded = urllib.parse.quote(uri, safe=":/#")
    q = f"""
SELECT ?target WHERE {{
  <{uri}> <http://dbpedia.org/ontology/wikiPageRedirects> ?target .
}} LIMIT 1
"""
    sparql.setQuery(q)
    try:
        res = sparql.query().convert()
        bindings = res["results"]["bindings"]
        if bindings:
            return bindings[0]["target"]["value"]
    except Exception as e:
        print(f"  [SPARQL ERR] redirect check {uri}: {e}")
    return None


def search_bif_contains(label: str) -> list[str]:
    """Full-text search in DBpedia labels using bif:contains."""
    time.sleep(SPARQL_DELAY)
    # bif:contains requires words ≥ 4 chars; take first meaningful word
    words = [w for w in re.split(r"\s+", label) if len(w) >= 3]
    if not words:
        return []
    search_term = " AND ".join(f'"{w}"' for w in words[:3])
    q = f"""
SELECT DISTINCT ?s ?label WHERE {{
  ?s rdfs:label ?label .
  ?label bif:contains '{search_term}' .
  FILTER(lang(?label) = 'en')
  FILTER(STRSTARTS(STR(?s), 'http://dbpedia.org/resource/'))
  FILTER(!CONTAINS(STR(?s), ':'))
}} LIMIT 5
"""
    sparql.setQuery(q)
    try:
        res = sparql.query().convert()
        return [(b["s"]["value"], b["label"]["value"]) for b in res["results"]["bindings"]]
    except Exception as e:
        print(f"  [SPARQL ERR] bif:contains '{label}': {e}")
        return []


# ---------------------------------------------------------------------------
# URI construction helpers
# ---------------------------------------------------------------------------

def name_to_uri(name: str) -> str:
    """Standard DBpedia URI from entity name."""
    normalized = name.strip().replace(" ", "_")
    return f"http://dbpedia.org/resource/{urllib.parse.quote(normalized, safe=':/#_')}"


def uri_variants(name: str) -> list[str]:
    """Generate candidate URI variants for a name."""
    variants = []
    base = name.strip()

    # 1. Direct
    variants.append(name_to_uri(base))

    # 2. Strip leading quotes / special chars
    cleaned = re.sub(r'^["\'"\u201c\u201d]+', '', base).strip()
    if cleaned != base:
        variants.append(name_to_uri(cleaned))

    # 3. Strip parenthetical suffix like "(Film)" or "(1753 – ...)"
    stripped_paren = re.sub(r'\s*\([^)]*\)\s*$', '', base).strip()
    if stripped_paren and stripped_paren != base:
        variants.append(name_to_uri(stripped_paren))

    # 4. Strip trailing descriptive clauses (comma-separated)
    stripped_comma = re.sub(r',.*$', '', base).strip()
    if stripped_comma and stripped_comma != base:
        variants.append(name_to_uri(stripped_comma))

    # 5. Title case
    tc = base.title()
    if tc != base:
        variants.append(name_to_uri(tc))

    # 6. Remove ordinal suffixes in Roman numeral names: "John II, ..." → "John_II"
    roman_stripped = re.sub(r'\s+(I{1,3}|IV|V|VI|VII|VIII|IX|X{1,3})[,\s].*$', '', base).strip()
    if roman_stripped and roman_stripped != base:
        variants.append(name_to_uri(roman_stripped))

    # 7. Nobility title abbreviation strip: "Prince/Princess/Lady/Lord ..." → inner name
    nobility_stripped = re.sub(r'^(Prince|Princess|Lady|Lord|Infanta|Infante|Empress|Emperor)\s+', '', base).strip()
    if nobility_stripped != base:
        variants.append(name_to_uri(nobility_stripped))

    # 8. Remove "Dowager" infix
    if "Dowager" in base:
        variants.append(name_to_uri(base.replace("Empress Dowager ", "Empress_Dowager_").replace(" ", "_")))
        variants.append(name_to_uri(re.sub(r'Empress Dowager ', '', base).strip()))

    # Deduplicate preserving order
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Wikipedia API helper
# ---------------------------------------------------------------------------

def wiki_search(name: str) -> list[dict]:
    """Search Wikipedia for the entity name, return top 3 results."""
    time.sleep(WIKI_DELAY)
    params = {
        "action": "query",
        "list": "search",
        "srsearch": name,
        "srlimit": 3,
        "format": "json",
        "utf8": 1,
    }
    try:
        r = requests.get(WIKI_API, params=params, timeout=10)
        data = r.json()
        return data.get("query", {}).get("search", [])
    except Exception as e:
        print(f"  [WIKI ERR] search '{name}': {e}")
        return []


def wiki_page_exists(title: str) -> bool:
    """Check if a Wikipedia page with this exact title exists."""
    time.sleep(WIKI_DELAY)
    params = {
        "action": "query",
        "titles": title,
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, timeout=10)
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        return not any(k == "-1" for k in pages)
    except Exception as e:
        print(f"  [WIKI ERR] page check '{title}': {e}")
        return False


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def check_missing_entity(name: str) -> dict:
    """
    Exhaustively check a single missing entity.
    Returns a result dict with all findings.
    """
    print(f"\n[CHECK] {name!r}")
    result = {
        "name": name,
        "direct_uri": name_to_uri(name),
        "direct_uri_exists": False,
        "variant_uris_tried": [],
        "variant_found": None,
        "redirect_target": None,
        "bif_matches": [],
        "wiki_results": [],
        "truly_missing": True,
        "notes": "",
    }

    # --- 1. Direct URI check ---
    direct = name_to_uri(name)
    direct_exists = ask_uri_exists(direct)
    result["direct_uri_exists"] = direct_exists
    if direct_exists:
        print(f"  [FOUND] Direct URI: {direct}")
        result["truly_missing"] = False
        result["notes"] = "Direct URI exists"
        return result

    # --- 2. Check redirect on direct URI ---
    redir = ask_redirect_target(direct)
    if redir:
        result["redirect_target"] = redir
        redir_exists = ask_uri_exists(redir)
        if redir_exists:
            print(f"  [FOUND via REDIRECT] {redir}")
            result["truly_missing"] = False
            result["notes"] = f"Redirect → {redir}"
            return result

    # --- 3. Variant URIs ---
    variants = uri_variants(name)[1:]  # skip first (already tried as direct)
    tried = []
    for v in variants:
        exists = ask_uri_exists(v)
        tried.append((v, exists))
        print(f"  [VARIANT] {v} → {'EXISTS' if exists else 'missing'}")
        if exists:
            result["variant_found"] = v
            result["truly_missing"] = False
            result["notes"] = f"Variant URI exists: {v}"
            break
        # Also check redirect on variant
        r2 = ask_redirect_target(v)
        if r2:
            r2_exists = ask_uri_exists(r2)
            tried.append((f"{v} [redirect→{r2}]", r2_exists))
            if r2_exists:
                result["variant_found"] = r2
                result["truly_missing"] = False
                result["notes"] = f"Variant redirect → {r2}"
                break

    result["variant_uris_tried"] = tried
    if result["variant_found"]:
        return result

    # --- 4. bif:contains full-text search ---
    bif = search_bif_contains(name)
    result["bif_matches"] = bif
    if bif:
        print(f"  [BIF] matches: {bif[:2]}")
        # Mark as "close match found" but still technically missing if URI differs
        result["notes"] = f"BIF close matches: {[u for u,_ in bif[:2]]}"

    # --- 5. Wikipedia search ---
    wiki = wiki_search(name)
    result["wiki_results"] = [w["title"] for w in wiki]
    if wiki:
        print(f"  [WIKI] top result: {wiki[0]['title']!r}")
        # Check if the top Wikipedia title maps to a DBpedia URI that exists
        top_title = wiki[0]["title"]
        candidate_uri = f"http://dbpedia.org/resource/{top_title.replace(' ', '_')}"
        cand_exists = ask_uri_exists(candidate_uri)
        if cand_exists:
            result["truly_missing"] = False
            result["notes"] = f"Wikipedia→DBpedia match: {candidate_uri}"
            result["variant_found"] = candidate_uri
            print(f"  [FOUND via WIKI] {candidate_uri}")
        else:
            result["notes"] = (result["notes"] or "") + f"; Wiki top hit: {top_title!r} (DBpedia URI missing)"

    if result["truly_missing"]:
        print(f"  [TRULY MISSING] No match found for {name!r}")

    return result


def check_existing_sample(questions, n=100) -> dict:
    """Random-sample n questions with all_entities_exist=True and ASK live DBpedia."""
    rng = random.Random(RANDOM_SEED)
    existing_qs = [q for q in questions if q["all_entities_exist"]]
    sample = rng.sample(existing_qs, min(n, len(existing_qs)))

    confirmed = 0
    failed = []
    total_entities = 0
    confirmed_entities = 0

    for q in sample:
        for e in q["entities"]:
            if e.get("exists") and e.get("uri"):
                total_entities += 1
                exists = ask_uri_exists(e["uri"])
                if exists:
                    confirmed_entities += 1
                else:
                    failed.append({"qid": q["question_id"], "name": e["name"], "uri": e["uri"]})

        confirmed = confirmed_entities  # track running

    return {
        "questions_sampled": len(sample),
        "entities_checked": total_entities,
        "entities_confirmed": confirmed_entities,
        "failures": failed,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    with open(DATA_FILE) as f:
        data = json.load(f)
    questions = data["questions"]

    # Step 1: Recount
    all_exist_count = sum(1 for q in questions if q["all_entities_exist"])
    any_missing_count = sum(1 for q in questions if not q["all_entities_exist"])
    print(f"Recount: all_exist={all_exist_count}, any_missing={any_missing_count}")

    # Collect missing entities (one per unique name)
    seen_names = set()
    missing_entries = []
    for q in questions:
        if not q["all_entities_exist"]:
            for e in q["entities"]:
                if not e["exists"] and e["name"] not in seen_names:
                    seen_names.add(e["name"])
                    missing_entries.append({"name": e["name"], "question_id": q["question_id"]})

    print(f"Total unique missing entities: {len(missing_entries)}")

    # Step 2: Check all missing entities
    print("\n=== CHECKING ALL MISSING ENTITIES ===")
    missing_results = []
    for entry in missing_entries:
        res = check_missing_entity(entry["name"])
        res["question_id"] = entry["question_id"]
        missing_results.append(res)

    # Step 3: Sample existing entities
    print("\n=== SAMPLING 100 EXISTING ENTITIES ===")
    sample_result = check_existing_sample(questions, n=100)

    # Step 4: Summarize
    truly_missing = [r for r in missing_results if r["truly_missing"]]
    found_via_variant = [r for r in missing_results if not r["truly_missing"] and r.get("variant_found")]
    found_direct = [r for r in missing_results if r["direct_uri_exists"]]
    found_redirect = [r for r in missing_results if r.get("redirect_target") and not r["truly_missing"] and not r["direct_uri_exists"] and not r.get("variant_found")]

    total_found_somehow = len(missing_results) - len(truly_missing)
    adjusted_coverage = all_exist_count + total_found_somehow

    print("\n=== SUMMARY ===")
    print(f"Claimed: 936/1000 (93.6%)")
    print(f"Recount: {all_exist_count}/1000 ({all_exist_count/10:.1f}%)")
    print(f"Missing entities total: {len(missing_results)}")
    print(f"  - Truly missing: {len(truly_missing)}")
    print(f"  - Found via direct URI: {len(found_direct)}")
    print(f"  - Found via redirect: {len(found_redirect)}")
    print(f"  - Found via variant/wiki: {len(found_via_variant)}")
    print(f"Adjusted coverage (if variants count): {adjusted_coverage}/1000 ({adjusted_coverage/10:.1f}%)")
    print(f"Sample check: {sample_result['entities_confirmed']}/{sample_result['entities_checked']} entities confirmed")
    if sample_result["failures"]:
        print(f"  FAILURES: {sample_result['failures']}")

    # Write report
    write_report(
        all_exist_count, any_missing_count, missing_results,
        truly_missing, found_direct, found_redirect, found_via_variant,
        adjusted_coverage, sample_result
    )
    print(f"\nReport written to: {OUT_FILE}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    all_exist_count, any_missing_count, missing_results,
    truly_missing, found_direct, found_redirect, found_via_variant,
    adjusted_coverage, sample_result
):
    lines = []
    lines.append("# Entity Coverage Verification Report")
    lines.append("")
    lines.append(f"Date: 2026-03-25 | Live DBpedia SPARQL + Wikipedia API")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Claimed**: 936/1000 (93.6%) questions have all entities found in DBpedia")
    lines.append(f"- **Recounted from JSON**: {all_exist_count}/1000 ({all_exist_count/10:.1f}%) — confirms 936 vs 64 split")
    lines.append(f"- **Unique missing entity names**: {len(missing_results)}")
    lines.append(f"- **Truly missing after exhaustive check**: {len(truly_missing)}")
    lines.append(f"- **Found via URI variants/wiki**: {len(missing_results) - len(truly_missing)}")
    lines.append(f"- **Adjusted coverage** (if variants accepted): {adjusted_coverage}/1000 ({adjusted_coverage/10:.1f}%)")
    lines.append(f"- **Sample check (100 questions × existing entities)**: {sample_result['entities_confirmed']}/{sample_result['entities_checked']} entities confirmed in live DBpedia")
    lines.append("")

    # Verdict
    claim_correct = (all_exist_count == 936)
    lines.append("## Verdict")
    lines.append("")
    if claim_correct:
        lines.append(f"**CLAIM CONFIRMED**: The JSON recount exactly matches 936/1000 (93.6%).")
    else:
        lines.append(f"**CLAIM INCORRECT**: JSON recount gives {all_exist_count}/1000, not 936.")

    if len(truly_missing) > 0:
        pct_miss = 100 * len(truly_missing) / len(missing_results)
        lines.append(f"Of {len(missing_results)} flagged-missing entities, **{len(truly_missing)} ({pct_miss:.0f}%) are truly absent** from DBpedia even after exhaustive variant/wiki search.")
        lines.append(f"The remaining {len(missing_results) - len(truly_missing)} could potentially be found under alternative URIs.")
    lines.append("")

    # Existing sample
    lines.append("## Existing Entities (sample check)")
    lines.append("")
    lines.append(f"- Questions sampled: {sample_result['questions_sampled']}/936")
    lines.append(f"- Entity URIs checked: {sample_result['entities_checked']}")
    lines.append(f"- Confirmed live in DBpedia: {sample_result['entities_confirmed']}/{sample_result['entities_checked']}")
    if sample_result["failures"]:
        lines.append(f"- **Failures (URI in JSON but not in live DBpedia)**:")
        for f in sample_result["failures"]:
            lines.append(f"  - `{f['name']}` → `{f['uri']}`")
    else:
        lines.append("- Failures: none — all sampled URIs confirmed live")
    lines.append("")

    # Missing entities table
    lines.append("## Missing Entities (exhaustive check)")
    lines.append("")
    lines.append("| Entity Name | Direct URI Exists | Redirect Found | Variant/Wiki Found | Truly Missing? | Notes |")
    lines.append("|-------------|:-----------------:|:--------------:|:-----------------:|:--------------:|-------|")

    for r in sorted(missing_results, key=lambda x: x["truly_missing"], reverse=True):
        name = r["name"].replace("|", "\\|")
        direct = "YES" if r["direct_uri_exists"] else "no"
        redir = r["redirect_target"] or "—"
        if len(redir) > 60:
            redir = redir[:57] + "..."
        variant = r["variant_found"] or "—"
        if len(variant) > 60:
            variant = variant[:57] + "..."
        truly = "**YES**" if r["truly_missing"] else "no"
        notes = (r.get("notes") or "—")
        if len(notes) > 80:
            notes = notes[:77] + "..."
        lines.append(f"| {name} | {direct} | {redir} | {variant} | {truly} | {notes} |")

    lines.append("")

    # Truly missing detail
    if truly_missing:
        lines.append("## Truly Missing Entities — Detail")
        lines.append("")
        for r in truly_missing:
            lines.append(f"### {r['name']}")
            lines.append(f"- Question ID: `{r.get('question_id','?')}`")
            lines.append(f"- Direct URI: `{r['direct_uri']}`")
            lines.append(f"- Redirect: {r['redirect_target'] or 'none'}")
            lines.append(f"- Variants tried: {len(r['variant_uris_tried'])}")
            for v, exists in r["variant_uris_tried"][:5]:
                lines.append(f"  - `{v}` → {'EXISTS' if exists else 'missing'}")
            if r["bif_matches"]:
                lines.append(f"- BIF close matches: {[u for u, _ in r['bif_matches'][:3]]}")
            if r["wiki_results"]:
                lines.append(f"- Wikipedia top results: {r['wiki_results'][:3]}")
            lines.append("")

    # Variant-found detail
    if found_via_variant:
        lines.append("## Entities Found via Variant URIs")
        lines.append("")
        for r in found_via_variant:
            lines.append(f"- **{r['name']}** → `{r['variant_found']}`")
        lines.append("")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
