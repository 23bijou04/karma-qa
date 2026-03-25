"""
Pass 2: Re-check the 54 "truly missing" entities from pass 1.
Fixes:
  - Wikipedia API: add User-Agent header
  - BIF search: sanitize quotes in entity names before querying
  - Additional DBpedia URI strategies per entity
"""
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests
from SPARQLWrapper import JSON, SPARQLWrapper

SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
WIKI_API = "https://en.wikipedia.org/w/api.php"
SPARQL_DELAY = 0.3
WIKI_DELAY = 0.5

WIKI_HEADERS = {
    "User-Agent": "DBpediaQA-EntityVerifier/1.0 (academic research; contact: researcher@example.com)"
}

sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.setReturnFormat(JSON)
sparql.setTimeout(15)
sparql.addCustomHttpHeader("User-Agent", "DBpediaQA-EntityVerifier/1.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask_uri_exists(uri: str) -> bool:
    time.sleep(SPARQL_DELAY)
    q = f"ASK {{ <{uri}> ?p ?o }}"
    sparql.setQuery(q)
    try:
        res = sparql.query().convert()
        return bool(res.get("boolean", False))
    except Exception as e:
        print(f"  [SPARQL ERR] ASK: {e}")
        return False


def name_to_uri(name: str) -> str:
    normalized = name.strip().replace(" ", "_")
    return f"http://dbpedia.org/resource/{urllib.parse.quote(normalized, safe=':/#_')}"


def bif_search(name: str) -> list:
    """Full-text search; sanitize the name to avoid SPARQL parse errors."""
    time.sleep(SPARQL_DELAY)
    # Strip all quote chars and pick longest alphanumeric words
    clean = re.sub(r'["\'\u201c\u201d\u2018\u2019]', '', name)
    words = [w for w in re.split(r'[\s\-,.()\[\]]+', clean) if len(w) >= 4]
    if not words:
        return []
    # Use top 3 words
    search_terms = " AND ".join(f'"{w}"' for w in words[:3])
    q = f"""
SELECT DISTINCT ?s ?label WHERE {{
  ?s rdfs:label ?label .
  ?label bif:contains '{search_terms}' .
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
        print(f"  [BIF ERR] {e}")
        return []


def wiki_search(name: str) -> list:
    time.sleep(WIKI_DELAY)
    # Clean name for search
    clean_name = re.sub(r'^["\'"\u201c\u201d]+', '', name).strip()
    # Remove parenthetical/descriptive suffixes
    clean_name = re.sub(r'\s*\(.*\)\s*$', '', clean_name).strip()
    clean_name = re.sub(r',.*$', '', clean_name).strip()
    params = {
        "action": "query",
        "list": "search",
        "srsearch": clean_name,
        "srlimit": 5,
        "format": "json",
        "utf8": 1,
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=10)
        data = r.json()
        return data.get("query", {}).get("search", [])
    except Exception as e:
        print(f"  [WIKI ERR] {e}")
        return []


def extra_uri_variants(name: str) -> list[str]:
    """Additional variants not tried in pass 1."""
    variants = []
    # Strip leading punctuation (quotes, curly quotes)
    stripped = re.sub(r'^[\"\'\u201c\u201d\u2018\u2019]+', '', name).strip()
    # Strip trailing punctuation
    stripped = re.sub(r'[\"\'\u201c\u201d\u2018\u2019]+$', '', stripped).strip()
    if stripped != name:
        variants.append(name_to_uri(stripped))

    # Strip "(died YYYY)" or "(c. YYYY-YYYY)" patterns
    date_stripped = re.sub(r'\s*\((died|c\.|fl\.)[^)]*\)\s*', ' ', name).strip()
    date_stripped = re.sub(r'\s+', ' ', date_stripped)
    if date_stripped != name and date_stripped:
        variants.append(name_to_uri(date_stripped))
        # Also strip comma suffix from date_stripped
        comma_stripped = re.sub(r',.*$', '', date_stripped).strip()
        if comma_stripped != date_stripped:
            variants.append(name_to_uri(comma_stripped))

    # "X was a Y" → "X"
    was_stripped = re.sub(r'\s+was\s+.*$', '', name, flags=re.IGNORECASE).strip()
    was_stripped = re.sub(r'\s+\(.*', '', was_stripped).strip()
    if was_stripped != name and was_stripped:
        variants.append(name_to_uri(was_stripped))

    # Remove "The " prefix
    if name.startswith("The "):
        variants.append(name_to_uri(name[4:]))

    # Try with "_(disambiguation)" suffix removed if present
    # Also try without "Jr." suffix
    jr_stripped = re.sub(r'\s+Jr\.$', '', name).strip()
    if jr_stripped != name:
        variants.append(name_to_uri(jr_stripped))

    # Deduplicate
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def check_entity(name: str) -> dict:
    print(f"\n[PASS2] {name!r}")
    result = {
        "name": name,
        "found": False,
        "found_uri": None,
        "found_via": None,
        "bif_matches": [],
        "wiki_results": [],
        "wiki_dbpedia_match": None,
        "notes": "",
    }

    # 1. Extra URI variants
    for v in extra_uri_variants(name):
        exists = ask_uri_exists(v)
        print(f"  [VARIANT] {v} → {'EXISTS' if exists else 'missing'}")
        if exists:
            result["found"] = True
            result["found_uri"] = v
            result["found_via"] = "variant_uri"
            return result

    # 2. BIF search
    bif = bif_search(name)
    result["bif_matches"] = [(u, l) for u, l in bif]
    if bif:
        print(f"  [BIF] {bif[:3]}")
        # Check each BIF match exists (it should since it came from a query, but verify)
        for uri, label in bif[:3]:
            if ask_uri_exists(uri):
                result["found"] = True
                result["found_uri"] = uri
                result["found_via"] = f"bif_search (label: {label!r})"
                print(f"  [FOUND via BIF] {uri}")
                return result

    # 3. Wikipedia search → try to map to DBpedia URI
    wiki = wiki_search(name)
    result["wiki_results"] = [w["title"] for w in wiki]
    if wiki:
        print(f"  [WIKI] top hits: {[w['title'] for w in wiki[:3]]}")
        for w in wiki[:3]:
            candidate = f"http://dbpedia.org/resource/{w['title'].replace(' ', '_')}"
            exists = ask_uri_exists(candidate)
            print(f"  [WIKI→DBpedia] {candidate} → {'EXISTS' if exists else 'missing'}")
            if exists:
                result["found"] = True
                result["found_uri"] = candidate
                result["found_via"] = f"wikipedia→dbpedia (wiki: {w['title']!r})"
                result["wiki_dbpedia_match"] = candidate
                print(f"  [FOUND via WIKI→DBpedia] {candidate}")
                return result

    print(f"  [STILL MISSING] {name!r}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TRULY_MISSING_FROM_PASS1 = [
    "Anne \"Henriette\" of France",
    "Thomas George Lyon-Bowes, Lord Glamis",
    "Jeanne Richomme Raunay",
    "The 3rd Count of Cardona",
    "John Ramon III Folch de Cardona",
    "Philippe of France, Duke of Anjou",
    "Sholto Charles Douglas, 15th Earl of Morton",
    "Ari'i-Otare Terii-maeva-rua III",
    "Lady Rose Mary Primrose McLaren",
    "Princess Catherine Frederica Charlotte of Württemberg",
    "Sherry Horman",
    "Joseph Ferdinand Leopold of Bavaria",
    "Universidad Militar Bolivariana de Venezuela",
    "Empress Dowager Từ Dụ",
    "Bernhard Albrecht, count of Limburg and Bronckhorst",
    "Ermengarde (or Irmingard) of Hesbaye",
    "Elene (1753 \u2013 17 June 1786) was a Georgia",
    "Marie of Namur",
    "Lev Mikhailovich Karakhan",
    "Sigrid Eriksdotter of Sweden",
    "Prince Charles Louis of Orléans, Duke of Chartres",
    "Princess Adrienne of Sweden, Duchess of Blekinge",
    "Infanta Adelgundes",
    "Sonia Haft Greene",
    "Ælfwine (c. 661\u2013679) was the King of Deira",
    "Eochu (or Eochaid) Uairches",
    "Tony Lee Bettenhausen Jr.",
    "Lord George William Montagu Douglas Scott",
    "Anne Baldwin Gust Brown",
    "Lady Dorothy Evelyn Macmillan",
    "Frederick Cavendish-Bentinck",
    "Gerhart \"Gerry\" Neugebauer",
    "\"Aren't You Kinda Glad We Did?",
    "Natalia Alexeievna",
    "Hamad bin Isa bin Salman Al Khalifa",
    "Gal Young ' Un",
    "Victor Amadeus of Anhalt-Bernburg",
    "Jean Everald Tangye",
    "Lady Helen Marina Lucy Taylor",
    "Lionel Edward Sackville",
    "Captain Napier George Henry Sturt, 3rd Baron Alington",
    "The University of Los Andes",
    "Tatiana Celia Kennedy Schlossberg",
    "Ferdinand Wilhelm Ernst, 2nd Prince of Solms-Braunfels",
    "William \"Bill\" Reeves",
    "Frederick William Dickens",
    "Alfred Lewin Copley",
    "Fernando Bermúdez",
    "976-EVIL II",
    "Filippa Eriksdotter Fleming",
    "Robert James Petre",
    "William Louis of Anhalt-Harzgerode",
    "Johann Ernst of Saxe-Eisenach",
    "Imam Abdul Aziz bin Muhammad",
]

def main():
    results = []
    for name in TRULY_MISSING_FROM_PASS1:
        res = check_entity(name)
        results.append(res)

    found = [r for r in results if r["found"]]
    still_missing = [r for r in results if not r["found"]]

    print(f"\n=== PASS 2 SUMMARY ===")
    print(f"Re-checked: {len(results)}")
    print(f"Found in pass 2: {len(found)}")
    print(f"Still missing: {len(still_missing)}")

    print("\n=== FOUND VIA PASS 2 ===")
    for r in found:
        print(f"  {r['name']!r} → {r['found_uri']} [{r['found_via']}]")

    print("\n=== STILL MISSING ===")
    for r in still_missing:
        bif = r.get("bif_matches", [])
        wiki = r.get("wiki_results", [])
        print(f"  {r['name']!r}")
        if bif:
            print(f"    BIF close: {[l for _, l in bif[:2]]}")
        if wiki:
            print(f"    Wiki hits: {wiki[:3]}")

    # Save as JSON for report merging
    out = Path("data/eval_1000q/pass2_missing_check.json")
    with open(out, "w") as f:
        json.dump({"found": found, "still_missing": still_missing}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
