"""SPARQL helper functions for MuSiQue DBpedia grounding analysis.

Provides: run_sparql(), resolve_title_to_uri(), check_connection(),
answer_grounded(). Adapted from verify_hotpotqa_dbpedia_grounding.py.
"""
import re
import time
from urllib.parse import quote, unquote

from SPARQLWrapper import JSON, SPARQLWrapper

# ── SPARQL setup ────────────────────────────────────────────────────────────
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
PREFIXES = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX dbr: <http://dbpedia.org/resource/>
"""
sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.setReturnFormat(JSON)
sparql.setTimeout(30)
sparql.addCustomHttpHeader("User-Agent", "DBpediaQA-MuSiQueGrounding/1.0")

_last_query_time = 0.0
# Cache: title → URI or None
uri_cache: dict[str, str | None] = {}


def _rate_limit():
    global _last_query_time
    elapsed = time.time() - _last_query_time
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
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return None


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def resolve_title_to_uri(title: str) -> str | None:
    """Resolve Wikipedia title → DBpedia URI (2-layer: direct + label search)."""
    if title in uri_cache:
        return uri_cache[title]
    # Layer 1: direct URI construction + ASK
    encoded = title.replace(" ", "_")
    uri = f"http://dbpedia.org/resource/{quote(encoded, safe='_,()·:')}"
    result = run_sparql(f"ASK {{ <{uri}> ?p ?o }}")
    if result and result.get("boolean"):
        uri_cache[title] = uri
        return uri
    # Layer 2: rdfs:label exact match
    escaped = _escape(title)
    result = run_sparql(f'SELECT ?s WHERE {{ ?s rdfs:label "{escaped}"@en }} LIMIT 1')
    if result:
        bindings = result.get("results", {}).get("bindings", [])
        if bindings:
            found = bindings[0]["s"]["value"]
            uri_cache[title] = found
            return found
    uri_cache[title] = None
    return None


NOISE_FILTERS = """
  FILTER(?p != <http://dbpedia.org/ontology/wikiPageWikiLink>)
  FILTER(?p != <http://www.w3.org/2002/07/owl#sameAs>)
  FILTER(?p != <http://www.w3.org/ns/prov#wasDerivedFrom>)
  FILTER(!STRSTARTS(STR(?p), "http://www.w3.org/"))
"""


def check_connection(uri1: str, uri2: str) -> tuple[bool, str]:
    """Check 1-hop then 2-hop connection. Returns (connected, type)."""
    # 1-hop
    q = f"ASK {{ {{ <{uri1}> ?p <{uri2}> }} UNION {{ <{uri2}> ?p <{uri1}> }} {NOISE_FILTERS} }}"
    result = run_sparql(q)
    if result and result.get("boolean"):
        return True, "1hop"
    # 2-hop
    q2 = f"""ASK {{
  {{ <{uri1}> ?p1 ?mid . ?mid ?p2 <{uri2}> }}
  UNION {{ <{uri2}> ?p1 ?mid . ?mid ?p2 <{uri1}> }}
  FILTER(?p1 != <http://dbpedia.org/ontology/wikiPageWikiLink>)
  FILTER(?p2 != <http://dbpedia.org/ontology/wikiPageWikiLink>)
  FILTER(!STRSTARTS(STR(?p1), "http://www.w3.org/"))
  FILTER(!STRSTARTS(STR(?p2), "http://www.w3.org/"))
}}"""
    result = run_sparql(q2, retries=1, timeout=15)
    if result is None:
        return False, "timeout"
    return result.get("boolean", False), "2hop"


def retrieve_values(uri: str) -> list[str]:
    """Fetch all dbo:/dbp: property values for an entity."""
    q = f"""SELECT ?o ?label WHERE {{
  <{uri}> ?p ?o .
  FILTER(STRSTARTS(STR(?p), "http://dbpedia.org/ontology/") ||
         STRSTARTS(STR(?p), "http://dbpedia.org/property/"))
  OPTIONAL {{ ?o rdfs:label ?label . FILTER(LANG(?label) = "en") }}
}} LIMIT 100"""
    result = run_sparql(q)
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


def answer_grounded(gold: str, norm_fn, values: list[str]) -> tuple[bool, str | None]:
    """Check if gold answer matches any value. Length-ratio guard >=0.4."""
    norm_gold = norm_fn(gold)
    if not norm_gold:
        return False, None
    for val in values:
        norm_val = norm_fn(val)
        if not norm_val:
            continue
        if norm_gold == norm_val:
            return True, val
        shorter, longer = sorted([norm_gold, norm_val], key=len)
        if len(longer) > 0 and len(shorter) / len(longer) >= 0.4:
            if norm_gold in norm_val or norm_val in norm_gold:
                return True, val
        if re.match(r"^\d{4}-\d{2}-\d{2}", val):
            if val[:4] == gold.strip():
                return True, val
    return False, None
