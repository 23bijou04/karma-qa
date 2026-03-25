#!/usr/bin/env python3
"""Low-level Wikidata/DBpedia SPARQL helpers for Phase 2 benchmark error detection.

Handles: rate-limited SPARQL requests, DBpedia owl:sameAs QID resolution,
Wikidata search API fallback, date parsing utilities.
"""

import logging
import re
import time

import requests

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
DBPEDIA_SPARQL = "https://dbpedia.org/sparql"
USER_AGENT = "DBpediaHybridQA/1.0 (academic research)"
# Wikidata requires application/json; DBpedia SPARQL requires sparql-results+json
_SPARQL_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

DATE_PROPS = {"P569", "P570", "P577", "P571"}
MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def is_date_like(value: str) -> bool:
    """Return True for values that are dates/literals (not resolvable as entities)."""
    v = value.strip()
    if not v:
        return True
    if v[0].isdigit():
        return True
    first_word = v.split()[0].lower() if " " in v else v.lower()
    return first_word in MONTH_NAMES


def extract_year(value: str) -> int | None:
    """Extract 4-digit year from a date string (benchmark or xsd:dateTime)."""
    m = re.search(r"\b(1[0-9]{3}|2[0-9]{3})\b", value)
    return int(m.group(1)) if m else None


def sparql_request(endpoint: str, query: str, delay: float = 1.0) -> list[dict]:
    """Execute a SPARQL SELECT query; return list of binding dicts.

    Retries up to 3 times with exponential backoff on 429/5xx responses.
    """
    time.sleep(delay)
    # DBpedia SPARQL requires sparql-results+json; Wikidata accepts application/json
    req_headers = _SPARQL_HEADERS if endpoint == DBPEDIA_SPARQL else HEADERS
    for attempt in range(3):
        try:
            resp = requests.get(
                endpoint,
                params={"query": query, "format": "json"},
                headers=req_headers,
                timeout=30,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 10 * (2 ** attempt)
                logger.warning(
                    "HTTP %s from %s; retrying in %ss", resp.status_code, endpoint, wait
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("results", {}).get("bindings", [])
        except requests.RequestException as exc:
            if attempt == 2:
                logger.error("SPARQL query failed after 3 attempts: %s", exc)
                return []
            time.sleep(10 * (2 ** attempt))
    return []


def dbpedia_batch_qids(uris: list[str]) -> dict[str, str]:
    """Resolve DBpedia URIs -> Wikidata QIDs via owl:sameAs (both directions).

    Uses full URI syntax in VALUES to handle entity names with commas.
    Queries both <dbr owl:sameAs wd> and <wd owl:sameAs dbr> directions.
    """
    values_str = " ".join(f"<{u}>" for u in uris)
    query = f"""
SELECT ?dbr ?wd WHERE {{
  VALUES ?dbr {{ {values_str} }}
  {{ ?dbr <http://www.w3.org/2002/07/owl#sameAs> ?wd }}
  UNION
  {{ ?wd <http://www.w3.org/2002/07/owl#sameAs> ?dbr }}
  FILTER(STRSTARTS(STR(?wd), "http://www.wikidata.org/entity/Q"))
}}
"""
    bindings = sparql_request(DBPEDIA_SPARQL, query, delay=1.0)
    result: dict[str, str] = {}
    for row in bindings:
        dbr = row.get("dbr", {}).get("value", "")
        wd = row.get("wd", {}).get("value", "")
        if dbr and wd:
            result[dbr] = wd.split("/")[-1]
    return result


def wikidata_search_qid(name: str) -> str | None:
    """Look up a Wikidata QID by entity name via wbsearchentities.

    Prefers exact label match (case-insensitive); falls back to top result.
    Retries with exponential backoff on 429.
    """
    for attempt in range(3):
        delay = 2.0 * (2 ** attempt)
        time.sleep(delay)
        try:
            resp = requests.get(
                WIKIDATA_SEARCH,
                params={
                    "action": "wbsearchentities",
                    "search": name,
                    "language": "en",
                    "limit": 5,
                    "format": "json",
                },
                headers=HEADERS,
                timeout=15,
            )
            if resp.status_code == 429:
                wait = 10 * (2 ** attempt)
                logger.warning("wbsearchentities 429 for %r; retrying in %ss", name, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            hits = resp.json().get("search", [])
            if not hits:
                return None
            for hit in hits:
                if hit.get("label", "").lower() == name.lower():
                    return hit["id"]
            return hits[0]["id"]
        except Exception as exc:
            if attempt == 2:
                logger.warning("wbsearchentities failed for %r: %s", name, exc)
                return None
    return None


def wikidata_claim_query(pid: str, subject_qids: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Query ALL Wikidata statements for a property across batched subject QIDs.

    Returns: {subject_qid: [(value_str, rank_label), ...]}
    """
    found: dict[str, list[tuple[str, str]]] = {}
    for i in range(0, len(subject_qids), 50):
        batch = subject_qids[i:i + 50]
        values_str = " ".join(f"wd:{q}" for q in batch)
        query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT ?entity ?value ?rank WHERE {{
  VALUES ?entity {{ {values_str} }}
  ?entity p:{pid} ?stmt .
  ?stmt ps:{pid} ?value .
  ?stmt wikibase:rank ?rank .
}}
"""
        bindings = sparql_request(WIKIDATA_SPARQL, query, delay=3.0)
        for row in bindings:
            subj = row.get("entity", {}).get("value", "").split("/")[-1]
            val = row.get("value", {}).get("value", "")
            rank = row.get("rank", {}).get("value", "").split("#")[-1]
            found.setdefault(subj, []).append((val, rank))
    return found
