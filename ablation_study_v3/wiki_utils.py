"""Wikipedia + LLM utilities with retry logic for ablation study."""

import re
import time
import logging
from typing import Optional
from urllib.parse import quote

import requests
from groq import Groq
from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)

WIKI_HEADERS = {"User-Agent": "AblationStudyV3/1.0"}
WIKI_DELAY = 0.5  # seconds between Wikipedia API calls
_last_wiki_call = 0.0


def _rate_limit_wiki():
    global _last_wiki_call
    elapsed = time.time() - _last_wiki_call
    if elapsed < WIKI_DELAY:
        time.sleep(WIKI_DELAY - elapsed)
    _last_wiki_call = time.time()


def _retry(func, max_attempts=3, backoff=2.0):
    """Retry a function with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.warning(f"Failed after {max_attempts} attempts: {e}")
                return None
            time.sleep(backoff ** attempt)
    return None


def llm_call_with_retry(client, prompt, max_attempts=5, initial_wait=10.0,
                        max_tokens=100, temperature=0):
    """Make a Groq LLM call with retry on 429 rate limit errors."""
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_attempts - 1:
                wait = initial_wait * (2 ** attempt)
                logger.info(f"Rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
            else:
                raise
    return None


def search_wikipedia(query: str, limit: int = 3) -> list[str]:
    """Search Wikipedia and return matching titles."""
    _rate_limit_wiki()

    def _do():
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search",
                    "srsearch": query, "srlimit": limit, "format": "json"},
            headers=WIKI_HEADERS, timeout=15,
        )
        return [r["title"] for r in resp.json().get("query", {}).get("search", [])]

    return _retry(_do) or []


def get_wikipedia_text_by_title(title: str, max_chars: int = 3000) -> Optional[str]:
    """Get Wikipedia article text by exact title."""
    _rate_limit_wiki()

    def _do():
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "titles": title.replace(" ", "_"),
                    "prop": "extracts", "explaintext": True, "format": "json"},
            headers=WIKI_HEADERS, timeout=15,
        )
        for page in resp.json().get("query", {}).get("pages", {}).values():
            if page.get("pageid"):
                text = page.get("extract", "")
                return text[:max_chars] if text else None
        return None

    return _retry(_do)


def get_wikipedia_text(title: str, max_chars: int = 3000, context: str = "") -> Optional[str]:
    """Get Wikipedia text with search fallback."""
    text = get_wikipedia_text_by_title(title, max_chars)
    if text:
        return text

    search_query = f"{title} {context}".strip()
    for search_title in search_wikipedia(search_query, limit=5):
        text = get_wikipedia_text_by_title(search_title, max_chars)
        if text:
            return text
    return None


def extract_answer_with_llm(
    client: Groq, text: str, relation: str,
    question: str, entity_context: str = "",
) -> Optional[str]:
    """Use LLM to extract a specific fact from Wikipedia text."""
    if not text:
        return None

    prompt = f"""Extract information from Wikipedia text to answer a question.

Question: {question}
Looking for: {relation}
{f"Context: {entity_context}" if entity_context else ""}

Wikipedia text:
{text}

Instructions:
1. Find the specific information requested ({relation})
2. Return ONLY the extracted value, nothing else
3. If the information is not in the text, respond with "NOT_FOUND"
4. For dates, return the full date if available
5. For names, return the full name

Answer:"""

    try:
        answer = llm_call_with_retry(client, prompt)
        if not answer or answer == "NOT_FOUND" or "not found" in answer.lower():
            return None
        return answer
    except Exception as e:
        logger.warning(f"LLM extraction error: {e}")
        return None


def compare_with_llm(
    client: Groq, question: str,
    entity1: str, value1: str, entity2: str, value2: str,
) -> Optional[str]:
    """Use LLM to compare two values and answer a comparison question."""
    prompt = f"""Answer this comparison question based on the extracted values.

Question: {question}

Entity 1: {entity1}
Value 1: {value1}

Entity 2: {entity2}
Value 2: {value2}

Instructions:
1. Compare the values based on what the question asks
2. For "born first" / "earlier" questions, the earlier date is the answer
3. For "born later" / "more recently" questions, the later date is the answer
4. For "same nationality" / "same country" questions, answer "yes" or "no"
5. Return ONLY the answer (entity name, "yes", or "no")

Answer:"""

    try:
        return llm_call_with_retry(client, prompt)
    except Exception as e:
        logger.warning(f"LLM comparison error: {e}")
        return None


# Entity URI overrides for common disambiguation
ENTITY_OVERRIDES = {
    "prince william": "William,_Prince_of_Wales",
    "prince harry": "Prince_Harry,_Duke_of_Sussex",
    "inception": "Inception",
    "the dark knight": "The_Dark_Knight",
    "harry potter": "Harry_Potter",
    "barack obama": "Barack_Obama",
    "donald trump": "Donald_Trump",
}


def entity_to_uri(entity: str, sparql: Optional[SPARQLWrapper] = None) -> str:
    """Convert entity name to DBpedia URI with improved resolution.

    Steps: overrides -> URL-encode -> ASK verify -> rdfs:label fallback.
    """
    entity_lower = entity.lower().strip()
    if entity_lower in ENTITY_OVERRIDES:
        return f"http://dbpedia.org/resource/{ENTITY_OVERRIDES[entity_lower]}"

    # URL-safe conversion: encode special chars but keep underscores
    uri_part = entity.strip().replace(" ", "_")
    uri_part = quote(uri_part, safe="_,().'")
    uri = f"http://dbpedia.org/resource/{uri_part}"

    if not sparql:
        return uri

    # Verify with ASK query
    try:
        sparql.setQuery(f"ASK WHERE {{ <{uri}> ?p ?o . }}")
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
        if result.get("boolean", False):
            return uri
    except Exception:
        pass

    # Fallback: rdfs:label search
    try:
        safe_label = entity.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
        sparql.setQuery(f"""
            SELECT ?entity WHERE {{
                ?entity rdfs:label "{safe_label}"@en .
                FILTER(STRSTARTS(STR(?entity), "http://dbpedia.org/resource/"))
            }} LIMIT 1
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["entity"]["value"]
    except Exception:
        pass

    return uri


def uri_to_wiki_title(uri: str) -> str:
    """Extract Wikipedia title from a DBpedia URI."""
    if "dbpedia.org/resource/" in uri:
        return uri.split("/resource/")[-1].replace("_", " ")
    return uri


def answers_match(extracted: str, gold: str) -> bool:
    """Fuzzy answer matching (for internal routing, not final scoring)."""
    if not extracted or not gold:
        return False

    def _norm(s):
        s = s.lower().strip()
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'\([^)]*\)', '', s).strip()
        return s

    ext, gld = _norm(extracted), _norm(gold)
    if ext == gld or gld in ext or ext in gld:
        return True

    ext_year = re.search(r'\d{4}', extracted)
    gold_year = re.search(r'\d{4}', gold)
    if ext_year and gold_year and ext_year.group() == gold_year.group():
        return True

    return False
