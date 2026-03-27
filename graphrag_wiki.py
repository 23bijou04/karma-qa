#!/usr/bin/env python3
"""General-purpose GraphRAG: DBpedia Spotlight + KG exploration + Wikipedia text + LLM.

Pipeline:
  1. Link entities via DBpedia Spotlight (spaCy fallback)
  2. Explore KG: follow 1-hop relations to discover related entities
  3. Retrieve Wikipedia text for all entities (URI-based + web search)
  4. Extract relevant sections using question keywords
  5. LLM reasoning over assembled context

Usage:
  python3 graphrag_wiki.py "Who destroys the last Horcrux of Voldemort?" --api-key $GROQ_API_KEY
  python3 graphrag_wiki.py "Who directed Inception?" --verbose
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import requests

# Ensure parent directory is on sys.path for dbpedia_qa package imports
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from ddgs import DDGS

from ablation_study_v3.wiki_utils import (
    get_wikipedia_text_by_title,
    uri_to_wiki_title,
)
from dbpedia_qa.dbpedia import DBpediaEntityLinker, DBpediaSPARQLClient, SchemaDiscovery
from dbpedia_qa.dbpedia.schema_discovery import CACHED_CLASS_SCHEMAS
from dbpedia_qa.models import Entity, extract_final_answer
from llm_provider import call_llm, create_llm_client

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 25_000
MAX_CHARS_PER_PAGE = 6_000

STOPWORDS = {
    "who", "what", "when", "where", "which", "how", "why",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "did", "does", "do", "has", "have", "had",
    "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "and", "or", "not", "that", "this", "it", "its",
    "first", "last", "same", "both", "more", "most",
    "after", "before", "between", "about", "into", "through",
}

# Relation-to-section mapping: when an entity was discovered via a specific
# predicate, paragraphs containing these header/topic keywords get 2x score.
RELATION_SECTION_BOOST = {
    "director": ["early life", "personal life", "career", "biography"],
    "producer": ["early life", "career", "biography"],
    "author": ["early life", "career", "biography", "works"],
    "writer": ["early life", "career", "biography"],
    "starring": ["early life", "career", "filmography"],
    "birthPlace": ["early life", "born", "birth", "geography"],
    "birthDate": ["early life", "born", "birth", "biography"],
    "deathPlace": ["death", "died", "later life"],
    "deathDate": ["death", "died", "later life"],
    "parent": ["early life", "family", "personal life"],
    "father": ["early life", "family", "personal life"],
    "mother": ["early life", "family", "personal life"],
    "spouse": ["personal life", "family", "marriage"],
    "country": ["geography", "politics", "government"],
    "foundedBy": ["history", "founding"],
    "award": ["awards", "honors", "career"],
    "performer": ["discography", "career", "music", "singles"],
    "artist": ["discography", "career", "music", "singles"],
    "almaMater": ["education", "early life", "career"],
    "nationality": ["nationality", "demographics", "early life", "biography"],
    "genre": ["genre", "music", "discography", "career"],
    "deathCause": ["death", "died", "later life"],
    "publicationDate": ["publication", "bibliography", "works", "history", "release"],
    "releaseDate": ["release", "discography", "singles", "history"],
    "language": ["language", "languages", "demographics"],
}


class _CircuitBreaker:
    """Lightweight circuit breaker for external services."""

    def __init__(self, threshold: int = 3, cooldown: float = 60.0):
        self._failures = 0
        self._threshold = threshold
        self._tripped_at: float | None = None
        self._cooldown = cooldown

    def is_open(self) -> bool:
        if self._tripped_at is None:
            return False
        if time.time() - self._tripped_at >= self._cooldown:
            self._failures = 0
            self._tripped_at = None
            return False
        return True

    def record_failure(self):
        self._failures += 1
        if self._failures >= self._threshold:
            self._tripped_at = time.time()

    def record_success(self):
        self._failures = 0
        self._tripped_at = None


_ddg_breaker = _CircuitBreaker(threshold=3, cooldown=60.0)


# ---------------------------------------------------------------------------
# 1. DBpedia Spotlight linking (primary) + spaCy fallback
# ---------------------------------------------------------------------------

def link_entities_to_dbpedia(question: str,
                             linker: DBpediaEntityLinker) -> list[Entity]:
    """Link question entities to DBpedia URIs.

    Delegates to linker.link_entities() which handles the full cascade:
    Spotlight -> spaCy mention extraction -> bif:contains fallback.
    Every returned entity has a resolved URI (no more Entity(uri="")).
    """
    return linker.link_entities(question, question=question)


# ---------------------------------------------------------------------------
# 1c. Schema-guided KG exploration
# ---------------------------------------------------------------------------

# Subclass → cached parent schema mapping
_CLASS_TO_SCHEMA = {
    "Scientist": "Person", "Monarch": "Person", "Politician": "Person",
    "Athlete": "Person", "MilitaryPerson": "Person", "SportsPlayer": "Person",
    "Royalty": "Person", "FictionalCharacter": "Person", "Artist": "Person",
    "Cleric": "Person", "Noble": "Person", "OfficeHolder": "Person",
    "Country": "Place", "City": "Place", "Settlement": "Place",
    "PopulatedPlace": "Place", "NaturalPlace": "Place", "River": "Place",
    "Mountain": "Place", "AdministrativeRegion": "Place",
    "Band": "MusicalArtist", "Singer": "MusicalArtist",
    "WrittenWork": "Book", "Novel": "Book", "Comic": "Book",
    "Song": "MusicalWork", "Single": "MusicalWork", "Album": "MusicalWork",
}

# Noise classes: neighbors typed ONLY as these are dropped (blocklist approach)
_NOISE_CLASSES = {
    "Genre", "Language", "TopicalConcept", "TimePeriod", "Award",
    "EthnicGroup", "Currency", "Colour", "OlympicResult",
}

# Fallback: old prefix+blocklist filter when schema lookup returns empty
_USEFUL_RELATION_PREFIXES = (
    "http://dbpedia.org/ontology/",
    "http://dbpedia.org/property/",
)
_SKIP_RELATIONS = {
    "http://dbpedia.org/ontology/wikiPageWikiLink",
    "http://dbpedia.org/ontology/wikiPageExternalLink",
    "http://dbpedia.org/ontology/wikiPageRedirects",
    "http://dbpedia.org/ontology/wikiPageDisambiguates",
    "http://dbpedia.org/ontology/wikiPageID",
    "http://dbpedia.org/ontology/wikiPageRevisionID",
    "http://dbpedia.org/ontology/wikiPageLength",
    "http://dbpedia.org/ontology/thumbnail",
    "http://dbpedia.org/ontology/abstract",
}


def _get_allowed_relations(entity_uri: str,
                           schema_discovery: SchemaDiscovery,
                           verbose: bool = False) -> tuple[set[str] | None, str | None]:
    """Determine allowed relations for an entity using its ontology class.

    Returns (allowed_set, class_name). allowed_set is None if fallback mode.
    """
    try:
        class_name = schema_discovery.get_entity_class(entity_uri)
    except Exception as e:
        logger.warning(f"Schema class lookup failed for {entity_uri}: {e}")
        return None, None

    if not class_name:
        return None, None

    # Resolve subclass to parent schema
    schema_key = class_name
    if class_name not in CACHED_CLASS_SCHEMAS:
        schema_key = _CLASS_TO_SCHEMA.get(class_name)

    if schema_key and schema_key in CACHED_CLASS_SCHEMAS:
        schema = CACHED_CLASS_SCHEMAS[schema_key]
        allowed = set(schema["dbo"] + schema["dbp"])
        if verbose:
            print(f"    [class: {class_name} → schema: {schema_key}]"
                  f" allowed: {{{', '.join(sorted(allowed)[:8])}...}}")
        return allowed, class_name

    # Dynamic fallback: query actual properties
    try:
        actual = schema_discovery.get_actual_properties(entity_uri)
        allowed = set(actual["dbo"] + actual["dbp"])
        if allowed:
            if verbose:
                print(f"    [class: {class_name} → dynamic]"
                      f" allowed: {{{', '.join(sorted(allowed)[:8])}...}}")
            return allowed, class_name
    except Exception as e:
        logger.warning(f"Dynamic schema fallback failed for {entity_uri}: {e}")

    return None, class_name


def explore_kg(entities: list[Entity], client: DBpediaSPARQLClient,
               schema_discovery: SchemaDiscovery,
               limit_per_entity: int = 50,
               verbose: bool = False) -> list[Entity]:
    """Schema-guided KG exploration: discover related entities via ontology.

    Two-layer filtering:
      Layer 1: Only follow relations defined in the entity's class schema
      Layer 2: Drop neighbors whose types are all noise classes (blocklist)

    Falls back to prefix+blocklist filter when schema lookup returns empty.
    """
    all_entities = list(entities)
    seen_uris = {e.uri for e in entities if e.uri}

    if not seen_uris:
        if verbose:
            print("  (no entities with URIs, skipping KG exploration)")
        return all_entities

    for entity in entities:
        if not entity.uri:
            continue

        if verbose:
            print(f"  Exploring: {entity.label}")

        # Layer 1: Determine allowed relations from ontology class
        allowed, class_name = _get_allowed_relations(
            entity.uri, schema_discovery, verbose=verbose,
        )

        try:
            neighbors = client.get_neighbors_with_types(
                entity.uri, direction="out", limit=limit_per_entity,
            )
        except Exception as e:
            logger.warning(f"KG exploration failed for {entity.label}: {e}")
            continue

        for relation_uri, neighbor_uri, neighbor_types in neighbors:
            relation_name = relation_uri.split("/")[-1]

            # Layer 1: Schema-guided relation filter
            if allowed is not None:
                if relation_name not in allowed:
                    continue
            else:
                # Fallback: old prefix+blocklist filter
                if not any(relation_uri.startswith(p)
                           for p in _USEFUL_RELATION_PREFIXES):
                    continue
                if relation_uri in _SKIP_RELATIONS:
                    continue

            # Layer 2: Neighbor class blocklist
            # Drop only if ALL types are noise; untyped neighbors pass through
            if neighbor_types and neighbor_types.issubset(_NOISE_CLASSES):
                if verbose:
                    label = uri_to_wiki_title(neighbor_uri)
                    print(f"      {entity.label} --{relation_name}"
                          f"--> {label} [SKIP: {neighbor_types}]")
                continue

            # Structural safeguards
            if neighbor_uri in seen_uris:
                continue
            if "/resource/" not in neighbor_uri:
                continue
            local = neighbor_uri.split("/resource/")[-1]
            if any(local.startswith(p)
                   for p in ("Category:", "Template:", "File:")):
                continue

            label = uri_to_wiki_title(neighbor_uri)
            all_entities.append(Entity(
                uri=neighbor_uri, label=label, source="kg_explore",
                discovery_relation=relation_name,
            ))
            seen_uris.add(neighbor_uri)

            if verbose:
                type_info = f" [{', '.join(sorted(neighbor_types))}]" if neighbor_types else ""
                print(f"      {entity.label} --{relation_name}--> {label}{type_info}")

    return all_entities


# ---------------------------------------------------------------------------
# 2. Section extraction (from demo_graphrag_horcrux.py)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving section headers."""
    chunks = []
    for part in re.split(r'(^==+\s.*?\s==+$)', text, flags=re.MULTILINE):
        part = part.strip()
        if not part:
            continue
        if re.match(r'^==+\s', part):
            chunks.append(part)
        else:
            # Split on sentence boundaries (period/!/? followed by space+uppercase)
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', part)
            chunks.extend(s.strip() for s in sentences if s.strip())
    return chunks


def _score_chunk(chunk: str, keywords: list[str],
                 rare_keywords: set[str]) -> int:
    """Score a text chunk by keyword hits (rare keywords count double)."""
    chunk_lower = chunk.lower()
    hits = [kw for kw in keywords if kw in chunk_lower]
    return sum(2 if kw in rare_keywords else 1 for kw in hits)


def extract_relevant_sections(text: str, keywords: list[str],
                              char_budget: int = MAX_CHARS_PER_PAGE,
                              relation_hint: str | None = None) -> str:
    """Extract text ranked by keyword density, sentence-level for long blocks.

    1. Split into paragraphs (double-newline boundaries).
    2. Score paragraphs by keyword hits (rare keywords = 2x weight).
    3. If a paragraph exceeds the budget, break it into sentences and
       keep the highest-scoring sentences within the budget.
    """
    if not text:
        return ""

    # Normalize keywords (strip trailing punctuation)
    keywords = [kw.strip("?.,!:;\"'") for kw in keywords]
    keywords = [kw for kw in keywords if kw]

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        return text[:char_budget]

    # Count in how many paragraphs each keyword appears (to find rare ones)
    kw_freq = {}
    for kw in keywords:
        kw_lower = kw.lower()
        count = sum(1 for p in paragraphs if kw_lower in p.lower())
        kw_freq[kw_lower] = count / len(paragraphs) if paragraphs else 0

    # A keyword is "rare" if it appears in <10% of paragraphs
    rare_keywords = {kw for kw, freq in kw_freq.items() if freq < 0.1}

    # Score and rank paragraphs by keyword relevance
    scored = []
    for para in paragraphs:
        para_lower = para.lower()
        hits = [kw for kw in keywords if kw.lower() in para_lower]
        has_rare = any(kw.lower() in rare_keywords for kw in hits)

        # Keep if: 2+ keyword hits, or 1 rare keyword hit
        if len(hits) >= 2 or has_rare:
            score = sum(2 if kw.lower() in rare_keywords else 1 for kw in hits)
            # Relation-guided boost: 2x for paragraphs matching discovery relation
            if relation_hint and relation_hint in RELATION_SECTION_BOOST:
                boost_terms = RELATION_SECTION_BOOST[relation_hint]
                if any(term in para_lower for term in boost_terms):
                    score *= 2
            scored.append((score, para))

    # Fallback: if density filter found nothing, use any single-keyword match
    if not scored:
        for para in paragraphs:
            para_lower = para.lower()
            hits = [kw for kw in keywords if kw.lower() in para_lower]
            if hits:
                score = len(hits)
                if relation_hint and relation_hint in RELATION_SECTION_BOOST:
                    boost_terms = RELATION_SECTION_BOOST[relation_hint]
                    if any(term in para_lower for term in boost_terms):
                        score *= 2
                scored.append((score, para))

    if not scored:
        return text[:char_budget]

    # Sort by score (highest first), keep within char budget
    scored.sort(key=lambda x: x[0], reverse=True)
    result = []
    total = 0
    for _, para in scored:
        if total + len(para) <= char_budget:
            result.append(para)
            total += len(para)
        else:
            # Paragraph too large for remaining budget — extract best sentences
            remaining = char_budget - total
            if remaining < 100:
                break
            sentences = _split_sentences(para)
            sent_scored = [
                (_score_chunk(s, keywords, rare_keywords), s)
                for s in sentences
            ]
            sent_scored.sort(key=lambda x: x[0], reverse=True)
            # Collect top sentences, then re-order by original position
            picked = []
            budget_left = remaining
            for sc, sent in sent_scored:
                if len(sent) <= budget_left:
                    picked.append((sentences.index(sent), sent))
                    budget_left -= len(sent)
            # Re-order by original position to maintain narrative flow
            picked.sort(key=lambda x: x[0])
            if picked:
                result.append(" ".join(s for _, s in picked))
                total += sum(len(s) for _, s in picked)

    return "\n\n".join(result)


# ---------------------------------------------------------------------------
# 3. Web search for Wikipedia pages
# ---------------------------------------------------------------------------

_WIKI_SEARCH_HEADERS = {"User-Agent": "DBpediaQA/1.0 (research; hybrid-qa)"}


def web_search_wikipedia(query: str, limit: int = 5) -> list[str]:
    """Search for Wikipedia pages. Primary: Wikipedia API. Fallback: DuckDuckGo.

    Wikipedia's action=query&list=search is official, reliable, and has no
    undocumented rate limits (unlike DDG which silently blocks after ~50 calls).
    """
    # Primary: Wikipedia native search API
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srnamespace": 0,
                "format": "json",
            },
            headers=_WIKI_SEARCH_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        titles = [r["title"] for r in results if r.get("title")]
        if titles:
            return titles
    except Exception as e:
        logger.warning(f"Wikipedia search API failed: {e}")

    # Fallback: DuckDuckGo (with circuit breaker)
    if _ddg_breaker.is_open():
        logger.debug("DDG circuit breaker open, skipping")
        return []

    try:
        results = list(DDGS().text(
            f"site:en.wikipedia.org {query}", max_results=limit,
        ))
        titles = []
        for r in results:
            url = r.get("href", "")
            if "en.wikipedia.org/wiki/" in url:
                title = unquote(url.split("/wiki/")[-1]).replace("_", " ")
                if ":" not in title:
                    titles.append(title)
        _ddg_breaker.record_success()
        return titles
    except Exception as e:
        logger.warning(f"DDG fallback failed: {e}")
        _ddg_breaker.record_failure()
        return []


# ---------------------------------------------------------------------------
# 4. Wikipedia retrieval
# ---------------------------------------------------------------------------

def retrieve_wikipedia(entities: list[Entity], question: str,
                       verbose: bool = False) -> dict[str, str]:
    """Retrieve Wikipedia text for entities + web-search-expanded pages.

    Entities WITH URIs: derive Wikipedia title from DBpedia URI (exact match).
    Entities WITHOUT URIs (spaCy fallback): name-based lookup + DuckDuckGo.
    """
    raw_texts: dict[str, str] = {}
    seen_titles: set[str] = set()

    for entity in entities:
        # Derive title from URI (exact) or fall back to entity label
        if entity.uri and "/resource/" in entity.uri:
            title = uri_to_wiki_title(entity.uri)
        else:
            title = entity.label.replace("_", " ")

        if title.lower() in seen_titles:
            continue

        if verbose:
            src = entity.source or "unknown"
            print(f"    Fetching: '{title}' (source={src})")
        text = get_wikipedia_text_by_title(title, max_chars=50_000)
        if text:
            raw_texts[title] = text
            seen_titles.add(title.lower())
            if verbose:
                print(f"      -> {len(text)} chars")
        else:
            # DuckDuckGo fallback for pages not found by exact title
            if verbose:
                print(f"      -> not found, web searching: '{entity.label}'")
            results = web_search_wikipedia(entity.label, limit=2)
            if verbose:
                print(f"      -> web results: {results}")
            for st in results:
                if st.lower() not in seen_titles:
                    text = get_wikipedia_text_by_title(st, max_chars=50_000)
                    if text:
                        raw_texts[st] = text
                        seen_titles.add(st.lower())
                        if verbose:
                            print(f"      -> {st}: {len(text)} chars")
                        break

    # Web search for extra pages using question keywords
    q_words = [w for w in question.lower().split() if w not in STOPWORDS and len(w) > 2]
    search_query = " ".join(q_words[:6])
    if search_query:
        if verbose:
            print(f"    Web search: '{search_query}'")
        extra_titles = web_search_wikipedia(search_query, limit=3)
        if verbose:
            print(f"      -> web results: {extra_titles}")
        for st in extra_titles:
            if st.lower() not in seen_titles:
                text = get_wikipedia_text_by_title(st, max_chars=50_000)
                if text:
                    raw_texts[st] = text
                    seen_titles.add(st.lower())
                    if verbose:
                        print(f"      -> {st}: {len(text)} chars")

    # Fallback: if nothing found, search with full question
    if not raw_texts:
        fallback_query = question[:100]
        if verbose:
            print(f"    Fallback web search: '{fallback_query}'")
        fallback_titles = web_search_wikipedia(fallback_query, limit=3)
        if verbose:
            print(f"      -> web results: {fallback_titles}")
        for st in fallback_titles:
            if st.lower() not in seen_titles:
                text = get_wikipedia_text_by_title(st, max_chars=50_000)
                if text:
                    raw_texts[st] = text
                    seen_titles.add(st.lower())
                    if verbose:
                        print(f"      -> {st}: {len(text)} chars")

    return raw_texts


def retrieve_and_chunk_wikipedia(entities: list[Entity], question: str,
                                 cache=None,
                                 verbose: bool = False):
    """Fetch Wikipedia pages and chunk them for RAG retrieval.

    Reuses existing retrieve_wikipedia() for page fetching, then applies
    section-aware chunking. Checks cache before re-chunking.

    Args:
        entities: Entities to retrieve Wikipedia pages for.
        question: Question (passed through to retrieve_wikipedia).
        cache: Optional WikiRAGCache instance for chunk persistence.
        verbose: Print progress.

    Returns:
        List of WikiChunk objects from all retrieved pages.
    """
    from rag.chunker import WikiChunk, chunk_wikipedia_page

    raw_texts = retrieve_wikipedia(entities, question, verbose=verbose)
    if not raw_texts:
        return []

    all_chunks: list[WikiChunk] = []
    for title, text in raw_texts.items():
        if cache:
            cached = cache.get_chunks(title)
            if cached:
                if verbose:
                    print(f"    {title}: {len(cached)} chunks (cached)")
                all_chunks.extend(cached)
                continue
        chunks = chunk_wikipedia_page(title, text)
        if cache:
            cache.store_chunks(title, chunks)
        if verbose:
            print(f"    {title}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# 3b. Context assembly (shared with hybrid_qa.py)
# ---------------------------------------------------------------------------

def assemble_context(
    raw_texts: dict[str, str],
    question: str,
    seed_titles: set[str] | None = None,
    kg_facts: str = "",
    relation_hints: dict | None = None,
    verbose: bool = False,
) -> tuple[str, dict[str, str]]:
    """Extract relevant sections, rank pages, and assemble LLM context.

    Args:
        raw_texts: {page_title: full_wiki_text}
        question: The question to answer
        seed_titles: Titles of seed entity pages (get 60% budget priority)
        kg_facts: Formatted KG facts to prepend (counted against budget)
        relation_hints: {page_title: relation_name} for section boosting
        verbose: Print progress

    Returns:
        (context_string, filtered_texts_dict)
    """
    q_keywords = [
        w.strip("?.,!:;\"'") for w in question.lower().split()
        if w.strip("?.,!:;\"'") not in STOPWORDS
        and len(w.strip("?.,!:;\"'")) > 2
    ]

    # Priority budget: seed entities get 60%, bridge entities 30%.
    # Remaining 10% is implicit slack absorbed by the final MAX_CONTEXT_CHARS cap.
    if seed_titles:
        seed_pages = {t for t in raw_texts if t in seed_titles}
        n_seed = max(len(seed_pages), 1)
        n_bridge = max(len(raw_texts) - len(seed_pages), 1)

        seed_budget = int(MAX_CONTEXT_CHARS * 0.60)
        bridge_budget = int(MAX_CONTEXT_CHARS * 0.30)

        seed_per_page = max(MAX_CHARS_PER_PAGE, seed_budget // n_seed)
        bridge_per_page = max(1500, bridge_budget // n_bridge)
    else:
        seed_pages = set()
        seed_per_page = max(MAX_CHARS_PER_PAGE, MAX_CONTEXT_CHARS // max(len(raw_texts), 1))
        bridge_per_page = seed_per_page

    filtered_texts = {}
    for title, full_text in raw_texts.items():
        budget = seed_per_page if title in seed_pages else bridge_per_page
        hint = relation_hints.get(title) if relation_hints else None
        relevant = extract_relevant_sections(full_text, q_keywords, budget, hint)
        if relevant:
            filtered_texts[title] = relevant
        if verbose:
            tag = "SEED" if title in seed_pages else "bridge"
            print(f"    {title} [{tag}]: {len(full_text)} -> {len(relevant)} chars (budget={budget})")

    if verbose:
        print(f"\n  Total pages: {len(filtered_texts)}")

    def _page_kw_score(text: str) -> int:
        t_lower = text.lower()
        return sum(t_lower.count(kw) for kw in q_keywords)

    # Rank: seed pages first (sorted by keyword score), then bridge pages
    seed_ranked = sorted(
        [(t, txt) for t, txt in filtered_texts.items() if t in seed_pages],
        key=lambda item: _page_kw_score(item[1]), reverse=True,
    )
    bridge_ranked = sorted(
        [(t, txt) for t, txt in filtered_texts.items() if t not in seed_pages],
        key=lambda item: _page_kw_score(item[1]), reverse=True,
    )
    ranked_pages = seed_ranked + bridge_ranked

    # Reserve space for kg_facts (injected by prompt template, not here)
    context_parts = []
    total_chars = len(kg_facts) if kg_facts else 0

    for title, text in ranked_pages:
        entry = f"=== {title} ===\n{text}"
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                context_parts.append(entry[:remaining])
            break
        context_parts.append(entry)
        total_chars += len(entry)
    context = "\n\n".join(context_parts)

    return context, filtered_texts


# ---------------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------------

def answer_question(question: str, api_key: str = None,
                    verbose: bool = False,
                    provider: str = "groq",
                    model: str = "llama-3.3-70b-versatile") -> str:
    """Answer any question using DBpedia Spotlight + KG exploration + Wikipedia + LLM."""
    llm_client = create_llm_client(provider, api_key=api_key)
    linker = DBpediaEntityLinker(min_confidence=0.5)
    kg_client = DBpediaSPARQLClient()
    schema = SchemaDiscovery()

    # Step 1: Entity linking (Spotlight primary, spaCy fallback)
    entities = link_entities_to_dbpedia(question, linker)
    if verbose:
        print(f"\n{'=' * 70}")
        print("STEP 1: Entity linking (Spotlight + spaCy fallback)")
        print(f"{'=' * 70}")
        for e in entities:
            src = e.source or "unknown"
            print(f"  - {e.label}  [{src}]")
            if e.uri:
                print(f"    URI: {e.uri}")
        if not entities:
            print("  (none found, will use question-based search)")

    # Step 2: Schema-guided KG exploration
    if verbose:
        print(f"\n{'=' * 70}")
        print("STEP 2: Schema-guided KG exploration")
        print(f"{'=' * 70}")
    seed_count = len(entities)
    all_entities = explore_kg(entities, kg_client, schema, verbose=verbose)
    if verbose:
        discovered = len(all_entities) - seed_count
        print(f"  Seed: {seed_count}, Discovered: {discovered}, Total: {len(all_entities)}")

    # Step 3: Retrieve Wikipedia text
    if verbose:
        print(f"\n{'=' * 70}")
        print("STEP 3: Wikipedia text retrieval")
        print(f"{'=' * 70}")
    raw_texts = retrieve_wikipedia(all_entities, question, verbose=verbose)

    if not raw_texts:
        print("\nFailed: no Wikipedia text retrieved.")
        return "Could not find answer"

    # Extract relevant sections, rank, and assemble context (seed entities get priority)
    seed_titles = {
        uri_to_wiki_title(e.uri) for e in entities if e.uri and "/resource/" in e.uri
    }
    context, filtered_texts = assemble_context(
        raw_texts, question, seed_titles=seed_titles, verbose=verbose,
    )

    prompt = f"""Answer this question using ONLY the Wikipedia text below.

Question: {question}

Wikipedia text:
{context}

Instructions:
- Use ONLY facts from the text above
- Show brief step-by-step reasoning
- End with exactly: FINAL ANSWER: <answer>
"""

    if verbose:
        print(f"\n{'=' * 70}")
        print("STEP 4: LLM reasoning")
        print(f"{'=' * 70}")
        print(f"  Context: {len(context)} chars from {len(filtered_texts)} pages")
        print("  Sending to LLM...")

    response = call_llm(llm_client, prompt, model, max_tokens=500)

    if verbose and response:
        print("\n  LLM Response:\n")
        for line in response.split("\n"):
            print(f"    {line}")

    # Extract final answer
    answer = extract_final_answer(response) if response else "Could not generate answer"

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULT")
    print(f"{'=' * 70}")
    print(f"  Question: {question}")
    print(f"  Answer:   {answer}")
    print(f"  Pages:    {len(filtered_texts)} ({', '.join(filtered_texts.keys())})")
    print("  Method:   Spotlight -> KG explore -> Wikipedia text -> LLM")
    print()

    return answer


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG: Answer questions via DBpedia + Wikipedia + LLM"
    )
    parser.add_argument("question", type=str, help="The question to answer")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: provider-specific env var)")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="LLM model (default: llama-3.3-70b-versatile)")
    parser.add_argument("--provider", default="groq", choices=["groq", "github"],
                        help="LLM provider (default: groq)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show step-by-step pipeline output")
    args = parser.parse_args()

    try:
        answer_question(
            args.question, api_key=args.api_key, verbose=args.verbose,
            provider=args.provider, model=args.model,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
