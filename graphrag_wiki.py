#!/usr/bin/env python3
"""General-purpose GraphRAG: DBpedia Spotlight + KG exploration + Wikipedia text + LLM.

Pipeline:
  1. Link entities via DBpedia Spotlight (spaCy fallback)
  2. Explore KG: follow 1-hop relations to discover related entities
  3. Retrieve Wikipedia text for all entities (URI-based + web search)
  4. Extract relevant sections using question keywords
  5. LLM reasoning over assembled context (Groq)

Usage:
  python3 graphrag_wiki.py "Who destroys the last Horcrux of Voldemort?" --api-key $GROQ_API_KEY
  python3 graphrag_wiki.py "Who directed Inception?" --verbose
"""

import os
import re
import sys
import logging
import argparse
from pathlib import Path

from urllib.parse import unquote

# Ensure parent directory is on sys.path for dbpedia_qa package imports
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from groq import Groq
from ddgs import DDGS

from dbpedia_qa.dbpedia import DBpediaEntityLinker, DBpediaSPARQLClient, SchemaDiscovery
from dbpedia_qa.dbpedia.schema_discovery import CACHED_CLASS_SCHEMAS
from dbpedia_qa.models import Entity
from ablation_study_v3.wiki_utils import (
    get_wikipedia_text_by_title,
    llm_call_with_retry,
    uri_to_wiki_title,
)

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12_000
MAX_CHARS_PER_PAGE = 3_000

STOPWORDS = {
    "who", "what", "when", "where", "which", "how", "why",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "did", "does", "do", "has", "have", "had",
    "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "and", "or", "not", "that", "this", "it", "its",
    "first", "last", "same", "both", "more", "most",
    "after", "before", "between", "about", "into", "through",
}


# ---------------------------------------------------------------------------
# 1. Entity extraction (spaCy NLP)
# ---------------------------------------------------------------------------

import spacy

_nlp = spacy.load("en_core_web_sm")

# Words to strip from the front of noun chunks (determiners, question words)
_STRIP_WORDS = STOPWORDS | {
    "who", "what", "when", "where", "which", "how", "why",
    "did", "does", "do", "are", "is", "was", "were",
    "has", "have", "had", "both", "either",
}


def extract_entities(question: str) -> list[str]:
    """Extract entity mentions using spaCy noun chunks + NER.

    Handles both proper nouns (Albert Einstein) and lowercase domain
    terms (angiogenesis, genomic variants) that are valid Wikipedia entities.
    """
    doc = _nlp(question)
    mentions = []

    # Quoted strings (spaCy may miss these)
    mentions.extend(re.findall(r'"([^"]+)"', question))

    # spaCy noun chunks: the core extraction
    for chunk in doc.noun_chunks:
        # Strip leading determiners/stopwords: "the last Horcrux" -> "Horcrux"
        words = chunk.text.split()
        while words and words[0].lower() in _STRIP_WORDS:
            words.pop(0)
        text = " ".join(words).strip()
        if text:
            mentions.append(text)

    # spaCy NER: catch entities that noun chunks might miss
    for ent in doc.ents:
        if ent.label_ not in ("CARDINAL", "ORDINAL", "QUANTITY", "PERCENT"):
            mentions.append(ent.text)

    # Title-case for Wikipedia/DBpedia lookup (preserves already-capped names)
    mentions = [m.title() if m[0].islower() else m for m in mentions if m]

    # Deduplicate: case-insensitive
    seen = set()
    unique = []
    for name in mentions:
        key = name.lower().strip()
        if key not in seen and key not in STOPWORDS and len(key) > 1:
            seen.add(key)
            unique.append(name)

    # Remove entries that are substrings of longer entries
    final = []
    for name in unique:
        if not any(
            name.lower() != other.lower() and name.lower() in other.lower()
            for other in unique
        ):
            final.append(name)

    return final


# ---------------------------------------------------------------------------
# 1b. DBpedia Spotlight linking (primary) + spaCy fallback
# ---------------------------------------------------------------------------

def link_entities_to_dbpedia(question: str,
                             linker: DBpediaEntityLinker) -> list[Entity]:
    """Link question entities to DBpedia URIs using Spotlight.

    Primary: DBpedia Spotlight API (disambiguation + linking).
    Fallback: spaCy extraction → Entity objects without URIs.
    """
    try:
        entities = linker.link_entities(question)
        if entities:
            return entities
    except Exception as e:
        logger.warning(f"Spotlight failed: {e}")

    # Fallback: spaCy extraction → Entity objects without URIs
    mentions = extract_entities(question)
    return [Entity(uri="", label=name, source="spacy", confidence=0.0) for name in mentions]


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
                              char_budget: int = MAX_CHARS_PER_PAGE) -> str:
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
            scored.append((score, para))

    # Fallback: if density filter found nothing, use any single-keyword match
    if not scored:
        for para in paragraphs:
            para_lower = para.lower()
            hits = [kw for kw in keywords if kw.lower() in para_lower]
            if hits:
                scored.append((len(hits), para))

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

def web_search_wikipedia(query: str, limit: int = 5) -> list[str]:
    """Search DuckDuckGo for Wikipedia pages matching query."""
    try:
        results = list(DDGS().text(
            f"site:en.wikipedia.org {query}", max_results=limit,
        ))
        titles = []
        for r in results:
            url = r.get("href", "")
            if "en.wikipedia.org/wiki/" in url:
                title = unquote(url.split("/wiki/")[-1]).replace("_", " ")
                if ":" not in title:  # skip Category:, File:, etc.
                    titles.append(title)
        return titles
    except Exception:
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


# ---------------------------------------------------------------------------
# 3b. Context assembly (shared with hybrid_qa.py)
# ---------------------------------------------------------------------------

def assemble_context(
    raw_texts: dict[str, str],
    question: str,
    verbose: bool = False,
) -> tuple[str, dict[str, str]]:
    """Extract relevant sections, rank pages, and assemble LLM context.

    Args:
        raw_texts: {page_title: full_wiki_text}
        question: The question to answer
        verbose: Print progress

    Returns:
        (context_string, filtered_texts_dict)
    """
    q_keywords = [
        w.strip("?.,!:;\"'") for w in question.lower().split()
        if w.strip("?.,!:;\"'") not in STOPWORDS
        and len(w.strip("?.,!:;\"'")) > 2
    ]
    per_page = max(MAX_CHARS_PER_PAGE, MAX_CONTEXT_CHARS // max(len(raw_texts), 1))
    filtered_texts = {}
    for title, full_text in raw_texts.items():
        relevant = extract_relevant_sections(full_text, q_keywords, per_page)
        if relevant:
            filtered_texts[title] = relevant
        if verbose:
            print(f"    {title}: {len(full_text)} -> {len(relevant)} chars (filtered, budget={per_page})")

    if verbose:
        print(f"\n  Total pages: {len(filtered_texts)}")

    def _page_kw_score(text: str) -> int:
        t_lower = text.lower()
        return sum(t_lower.count(kw) for kw in q_keywords)

    ranked_pages = sorted(
        filtered_texts.items(),
        key=lambda item: _page_kw_score(item[1]),
        reverse=True,
    )
    context_parts = []
    total_chars = 0
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

def answer_question(question: str, api_key: str,
                    verbose: bool = False) -> str:
    """Answer any question using DBpedia Spotlight + KG exploration + Wikipedia + LLM."""
    llm_client = Groq(api_key=api_key)
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

    # Extract relevant sections, rank, and assemble context
    context, filtered_texts = assemble_context(raw_texts, question, verbose=verbose)

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

    response = llm_call_with_retry(llm_client, prompt, max_tokens=500)

    if verbose and response:
        print(f"\n  LLM Response:\n")
        for line in response.split("\n"):
            print(f"    {line}")

    # Extract final answer
    answer = response or "Could not generate answer"
    if "FINAL ANSWER:" in answer:
        answer = answer.split("FINAL ANSWER:")[-1].strip()

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULT")
    print(f"{'=' * 70}")
    print(f"  Question: {question}")
    print(f"  Answer:   {answer}")
    print(f"  Pages:    {len(filtered_texts)} ({', '.join(filtered_texts.keys())})")
    print(f"  Method:   Spotlight -> KG explore -> Wikipedia text -> LLM")
    print()

    return answer


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG: Answer questions via DBpedia + Wikipedia + LLM"
    )
    parser.add_argument("question", type=str, help="The question to answer")
    parser.add_argument("--api-key", default=os.environ.get("GROQ_API_KEY"),
                        help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show step-by-step pipeline output")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set GROQ_API_KEY env var")
        sys.exit(1)

    answer_question(args.question, args.api_key, verbose=args.verbose)


if __name__ == "__main__":
    main()
