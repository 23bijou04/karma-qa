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

from dbpedia_qa.dbpedia import DBpediaEntityLinker, DBpediaSPARQLClient
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
# 1c. KG exploration: follow 1-hop relations to discover related entities
# ---------------------------------------------------------------------------

# Only follow structured DBpedia ontology/property relations
_USEFUL_RELATION_PREFIXES = (
    "http://dbpedia.org/ontology/",
    "http://dbpedia.org/property/",
)

# Skip these even within the ontology namespace (noisy/meta)
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


def explore_kg(entities: list[Entity], client: DBpediaSPARQLClient,
               limit_per_entity: int = 50,
               verbose: bool = False) -> list[Entity]:
    """Use DBpedia KG to discover related entities via structured properties.

    Only follows dbo:/dbp: ontology relations (director, birthPlace, etc.).
    Skips noisy wikiPageWikiLink edges — web search handles those.
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
        try:
            neighbors = client.get_neighbors(
                entity.uri, direction="out", limit=limit_per_entity,
            )
        except Exception as e:
            logger.warning(f"KG exploration failed for {entity.label}: {e}")
            continue

        for relation_uri, neighbor_uri, conf in neighbors:
            # Only follow structured ontology/property relations
            if not any(relation_uri.startswith(p) for p in _USEFUL_RELATION_PREFIXES):
                continue
            if relation_uri in _SKIP_RELATIONS:
                continue
            if neighbor_uri in seen_uris:
                continue
            if "/resource/" not in neighbor_uri:
                continue
            # Skip categories, templates, files
            local = neighbor_uri.split("/resource/")[-1]
            if any(local.startswith(p) for p in ("Category:", "Template:", "File:")):
                continue

            label = uri_to_wiki_title(neighbor_uri)
            all_entities.append(Entity(
                uri=neighbor_uri, label=label, source="kg_explore",
            ))
            seen_uris.add(neighbor_uri)

            if verbose:
                rel_name = relation_uri.split("/")[-1] if "/" in relation_uri else relation_uri
                print(f"      {entity.label} --{rel_name}--> {label}")

    return all_entities


# ---------------------------------------------------------------------------
# 2. Section extraction (from demo_graphrag_horcrux.py)
# ---------------------------------------------------------------------------

def extract_relevant_sections(text: str, keywords: list[str],
                              char_budget: int = MAX_CHARS_PER_PAGE) -> str:
    """Extract paragraphs ranked by keyword density.

    Keeps paragraphs with 2+ distinct keyword hits, or 1 hit if the keyword
    is rare across the full text (appears in <10% of paragraphs).
    Falls back to any-keyword-match if density filter returns nothing.
    """
    if not text:
        return ""

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
        if total + len(para) > char_budget:
            # Always include at least the first paragraph (truncated if needed)
            if not result:
                result.append(para[:char_budget])
            break
        result.append(para)
        total += len(para)

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
# 4. Main pipeline
# ---------------------------------------------------------------------------

def answer_question(question: str, api_key: str,
                    verbose: bool = False) -> str:
    """Answer any question using DBpedia Spotlight + KG exploration + Wikipedia + LLM."""
    llm_client = Groq(api_key=api_key)
    linker = DBpediaEntityLinker(min_confidence=0.5)
    kg_client = DBpediaSPARQLClient()

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

    # Step 2: KG exploration (follow 1-hop relations)
    if verbose:
        print(f"\n{'=' * 70}")
        print("STEP 2: KG exploration (1-hop neighbors)")
        print(f"{'=' * 70}")
    seed_count = len(entities)
    all_entities = explore_kg(entities, kg_client, verbose=verbose)
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

    # Extract relevant sections with dynamic per-page budget
    q_keywords = [w for w in question.lower().split()
                  if w.strip("?.,!") not in STOPWORDS and len(w) > 2]
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

    # Step 4: LLM reasoning
    context = "\n\n".join(
        f"=== {title} ===\n{text}" for title, text in filtered_texts.items()
    )
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n... [truncated]"

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
