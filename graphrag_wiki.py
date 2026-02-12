#!/usr/bin/env python3
"""General-purpose GraphRAG: DBpedia entity linking + Wikipedia text + LLM.

Pipeline:
  1. Extract entities from question (pattern-based)
  2. Link entities to DBpedia URIs (verify existence)
  3. Retrieve Wikipedia full text for each entity + search for extras
  4. Extract relevant sections using question keywords
  5. LLM reasoning over assembled context (Groq)

Usage:
  python3 graphrag_wiki.py "Who destroys the last Horcrux of Voldemort?" --api-key $GROQ_API_KEY
  python3 graphrag_wiki.py "Who directed Inception?" --verbose
"""

import os
import re
import sys
import argparse

from urllib.parse import unquote

from groq import Groq
from ddgs import DDGS
from SPARQLWrapper import SPARQLWrapper, JSON

from ablation_study_v3.wiki_utils import (
    entity_to_uri,
    get_wikipedia_text_by_title,
    llm_call_with_retry,
)

DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
MAX_CONTEXT_CHARS = 12_000
MAX_CHARS_PER_PAGE = 3_000

STOPWORDS = {
    "who", "what", "when", "where", "which", "how", "why",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "did", "does", "do", "has", "have", "had",
    "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "and", "or", "not", "that", "this", "it", "its",
    "first", "last", "same", "both", "more", "most",
}


# ---------------------------------------------------------------------------
# 1. Entity extraction
# ---------------------------------------------------------------------------

QUESTION_WORDS = {
    "who", "what", "when", "where", "which", "how", "why",
    "did", "does", "do", "are", "is", "was", "were",
    "has", "have", "had", "both", "either",
}


def extract_entities(question: str) -> list[str]:
    """Extract entity mentions from a question using patterns."""
    mentions = []

    # Quoted strings
    mentions.extend(re.findall(r'"([^"]+)"', question))

    # Capitalized phrases: "Albert Einstein", "The Dark Knight", "Napoleon"
    # Supports: uppercase words, hyphenated (Hewlett-Packard), Roman numerals (II, III)
    # Connectors: "the", "de", "von", "van" (name particles only, NOT "and"/"of")
    cap_pattern = r'\b([A-Z][a-z]+(?:[-\s]+(?:[A-Z][a-z]+|[A-Z]{2,3}|the|de|von|van))*)\b'
    for m in re.findall(cap_pattern, question):
        mentions.append(m)

    # "X or/and Y" — only capture capitalized name portions
    m = re.search(
        r'([A-Z][a-z]+(?:[\s-]+[A-Z][a-z]+)*)\s+(?:or|and)\s+'
        r'([A-Z][a-z]+(?:[\s-]+[A-Z][a-z]+)*)',
        question,
    )
    if m:
        mentions.extend([m.group(1).strip(), m.group(2).strip()])

    # "of [Entity]" — capture until next preposition, punctuation, or lowercase word
    for m in re.findall(
        r'of\s+((?:[A-Z][a-z]+(?:[-\s]+(?:[A-Z][a-z]+|[A-Z]{2,3}))*))(?:\?|,|\.|$|\s+(?:go|from|who|that|which|born))',
        question,
    ):
        mentions.append(m.strip())

    # Lowercase fallback: known patterns like "the capital of france"
    for m in re.findall(r'of\s+(?:the\s+)?([a-z]{3,}(?:\s+[a-z]+)*?)(?:\?|$)', question):
        # Only keep if it looks like a proper noun used lowercase
        mentions.append(m.strip().title())

    # Deduplicate: case-insensitive, remove question words, remove substrings
    seen = set()
    unique = []
    for name in mentions:
        # Strip leading question words: "Did Albert Einstein" -> "Albert Einstein"
        words = name.split()
        while words and words[0].lower() in QUESTION_WORDS:
            words.pop(0)
        name = " ".join(words).strip()

        key = name.lower()
        if key not in seen and key not in STOPWORDS and len(key) > 1:
            seen.add(key)
            unique.append(name)

    # Remove entries that are substrings of longer entries
    # e.g. "Queen Elizabeth" is substring of "Queen Elizabeth II"
    final = []
    for name in unique:
        if not any(
            name.lower() != other.lower() and name.lower() in other.lower()
            for other in unique
        ):
            final.append(name)

    return final


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

def retrieve_wikipedia(entities: list[str], question: str,
                       verbose: bool = False) -> dict[str, str]:
    """Retrieve Wikipedia text for entities + web-search-expanded pages."""
    raw_texts: dict[str, str] = {}
    seen_titles: set[str] = set()

    # Fetch pages for each entity (direct Wikipedia lookup)
    for name in entities:
        title = name.replace("_", " ")
        if title.lower() in seen_titles:
            continue

        if verbose:
            print(f"    Fetching page: '{title}'")
        text = get_wikipedia_text_by_title(title, max_chars=50_000)
        if text:
            raw_texts[title] = text
            seen_titles.add(title.lower())
            if verbose:
                print(f"      -> {len(text)} chars (direct)")
        else:
            # Search fallback via DuckDuckGo
            if verbose:
                print(f"      -> not found, web searching: '{name}'")
            results = web_search_wikipedia(name, limit=2)
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
    """Answer any question using DBpedia entity linking + Wikipedia + LLM."""
    client = Groq(api_key=api_key)
    sparql = SPARQLWrapper(DBPEDIA_ENDPOINT)
    sparql.setReturnFormat(JSON)

    # Step 1: Extract entities
    entities = extract_entities(question)
    if verbose:
        print(f"\n{'=' * 70}")
        print("STEP 1: Entity extraction")
        print(f"{'=' * 70}")
        for e in entities:
            print(f"  - {e}")
        if not entities:
            print("  (none found, will use question-based search)")

    # Step 2: Link to DBpedia
    linked = {}
    for name in entities:
        uri = entity_to_uri(name, sparql)
        linked[name] = uri
    if verbose:
        print(f"\n{'=' * 70}")
        print("STEP 2: DBpedia entity linking")
        print(f"{'=' * 70}")
        for name, uri in linked.items():
            print(f"  {name}")
            print(f"    URI: {uri}")

    # Step 3: Retrieve Wikipedia text
    if verbose:
        print(f"\n{'=' * 70}")
        print("STEP 3: Wikipedia text retrieval")
        print(f"{'=' * 70}")
    raw_texts = retrieve_wikipedia(entities, question, verbose=verbose)

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

    response = llm_call_with_retry(client, prompt, max_tokens=500)

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
    print(f"  Method:   DBpedia entity linking -> Wikipedia text -> LLM")
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
