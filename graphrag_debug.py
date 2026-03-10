#!/usr/bin/env python3
"""Debug script: show all intermediate results of the GraphRAG pipeline.

Usage:
  python3 graphrag_debug.py "Who destroys the last Horcrux of Voldemort?"
  python3 graphrag_debug.py "Who directed Inception?" --api-key $GROQ_API_KEY
  python3 graphrag_debug.py "Who directed Inception?" --provider github --model gpt-4o
"""

import argparse
import re
import sys
from pathlib import Path

# Ensure parent directory is on sys.path for dbpedia_qa package imports
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from dbpedia_qa.dbpedia import DBpediaEntityLinker, DBpediaSPARQLClient, SchemaDiscovery
from dbpedia_qa.models import extract_final_answer
from graphrag_wiki import (
    MAX_CHARS_PER_PAGE,
    MAX_CONTEXT_CHARS,
    STOPWORDS,
    explore_kg,
    extract_relevant_sections,
    link_entities_to_dbpedia,
    retrieve_wikipedia,
)


def wiki_url(title: str) -> str:
    """Build Wikipedia URL from page title."""
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG debug: show all intermediate results"
    )
    parser.add_argument("question", type=str, help="The question to answer")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: provider-specific env var)")
    parser.add_argument("--provider", default="groq", choices=["groq", "github"],
                        help="LLM provider (default: groq)")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="LLM model name")
    args = parser.parse_args()

    question = args.question
    sep = "=" * 80
    linker = DBpediaEntityLinker(min_confidence=0.5)
    kg_client = DBpediaSPARQLClient()
    schema = SchemaDiscovery()

    # Step 1: Entity linking (Spotlight + spaCy fallback)
    entities = link_entities_to_dbpedia(question, linker)
    print(sep)
    print("STEP 1: ENTITY LINKING (Spotlight primary, spaCy fallback)")
    print(sep)
    if entities:
        for e in entities:
            src = e.source or "unknown"
            print(f"  - {e.label}  [{src}, conf={e.confidence:.2f}]")
            if e.uri:
                print(f"    DBpedia: {e.uri}")
                print(f"    Wikipedia: {wiki_url(e.label)}")
    else:
        print("  (none found, will use question-based search)")

    # Step 2: KG exploration (1-hop neighbors)
    print(f"\n{sep}")
    print("STEP 2: KG EXPLORATION (1-hop neighbors)")
    print(sep)
    seed_count = len(entities)
    all_entities = explore_kg(entities, kg_client, schema, verbose=True)
    discovered = len(all_entities) - seed_count
    print(f"\n  Seed: {seed_count}, Discovered: {discovered}, Total: {len(all_entities)}")
    if discovered > 0:
        print("  Discovered entities:")
        for e in all_entities[seed_count:]:
            print(f"    - {e.label} ({e.uri})")

    # Step 3: Wikipedia retrieval
    print(f"\n{sep}")
    print("STEP 3: WIKIPEDIA PAGES RETRIEVED")
    print(sep)
    raw_texts = retrieve_wikipedia(all_entities, question, verbose=True)
    for title, text in raw_texts.items():
        print(f"\n  [{title}]")
        print(f"    URL:   {wiki_url(title)}")
        print(f"    Size:  {len(text)} chars")

    # Step 4: Section extraction with dynamic per-page budget
    q_keywords = [
        w for w in question.lower().split()
        if w.strip("?.,!") not in STOPWORDS and len(w) > 2
    ]
    per_page = max(MAX_CHARS_PER_PAGE, MAX_CONTEXT_CHARS // max(len(raw_texts), 1))
    print(f"\n{sep}")
    print(f"STEP 4: PARAGRAPH FILTERING (keywords: {q_keywords}, budget={per_page}/page)")
    print(sep)
    filtered_texts = {}
    for title, full_text in raw_texts.items():
        relevant = extract_relevant_sections(full_text, q_keywords, per_page)
        if relevant:
            filtered_texts[title] = relevant

        # Split into paragraphs for display
        paras = [p.strip() for p in re.split(r'\n\s*\n', relevant) if p.strip()]
        print(f"\n  [{title}] {len(full_text)} -> {len(relevant)} chars, {len(paras)} paragraphs")
        print(f"    URL: {wiki_url(title)}")
        for i, para in enumerate(paras, 1):
            preview = para[:120].replace('\n', ' ')
            if len(para) > 120:
                preview += "..."
            print(f"    P{i}: {preview}")

    # Step 5: Full prompt
    context = "\n\n".join(
        f"=== {t} ===\n{txt}" for t, txt in filtered_texts.items()
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

    print(f"\n{sep}")
    print(f"STEP 5: LLM PROMPT ({len(prompt)} chars, {len(filtered_texts)} pages)")
    print(sep)
    print(prompt)

    # Step 6: Full LLM response
    print(f"\n{sep}")
    print("STEP 6: LLM RESPONSE")
    print(sep)
    from llm_provider import call_llm, create_llm_client
    llm_client = create_llm_client(args.provider, api_key=args.api_key)
    response = call_llm(llm_client, prompt, args.model, max_tokens=500)
    print(response)

    if response:
        answer = extract_final_answer(response)
        if answer != response:
            print(f"\n{sep}")
            print(f"EXTRACTED ANSWER: {answer}")
            print(sep)


if __name__ == "__main__":
    main()
