#!/usr/bin/env python3
"""Debug script: show all intermediate results of the GraphRAG pipeline.

Usage:
  python3 graphrag_debug.py "Who destroys the last Horcrux of Voldemort?"
  python3 graphrag_debug.py "Who directed Inception?" --api-key gsk_...
"""

import os
import re
import sys
import argparse

from groq import Groq
from SPARQLWrapper import SPARQLWrapper, JSON

from graphrag_wiki import (
    extract_entities,
    retrieve_wikipedia,
    extract_relevant_sections,
    STOPWORDS,
    MAX_CONTEXT_CHARS,
)
from ablation_study_v3.wiki_utils import entity_to_uri, llm_call_with_retry


def wiki_url(title: str) -> str:
    """Build Wikipedia URL from page title."""
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG debug: show all intermediate results"
    )
    parser.add_argument("question", type=str, help="The question to answer")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GROQ_API_KEY"),
        help="Groq API key (or set GROQ_API_KEY env var)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set GROQ_API_KEY env var")
        sys.exit(1)

    question = args.question
    sep = "=" * 80

    # Step 1: Entity extraction
    entities = extract_entities(question)
    print(sep)
    print("STEP 1: EXTRACTED ENTITIES")
    print(sep)
    if entities:
        for e in entities:
            print(f"  - {e}")
    else:
        print("  (none found, will use question-based search)")

    # Step 2: DBpedia linking
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)
    linked = {name: entity_to_uri(name, sparql) for name in entities}
    print(f"\n{sep}")
    print("STEP 2: DBPEDIA LINKING")
    print(sep)
    if linked:
        for name, uri in linked.items():
            print(f"  {name}")
            print(f"    DBpedia: {uri}")
            print(f"    Wikipedia: {wiki_url(name)}")
    else:
        print("  (none)")

    # Step 3: Wikipedia retrieval
    print(f"\n{sep}")
    print("STEP 3: WIKIPEDIA PAGES RETRIEVED")
    print(sep)
    raw_texts = retrieve_wikipedia(entities, question, verbose=True)
    for title, text in raw_texts.items():
        print(f"\n  [{title}]")
        print(f"    URL:   {wiki_url(title)}")
        print(f"    Size:  {len(text)} chars")

    # Step 3b: Section extraction with dynamic per-page budget
    q_keywords = [
        w for w in question.lower().split()
        if w.strip("?.,!") not in STOPWORDS and len(w) > 2
    ]
    from graphrag_wiki import MAX_CHARS_PER_PAGE
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
            # Show first 120 chars of each paragraph
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
    client = Groq(api_key=args.api_key)
    response = llm_call_with_retry(client, prompt, max_tokens=500)
    print(response)

    if response and "FINAL ANSWER:" in response:
        answer = response.split("FINAL ANSWER:")[-1].strip()
        print(f"\n{sep}")
        print(f"EXTRACTED ANSWER: {answer}")
        print(sep)


if __name__ == "__main__":
    main()
