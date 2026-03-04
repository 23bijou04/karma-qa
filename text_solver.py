"""Unified TEXT solver: Wikipedia retrieval + context assembly + LLM reasoning.

Single entry point `solve()` used by both the main text fallback in hybrid_qa.py
and per-sub-question TEXT resolution in logicrag_decomposition.py.
"""

import logging

from ablation_study_v3.wiki_utils import uri_to_wiki_title
from dbpedia_qa.models import Entity, extract_final_answer
from graphrag_wiki import assemble_context, retrieve_wikipedia
from sparql_evaluation import call_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TEXT_REASON_PROMPT = """Answer this question using the sources below.

Question: {question}

{kg_facts}Wikipedia text:
{context}

Instructions:
- Knowledge graph facts (if present) are verified structured data — trust them
- Use Wikipedia text to fill gaps not covered by KG facts
- Show brief step-by-step reasoning
- End with exactly: FINAL ANSWER: <answer>
"""

TEXT_SUBQ_PROMPT = """Answer this sub-question using the sources below.

Sub-question: {question}

{kg_facts}Wikipedia text:
{context}

Instructions:
- Knowledge graph facts (if present) are verified structured data -- trust them
- Use Wikipedia text to fill gaps not covered by KG facts
- Give a brief, factual answer
- End with exactly: FINAL ANSWER: <answer>
"""


def solve(
    question: str,
    entities: list[Entity],
    known_facts: str = "",
    relation_hints: dict[str, str] | None = None,
    llm_client=None,
    model: str = "",
    verbose: bool = False,
    is_subquestion: bool = False,
    seed_titles: set[str] | None = None,
) -> str | None:
    """Retrieve Wikipedia text, assemble context, LLM reasoning.

    Args:
        question: The question to answer.
        entities: Entities to retrieve Wikipedia pages for.
        known_facts: Pre-formatted KG facts or accumulated sub-Q answers.
        relation_hints: Title->relation map for section boosting.
        llm_client: Groq LLM client.
        model: LLM model name.
        verbose: Print pipeline steps.
        is_subquestion: Use sub-question prompt (shorter, more direct).
        seed_titles: Override seed titles for budget allocation.

    Returns:
        Answer string if successful, None if resolution fails.
    """
    raw_texts = retrieve_wikipedia(entities, question, verbose=verbose and not is_subquestion)
    if not raw_texts:
        if verbose:
            print("  No Wikipedia text retrieved.")
        return None

    # Build seed titles if not provided
    if seed_titles is None:
        seed_titles = {
            uri_to_wiki_title(e.uri) for e in entities
            if e.uri and "/resource/" in e.uri
        }

    context, _ = assemble_context(
        raw_texts, question, seed_titles=seed_titles,
        kg_facts=known_facts, relation_hints=relation_hints or {},
        verbose=verbose and not is_subquestion,
    )

    # Choose prompt
    if is_subquestion:
        prompt = TEXT_SUBQ_PROMPT.format(
            question=question, context=context, kg_facts=known_facts,
        )
    else:
        prompt = TEXT_REASON_PROMPT.format(
            question=question, context=context, kg_facts=known_facts,
        )

    response = call_llm(llm_client, prompt, model)
    answer = extract_final_answer(response) if response else None

    if verbose and answer:
        print(f"  TEXT answer: {answer}")

    return answer
