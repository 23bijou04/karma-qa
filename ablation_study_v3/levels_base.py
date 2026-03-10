"""Level 0 (No KG), Level 1 (Entity-Only KG), Level 4 (Oracle) + LevelResult."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from SPARQLWrapper import SPARQLWrapper

from .wiki_utils import (
    compare_with_llm,
    entity_to_uri,
    extract_answer_with_llm,
    get_wikipedia_text,
    get_wikipedia_text_by_title,
    llm_call_with_retry,
    uri_to_wiki_title,
)

logger = logging.getLogger(__name__)


@dataclass
class LevelResult:
    """Standardized result from any ablation level."""
    question_id: str
    question_type: str
    gold_answer: str
    predicted_answer: str = ""
    answer_source: str = "none"  # kg, wikipedia, oracle, none
    em: float = 0.0
    f1: float = 0.0
    llm_calls: int = 0
    wiki_calls: int = 0
    sparql_calls: int = 0
    latency_seconds: float = 0.0
    error: str = ""
    metadata: dict = field(default_factory=dict)


def _get_all_entity_names(evidences: list) -> list[str]:
    """Extract unique entity names from evidence triples."""
    names = []
    seen = set()
    for ev in evidences:
        name = ev[0]
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def answer_level_0(question: dict, client: OpenAI, **kwargs) -> LevelResult:
    """Level 0: No KG. Pure Wikipedia search + LLM answer."""
    start = time.time()
    result = LevelResult(
        question_id=question["_id"],
        question_type=question["type"],
        gold_answer=question["answer"],
    )

    evidences = question.get("evidences", [])
    entity_names = _get_all_entity_names(evidences)

    # Retrieve Wikipedia text via search (no entity linking)
    wiki_texts = []
    for name in entity_names:
        text = get_wikipedia_text(name, max_chars=3000)
        result.wiki_calls += 1
        if text:
            wiki_texts.append(f"--- {name} ---\n{text}")

    if not wiki_texts:
        result.error = "No Wikipedia text found"
        result.latency_seconds = time.time() - start
        return result

    context = "\n\n".join(wiki_texts)
    prompt = f"""Answer the following question using ONLY the provided text.

Question: {question['question']}

Text:
{context}

Answer with ONLY the answer (a name, date, yes/no, or short phrase).
If you cannot determine the answer, say "UNKNOWN".

Answer:"""

    try:
        answer = llm_call_with_retry(client, prompt)
        if answer:
            result.predicted_answer = answer
            result.llm_calls = 1
            result.answer_source = "wikipedia"
    except Exception as e:
        result.error = str(e)

    result.latency_seconds = time.time() - start
    return result


def _answer_via_wiki_extraction(
    question: dict, client: OpenAI, entity_texts: dict[str, Optional[str]],
) -> tuple[Optional[str], int]:
    """Answer a question by extracting facts from Wikipedia texts.

    Routes by question type. Returns (answer, llm_calls).
    """
    qtype = question["type"]
    evidences = question["evidences"]
    q_text = question["question"]
    llm_calls = 0

    if qtype == "comparison":
        entity1, prop1, _ = evidences[0]
        entity2, prop2, _ = evidences[1]
        text1 = entity_texts.get(entity1)
        text2 = entity_texts.get(entity2)

        val1 = extract_answer_with_llm(client, text1, prop1, q_text, entity1)
        val2 = extract_answer_with_llm(client, text2, prop2, q_text, entity2)
        llm_calls += 2

        if val1 and val2:
            answer = compare_with_llm(client, q_text, entity1, val1, entity2, val2)
            llm_calls += 1
            return answer, llm_calls
        return None, llm_calls

    elif qtype == "bridge_comparison":
        entity1, bridge_rel1, _ = evidences[0]
        entity2, bridge_rel2, _ = evidences[1]
        _, compare_rel1, _ = evidences[2]
        _, compare_rel2, _ = evidences[3]

        text1 = entity_texts.get(entity1)
        text2 = entity_texts.get(entity2)

        bridge1 = extract_answer_with_llm(
            client, text1, bridge_rel1,
            f"Who is the {bridge_rel1} of {entity1}?", entity1,
        )
        bridge2 = extract_answer_with_llm(
            client, text2, bridge_rel2,
            f"Who is the {bridge_rel2} of {entity2}?", entity2,
        )
        llm_calls += 2

        if bridge1 and bridge2:
            text_b1 = get_wikipedia_text(bridge1)
            text_b2 = get_wikipedia_text(bridge2)

            cmp1 = extract_answer_with_llm(
                client, text_b1, compare_rel1,
                f"What is the {compare_rel1} of {bridge1}?", bridge1,
            )
            cmp2 = extract_answer_with_llm(
                client, text_b2, compare_rel2,
                f"What is the {compare_rel2} of {bridge2}?", bridge2,
            )
            llm_calls += 2

            if cmp1 and cmp2:
                answer = compare_with_llm(client, q_text, entity1, cmp1, entity2, cmp2)
                llm_calls += 1
                return answer, llm_calls

        return None, llm_calls

    else:  # compositional or inference
        entity1, rel1, _ = evidences[0]
        _, rel2, _ = evidences[1]

        text1 = entity_texts.get(entity1)
        context = "film" if "director" in rel1.lower() else rel1

        intermediate = extract_answer_with_llm(
            client, text1, rel1,
            f"Who/what is the {rel1} of {entity1}?", entity1,
        )
        llm_calls += 1

        if intermediate:
            text2 = get_wikipedia_text(intermediate, context=context)
            answer = extract_answer_with_llm(
                client, text2, rel2,
                f"What is the {rel2} of {intermediate}?", intermediate,
            )
            llm_calls += 1
            return answer, llm_calls

        return None, llm_calls


def answer_level_1(
    question: dict, client: OpenAI, sparql: Optional[SPARQLWrapper] = None, **kwargs,
) -> LevelResult:
    """Level 1: Entity-Only KG. DBpedia for entity linking, Wikipedia for answers."""
    start = time.time()
    result = LevelResult(
        question_id=question["_id"],
        question_type=question["type"],
        gold_answer=question["answer"],
    )

    evidences = question.get("evidences", [])
    entity_names = _get_all_entity_names(evidences)

    # Entity linking via DBpedia (the only KG operation)
    entity_texts: dict[str, Optional[str]] = {}
    for name in entity_names:
        uri = entity_to_uri(name, sparql=sparql)
        result.sparql_calls += 1
        wiki_title = uri_to_wiki_title(uri)
        text = get_wikipedia_text_by_title(wiki_title, max_chars=3000)
        result.wiki_calls += 1

        # Fallback to search if exact title fails
        if not text:
            text = get_wikipedia_text(name, max_chars=3000)
            result.wiki_calls += 1

        entity_texts[name] = text
        result.metadata[f"uri_{name}"] = uri

    answer, llm_calls = _answer_via_wiki_extraction(question, client, entity_texts)
    result.llm_calls = llm_calls

    if answer:
        result.predicted_answer = answer
        result.answer_source = "wikipedia"
    else:
        result.error = "Could not extract answer from Wikipedia text"

    result.latency_seconds = time.time() - start
    return result


def answer_level_4(question: dict, client: OpenAI, **kwargs) -> LevelResult:
    """Level 4: Oracle KG. Use ground-truth evidence triples directly."""
    start = time.time()
    result = LevelResult(
        question_id=question["_id"],
        question_type=question["type"],
        gold_answer=question["answer"],
        answer_source="oracle",
    )

    evidences = question.get("evidences", [])
    qtype = question["type"]

    if qtype in ("compositional", "inference"):
        # Last evidence value is the answer
        result.predicted_answer = evidences[-1][2]
        result.latency_seconds = time.time() - start
        return result

    # Comparison / bridge_comparison: need LLM to reason over oracle values
    facts = "\n".join(f"- {ev[0]}'s {ev[1]} is {ev[2]}" for ev in evidences)
    prompt = f"""Answer this question using ONLY the provided facts.

Question: {question['question']}

Known facts:
{facts}

Return ONLY the answer (entity name, "yes", or "no").

Answer:"""

    try:
        answer = llm_call_with_retry(client, prompt)
        if answer:
            result.predicted_answer = answer
            result.llm_calls = 1
    except Exception as e:
        result.error = str(e)

    result.latency_seconds = time.time() - start
    return result
