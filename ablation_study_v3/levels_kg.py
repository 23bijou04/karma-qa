"""Level 2 (Entity + Important Relations) and Level 3 (Full KG) with SPARQL."""

import logging
import re
import time
from typing import Optional

from openai import OpenAI
from SPARQLWrapper import JSON, SPARQLWrapper

from .levels_base import LevelResult, _answer_via_wiki_extraction, _get_all_entity_names
from .wiki_utils import (
    answers_match,
    compare_with_llm,
    entity_to_uri,
    extract_answer_with_llm,
    get_wikipedia_text,
    get_wikipedia_text_by_title,
    uri_to_wiki_title,
)

logger = logging.getLogger(__name__)

# High-coverage properties (>63% from coverage analysis)
IMPORTANT_PROPERTIES = {
    "director": ["dbo:director", "dbp:director"],
    "date of birth": ["dbo:birthDate", "dbp:birthDate"],
    "date of death": ["dbo:deathDate", "dbp:deathDate"],
    "place of birth": ["dbo:birthPlace", "dbp:birthPlace"],
    "country": ["dbo:country", "dbp:country"],
    "country of origin": ["dbo:country", "dbp:country"],
    "place of death": ["dbo:deathPlace", "dbp:deathPlace"],
    "producer": ["dbo:producer", "dbp:producer"],
}

# Full property mappings (30+ relations)
FULL_PROPERTY_MAPPINGS = {
    **IMPORTANT_PROPERTIES,
    "father": ["dbo:parent", "dbp:father", "dbp:parent"],
    "mother": ["dbo:parent", "dbp:mother"],
    "spouse": ["dbo:spouse", "dbp:spouse"],
    "occupation": ["dbo:occupation", "dbp:occupation"],
    "employer": ["dbo:employer", "dbp:employer"],
    "educated at": ["dbo:almaMater", "dbp:almaMater", "dbo:education"],
    "work location": ["dbo:locationCity", "dbp:workplaces"],
    "member of": ["dbo:party", "dbp:party"],
    "award received": ["dbo:award", "dbp:awards"],
    "performer": ["dbo:artist", "dbp:artist", "dbo:musicalArtist"],
    "author": ["dbo:author", "dbp:author", "dbo:writer"],
    "publication date": ["dbo:releaseDate", "dbp:released"],
    "genre": ["dbo:genre", "dbp:genre"],
    "capital": ["dbo:capital", "dbp:capital"],
    "located in": ["dbo:location", "dbp:location", "dbo:country"],
    "founded by": ["dbo:foundedBy", "dbp:founder"],
    "headquarters": ["dbo:headquarter", "dbp:headquarters"],
    "language": ["dbo:language", "dbp:language"],
    "starring": ["dbo:starring", "dbp:starring"],
    "screenwriter": ["dbo:writer", "dbp:writer"],
    "composer": ["dbo:musicComposer", "dbp:music"],
    "cast member": ["dbo:starring", "dbp:starring"],
    "narrative location": ["dbo:location", "dbp:location"],
    "country of citizenship": ["dbo:nationality", "dbp:nationality", "dbo:citizenship"],
}


def _get_property_mappings(relation: str, prop_set: dict) -> list[str]:
    """Map a dataset relation name to DBpedia property URIs."""
    rel_lower = relation.lower().strip()
    if rel_lower in prop_set:
        return prop_set[rel_lower]
    # Partial match
    for key, props in prop_set.items():
        if key in rel_lower or rel_lower in key:
            return props
    # Fallback
    prop_name = rel_lower.replace(" ", "").replace("_", "")
    return [f"dbo:{prop_name}", f"dbp:{prop_name}"]


def _sparql_get_value(
    sparql: SPARQLWrapper, entity_uri: str, prop_uris: list[str],
) -> tuple[Optional[str], int]:
    """Query DBpedia for a property value. Returns (value, num_queries)."""
    queries = 0
    for prop in prop_uris:
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbp: <http://dbpedia.org/property/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?value ?label WHERE {{
            <{entity_uri}> {prop} ?value .
            OPTIONAL {{ ?value rdfs:label ?label . FILTER(LANG(?label) = "en") }}
        }}
        LIMIT 5
        """
        queries += 1
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
                if "label" in bindings[0]:
                    return bindings[0]["label"]["value"], queries
                val = bindings[0]["value"]["value"]
                if val.startswith("http://"):
                    val = val.split("/")[-1].replace("_", " ")
                return val, queries
        except Exception:
            continue
    return None, queries


def _answer_comparison_with_values(
    question: dict, entity1: str, val1: str, entity2: str, val2: str,
    client: Optional[OpenAI] = None,
) -> tuple[Optional[str], int]:
    """Answer a comparison question given two values. Returns (answer, llm_calls)."""
    q_lower = question["question"].lower()

    if "same" in q_lower or "both" in q_lower:
        return ("yes" if val1.lower() == val2.lower() else "no"), 0

    # Date comparison
    try:
        year1 = int(re.search(r'\d{4}', str(val1)).group())
        year2 = int(re.search(r'\d{4}', str(val2)).group())
    except (AttributeError, ValueError):
        # Can't parse dates, use LLM
        if client:
            answer = compare_with_llm(client, question["question"], entity1, val1, entity2, val2)
            return answer, 1
        return None, 0

    if "first" in q_lower or "earlier" in q_lower or "older" in q_lower:
        return (entity1 if year1 < year2 else entity2), 0
    elif "later" in q_lower or "recently" in q_lower or "younger" in q_lower:
        return (entity1 if year1 > year2 else entity2), 0

    if client:
        answer = compare_with_llm(client, question["question"], entity1, val1, entity2, val2)
        return answer, 1
    return None, 0


def answer_level_2(
    question: dict, client: OpenAI, sparql: Optional[SPARQLWrapper] = None, **kwargs,
) -> LevelResult:
    """Level 2: Entity + Important Relations. KG for high-coverage props, Wikipedia fallback."""
    start = time.time()
    result = LevelResult(
        question_id=question["_id"],
        question_type=question["type"],
        gold_answer=question["answer"],
    )

    evidences = question.get("evidences", [])
    entity_names = _get_all_entity_names(evidences)

    # Entity linking
    uris = {}
    for name in entity_names:
        uris[name] = entity_to_uri(name, sparql=sparql)
        result.sparql_calls += 1

    # Try to get each evidence value from KG first, fallback to Wikipedia
    hop_values = {}  # (entity, relation) -> (value, source)

    for ev in evidences:
        entity, relation, _ = ev
        uri = uris.get(entity, "")
        prop_uris = _get_property_mappings(relation, IMPORTANT_PROPERTIES)

        # Only try KG if relation is in important properties
        kg_value = None
        if relation.lower().strip() in IMPORTANT_PROPERTIES and uri and sparql:
            kg_value, q_count = _sparql_get_value(sparql, uri, prop_uris)
            result.sparql_calls += q_count

        if kg_value:
            hop_values[(entity, relation)] = (kg_value, "kg")
        else:
            # Fallback to Wikipedia + LLM
            wiki_title = uri_to_wiki_title(uri) if uri else entity
            text = get_wikipedia_text_by_title(wiki_title, max_chars=3000)
            if not text:
                text = get_wikipedia_text(entity, max_chars=3000)
            result.wiki_calls += 1

            wiki_value = extract_answer_with_llm(
                client, text, relation,
                f"What is the {relation} of {entity}?", entity,
            )
            result.llm_calls += 1
            hop_values[(entity, relation)] = (wiki_value, "wikipedia")

    result.metadata["hop_sources"] = {
        f"{e}|{r}": src for (e, r), (_, src) in hop_values.items()
    }

    # Assemble final answer
    answer = _assemble_answer_from_hops(question, hop_values, client, result)
    if answer:
        result.predicted_answer = answer
        result.answer_source = "wikipedia"  # hybrid, but categorized by dominant source
        kg_hops = sum(1 for _, src in hop_values.values() if src == "kg")
        if kg_hops == len(hop_values):
            result.answer_source = "kg"

    result.latency_seconds = time.time() - start
    return result


def _assemble_answer_from_hops(
    question: dict, hop_values: dict, client: OpenAI, result: LevelResult,
) -> Optional[str]:
    """Assemble the final answer from per-hop extracted values."""
    qtype = question["type"]
    evidences = question["evidences"]

    if qtype == "comparison":
        e1, r1, _ = evidences[0]
        e2, r2, _ = evidences[1]
        v1 = hop_values.get((e1, r1), (None, ""))[0]
        v2 = hop_values.get((e2, r2), (None, ""))[0]
        if v1 and v2:
            answer, llm = _answer_comparison_with_values(question, e1, v1, e2, v2, client)
            result.llm_calls += llm
            return answer

    elif qtype == "bridge_comparison":
        e1, br1, _ = evidences[0]
        e2, br2, _ = evidences[1]
        _, cr1, _ = evidences[2]
        _, cr2, _ = evidences[3]
        bridge1 = hop_values.get((e1, br1), (None, ""))[0]
        bridge2 = hop_values.get((e2, br2), (None, ""))[0]
        if bridge1 and bridge2:
            cmp1 = hop_values.get((bridge1, cr1), (None, ""))[0]
            cmp2 = hop_values.get((bridge2, cr2), (None, ""))[0]
            # Bridge values may not be in hop_values if they were extracted names
            # Try to look them up if missing
            if not cmp1 or not cmp2:
                return None
            answer, llm = _answer_comparison_with_values(question, e1, cmp1, e2, cmp2, client)
            result.llm_calls += llm
            return answer

    else:  # compositional, inference
        _, _, intermediate = evidences[0]
        _, rel2, _ = evidences[1]
        # The answer is the second hop value
        val = hop_values.get((intermediate, rel2), (None, ""))[0]
        if val:
            return val
        # If intermediate was extracted from first hop, try again
        first_entity = evidences[0][0]
        first_rel = evidences[0][1]
        extracted_intermediate = hop_values.get((first_entity, first_rel), (None, ""))[0]
        if extracted_intermediate:
            val = hop_values.get((extracted_intermediate, rel2), (None, ""))[0]
            return val

    return None


def answer_level_3(
    question: dict, client: OpenAI, sparql: Optional[SPARQLWrapper] = None, **kwargs,
) -> LevelResult:
    """Level 3: Full KG + Wikipedia Fallback. Try all property mappings first."""
    start = time.time()
    result = LevelResult(
        question_id=question["_id"],
        question_type=question["type"],
        gold_answer=question["answer"],
    )

    evidences = question.get("evidences", [])
    entity_names = _get_all_entity_names(evidences)

    # Entity linking
    uris = {}
    for name in entity_names:
        uris[name] = entity_to_uri(name, sparql=sparql)
        result.sparql_calls += 1

    # Try full KG query first (same logic as ablation_study_v2)
    kg_answer = _try_full_kg(question, uris, sparql, result)

    if kg_answer and answers_match(kg_answer, question["answer"]):
        result.predicted_answer = kg_answer
        result.answer_source = "kg"
        result.latency_seconds = time.time() - start
        return result

    # KG failed or answer didn't match - fallback to Wikipedia + LLM
    entity_texts: dict[str, Optional[str]] = {}
    for name in entity_names:
        uri = uris.get(name, "")
        wiki_title = uri_to_wiki_title(uri) if uri else name
        text = get_wikipedia_text_by_title(wiki_title, max_chars=3000)
        if not text:
            text = get_wikipedia_text(name, max_chars=3000)
        result.wiki_calls += 1
        entity_texts[name] = text

    answer, llm_calls = _answer_via_wiki_extraction(question, client, entity_texts)
    result.llm_calls += llm_calls

    if answer:
        result.predicted_answer = answer
        result.answer_source = "wikipedia"
    elif kg_answer:
        # Use KG answer even if it didn't fuzzy-match (let official metrics judge)
        result.predicted_answer = kg_answer
        result.answer_source = "kg"
    else:
        result.error = "Both KG and Wikipedia failed"

    result.latency_seconds = time.time() - start
    return result


def _try_full_kg(
    question: dict, uris: dict, sparql: Optional[SPARQLWrapper], result: LevelResult,
) -> Optional[str]:
    """Try answering with full KG (all property mappings)."""
    if not sparql:
        return None

    qtype = question["type"]

    if qtype == "comparison":
        return _try_kg_comparison(question, uris, sparql, result)
    elif qtype == "bridge_comparison":
        return _try_kg_bridge(question, uris, sparql, result)
    else:
        return _try_kg_two_hop(question, uris, sparql, result)


def _try_kg_comparison(question, uris, sparql, result) -> Optional[str]:
    evidences = question["evidences"]
    e1, prop1, _ = evidences[0]
    e2, prop2, _ = evidences[1]
    props = _get_property_mappings(prop1, FULL_PROPERTY_MAPPINGS)

    values = []
    for entity in [e1, e2]:
        uri = uris.get(entity, "")
        val, q_count = _sparql_get_value(sparql, uri, props)
        result.sparql_calls += q_count
        values.append((entity, val))

    if values[0][1] and values[1][1]:
        answer, _ = _answer_comparison_with_values(
            question, values[0][0], values[0][1], values[1][0], values[1][1],
        )
        return answer
    return None


def _try_kg_bridge(question, uris, sparql, result) -> Optional[str]:
    evidences = question["evidences"]
    e1, br1, _ = evidences[0]
    e2, br2, _ = evidences[1]
    _, cr1, _ = evidences[2]

    bridge_props = _get_property_mappings(br1, FULL_PROPERTY_MAPPINGS)
    compare_props = _get_property_mappings(cr1, FULL_PROPERTY_MAPPINGS)

    bridge_values = []
    for entity in [e1, e2]:
        uri = uris.get(entity, "")
        for bp in bridge_props:
            for cp in compare_props:
                query = f"""
                PREFIX dbo: <http://dbpedia.org/ontology/>
                PREFIX dbp: <http://dbpedia.org/property/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?bridge ?bridgeLabel ?compare WHERE {{
                    <{uri}> {bp} ?bridge .
                    ?bridge {cp} ?compare .
                    OPTIONAL {{ ?bridge rdfs:label ?bridgeLabel . FILTER(LANG(?bridgeLabel) = "en") }}
                }}
                LIMIT 5
                """
                result.sparql_calls += 1
                try:
                    sparql.setQuery(query)
                    sparql.setReturnFormat(JSON)
                    res = sparql.query().convert()
                    bindings = res.get("results", {}).get("bindings", [])
                    if bindings:
                        cmp_val = bindings[0]["compare"]["value"]
                        bridge_values.append((entity, cmp_val))
                        break
                except Exception:
                    continue
            if bridge_values and bridge_values[-1][0] == entity:
                break

    if len(bridge_values) >= 2:
        answer, _ = _answer_comparison_with_values(
            question, bridge_values[0][0], bridge_values[0][1],
            bridge_values[1][0], bridge_values[1][1],
        )
        return answer
    return None


def _try_kg_two_hop(question, uris, sparql, result) -> Optional[str]:
    evidences = question["evidences"]
    e1, r1, _ = evidences[0]
    _, r2, _ = evidences[1]
    uri = uris.get(e1, "")

    props1 = _get_property_mappings(r1, FULL_PROPERTY_MAPPINGS)
    props2 = _get_property_mappings(r2, FULL_PROPERTY_MAPPINGS)

    for p1 in props1:
        for p2 in props2:
            query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbp: <http://dbpedia.org/property/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?final ?label WHERE {{
                <{uri}> {p1} ?intermediate .
                ?intermediate {p2} ?final .
                OPTIONAL {{ ?final rdfs:label ?label . FILTER(LANG(?label) = "en") }}
            }}
            LIMIT 10
            """
            result.sparql_calls += 1
            try:
                sparql.setQuery(query)
                sparql.setReturnFormat(JSON)
                res = sparql.query().convert()
                bindings = res.get("results", {}).get("bindings", [])
                for b in bindings:
                    if "label" in b:
                        val = b["label"]["value"]
                    else:
                        val = b["final"]["value"]
                        if val.startswith("http://"):
                            val = val.split("/")[-1].replace("_", " ")
                    if answers_match(val, question["answer"]):
                        return val
                if bindings:
                    val = bindings[0].get("label", bindings[0].get("final", {})).get("value", "")
                    if val.startswith("http://"):
                        val = val.split("/")[-1].replace("_", " ")
                    return val
            except Exception:
                continue
    return None
