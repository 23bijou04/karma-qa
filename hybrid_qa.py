#!/usr/bin/env python3
"""Schema-Routed Hybrid QA: uses KG schema to route SPARQL vs text.

Pipeline:
  1. EntityLink(q)            — DBpedia Spotlight (spaCy fallback)
  2. SchemaDiscovery(E)       — entity classes, properties, sample values
  3. RouteDecision(q, S)      — rule-based first, LLM fallback -> SPARQL | TEXT
  4. if SPARQL: GenerateSPARQL(q, S) -> Execute with retry -> answer
  5. if TEXT (or SPARQL fail): context entity discovery:
     a) LLM-generated SPARQL  — targeted context entities
     b) SPARQL neighbor query  — reliable fallback if a finds nothing new
  6. WikipediaRetrieval       — fetch pages for context entities
  7. SectionExtraction        — sentence-level keyword scoring
  8. LLM reads passages       -> FINAL ANSWER

Usage:
  python3 hybrid_qa.py "Who directed Inception?" --verbose
  python3 hybrid_qa.py "Who destroys the last Horcrux?" --verbose
  python3 hybrid_qa.py "When was Albert Einstein born?" --api-key $GROQ_API_KEY
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from groq import Groq

from ablation_study_v3.wiki_utils import uri_to_wiki_title
from dbpedia_qa.dbpedia import DBpediaEntityLinker, DBpediaSPARQLClient, SchemaDiscovery
from dbpedia_qa.dbpedia.schema_discovery import CACHED_CLASS_SCHEMAS, PROPERTY_RANGES
from dbpedia_qa.models import Entity, HybridResult, extract_final_answer
from graphrag_wiki import (
    assemble_context,
    link_entities_to_dbpedia,
    retrieve_wikipedia,
)
from sparql_evaluation import (
    ANSWER_PROMPT,
    SPARQL_GEN_PROMPT,
    SPARQL_RETRY_PROMPT,
    call_llm,
    execute_sparql,
    extract_sparql,
    get_schema_for_uris,
    sanitize_sparql_uris,
)

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

ROUTE_PROMPT = """You are a knowledge graph query router for DBpedia. \
Decide if this question can be answered via SPARQL.

{schema}

QUESTION: {question}
{hint}
Reasoning steps:
1. What information does this question need? \
(a date, a name, a place, a comparison, etc.)
2. What property or relationship provides that information? \
(e.g., birthDate, director, foundedBy, nationality)
3. Does the schema above contain that property, \
or a 2-hop chain that leads to it?
   - Direct: the property exists on the entity \
(e.g., dbo:birthDate in "Actual properties")
   - 2-hop: a chain exists \
(e.g., "dbo:director -> Person: birthDate")
4. If a complete property path exists: SPARQL
5. If the required property is absent or the chain is incomplete: TEXT

Reply with EXACTLY one word: SPARQL or TEXT"""

CONTEXT_SPARQL_PROMPT = """You are a SPARQL expert for DBpedia. The KG schema does not have enough properties to answer this question directly via SPARQL. Instead, we need to find related DBpedia entities whose Wikipedia pages likely contain the answer.

Generate a SPARQL query that finds these related entities. Return entity URIs and their labels.

{schema}

QUESTION: {question}

RULES:
1. Use the EXACT entity URIs from the schema above
2. Only follow properties that are RELEVANT to the question topic — use the schema to pick the right properties
3. Do NOT blindly follow all dbo:/dbp: relations — that returns noise
4. NEVER use dbo:wikiPageWikiLink
5. Return ?entity and ?label columns
6. Include the seed entities themselves in the results
7. Use UNION to combine multiple relation paths
8. For questions about relatives (father, mother, parent, spouse, child):
   use dbo:parent, dbp:father, dbp:mother, dbo:spouse, dbo:child to find the relative entity
9. Add LIMIT 6
10. Generate ONLY the SPARQL query, no explanation

EXAMPLE for "Who directed Inception?":
SELECT DISTINCT ?entity ?label WHERE {{
  {{ VALUES ?entity {{ <http://dbpedia.org/resource/Inception> }}
     ?entity rdfs:label ?label . FILTER(LANG(?label) = "en") }}
  UNION
  {{ <http://dbpedia.org/resource/Inception> dbo:director ?entity .
     ?entity rdfs:label ?label . FILTER(LANG(?label) = "en") }}
}} LIMIT 6"""

SPARQL_DECOMPOSE_PROMPT = """You are a SPARQL expert for DBpedia. Generate a query for ONE entity relevant to a comparison question.

{schema}

QUESTION: {question}
TARGET ENTITY: {entity_name} (<{entity_uri}>)

Generate a SPARQL query that extracts the relevant property value(s) for ONLY this entity.
Use OPTIONAL patterns so the query returns results even if some properties are missing.
Add LIMIT 10.
Generate ONLY the SPARQL query, no explanation."""

CONTEXT_SPARQL_RETRY_PROMPT = """Your previous context-retrieval SPARQL query returned {problem}.

Previous query:
{sparql}

Fix the query. We need to find related DBpedia entities whose Wikipedia pages help answer the question.

{schema}

QUESTION: {question}

Generate ONLY the corrected SPARQL query:"""


# ---------------------------------------------------------------------------
# Route decision
# ---------------------------------------------------------------------------

def _parse_2hop_chains(schema_ctx: str) -> dict[str, tuple[str, list[str]]]:
    """Extract 2-hop chain info from the '2-Hop chains:' section only.

    Section-anchored: only parses lines inside '2-Hop chains:' blocks,
    NOT 'Actual properties' lines like 'dbo:director -> Christopher Nolan'.

    Returns: {"director": ("Person", ["birthDate", "deathDate", ...])}
    """
    chains: dict[str, tuple[str, list[str]]] = {}
    in_section = False
    for line in schema_ctx.split("\n"):
        stripped = line.strip()
        if stripped == "2-Hop chains:":
            in_section = True
            continue
        if in_section:
            if not stripped or not stripped.startswith("- dbo:"):
                in_section = False
                continue
            m = re.match(
                r'-\s*dbo:(\w+)\s*->\s*(\w+):\s*(\w+(?:,\s*\w+)*)', stripped
            )
            if m:
                chains[m.group(1)] = (
                    m.group(2),
                    [p.strip() for p in m.group(3).split(",")],
                )
    return chains


def _rule_based_route(question: str, schema_ctx: str) -> tuple[str, str] | None:
    """Return ('SPARQL', reason) if rules match, None if inconclusive (let LLM decide)."""
    q_lower = question.lower()

    # Parse properties from both schema formats:
    #   Format 1: "dbo:prop -> value" (actual properties section)
    actual_props = set(re.findall(r'(?:dbo|dbp):\w+', schema_ctx))
    #   Format 2: "  dbo: prop1, prop2, ..." (cached ontology list, may be truncated)
    for m in re.finditer(r'^\s+(dbo|dbp): ([\w]+(?:,\s*[\w]+)*)', schema_ctx, re.MULTILINE):
        prefix = m.group(1)
        for name in m.group(2).split(','):
            name = name.strip()
            if name:
                actual_props.add(f"{prefix}:{name}")

    comparison_words = [
        "earlier", "later", "older", "younger", "first", "before", "after",
        "same", "both", "more", "less", "longer", "shorter",
    ]
    date_props = {
        "dbo:birthDate", "dbp:birthDate", "dbo:deathDate", "dbp:deathDate",
        "dbo:activeYearsStartYear", "dbp:established", "dbo:foundingYear",
        "dbp:foundingDate", "dbo:releaseDate", "dbp:released",
    }
    country_props = {
        "dbo:nationality", "dbp:nationality", "dbo:country", "dbp:country",
        "dbo:citizenship", "dbp:citizenship", "dbo:birthPlace", "dbp:birthPlace",
        "dbo:stateOfOrigin", "dbp:placeOfBirth",
    }

    # Comparison + date properties on the entity itself -> SPARQL
    if any(w in q_lower for w in comparison_words):
        if actual_props & date_props:
            return ("SPARQL", "rule:direct_date_comparison")
        # "same country / nationality" comparisons
        country_keywords = ["country", "nationality", "citizen", "born in"]
        if any(kw in q_lower for kw in country_keywords) and actual_props & country_props:
            return ("SPARQL", "rule:direct_country_comparison")

    # Bridge chain validation: verify the relationship word maps to a target class
    # whose properties are relevant to the question (replaces blind class+word check)
    bridge_word_to_prop = {
        "director": "director", "directed": "director",
        "producer": "producer", "produced": "producer",
        "writer": "writer", "wrote": "writer",
        "author": "author",
        "composer": "musicComposer",
        "starring": "starring", "actor": "starring", "actress": "starring",
        "founded": "foundedBy", "founder": "foundedBy",
    }

    chains_2hop = _parse_2hop_chains(schema_ctx)

    def _has_bridge_word(word: str) -> bool:
        return bool(re.search(rf"\b{word}\b", q_lower))

    if any(w in q_lower for w in comparison_words):
        for word, prop_name in bridge_word_to_prop.items():
            if not _has_bridge_word(word):
                continue
            # Check if this property has a chain in schema text
            if prop_name in chains_2hop:
                target_cls, target_props = chains_2hop[prop_name]
                if any(p in target_props for p in [
                    "birthDate", "deathDate", "foundingDate",
                    "activeYearsStartYear",
                ]):
                    return ("SPARQL", "rule:bridge_comparison")
            # Fallback: check PROPERTY_RANGES directly
            range_key = f"dbo:{prop_name}"
            if range_key in PROPERTY_RANGES:
                target_cls = PROPERTY_RANGES[range_key]
                if target_cls in CACHED_CLASS_SCHEMAS:
                    target_schema = CACHED_CLASS_SCHEMAS[target_cls]
                    if any(p in target_schema.get("dbo", []) for p in [
                        "birthDate", "deathDate", "foundingDate",
                    ]):
                        return ("SPARQL", "rule:bridge_comparison")

    # Non-comparison bridge: "When was the director of X born?"
    # Detects: bridge_word + hop-2 property keyword
    hop2_keywords = {
        "born": ["birthDate", "birthPlace"],
        "died": ["deathDate", "deathPlace"],
        "birth": ["birthDate", "birthPlace"],
        "death": ["deathDate", "deathPlace"],
        "country": ["country", "nationality"],
    }
    for word, prop_name in bridge_word_to_prop.items():
        if not _has_bridge_word(word):
            continue
        range_key = f"dbo:{prop_name}"
        if range_key not in PROPERTY_RANGES:
            continue
        target_cls = PROPERTY_RANGES[range_key]
        if target_cls not in CACHED_CLASS_SCHEMAS:
            continue
        target_dbo = CACHED_CLASS_SCHEMAS[target_cls].get("dbo", [])
        for kw, needed_props in hop2_keywords.items():
            if re.search(rf"\b{kw}\b", q_lower) and any(
                p in target_dbo for p in needed_props
            ):
                return ("SPARQL", "rule:bridge_hop2")

    # Parent chain: father/mother + any property question -> SPARQL
    # Handles: "Where did X's father graduate?", "When was X's mother born?"
    parent_words = ["father", "mother", "parent", "paternal", "maternal"]
    parent_props = {"dbo:parent", "dbp:father", "dbp:mother"}
    if any(w in q_lower for w in parent_words) and actual_props & parent_props:
        return ("SPARQL", "rule:parent_chain")

    # Direct property lookup (includes morphological variants for word boundary matching)
    _director_props = {"dbo:director", "dbp:director"}
    _author_props = {"dbo:author", "dbp:author", "dbo:writer", "dbp:writer"}
    _birth_props = {"dbo:birthDate", "dbp:birthDate", "dbo:birthPlace", "dbp:birthPlace"}
    _award_props = {"dbo:award", "dbp:award"}
    _founder_props = {"dbo:foundingYear", "dbp:foundingDate", "dbo:founder", "dbp:founder"}
    direct_map = {
        "direct": _director_props,
        "directed": _director_props,
        "director": _director_props,
        "wrote": _author_props,
        "written": _author_props,
        "author": _author_props,
        "born": _birth_props,
        "died": {"dbo:deathDate", "dbp:deathDate", "dbo:deathPlace"},
        "capital": {"dbo:capital", "dbp:capital"},
        "population": {"dbo:populationTotal", "dbp:population"},
        "spouse": {"dbo:spouse", "dbp:spouse"},
        "married": {"dbo:spouse", "dbp:spouse"},
        "founded": _founder_props,
        "founder": _founder_props,
        "nationality": {"dbo:nationality", "dbp:nationality", "dbo:citizenship"},
        "father": {"dbo:parent", "dbp:father"},
        "mother": {"dbo:parent", "dbp:mother"},
        "award": _award_props,
        "awarded": _award_props,
        "occupation": {"dbo:occupation", "dbp:occupation"},
    }
    for keyword, props in direct_map.items():
        if re.search(rf"\b{keyword}\b", q_lower) and actual_props & props:
            return ("SPARQL", "rule:direct_property")

    return None  # inconclusive — let LLM decide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def entities_to_schema_dicts(entities: list[Entity]) -> list[dict]:
    """Convert Spotlight Entity objects to dicts for get_schema_for_uris()."""
    return [
        {"uri": e.uri, "name": e.label}
        for e in entities if e.uri
    ]


def _is_comparison_question(question: str, entities: list[Entity]) -> bool:
    """Check if this is a comparison question with 2+ entities."""
    q_lower = question.lower()
    # Superset of comparison_words in _rule_based_route — intentionally broader
    # to catch comparison patterns (same/both/longer/shorter) that may not require
    # date-based routing but do benefit from per-entity decomposition.
    comparison_words = [
        "earlier", "later", "older", "younger", "first", "before", "after",
        "more recently", "longer", "shorter", "same", "both",
    ]
    return len(entities) >= 2 and any(w in q_lower for w in comparison_words)


def _try_per_entity_sparql(question: str, entities: list[Entity],
                           schema_ctx: str, llm_client, model: str,
                           verbose: bool = False) -> list[dict]:
    """Decompose a comparison query into per-entity SPARQL queries.

    Returns combined results if BOTH entities return data, else empty list.
    Only processes the first 2 entities (2WikiMultiHopQA comparisons are always 2-entity).
    """
    sep = "=" * 70
    if verbose:
        print(f"\n{sep}")
        print("STEP 4c: Per-Entity SPARQL Decomposition (comparison recovery)")
        print(sep)
        print(f"  Entities: {', '.join(e.label for e in entities[:2])}")

    per_entity_results = []
    for i, entity in enumerate(entities[:2]):
        if verbose:
            print(f"\n  --- Entity {i + 1}: {entity.label} ---")

        prompt = SPARQL_DECOMPOSE_PROMPT.format(
            schema=schema_ctx, question=question,
            entity_name=entity.label, entity_uri=entity.uri,
        )
        resp = call_llm(llm_client, prompt, model)
        query = sanitize_sparql_uris(extract_sparql(resp))
        if verbose:
            print(f"  Generated SPARQL:\n    {query}")

        results, err = execute_sparql(query)
        if verbose:
            if err:
                print(f"  Results: {len(results)} rows, error: '{err[:80]}'")
            else:
                print(f"  Results: {len(results)} rows")
            if results:
                for row in results[:5]:
                    print(f"    {row}")

        if not results or err:
            if verbose:
                print(f"  No data for {entity.label} — decomposition failed, falling to TEXT")
            return []

        for row in results:
            row["entity_name"] = entity.label
        per_entity_results.extend(results)

    if verbose:
        print(f"\n  Combined: {len(per_entity_results)} rows from both entities")

    return per_entity_results


# ---------------------------------------------------------------------------
# Compositional decomposition (Phase 2)
# ---------------------------------------------------------------------------

_ROLE_TO_DBO = {
    "director": "dbo:director", "producer": "dbo:producer",
    "author": "dbo:author", "writer": "dbo:writer",
    "composer": "dbo:musicComposer", "founder": "dbo:foundedBy",
    "starring": "dbo:starring",
    # Family roles
    "father": "dbo:parent", "mother": "dbo:parent",
    "husband": "dbo:spouse", "wife": "dbo:spouse",
    "spouse": "dbo:spouse",
}
_PROP_TO_DBO = {
    "born": "dbo:birthPlace", "birth": "dbo:birthPlace",
    "birthplace": "dbo:birthPlace",
    "die": "dbo:deathPlace", "died": "dbo:deathPlace",
    "deathplace": "dbo:deathPlace",
    "buried": "dbo:deathPlace",
    "educated": "dbo:almaMater", "graduate": "dbo:almaMater",
    "graduated": "dbo:almaMater",
    "nationality": "dbo:nationality", "country": "dbo:nationality",
    # Multi-word phrases (extracted by possessive patterns)
    "place of birth": "dbo:birthPlace",
    "place of death": "dbo:deathPlace",
    "married": "dbo:spouse",
    "work": "dbo:occupation", "worked": "dbo:occupation",
    "live": "dbo:residence", "lived": "dbo:residence",
}
_PROP_DATE_DBO = {
    "born": "dbo:birthDate", "die": "dbo:deathDate", "died": "dbo:deathDate",
    "birthdate": "dbo:birthDate",
    "deathdate": "dbo:deathDate",
    "date of birth": "dbo:birthDate",
    "date of death": "dbo:deathDate",
}


def _decompose_compositional(
    question: str, entities: list,
) -> tuple[str, str, str] | None:
    """Decompose compositional question into (role, entity_name, property_word).

    Returns None if question doesn't match known patterns.
    """
    q = question.strip().rstrip("?")
    # Pattern 1: "Where/When was the director of Inception born"
    m = re.match(
        r"(?:where|when|how)\s+(?:was|is|did)\s+the\s+(\w+)\s+of\s+"
        r"(?:the\s+(?:film|movie|book|novel|show|series)\s+)?"
        r"(.+?)\s+(born|die|died|buried|educated|graduate\w*)",
        q, re.IGNORECASE,
    )
    if m:
        return m.group(1), m.group(2).strip(), m.group(3)
    # Pattern 2: "What is the birthplace of the director of X"
    m = re.match(
        r"what\s+(?:is|was)\s+the\s+(\w+)\s+of\s+the\s+(\w+)\s+of\s+(.+)",
        q, re.IGNORECASE,
    )
    if m:
        return m.group(2), m.group(3).strip(), m.group(1)

    # --- Possessive patterns (X's <role>) ---

    # Pattern 3: "Where/When was/did X's <role> born/die/graduate..."
    m = re.match(
        r"(?:where|when|how|why)\s+(?:was|is|did|were)\s+"
        r"(.+?)'s\s+(\w+)\s+(born|die|died|buried|educated|graduate\w*"
        r"|live\w*|married|work\w*)",
        q, re.IGNORECASE,
    )
    if m:
        return m.group(2), m.group(1).strip(), m.group(3)

    # Pattern 4: "What is/was the <prop-phrase> of X's <role>"
    # Greedy (.+) for prop so "place of birth" isn't split at first "of"
    m = re.match(
        r"what\s+(?:is|was)\s+(?:the\s+)?(.+)\s+of\s+(.+?)'s\s+(\w+)\s*$",
        q, re.IGNORECASE,
    )
    if m:
        return m.group(3), m.group(2).strip(), m.group(1).strip()

    # Pattern 5: "What <noun> is X's <role>"
    m = re.match(
        r"what\s+(\w+)\s+(?:is|was)\s+(.+?)'s\s+(\w+)\s*$",
        q, re.IGNORECASE,
    )
    if m:
        return m.group(3), m.group(2).strip(), m.group(1)

    # Pattern 6: "Which country X's <role> is from"
    m = re.match(
        r"which\s+(\w+)\s+(.+?)'s\s+(\w+)\s+is\s+from",
        q, re.IGNORECASE,
    )
    if m:
        return m.group(3), m.group(2).strip(), m.group(1)

    return None


def _try_compositional_decomposition(
    question: str, entities: list[Entity],
    schema_ctx: str, llm_client, model: str,
    kg_client: DBpediaSPARQLClient,
    verbose: bool = False,
) -> str | None:
    """Hop-by-hop SPARQL decomposition for compositional questions.

    Returns answer string if successful, None otherwise.
    """
    from dbpedia_qa.dbpedia.sparql_utils import escape_sparql_uri

    decomp = _decompose_compositional(question, entities)
    if not decomp:
        return None

    role, entity_name, prop_word = decomp
    role_lower = role.lower()
    prop_lower = prop_word.lower()

    sep = "=" * 70
    if verbose:
        print(f"\n{sep}")
        print("STEP 4d: Compositional Decomposition")
        print(sep)
        print(f"  Role: {role}, Entity: {entity_name}, Property: {prop_word}")

    # Map role to DBpedia property
    role_dbo = _ROLE_TO_DBO.get(role_lower)
    if not role_dbo:
        if verbose:
            print(f"  No DBO mapping for role '{role}', skipping decomposition")
        return None

    # Find seed entity URI
    seed_uri = None
    for e in entities:
        if e.uri and (entity_name.lower() in e.label.lower()
                      or e.label.lower() in entity_name.lower()):
            seed_uri = e.uri
            break
    if not seed_uri and entities:
        seed_uri = entities[0].uri

    if not seed_uri:
        return None

    # Hop 1: find bridge entity via role relation
    safe_uri = escape_sparql_uri(seed_uri)
    hop1_query = (
        f"SELECT ?bridge ?label WHERE {{ "
        f"<{safe_uri}> {role_dbo} ?bridge . "
        f"?bridge rdfs:label ?label . FILTER(LANG(?label) = \"en\") "
        f"}} LIMIT 5"
    )
    if verbose:
        print(f"  Hop 1 SPARQL: {hop1_query[:200]}")

    hop1_results, hop1_err = execute_sparql(hop1_query, preserve_uris=True)
    if verbose:
        print(f"  Hop 1: {len(hop1_results)} results"
              + (f", error: {hop1_err[:60]}" if hop1_err else ""))

    if not hop1_results:
        if verbose:
            print("  Hop 1 failed — no bridge entity found")
        return None

    bridge_uri = None
    bridge_label = None
    for row in hop1_results:
        for key in ("bridge", "entity", "uri"):
            if key in row and row[key] and row[key].startswith("http://"):
                bridge_uri = row[key]
                break
        for key in ("label", "name"):
            if key in row and row[key]:
                bridge_label = row[key]
                break
        if bridge_uri:
            break

    if not bridge_uri:
        return None

    if verbose:
        print(f"  Bridge entity: {bridge_label} ({bridge_uri})")

    # Hop 2: find property on bridge entity
    # Try both place and date mappings
    prop_dbo = _PROP_TO_DBO.get(prop_lower)
    date_dbo = _PROP_DATE_DBO.get(prop_lower)

    hop2_props = []
    if prop_dbo:
        hop2_props.append(prop_dbo)
    if date_dbo:
        hop2_props.append(date_dbo)

    if not hop2_props:
        if verbose:
            print(f"  No DBO mapping for property '{prop_word}'")
        return None

    safe_bridge = escape_sparql_uri(bridge_uri)
    for dbo_prop in hop2_props:
        hop2_query = (
            f"SELECT ?value WHERE {{ "
            f"<{safe_bridge}> {dbo_prop} ?value "
            f"}} LIMIT 5"
        )
        if verbose:
            print(f"  Hop 2 SPARQL ({dbo_prop}): {hop2_query[:200]}")

        hop2_results, hop2_err = execute_sparql(hop2_query)
        if hop2_results:
            values = []
            for row in hop2_results:
                for val in row.values():
                    if val:
                        values.append(str(val))
            if values:
                answer = values[0]
                if verbose:
                    print(f"  Hop 2 answer: {answer}")
                return answer

    # Hop 2 SPARQL failed — fall to targeted text retrieval for bridge entity
    if verbose:
        print("  Hop 2 SPARQL failed — trying targeted text retrieval")

    hop1_facts = _format_kg_facts([{
        "role": role, "entity": entity_name,
        "bridge": bridge_label or bridge_uri.split("/")[-1].replace("_", " "),
    }])

    bridge_entity = Entity(
        uri=bridge_uri,
        label=bridge_label or bridge_uri.split("/")[-1].replace("_", " "),
        source="decomposed",
        discovery_relation=role_lower,
    )
    raw_texts = retrieve_wikipedia([bridge_entity], question, verbose=verbose)
    if not raw_texts:
        return None

    context, _ = assemble_context(
        raw_texts, question, kg_facts=hop1_facts, verbose=verbose,
    )
    prompt = TEXT_REASON_PROMPT.format(
        question=question, context=context, kg_facts=hop1_facts,
    )
    response = call_llm(llm_client, prompt, model)
    return extract_final_answer(response) if response else None


def _sparql_neighbors(entities: list[Entity],
                      kg_client: DBpediaSPARQLClient,
                      limit_per_entity: int = 10,
                      verbose: bool = False) -> list[Entity]:
    """Get direct dbo:/dbp: neighbors via a simple SPARQL query.

    Much simpler than explore_kg: no schema lookup, no type blocklist.
    Just gets the first N structured neighbors for each seed entity.
    """
    from dbpedia_qa.dbpedia.sparql_utils import escape_sparql_uri

    all_entities = list(entities)
    seen_uris = {e.uri for e in entities if e.uri}

    for entity in entities:
        if not entity.uri:
            continue
        safe_uri = escape_sparql_uri(entity.uri)
        query = f"""
        SELECT DISTINCT ?p ?neighbor ?label WHERE {{
            <{safe_uri}> ?p ?neighbor .
            FILTER(isIRI(?neighbor))
            FILTER(
                STRSTARTS(STR(?p), "http://dbpedia.org/ontology/")
                || STRSTARTS(STR(?p), "http://dbpedia.org/property/")
            )
            FILTER(?p != <http://dbpedia.org/ontology/wikiPageWikiLink>)
            FILTER(STRSTARTS(STR(?neighbor), "http://dbpedia.org/resource/"))
            FILTER(!STRSTARTS(STR(?neighbor), "http://dbpedia.org/resource/Category:"))
            ?neighbor rdfs:label ?label .
            FILTER(LANG(?label) = "en")
        }} LIMIT {limit_per_entity}
        """
        try:
            raw = kg_client.query(query)
            bindings = raw.get("results", {}).get("bindings", [])
        except Exception as e:
            logger.warning(f"Neighbor query failed for {entity.label}: {e}")
            continue

        for b in bindings:
            uri = b["neighbor"]["value"]
            if uri in seen_uris:
                continue
            seen_uris.add(uri)
            label = b.get("label", {}).get("value", "")
            if not label:
                label = unquote(uri.split("/")[-1]).replace("_", " ")
            prop = b["p"]["value"].split("/")[-1]
            all_entities.append(Entity(
                uri=uri, label=label, source="sparql_neighbor",
                discovery_relation=prop,
            ))
            if verbose:
                print(f"    {entity.label} --{prop}--> {label}")

    return all_entities


def _sparql_results_to_entities(results: list[dict],
                                seed_uris: set[str]) -> list[Entity]:
    """Convert SPARQL context query results to Entity objects."""
    from urllib.parse import quote

    entities = []
    seen = set()
    for row in results:
        # Try common column names for entity URIs
        uri = None
        label = None
        for key in ("entity", "related", "neighbor", "uri", "s", "o"):
            if key in row and row[key]:
                val = row[key]
                if val.startswith("http://dbpedia.org/resource/"):
                    uri = val  # full URI — use directly, no re-encoding
                elif "/" not in val and val.strip():
                    # Reconstructed local name from execute_sparql's deconstruction
                    local_name = val.replace(" ", "_")
                    # DBpedia convention: keep (), ', , unencoded (RFC 2396 §2.4.3)
                    encoded = quote(local_name, safe="/:@!$&'()*+,;=-._~%")
                    uri = f"http://dbpedia.org/resource/{encoded}"
                if uri:
                    break
        for key in ("label", "entityLabel", "name"):
            if key in row and row[key]:
                label = row[key]
                break

        if not uri or uri in seen:
            continue
        seen.add(uri)

        if not label:
            label = unquote(uri.split("/")[-1]).replace("_", " ")

        # Extract predicate for discovery_relation
        relation = None
        for key in ("p", "relation", "predicate", "property"):
            if key in row and row[key]:
                val = row[key]
                relation = val.split("/")[-1] if "/" in val else val
                break

        source = "seed" if uri in seed_uris else "sparql_context"
        entities.append(Entity(
            uri=uri, label=label, source=source,
            discovery_relation=relation,
        ))

    return entities


def _format_kg_facts(results: list[dict], max_chars: int = 800) -> str:
    """Format SPARQL result rows as 'Known facts from KG' text block."""
    if not results:
        return ""
    lines = []
    header = "Known facts from knowledge graph:\n"
    total = len(header)
    for row in results:
        parts = []
        for key, val in row.items():
            if val and not key.startswith("_"):
                if isinstance(val, str) and val.startswith("http://"):
                    val = val.split("/")[-1].replace("_", " ")
                parts.append(f"{key}: {val}")
        if parts:
            line = "- " + ", ".join(parts)
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
    if not lines:
        return ""
    return header + "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Text fallback: LLM-generated context SPARQL
# ---------------------------------------------------------------------------

def _find_context_entities(
    question: str, entities: list[Entity],
    schema_ctx: str, llm_client, model: str,
    kg_client: DBpediaSPARQLClient,
    verbose: bool = False,
) -> tuple[list[Entity], list[dict]]:
    """Find context entities for Wikipedia retrieval.

    Two-tier approach:
      1. LLM-generated SPARQL (targeted, may fail)
      2. Simple SPARQL neighbor query (reliable fallback)
    Returns (entities, sparql_results) — entities for page fetch, raw results for KG facts.
    """
    sep = "=" * 70
    seed_uris = {e.uri for e in entities if e.uri}

    context_entities = []
    ctx_sparql_results: list[dict] = []
    used_llm_sparql = False

    # --- Tier 1: LLM-generated context SPARQL (only if schema available) ---
    if schema_ctx:
        if verbose:
            print(f"\n{sep}")
            print("STEP 4a: Context entities — LLM-generated SPARQL")
            print(sep)

        prompt = CONTEXT_SPARQL_PROMPT.format(schema=schema_ctx, question=question)
        resp = call_llm(llm_client, prompt, model)
        sparql_query = sanitize_sparql_uris(extract_sparql(resp))
        if verbose:
            print(f"  Context SPARQL:\n    {sparql_query[:300]}")

        results, err = execute_sparql(sparql_query, preserve_uris=True)
        if verbose:
            status = f"{len(results)} rows"
            if err:
                status += f", error: '{err[:60]}'"
            print(f"  Results: {status}")

        # Retry if empty or error
        if not results or err:
            problem = f"an error: {err}" if err else "empty results (no rows)"
            retry_prompt = CONTEXT_SPARQL_RETRY_PROMPT.format(
                problem=problem, sparql=sparql_query,
                schema=schema_ctx, question=question,
            )
            retry_resp = call_llm(llm_client, retry_prompt, model)
            retry_query = sanitize_sparql_uris(extract_sparql(retry_resp))
            if verbose:
                print(f"  Retry SPARQL:\n    {retry_query[:300]}")
            results2, err2 = execute_sparql(retry_query, preserve_uris=True)
            if verbose:
                print(f"  Retry results: {len(results2)} rows")
            if results2:
                results = results2

        # Convert to entities — trust all LLM SPARQL results (no word-overlap
        # filter). The SPARQL prompt already targets relevant relations and has
        # LIMIT 6; downstream assemble_context() caps total chars at 12K.
        ctx_sparql_results = results
        context_entities = _sparql_results_to_entities(results, seed_uris)
        used_llm_sparql = True

    # --- Tier 2: SPARQL neighbor fallback (if LLM SPARQL found no new entities) ---
    new_uris = {e.uri for e in context_entities if e.uri} - seed_uris
    if not new_uris:
        if verbose:
            if used_llm_sparql:
                print(f"\n{sep}")
                print("STEP 4b: LLM SPARQL found no new entities — neighbor fallback")
                print(sep)
            else:
                print(f"\n{sep}")
                print("STEP 4: Context entities — SPARQL neighbors")
                print(sep)
        context_entities = _sparql_neighbors(
            entities, kg_client, verbose=verbose,
        )
        if verbose:
            print(f"  Seed: {len(entities)}, Total: {len(context_entities)}")
        return context_entities, []

    # Always include seed entities (in case SPARQL missed them)
    seen_uris = {e.uri for e in context_entities}
    for e in entities:
        if e.uri and e.uri not in seen_uris:
            context_entities.insert(0, e)

    if verbose:
        print(f"  Context entities ({len(context_entities)}):")
        for e in context_entities:
            print(f"    - {e.label} [{e.source}]")

    return context_entities, ctx_sparql_results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def hybrid_answer(question: str, api_key: str,
                  model: str = "llama-3.3-70b-versatile",
                  verbose: bool = False,
                  llm_client=None, linker=None,
                  kg_client=None, schema_disc=None) -> HybridResult:
    """Schema-routed hybrid QA: SPARQL when schema supports it, text otherwise.

    Args:
        llm_client: Optional pre-created Groq client (for batch reuse)
        linker: Optional pre-created DBpediaEntityLinker (for batch reuse)
        kg_client: Optional pre-created DBpediaSPARQLClient (for batch reuse)
        schema_disc: Optional pre-created SchemaDiscovery (for batch reuse)

    Returns: HybridResult with answer, method, route_source, and sparql_fallback.
    """
    sep = "=" * 70
    llm_client = llm_client or Groq(api_key=api_key)
    linker = linker or DBpediaEntityLinker(min_confidence=0.5)
    kg_client = kg_client or DBpediaSPARQLClient()
    schema_disc = schema_disc or SchemaDiscovery()

    # === Step 1: Entity Linking ===
    entities = link_entities_to_dbpedia(question, linker)
    if verbose:
        print(f"\n{sep}")
        print("STEP 1: Entity Linking (Spotlight + spaCy fallback)")
        print(sep)
        for e in entities:
            print(f"  - {e.label}  [{e.source or 'unknown'}]")
            if e.uri:
                print(f"    URI: {e.uri}")
        if not entities:
            print("  (none found)")

    # === Step 2: Schema Discovery ===
    schema_dicts = entities_to_schema_dicts(entities)
    schema_ctx = get_schema_for_uris(schema_disc, schema_dicts) if schema_dicts else ""
    if verbose:
        print(f"\n{sep}")
        print("STEP 2: Schema Discovery")
        print(sep)
        if schema_ctx:
            for line in schema_ctx.split("\n")[:20]:
                print(f"  {line}")
            if schema_ctx.count("\n") > 20:
                print(f"  ... ({schema_ctx.count(chr(10))} lines total)")
        else:
            print("  (no URIs to query schema for)")

    # === Step 3: Route Decision — can SPARQL answer this? ===
    sparql_answer = ""
    route = "TEXT"
    route_source = "no_schema_fallback"
    if schema_ctx:
        # Rule analysis as hint (not hard decision — LLM always makes final call)
        rule_result = _rule_based_route(question, schema_ctx)
        if rule_result:
            hint = f"\nSchema analysis suggests: {rule_result[1]} (SPARQL likely feasible)\n"
        else:
            hint = ""

        # Escape curly braces in dynamic content to avoid .format() crashes
        safe_schema = schema_ctx.replace("{", "{{").replace("}", "}}")
        safe_question = question.replace("{", "{{").replace("}", "}}")
        route_prompt = ROUTE_PROMPT.format(
            schema=safe_schema, question=safe_question, hint=hint,
        )
        route_resp = call_llm(llm_client, route_prompt, model).strip()
        route_match = re.search(r'\b(SPARQL|TEXT)\b', route_resp, re.IGNORECASE)
        route = route_match.group(1).upper() if route_match else "TEXT"

        # Preserve rule route_source for bridge budget detection when LLM agrees
        if rule_result and route == "SPARQL":
            route_source = rule_result[1]
        else:
            route_source = "llm"

        if verbose:
            print(f"\n{sep}")
            print(f"STEP 3: Route Decision → {route} ({route_source})")
            print(sep)
            if rule_result:
                print(f"  Rule hint: {rule_result[1]}")
            print(f"  LLM response: {route_resp[:120]}")

    # === Step 4: SPARQL Generation + Execution (only if routed) ===
    sparql_partial_facts = ""  # KG facts from partial SPARQL (for text injection)
    executed_sparql_query = ""  # provenance: the SPARQL query that produced the answer
    decomp_attempted = False
    if route == "SPARQL" and schema_ctx:
        if verbose:
            print(f"\n{sep}")
            print("STEP 4: SPARQL Generation + Execution")
            print(sep)

        sparql_prompt = SPARQL_GEN_PROMPT.format(schema=schema_ctx, question=question)
        sparql_resp = call_llm(llm_client, sparql_prompt, model)
        sparql_query = sanitize_sparql_uris(extract_sparql(sparql_resp))
        executed_sparql_query = sparql_query
        if verbose:
            print(f"  Generated SPARQL:\n    {sparql_query[:200]}")

        results, err = execute_sparql(sparql_query)
        if verbose:
            print(f"  Results: {len(results)} rows, error: '{err[:80]}'" if err else f"  Results: {len(results)} rows")

        if not results or err:
            problem = f"an error: {err}" if err else "empty results (no rows)"
            retry_prompt = SPARQL_RETRY_PROMPT.format(
                problem=problem, sparql=sparql_query,
                schema=schema_ctx, question=question,
            )
            retry_resp = call_llm(llm_client, retry_prompt, model)
            retry_query = sanitize_sparql_uris(extract_sparql(retry_resp))
            if verbose:
                print(f"  Retry SPARQL:\n    {retry_query[:200]}")
            results2, err2 = execute_sparql(retry_query)
            if verbose:
                print(f"  Retry results: {len(results2)} rows")
            if results2:
                results = results2
                executed_sparql_query = retry_query

        # Per-entity decomposition for comparison questions (JOIN failure recovery)
        if not results and _is_comparison_question(question, entities):
            results = _try_per_entity_sparql(
                question, entities, schema_ctx, llm_client, model, verbose,
            )

        # Compositional decomposition for non-comparison questions
        if not results and not _is_comparison_question(question, entities):
            decomp_attempted = True
            decomp_answer = _try_compositional_decomposition(
                question, entities, schema_ctx, llm_client, model,
                kg_client, verbose,
            )
            if decomp_answer:
                return HybridResult(
                    decomp_answer, "decomposed", route_source,
                    sparql_query="[decomposed]",
                )

        # Partial SPARQL guard: comparison with data for only 1 entity
        if results and _is_comparison_question(question, entities):
            entity_names_in_results = {
                row.get("entity_name") for row in results if row.get("entity_name")
            }
            if len(entity_names_in_results) == 1 and len(entities) >= 2:
                if verbose:
                    print(f"  Partial data: only {entity_names_in_results}. "
                          f"Injecting as KG context, falling to TEXT.")
                sparql_partial_facts = _format_kg_facts(results)
                results = []  # force text fallback with partial data preserved

        if results:
            fmt_results = json.dumps(results[:20], indent=2)
            answer_prompt = ANSWER_PROMPT.format(
                question=question, results=fmt_results,
            )
            sparql_answer = extract_final_answer(
                call_llm(llm_client, answer_prompt, model)
            )
            if verbose:
                print(f"  SPARQL answer: {sparql_answer}")

    # === Return SPARQL answer or fall back to text ===
    if sparql_answer:
        if verbose:
            print(f"\n{sep}")
            print(f"RESULT (method=sparql, route={route_source})")
            print(sep)
            print(f"  Question: {question}")
            print(f"  Answer:   {sparql_answer}")
        return HybridResult(
            sparql_answer, "sparql", route_source,
            sparql_query=executed_sparql_query,
        )

    # === Text Fallback — find context entities + Wikipedia ===
    method_label = "text"
    sparql_fallback = route == "SPARQL"  # originally routed SPARQL but it failed

    # Compositional decomposition on TEXT route (skip if already tried on SPARQL path)
    if not decomp_attempted and not _is_comparison_question(question, entities):
        decomp_answer = _try_compositional_decomposition(
            question, entities, schema_ctx, llm_client, model,
            kg_client, verbose,
        )
        if decomp_answer:
            return HybridResult(
                decomp_answer, "decomposed", route_source,
                sparql_fallback=sparql_fallback,
                sparql_query="[decomposed]",
            )

    all_entities, ctx_sparql_results = _find_context_entities(
        question, entities, schema_ctx, llm_client, model,
        kg_client, verbose,
    )

    # Build KG facts: prefer partial SPARQL facts, then context SPARQL results
    kg_facts = sparql_partial_facts or _format_kg_facts(ctx_sparql_results)

    # Build relation hints for section boosting (used by Phase 3 if enabled)
    relation_hints = {}
    for e in all_entities:
        if e.discovery_relation and e.uri and "/resource/" in e.uri:
            title = uri_to_wiki_title(e.uri)
            relation_hints[title] = e.discovery_relation

    # === Wikipedia retrieval (shared by both methods) ===
    if verbose:
        print(f"\n{sep}")
        print("STEP 5: Wikipedia retrieval")
        print(sep)
        if kg_facts:
            print(f"  KG facts ({len(kg_facts)} chars) will be injected into prompt")
    raw_texts = retrieve_wikipedia(all_entities, question, verbose=verbose)
    if not raw_texts:
        if verbose:
            print("  No Wikipedia text retrieved.")
        return HybridResult(
            "Could not find answer", method_label, route_source,
            sparql_fallback, kg_context=kg_facts,
        )

    # Section extraction and context assembly (seed entities get budget priority)
    if route_source in ("rule:bridge_comparison", "rule:bridge_hop2", "rule:parent_chain"):
        # Bridge/parent questions: answer is on discovered entities (directors,
        # parents), not the Spotlight entities (films). Prioritize bridge pages.
        spotlight_uris = {e.uri for e in entities if e.uri}
        # When no bridge entities found, seed_titles is empty (falsy) ->
        # assemble_context uses uniform budget (intended fallback).
        seed_titles = {
            uri_to_wiki_title(e.uri) for e in all_entities
            if e.uri and "/resource/" in e.uri and e.uri not in spotlight_uris
        }
    else:
        seed_titles = {
            uri_to_wiki_title(e.uri) for e in entities if e.uri and "/resource/" in e.uri
        }
    context, filtered_texts = assemble_context(
        raw_texts, question, seed_titles=seed_titles,
        kg_facts=kg_facts, relation_hints=relation_hints, verbose=verbose,
    )

    # LLM reasoning
    if verbose:
        print(f"\n{sep}")
        print("STEP 6: LLM Reasoning over text")
        print(sep)
        print(f"  Context: {len(context)} chars from {len(filtered_texts)} pages")
        if kg_facts:
            print(f"  KG facts: {len(kg_facts)} chars injected")

    prompt = TEXT_REASON_PROMPT.format(
        question=question, context=context, kg_facts=kg_facts,
    )
    response = call_llm(llm_client, prompt, model)

    answer = extract_final_answer(response) if response else "Could not generate answer"

    if verbose:
        fb_tag = " [sparql_fallback]" if sparql_fallback else ""
        print(f"\n{sep}")
        print(f"RESULT (method={method_label}, route={route_source}{fb_tag})")
        print(sep)
        print(f"  Question: {question}")
        print(f"  Answer:   {answer}")
        print(f"  Pages:    {len(filtered_texts)} ({', '.join(filtered_texts.keys())})")

    return HybridResult(
        answer, method_label, route_source, sparql_fallback,
        kg_context=kg_facts,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Schema-Routed Hybrid QA: SPARQL-first + text fallback"
    )
    parser.add_argument("question", type=str, help="The question to answer")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GROQ_API_KEY"),
        help="Groq API key (or set GROQ_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="LLM model (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show pipeline steps")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set GROQ_API_KEY env var")
        sys.exit(1)

    result = hybrid_answer(
        args.question, args.api_key, model=args.model,
        verbose=args.verbose,
    )
    if not args.verbose:
        fb = " [sparql_fallback]" if result.sparql_fallback else ""
        print(f"Answer ({result.method}, route={result.route_source}{fb}): {result.answer}")


if __name__ == "__main__":
    main()
