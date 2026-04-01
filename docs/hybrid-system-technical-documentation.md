# DBpedia Hybrid QA System — Comprehensive Technical Documentation

**Version:** 1.2
**Last Updated:** 2026-03-27
**Scope:** Complete technical specification of the DBpedia Hybrid QA system for 2WikiMultiHopQA benchmark
**Target Audience:** Thesis researchers, system maintainers, hyperparameter tuners

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Pipeline Architecture](#2-core-pipeline-architecture)
3. [Entity Linking Subsystem](#3-entity-linking-subsystem)
4. [Schema Discovery & Projection](#4-schema-discovery--projection)
5. [Routing Architecture](#5-routing-architecture)
6. [SPARQL Solver](#6-sparql-solver)
7. [LogicRAG Decomposition](#7-logicrag-decomposition)
8. [Text Retrieval & RAG Pipeline](#8-text-retrieval--rag-pipeline)
9. [Entity Discovery](#9-entity-discovery)
10. [LLM Provider Infrastructure](#10-llm-provider-infrastructure)
11. [Data Models](#11-data-models)
12. [Evaluation Framework](#12-evaluation-framework)
13. [Complete Hyperparameter Catalog](#13-complete-hyperparameter-catalog)
14. [External Service Dependencies](#14-external-service-dependencies)
15. [Caching Architecture](#15-caching-architecture)
16. [Design Decisions & Tradeoffs](#16-design-decisions--tradeoffs)

---

## 1. System Overview

### Architecture Diagram (ASCII)

```
Question Input
    |
    v
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 1: Entity Linking                          │
│  DBpedia Spotlight + spaCy fallback → Entity URIs + Types       │
│  5-stage mention extraction, confidence ≥ 0.5                   │
└─────────────────────────────────────────────────────────────────┘
    |
    v
┌─────────────────────────────────────────────────────────────────┐
│         STEP 2: Schema Discovery & Entity Profiling              │
│  Build EntityProfile (URI → class + properties + values)        │
│  Project schema for SPARQL, LogicRAG, Text contexts             │
└─────────────────────────────────────────────────────────────────┘
    |
    v
┌─────────────────────────────────────────────────────────────────┐
│         STEP 3: Route Decision (3-Stage)                        │
│  Stage 1: Rule-based (property/chain matching)                  │
│  Stage 2: SPARQL ASK verification                               │
│  Stage 3: LLM fallback decision                                 │
└─────────────────────────────────────────────────────────────────┘
    |
    +─────────────────────────────────────────────┐
    |                                             |
    v                                             v
┌──────────────────────────┐         ┌────────────────────────┐
│  STEP 4: Direct SPARQL   │         │  ROUTE: DECOMPOSE      │
│  Generation + Execution  │         │    or TEXT             │
│  Single/multi-hop triples│         │                        │
└──────────────────────────┘         └────────────────────────┘
    |                                     |
    | Answer found                        v
    |                          ┌──────────────────────┐
    |                          │ STEP 5: LogicRAG     │
    |                          │ Decomposition        │
    |                          │ Sub-Qs + dependency  │
    |                          │ routing              │
    |                          └──────────────────────┘
    |                              |
    |                              | Answer found
    |                              |
    |                              v
    |                          ┌──────────────────────┐
    |                          │STEP 5b: SPARQL       │
    |                          │Pre-check (safety net)│
    |                          │Direct properties     │
    |                          └──────────────────────┘
    |                              |
    |                              v
    |                          ┌──────────────────────┐
    |                          │STEP 6: Text Fallback │
    |                          │Wikipedia + RAG/KW    │
    |                          │+ LLM reasoning       │
    |                          └──────────────────────┘
    |                              |
    +─────────────┬────────────────+
                  |
                  v
          ┌──────────────┐
          │ HybridResult │
          │(answer,method│
          │ ,route)      │
          └──────────────┘
```

### Four Pipelines Overview

The system implements four distinct QA pipelines, each suited to different question characteristics:

1. **Hybrid Pipeline** (`hybrid_qa.py`, 401 lines) — Main orchestrator with two-route architecture:
   - SPARQL Direct: when schema projection confirms the KG covers the question
   - LogicRAG Decomposition: for multi-hop or incomplete schema — decomposes into sub-questions, each independently uses SPARQL or text retrieval
   - Text retrieval (Wikipedia + RAG) is the fallback mechanism within sub-questions and a last resort, not a separate top-level route

2. **GraphRAG Text-Only** (`graphrag_wiki.py`, 847 lines) — Pure text pipeline:
   - Spotlight entity linking → KG exploration → Wikipedia retrieval → LLM reasoning
   - No SPARQL, suitable for schema-sparse domains

3. **GRASP Baseline** (`grasp_evaluation.py`) — External WebSocket API:
   - Third-party KG QA system (`wss://grasp.cs.uni-freiburg.de/api/live`)
   - Uses OpenAI GPT-5-mini model

4. **Schema-Guided SPARQL Component** (`sparql_evaluation.py`) — Hybrid's direct SPARQL path evaluated standalone:
   - Schema discovery + LLM-driven SPARQL generation
   - No text fallback or decomposition (used for component-level evaluation only)

### Design Philosophy

**Thesis:** Entity-only knowledge graph (no relations) + Wikipedia text retrieval can effectively answer multi-hop QA without full relation coverage.

**Key Insights:**
- DBpedia entity coverage is high (93.6% on 1000q benchmark)
- Property coverage is moderate (36.8% hop-verified)
- Combining KG entities with Wikipedia text provides robust fallback
- LLM-driven decomposition bridges multi-hop reasoning gaps

**Core Innovation:**
Three-stage routing with deterministic rules → ASK verification → LLM fallback allows adaptive path selection without over-committing to SPARQL for complex questions.

---

## 2. Core Pipeline Architecture

### Step-by-Step Flow

The `hybrid_answer()` function in `hybrid_qa.py` implements six sequential steps with fallback logic:

#### Step 1: Entity Linking
**Entry:** Raw question string
**Function:** `link_entities_to_dbpedia(question, linker)` (from `graphrag_wiki.py`)
**Process:**
- DBpedia Spotlight API with `min_confidence=0.5`
- 5-stage mention extraction: quoted strings → noun chunks → NER → title-after-keyword → dedup
- Circuit breaker protection (3 failures, 120s cooldown)
- Fallback to spaCy NER if Spotlight fails
- Type validation via RELATION_TYPE_HINTS and TYPE_HINTS
- Return: List of Entity objects with URI, label, confidence, source

**Output:** `entities` list (Entity dataclass)

#### Step 2: Schema Discovery & Entity Profiling
**Input:** entities (with URIs)
**Function:** `build_entity_profile()` + `project()`
**Process:**
- For each linked entity URI, fetch:
  - Entity class (Person, Film, Company, etc.)
  - Actual properties (dbo: and dbp: prefixes)
  - Sample values for each property
- Build EntityProfile dataclass with: uri, label, all_classes, cached_classes, actual_properties, actual_property_set, all_2hop_chains
- SchemaProjection renders profile for:
  - SPARQL generation via `render_for_sparql()` (~300 tokens, includes prefixes, full dbo/dbp listings)
  - Routing decision via `render_for_routing()` (~150 tokens, condensed)
  - Text context via `render_for_text()` (~50 tokens, minimal)

**Completeness:** Categorical string (`"complete"`, `"partial"`, `"missing"`) via prefix-aware subset checks — only `dbo:` properties count
**Output:** `projections` list (SchemaProjection objects), `schema_ctx` (rendered SPARQL context)

#### Step 3: Route Decision (Three-Stage Routing)
**Input:** question, entities, projections, schema context
**Function:** `route_question()` (from `routing.py`)
**Process:**
1. **Stage 1 — Deterministic Rules** (`_rule_based_route()`):
   - Direct property match: target properties found in entity schema
   - 2-hop chain match: bridge + hop2 keywords with matching schema chains
   - Comparison routing: date/country property detection
   - Bridge/parent chain routing: bridge keyword + appropriate properties
   - Returns `('SPARQL', reason)` or `None`

2. **Stage 2 — SPARQL ASK Verification** (`verify_sparql_feasibility()`):
   - For bridge/hop2/parent rules: verify ASK query succeeds
   - Downgrades to `DECOMPOSE` if ASK fails (missing triple)
   - Returns `ask_verified` or `ask_missing`

3. **Stage 3 — LLM Property Extraction Fallback**:
   - If Stage 1 returns `None`, LLM extracts needed properties via `PROPERTY_EXTRACTION_PROMPT`
   - Extracted properties are re-fed into deterministic rules for a second pass
   - If still no match, defaults to TEXT route

**Output:** `route` ∈ {SPARQL, DECOMPOSE, TEXT}, `route_source` (reason tag)

#### Step 4: Direct SPARQL
**Condition:** `route == "SPARQL"` and `schema_ctx` is non-empty
**Function:** `sparql_solver.solve(question, schema_ctx, entities, llm_client, model)`
**Process:**
- Generate SPARQL query via `UNIFIED_SPARQL_PROMPT` using question + schema + entities
- Execute via DBpedia SPARQL endpoint with 60s timeout
- 2 total attempts (not 2 retries) on `URLError`, `ConnectionError`, `TimeoutError` only — HTTP status codes (429, 500) are not retried
- Extract final answer from result using `extract_final_answer()`
- Return: (answer_string, query_string, discovered_entities)

**Early Exit:** If answer found, return HybridResult with method="sparql"

#### Step 5: LogicRAG Decomposition
**Condition:** SPARQL answered=None (failed or route was DECOMPOSE)
**Function:** `_try_logicrag_decomposition(question, entities, schema_ctx, ...)`
**Process:**
- Decompose question into sub-questions via LLM + JSON parsing
- Build dependency graph (topological sort via Kahn's algorithm)
- Route each sub-Q via `_rule_based_route()`: SPARQL or TEXT
- Gather grounded answers and synthesize final answer
- Return: (answer_string, sub_question_log)

**Early Exit:** If answer found, return HybridResult with method="logicrag_decomposed"

**Special Tracking:** `sparql_fallback=True` if originally routed SPARQL but fell through

#### Step 5b: SPARQL Pre-Check (Safety Net)
**Condition:** LogicRAG failed, but entities exist + target_properties + schema_ctx
**Process:**
- For each entity, check ASK query for each target property
- If any property exists: attempt SPARQL generation again (second chance)
- Catches gaps from entity-linking failures where properties exist but routing missed them

**Early Exit:** If answer found, return HybridResult with sparql_fallback=True

#### Step 6: Text Fallback
**Input:** all_entities (seed + discovered neighbors), question
**Function:** `text_solver.solve(question, all_entities, ...)`
**Process:**
1. Context entity discovery: type-guided wikiPageWikiLink traversal (ASK limit=10)
2. Wikipedia retrieval: URI-based → label → DuckDuckGo → keywords
3. RAG indexing: chunk Wikipedia text, FAISS IndexFlatIP with all-MiniLM-L6-v2
4. Bi-encoder retrieval (top_k=20) + cross-encoder reranking (top_k=8)
5. Assemble 12K-char context
6. LLM reasoning with TEXT_REASON_PROMPT

**Output:** answer_string via `extract_final_answer()`

### Debug Info Assembly

During each step, metadata is collected:
- `entities_linked`: List of Entity objects with confidence scores
- `route`, `route_meta`: Routing decision + intermediate analysis
- `analysis`: QuestionAnalysis (hop_structure, comparison, target_properties, bridge keywords)
- `projections`: Per-entity class, completeness, missing hops
- `stage_timings`: elapsed seconds for entity linking, schema, routing, SPARQL, LogicRAG, text

All wrapped in try/except to prevent metadata failure from crashing the question.

---

## 3. Entity Linking Subsystem

### Spotlight API Configuration

**Endpoint:** `https://api.dbpedia-spotlight.org/en/annotate`
**Parameters:**
- `text`: Question string
- `confidence`: 0.5 (minimum acceptance threshold)
- `support`: 5 (default, counts in Wikipedia)

**Timeout:** 30s (via httpx client)

**Circuit Breaker** (`_SpotlightCircuitBreaker` at entity_linker.py:26):
- Threshold: 3 consecutive failures
- Cooldown: 120s (2 minutes)
- After cooldown, automatic retry

### Mention Extraction (5 Stages)

The system extracts entity mention candidates in order of confidence:

1. **Quoted Strings** (highest confidence):
   - Regex: `"([^"]+)"` and `'([^']+)'`
   - Direct pass to Spotlight

2. **Noun Chunks** (via spaCy):
   - spaCy `noun_chunks` iterator
   - Filter: drop single-letter chunks
   - Pass to Spotlight if Spotlight available

3. **Named Entity Recognition** (spaCy NER):
   - Extract all entities EXCEPT those with `CARDINAL`, `ORDINAL`, `QUANTITY`, `PERCENT` labels (exclusion filter, not inclusion)
   - Split entities containing " or " into parts

4. **Title-After-Keyword** (rule-based):
   - Pattern: `"who directed|who wrote|who starred" <Title>`
   - Extract proper nouns after keywords
   - High precision, low recall

5. **Deduplication**:
   - Lowercase + strip whitespace comparison
   - Merge multiple extractions of same entity
   - Keep highest confidence source

### Strip Words & Blocklist

**Strip Words (39 total):**
Common words removed from mentions before Spotlight query:
```
a, an, and, are, at, be, both, by, did, do, does, either, for, from,
had, has, have, how, in, is, it, its, not, of, on, or, that, the, this,
to, was, were, what, when, where, which, who, why, with
```

**Mention Blocklist (43 entities):**
High-frequency domain nouns dropped:
```
"actor", "actress", "age", "album", "author", "band", "book", "born", "city",
"company", "composer", "country", "date", "died", "different", "director",
"earlier", "fewer", "film", "first", "last", "later", "more", "movie",
"musician", "name", "novel", "number", "older", "performer", "person", "place",
"politician", "publisher", "same", "school", "singer", "song", "time",
"university", "writer", "year", "younger"
```

### Entity Object Structure

Each Entity is a dataclass with:
- `uri`: Full DBpedia resource URI (http://dbpedia.org/resource/...)
- `label`: Human-readable name (possibly disambiguation)
- `types`: list[str] — entity type URIs (default empty list)
- `abstract`: Optional[str] — entity abstract text (default None)
- `confidence`: Float [0, 1] from Spotlight (default 1.0)
- `source`: "dbpedia", "spotlight", "sparql", "logicrag_guessed", "wiki_resolve", etc.
- `discovery_relation`: Optional relation predicate that discovered this entity

### Type Hints & Validation

**RELATION_TYPE_HINTS** (regex pattern keys → single `dbo:` class string):
```python
{
    r'\bdirector\s+of\b': 'dbo:Film',
    r'\bdirected\b': 'dbo:Film',
    r'\bcast\s+of\b': 'dbo:Film',
    r'\bstarring\s+in\b': 'dbo:Film',
    r'\bauthor\s+of\b': 'dbo:Book',
    r'\bwrote\b': 'dbo:Book',
    # ... 13 total entries
}
```
Note: Semantics map the *relation phrase* to the *subject type* (e.g., "director of" → Film, not Person).

**TYPE_HINTS** (regex pattern keys → single `dbo:` class string):
```python
{
    r'\bfilm\b': 'dbo:Film',
    r'\bmovie\b': 'dbo:Film',
    r'\bdirector\b': 'dbo:Film',
    r'\bbook\b': 'dbo:Book',
    r'\bauthor\b': 'dbo:Book',
    # ... 14 total entries
}
```

### Confidence Stratification

Entities split into two tiers:
- **High confidence:** confidence ≥ 0.7 (prioritized for schema discovery)
- **Low confidence:** 0.5 ≤ confidence < 0.7 (fallback only)

### Unlinked Resolution Cascade (4 Layers)

When Spotlight fails or returns no results, attempt to recover:

1. **URI Probe** — Try construct URI from title:
   - Example: "Albert Einstein" → `http://dbpedia.org/resource/Albert_Einstein`
   - Live SPARQL query to verify entity exists

2. **Fulltext Search** — DBpedia SPARQL fulltext search:
   - `PREFIX dbo: <...> SELECT ?s WHERE { ?s rdfs:label ?label . FILTER(regex(?label, "pattern", "i")) }`
   - Return top match

3. **Wikipedia Resolver** — WikipediaResolver class:
   - Query Wikipedia search API
   - Extract redirect chains if applicable
   - Obtain Wikipedia title → convert to DBpedia URI

4. **SPARQL Label Fallback** — Direct label lookup:
   - SPARQL: `SELECT ?s WHERE { ?s rdfs:label "exact_label"@en }`
   - Return first match

### Jaccard Similarity Threshold

When resolving entities, similarity score computed:
```
similarity(label_a, label_b) = |set(tokens_a) ∩ set(tokens_b)| /
                               |set(tokens_a) ∪ set(tokens_b)|
```

**Threshold:** 0.5 (must share ≥50% of token vocabulary)

### LLM Disambiguation

When `bif:contains` text search or Wikipedia resolver returns multiple candidates (NOT Spotlight):
- LLM reranker callback created via `create_llm_reranker()`
- Call LLM with: mention, list of candidate URIs + labels + types, question context
- Returns best-matching Entity from candidates
- Temperature: 0, max_tokens: 10
- Max 2 retry attempts

---

## 4. Schema Discovery & Projection

### CACHED_CLASS_SCHEMAS (7 Classes)

Pre-computed schema for common DBpedia classes. Updated manually via offline SPARQL queries.

#### Person Class
```
dbo properties (ontology):
  birthDate, deathDate, birthPlace, deathPlace,
  parent, child, spouse, nationality, occupation,
  almaMater, residence, religion, ethnicity,
  father, mother, deathCause

dbp properties (wikipedia infobox):
  father, mother, birthDate, birthPlace, deathDate, deathPlace,
  spouse, children, nationality, occupation, education, residence

2-hop chains (reachable classes + hop-2 properties):
  parent → Person (birthDate, deathDate, birthPlace)
  father → Person (birthDate, deathDate, birthPlace)
  mother → Person (birthDate, deathDate, birthPlace)
  spouse → Person (birthDate, deathDate, birthPlace, nationality)
  child → Person (birthDate, deathDate, birthPlace)
  birthPlace → Place (country, isPartOf, populationTotal)
  deathPlace → Place (country, isPartOf)
```

#### Film Class
```
dbo properties:
  director, producer, starring, releaseDate, runtime,
  distributor, country, budget, gross, cinematography,
  musicComposer, editing, writer

dbp properties:
  director, producer, starring, released, runtime, studio,
  distributor, country, budget, gross, music, cinematography,
  editing, writer

2-hop chains:
  director → Person
  producer → Person
  starring → Person
  writer → Person
  musicComposer → Person
  distributor → Company
```

#### Company Class
```
dbo properties:
  foundedBy, foundingDate, locationCity, locationCountry,
  industry, numberOfEmployees, revenue, netIncome,
  parentCompany, subsidiary, keyPerson

dbp properties:
  founder, founded, location, headquarters, industry,
  employees, revenue, netIncome, parent, subsidiaries,
  keyPeople, products, services
```

#### Place Class
```
dbo properties:
  country, isPartOf, populationTotal, areaTotal,
  leader, leaderName, capital, largestCity

dbp properties:
  country, subdivision, population, area, leader,
  capital, largestCity, timezone
```

#### MusicalArtist Class
```
dbo properties:
  genre, recordLabel, associatedMusicalArtist, instrument,
  birthDate, birthPlace, activeYearsStartYear

dbp properties:
  genre, label, associatedActs, instrument,
  birthDate, birthPlace, yearsActive, origin
```

#### Book Class
```
dbo properties:
  author, publisher, publicationDate, numberOfPages,
  isbn, literaryGenre, language

dbp properties:
  author, publisher, published, pages, isbn, genre, language, country
```

#### MusicalWork Class
```
dbo properties:
  artist, genre, recordLabel, releaseDate, runtime,
  writer, producer, album

dbp properties:
  artist, genre, label, released, length, writer, producer
```

### PROPERTY_RANGES (30 Properties Mapped)

Maps `dbo:` property URIs to their range (expected target class). All entries use `dbo:` prefix only:

```
# Film → Person
dbo:director → Person
dbo:producer → Person
dbo:starring → Person
dbo:writer → Person
dbo:musicComposer → Person
dbo:cinematography → Person
dbo:editing → Person
# Film → Company
dbo:distributor → Company
# Book → Person/Company
dbo:author → Person
dbo:publisher → Company
# Person → Place
dbo:birthPlace → Place
dbo:deathPlace → Place
dbo:residence → Place
dbo:almaMater → Place
# Person → Person
dbo:parent → Person
dbo:father → Person
dbo:mother → Person
dbo:spouse → Person
dbo:child → Person
# Company → Person/Company/Place
dbo:foundedBy → Person
dbo:keyPerson → Person
dbo:parentCompany → Company
dbo:subsidiary → Company
dbo:locationCity → Place
dbo:locationCountry → Place
# Place → Place
dbo:country → Place
dbo:isPartOf → Place
dbo:capital → Place
# MusicalArtist/Work
dbo:associatedMusicalArtist → MusicalArtist
dbo:artist → Person
```

Used by routing to verify 2-hop chains without live schema queries.

### EntityProfile Dataclass

```python
@dataclass
class EntityProfile:
    uri: str
    label: str
    all_classes: list[str] = field(default_factory=list)       # All rdf:type classes
    cached_classes: list[str] = field(default_factory=list)     # Classes with cached schemas
    actual_properties: dict = field(default_factory=dict)       # {"dbo": [...], "dbp": [...]}
    actual_property_set: frozenset[str] = field(default_factory=frozenset)  # {"dbo:birthDate", ...}
    all_2hop_chains: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
```

**Construction:** `build_entity_profile(uri, label, schema_disc)`
1. Query all rdf:type classes via `schema_disc.get_all_cached_classes(uri)` (1 SPARQL query)
2. Query actual dbo: and dbp: properties via `schema_disc.get_actual_properties(uri)` (1 SPARQL query)
3. Build `actual_property_set`: frozenset of `"prefix:name"` strings
4. Merge 2-hop chains from ALL matching cached class schemas

### SchemaProjection Class

Implements 4-step projection algorithm:

1. **Class Selection** via `_score_class()`:
   - Score each cached class by relevance to question: 3 = contains target_property, 2 = contains bridge_property, 1 = contains hop2 property, 0 = no match
   - Pick highest-scoring class (tie-break: longer name = more specific)
   - No SPARQL fallback or type alias handling — pure scoring against cached schemas

2. **Properties Assembly**:
   - `relevant_properties`: intersection of analysis target properties with `actual_property_set`
   - Includes hop2 properties if present
   - Deduplicated, preserving order

3. **Chain Assembly**:
   - `relevant_chains` dict: {property: (target_class, target_properties)}
   - Filtered from `all_2hop_chains` — keeps chains whose entry property matches bridge or target
   - Example: `director` → (Person, [birthDate, deathDate, birthPlace, ...])

4. **Completeness Calculation** (categorical, not ratio-based):
   - Prefix-aware: only `dbo:` properties count (dbp-only properties cause false completeness)
   - For bridge questions: checks if `bridge_property` exists in `dbo:` properties
   - Result is a string: `"complete"` (all needed ⊆ have), `"partial"` (some overlap), `"missing"` (disjoint)
   - `missing_hops`: list of property names not found (e.g., `["hop1:director"]`)

### Rendering Modes

**`render_for_sparql()`** (~300 tokens):
```
SPARQL Prefixes:
  PREFIX dbr: <http://dbpedia.org/resource/>
  PREFIX dbo: <http://dbpedia.org/ontology/>
  PREFIX dbp: <http://dbpedia.org/property/>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

Entity: Albert Einstein
URI: <http://dbpedia.org/resource/Albert_Einstein>
Class: dbo:Person
  dbo: birthDate, deathDate, birthPlace, ...
  dbp: ...
  Property hints:
    - birthDate: Date of birth
  2-hop chains: dbo:birthPlace -> Place: country, ...

Actual properties found:
  dbo:birthDate -> 1879-03-14
  ...
```

**`render_for_routing()`** (~150 tokens):
```
Entity: Albert Einstein | Class: Person | Completeness: complete
Props: dbo:birthDate, dbo:deathDate, dbo:birthPlace
Chains: dbo:birthPlace -> Place: country, populationTotal
```

**`render_for_text()`** (~50 tokens):
```
Expect: Place via birthPlace
Discovery: wikiPageWikiLink + type filter dbo:Place
```

---

## 5. Routing Architecture

### Three-Stage Routing Design

Routing decides: SPARQL (complete schema, high confidence), DECOMPOSE (multi-hop with gaps), or TEXT (schema-sparse fallback).

#### Stage 1: Deterministic Rule-Based Routing

Function: `_rule_based_route(analysis, projections, debug=False)`

**Rule Execution Order:**

1. **Comparison + Date/Country Properties**:
   - Condition: `is_comparison=True` (≥2 entities, comparison word detected)
   - Check property intersection (all entities must have property):
   ```python
   common_props = intersection(entity1.props, entity2.props, ...)
   if common_props & DATE_PROPERTIES:  # birthDate, deathDate, etc.
       return ("SPARQL", "rule:direct_date_comparison")
   if (country_keywords & question.keywords) and (common_props & COUNTRY_PROPERTIES):
       return ("SPARQL", "rule:direct_country_comparison")
   ```
   - Rationale: Comparisons need complete data across all entities; partial data leads to wrong conclusions

2. **Comparison Bridge Chain**:
   - Condition: `is_comparison=True`, `bridge_keyword` (director, producer, etc.)
   - Check if bridge property exists in schema AND target class has date properties:
   ```python
   if bridge_property in proj.relevant_chains:
       target_cls, target_props = proj.relevant_chains[bridge_property]
       if any(p in target_props for p in ["birthDate", "deathDate", "foundingDate"]):
           return ("SPARQL", "rule:bridge_comparison")
   ```
   - Example: "Is X's director older than Y's director?" → director chain → Person → birthDate

3. **Non-Comparison Bridge Hop2**:
   - Condition: `bridge_keyword` (director, producer) AND `hop2_keyword` (birthDate, country, etc.)
   - Check if bridge property exists in actual properties:
   ```python
   if f"dbo:{bridge_property}" in all_actual_properties:
       return ("SPARQL", "rule:bridge_hop2")
   ```
   - Example: "Who directed the film whose composer was born in France?" → director chain → Person (composer exists) → birthPlace

4. **Parent Chain**:
   - Condition: `bridge_keyword ∈ {father, mother, parent, paternal, maternal}`
   - Check if parent properties exist:
   ```python
   if all_actual_properties & PARENT_PROPS:  # dbo:parent, dbo:father, dbo:mother, dbp:father, dbp:mother
       return ("SPARQL", "rule:parent_chain")
   ```

5. **Direct Property Match**:
   - Condition: Single-hop, `target_properties` (keywords → properties) found in schema
   - For comparisons: require property in ALL entities (intersection)
   - For non-comparisons: require property in ANY entity (union)
   ```python
   if analysis.is_comparison:
       matched = target_properties & common_props
   else:
       matched = target_properties & all_actual_properties
   if matched:
       return ("SPARQL", "rule:direct_property")
   ```

**Gap Analysis Debug Mode:**
If no rules match, return `None` with debug output showing:
- Actual properties found in entity schema
- Target properties extracted from question
- Recommendation: "Route: None (inconclusive → LLM decides)"

#### Stage 2: SPARQL ASK Verification

Function: `verify_sparql_feasibility(analysis, projections, entities, kg_client, rule_result)`

**Triggered for:** bridge/hop2/parent rules only
**Rationale:** Rules are necessary but not sufficient; must verify triples actually exist

**ASK Query Template** (for bridge hop2):
```sparql
PREFIX dbo: <http://dbpedia.org/ontology/>

ASK {
  <URI_ENTITY> dbo:director ?person .
  ?person dbo:birthDate ?date
}
```

**Outcomes:**
- `ask_verified=True`: Triple exists, proceed with SPARQL route
- `ask_missing=True`: Triple doesn't exist, downgrade to DECOMPOSE or TEXT

**Special Handling:**
- Direct property rules (direct_date_comparison, direct_property): skip ASK (property already in schema)
- Bridge rules: perform ASK only on hop2 part

#### Stage 3: LLM Property Extraction Fallback

Function: `_extract_properties_llm()` (called from `route_question()`)

**Triggered:** When Stage 1 returns `None` (no deterministic rule matched)

**Mechanism:** NOT a binary SPARQL/TEXT LLM decision. Instead:
1. LLM extracts needed property names from question + schema via `PROPERTY_EXTRACTION_PROMPT`
2. Extracted properties are merged into the analysis via `enrich_analysis()`
3. Deterministic rules re-run with the enriched property set
4. If still no match → default to TEXT

**Prompt:**
```
Given this question and DBpedia schema, which properties are needed to answer it?

{schema}

QUESTION: {question}

Output a comma-separated list of property names (no dbo:/dbp: prefix).
Only output properties listed above. For 2-hop paths (e.g., "director's birthdate"),
list both: director, birthDate
If no properties can help, output: NONE

Properties:
```

### QuestionAnalysis

Function: `analyze_question(question, profiles)`

Extracts structural features:

**hop_structure**: String enum — `"direct"`, `"bridge"`, `"comparison"`, or `"compositional"`
**is_comparison**: Boolean, ≥2 entities + comparison word detected

**target_keywords_set**: Keywords extracted from question
```python
# Keywords mapped to property LISTS (53 keys, values are lists)
KEYWORD_TO_PROPERTIES: dict[str, list[str]] = {
    "director": ["director"], "directed": ["director"], "direct": ["director"],
    "producer": ["producer"], "produced": ["producer"],
    "writer": ["writer", "author"], "wrote": ["writer", "author"],
    "author": ["author"],
    "composer": ["musicComposer"],
    "starring": ["starring"], "actor": ["starring"], "actress": ["starring"],
    "founded": ["foundedBy", "foundingDate", "foundingYear"],
    "founder": ["foundedBy"],
    "father": ["father", "parent"], "mother": ["mother", "parent"],
    "spouse": ["spouse"], "married": ["spouse"],
    "born": ["birthDate", "birthPlace"],
    "died": ["deathDate", "deathPlace"],
    "country": ["country", "nationality"],
    "nationality": ["nationality", "citizenship"],
    "capital": ["capital"],
    "population": ["populationTotal"],
    # ... 30+ more mappings (53 total keys)
}
```

**target_properties**: Set of ontology properties needed to answer

**bridge_keyword**: First 2-hop keyword (director, producer, etc.)
**bridge_property**: Property name for bridge (director → dbo:director)

**hop2_keyword**: Property target in second hop (birthDate, country, etc.)
**hop2_properties**: Set of properties for hop2

### COMPARISON_WORDS Set

Indicators of comparison questions (13 words):
```python
{"earlier", "later", "older", "younger", "first", "before",
 "after", "same", "both", "more", "less", "longer", "shorter"}
```

Example: "Who is older, Einstein or Marie Curie?" → is_comparison=True

### _BRIDGE_KEYWORDS Set

Bridge relations for hop-1 detection (27 keywords, in `question_analysis.py`):
```
director, directed, direct, producer, produced, writer, wrote, written,
author, composer, starring, actor, actress, founded, founder,
father, mother, parent, paternal, maternal, performer,
spouse, married, wife, husband, grandfather, grandmother
```

### _HOP2_KEYWORDS Set

Second-hop property keywords (16 keywords, in `question_analysis.py`):
```
born, birth, died, death, country, nationality, capital,
citizenship, citizen, education, studied, attended, educated,
spouse, wife, husband
```

### DATE_PROPERTIES & COUNTRY_PROPERTIES

**_DATE_PROPERTIES** (13):
```
dbo:birthDate, dbp:birthDate, dbo:deathDate, dbp:deathDate,
dbo:activeYearsStartYear, dbp:established, dbo:foundingYear,
dbp:foundingDate, dbo:releaseDate, dbp:released,
dbo:publicationDate, dbp:publicationDate, dbp:published
```

**_COUNTRY_PROPERTIES** (10):
```
dbo:nationality, dbp:nationality, dbo:country, dbp:country,
dbo:citizenship, dbp:citizenship, dbo:birthPlace, dbp:birthPlace,
dbo:stateOfOrigin, dbp:placeOfBirth
```

---

## 6. SPARQL Solver

### Overview

Module: `sparql_solver.py` (233 lines)
Entry point: `solve(question, schema_ctx, entities, llm_client, model, verbose, debug)`

### Execution Flow

1. **Generation**: LLM generates SPARQL query via `UNIFIED_SPARQL_PROMPT`
2. **Execution**: Query executed against DBpedia SPARQL endpoint (60s timeout)
3. **Retry**: 2 retries on transient errors with exponential backoff
4. **Answer Extraction**: Parse SPARQL result, extract final answer

### Read-Only Guard

Security check via regex injection prevention:

```python
_WRITE_OPS = re.compile(
    r'\b(INSERT|DELETE|DROP|CLEAR|CREATE|LOAD|COPY|MOVE|ADD|SERVICE)\b',
    re.IGNORECASE
)
# 10 SPARQL Update keywords blocked via regex
```

Prevents accidental write operations on the live DBpedia endpoint.

### UNIFIED_SPARQL_PROMPT

Template for SPARQL generation:

```
You are a SPARQL expert for DBpedia. Generate a query to answer the question.

{schema}
{known_facts_block}
QUESTION: {question}

RULES:
1. Use the EXACT entity URIs provided in the schema
2. For multi-hop queries, chain triple patterns using dbo: properties (they return URIs, not strings).
   Use the "2-Hop chains" section in the schema to identify valid chains.
   Example: if schema shows dbo:director -> Person: birthDate, write:
   ?film dbo:director ?person . ?person dbo:birthDate ?date .
3. Prefer dbo: properties for multi-hop joins; only use dbp: for single-hop leaf values
4. Always add LIMIT 100
5. For comparison questions (who/which was born first, etc.), return data for BOTH entities so results can be compared
6. If an entity URI contains special characters (apostrophes, commas, parentheses), use full IRI syntax <http://dbpedia.org/resource/...> instead of dbr: prefix
7. If known facts provide entity URIs, use those URIs directly in your query
8. Generate ONLY the SPARQL query, no explanation
```

**Template Variables:**
- `schema`: SchemaProjection rendering (~300 tokens)
- `known_facts_block`: Previously resolved facts from LogicRAG sub-questions (empty for direct SPARQL)
- `question`: Original question string

**Key Design Rules:**
- LIMIT 100: Prevents runaway result sets
- dbo: for joins, dbp: for leaf values only
- Chain patterns via 2-hop schema hints (not intermediate variables)

### Execution

**Endpoint:** `https://dbpedia.org/sparql`
**Method:** POST via SPARQLWrapper
**Format:** application/json (SPARQL JSON format)

**Timeout:** 60 seconds per query (via `execute_sparql(timeout=60)` in `sparql_evaluation.py:274`)
**Rate Limiting:** 10 queries/second via token bucket (DBpediaSPARQLClient, `dbpedia/client.py:63`)

### Retry Logic

SPARQL retry is handled at two levels:

**Level 1 — `execute_sparql()` (network retries)** in `sparql_evaluation.py`:
- `max_retries=2` → `range(max_retries)` = 2 total attempts
- Wait: `2 * (attempt + 1)` seconds between attempts (2s, 4s)
- Catches ONLY: `urllib.error.URLError`, `ConnectionError`, `TimeoutError`
- HTTP error codes (429, 500, etc.) fall to generic `except Exception` and return immediately without retry
- Does NOT retry on SPARQL syntax errors or any HTTP status code errors

**Level 2 — `solve()` (LLM-driven query correction)** in `sparql_solver.py`:
- If first query returns empty results or error, generates retry prompt via `SPARQL_RETRY_PROMPT`
- LLM generates a corrected SPARQL query
- Corrected query executed via `execute_sparql()` (which has its own network retries)
- Total: up to 2 LLM-generated queries × 2 network attempts each = 4 SPARQL executions max

### URI Sanitization

Special characters in entity URIs converted to proper IRIs:

```python
def escape_sparql_uri(uri):
    # Examples:
    # "Albert Einstein" → "Albert%20Einstein"
    # "Bach (composer)" → "Bach%20%28composer%29"
    # Result: <http://dbpedia.org/resource/Bach%20%28composer%29>
```

Used in SPARQL query generation to handle names with spaces, parentheses, etc.

### NULL_ANSWERS Set (9 Strings)

Responses to filter as non-answers:

```python
NULL_ANSWERS = {
    "could not",
    "n/a",
    "unknown",
    "none",
    "not found",
    "no answer",
    "could not determine",
    "no information",
    "not available",
}
```

If LLM answer extraction returns any of these strings, treated as answer=None (fallthrough).

### LIST_ARTICLE_PATTERN

Regex to reject list/disambiguation pages as SPARQL results:

```python
pattern = re.compile(
    r"(?:^|/)List[_ ]of[_ ]|discography$|filmography$|bibliography$|Awards[_ ]and[_ ]nominations",
    re.IGNORECASE
)
```

Note: URI-aware patterns (not start-anchored for "List" — handles URI paths). End-anchored for `discography$`, `filmography$`, `bibliography$`.

### Answer Extraction

Post-execution logic:

1. **Parse SPARQL JSON result** (variable bindings)
2. **Format results** (first 20 rows, variable names, values)
3. **Call LLM** to extract answer from formatted results (first 20 rows)
4. **Apply filters:**
   - If answer in NULL_ANSWERS → return None
   - If answer matches LIST_ARTICLE_PATTERN → return None
   - Otherwise → return answer

### SPARQL_RETRY_PROMPT

Used when SPARQL query fails and needs correction (imported from `sparql_evaluation.py`):

```
Your previous SPARQL query returned {problem}.

Previous query:
{sparql}

Fix the query using the schema below.

{schema}

QUESTION: {question}

Generate ONLY the corrected SPARQL query:
```

Note: This is for retrying *failed queries*, not for answer extraction from results.

### Metadata Tracking

Module-level dict: `_last_solve_meta`

```python
_last_solve_meta = {
    "sparql_gen_prompt": str,      # Full prompt sent to LLM for SPARQL generation
    "sparql_gen_response": str,    # Raw LLM response
    "initial_query": str,          # Extracted & sanitized SPARQL query
    "initial_results": list,       # First 20 result rows
    "initial_result_count": int,   # Total result count
    "initial_error": str,          # Error message (if any)
    "retry_prompt": str,           # Retry prompt (if retry triggered)
    "retry_response": str,         # Retry LLM response
    "retry_query": str,            # Corrected SPARQL query
    "retry_results": list,         # Retry result rows
    "retry_result_count": int,     # Retry result count
    "retry_error": str,            # Retry error
    "used_retry": bool,            # Whether retry was triggered
}
```

Stored in HybridResult.debug_info for post-execution analysis.

---

## 7. LogicRAG Decomposition

### Overview

Module: `logicrag_decomposition.py` (758 lines)
Entry point: `_try_logicrag_decomposition(question, entities, schema_ctx, llm_client, model, ...)`

**Purpose:** Handle multi-hop questions with incomplete schema via sub-question decomposition

### SubQuestion Dataclass

```python
@dataclass
class SubQuestion:
    text: str
    depends_on: list[int] = field(default_factory=list)  # Indices this depends on
    original_idx: int = -1  # Position in decomposed list
    answer: str | None = None  # Grounded answer (filled during resolution)
    method: str | None = None  # "sparql", "text", "fallback"
    entities: list[Entity] | None = None  # Linked entities for this sub-Q
```

### Three Prompt Templates

#### 1. DECOMPOSE_PROMPT (LLM-driven decomposition)

```
Decompose this multi-hop question into independent sub-questions.

QUESTION: {question}
ENTITIES: {entity_names}
SCHEMA SUMMARY: {schema_summary}
{type_hint}
Rules:
- Each sub-question should ask for ONE fact
- If a sub-question needs the answer from another, mark the dependency
- For comparison questions: extract the same property for each entity (parallel, no dependencies)
- For bridge questions: first find the intermediate entity, then ask about its property
- If the question is simple (single fact lookup), return exactly 1 sub-question with no dependencies

Return JSON:
{
  "sub_questions": ["sub-Q1 text", "sub-Q2 text", ...],
  "dependencies": [[dependent_idx, dependency_idx], ...]
}

Example - Bridge: "When was the director of Inception born?"
{
  "sub_questions": ["Who directed Inception?", "When was [Q0] born?"],
  "dependencies": [[1, 0]]
}
```

**Processing:**
- LLM returns JSON with `sub_questions` (list of strings) and `dependencies` (list of [idx, idx] pairs)
- Parser validates: dependency indices within range, no cycles
- Topological sort via Kahn's algorithm (handle cycles with sequential fallback)

#### 2. DEPENDENCY_RAG_PROMPT (Accumulated-facts check)

```
We pre-parsed the question into a list of dependencies, and the
dependencies are sorted in a topological order, below is the question, the information
summary, and the decomposed dependencies:

Question: {question}

Available Information:
{info_summary}

Decomposed dependencies:
{dependencies}

Current dependency to be answered:
{current_dependency}

Please analyze:
1. Can the question be answered completely with this information? (Yes/No)
2. Summarize our current understanding based on available information.

Please format your response as a JSON object with these keys:
- "can_answer": boolean
- "current_understanding": string
```

**Used for:** Checking if accumulated facts from resolved sub-Qs already suffice to answer the full question — enables early exit without resolving all sub-Qs

#### 3. FINAL_ANSWER_PROMPT (Direct answer synthesis)

```
You must give ONLY the direct answer in the most concise way possible.
DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

Question: {question}

Information Summary:
{info_summary}

End your response with exactly: FINAL ANSWER: <your answer>
```

### Decomposition Algorithm

**Step 1: Schema Summarization**
```python
schema_summary = """
Entity 1 (Type): property1 → Type2, property2 (scalar)
Entity 2 (Type): property1 → Type3, property3 (scalar)
"""
```

**Step 2: Type Hint Construction**
```python
type_hints = {entity_label: entity_type for entity in entities}
# Example: {"Inception": "Film", "Christopher Nolan": "Person"}
```

**Step 3: LLM Decomposition**
- Call LLM with DECOMPOSE_PROMPT
- Max tokens: 1000 (default from `call_llm()` in `llm_provider.py`)
- Temperature: 0 (deterministic)
- Parse JSON response

**Step 4: Validation**
```python
# Check:
# 1. All dependency IDs are valid (≤ sub-Q count)
# 2. No cycles in dependency graph
# 3. ≤ 6 sub-questions (hardcoded cap)
# 4. Each sub-Q is non-empty
```

**Step 5: Topological Sort**
```python
# Kahn's algorithm for DAG ordering
# Input: dependency graph
# Output: ordered list of sub-Q IDs
# Fallback: Sequential 0..n-1 if cycle detected
```

**Step 6: Sub-Question Routing**
For each sub-Q (in topological order):
```python
# Substitute previous answers into sub-Q text
sub_q_with_answers = _substitute_answers(sub_q.text, known_facts)

# Route via _rule_based_route() (rule-based only, no LLM route prompt)
route_result = _rule_based_route(sub_analysis, sub_projections, debug=debug)

# Only attempt SPARQL when routing rules explicitly matched;
# bool(combined_schema) is nearly always True (any entity with a URI),
# which defeated routing and wasted LLM+SPARQL calls on TEXT sub-Qs.
attempt_sparql = route_result is not None

if attempt_sparql and combined_schema:
    answer = sparql_solver.solve(...)
if not answer:  # TEXT fallback
    answer = text_solver.solve(...)
```

### Answer Substitution (3-Pass Strategy)

When sub-questions reference previous answers, substitute before routing:

**Pass 1 (Indexed Placeholders):**
```
"Who directed [Q0]?" where Q0 answer = "Parasite"
→ Replace [Q0] with "Parasite" (case-insensitive, exact match)
```

**Pass 2 (Positional Fallback):**
```
Any remaining [.*?] brackets → Replace with next unmatched answer (sorted index order)
Only applies to indices not already handled in Pass 1
```

**Pass 3 (Direct Token Substitution):**
```
"Q0", "sub-question 0", "answer 0" → Replace with answer (case-insensitive)
Only if answer text not already in result string
```

### Entity Resolution for Bridge Linking

When decomposition introduces new entities not in seed set (`_make_fallback_entity()`):
- Construct URI from name: `http://dbpedia.org/resource/{name}`
- Verify entity exists via `ASK { <uri> ?p ?o }` (existence check, NOT `rdf:type` lookup)
- Add to all_entities with `source="logicrag_guessed"`

### Sub-Question Routing Example

```
Question: "Who directed the film that won Best Picture in 2020?"

Decomposition:
1. Which film won Best Picture in 2020?
2. Who directed that film?

Sub-Q 1 Routing:
- Keywords: "won", "award", "Best Picture"
- Entities: none extracted
- Rule: Comparison? No. Direct property? No (no entity for "Best Picture award")
- Route: TEXT (search Wikipedia for "Academy Awards 2020")

Sub-Q 2 Routing:
- Keywords: "directed"
- Entities: [Film from Sub-Q 1 answer]
- Rule: Direct property? Yes (director property exists for Film class)
- Route: SPARQL

Synthesis:
- Sub-Q 1: Answer = "Parasite"
- Sub-Q 2: Answer = "Bong Joon-ho"
- Final: "Bong Joon-ho"
```

### Early Stopping via DEPENDENCY_RAG_PROMPT

After resolving each sub-Q, check if accumulated facts already answer the full question:
```python
# Call LLM with accumulated info_summary + dependencies
result = call_llm(DEPENDENCY_RAG_PROMPT, question, info_summary, dependencies)

if result["can_answer"]:
    # Skip remaining sub-Qs, go directly to FINAL_ANSWER_PROMPT
    return synthesize_answer(question, info_summary)
```

Enables early exit when the first 1-2 sub-Qs provide enough information.

### Orchestration Flow

```python
def _try_logicrag_decomposition(...):
    # 1. Generate schema summary
    # 2. Decompose via LLM
    # 3. Build dependency graph
    # 4. Topological sort
    # 5. For each sub-Q:
    #    a. Substitute previous answers
    #    b. Route (rule-based or LLM)
    #    c. Execute (SPARQL or TEXT)
    #    d. Store answer
    # 6. Synthesize final answer via LLM
    # 7. Return (answer, sub_q_log)
```

**Sub-Q Log:** Full trace of decomposition, routing, execution per sub-Q (for debugging)

---

## 8. Text Retrieval & RAG Pipeline

### Overview

Module: `text_solver.py` (304 lines)
Entry point: `solve(question, entities, known_facts, relation_hints, llm_client, model, seed_titles, ...)`

### Execution Flow

1. **Wikipedia Retrieval**: Fetch Wikipedia text for all entities
2. **Chunking**: Split text into 512-char sections with 64-char overlap
3. **FAISS Indexing**: Build flat index with all-MiniLM-L6-v2 embeddings
4. **Bi-Encoder Retrieval**: Retrieve top-20 chunks via dot-product similarity (`text_solver.py:259`)
5. **Seed Boost**: Upweight chunks from seed entity pages (1.2x multiplier)
6. **Cross-Encoder Reranking**: Score top-20 with ms-marco-MiniLM-L-6-v2
7. **Context Assembly**: Select top-8 chunks, assemble 12K-char context
8. **LLM Reasoning**: Prompt with context + question, extract final answer

### WikiChunk Dataclass

```python
@dataclass
class WikiChunk:
    text: str              # Chunk content (up to 512 chars)
    page_title: str        # Wikipedia article title
    section_header: str    # Section heading (empty for intro)
    chunk_index: int       # Sequential index within article
    char_offset: int       # Character offset in original article text
```

Note: No `embedding` or `relevance_score` fields — embeddings stored separately in FAISS index and SQLite cache; scores computed during retrieval.

### Chunking Strategy

**Parameters:**
- Target chunk size: 512 characters (soft cap)
- Overlap: 64 characters (for context continuity)
- Minimum chunk: 100 characters

**Algorithm (Section-Aware):**
1. Parse Wikipedia article into sections (heading hierarchy)
2. For each section, split into 512-char chunks
3. Add 64-char overlap from previous chunk if available
4. Include section heading in each chunk for context

**Example:**
```
Article: Albert Einstein

Intro (200 chars):
"Albert Einstein was a German-born theoretical physicist..."

Section "Early Life" (4000 chars):
[chunk 1: 512 chars, 0-64 overlap from intro]
[chunk 2: 512 chars, includes heading, 64 overlap from chunk 1]
[chunk 3: 512 chars, etc.]

Section "Career" (5000 chars):
[chunk 1: 512 chars, includes heading, etc.]
```

### FAISS IndexFlatIP

**Model:** `all-MiniLM-L6-v2` (384-dimensional)

**Configuration:**
- Index type: IndexFlatIP (flat brute-force, exact inner-product search)
- No quantization (exact matching)
- No HNSW (no approximation)
- No batch size — all chunks indexed at once via single `faiss.IndexFlatIP.add()` call

**Rationale:** Small corpus (~1K chunks per question), exact search preferred over approximate.

### Bi-Encoder Retrieval

**Step 1: Over-fetch**
```python
fetch_k = min(top_k * 2, index.ntotal)  # 2x oversample, capped at corpus size
scores, indices = index.search(query_embedding, fetch_k)
# With top_k=20 at call site → fetch_k=40 candidates
```

**Step 2: Seed Boost**
```python
for i, (chunk, score) in enumerate(scored_chunks):
    if chunk.title in seed_titles:
        scores[i] *= 1.2  # Boost seed pages
    scored_chunks[i] = (chunk, scores[i])
```

**Seed Titles:** Articles for seed entities (from Entity Linking step 1)

**Seed Boost Multiplier:** 1.2 (20% upweight)

**Rationale:** Seed entities are direct targets; prioritize their pages.

**Default top_k:** 30 in `RAGIndex.retrieve()` function signature, but called with `top_k=20` at the `text_solver.solve()` call site

### Cross-Encoder Reranking

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Input:** (question, chunk) pairs (from bi-encoder top-20)

**Scoring:**
```python
for chunk in bi_encoder_topk:
    # Augment question with relation hints
    augmented_q = question
    if chunk.page_title in relation_hints:
        relation = relation_hints[chunk.page_title]
        augmented_q = f"{query} [context: discovered via {relation} relation]"

    # Score
    score = reranker.predict((augmented_q, chunk.text))
    chunk.relevance_score = score
```

**Relation Hints:** Keywords indicating semantic relations (from entity discovery)
- Example: {"Marie Curie": "spouse", "France": "birth place"}

**Output top_k:** 15 in `Reranker.rerank()` function signature, but called with `top_k=8` at the `text_solver.solve()` call site

**Rationale:** Cross-encoders provide fine-grained ranking; BERT-style models capture complex semantic interactions.

### Context Assembly

**Algorithm** (`_assemble_from_chunks()` in `text_solver.py`):
```
# Function defaults: max_chars=25,000, max_per_entity=4,000
# RAG call site overrides: max_chars=12,000 (tighter budget for RAG chunks)
# Keyword path uses defaults: max_chars=25,000

# Prepend KG facts if available
if kg_facts: parts.append(kg_facts)

for chunk in reranked_chunks:
    entry = f"=== {chunk.page_title} / {chunk.section_header} ===\n{chunk.text}"

    if total + len(entry) > max_chars:
        break  # Hard budget stop — no partial chunks

    if entity_used[chunk.page_title] + len(entry) > max_per_entity:
        continue  # Skip — this entity already has enough context

    parts.append(entry)
```

**Output:** Concatenated text joined with `\n\n`, capped at 12,000 chars (RAG) or 25,000 chars (keyword)

### Legacy Keyword Path (Fallback)

If RAG dependencies (sentence-transformers, FAISS) not installed:

**Keyword Scoring** (`extract_relevant_sections()` in `graphrag_wiki.py`):
```python
# Paragraph-level rare-keyword scoring, not chunk-level
# Splits text into paragraphs, scores each by keyword hits
for para in paragraphs:
    rare_keywords = [kw for kw in question_keywords if freq(kw) < threshold]
    score = sum(2 if kw in rare_keywords else 1 for kw in hits)
    # Relation-guided 2x boost via RELATION_SECTION_BOOST
    if relation_hint in RELATION_SECTION_BOOST:
        if any(term in para for term in RELATION_SECTION_BOOST[relation_hint]):
            score *= 2
```

**RELATION_SECTION_BOOST (25 entries):**
```python
RELATION_SECTION_BOOST = {
    "director": ["early life", "personal life", "career", "biography"],
    "producer": ["early life", "career", "biography"],
    "author": ["early life", "career", "biography", "works"],
    # ... 25 total entries mapping relation predicates to section keywords
    # Full list: director, producer, author, writer, starring, birthPlace,
    # birthDate, deathPlace, deathDate, parent, father, mother, spouse,
    # country, foundedBy, award, performer, artist, almaMater, nationality,
    # genre, deathCause, publicationDate, releaseDate, language
}
```

**Budget Allocation:**
- Seed entities: 60% of 25K = 15,000 chars
- Bridge entities: 30% of 25K = 7,500 chars
- Remaining context: 2,500 chars

### Wikipedia Retrieval Cascade

**Priority Order:**

1. **URI-Based Lookup**:
   - Convert DBpedia URI → Wikipedia title via `uri_to_wiki_title()`
   - Direct API fetch: `action=query&titles={title}`

2. **Label Lookup**:
   - Query Wikipedia search API: `action=query&list=search&srsearch={label}`
   - Extract top-1 result

3. **DuckDuckGo Search**:
   - Query DuckDuckGo: `{label} Wikipedia`
   - Extract first Wikipedia result
   - Circuit breaker: 3 failures, 60s cooldown (DDG rate-limits aggressively)

4. **Question Keywords**:
   - Extract key nouns from question
   - Query Wikipedia search API with question terms
   - Fallback if entity-based retrieval fails

**Rate Limiting:**
- Wikipedia API: 0.5s delay between calls (politeness)
- DuckDuckGo: Circuit breaker (3 failures, 60s cooldown)
- HTTP timeout: 15s per request

### SQLite Cache (wiki_cache.db)

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS wiki_pages (
    title TEXT PRIMARY KEY,
    chunks_json TEXT,      -- JSON-serialized list of WikiChunk dicts
    embeddings BLOB,       -- numpy array of all chunk embeddings
    fetched_at REAL        -- Unix timestamp
);
```

Single table stores chunks + embeddings together per page title.

**Eviction Policy:**
- Time-based: 7 days TTL (`_EVICTION_DAYS = 7` at `rag/cache.py:13`)
- Eviction runs on each cache instantiation: `DELETE FROM wiki_pages WHERE fetched_at < cutoff`

**Location:** `data/wiki_cache.db` (auto-created)

### Two Prompt Templates

#### TEXT_REASON_PROMPT

```
Answer this question using the sources below.

Question: {question}

{kg_facts}Wikipedia text:
{context}

Instructions:
- Knowledge graph facts (if present) are verified structured data — trust them
- Use Wikipedia text to fill gaps not covered by KG facts
- Show brief step-by-step reasoning
- End with exactly: FINAL ANSWER: <answer>
```

**Max tokens:** 1000 (default from `call_llm`)
**Temperature:** 0

#### TEXT_SUBQ_PROMPT

```
Answer this sub-question using the sources below.

Sub-question: {question}

{kg_facts}Wikipedia text:
{context}

Instructions:
- Knowledge graph facts (if present) are verified structured data -- trust them
- Use Wikipedia text to fill gaps not covered by KG facts
- Give a brief, factual answer
- End with exactly: FINAL ANSWER: <answer>
```

Used during LogicRAG sub-question execution.

---

## 9. Entity Discovery

### Overview

Function: `discover_context_entities(seed_entities, projections, kg_client, question)`

**Purpose:** Expand seed entities with neighbors to provide context for text reasoning

### Type-Guided wikiPageWikiLink Traversal

**Algorithm:**
```python
def discover_context_entities(
    seed_entities, projections, kg_client,
    question="", max_per_seed=5,     # Default: top-5 per seed entity
) -> list[Entity]:
    # For each seed entity:
    # 1. Determine expected type from SchemaProjection's expected_target_type
    # 2. SPARQL query: wikiPageWikiLink neighbors (limit=30, optionally type-filtered)
    # 3. Score candidates via _score_candidates() (two-pass)
    # 4. Keep top max_per_seed (default 5) by score
    # Fallback chain:
    #   1. Typed neighbors (wikiPageWikiLink + type filter)
    #   2. Untyped neighbors (wikiPageWikiLink, no type)
    #   3. Return seed entities only
```

### Candidate Scoring (Two-Pass)

Function: `_score_candidates(candidates, seed_uri, question, hop2_property, actual_uris, kg_client, ask_limit=10)`

**Pass 1 — Cheap signals (no SPARQL):**
```python
# Label overlap with question words (+1 per overlapping word, stop words excluded)
# Already-known URI penalty (-2): if URI already in entity's actual property values
```

**Pass 2 — SPARQL ASK on top candidates only (up to ask_limit=10):**
```python
# Backlink check (+3): ASK { <candidate> wikiPageWikiLink <seed> }
#   Single boolean ASK query, flat +3 if backlink exists (not count-based)
# Hop2 property match (+2): ASK { <candidate> dbo:{hop2_property} ?x }
#   Checks if candidate has the target hop-2 property
```

### ASK Limit

Maximum 10 ASK queries per entity discovery (control execution time, `ask_limit=10`).

### Fallback Chain

If typed neighbors insufficient:
1. **Typed neighbors:** wikiPageWikiLink + type filter (limit=30)
2. **Untyped neighbors:** wikiPageWikiLink without type (if typed returns empty)
3. **Seed-only:** Return just seed entities (if untyped also returns nothing)

---

## 10. LLM Provider Infrastructure

### Overview

Module: `llm_provider.py` (102 lines)

Unified abstraction for three LLM providers: Groq, GitHub Models, Ollama

### Provider Abstraction

**Factory Function:**
```python
PROVIDERS = {
    "groq":   {"base_url": "https://api.groq.com/openai/v1",        "env_key": "GROQ_API_KEY"},
    "github": {"base_url": "https://models.inference.ai.azure.com",  "env_key": "GITHUB_TOKEN"},
    "ollama": {"base_url": "http://localhost:11434/v1",              "env_key": None},
}

def create_llm_client(provider: str = "groq", api_key: str | None = None) -> OpenAI:
    cfg = PROVIDERS.get(provider)
    if not cfg:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
    if cfg["env_key"] is None:
        key = api_key or "ollama"
    else:
        key = api_key or os.environ.get(cfg["env_key"])
        if not key:
            raise ValueError(f"API key required: pass --api-key or set {cfg['env_key']}")
    return OpenAI(base_url=cfg["base_url"], api_key=key)
```

All providers use a single `OpenAI()` call — no separate `groq` import.

### call_llm() Retry Strategy

**Function:**
```python
def call_llm(
    client, prompt: str, model: str,          # model is REQUIRED 3rd positional arg
    max_attempts: int = 5, initial_wait: float = 10.0,
    max_tokens: int = 1000, temperature: float = 0,
) -> str:
    """Make an LLM call with retry on transient errors (429, 500, 503, timeout)."""
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            if _is_retryable(e) and attempt < max_attempts - 1:
                wait = initial_wait * (2 ** attempt)
                time.sleep(wait)
            else:
                raise
```

**Retryable check:** `_is_retryable(e)` — returns `True` for `openai.RateLimitError` or if error string contains any of 7 patterns:
```
"429", "500", "502", "503", "504", "timeout", "connection"
```

**Non-Retryable:** All other exceptions (400, 401, 404, etc.) re-raised immediately.

**Backoff Schedule** (exponential, no cap):
```
Attempt 1: immediate (0s wait)
Attempt 2: 10s wait
Attempt 3: 20s wait
Attempt 4: 40s wait
Attempt 5: 80s wait (or re-raises if last attempt)
```

Note: The 5th attempt's wait is only reached if `max_attempts > 5`. On the last attempt (`attempt == max_attempts - 1`), exceptions are re-raised without waiting.

### Default Configuration

**Temperature:** 0 (deterministic greedy sampling)
**Max tokens:** 1000 (generic response budget)
**Default model:** `llama-3.3-70b-versatile` (Groq)
**Default provider:** `groq`

**Model Options:**
- Groq: `llama-3.3-70b-versatile` (default)
- GitHub: `gpt-4o`
- Ollama: any locally installed model

Model is selected at runtime via `--model` flag; no model list is hardcoded. All support OpenAI-compatible chat API.

---

## 11. Data Models

### Entity Dataclass

```python
@dataclass
class Entity:
    uri: str
    label: str
    types: list[str] = field(default_factory=list)  # rdf:type classes
    abstract: Optional[str] = None                   # DBpedia abstract
    confidence: float = 1.0                          # [0, 1] from Spotlight
    source: str = "dbpedia"                          # "dbpedia", "spotlight", "sparql"
    discovery_relation: Optional[str] = None         # Predicate that discovered this entity

    def __hash__(self):
        return hash(self.uri)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.uri == other.uri
```

Note: No `entity_type` field — type information stored in `types` list.

### Triple Dataclass

```python
@dataclass
class Triple:
    subject: str
    predicate: str
    obj: str                                    # Can be URI or literal
    subject_label: Optional[str] = None
    predicate_label: Optional[str] = None
    obj_label: Optional[str] = None
    confidence: float = 1.0
    source: Optional[str] = None

    def __hash__(self):
        return hash((self.subject, self.predicate, self.obj))

    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.subject == other.subject and
                self.predicate == other.predicate and
                self.obj == other.obj)
```

Includes `to_natural_language()` method for converting triples to readable sentences.

### HybridResult Dataclass

```python
@dataclass
class HybridResult:
    answer: str                                              # Final answer text
    method: str                                              # "sparql", "text", or "logicrag_decomposed"
    route_source: str = ""                                   # "rule:*[+ask_verified|ask_missing]", "llm_prop:*", etc.
    sparql_fallback: bool = False                            # True if routed SPARQL but fell back to text
    sparql_query: Optional[str] = None                       # Executed SPARQL query (provenance)
    kg_context: str = ""                                     # Formatted KG facts injected into text prompt
    sub_question_log: list[dict] = field(default_factory=list)   # Per-sub-Q method/query/answer
    debug_info: dict = field(default_factory=dict)               # Full metadata
```

Note: `sparql_fallback` comes before `sparql_query`. Default values use `""`, `list()`, and `dict()` — not `None`.

### extract_final_answer() Function

```python
def extract_final_answer(text: str) -> str:
    """Extract answer after last 'FINAL ANSWER:' marker (case-insensitive).

    Last-match semantics: if multiple markers, use the last one.
    Returns text AFTER the marker (across newlines), not a single-line capture.
    Fallback: return full text if no marker found.
    """
    if not text:
        return text or ""
    matches = list(re.finditer(r"FINAL ANSWER:", text, re.IGNORECASE))
    if matches:
        return text[matches[-1].end():].strip()
    return text
```

**Key detail:** The regex only matches the marker itself (`r"FINAL ANSWER:"`), then captures **all remaining text** via `text[match.end():]`. This differs from a single-line capture group — it preserves multi-line answers.

**Rationale:** Multi-turn reasoning may include intermediate "FINAL ANSWER" lines; last one is definitive.

---

## 12. Evaluation Framework

### Metrics Module

**File:** `evaluation/metrics.py` (sole source of truth for metrics)

#### normalize_answer() Pipeline

```python
def normalize_answer(answer: str) -> str:
    """Normalize answer for EM/F1 comparison."""
    # 1. Date normalization FIRST (before punctuation removal destroys date hyphens)
    #    Short-circuits: if input is a recognized date, return canonical form immediately
    canonical_date = normalize_date(answer)  # Note: singular, not "normalize_dates"
    if canonical_date is not None:
        return canonical_date  # e.g., "16 August 1932" → "1932-08-16"

    # 2. Lowercase
    answer = answer.lower()

    # 3. Remove punctuation (ALL 32 chars in string.punctuation, not a limited regex)
    answer = ''.join(ch for ch in answer if ch not in string.punctuation)

    # 4. Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)

    # 5. Remove extra whitespace
    answer = ' '.join(answer.split())

    return answer.strip()

def normalize_date(text: str) -> str | None:
    """Convert date strings to ISO format YYYY-MM-DD. Returns None if not a date."""
    # ISO: 1932-08-16 → keep (re-formatted with zero-padding)
    # DMY: 16 August 1932 → 1932-08-16
    # MDY: September 7, 1868 → 1868-09-07
    # Does NOT handle numeric formats like 01/15/2020
```

#### EM (Exact Match)

```python
def compute_em(prediction: str, ground_truth: str) -> float:
    """Binary: normalized prediction == normalized ground truth."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))
```

#### F1 (Token-Level with Multiplicity)

```python
def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 using Counter (preserves token multiplicity)."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    # Counter intersection — NOT set intersection
    # This preserves multiplicity: "new new york" vs "new york" differs from set-based F1
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)
```

#### Containment

```python
def compute_containment(prediction: str, ground_truth: str) -> float:
    """Binary substring check — NOT fractional token recall.
    Returns 1.0 if normalized gold appears anywhere in normalized prediction, else 0.0.
    This is an upper bound metric (intentionally lenient).
    """
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))
```

### Batch Evaluation

**Script:** `scripts/hybrid_batch_eval.py`

**Features:**
- Auto-resume: reads completed question IDs from existing JSONL
- Append-only: new results appended to existing file
- 5-minute timeout per question
- Batch size: 1 (serial execution, prevents timeout masking)

**JSONL Format:**
```jsonl
{"question_id": 1, "question": "...", "answer": "...", "method": "sparql", "em": 0, "f1": 0.5}
{"question_id": 2, "question": "...", "answer": "...", "method": "text", "em": 1, "f1": 1.0}
```

### LLM-as-Judge

**Script:** `scripts/judge_hybrid_eval.py`

**System Message:**
```
You are a strict evaluation judge for a QA benchmark.
Determine if two answers refer to the exact same entity or fact.
Precision matters more than recall — when in doubt, answer no.
```

**Prompt Template** (binary VERDICT format):
```
Determine if the predicted answer is semantically equivalent to the gold answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Matching rules (answer YES):
- Same entity/fact with different surface forms (e.g. "July 16, 2013" vs "2013-07-16")
- Gold contained in a more specific predicted answer
- Added title/honorific/disambiguator
- ... (7 matching rules total)

Rejection rules (answer NO):
- Wrong, empty, or completely different entity
- Predicted is broader/vaguer than gold
- Year only when gold has full date
- ... (7 rejection rules total)

First, explain in ONE sentence why the answers match or don't.
Then write your verdict on a new line: VERDICT: yes or VERDICT: no
```

**Fast-path precheck:** Before calling LLM, `_containment_precheck()` checks if gold words (≥2 words required) are a subset of prediction words. If yes, returns True without LLM call.

**LLM Config:**
- Default provider/model: `ollama` / `gemma3:12b` (configurable via `--provider` / `--model`)
- Temperature: 0 (deterministic judging)
- Max tokens: 200

### Statistical Testing

#### McNemar Test

```python
def mcnemar_test(system_a_correct, system_b_correct, alpha=0.05):
    """
    Paired test for two systems.
    contingency[0,1] = cases where A correct, B wrong
    contingency[1,0] = cases where A wrong, B correct
    """
    from scipy.stats import mcnemar

    contingency = np.array([[agree_both, a_correct_b_wrong],
                            [a_wrong_b_correct, disagree_both]])

    statistic, p_value = mcnemar(contingency)
    return p_value < alpha, p_value
```

#### Wilson Score Interval

```python
def wilson_confidence_interval(successes, n, confidence=0.95):
    """
    Compute 95% CI for proportion (not symmetric like Clopper-Pearson).
    Used for EM/F1 reporting.
    """
    from statsmodels.stats.proportion import proportion_confint

    lower, upper = proportion_confint(successes, n, alpha=1-confidence, method='wilson')
    return lower, upper
```

#### Bootstrap CI (10K iterations)

```python
def bootstrap_ci(data, statistic_fn, n_iterations=10000, seed=42, alpha=0.05):
    """
    Resample with replacement, compute statistic, extract percentile CI.
    """
    np.random.seed(seed)
    bootstrap_stats = []

    for _ in range(n_iterations):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_fn(sample))

    lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)

    return lower, upper
```

#### Bonferroni Correction

```
# 4-way comparison (SPARQL component vs GRASP vs Hybrid vs GraphRAG)
# Note: SPARQL here is Hybrid's internal component run standalone, not an independent system
# Bonferroni: alpha_individual = 0.05 / 4 = 0.0125
# Apply to all 6 pairwise tests
```

---

## 13. Complete Hyperparameter Catalog

### 13.1 Entity Linking (8 Parameters)

| Parameter | Value | Location | Type | Notes |
|-----------|-------|----------|------|-------|
| Spotlight circuit breaker threshold | 3 failures | dbpedia/entity_linker.py:26,29 | Hardcoded | Trips breaker after 3 consecutive failures |
| Spotlight circuit breaker cooldown | 120.0 sec | dbpedia/entity_linker.py:26,29 | Hardcoded | 2-minute cooldown before retrying Spotlight API |
| HTTP client timeout | 30.0 sec | dbpedia/entity_linker.py:187 | Hardcoded | httpx timeout for entity linking HTTP requests |
| min_confidence threshold | 0.5 | hybrid_qa.py:80 | Hardcoded | Minimum confidence for entity linking acceptance |
| Spotlight disamb. max_tokens | 10 | dbpedia/entity_linker.py:650-652 | Hardcoded | Budget for LLM disambiguation calls |
| Spotlight disamb. temperature | 0 | dbpedia/entity_linker.py:650-652 | Hardcoded | Deterministic decoding for Spotlight resolution |
| Spotlight disamb. max_attempts | 2 | dbpedia/entity_linker.py:650-652 | Hardcoded | Max retry attempts for disambiguation |
| Wikipedia search result limit | 3 | dbpedia/wiki_resolver.py:71 | Hardcoded | srlimit=3 in Wikipedia search_candidates() |

### 13.2 SPARQL Execution & DBpedia (9 Parameters)

| Parameter | Value | Location | Type | Notes |
|-----------|-------|----------|------|-------|
| DBpedia endpoint | https://dbpedia.org/sparql | dbpedia/client.py:61 | Hardcoded | Official DBpedia SPARQL endpoint |
| SPARQL timeout | 30 sec | dbpedia/client.py:62,70 | Hardcoded | Timeout per SPARQL query |
| Rate limiter (queries/sec) | 10.0 | dbpedia/client.py:25,63 | Hardcoded | Token bucket rate: 10 queries/second |
| Retry attempts (SPARQL) | 3 | dbpedia/client.py:74 | Hardcoded | tenacity @retry: 3 attempts with exponential backoff |
| Retry max wait | 10 sec | dbpedia/client.py:74 | Hardcoded | Max exponential backoff wait time |
| SPARQL LIMIT clause | 100 | sparql_solver.py:77 | In prompt rule | "Always add LIMIT 100" instruction |
| NULL_ANSWERS set | 9 strings | sparql_solver.py:44-47 | Hardcoded | {"could not", "n/a", "unknown", "none", "not found", "no answer", ...} |
| List-article rejection regex | Pattern | sparql_solver.py:51-57 | Hardcoded | Filters "List of...", "discography", "filmography", "bibliography", "Awards..." |
| Max results in prompts | 20 | sparql_solver.py:145,207 | Hardcoded | Show first 20 rows for answer extraction |

### 13.3 Text Retrieval & Wikipedia (13 Parameters)

| Parameter | Value | Location | Type | Notes |
|-----------|-------|----------|------|-------|
| Wikipedia API rate limit | 0.5 sec | dbpedia/wiki_resolver.py:15 | Hardcoded | Politeness delay between Wikipedia API calls |
| Wikipedia API timeout | 10.0 sec | dbpedia/wiki_resolver.py:74,105 | Hardcoded | httpx timeout for Wikipedia search/redirect |
| MAX_CONTEXT_CHARS | 25,000 | graphrag_wiki.py:44 | Hardcoded | Hard cap on total context size (text pipeline) |
| MAX_CHARS_PER_PAGE | 6,000 | graphrag_wiki.py:45 | Hardcoded | Max chars per Wikipedia page (prevents dominance) |
| SEED budget fraction | 0.60 (60%) | graphrag_wiki.py:663 | Hardcoded | Budget allocated to seed entities |
| BRIDGE budget fraction | 0.30 (30%) | graphrag_wiki.py:664 | Hardcoded | Budget allocated to bridge entities |
| MIN bridge page budget | 1,500 chars | graphrag_wiki.py:667 | Hardcoded | Minimum per bridge page regardless of count |
| Remaining context threshold | 200 chars | graphrag_wiki.py:710 | Hardcoded | Min chars to include partial final entry |
| Max Wikipedia text per page | 50,000 chars | graphrag_wiki.py:522,537,556,573 | Hardcoded | Limits memory during Wikipedia fetching |
| DuckDuckGo circuit breaker threshold | 3 | graphrag_wiki.py:91,116 | Hardcoded | Failures before DuckDuckGo circuit breaks |
| DuckDuckGo circuit breaker cooldown | 60.0 sec | graphrag_wiki.py:91,116 | Hardcoded | 1-minute cooldown for DuckDuckGo |
| Wiki delay (alternate) | 0.5 sec | ablation_study_v3/wiki_utils.py:17 | Hardcoded | Alternative Wikipedia rate limit |
| Spotlight delay (test harness) | 1.0 sec | scripts/test_ensemble_el_live.py:49 | Hardcoded | Rate limit for Spotlight in test ensemble |

### 13.4 RAG (Vector Store, Chunking, Reranking) — 13 Parameters

| Parameter | Value | Location | Type | Notes |
|-----------|-------|----------|------|-------|
| Default chunk size | 512 chars | rag/chunker.py:27 | Configurable | Target (soft) chunk size in characters |
| Chunk overlap | 64 chars | rag/chunker.py:28 | Configurable | Character overlap for context continuity |
| Min effective chunk | 100 chars | rag/chunker.py:121-122 | Hardcoded | Minimum after header subtraction |
| Bi-encoder model | all-MiniLM-L6-v2 | rag/vector_store.py:15,38 | Configurable default | Sentence-transformer 384-dim model |
| Fetch_k multiplier | 2x top_k | rag/vector_store.py:145 | Hardcoded | Over-fetch before seed boost reranking |
| Seed boost multiplier | 1.2 | rag/vector_store.py:124 | Configurable | Score multiplier for seed pages |
| Default top_k retrieval | 30 | rag/vector_store.py:122 | Configurable | Default chunks before reranking |
| Cross-encoder model | cross-encoder/ms-marco-MiniLM-L-6-v2 | rag/reranker.py:13 | Configurable default | MS MARCO MiniLM-L-6 reranker |
| Reranker top_k output | 15 | rag/reranker.py:44 | Configurable | Keep top-15 after reranking |
| RAG retrieval top_k | 20 | text_solver.py:259 | Hardcoded | Bi-encoder chunks for reranker |
| RAG reranker top_k | 8 | text_solver.py:272 | Hardcoded | Final chunks after cross-encoder |
| RAG assembly max_chars | 12,000 | text_solver.py:278 | Hardcoded | Tighter budget for RAG context |
| Text assembly max_per_entity | 4,000 | text_solver.py:89 | Hardcoded | Per-entity char limit in assembly |

### 13.5 Routing & Stage Thresholds — 20+ Parameters

**KEYWORD_TO_PROPERTIES mapping (60+ entries)** → See Section 5 (Question Analysis)

**COMPARISON_WORDS set** (13 words) → See Section 5

**_BRIDGE_KEYWORDS set** (27 words) → See Section 5

**_HOP2_KEYWORDS set** (16 words) → See Section 5

**_DATE_PROPERTIES** (13) → See Section 5

**_COUNTRY_PROPERTIES** (10) → See Section 5

**_PARENT_PROPS** (5) → See Section 5

### 13.6 LogicRAG Decomposition — 2 Parameters

| Parameter | Value | Location | Type | Notes |
|-----------|-------|----------|------|-------|
| Topological sort algorithm | Kahn's algorithm | logicrag_decomposition.py:106-136 | Hardcoded | DAG-based dependency resolution with cycle fallback |
| Cycle fallback strategy | Sequential 0..n-1 | logicrag_decomposition.py:135-136 | Hardcoded | Avoids deadlock; processes linearly on cycle detection |

### 13.7 LLM Configuration — 20 Parameters

| Parameter | Value | Location | Type | Notes |
|-----------|-------|----------|------|-------|
| Default max_attempts | 5 | llm_provider.py:70 | Configurable | Retry attempts for transient errors |
| Initial wait time | 10.0 sec | llm_provider.py:70 | Configurable | First retry wait (exponential backoff) |
| Default max_tokens | 1000 | llm_provider.py:71 | Configurable | Generic response budget |
| Default temperature | 0 | llm_provider.py:71 | Configurable | Deterministic/greedy decoding |
| Retryable error codes | 7 patterns | llm_provider.py:36 | Hardcoded | {429, 500, 502, 503, 504, timeout, connection} |
| Default model | llama-3.3-70b-versatile | hybrid_qa.py:60 | CLI configurable | Primary model across eval scripts |
| Default provider | groq | hybrid_qa.py:64 | CLI configurable | Groq API endpoint |
| GraphRAG max_tokens (final answer) | 500 | graphrag_wiki.py:798 | Hardcoded | Answer reasoning response budget |
| Judge LLM temperature | 0 | scripts/judge_hybrid_eval.py:126 | Hardcoded | Deterministic judging |
| Judge LLM max_tokens | 100 | scripts/judge_hybrid_eval.py:127 | Hardcoded | Judgment response budget |
| Ablation LLM retry attempts | 5 | ablation_study_v3/wiki_utils.py:42 | Hardcoded | LLM call retries in ablation |
| Ablation LLM initial_wait | 10.0 | ablation_study_v3/wiki_utils.py:42 | Hardcoded | Initial retry wait in ablation |
| Ablation LLM max_tokens | 100 | ablation_study_v3/wiki_utils.py:43 | Hardcoded | Response budget in ablation |
| Ablation LLM temperature | 0 | ablation_study_v3/wiki_utils.py:43 | Hardcoded | Deterministic in ablation |

### 13.8 Evaluation & Statistical Testing — 15+ Parameters

| Parameter | Value | Location | Type | Notes |
|-----------|-------|----------|------|-------|
| Evaluation QUESTION_TIMEOUT | 300 sec | scripts/hybrid_batch_eval.py:38 | Hardcoded | 5-minute timeout per question |
| GRASP WS_TIMEOUT | 180 sec | grasp_evaluation.py:39 | Hardcoded | 3-minute per WebSocket message |
| GRASP QUESTION_TIMEOUT | 600 sec | grasp_evaluation.py:40 | Hardcoded | 10-minute total per question |
| GRASP MAX_RETRIES | 3 | grasp_evaluation.py:41 | Hardcoded | Retry transient failures |
| GRASP MIN_REQUEST_DELAY | 2 sec | grasp_evaluation.py:42 | Hardcoded | Politeness delay between requests |
| Bonferroni alpha | 0.0125 | ablation_study_v3/reporting.py:17 | Hardcoded | 0.05 / 4 tests (4-way comparison) |
| Bootstrap iterations | 10,000 | scripts/unified_comparison.py:106 | Hardcoded | Monte Carlo resampling precision |
| Bootstrap seed | 42 | scripts/unified_comparison.py:106 | Hardcoded | RNG seed for reproducibility |
| Bootstrap CI alpha | 0.05 | scripts/unified_comparison.py:106 | Hardcoded | Two-sided significance level (95% CI) |
| Standard alpha (single test) | 0.05 | scripts/unified_comparison.py:402 | Hardcoded | Used when no Bonferroni correction |
| Per-scenario sample | 10 | scripts/scenario_eval.py:271 | CLI configurable | Questions per scenario type |
| Sampling random seed | 42 | scripts/scenario_eval.py:273 | CLI configurable | RNG seed for scenario selection |
| Request delay (scenario eval) | 2.0 sec | scripts/scenario_eval.py:274 | CLI configurable | Rate-limiting between API calls |
| Proximity window (benchmark check) | 200 chars | scripts/detect_benchmark_errors.py:32 | Hardcoded | Distance threshold for keyword proximity to entity |

### 13.9 Caching Systems — 11 Parameters

| Parameter | Value | Location | Type | Notes |
|-----------|-------|----------|------|-------|
| Memory cache max_size | 10,000 entries | dbpedia/cache.py:12 | Configurable | FIFO eviction when full |
| Memory cache TTL | 3600 sec (1 hour) | dbpedia/cache.py:12 | Configurable | Default expiry time |
| Redis host | localhost | dbpedia/cache.py:63 | Configurable | Redis server hostname |
| Redis port | 6379 | dbpedia/cache.py:64 | Configurable | Standard Redis port |
| Redis DB | 0 | dbpedia/cache.py:65 | Configurable | Database number |
| Redis TTL | 3600 sec | dbpedia/cache.py:70 | Configurable | Cache expiry in Redis |
| Redis key prefix | "dbpedia_qa:" | dbpedia/cache.py:68 | Configurable | Namespace for keys |
| Wiki cache path | data/wiki_cache.db | rag/cache.py:12 | Configurable | SQLite database location |
| Wiki cache eviction | 7 days | rag/cache.py:13 | Hardcoded | Stale entry cleanup |
| Embedding vector dim | 384 | rag/cache.py:83 | Hardcoded | Matches all-MiniLM-L6-v2 |
| Default DB path | data/wiki_cache.db | rag/cache.py:12 | Hardcoded default | Auto-created in data/ |

---

## 14. External Service Dependencies

### DBpedia Spotlight API

**URL:** `https://api.dbpedia-spotlight.org/en/annotate`
**Method:** POST (multipart form-data)
**Parameters:**
- `text`: Question string
- `confidence`: 0.5
- `support`: 5

**Response:** XML with `<Resource>` tags containing uri, surfaceForm, offset, similarity (confidence)

**Reliability:** Publicly available, occasionally slow (rate-limited to ~5 req/sec per IP)

**Fallback:** spaCy NER if Spotlight times out

### DBpedia SPARQL Endpoint

**URL:** `https://dbpedia.org/sparql`
**Method:** POST (application/x-www-form-urlencoded)
**Parameters:**
- `query`: SPARQL query
- `format`: application/json

**Response:** JSON with variable bindings (for SELECT) or boolean (for ASK)

**Timeout:** 30s per query
**Rate Limit:** ~10 queries/sec (enforced server-side; exceeded → HTTP 429)

**IPv4 Forcing:** Monkey-patch in dbpedia/client.py forces IPv4 (IPv6 unreachable on some networks)

### Wikipedia API

**URL:** `https://en.wikipedia.org/w/api.php`
**Methods:** `query`, `search`, `redirect`

**Parameters:**
- `action=query&titles=...`: Fetch article by title
- `action=query&list=search&srsearch=...`: Search API
- `redirects`: Follow redirects

**Response:** JSON with pages (article content, sections, links)

**Rate Limit:** 0.5s politeness delay between calls
**Timeout:** 10s per request

### DuckDuckGo Search

**URL:** `https://duckduckgo.com/` (via ddgs Python library)
**Method:** Implicit (uses requests + parsing)

**Reliability:** Very unreliable; blocks after ~50 calls in short time
**Circuit Breaker:** 3 failures, 60s cooldown (built into graphrag_wiki.py)

### GRASP WebSocket API

**URL:** `wss://grasp.cs.uni-freiburg.de/api/live`
**Protocol:** WebSocket with message acknowledgment

**Message Format (request):**
```json
{
  "question": "Who directed Inception?"
}
```

**Message Format (response):**
```json
{
  "sparql": "SELECT ...",
  "results": [...],
  "final_answer": "..."
}
```

**Acknowledgment:** Client must send `{"received": true}` after each message

**Timeout:** 3 min per message, 10 min per question
**Rate Limit:** 2s minimum delay between requests

**Reliability:** Academic service, occasionally goes offline
**Model:** Uses OpenAI GPT-5-mini (via API key)

---

## 15. Caching Architecture

### In-Memory Cache

**Location:** `dbpedia/cache.py`

**Implementation:** FIFO dict with TTL tracking (evicts oldest key when full, not least recently used)

**Capacity:** 10,000 entries
**TTL:** 1 hour (3600 sec)

**Use Cases:**
- SPARQL query results
- Entity type lookups
- Wikipedia article fetches (temporary)

### Redis Cache (Optional)

**Location:** `dbpedia/cache.py` (Redis backend)

**Configuration:**
- Host: localhost
- Port: 6379
- DB: 0
- Key prefix: "dbpedia_qa:"
- TTL: 1 hour

**Use Cases:** Distributed caching (if multiple processes/workers)

### SQLite Wiki Cache (wiki_cache.db)

**Location:** `data/wiki_cache.db`

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS wiki_pages (
    title TEXT PRIMARY KEY,
    chunks_json TEXT,      -- JSON-serialized list of WikiChunk dicts
    embeddings BLOB,       -- numpy array of all chunk embeddings
    fetched_at REAL        -- Unix timestamp
);
```

Single table stores chunks + embeddings together per page title.

**Eviction Policy:**
- **Time-based:** 7-day TTL (`_EVICTION_DAYS = 7`, `rag/cache.py:13`)
- Eviction runs on each `WikiRAGCache` instantiation: `DELETE FROM wiki_pages WHERE fetched_at < cutoff`

**Initialization:** Auto-created on first run; no manual setup required

---

## 16. Design Decisions & Tradeoffs

### 1. Why Entity-Only KG (No Wikidata)

**Decision:** Use DBpedia knowledge graph entities but acknowledge property coverage gap.

**Rationale:**
- **Benchmark constraint:** 2WikiMultiHopQA sourced from Wikidata; using Wikidata for answers = data leakage
- **Entity-heavy assumption:** Entities are well-covered (93.6% on 1000q); facts encoded in Wikipedia text
- **Thesis claim:** Text + entity linking can bridge property gaps

**Tradeoff:**
- ✓ Avoids data leakage
- ✗ Limited property coverage (36.8% hop-verified)
- Compensated by LogicRAG decomposition + text fallback

### 2. Why Cached Schemas vs Dynamic Discovery

**Decision:** Pre-compute CACHED_CLASS_SCHEMAS for 7 common classes; fallback to SPARQL for unknowns.

**Rationale:**
- **Speed:** Cached schemas available instantly, no network latency
- **Reliability:** Pre-computed at system startup, fixed metadata
- **Coverage:** 7 classes cover ~80% of benchmark questions (Person, Film, Company, Place, etc.)
- **Scalability:** Dynamic SPARQL per entity would exceed rate limits

**Tradeoff:**
- ✓ Low latency, high throughput
- ✗ Stale schemas (updated manually, not live)
- Mitigated by extensive testing and offline SPARQL verification

### 3. Why Three-Stage Routing

**Decision:** Deterministic rules → SPARQL ASK verification → LLM fallback

**Rationale:**
- **Stage 1 (Rules):** Fast, explainable, covers 60% of questions
- **Stage 2 (ASK):** Verify rules before expensive SPARQL execution
- **Stage 3 (LLM property extraction):** Extract needed properties, re-run rules, default to TEXT

**Tradeoff:**
- ✓ Adaptive: doesn't over-commit to SPARQL for complex questions
- ✗ Adds latency (3 queries per question in worst case)
- Justified by avoiding false-positive SPARQL routing

### 4. Why LogicRAG Decomposition Over Monolithic Prompting

**Decision:** Generate sub-questions with dependency graph, route each, synthesize.

**Rationale:**
- **Compositionality:** Break multi-hop into single-hop sub-problems
- **Route flexibility:** Each sub-Q routed independently (SPARQL or TEXT)
- **Grounding:** Intermediate answers provide explicit reasoning trace
- **Interpretability:** Debug full sub-Q log in debug_info

**Tradeoff:**
- ✓ Better accuracy for complex multi-hop
- ✗ Higher latency (multiple LLM calls + SPARQL/text per sub-Q)
- ✗ Decomposition errors can propagate (if sub-Q answer wrong)

### 5. Why Bi-Encoder + Cross-Encoder Two-Stage Retrieval

**Decision:** FAISS bi-encoder top-30 → ms-marco cross-encoder top-8

**Rationale:**
- **Bi-encoder (fast):** Scales to large corpus (1K chunks), dot-product is O(1) per query
- **Cross-encoder (accurate):** Fine-grained ranking, captures semantic nuance
- **Two-stage:** First stage filters, second stage ranks accurately

**Tradeoff:**
- ✓ Balanced speed/accuracy (bi-encoder filters noise, cross-encoder refines)
- ✗ Two model loads (384-dim + BERT), 2-3x more latency than single-stage
- Justified by 5-10% F1 improvement over bi-encoder alone

### 6. Why FAISS IndexFlatIP (Brute-Force Exact Search)

**Decision:** No quantization, no approximation, no HNSW.

**Rationale:**
- **Corpus size:** ~1K chunks per question, manageable brute-force
- **Accuracy:** Exact dot-product search (no information loss)
- **Simplicity:** No index tuning, reproducible results
- **Reliability:** No approximate neighbor misses

**Tradeoff:**
- ✓ 100% recall, deterministic results
- ✗ O(K×D) per query (K=chunk count, D=384-dim)
- Acceptable for small corpus; would not scale to 1M+ chunks

### 7. Why Seed Boost (1.2x) Instead of Page-Level Filtering

**Decision:** Upweight seed entity pages, don't filter non-seed pages.

**Rationale:**
- **Inclusion:** Bridge entities often mentioned in seed articles (co-occurrence)
- **Flexibility:** Ranking can still include non-seed content if more relevant
- **Precision:** Soft boost (1.2x) gentler than hard filtering
- **Recall:** Don't lose information by hard-filtering

**Tradeoff:**
- ✓ Higher recall (includes bridge entities even if not in seed articles)
- ✗ Slightly noisier results if seed boost too low
- Tuned empirically (1.2x chosen for this corpus)

### 8. Why Last-Match FINAL ANSWER Semantics

**Decision:** If multiple "FINAL ANSWER:" markers in LLM response, use last one.

**Rationale:**
- **Multi-turn reasoning:** LLM may output intermediate reasoning steps with "FINAL ANSWER"
- **Correctness:** Last occurrence is final refinement
- **Robustness:** Handles malformed LLM outputs gracefully

**Tradeoff:**
- ✓ Handles varied LLM output formats
- ✗ Fragile to LLM outputting multiple plausible answers
- Mitigated by temperature=0 (deterministic, single answer typically)

---

## Appendix: Critical Missing Documentation

The following parameters lack empirical justification or sensitivity analysis:

1. **Seed budget split (60/30)** — No ablation study on allocation ratio
2. **MAX_CONTEXT_CHARS (25K)** — No token-count to length study
3. **RAG top_k values (20→8)** — No precision/recall tradeoff analysis
4. **Chunk size (512)** — No semantic coherence vs granularity study
5. **Confidence threshold (0.5)** — No precision/recall curve documented
6. **Rate limits** — Set conservatively without load testing data
7. **Timeout values** — Appear arbitrary; no distribution analysis
8. **Circuit breaker cooldowns** — No failure recovery metrics

---

## Appendix: Recommendations for Future Work

### High-Impact for Thesis:
- Systematically tune seed budget allocation (test 40/40, 50/40, 70/20, etc.)
- Vary bi-encoder top_k (15, 20, 30) and reranker top_k (5, 8, 12) on benchmark
- Measure impact of MAX_CONTEXT_CHARS reduction (20K, 22.5K, 25K, 27.5K)
- Document empirical justification for confidence thresholds

### For Production:
- Document all CLI-configurable parameters with range constraints
- Add config file support (YAML/JSON) instead of scattered CLI defaults
- Implement hyperparameter validation at startup
- Add telemetry for cache hit rates, timeout frequency, circuit breaker trips

---

**End of Technical Documentation**

Total lines: ~2,430 (including tables, code blocks, ASCII diagrams)

This document is thesis-ready for citing specific hyperparameter choices and documenting experimental variations.
