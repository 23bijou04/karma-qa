# Schema-Routed Hybrid QA: System Architecture Report

## 1. Overview

This document describes the architecture of our Schema-Routed Hybrid QA system for multi-hop question answering over DBpedia. The system addresses a fundamental limitation: only 31.1% of 2WikiMultiHopQA questions are fully answerable via SPARQL against DBpedia (entity coverage is 92.5%, but property coverage is only 60.1%). Rather than attempting SPARQL for all questions, the system uses the KG schema itself to route each question to the optimal answering strategy.

**Entry point:** `hybrid_qa.py`
**Implementation:** ~500 lines of Python

## 2. Algorithm

```
Algorithm: Schema-Routed Hybrid QA

Input:  question q, LLM model M
Output: answer a, method m

1. ENTITY LINKING
   E <- Spotlight(q, confidence >= 0.5) + spaCy_NER_fallback(q)

2. SCHEMA DISCOVERY
   S <- {class, properties, sample_values} for each e in E via SPARQL

3. ROUTE DECISION
   route <- LLM(S, q) -> "SPARQL" | "TEXT"

4. IF route = "SPARQL":
     sparql <- LLM_generate(S, q)
     results <- execute(sparql)
     if results = empty:
       sparql' <- LLM_retry(sparql, error, S, q)
       results <- execute(sparql')
     if results != empty:
       return LLM_extract_answer(q, results), "sparql"

5. TEXT FALLBACK (auto-chain):
   5a. context_entities <- LLM_generate_context_SPARQL(S, q)
       context_entities <- filter_by_relevance(context_entities, q, E)
   5b. if no new entities found:
         context_entities <- explore_kg(E, schema_filter=True)

6. WIKIPEDIA RETRIEVAL
   pages <- fetch_wikipedia(context_entities) + web_search(keywords(q))

7. SECTION EXTRACTION
   per_page_budget <- max(3000, 12000 / |pages|)
   for each page: extract top sentences by keyword density (budget=per_page_budget)
   context <- rank pages by keyword density, fill up to 12000 chars

8. ANSWER
   a <- LLM(q, context) -> extract "FINAL ANSWER: ..."
   return a, method
```

## 3. Pipeline Components

### 3.1 Entity Linking (Step 1)

**Primary:** DBpedia Spotlight API (`https://api.dbpedia-spotlight.org/en/annotate`) with `min_confidence=0.5`. The confidence threshold filters noise (e.g., "Albert" mapping to "Uncle Albert" at confidence 0.40).

**Fallback:** spaCy NER (`en_core_web_sm`) extracts named entities when Spotlight returns nothing, followed by SPARQL `rdfs:label` search to resolve URIs.

**Retry:** Tenacity-based retry (3 attempts, exponential backoff) for Spotlight API resilience.

### 3.2 Schema Discovery (Step 2)

For each linked entity, the system queries DBpedia for:
- **Class membership** (e.g., `dbo:Person`, `dbo:Film`)
- **Available properties** (both `dbo:` and `dbp:` namespaces)
- **Sample values** for key properties (e.g., `dbo:birthDate -> "1879-03-14"`)

Six common classes (`Person`, `Film`, `Organisation`, `Place`, `Work`, `Band`) have cached schemas for performance. Dynamic SPARQL discovery handles all other classes.

The schema is formatted as structured text passed to the LLM, including exact URIs, property names, and value hints. This format is critical --- the LLM uses the exact URIs to generate valid SPARQL.

### 3.3 Schema-Based Route Decision (Step 3)

**Key insight:** The LLM examines the discovered schema and decides whether the available properties can answer the question *before* generating any SPARQL query.

**Prompt strategy:** The `ROUTE_PROMPT` presents the schema and question, asking the LLM to reply with exactly one word: `SPARQL` or `TEXT`.

**Examples:**
- "When was Einstein born?" + schema shows `dbo:birthDate` -> route `SPARQL`
- "Who destroys the last Horcrux?" + schema shows only `dbo:description` -> route `TEXT`

**Benefits:**
1. Saves 2-3 LLM calls when SPARQL cannot answer (no wasted generation + retry)
2. Avoids wrong answers from SPARQL queries that match incorrect properties but return some results
3. Reduces end-to-end latency for text-routed questions

### 3.4 SPARQL Generation + Execution (Step 4)

When routed to SPARQL:
1. **Generate:** LLM produces a SPARQL query using the schema context (exact URIs, property hints)
2. **Execute:** Query runs against DBpedia's public SPARQL endpoint
3. **Retry:** If the query fails (syntax error, empty results), the error message is fed back to the LLM for self-correction
4. **Extract:** If results are obtained, LLM extracts a concise answer from the structured results

### 3.5 Text Fallback with Auto-Chaining (Step 5)

When routed to TEXT (or when SPARQL fails), the system identifies which Wikipedia pages to retrieve. Two methods are available, with automatic chaining:

**Method A --- LLM-Generated Context SPARQL:**
The LLM generates a SPARQL query to find DBpedia entities whose Wikipedia pages likely contain the answer. Results are filtered by label relevance (entity labels must share words with the question or seed entities) to remove noise. This prevents irrelevant entities like "Dragon Challenge" or "Lenny Henry" from diluting the context.

**Method B --- Programmatic KG Exploration:**
Schema-guided 1-hop neighbor traversal. Filters to `dbo:`/`dbp:` predicates only (skipping `wikiPageWikiLink`). Uses the entity's class schema to identify meaningful relation types.

**Auto-chain logic:** Try Method A first. If it finds no new entities beyond the seeds, automatically fall back to Method B. This is controlled by the `--fallback` flag: `auto` (default, chains both), `sparql_context` (A only), `explore` (B only).

### 3.6 Wikipedia Retrieval (Step 6)

For each entity, the system:
1. Derives the Wikipedia title from the DBpedia URI (exact match)
2. Fetches full article text via the Wikipedia API (up to 50,000 chars)
3. Falls back to DuckDuckGo web search if the Wikipedia title doesn't resolve
4. Supplements with question-keyword web search for additional relevant pages

### 3.7 Section Extraction (Step 7)

Wikipedia articles can be 50,000+ characters. The system extracts the most relevant content within a budget:

**Sentence-level extraction for long paragraphs:** Wikipedia plot summaries often appear as a single giant paragraph (e.g., 4198 chars for Deathly Hallows). Rather than truncating at the budget limit (which would lose content at the end), the system splits into sentences and keeps the highest-scoring sentences by keyword density.

**Keyword scoring:**
- Keywords are extracted from the question (stopwords removed, punctuation stripped)
- Each sentence is scored: rare keywords (appearing in <10% of paragraphs) count 2x
- Sentences are re-ordered by original position to maintain narrative flow

**Context assembly:** Pages are ranked by total keyword density. The most relevant pages fill the 12,000-character context budget first, ensuring noise pages are dropped rather than displacing relevant content.

### 3.8 LLM Reasoning (Step 8)

The final LLM call receives the assembled Wikipedia context and the question. The prompt instructs:
- Use ONLY facts from the provided text
- Show brief step-by-step reasoning
- End with `FINAL ANSWER: <answer>`

The answer is extracted by splitting on the `FINAL ANSWER:` marker.

## 4. Design Decisions and Rationale

### Why Schema-Based Routing?

Without routing, the system always attempts SPARQL generation, which:
- Wastes 2-3 LLM calls on questions where the schema clearly lacks needed properties
- Risks wrong answers when SPARQL matches incorrect properties but returns plausible-looking results
- Adds unnecessary latency

The schema provides a reliable signal: if `dbo:birthDate` exists for a Person entity, a birth-date question can be answered via SPARQL. If only `dbo:description` exists for a fictional object, SPARQL cannot help.

### Why Entity Relevance Filtering?

LLM-generated context SPARQL queries tend to follow generic `dbo:` relations, returning many irrelevant entities (e.g., querying Harry Potter's relations returns Dragon Challenge, Lenny Henry, Iranian protests). The relevance filter checks if each entity's label shares words with the question or seed entities, keeping only topically relevant entities.

### Why Sentence-Level Section Extraction?

Wikipedia articles often have plot summaries in a single paragraph. The standard approach (truncate at budget) systematically loses content at the end of long paragraphs. For "Who destroys the last Horcrux?", the answer (Neville Longbottom at char 3106) was consistently lost by a 3000-char budget. Sentence-level extraction keeps the most keyword-relevant sentences regardless of their position in the paragraph.

### Why Keyword-Ranked Context Assembly?

When many pages are retrieved (e.g., 14 from sparql_context), a naive concatenation fills the 12,000-char budget with the first pages encountered, regardless of relevance. Keyword-ranking ensures the most relevant pages (e.g., "Harry Potter and the Deathly Hallows") are included before noise pages (e.g., "List of motorcycle manufacturers").

## 5. Experimental Observations

### Routing Accuracy

| Question | Schema Signal | Route | Method | Answer |
|----------|--------------|-------|--------|--------|
| "When was Einstein born?" | `dbo:birthDate` present | SPARQL | Direct SPARQL | March 14, 1879 |
| "Who directed Inception?" | `dbo:director` present | SPARQL | Direct SPARQL | Christopher Nolan |
| "Who destroys the last Horcrux?" | Only `dbo:description` | TEXT | Auto fallback | Neville Longbottom |

### Text Fallback Auto-Chaining

The auto-chain (sparql_context -> explore) handles cases where the LLM-generated context SPARQL returns no useful entities. When sparql_context finds relevant new entities (e.g., "Harry Potter and the Deathly Hallows"), it is used directly. When it finds only noise or nothing, the system falls back to programmatic KG exploration.

### Section Extraction Impact

Before the sentence-level fix, the Horcrux question consistently returned wrong answers ("Harry Potter" or "Voldemort") because the Neville/Nagini content at char 3106 of a 4198-char paragraph was always truncated by the 3000-char budget. After the fix, the relevant sentences are preserved regardless of position.

## 6. System Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `llama-3.3-70b-versatile` | LLM model (Groq API) |
| `--fallback` | `auto` | Text fallback: `auto`, `explore`, `sparql_context` |
| `--verbose` | off | Show all intermediate pipeline steps |
| `min_confidence` | 0.5 | DBpedia Spotlight entity linking threshold |
| `MAX_CONTEXT_CHARS` | 12,000 | Total context budget for LLM reasoning |
| `MAX_CHARS_PER_PAGE` | 3,000 | Per-page section extraction budget |

## 7. File Structure

```
hybrid_qa.py              -- Main pipeline (entry point)
graphrag_wiki.py           -- Entity linking, KG exploration, Wikipedia retrieval,
                              section extraction
sparql_evaluation.py       -- SPARQL generation/execution, call_llm, extract_sparql
dbpedia/
  entity_linker.py         -- DBpedia Spotlight + spaCy NER
  client.py                -- SPARQL client (get_neighbors, get_triples)
  schema_discovery.py      -- Class schemas, property discovery
  cache.py                 -- Caching layer for SPARQL results
models.py                  -- Entity, Triple dataclasses
evaluation/metrics.py      -- EM/F1 metrics (2WikiMultiHopQA official)
ablation_study_v3/         -- Ablation study (L0-L4 levels)
```
