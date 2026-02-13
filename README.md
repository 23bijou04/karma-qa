# DBpedia Hybrid QA for 2WikiMultiHopQA

A hybrid question-answering system that combines DBpedia knowledge graph with Wikipedia text retrieval and LLM reasoning to answer multi-hop questions from the [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) benchmark (12,576 dev questions, 4 types).

**Thesis:** Entity-only KG + Wikipedia text can answer multi-hop QA without full relation coverage.

## Architecture

The main system is the **Schema-Routed Hybrid Pipeline** (`hybrid_qa.py`). Two additional evaluation baselines are provided for comparison:

| Pipeline | Entry Point | Strategy |
|----------|-------------|----------|
| **Hybrid (main)** | `hybrid_qa.py` | Schema-routed: SPARQL when schema supports it, text fallback otherwise |
| GraphRAG | `graphrag_wiki.py` | Text-only: KG exploration + Wikipedia + LLM |
| GRASP | `grasp_evaluation.py` | External KG QA system (WebSocket API) |
| LLM-SPARQL | `sparql_evaluation.py` | SPARQL-only: schema discovery + LLM SPARQL generation |

### GraphRAG Pipeline (graphrag_wiki.py)

```
Question
   |
1. Entity Linking (DBpedia Spotlight, min_confidence=0.5)
   |
2. KG Exploration (1-hop neighbors, dbo:/dbp: predicates only)
   |
3. Wikipedia Retrieval (URI-based + DuckDuckGo fallback)
   |
4. Section Extraction (keyword-based relevance)
   |
5. LLM Reasoning (Groq llama-3.3-70b-versatile)
   |
Answer
```

### Schema-Routed Hybrid Pipeline (hybrid_qa.py)

```
Question
   |
1. Entity Linking (Spotlight + spaCy fallback)
   |
2. Schema Discovery (classes, properties, sample values via SPARQL)
   |
3. Route Decision — LLM examines schema → SPARQL or TEXT?
   |
   ├─ SPARQL route:
   │    4a. LLM generates SPARQL query using schema URIs
   │    4b. Execute with retry (feed error back to LLM)
   │    4c. Extract answer from structured results
   │         (fall through to TEXT on failure)
   │
   └─ TEXT route (or SPARQL failure):
        5. Context building (auto-chain):
           a) sparql_context — LLM generates context SPARQL to find entities
              + entity relevance filter (label words ∩ question words)
           b) explore — schema-guided 1-hop KG traversal (if a finds nothing)
        6. Wikipedia retrieval for context entities
        7. Section extraction (sentence-level, keyword-scored)
        8. Context assembly (pages ranked by keyword density, 12K budget)
        9. LLM reads passages → FINAL ANSWER
   |
Answer
```

## Question Types

| Type | Strategy | Hops | Example |
|------|----------|------|---------|
| **Comparison** | Get same property for 2 entities, compare | 0-1 | "Who was born first, Einstein or Newton?" |
| **Inference** | Follow logical chain (e.g., grandfather = father->father) | 2 | "Who is the paternal grandfather of X?" |
| **Compositional** | Find bridge entity, then target property | 2 | "Who founded the company that distributed La La Land?" |
| **Bridge-Comparison** | Find bridges for 2 entities, then compare | 2-3 | "Which film has the director born first?" |

## Evaluation Results

Evaluated on 276 answerable questions from 2WikiMultiHopQA dev set:

| System | EM | F1 | Judge EM | Best At |
|--------|-----|------|----------|---------|
| **LLM-SPARQL** | **67.8%** | **76.9%** | **82.6%** | comparison (89.8%), bridge_comparison (82.9%) |
| GRASP | 33.7% | 51.9% | 75.7% | bridge_comparison (65.8%) |
| GraphRAG | -- | -- | -- | text-heavy questions |

### Per Question Type (LLM-SPARQL)

| Type | n | EM | F1 | Judge EM |
|------|---|-----|------|----------|
| Comparison | 49 | 89.8% | 90.5% | 93.9% |
| Bridge-Comparison | 111 | 82.9% | 82.9% | 82.9% |
| Inference | 63 | 63.5% | 80.6% | 84.1% |
| Compositional | 53 | 20.8% | 47.2% | 69.8% |

## DBpedia Answerability

Based on analysis of the full 2WikiMultiHopQA dev set (12,576 questions):

| Metric | Value |
|--------|-------|
| Entity Coverage | 92.5% |
| Property Coverage (per pair) | 60.1% |
| **Full Answerability (SPARQL only)** | **31.1%** |

Property coverage is the bottleneck. The Wikipedia text fallback (GraphRAG) covers the remaining ~69% of questions. See [docs/wikimultihop-dbpedia-answerability.md](docs/wikimultihop-dbpedia-answerability.md) for the full analysis.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package in development mode
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm

# Additional dependencies
pip install groq ddgs
```

**Requirements:** Python 3.10+

## Quick Start

### GraphRAG Pipeline

Answers any question using DBpedia entity linking + Wikipedia full text + Groq LLM:

```bash
export GROQ_API_KEY="your-key-here"

# Answer a question (verbose shows pipeline steps)
python graphrag_wiki.py "Who directed Inception?" --verbose

# Multi-hop question
python graphrag_wiki.py "Who is the paternal grandfather of Eadred?" --verbose
```

### Hybrid Pipeline (Schema-Routed)

Uses KG schema to decide SPARQL vs text retrieval per question:

```bash
python hybrid_qa.py "Who directed Inception?" --verbose

# With explicit fallback mode
python hybrid_qa.py "Who destroys the last Horcrux?" --fallback sparql_context --verbose
```

### Debug Mode

Shows all intermediate pipeline results (entities, KG triples, Wikipedia text):

```bash
python graphrag_debug.py "Who directed Inception?" --api-key $GROQ_API_KEY
```

### Evaluation Pipelines

```bash
# GRASP evaluation (WebSocket-based, ~1.3 min/question)
python grasp_evaluation.py --limit 5

# LLM-SPARQL evaluation
python sparql_evaluation.py --limit 5

# Offline re-evaluation of GRASP results
python grasp_evaluate_results.py data/grasp_eval_dbpedia.jsonl

# Failure analysis (questions where both systems fail)
python graphrag_eval_failures.py
```

### Ablation Study

```bash
python -m ablation_study_v3.runner --api-key $GROQ_API_KEY --n-per-type 5 --levels 0 1 4
```

## Configuration

```bash
# DBpedia endpoints
export DBPEDIA_SPARQL_ENDPOINT="https://dbpedia.org/sparql"
export DBPEDIA_SPOTLIGHT_ENDPOINT="https://api.dbpedia-spotlight.org/en/annotate"

# LLM (Groq)
export GROQ_API_KEY="your-key-here"
```

## Project Structure

```
dbpedia_qa/
+-- __init__.py              # Package exports (Entity, Triple)
+-- models.py                # Data models (Entity, Triple)
+-- graphrag_wiki.py         # GraphRAG pipeline: Spotlight -> KG -> Wikipedia -> LLM
+-- graphrag_debug.py        # Debug script showing all intermediate results
+-- hybrid_qa.py             # Main pipeline: schema-routed SPARQL + text fallback
+-- sparql_evaluation.py     # LLM-SPARQL evaluation pipeline
+-- grasp_evaluation.py      # GRASP WebSocket evaluation client
+-- grasp_evaluate_results.py# Offline re-evaluation of GRASP results
+-- graphrag_eval_failures.py# Failure analysis across systems
+-- dbpedia/
|   +-- client.py            # SPARQL client (queries, neighbors, type-aware)
|   +-- entity_linker.py     # DBpedia Spotlight integration + fallback
|   +-- schema_discovery.py  # Hybrid schema discovery (cached + dynamic)
|   +-- cache.py             # In-memory cache manager
+-- evaluation/
|   +-- metrics.py           # Official EM/F1 computation
+-- ablation_study_v3/
|   +-- runner.py            # Ablation study runner
|   +-- sampling.py          # Stratified sampling (50/type, seed=42)
|   +-- levels_base.py       # Levels 0, 1, 4
|   +-- levels_kg.py         # Levels 2, 3 (KG-based)
|   +-- llm_judge.py         # LLM-as-judge evaluation
|   +-- wiki_utils.py        # Wikipedia text utilities
|   +-- reporting.py         # Report generation
+-- data/
|   +-- coverage_analysis_800.json   # 276 answerable questions
|   +-- grasp_eval_dbpedia*.jsonl    # GRASP evaluation results
|   +-- sparql_eval*.jsonl           # SPARQL evaluation results
|   +-- graphrag_eval_failures*.jsonl# Failure analysis data
|   +-- both_systems_failures.json   # Cross-system failure comparison
+-- docs/
    +-- wikimultihop-dbpedia-answerability.md  # Coverage analysis
    +-- evaluation-section.md                  # Comparative evaluation
    +-- hybrid-pipeline-report.md              # Hybrid pipeline design
```

## Ablation Study

The `ablation_study_v3/` module evaluates 5 levels of KG detail to measure the impact of structured knowledge on answer quality:

| Level | Description | Context Provided |
|-------|-------------|-----------------|
| 0 | No KG (Entity Names Only) | Just entity names from the question |
| 1 | Entity-Only KG | Verified DBpedia entity URIs |
| 2 | Entity + Important Relations | Key properties from schema discovery |
| 3 | Full KG + Wikipedia Fallback | Complete KG triples + Wikipedia text |
| 4 | Oracle KG | Ground truth evidence triples |

## Key Technical Decisions

### Schema-Based Route Decision
- LLM examines discovered schema *before* generating SPARQL
- If properties can answer the question (e.g., `dbo:director` for "Who directed X?") -> SPARQL route
- If schema lacks needed properties (e.g., only `dbo:description` for plot questions) -> TEXT route
- Prevents wasted SPARQL generation and wrong answers from incorrect property matches

### DBpedia Spotlight Entity Linking
- `min_confidence=0.5` filters noise (e.g., "Albert" -> Uncle Albert at 0.40)
- Built-in fallback to SPARQL label search when Spotlight fails
- Retries via tenacity (3 attempts, exponential backoff)

### KG Exploration
- Filters to `dbo:`/`dbp:` predicates only, skips `wikiPageWikiLink`
- Type-aware neighbor retrieval via `get_neighbors_with_types()`
- KG exploration adds value ~60% of the time (property coverage)

### Schema Discovery
1. **Cached class schemas** for 6 common DBpedia classes (Person, Film, Organisation, Place, Work, Band)
2. **Dynamic property discovery** via SPARQL queries for unknown entities
3. Schema includes exact URIs, property names, and sample values for LLM context

### Text Fallback: Auto-Chaining + Relevance Filtering
- **Auto-chain**: tries LLM-generated context SPARQL first, falls back to KG exploration if no new entities found
- **Entity relevance filter**: drops context entities whose labels share no words with the question or seed entities
- **Sentence-level section extraction**: splits oversized paragraphs into sentences, scores by keyword density
- **Keyword-ranked context assembly**: pages ranked by keyword density, most relevant fill 12K budget first

## Dependencies

### Core
- `sparqlwrapper` - SPARQL query execution
- `httpx` - HTTP client (entity linking)
- `requests` - HTTP client (Wikipedia API)
- `spacy` (en_core_web_sm) - NLP entity extraction
- `tenacity` - Retry logic

### LLM & Search
- `groq` - Groq API (llama-3.3-70b-versatile)
- `ddgs` - DuckDuckGo search for Wikipedia page discovery

### Evaluation
- `numpy` - Numerical computation
- `tqdm` - Progress bars
- `websockets` - GRASP WebSocket client

## License

MIT License
