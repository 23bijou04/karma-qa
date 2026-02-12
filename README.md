# DBpedia GraphRAG for 2WikiMultiHopQA

A GraphRAG system that combines the DBpedia knowledge graph with Wikipedia text retrieval and LLM reasoning to answer multi-hop questions from the [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) benchmark (192K questions, 4 types).

## Architecture

```
                          Question
                             |
              +--------------+--------------+
              |                             |
     SPARQL-based Pipeline         Wikipedia-based Pipeline
     (graphrag_wiki.py /            (graphrag_wiki.py)
      dbpedia_qa package)
              |                             |
   +----------+----------+      +----------+----------+
   | 1. Extract entities  |      | 1. Extract entities  |
   |    (spaCy NER +      |      |    (spaCy NER +      |
   |     pattern-based)   |      |     pattern-based)   |
   +----------+----------+      +----------+----------+
              |                             |
   +----------+----------+      +----------+----------+
   | 2. Link to DBpedia   |      | 2. Link to DBpedia   |
   |    (URI resolution +  |      |    (URI verification) |
   |     schema discovery) |      +----------+----------+
   +----------+----------+               |
              |                  +-------+--------+
   +----------+----------+      | 3. Retrieve     |
   | 3. Generate SPARQL   |      |    Wikipedia    |
   |    (LLM or template) |      |    full text    |
   +----------+----------+      +-------+--------+
              |                          |
   +----------+----------+      +-------+--------+
   | 4. Execute SPARQL    |      | 4. Extract      |
   |    against DBpedia   |      |    relevant     |
   +----------+----------+      |    sections     |
              |                  +-------+--------+
   +----------+----------+               |
   | 5. Get Wikipedia      |              |
   |    sentences          |              |
   +----------+----------+               |
              |                          |
              +-----------+--------------+
                          |
               +----------+----------+
               | LLM Reasoning       |
               | (Groq / OpenAI /    |
               |  Ollama)            |
               +----------+----------+
                          |
                       Answer
```

### Detailed SPARQL Pipeline

```
Question: "Who is the paternal grandfather of Eadred?"
     |
     v
STEP 1: Generate SPARQL Query
  SELECT ?answer WHERE {
    dbr:Eadred dbo:father ?father .
    ?father dbo:father ?answer .
  }
     |
     v
STEP 2: Execute SPARQL -> Get Triples
  (Eadred, father, Edward_the_Elder)
  (Edward_the_Elder, father, Alfred_the_Great)
     |
     v
STEP 3: Get Wikipedia Sentences
  [Eadred] "He was the son of Edward the Elder."
  [Edward] "He was the son of Alfred the Great."
     |
     v
STEP 4: Assemble Context
  Knowledge Graph Facts:
    - Eadred -> father -> Edward the Elder
    - Edward the Elder -> father -> Alfred the Great
  Wikipedia Evidence:
    [Eadred:0] "He was the son of Edward the Elder."
    [Edward:0] "He was the son of Alfred the Great."
     |
     v
STEP 5: LLM Generates Answer
  ANSWER: Alfred the Great
  SUPPORTING_FACTS: [Eadred:0], [Edward the Elder:0]
```

## Question Types

| Type | Strategy | Hops | Example |
|------|----------|------|---------|
| **Comparison** | Get same property for 2 entities, compare | 0-1 | "Who was born first, Einstein or Newton?" |
| **Inference** | Follow logical chain (e.g., grandfather = father->father) | 2 | "Who is the paternal grandfather of X?" |
| **Compositional** | Find bridge entity, then target property | 2 | "Who founded the company that distributed La La Land?" |
| **Bridge-Comparison** | Find bridges for 2 entities, then compare | 2-3 | "Which film has the director born first?" |

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

# Install with LLM support (Ollama/OpenAI)
pip install -e ".[dev,llm]"

# Download spaCy model
python -m spacy download en_core_web_sm

# Additional dependencies for Wikipedia pipeline
pip install groq ddgs
```

**Requirements:** Python 3.10+

## Quick Start

### Standalone Wikipedia Pipeline (graphrag_wiki.py)

Answers any question using DBpedia entity linking + Wikipedia full text + Groq LLM:

```bash
# Set your Groq API key
export GROQ_API_KEY="your-key-here"

# Answer a question
python graphrag_wiki.py "Who is the paternal grandfather of Eadred?" --verbose

# Simple query
python graphrag_wiki.py "Who directed Inception?"
```

### Python API (dbpedia_qa package)

```python
from dbpedia_qa import create_system

system = create_system()

answer, supporting_facts, evidence = system.answer_question(
    "Who was born first, Einstein or Lincoln?"
)

print(f"Answer: {answer.text}")
print(f"Confidence: {answer.confidence:.2%}")
print(f"Supporting facts: {supporting_facts}")
```

### SPARQL Generation

```python
from dbpedia_qa import SPARQLGenerator

generator = SPARQLGenerator()

# Template-based (no LLM needed)
query = generator.generate("Who is the paternal grandfather of Eadred?")
print(query)
```

### Command-Line Interface

```bash
# Run component tests
python cli.py test --dry-run

# Interactive demo mode
python cli.py demo

# Answer a single question
python cli.py answer "Who was born first, Einstein or Newton?"

# Run benchmark evaluation
python cli.py benchmark path/to/2wikimultihopqa.json --limit 100 -o results.json
```

### Benchmark Runner

Run the 10-question benchmark with schema discovery and LLM-generated SPARQL:

```bash
python benchmark_runner.py --limit 10 --output results.json
```

## Configuration

The system can be configured via environment variables or programmatically:

```bash
# DBpedia settings
export DBPEDIA_SPARQL_ENDPOINT="https://dbpedia.org/sparql"
export DBPEDIA_SPOTLIGHT_ENDPOINT="https://api.dbpedia-spotlight.org/en/annotate"

# Redis caching (optional)
export REDIS_HOST="localhost"
export REDIS_PORT="6379"

# LLM settings
export LLM_PROVIDER="ollama"  # or "openai"
export LLM_MODEL="llama3.1:70b"
export OPENAI_API_KEY="sk-..."  # if using OpenAI

# Groq (for graphrag_wiki.py and benchmark_runner.py)
export GROQ_API_KEY="your-key-here"
```

Or in Python:

```python
from dbpedia_qa import Config, create_system

config = Config()
config.llm.provider = "ollama"
config.llm.model = "llama3.1:8b"
config.traversal.max_hops = 3
config.traversal.beam_width = 10

system = create_system(config)
```

## Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=dbpedia_qa

# Run specific test file
pytest tests/test_basic.py -v

# Run SPARQL generator tests
pytest tests/test_sparql_generator.py -v
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

Uses stratified sampling across question types with LLM-as-judge evaluation.

## Project Structure

```
dbpedia_qa/
+-- __init__.py              # Package exports
+-- config.py                # Configuration settings
+-- models.py                # Data models (Entity, Triple, etc.)
+-- system.py                # Main GraphRAG system
+-- cli.py                   # Command-line interface
+-- sparql_generator.py      # LLM/template SPARQL generation
+-- graphrag_wiki.py         # Standalone Wikipedia pipeline
+-- benchmark_runner.py      # Benchmark with schema discovery
+-- graphrag_debug.py        # Debug utilities
+-- dbpedia/
|   +-- client.py            # SPARQL client + caching
|   +-- property_mapper.py   # Wikidata -> DBpedia mapping
|   +-- entity_linker.py     # Spotlight integration
|   +-- cache.py             # Redis/in-memory cache
|   +-- schema_discovery.py  # Hybrid schema discovery for entities
|   +-- schema_explorer.py   # Interactive schema exploration
|   +-- ontology_manager.py  # DBpedia ontology management
+-- question_processing/
|   +-- classifier.py        # Type classification
|   +-- decomposer.py        # Question decomposition
|   +-- entity_extractor.py  # Entity extraction
+-- traversal/
|   +-- ranked_bfs.py        # Core traversal algorithm
|   +-- relevance_scorer.py  # Scoring functions
|   +-- strategies.py        # Type-specific strategies
+-- wikipedia/
|   +-- client.py            # Wikipedia API
|   +-- supporting_facts.py  # Evidence extraction
|   +-- section_retriever.py # Section-level retrieval
+-- context/
|   +-- textualizer.py       # Triple -> text
|   +-- assembler.py         # Context assembly
+-- llm/
|   +-- prompts.py           # Prompt templates
|   +-- generator.py         # Answer generation
|   +-- parser.py            # Response parsing
+-- evaluation/
|   +-- metrics.py           # EM/F1 computation
|   +-- runner.py            # Benchmark runner
+-- ablation_study_v3/
|   +-- runner.py            # Ablation study runner
|   +-- sampling.py          # Stratified sampling
|   +-- levels_base.py       # Levels 0, 1, 4
|   +-- levels_kg.py         # Levels 2, 3 (KG-based)
|   +-- llm_judge.py         # LLM-as-judge evaluation
|   +-- wiki_utils.py        # Wikipedia text utilities
|   +-- reporting.py         # Report generation
+-- tests/
|   +-- test_basic.py        # Unit tests
|   +-- test_integration.py  # Integration tests
|   +-- test_sparql_generator.py  # SPARQL generation tests
|   +-- test_llm_sparql_with_schema.py  # Schema-aware tests
+-- docs/
|   +-- wikimultihop-dbpedia-answerability.md  # Coverage analysis
```

## Key Technical Decisions

### Hybrid Schema Discovery

The SPARQL generator uses a hybrid approach:
1. **Cached class schemas** for common DBpedia classes (Person, Film, Place, etc.)
2. **Dynamic property discovery** via SPARQL queries for unknown entities
3. **LLM-based generation** when an LLM client is available, falling back to template patterns

### Wikidata -> DBpedia Property Mapping (~85% coverage)

| Wikidata | DBpedia | Usage |
|----------|---------|-------|
| P569 | dbo:birthDate | date of birth |
| P570 | dbo:deathDate | date of death |
| P22 | dbo:father | father |
| P25 | dbo:mother | mother |
| P57 | dbo:director | film director |
| P577 | dbo:releaseDate | publication date |

### Graph Traversal Parameters

- **Beam width:** 10 neighbors per hop
- **Max hops:** 2-3 (question-type dependent)
- **Relevance scoring:** Keyword matching + optional embeddings

## Dependencies

### Core
- `sparqlwrapper` - SPARQL query execution
- `requests` / `httpx` - HTTP clients
- `spacy` (en_core_web_sm) - NLP entity extraction
- `pydantic` - Data models and validation
- `redis` - Optional caching backend
- `tenacity` - Retry logic

### LLM Providers
- `groq` - Groq API (used by graphrag_wiki.py and benchmark_runner.py)
- `openai` - OpenAI-compatible APIs
- `ollama` - Local LLM inference (optional)

### Search & Retrieval
- `ddgs` - DuckDuckGo search for Wikipedia page discovery
- `rdflib` - RDF/OWL ontology parsing

### Evaluation
- `numpy` - Numerical computation
- `tqdm` - Progress bars
- `pytest` - Testing framework

## License

MIT License
