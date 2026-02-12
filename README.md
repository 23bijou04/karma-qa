# DBpedia GraphRAG for 2WikiMultiHopQA

A GraphRAG system using DBpedia knowledge graph to answer multi-hop questions from the 2WikiMultiHopQA benchmark (192K questions, 4 types).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Question                                                        │
│  "Who is the paternal grandfather of Eadred?"                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Generate SPARQL Query                                   │
│                                                                  │
│  • LLM/template generates multi-hop SPARQL from question         │
│  • Handles all question types with single code path              │
│                                                                  │
│  SELECT ?answer WHERE {                                          │
│    dbr:Eadred dbo:father ?father .                              │
│    ?father dbo:father ?answer .                                 │
│  }                                                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Execute SPARQL → Get Triples                            │
│                                                                  │
│  • Query DBpedia endpoint                                        │
│  • Extract entities and relationships                            │
│  • Caching + Rate Limiting                                       │
│                                                                  │
│  (Eadred, father, Edward_the_Elder)                              │
│  (Edward_the_Elder, father, Alfred_the_Great)                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Get Wikipedia Sentences                                 │
│                                                                  │
│  • Fetch Wikipedia pages for entities in triples                 │
│  • Find sentences containing entity names/relations              │
│                                                                  │
│  [Eadred] "He was the son of Edward the Elder."                  │
│  [Edward] "He was the son of Alfred the Great."                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Assemble Context                                        │
│                                                                  │
│  Knowledge Graph Facts:                                          │
│    - Eadred → father → Edward the Elder                          │
│    - Edward the Elder → father → Alfred the Great                │
│                                                                  │
│  Wikipedia Evidence:                                             │
│    [Eadred:0] "He was the son of Edward the Elder."              │
│    [Edward:0] "He was the son of Alfred the Great."              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: LLM Generates Answer                                    │
│                                                                  │
│  • Question + Context → LLM                                      │
│  • Structured output parsing                                     │
│                                                                  │
│  ANSWER: Alfred the Great                                        │
│  SUPPORTING_FACTS: [Eadred:0], [Edward the Elder:0]              │
└─────────────────────────────────────────────────────────────────┘
```

## Question Types

| Type | Strategy | Hops | Example |
|------|----------|------|---------|
| **Comparison** | Get same property for 2 entities, compare | 0-1 | "Who was born first, Einstein or Newton?" |
| **Inference** | Follow logical chain (e.g., grandfather = father→father) | 2 | "Who is the paternal grandfather of X?" |
| **Compositional** | Find bridge entity, then target property | 2 | "Who founded the company that distributed La La Land?" |
| **Bridge-Comparison** | Find bridges for 2 entities, then compare | 2-3 | "Which film has the director born first?" |

## Installation

```bash
# Install package in development mode
pip install -e ".[dev]"

# Install with LLM support (Ollama/OpenAI)
pip install -e ".[dev,llm]"
```

## Quick Start

### Python API

```python
from dbpedia_qa import create_system

# Create the system
system = create_system()

# Answer a question
answer, supporting_facts, evidence = system.answer_question(
    "Who was born first, Einstein or Lincoln?"
)

print(f"Answer: {answer.text}")
print(f"Confidence: {answer.confidence:.2%}")
print(f"Supporting facts: {supporting_facts}")
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
```

## Evaluation

### Run Benchmark

```python
from dbpedia_qa.evaluation.runner import BenchmarkRunner
from dbpedia_qa import create_system

system = create_system()
runner = BenchmarkRunner(system, dataset_path="2wikimultihopqa_dev.json")

# Run evaluation
results = runner.run(limit=100, verbose=True)

# Print report
print(runner.report())

# Save results
runner.save_results("evaluation_results.json")
```

### Target Metrics

| Metric | LLM-only Baseline | GraphRAG Target |
|--------|-------------------|-----------------|
| Answer F1 | ~42% | 55-60% |
| Supporting Facts F1 | ~65% | 70-75% |
| Evidence F1 | ~15% | 25-35% |
| Joint F1 | ~5% | 15-20% |

## Project Structure

```
dbpedia_qa/
├── __init__.py              # Package exports
├── config.py                # Configuration settings
├── models.py                # Data models (Entity, Triple, etc.)
├── system.py                # Main GraphRAG system
├── cli.py                   # Command-line interface
├── dbpedia/
│   ├── client.py            # SPARQL client + caching
│   ├── property_mapper.py   # Wikidata → DBpedia mapping
│   ├── entity_linker.py     # Spotlight integration
│   └── cache.py             # Redis/in-memory cache
├── question_processing/
│   ├── classifier.py        # Type classification
│   ├── decomposer.py        # Question decomposition
│   └── entity_extractor.py  # Entity extraction
├── traversal/
│   ├── ranked_bfs.py        # Core traversal algorithm
│   ├── relevance_scorer.py  # Scoring functions
│   └── strategies.py        # Type-specific strategies
├── wikipedia/
│   ├── client.py            # Wikipedia API
│   └── supporting_facts.py  # Evidence extraction
├── context/
│   ├── textualizer.py       # Triple → text
│   └── assembler.py         # Context assembly
├── llm/
│   ├── prompts.py           # Prompt templates
│   ├── generator.py         # Answer generation
│   └── parser.py            # Response parsing
├── evaluation/
│   ├── metrics.py           # EM/F1 computation
│   └── runner.py            # Benchmark runner
└── tests/
    ├── test_basic.py        # Unit tests
    └── test_integration.py  # Integration tests
```

## Key Technical Decisions

### Wikidata → DBpedia Property Mapping (~85% coverage)

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

```
# Core
sparqlwrapper>=2.0
requests>=2.28
redis>=4.0 (optional)

# NLP
spacy>=3.5

# LLM (Open-source)
ollama>=0.1
openai>=1.0 (optional)

# Tokenization
tiktoken>=0.5

# Evaluation
numpy>=1.24
tqdm>=4.65
pytest>=7.0
```

## License

MIT License
