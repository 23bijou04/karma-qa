# DBpedia Hybrid QA for 2WikiMultiHopQA

A hybrid question-answering system that combines **DBpedia knowledge graph** with **Wikipedia text retrieval** and **LLM reasoning** to answer multi-hop questions from the [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) benchmark (12,576 dev questions, 4 types).

**Thesis:** An entity-only knowledge graph (DBpedia) combined with Wikipedia text can answer multi-hop questions *without* full relation coverage — by using schema-aware routing to choose SPARQL when the KG has the data and falling back to text retrieval when it doesn't.

## Key Results

Evaluated on a 1000-question subset (from [LinearRAG](https://github.com/Zly0523/linear-rag) evaluation set). All systems judged with the same LLM judge (gemma3:12b) for fair comparison.

### Full 1000q (All Questions)

| System | Judge EM | 95% CI |
|--------|----------|--------|
| **Hybrid (ours)** | **89.4%** (894/1000) | [87.5%, 91.3%] |
| LinearRAG | 84.0% (840/1000) | [81.6%, 86.1%] |
| LogicRAG | 74.5% (745/1000) | [71.7%, 77.1%] |

### Hop-Verified 368q (KG-Answerable Subset)

368 questions where DBpedia has all required predicates at every reasoning hop.

| System | Judge EM | 95% CI |
|--------|----------|--------|
| **Hybrid (ours)** | **92.9%** (342/368) | [89.8%, 95.1%] |
| SPARQL component | 91.6% (337/368) | [88.3%, 94.0%] |
| LinearRAG | 88.0% (324/368) | [84.3%, 91.0%] |
| GRASP | 79.1% (291/368) | [74.6%, 82.9%] |
| LogicRAG | 73.9% (272/368) | [69.2%, 78.1%] |

### By Question Type (1000q)

| System | bridge_comparison | comparison | compositional | inference |
|--------|:-:|:-:|:-:|:-:|
| **Hybrid** | 91.9% (216/235) | **95.9%** (234/244) | **82.6%** (341/413) | **95.4%** (103/108) |
| LinearRAG | **93.6%** (235) | 96.7% (244) | 71.9% (413) | 80.6% (108) |
| LogicRAG | 72.8% (235) | 92.6% (244) | 61.7% (413) | 86.1% (108) |

## Architecture

### Schema-Routed Hybrid Pipeline (`hybrid_qa.py`)

The core idea: **use the KG schema to decide the solving strategy**. The system inspects what DBpedia actually knows about the entities in the question, then picks one of two routes:

```
Question
   |
1. Entity Linking — DBpedia Spotlight (spaCy fallback)
   |
2. Schema Discovery — entity classes, properties, sample values via SPARQL
   |
3. Schema Projection — question-aware property filtering (4-layer pipeline)
   |
4. Route Decision — 3-stage: deterministic rules → SPARQL ASK → LLM fallback
   |
   ├─ SPARQL Direct (schema covers the question)
   │    → LLM generates SPARQL using projected schema URIs
   │    → Execute with retry (feed errors back to LLM)
   │
   └─ LogicRAG Decomposition (multi-hop or incomplete schema)
        → LLM decomposes into dependency-ordered sub-questions
        → Each sub-Q solved independently (SPARQL when KG has it, text when not)
        → Grounded answers chained downstream → final synthesis
   |
Answer (with method + route source attribution)
```

**Text retrieval** (Wikipedia + RAG) is not a separate route — it's the fallback mechanism *within* each sub-question when SPARQL can't answer it, and the last resort if both routes above fail entirely.

### Three-Stage Routing

The routing system (`routing.py`) decides SPARQL-direct vs decomposition:

1. **Deterministic rules** — keyword-to-property lookup (e.g., "directed" → `dbo:director`), 2-hop chain matching, comparison detection
2. **SPARQL ASK verification** — tests if the proposed triple actually exists in DBpedia before committing to SPARQL
3. **LLM fallback** — if no rule matches, asks the LLM to classify based on discovered schema

### Comparison Systems

| System | Entry Point | Strategy | Scope |
|--------|-------------|----------|-------|
| **Hybrid (main)** | `hybrid_qa.py` | Schema-routed: SPARQL direct or LogicRAG decomposition | Our contribution |
| LLM-SPARQL | `sparql_evaluation.py` | SPARQL-only with schema discovery | Component baseline |
| GRASP | `grasp_evaluation.py` | External KG QA via WebSocket API | External baseline |
| LinearRAG | `baselines/LinearRAG/` | Linear decomposition + retrieval | External baseline |
| LogicRAG | `baselines/LogicRAG/` | Logic-based decomposition | External baseline |
| GraphRAG | `graphrag_wiki.py` | Text-only: KG exploration + Wikipedia + LLM | Earlier prototype |

## Question Types

The 2WikiMultiHopQA benchmark has 4 question types, each requiring different reasoning strategies:

| Type | Hops | Strategy | Example |
|------|:----:|----------|---------|
| **Comparison** | 1 | Get same property for 2 entities, compare | "Who was born first, Einstein or Newton?" |
| **Inference** | 2 | Follow logical chain (e.g., grandfather = father→father) | "Who is the paternal grandfather of X?" |
| **Compositional** | 2 | Find bridge entity, then target property | "Who founded the company that distributed La La Land?" |
| **Bridge-Comparison** | 2–3 | Find bridges for 2 entities, then compare | "Which film has the director born first?" |

## DBpedia Answerability

Not all questions can be answered from DBpedia. Our hop-by-hop verification measures this:

| Metric | Full Dev Set (12,576q) | 1000q Eval Subset |
|--------|:-:|:-:|
| Entity Coverage | 92.5% | 93.6% (verified live) |
| Property Coverage (all hops) | — | 36.8% (368/1000) |
| Strict Value Match | — | 11.9% (119/1000) |

**Property coverage is the bottleneck**, not entity coverage. Of the 1000 questions, only 368 have all required DBpedia predicates at every reasoning hop. The low strict value match (11.9%) is due to date format mismatches, entity disambiguation, and label normalization — not missing data.

Ground truth: `data/eval_1000q/hop_verification_results.json` (script: `scripts/verify_hop_coverage.py`). See [docs/wikimultihop-dbpedia-answerability.md](docs/wikimultihop-dbpedia-answerability.md) for full analysis.

## Installation

**Requirements:** Python 3.10+

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install core package in development mode
pip install -e ".[dev]"

# Install RAG dependencies (semantic retrieval with FAISS + cross-encoder)
pip install -e ".[rag]"

# Install LLM client and search
pip install groq ddgs

# Download spaCy NLP model (used for entity extraction fallback)
python -m spacy download en_core_web_sm

# Pre-download embedding models (~100MB each, required for RAG path)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

## Data

**To verify our results:** All evaluation results are committed in `data/eval_1000q/` — no additional downloads needed. The 1000-question subset, ground truth, and all judged results are in the repository.

**To re-run evaluations from scratch:** The evaluation scripts load questions from `data/eval_1000q/ground-truth/coverage_analysis_1000.json` (committed), so you can re-run the Hybrid pipeline without the full benchmark dataset.

**To run analysis scripts** (hop verification, benchmark error detection, entity linking accuracy): These require the full 2WikiMultiHopQA dev set (`dev_full.json`, 53MB, gitignored). Download from the [2WikiMultiHopQA repository](https://github.com/Alab-NII/2wikimultihop) and place in `data/dev_full.json`.

**Baselines:** LinearRAG and LogicRAG are external systems with their own environments:

```bash
git clone https://github.com/DEEP-PolyU/LinearRAG.git baselines/LinearRAG
git clone https://github.com/chensyCN/LogicRAG.git baselines/LogicRAG
# Each baseline has a REPRODUCE.md with setup instructions
```

## Quick Start

### 1. Set your API key

```bash
# Option A: Groq (default, free tier at console.groq.com)
export GROQ_API_KEY="your-key-here"

# Option B: GitHub Models
export GITHUB_TOKEN="your-token-here"
```

### 2. Ask a question

```bash
# Simple one-liner
python hybrid_qa.py "Who directed Inception?"
# → Answer (sparql, route=direct_property): Christopher Nolan

# See the full pipeline in action (recommended!)
python hybrid_qa.py "Who directed Inception?" --verbose

# Multi-hop question (triggers LogicRAG decomposition)
python hybrid_qa.py "Where was the director of Inception born?" --verbose

# Comparison question
python hybrid_qa.py "Who is older, Obama or Trump?" --verbose

# Use GitHub Models instead of Groq
python hybrid_qa.py "When was Einstein born?" --provider github --model gpt-4o
```

### 3. Debug mode (maximum detail)

```bash
# Shows SPARQL gap analysis, decomposition traces, and all intermediate results
python hybrid_qa.py "Who directed Inception?" --debug
```

### 4. Run evaluations

```bash
# Hybrid batch evaluation (1000q, auto-resumes from existing results)
python scripts/hybrid_batch_eval.py --api-key $GROQ_API_KEY --limit 5

# LLM judge for hybrid results
python scripts/judge_hybrid_eval.py data/eval_1000q/hybrid/hybrid_eval_1000.jsonl

# Cross-system comparison (Hybrid vs SPARQL vs GRASP vs LinearRAG)
python scripts/unified_comparison.py

# Statistical significance tests
python scripts/compare_evals_statistical.py
```

### 5. Cross-benchmark evaluation (HotPotQA, MuSiQue)

```bash
# Prepare HotPotQA + MuSiQue datasets
python scripts/prepare-cross-benchmark-data.py

# Run Hybrid on HotPotQA + MuSiQue
python scripts/run-hybrid-cross-benchmark.py --benchmark hotpotqa
python scripts/run-hybrid-cross-benchmark.py --benchmark musique

# Run LogicRAG baseline on same benchmarks
python scripts/run-logicrag-cross-benchmark.py --benchmark hotpotqa
python scripts/run-logicrag-cross-benchmark.py --benchmark musique

# Aggregate metrics + statistical tests
python scripts/compute-cross-benchmark-metrics.py
python scripts/cross-benchmark-statistical-analysis.py
```

## Configuration

```bash
# Required: one of these (depending on --provider)
export GROQ_API_KEY="your-key-here"      # for --provider groq (default)
export GITHUB_TOKEN="your-token-here"    # for --provider github

# Optional: custom endpoints (defaults work fine)
export DBPEDIA_SPARQL_ENDPOINT="https://dbpedia.org/sparql"
export DBPEDIA_SPOTLIGHT_ENDPOINT="https://api.dbpedia-spotlight.org/en/annotate"
```

### LLM Provider Selection

All pipelines support `--provider groq|github` and `--model <name>`:

| Provider | Default Model | Env Var | Flag |
|----------|---------------|---------|------|
| **Groq** (default) | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | `--provider groq` |
| GitHub Models | `gpt-4o` | `GITHUB_TOKEN` | `--provider github` |

## Project Structure

```
.
├── hybrid_qa.py                # Main pipeline: schema-routed SPARQL + LogicRAG
├── routing.py                  # 3-stage routing: rules → ASK → LLM
├── sparql_solver.py            # SPARQL generation, execution, retry
├── text_solver.py              # Text retrieval path (RAG + Wikipedia)
├── logicrag_decomposition.py   # Multi-hop sub-question decomposition
├── logicrag_prompts.py         # Prompt templates for LogicRAG
├── schema_projection.py        # Question-aware schema projection (4-layer)
├── question_analysis.py        # Question type classification
├── entity_discovery.py         # Type-guided entity discovery from KG
├── entity_profile.py           # Entity profile construction from KG
├── graphrag_wiki.py            # Wikipedia retrieval + entity linking helpers
├── llm_provider.py             # Unified LLM factory (Groq, GitHub Models)
├── models.py                   # Entity, Triple, HybridResult dataclasses
├── wiki_utils.py               # Wikipedia text fetch utilities
├── sampling.py                 # Stratified sampling + McNemar test
├── llm_judge.py                # LLM-as-judge helper
│
├── dbpedia/                    # KG integration layer
│   ├── client.py               #   SPARQL client with rate limiting + caching
│   ├── entity_linker.py        #   Spotlight API + SPARQL label fallback
│   ├── schema_discovery.py     #   Cached + dynamic schema discovery
│   └── cache.py                #   In-memory cache manager
│
├── rag/                        # Semantic retrieval (bi-encoder + cross-encoder)
│   ├── chunker.py              #   Section-aware Wikipedia text splitting
│   ├── vector_store.py         #   FAISS index + sentence-transformers
│   ├── reranker.py             #   Cross-encoder reranking with relation hints
│   └── cache.py                #   SQLite persistence (data/wiki_cache.db)
│
├── evaluation/
│   └── metrics.py              # Official EM/F1/containment metrics
│
├── sparql_evaluation.py        # LLM-SPARQL evaluation baseline + helper library
├── grasp_evaluation.py         # GRASP WebSocket evaluation client
│
├── scripts/                    # Evaluation scripts (reproduction)
│   ├── hybrid_batch_eval.py                # Hybrid batch runner (1000q, auto-resume)
│   ├── judge_hybrid_eval.py                # LLM-as-judge
│   ├── unified_comparison.py               # Cross-system table (2WikiMultiHopQA)
│   ├── compare_evals_statistical.py        # McNemar + Wilson CI
│   ├── run-hybrid-cross-benchmark.py       # Hybrid on HotPotQA/MuSiQue
│   ├── run-logicrag-cross-benchmark.py     # LogicRAG on HotPotQA/MuSiQue
│   ├── run-logicrag-1000q.py               # LogicRAG on 2WikiMultiHopQA
│   ├── convert-linearrag-predictions.py    # LinearRAG output → eval format
│   ├── prepare-cross-benchmark-data.py     # HotPotQA + MuSiQue data prep
│   ├── compute-cross-benchmark-metrics.py  # Cross-benchmark metrics
│   ├── cross-benchmark-statistical-analysis.py  # Cross-benchmark stats
│   └── grasp_evaluate_results.py           # GRASP offline re-scoring
│
├── baselines/
│   ├── LinearRAG/              # LinearRAG baseline (self-contained, own venv)
│   └── LogicRAG/               # LogicRAG baseline
│
├── data/
│   ├── eval_1000q/             # 2WikiMultiHopQA 1000q results (JSONL)
│   └── cross_benchmark/        # HotPotQA + MuSiQue results (JSONL)
│
├── tests/                      # Unit/integration tests
└── docs/                       # Project documentation
```

## How It Works (Step by Step)

Every question goes through the same first three steps, then diverges based on schema coverage:

**Shared steps (every question):**
1. **Entity Linking** — DBpedia Spotlight maps mentions to KG entities (e.g., "Inception" → `dbr:Inception_(film)`)
2. **Schema Discovery** — Queries DBpedia for entity classes and properties (e.g., Film has `dbo:director`, `dbo:starring`, ...)
3. **Schema Projection** — Filters to properties relevant to *this* question (e.g., "directed" → `dbo:director`)

**Example A — SPARQL Direct:** "Who directed Inception?"
- Routing finds `dbo:director` directly in the schema → SPARQL route
- Generates `SELECT ?x WHERE { dbr:Inception_(film) dbo:director ?x }` → **Christopher Nolan**

**Example B — LogicRAG Decomposition:** "Where was the director of Inception born?"
- This is 2-hop: the schema has `dbo:director` but needs a second hop to `birthPlace`
- Decomposes into sub-questions:
  - Sub-Q1 (SPARQL): "Who directed Inception?" → Christopher Nolan
  - Sub-Q2 (SPARQL): "Where was Christopher Nolan born?" → London
- Chains the grounded answers → **London**

Within decomposition, each sub-question independently uses SPARQL when the KG has the data or falls back to Wikipedia text retrieval (RAG with bi-encoder + cross-encoder reranking) when it doesn't.

## Scripts

All evaluation scripts live in `scripts/`.

> **Reproducing the paper numbers:** check out the `paper-baseline` tag before
> running any evaluation script. Later commits on `main` include post-submission
> bug fixes that change runtime behavior on a minority of questions and will
> produce slightly different numbers.
>
> ```bash
> git checkout paper-baseline    # frozen state that produced the reported results
> git checkout main              # resume development branch
> ```

### Reproduction (2WikiMultiHopQA)

| Script | Purpose |
|--------|---------|
| `hybrid_batch_eval.py` | Batch Hybrid evaluation on 1000q (auto-resume JSONL) |
| `run-logicrag-1000q.py` | LogicRAG baseline on 1000q |
| `convert-linearrag-predictions.py` | Convert LinearRAG output to eval format |
| `judge_hybrid_eval.py` | LLM-as-judge (adds `judge_correct` field) |
| `unified_comparison.py` | Cross-system comparison table (Hybrid, SPARQL, GRASP, LinearRAG, LogicRAG) |
| `compare_evals_statistical.py` | McNemar + per-type breakdown |

### Reproduction (HotPotQA + MuSiQue)

| Script | Purpose |
|--------|---------|
| `prepare-cross-benchmark-data.py` | Download + verify HotPotQA/MuSiQue datasets |
| `run-hybrid-cross-benchmark.py` | Hybrid on HotPotQA/MuSiQue |
| `run-logicrag-cross-benchmark.py` | LogicRAG on HotPotQA/MuSiQue |
| `compute-cross-benchmark-metrics.py` | Aggregate cross-benchmark metrics |
| `cross-benchmark-statistical-analysis.py` | McNemar across 3 systems × 3 benchmarks |

### GRASP

| Script | Purpose |
|--------|---------|
| `grasp_evaluate_results.py` | Offline re-scoring of GRASP results with LLM judge |

## Key Technical Decisions

### Schema-Aware Routing (Not Blind SPARQL)
The LLM examines the discovered schema *before* generating SPARQL. If properties can answer the question (e.g., `dbo:director` for "Who directed X?"), it takes the SPARQL path. If the schema lacks needed properties, it skips SPARQL entirely — preventing wasted queries and wrong answers from incorrect property matches.

### LogicRAG-Style Decomposition for Multi-Hop
Complex questions are broken into dependency-ordered sub-questions, each routed independently. This lets the system use SPARQL for hops where DBpedia has data and text retrieval for hops where it doesn't — combining strengths of both approaches.

### RAG with Relation-Hint Reranking
The text retrieval path uses a two-stage retrieval pipeline: bi-encoder (sentence-transformers) for recall, cross-encoder for precision. Relation hints from the KG boost Wikipedia sections that match discovered properties (e.g., if we know the question is about `dbo:birthPlace`, sections mentioning "born in" get boosted).

### DBpedia Spotlight Entity Linking
- `min_confidence=0.5` filters noise (e.g., "Albert" → Uncle Albert at 0.40)
- Built-in fallback to SPARQL label search when Spotlight fails
- Post-hoc type validation drops false positives
- Retries via tenacity (3 attempts, exponential backoff)

### External Service Resilience
All external calls (DBpedia SPARQL, Spotlight, Wikipedia API, LLM) have retry logic with exponential backoff. DBpedia SPARQL is rate-limited via a token bucket. IPv4 is forced (IPv6 unreachable on some networks). DuckDuckGo has a circuit breaker (3 failures → 60s cooldown).

## Dependencies

### Core
- `sparqlwrapper` — SPARQL query execution against DBpedia
- `httpx` — HTTP client for entity linking
- `requests` — HTTP client for Wikipedia API
- `spacy` (en_core_web_sm) — NLP fallback for entity extraction
- `tenacity` — Retry logic with exponential backoff
- `openai` — OpenAI-compatible client (used for Groq and GitHub Models)

### RAG (optional, installed via `pip install -e ".[rag]"`)
- `sentence-transformers` — Bi-encoder embeddings (all-MiniLM-L6-v2)
- `faiss-cpu` — Vector similarity search

### LLM & Search
- `groq` — Groq API client
- `ddgs` — DuckDuckGo search for Wikipedia page discovery

### Evaluation
- `numpy` — Numerical computation
- `tqdm` — Progress bars
- `websockets` — GRASP WebSocket client

## Running Tests

```bash
# Run all tests
pytest

# Run a specific test
pytest tests/test_routing_2hop.py::test_name

# Lint
ruff check .
ruff check . --fix   # auto-fix
```

## Citation

If you use this system in your research, please cite:

```bibtex
@mastersthesis{dbpedia_hybrid_qa_2026,
  title     = {Schema-Routed Hybrid Question Answering over DBpedia and Wikipedia},
  author    = {Pham, Thi},
  year      = {2026},
  school    = {University},
  note      = {\url{https://github.com/phamthi1812/dbpedia-qa}}
}
```

## License

MIT License
