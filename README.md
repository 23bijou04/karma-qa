# DBpedia Hybrid QA

Hybrid QA over DBpedia + Wikipedia for 2WikiMultiHopQA, HotPotQA, MuSiQue.

## Key Results

Evaluated on a 1000-question subset (from [LinearRAG](https://github.com/Zly0523/linear-rag) evaluation set). All systems judged with the same LLM judge (gemma3:12b) for fair comparison.

### Full 1000q (All Questions)

| System | Judge EM | 95% CI |
|--------|----------|--------|
| **Hybrid (ours)** | **89.4%** (894/1000) | [87.3%, 91.2%] |
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

### Cross-Benchmark (1000q each)

| System    | 2WikiMultiHopQA | HotPotQA |
|-----------|:-:|:-:|
| Hybrid    | 89.4%      | 86.1%      |
| LinearRAG | 84.0%      | 86.9%      |
| LogicRAG  | 74.5%      | 78.8%      |

### By Question Type (1000q)

| System | bridge_comparison | comparison | compositional | inference |
|--------|:-:|:-:|:-:|:-:|
| **Hybrid** | 91.9% (216/235) | **95.9%** (234/244) | **82.6%** (341/413) | **95.4%** (103/108) |
| LinearRAG | **93.6%** (220/235) | 96.7% (236/244) | 71.9% (297/413) | 80.6% (87/108) |
| LogicRAG | 72.8% (171/235) | 92.6% (226/244) | 61.7% (255/413) | 86.1% (93/108) |

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

## Requirements

- Python 3.10+
- macOS or Linux (uses `signal.SIGALRM`)
- `GROQ_API_KEY` (free tier: <https://console.groq.com>) **or** `GITHUB_TOKEN`
- Optional: Ollama + `gemma3:12b` (for the LLM judge step)

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]" ".[rag]"
python -m spacy download en_core_web_sm

# Pre-cache embedding models (~100 MB each)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Clone baselines used by evaluation scripts
git clone https://github.com/chensyCN/LogicRAG.git baselines/LogicRAG
git clone https://github.com/Zly0523/linear-rag.git baselines/LinearRAG
# LinearRAG is self-contained with its own venv; see baselines/LinearRAG/REPRODUCE.md

# LLM judge (gemma3:12b via Ollama) — required to reproduce headline Judge EM numbers
# macOS: brew install ollama && brew services start ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh && systemctl --user start ollama
ollama pull gemma3:12b   # ~8 GB, one-time
ollama list              # verify gemma3:12b is present
```

Key deps (declared in `pyproject.toml`): `groq`, `openai`, `sentence-transformers`, `faiss-cpu`, `spacy`, `tenacity`, `ddgs`, `websockets`, `huggingface_hub`.

## Run the main experiment

```bash
export GROQ_API_KEY="..."

# 2WikiMultiHopQA (1000q, ~5 h on Groq free tier, auto-resumes on 429s)
python scripts/hybrid_batch_eval.py --api-key $GROQ_API_KEY

# Cross-benchmark
python scripts/run-hybrid-cross-benchmark.py --benchmark hotpotqa --api-key $GROQ_API_KEY
python scripts/run-hybrid-cross-benchmark.py --benchmark musique  --api-key $GROQ_API_KEY

# Judge raw outputs (gemma3:12b via Ollama; or use --provider groq)
python scripts/judge_hybrid_eval.py data/eval_1000q/hybrid/hybrid_eval_1000.jsonl
```

LLM settings: `temperature=0`, exponential-backoff retry on 429s. Groq routes across heterogeneous accelerators — expect ±0.3pp drift across reruns.

## Result files

Headline: **Hybrid 89.4 %** (894/1000) Judge EM on 2WikiMultiHopQA, **92.9 %** (342/368) on the hop-verified subset.

| Result | JSONL |
|---|---|
| Hybrid 1000q (judged) | `data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl` |
| LinearRAG 1000q (judged) | `data/eval_1000q/linearrag/linearrag_1000q_judged.jsonl` |
| LogicRAG 1000q (judged) | `data/eval_1000q/logicrag/logicrag_1000q_traced_judged.jsonl` |
| SPARQL component 368q | `data/eval_1000q/sparql/sparql_*_judged.jsonl` |
| GRASP 368q | `data/eval_1000q/grasp/grasp_*_judged.jsonl` |
| Hybrid HotPotQA | `data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged.jsonl` |
| Hybrid MuSiQue | `data/cross_benchmark/musique/hybrid_musique_1000q_judged.jsonl` |
| 1000q manifest | `data/eval_1000q/ground-truth/coverage_analysis_1000.json` |
| 368q hop verification | `data/eval_1000q/ground-truth/hop_verification_results.json` |

Verify a row: `jq '[.judge_correct] | add' <file>`.

## Project Structure

Selected files shown — see repo for full layout.

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
├── models.py                   # Entity, HybridResult dataclasses + extract_final_answer()
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
│   ├── grounding/                          # Hop-verification scripts (2Wiki/HotPotQA/MuSiQue)
│   └── ...
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
