# Scripts

## Evaluation Runners

| Script | Purpose | Output |
|---|---|---|
| `hybrid_batch_eval.py` | Batch-run SchemaRAG pipeline on 1000q set (auto-resumes from JSONL) | `data/eval_1000q/hybrid/hybrid_eval_*.jsonl` |
| `judge_hybrid_eval.py` | LLM-as-judge re-scoring of SchemaRAG eval results | `*_judged.jsonl` variants |
| `run-hybrid-cross-benchmark.py` | Run SchemaRAG on HotPotQA / MuSiQue 1000q subsets | `data/cross_benchmark/<bench>/hybrid_<bench>_1000q.jsonl` |
| `run-logicrag-1000q.py` | Run LogicRAG baseline on 2WikiMultiHopQA 1000q | `data/eval_1000q/logicrag/logicrag_1000q_traced.jsonl` |
| `run-logicrag-cross-benchmark.py` | Run LogicRAG on HotPotQA / MuSiQue | `data/cross_benchmark/<bench>/logicrag_<bench>_1000q.jsonl` |
| `convert-linearrag-predictions.py` | Convert LinearRAG predictions JSON → eval-compatible JSONL | `data/eval_1000q/linearrag/linearrag_1000q.jsonl` |

## Analysis & Statistics

| Script | Purpose | Output |
|---|---|---|
| `unified_comparison.py` | Cross-pipeline comparison tables (Schema-Guided SPARQL vs SchemaRAG vs LinearRAG) | stdout / markdown |
| `compare_evals_statistical.py` | Statistical significance: McNemar, Wilson CI, Bonferroni | stdout |
| `cross-benchmark-statistical-analysis.py` | Cross-benchmark significance tests | stdout |
| `compute-cross-benchmark-metrics.py` | Aggregate metrics across benchmarks | `data/cross_benchmark/metrics_summary.json` |
| `compute-routing-breakdown.py` | Routing decision breakdown (Table 1 of supplementary) | stdout |

## Coverage / Grounding (`grounding/`)

| Script | Purpose | Output |
|---|---|---|
| `verify-2wiki-grounding.py` | Hop-by-hop verification of 2WikiMultiHopQA against DBpedia | `data/eval_1000q/ground-truth/hop_verification_results.json` |
| `verify-hotpotqa-grounding.py` | Same for HotPotQA | `data/cross_benchmark/hotpotqa/dbpedia_grounding_results.json` |
| `verify-musique-grounding.py` | Same for MuSiQue (per-hop) | `data/cross_benchmark/musique/ablation/dbpedia_grounding_results.json` |
| `aggregate-coverage-tables.py` | Aggregate cross-benchmark coverage tables | stdout / markdown |
| `musique-sparql-helpers.py`, `musique-utils.py` | Helpers for MuSiQue grounding | (imported) |

## Usage

All scripts assume the virtual environment is activated and the project is installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev,rag]"
```

Evaluation runners require `GROQ_API_KEY` or `GITHUB_TOKEN`. Most support `--limit N` for quick tests.
