# Scripts

## Evaluation Runners

| Script | Purpose | Output |
|---|---|---|
| `hybrid_batch_eval.py` | Batch-run hybrid pipeline on 1000q set (auto-resumes from JSONL) | `data/eval_1000q/hybrid_eval_*.jsonl` |
| `judge_hybrid_eval.py` | LLM-as-judge re-scoring of hybrid eval results | `*_judged.jsonl` variants |
| `unified_comparison.py` | Cross-pipeline comparison tables (SPARQL vs GRASP vs Hybrid) | stdout / markdown |
| `compare_evals_statistical.py` | Statistical significance: McNemar test, Wilson CI, Bonferroni | stdout |
| `graphrag_eval_failures.py` | Analyze and categorize GraphRAG pipeline failures | `data/eval_1000q/graphrag_eval_failures.jsonl` |
| `grasp_evaluate_results.py` | Offline re-evaluation/re-scoring of GRASP JSONL results | stdout |

## Analysis Tools

| Script | Purpose | Output |
|---|---|---|
| `check_answer_sources.py` | Analyze where correct answers come from (KG vs Wikipedia text) | `data/eval_1000q/answer_source_analysis_1000.json` |
| `classify_1000q_scenarios.py` | Classify questions by KG coverage scenario (full/partial/none) | `data/eval_1000q/scenario_classification_1000.json` |
| `extract_linearrag_1k.py` | Extract LinearRAG results aligned to our 1000q subset | `data/old_800q_results/linearrag_eval_subset.json` |

## Usage

All scripts assume the virtual environment is activated and the project is installed:

```bash
source venv/bin/activate
pip install -e ".[dev]"
```

Evaluation runners require `GROQ_API_KEY` or `GITHUB_TOKEN`. Most support `--limit N` for quick tests.
