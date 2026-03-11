# Data Directory

## Benchmark Data

- `dev_full.json` — 2WikiMultiHopQA dev set: 12,576 questions (53MB, gitignored)
- `test_full.json` — 2WikiMultiHopQA test set (53MB, gitignored)
- `train_full.json` — 2WikiMultiHopQA train set (682MB, gitignored)
- `sample_2wikimultihopqa.json` — Small sample for quick testing

## Evaluation Results (`eval_1000q/`)

Primary evaluation campaign on 1000-question stratified sample.

### Coverage & Classification
- `coverage_analysis_1000.json` — Per-question KG coverage + answerability flags (produced by coverage analysis pipeline)
- `scenario_classification_1000.json` — Questions classified by KG coverage scenario (produced by `scripts/classify_1000q_scenarios.py`)
- `answer_source_analysis_1000.json` — Analysis of where correct answers originate (produced by `scripts/check_answer_sources.py`)

### SPARQL-Only Baseline
- `sparql_eval_1000.jsonl` — 346 evaluated questions (JSONL, append-only with resume)
- `sparql_eval_1000_summary.json` — Aggregate metrics

### GRASP Baseline
- `grasp_eval_dbpedia.jsonl` — 346 evaluated questions
- `grasp_eval_dbpedia_summary.json` — Aggregate metrics
- `grasp_eval_dbpedia_traces/` — Full WebSocket traces per question (348 files)
- `grasp_failure_deep_analysis.md` — Manual analysis of GRASP failure modes
- `grasp_trace_examples.md` — Annotated trace examples

### Hybrid Pipeline
- `hybrid_eval_100_2hop.jsonl` — First 100-question eval (2-hop subset)
- `hybrid_eval_100_2hop_judged.jsonl` — Same with LLM-as-judge scores
- `hybrid_eval_100_2hop_report.md` — Summary report
- `hybrid_eval_100_post_fix.jsonl` — Re-eval after bug fixes
- `hybrid_eval_100_llm_hints.jsonl` — Eval with LLM hint routing
- `hybrid_eval_100_decomp_fix.jsonl` — Eval after decomposition fixes
- `*_judged.jsonl` variants — Each eval re-scored by LLM judge
- `_rerun_100_ids.json` — Question IDs for the 100-question subset

### Cross-Pipeline
- `evaluation_report_sparql_grasp.md` — SPARQL vs GRASP comparison report

## Historical Results (`old_800q_results/`)

Earlier evaluation campaign on 800-question sample (superseded by 1000q).

- `coverage_analysis_800.json` — Per-question KG coverage (earlier version)
- `sparql_eval.jsonl` / `sparql_eval_judged.jsonl` — SPARQL baseline results
- `grasp_eval_dbpedia.jsonl` / `grasp_eval_dbpedia_judged.jsonl` — GRASP results
- `graphrag_eval_failures.jsonl` — GraphRAG failure cases
- `linearrag_eval_subset.json` — LinearRAG results for comparison
- `grasp_verification_results.json` — GRASP answer verification
- `*_traces/` directories — Per-question debug traces
- `*_summary.json` — Aggregate metrics per pipeline

## Other

- `wiki_cache.db` — SQLite cache for Wikipedia page content (used by `rag/cache.py`)
- `ontology/` — DBpedia ontology reference files
- `sparql_eval_traces/` — SPARQL evaluation debug traces (top-level, from early experiments)
