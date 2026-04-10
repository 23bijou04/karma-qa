# Data Directory

## Benchmark Data

- `dev_full.json` — 2WikiMultiHopQA dev set: 12,576 questions (53MB, gitignored)
- `test_full.json` — 2WikiMultiHopQA test set (53MB, gitignored)
- `train_full.json` — 2WikiMultiHopQA train set (682MB, gitignored)
- `sample_2wikimultihopqa.json` — Small sample for quick testing

## Evaluation Results (`eval_1000q/`)

Primary evaluation campaign on 1000-question subset (from LinearRAG evaluation set, Zly0523/linear-rag).

### Coverage & Classification
- `coverage_analysis_1000.json` — Per-question KG coverage + answerability flags (original analysis, narrow PROPERTY_MAP: 38.5% property, 34.6% answerable)
- `coverage_analysis_1000_corrected.json` — Predicate-expanded correction (41.6% property, 37.5% answerable; intermediate, superseded by hop verification)
- `hop_verification_results.json` — **Ground truth**: hop-by-hop chain verification against live DBpedia (36.8% predicate-complete, 11.9% strict value match). Script: `scripts/verify_hop_coverage.py`
- `scenario_classification_1000.json` — Questions classified by KG coverage scenario (produced by `scripts/classify_1000q_scenarios.py`, based on original analysis)
- `answer_source_analysis_1000.json` — Analysis of where correct answers originate (produced by `scripts/check_answer_sources.py`)

### SPARQL-Only Baseline
- `sparql_eval_1000.jsonl` — 346 evaluated questions (original answerable set; JSONL, append-only with resume)
- `sparql_eval_1000_summary.json` — Aggregate metrics

### GRASP Baseline
- `grasp_eval_dbpedia.jsonl` — 346 evaluated questions (original answerable set)
- `grasp_eval_dbpedia_summary.json` — Aggregate metrics
- `grasp_eval_dbpedia_traces/` — Full WebSocket traces per question (348 files)
- `grasp_failure_deep_analysis.md` — Manual analysis of GRASP failure modes
- `grasp_trace_examples.md` — Annotated trace examples

### Hybrid Pipeline
- `hybrid_eval_1000.jsonl` — Full 1000-question eval (in progress, auto-resumes)

### Scenario Evals
- `scenario_eval_hybrid.jsonl` — Hybrid pipeline on 60 scenario-stratified questions (10 per A–F)
- `scenario_eval_hybrid_judged.jsonl` — Same with LLM-as-judge scores
- `scenario_eval_logicrag_baseline.jsonl` — LogicRAG-only baseline on same 60 questions
- `scenario_eval_logicrag_baseline_judged.jsonl` — Same with LLM-as-judge scores

### Cross-Pipeline
- `evaluation_report_sparql_grasp.md` — SPARQL vs GRASP comparison report

## Historical Results (`old_800q_results/`)

Earlier evaluation campaign on 800-question sample (superseded by 1000q).

- `coverage_analysis_800.json` — Per-question KG coverage (earlier version)
- `sparql_eval.jsonl` / `sparql_eval_judged.jsonl` — SPARQL component results (Hybrid's SPARQL path standalone)
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
