# Data Directory

## Benchmark Data

- `dev_full.json` — 2WikiMultiHopQA dev set: 12,576 questions (gitignored, fetch from upstream: https://github.com/Alab-NII/2wikimultihop)
- `sample_2wikimultihopqa.json` — Small sample for quick testing
- `ontology_cache.json` — DBpedia ontology cache

## Evaluation Results

Primary evaluation campaign on 1000-question subset (sourced from the LinearRAG evaluation set, `Zly0523/linear-rag`).

### `eval_1000q/`

#### Ground truth
- `ground-truth/coverage_analysis_1000.json` — Per-question KG coverage + answerability flags
- `ground-truth/coverage_analysis_1000_corrected.json` — Predicate-expanded correction (intermediate, superseded)
- `ground-truth/hop_verification_results.json` — Hop-by-hop chain verification against live DBpedia (primary ground truth)
- `ground-truth/hop_verification_results_260414.json` — Re-run (zero drift)
- `ground-truth/property_coverage_verification.json`, `property_verification_results.json` — Property-level checks
- `ground-truth/scenario_classification_1000.json` — KG-coverage scenario labels (A–F)
- `ground-truth/entity_qid_cache.json`, `benchmark_error_flags.json` — Caches/flags

#### Per-system results (raw + LLM-judged)
- `hybrid/hybrid_eval_1000{,_judged,_run2_judged}.jsonl` — SchemaRAG primary
- `hybrid/hybrid_hop368_judged.jsonl` — Hop-verified 368q subset
- `hybrid/ablation/*.jsonl`, `mixed_route_cohort_ids.txt` — Ablation cohorts
- `linearrag/linearrag_1000q{,_judged,_run2_judged}.jsonl` + `linearrag_hop368_judged.jsonl`
- `linearrag_predictions_1000{,_run2}.json` — Raw LinearRAG predictions
- `logicrag/logicrag_1000q_traced{,_judged,_judged_claude,_run2_judged}.jsonl` + `logicrag_hop368_judged.jsonl`
- `sparql/hop368/sparql_eval_1000_hop368{,_judged,_summary}.{jsonl,json}` — Schema-Guided SPARQL component
- `sparql/original-346q/sparql_eval_1000{,_judged,_summary}.{jsonl,json}` — Earlier SPARQL run
- `unified_comparison_1000.json` — Cross-system per-question comparison

### `cross_benchmark/`

#### HotPotQA (`hotpotqa/`) and MuSiQue (`musique/`)
For each benchmark:
- `{hybrid,linearrag,logicrag}_<bench>_1000q{,_judged,_run2_judged}.jsonl`
- `dbpedia_grounding_results.json` — Entity/connectivity/answer grounding stats
- `ablation/*.jsonl`, `mixed_route_cohort_ids.txt` — Mixed-route / text-only-decomp cohorts

`metrics_summary.json` — Aggregated cross-benchmark metrics.

## Other

- `wiki_cache.db` — SQLite cache for Wikipedia page content (used by `rag/cache.py`, gitignored)
