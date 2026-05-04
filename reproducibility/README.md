# Reproduce `input/xp.tex`

`reproduce-xp-numbers.ipynb` re-derives every number, table, and figure
in the paper's experiment section from raw eval JSONLs.

## Run

From the repository root:

```bash
cd reproducibility
python build-notebook.py        # regenerate the source notebook (optional; tracked in repo)
jupyter nbconvert --to notebook --execute reproduce-xp-numbers.ipynb \
    --output reproduce-xp-numbers-executed.ipynb \
    --ExecutePreprocessor.timeout=300
```

Or open in Jupyter and run all cells (Run → Run All). The notebook
auto-detects the repo root from the current working directory; running
from either the repo root or `reproducibility/` works.

## What it reproduces

| Section | Source in xp.tex | Status |
|---|---|---|
| Dataset stats (per-type counts) | §4.1 L37-74 | OK |
| KG coverage rates | §4.1 L83 / coverage-methodology.tex | OK |
| Table 1 — End-to-end accuracy run-1 | L177-212 | OK |
| Table 2 — Per-type accuracy mean of 2 runs | L290-317 | OK |
| Table 3 — Routing breakdown | L393-431 | OK |
| Figure 1 — Routing distribution | L433 | OK |
| Inline prose numbers (engagement, decomposition, hop-conn, error count) | L222-286, L500-560 | OK |
| Appendix Table 6 — Per-run breakdown | appendix.tex L24-63 | OK |
| Appendix Table — `tab:hop-depth` | error-analysis.tex L133-152 | OK |

## Data locations (relative to repo root)

- 2Wiki run-1 (gemma judged): `data/eval_1000q/{hybrid,linearrag,logicrag}/...`
- HotpotQA + MuSiQue: `data/cross_benchmark/{hotpotqa,musique}/...`
- Run-2 variants: same paths, `_run2_judged` suffix.
- Coverage analysis: `data/eval_1000q/ground-truth/coverage_analysis_1000.json`
- HotpotQA grounding: `data/cross_benchmark/hotpotqa/dbpedia_grounding_results.json`
- MuSiQue grounding: `data/cross_benchmark/musique/ablation/dbpedia_grounding_results.json`

## Implementation notes

- `normalize_date()` and `normalize_answer()` are inlined from
  `evaluation/metrics.py`. Containment is recomputed against
  the union {gold_answer} ∪ gold_aliases (matches the stored field on
  benchmarks where aliases exist, e.g. MuSiQue/HotpotQA).
- `judge_correct` is the Gemma-3-12B LLM-as-judge field. Run-2 judging
  is recorded in `_run2_judge_logs/` files under each system folder.
- LinearRAG MuSiQue run-2 jsonl drops `question_type`; the notebook
  reconstructs types via `question_id` → run-1 type lookup.
- The KG-only system (`\sysnameKG`) numbers are derived from the Hybrid
  jsonl: a question is "SPARQL-only" iff `method=="sparql"` OR
  (`sub_question_log` is non-empty AND every entry has
  `method=="sparql"`). The non-empty constraint avoids vacuous
  truth on records with missing/empty decomposition logs.
  Acc = #(SPARQL-only ∧ judge_correct) / 1000.
- Per-run-breakdown table (Appendix Table 6): a handful of cells
  may differ by ±0.1pp from the paper. The paper rounds from raw
  count ratios; the notebook uses Python's banker's-rounding
  `round(x, 1)`. The drift is cosmetic and within rounding
  tolerance — the underlying counts agree.

## Runtime

Full notebook executes in ~10-15 seconds on first run (loads ~36 JSONL
files of 1000 records each plus a 1k-question coverage JSON). No LLM
calls or network I/O.

## Dependencies

- Python 3.10+
- pandas, numpy, matplotlib (stdlib otherwise)
- jupyter / nbconvert if running headless

## Sanity check

The final cell prints PASS/FAIL per section. Expected: `OVERALL: PASS`.
