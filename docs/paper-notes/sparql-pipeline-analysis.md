# SPARQL Pipeline Analysis

**Sources:** Consolidated from investigation sessions (2026-03-20 through 2026-03-27). Original session reports no longer available; facts verified against code and data files directly.

---

## Routing and Execution Gap (60-question sample)

| Stage | Count | % |
|-------|-------|---|
| Questions attempted via SPARQL | 28/60 | 47% |
| SPARQL attempts that succeeded | 7/28 | 25% |
| Wrong routing (routed to SPARQL, should not have been) | ~11% | — |
| Execution gap (routed correctly, but SPARQL failed) | ~64% | of SPARQL attempts |

**Key finding:** Only 7/60 questions (11.7%) ultimately answered correctly via SPARQL. Routing is a minor issue; execution failure is dominant.

---

## Routing Architecture (3 Stages)

**Stage 1 — Deterministic rules** (`_rule_based_route()`):
- Keyword-to-property dictionary (e.g., "directed" → `dbo:director`)
- 2-hop chain match via `schema_projection.relevant_chains`
- Comparison routing: date/country property detection
- Bridge/parent chain: keyword pattern match

**Stage 2 — LLM property extraction** (`route_question()`):
- If Stage 1 rules return None, LLM extracts properties from question text
- Re-runs deterministic rules with LLM-extracted properties + ASK verification
- NOT a binary SPARQL/TEXT classifier — uses LLM for vocabulary mapping only

**Stage 2b — SPARQL ASK verification** (`verify_sparql_feasibility()`):
- Direct property match and date comparison: pass through without ASK
- Bridge/hop2/parent rules: verify via SPARQL ASK that triple exists
- Returns `ask_verified` or `ask_missing` (downgrades to DECOMPOSE)

**Stage 3 — TEXT fallback:**
- If no property match after Stage 2, unconditionally falls back to TEXT

---

## Comparison Guard Bug (Critical) — FIXED

**Location:** `sparql_solver.py` — comparison query result checking

**Bug:** After executing a comparison SPARQL query, the code checks `if sparql_rows:` to determine if results were returned. However, for comparison queries, results are stored in a different variable — `sparql_rows` is always `[]`.

**Status:** Fixed. The `sparql_rows` variable no longer exists in `sparql_solver.py`. Comparison results are now handled correctly.

**Impact:** 7 correct SPARQL answers discarded for comparison questions. System falls through to text path unnecessarily.

**Affected question types:** All comparison questions routed to direct SPARQL path.

---

## DBpedia Prefix Asymmetry (False ask_verified)

**Issue:** `dbp:` properties exist in DBpedia KG, but ASK verification queries use `dbo:` namespace.

**Result:** ASK query finds triple (via `dbo:` match for a different predicate), returns `True` → `ask_verified` → routes to SPARQL → SPARQL generation fails to produce valid query → silent execution failure.

**Impact:** 3/5 Scenario C failures directly caused by this. False positives in ASK verification.

**Diagnosis:** Not a Spotlight or routing logic bug — purely a namespace prefix mismatch in ASK query generation.

---

## SPARQL Prompt Consistency Issue (Fairness)

**Baseline vs Hybrid prompt divergence:**

| Component | sparql_evaluation.py (baseline) | sparql_solver.py (hybrid) |
|-----------|--------------------------------|--------------------------|
| Prompt | `SPARQL_GEN_PROMPT` (7 rules) | `UNIFIED_SPARQL_PROMPT` (8 rules) |
| Known facts | None | Optional (rule 7 + parameter) |
| Placeholders | `{schema}`, `{question}` | `{schema}`, `{known_facts_block}`, `{question}` |

**Rule 7 (hybrid only):** "If known facts provide entity URIs, use those URIs directly in your query"

**Fairness implication:** Baseline was tested with 7-rule prompt; hybrid SPARQL fallback uses 8-rule prompt. Direct comparison of SPARQL accuracy between baseline and hybrid is not strictly fair.

**Shared functions (identical implementation):** `extract_sparql()`, `sanitize_sparql_uris()`, `execute_sparql()`, `ANSWER_PROMPT`, `SPARQL_RETRY_PROMPT`

**Options for fairness:**
- Option A (preferred): Re-run baseline with `UNIFIED_SPARQL_PROMPT` (empty `known_facts_block`)
- Option B: Document the difference as intentional design choice
- Option C: Align both to same 7-rule prompt

---

## Execution Failure Modes

| Failure Mode | Frequency | Description |
|-------------|-----------|-------------|
| dbp/dbo prefix mismatch | High | False ask_verified; SPARQL then fails on wrong namespace |
| Multi-valued property ordering | Medium | DBpedia returns first listed value; not always the intended one |
| Comparison guard bug (FIXED) | High | `sparql_rows` always empty for comparison queries |
| Transient DBpedia errors | Low | Endpoint timeout; auto-resume handles most |
| Complex aggregation | Medium | COUNT/GROUP BY over large result sets times out |

---

## Key Numbers for Paper

- 28/60 SPARQL attempts → 7 successes = 25% SPARQL execution success rate (60-question sample)
- Comparison guard bug kills 7 correct answers (fixable with one-line code change)
- 3/5 Scenario C failures from prefix asymmetry (fixable)
- Baseline prompt: 7 rules; hybrid: 8 rules — not directly comparable without re-running baseline
