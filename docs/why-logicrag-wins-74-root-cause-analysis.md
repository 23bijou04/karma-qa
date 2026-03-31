# Why LogicRAG Wins on 74 Questions: Deep Root Cause Analysis

**Date**: 2026-03-30 | **Validated**: 2026-03-31 | **Status**: For Discussion | **Branch**: `experiment/fair-comparison-analysis`

## Context

In the 1000q head-to-head comparison, the hybrid pipeline wins on 170 questions but **loses on 74** where plain LogicRAG succeeds. This document synthesizes findings from three parallel expert investigations that traced the exact code paths behind each failure.

**Critical revision**: The original 5-category classification was a surface-level grouping. The investigations found the actual root mechanisms are different — and in some cases, the original hypotheses were wrong.

---

## TL;DR — What We Found

| Original Category | Original Count | Actual Mechanism | Verdict |
|---|---|---|---|
| SPARQL comparison inversion | 16 | **Not an inversion bug.** Coverage asymmetry: only 1 of 2 entities has data → LLM defaults to that entity → always wrong | Hypothesis refuted |
| Entity linking errors | 10 | **Much bigger than reported.** `bif:contains` injects noise entities in 38.8% of all 1000 questions | Understated |
| Decomposition comparison | 21 | Three sub-patterns: country label non-normalization, unstable "How old" sub-Qs, Spotlight errors | Confirmed + refined |
| Decomposition general | 21 | Three sub-patterns: SPARQL retry confabulation, multi-variable projection strips names, TEXT→SPARQL bridge propagation failure | Confirmed + refined |
| Evidence hedging | 6 | Two sub-mechanisms: entity cross-contamination in UNION queries, `known_facts` bleeding between sub-Qs | Confirmed + refined |

---

## Root Cause 1: SPARQL Coverage Asymmetry (was "Comparison Inversion")

### What the original analysis said
> "SPARQL picks the EXACT OPPOSITE film/person — all 16/16 inverted. This smells like a systematic date comparison bug."

### What we actually found
**There is no comparison operator bug.** No wrong `<` vs `>`, no string-vs-numeric date issue, no ORDER BY inversion.

The actual mechanism is **coverage asymmetry with LLM default behavior**:

```
Question: "Which film has the director born first, Film A or Film B?"

What happens:
1. SPARQL generates query for both films' directors + birth dates
2. DBpedia has birth date for Film A's director but NOT Film B's
3. LLM sees one result row → defaults to Film A as the answer
4. Film A is ALWAYS wrong (the film with data is always the wrong answer)

Why always wrong: DBpedia's coverage gaps are systematic — the correct
answer's director is typically more obscure (less-covered in KG).
```

**Evidence**: 19/19 tested cases show `result_film == predicted_answer` (the film with available data), and the gold answer is always the film with zero data rows.

### Code path
```
routing.py:134        → rule:bridge_comparison matched
routing.py:303        → verify_sparql_feasibility() → ASK succeeds (for ONE entity)
sparql_solver.py:64   → UNIFIED_SPARQL_PROMPT generates comparison SPARQL
sparql_evaluation.py:276 → execute_sparql() returns data for 1 of 2 entities
sparql_evaluation.py:356 → ANSWER_PROMPT — NO instruction for partial data
                           → LLM defaults to entity with data → WRONG
```

### Fix
Add one instruction to `ANSWER_PROMPT` in `sparql_evaluation.py:356`:

> "For comparison questions, if data for one entity is missing, return 'could not determine' rather than defaulting to the entity with available data."

This makes `sparql_solver.py:218` treat it as a null answer and fall through to text.

**⚠ Implementation note (2026-03-31 validation):** The null-answer string MUST match `_NULL_ANSWERS` in `sparql_solver.py:44–47`, which checks for `"could not determine"`. Using `"cannot determine"` (as originally proposed) would bypass the null filter and produce a wrong-shaped answer instead of triggering text fallback.

**Estimated impact**: 19 questions recovered (Category A cases).

> **Partial fix applied (2026-03-31):** ASK verification fix in `routing.py` recovered 12/23 bridge_comparison questions (12 EM 0→1, 0 regressions). The 11 remaining failures are coverage-asymmetry cases not addressable by routing alone.

---

## Root Cause 2: Entity Linking Noise Injection (was "Entity Linking Errors")

### What the original analysis said
> "10 questions, 13.5%. Spotlight links to wrong DBpedia entities."

### What we actually found
**The scale is much larger than reported.** Entity linking injects noise entities in **38.8%** (388/1000) of all questions. The primary source is NOT Spotlight — it's the `bif:contains` fulltext fallback.

### Four distinct failure modes

#### Mode A — `bif:contains` Film Entity Injection (388 questions, 38.8%)

Two DBpedia entities appear in results for nearly every film/director question:
- `Kenji Mizoguchi: The Life of a Film Director` — 270 questions
- `Film in Which There Appear Edge Lettering, Sprocket Holes, Dirt Particles, Etc.` — 263 questions

**Root mechanism**: `_extract_type_hint()` returns `dbo:Film`, and these entities satisfy the type filter because they ARE typed as `dbo:Film`. The type filter constrains to a class, not semantic relevance.

**Critical gap**: Results from the `bif:contains` unlinked loop are **never post-validated** by `_validate_entity_types()` (entity_linker.py:371). This function only runs on Spotlight entities.

**Impact**: Of 388 affected questions, 92 have EM=0. Most failures occur on the TEXT route (noise entities pollute Wikipedia context retrieval), not SPARQL.

#### Mode B — Generic Word Entities (~440 questions, ~44%)

`'birth'` and `'death'` are NOT in `_MENTION_BLOCKLIST` (only `'born'` and `'died'` are).
These resolve to `dbp:birth`, `dbp:death` via exact label match. Similarly `'What'` → `dbr:What`, `'father'` → `dbr:Father_(disambiguation)`.

**Impact**: These clutter schema context but rarely cause EM=0 because SPARQL generation skips property-type URIs.

#### Mode C — Complete Entity Linking Failure (rare, ~1 confirmed SPARQL failure)

When Spotlight AND URI probe both fail, `bif:contains` returns only semantically irrelevant entities. Example: "The Private Life Of Cinema" → only entity found is `Kenji_Mizoguchi:_The_Life_of_a_Film_Director` → SPARQL queries wrong director's birthplace.

#### Mode D — URI Suffix Mismatch (~58 cases)

Linker returns `dbr:X_(film)` but hop-verification expects `dbr:X`. Systematic scan finds ~58 affected questions (original "13" was from a 20-question spot-check). Mostly a measurement artifact — SPARQL queries the `_(film)` URI which has correct properties.

### Fixes ranked by impact/effort

| Priority | Fix | Location | Impact |
|---|---|---|---|
| 1 | Add `'birth'`, `'death'`, `'father'`, `'law'` to `_MENTION_BLOCKLIST` | `entity_linker.py` | ~440 questions cleaner |
| 2 | Jaccard gate on `bif:contains` results (reject Jaccard < 0.3 vs query mention) | `entity_linker.py` | 388 noise entities removed |
| 3 | Block known noise URIs (the two recurrent offenders) | `entity_linker.py` | Immediate, but brittle |
| 4 | Extend `_validate_entity_types` to `bif:contains` results | `entity_linker.py` | Prevents unvalidated noise |

---

## Root Cause 3: Decomposition Comparison Failures (~26 cases)

*Note: Sub-pattern counts below are pattern estimates from 1–2 documented examples each, not full enumerations. Treat as directional.*

### Three sub-patterns

#### Sub-pattern A1 — Country Label Non-Normalization (~9 cases)

Sub-Qs return DBpedia values at wrong granularity: `"New York City"` vs `"Hungary"`, or `"East Germany"` vs `"Germany"`. `FINAL_ANSWER_PROMPT` sees two different strings and says "No" (different country), when the correct answer is "Yes" (same country).

**Example**: "Are directors of Film X and Film Y from the same country?"
- Sub-Q1 returns `"New York City"` (city, not country)
- Sub-Q2 returns `"Hungary"`
- LLM: "No, different countries" → Wrong (both are actually from the same country)

**Why LogicRAG avoids this**: It retrieves full Wikipedia text that says "American film director", not a DBpedia property value.

**Fix**: Add normalization instruction to `FINAL_ANSWER_PROMPT` in `logicrag_prompts.py:35`: "Treat cities/states as their containing country. Treat 'East Germany'/'West Germany' as 'Germany'."

#### Sub-pattern A2 — "How Old" Sub-Questions (~11 cases)

`DECOMPOSE_PROMPT` sometimes generates sub-Qs like "How old is the director of Film X?" instead of "When was [Q0] born?". The TEXT path retrieves a Wikipedia age (e.g., 66) rather than a birth year. Since age changes over time and is lost at death, the comparison is unreliable.

**Example**: Inside Man → Spike Lee (66) vs Alberto Lattuada (62 at death, but actually born 1914 vs 1957 — Lattuada is far older).

**Fix**: Add to `DECOMPOSE_PROMPT` at `logicrag_decomposition.py:62`: "NEVER ask 'How old is X?'. Always ask for the birth date/year instead."

#### Sub-pattern A3 — Spotlight Entity Disambiguation (3+ cases)

"St. Mary's School, Patiala" → Spotlight links to `St._Mary's_Catholic_High_School,_Fujairah` (UAE). Sub-Q returns "UAE" instead of "India".

**Fix**: Year-hint validation post-Spotlight; disambiguation via question context.

---

## Root Cause 4: Decomposition Bridge/General Failures (~22 cases)

*Note: Sub-pattern counts below are pattern estimates, not full enumerations.*

### Three sub-patterns

#### Sub-pattern B1 — SPARQL Retry Confabulation (~10 cases)

When initial SPARQL returns empty results, the retry generates a broader query that returns multiple rows. The LLM picks the **first** result, which is arbitrary.

**Example**: Birthplace of Christian August, Prince of Anhalt-Zerbst → retry returns `[Gommern, Holy_Roman_Empire, Principality_of_Anhalt-Dornburg]` → answer "Gommern" (first). Gold: Dornburg (third).

**Worst case**: "Archduchess Maria Antonia Of Austria (1899–1977)" → Spotlight links to `Marie_Antoinette` (confidence 1.0, the 1755 figure). The `(1899–1977)` year hint is completely ignored.

**Fix**: In `sparql_solver.py:194-215`, pass all rows to LLM and instruct: "Pick the most specific and contextually relevant result, not the first."

#### Sub-pattern B2 — Multi-Variable SPARQL Strips Entity Name (~3 cases)

Sub-Q for "Who is X's husband?" generates `SELECT ?husband ?husbandBirthDate WHERE { ... }` but the `?person` binding is never projected. LLM reads `[{'husbandBirthDate': '1943-11-03'}]` and says "husband unknown". Sub-Q1 then searches for "person born 1943-11-03" → noise results.

**Fix**: In SPARQL generation prompt, instruct: "Always project the primary entity variable (name/label) alongside its properties."

#### Sub-pattern B3 — TEXT Sub-Q Propagation Failure (~12 cases)

`text_solver.solve()` returns a plain string (no entity list). When sub-Q0 resolves via TEXT, `known_entities[0]` stays empty. The entity scoping logic (lines 627–633 in `logicrag_decomposition.py`) gives dependent sub-Q1 an empty entity list. Sub-Q1 then falls back to root-question entities (which may be wrong due to Spotlight errors).

**Corrected mechanism (2026-03-31 validation):** `_resolve_answer_entities()` (defined at line 319) actually runs for ALL dependent sub-Qs at lines 482–489 regardless of SPARQL/TEXT route — it adds entities to `all_entities` (context). The actual gap is that its output is **never propagated back to `known_entities`** (the scoping dict at lines 664–666). So the entity scoping block at 627–633 still gets an empty list for TEXT-resolved dependencies.

**Fix**: At `logicrag_decomposition.py:664-666`, propagate `_resolve_answer_entities()` output into `known_entities[i]` so downstream sub-Qs can scope to TEXT-resolved entities.

---

## Root Cause 5: Evidence Hedging (6 cases)

### Two distinct sub-mechanisms

#### Sub-mechanism A — SPARQL UNION Cross-Contamination (2–3 cases)

For parallel comparison sub-Qs in LogicRAG decomposition, `effective_parents = list(entities)` at `logicrag_decomposition.py:633` passes BOTH comparison entities to the SPARQL solver. The LLM generates a UNION query fetching all dates together with **no entity labels per row**.

**Example** (Thomas Praytor vs Matthew Knox): SPARQL returns `1990-02-28` (Praytor) and `1999-12-22` (Knox, appears twice via `dbo:` + `dbp:`). LLM picks majority value `1999-12-22` as Praytor's birthday → both appear equal → "same age."

**Fix**: Scope `all_entities` to only the sub-Q's named entity before passing to `sparql_solver.solve()`.

#### Sub-mechanism B — `known_facts` Bleeding (2 cases)

`logicrag_decomposition.py:551` passes ALL accumulated `known_facts` to `text_solver.solve()` regardless of dependency. The text solver prompt says "trust KG facts — they are verified structured data." The LLM treats a prior sub-Q's answer (e.g., Manuel Díaz Rodríguez's birthdate) as the current sub-Q's answer (Mona Tyndall), overriding Wikipedia text.

**Fix**: At line 551, filter `known_facts` to include only facts from `sub_q.depends_on` sub-Qs.

---

## Architectural Analysis: Why LogicRAG Avoids All 5 Failure Modes

| Hybrid Component | Failure Mode | LogicRAG Equivalent | Why LogicRAG Succeeds |
|---|---|---|---|
| Entity linking (Spotlight + bif:contains) | Noise entities, wrong entities, cross-contamination | None — corpus search by text | No entity linking = no entity linking errors |
| SPARQL generation + execution | Coverage asymmetry, retry confabulation, projection errors | None — all retrieval from corpus | No SPARQL = no SPARQL bugs |
| Per-sub-Q routing (SPARQL vs TEXT) | Each hop can fail independently, error compounds | None — all sub-Qs go to corpus | No routing = no routing errors |
| Schema projection | Wrong type hints, missing properties | None — no schema | No schema = no schema errors |
| `known_facts` + entity scoping | Cross-contamination between parallel sub-Qs | Accumulated summary (single text blob) | Summary is additive, not entity-scoped |
| `FINAL_ANSWER_PROMPT` with raw KG values | Non-normalized values ("NYC" vs "Hungary") | LLM reads Wikipedia prose ("American director") | Natural language is already normalized |

**The fundamental trade-off**: LogicRAG's corpus-only approach is immune to KG pipeline errors because it has no KG pipeline. But it's less precise when KG data IS correct — which is why hybrid wins on 170 other questions. The hybrid pipeline's additional components (entity linking, SPARQL, routing, schema) each add both capability and failure surface area.

---

## Fix Priority Matrix

Ranked by (questions recovered × feasibility):

| Priority | Fix | Est. Impact | Effort | Location |
|---|---|---|---|---|
| **P0** | Add partial-data instruction to ANSWER_PROMPT | ~19 questions | 1 line | `sparql_evaluation.py:356` |
| **P0** | Scope `all_entities` per sub-Q for comparisons | ~5 questions | 2 lines | `logicrag_decomposition.py:633` |
| **P0** | Filter `known_facts` by `depends_on` | ~5 questions | 1 line | `logicrag_decomposition.py:551` |
| **P1** | Prohibit "How old" sub-Qs in DECOMPOSE_PROMPT | ~7 questions | 1 paragraph | `logicrag_decomposition.py:62` |
| **P1** | Country normalization in FINAL_ANSWER_PROMPT | ~5 questions | 2 sentences | `logicrag_prompts.py:35` |
| **P1** | Blocklist `'birth'`/`'death'`/`'father'`/`'law'` | ~440q cleaner | 1 line | `entity_linker.py` |
| **P1** | Jaccard gate on `bif:contains` (reject < 0.3) | 388q cleaner | 5 lines | `entity_linker.py` |
| **P2** | Pass all SPARQL retry rows + pick-best instruction | ~6 questions | 3 lines | `sparql_solver.py:194` |
| **P2** | Always project entity name in multi-var SPARQL | ~3 questions | 1 sentence | SPARQL prompt |
| **P2** | Propagate `_resolve_answer_entities()` output to `known_entities` | ~5 questions | 5 lines | `logicrag_decomposition.py:664` |
| **P2** | Extend `_validate_entity_types` to `bif:contains` | noise reduction | 10 lines | `entity_linker.py:371` |
| **P3** | Year-hint validation post-Spotlight | ~4 questions | 15 lines | `hybrid_qa.py` |

**Total estimated recovery**: ~45–55 of 74 questions with P0+P1+P2 fixes. The remaining ~20 involve DBpedia data quality errors (wrong dates, missing entities) that no code fix can address.

---

## Discussion Points

### 1. Should we implement these fixes for the thesis?

**Pro**: Recovering ~50 questions would push hybrid from 85.3% → ~89% judge-correct, widening the gap vs LogicRAG's 74.5%. It demonstrates our pipeline is fixable, not fundamentally flawed.

> **Update (2026-03-31):** The ASK verification fix in `routing.py` (comparison questions now use `all()` logic for multi-entity ASK) recovered 12 of the 23 affected bridge_comparison questions. Hybrid 1000q Judge EM improved from 84.1% to 85.3% (+1.2pp). This is a partial implementation of the P0 fixes listed above; remaining fixes (entity scoping, known_facts filtering, ANSWER_PROMPT partial-data instruction) are not yet applied.

**Con**: These are post-hoc fixes derived from analyzing the failure set. Any reported improvements should clearly state they're "post-analysis corrections" to avoid overfitting claims.

### 2. Is the coverage asymmetry finding publishable?

The finding that "16/16 SPARQL comparison inversions are actually coverage gaps, not operator bugs" is a non-obvious result. It reframes the problem from "our comparison logic is buggy" to "KG coverage gaps cause systematic LLM defaults." This is a genuine insight about KG-augmented QA systems.

### 3. Entity linking noise — systemic vs local?

The `bif:contains` noise (38.8% of questions affected) is a systemic design flaw, not a per-question bug. The question is whether to:
- (a) Fix entity linking quality (Jaccard gate, extended validation) — improves the system
- (b) Document it as a limitation of KG-augmented approaches — adds to the thesis argument
- (c) Both — fix what we can, document what we can't

### 4. LogicRAG's corpus advantage — is it "unfair"?

LogicRAG retrieves from the 2WikiMultiHopQA corpus, which contains curated Wikipedia passages already aligned with the questions. Our hybrid pipeline retrieves from live Wikipedia, which is noisier but more general. This corpus advantage partly explains LogicRAG's success on the 74 questions — the answer is often sitting in a pre-curated passage.

**Question**: Should we run hybrid with the same corpus to isolate the KG pipeline effect from the retrieval source effect?

### 5. ~20 questions are DBpedia data quality errors

Several failures trace to DBpedia recording wrong dates or missing entities (e.g., "Moguds Pellams" director born 1946 per DBpedia, gold answer says 1949). These are fundamentally unrecoverable by code fixes. They should be documented as a KG data quality limitation.

---

## Unresolved Questions

1. **Retry confabulation**: Is the SPARQL retry generating broader queries by design, or is the LLM hallucinating when the original query fails? Need to audit retry prompts.

2. **Entity scoping completeness**: The fix for comparison entity cross-contamination (scope `all_entities` per sub-Q) may break bridge entity chains where the parent entity IS needed. Need to test both comparison and bridge paths.

3. **TEXT→SPARQL bridge propagation**: `_resolve_answer_entities()` already runs for all dependent sub-Qs (lines 482–489), but its output feeds `all_entities` without propagating back to `known_entities`. The fix is propagation, not invocation — but does `_resolve_answer_entities()` reliably extract entities from free-text answers (vs structured SPARQL URIs)?

4. **`bif:contains` replacement**: If we add a Jaccard gate, what threshold? 0.3 seems reasonable but needs empirical validation on the 1000q set.

---

## Validation Log (2026-03-31)

Three independent debugger agents verified every code-path claim against source. Corrections applied:

| # | Correction | Severity |
|---|---|---|
| 1 | `execute_sparql()` citation: `sparql_solver.py:82` → `sparql_evaluation.py:276` | Minor (non-material) |
| 2 | Fix 1 null-string: `"cannot determine"` → `"could not determine"` to match `_NULL_ANSWERS` | **Critical** (would cause silent failure) |
| 3 | Mode B count: `421` → `~440` (systematic scan = 436–446) | Minor (ballpark was correct) |
| 4 | Mode D count: `13` → `~58` (original from 20-question spot-check, 4.5x undercount) | **Significant** |
| 5 | RC4 B3 mechanism: `_resolve_answer_entities()` runs for ALL paths, not SPARQL-only; actual gap is `known_entities` propagation | **Significant** (fix unchanged, mechanism description corrected) |
| 6 | Unresolved Q3: corrected same misattribution as #5 | Minor |
| 7 | RC3/RC4 sub-pattern counts flagged as pattern estimates (~), not enumerations | Transparency |

**Verification scores**: RC1 9/10 verified, RC2 5/8 verified (2 counts corrected, 1 location), RC3 code 3/3 verified (counts unverified), RC4 3/5 verified (1 significant mechanism correction), RC5 3/3 fully verified.

**Net assessment**: All 5 root cause mechanisms are architecturally sound. The recovery estimate (~45–55 questions) should be treated as directional, not precise — underlying cluster counts are estimates without complete question-ID enumeration.

Validation reports: `plans/reports/debugger-260331-0847-rc*-claim-validation.md`

---

## Source Reports

Detailed investigation reports with full code traces and line numbers:

1. [SPARQL Comparison Inversion + Hedging](../plans/reports/investigation-260330-sparql-comparison-inversion.md) — 16 comparison failures + 6 hedging cases
2. [Entity Linking Failures](../plans/reports/investigation-260330-entity-linking-failures.md) — `bif:contains` noise injection, type validation gaps
3. [Decomposition Failures](../plans/reports/investigation-260330-decomposition-failures.md) — 26 comparison + 22 bridge/general failures
4. [Original Head-to-Head Analysis](../plans/reports/factcheck-260330-logicrag-head-to-head-analysis.md) — Surface-level 5-category classification
5. [LogicRAG Architecture Deep Dive](../plans/reports/Explore-260329-LogicRAG-Pipeline.md) — Architectural comparison
