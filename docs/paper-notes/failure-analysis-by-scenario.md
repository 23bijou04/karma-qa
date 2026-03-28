# Failure Analysis by Scenario

**Source:** `plans/reports/scenario_{a-f}_failure_analysis.md`
**Scenario classification:** A=KG_COMPLETE, B=HOP1_MISSING, C=HOP2_MISSING, D=BOTH_HOPS_MISSING, E=ENTITY_NOT_FOUND, F=ENTITY_NOT_FOUND+PROP

---

## Scenario A — KG_COMPLETE (n=10)

**Definition:** Both hops fully present in DBpedia KG.

**Performance:** 90% raw EM / 100% LLM judge EM

**Failures (1):**
- Compound DBpedia literal: `"New York City, New York"` vs gold `"New York City"` — containment=1.0, EM=0.0
- Root cause: DBpedia stores city+state as a compound literal; gold is city-only

**Key insight:** Near-perfect on KG-complete questions. Single failure is metric artifact (containment passes, EM fails due to over-specific KG value).

---

## Scenario B — HOP1_MISSING (n=10)

**Definition:** First hop property absent from DBpedia; second hop present.

**Performance:** 80% LLM judge EM

**Failures (2):**
1. Transient DBpedia + multi-valued property truncation: SPARQL returned first listed value, not the intended one (data ordering issue)
2. `direct_date_comparison` rule fires on UNION not intersection: partial SPARQL result returned (only one entity's dates, not both)

**Key insight:** Comparison routing fires incorrectly when property matches one entity but not both. Systemic bug in comparison guard logic.

---

## Scenario C — HOP2_MISSING (n=10)

**Definition:** First hop present; second hop property absent from DBpedia.

**Performance:** 80% LLM judge EM

**Failures (5):**
- 3/5 systemic: dbp-only bridge triggers false `ask_verified` routing
  - `dbp:` properties exist in the KG but ASK query uses `dbo:` namespace → query succeeds spuriously → system routes to SPARQL → SPARQL execution fails silently
- 1/5 benchmark annotation error (gold answer incorrect in dataset)
- 1/5 genuine knowledge gap

**Root cause (3/5):** DBpedia prefix asymmetry — `dbo:` vs `dbp:` causes false positives in SPARQL ASK verification.

**Key insight:** ASK verification does not distinguish namespace prefixes. The most fixable defect in Scenario C.

---

## Scenario D — BOTH_HOPS_MISSING (n=10)

**Definition:** Both hop properties absent from DBpedia; entity URIs present.

**Performance:** 10% raw EM / 30% LLM judge EM

**Failures (7):**
- 2 pure knowledge gaps: information not in Wikipedia or DBpedia
- 3 wrong text retrieval: RAG retrieved irrelevant passages; entity pollution (parent entities contaminated sub-question context)
- 2 system bugs: entity pollution + RAG volume bias

**Example (Case 4):** "Margaretha van Mechelen" — Wikipedia describes her as "mistress" (consistent since 2008); question asks about romantic relationship type. Correct answer present but answer extraction confused by relationship category ambiguity.

**Key insight:** Text retrieval is the primary bottleneck here. Without KG support, system relies entirely on Wikipedia RAG quality.

---

## Scenario E — ENTITY_NOT_FOUND (n=10)

**Definition:** Entity URI not found in DBpedia; entity may be identifiable by name.

**Performance:** 40% raw EM / 90% LLM judge EM — **+50 percentage point gap**

**Failures (5 EM-only, 1 genuine):**
- 5 metric-only failures: LLM judge accepts answer as correct; EM rejects due to format differences (e.g., "United States" vs "USA")
- 1 genuine failure: geographic granularity error (city vs country)

**Key insight:** The 50pp gap between EM and judge is metric pathology, not system failure. Over-specific or paraphrased correct answers fail EM but pass semantic judge. This scenario demonstrates strongest case for using LLM-as-judge.

---

## Scenario F — ENTITY_NOT_FOUND + PROP_MISSING (n=10)

**Definition:** Entity URI absent AND property missing; hardest scenario.

**Performance:** 20% raw EM / 30% LLM judge EM

**Failures (7):**
- F4, F5, F6, F7, F8 (5/7): fixable with better entity linking + text retrieval
- 2/7: hard knowledge gaps (information not in accessible corpora)

**Fixable failure modes:**
- Entity not linked → wrong Wikipedia page retrieved
- Property gap → text retrieval cannot recover structured relation
- Entity pollution contaminating sub-question context

**Key insight:** 5/7 failures are potentially fixable (entity linking improvement + entity scope isolation). 2/7 are irreducible knowledge gaps.

---

## Cross-Scenario Summary

| Scenario | n | Raw EM | Judge EM | Primary Failure Mode |
|----------|---|--------|----------|---------------------|
| A (KG_COMPLETE) | 10 | 90% | 100% | Metric artifact (compound literal) |
| B (HOP1_MISSING) | 10 | — | 80% | Comparison UNION bug |
| C (HOP2_MISSING) | 10 | — | 80% | dbp/dbo prefix asymmetry (3/5) |
| D (BOTH_MISSING) | 10 | 10% | 30% | Text retrieval + entity pollution |
| E (ENTITY_NOT_FOUND) | 10 | 40% | 90% | Metric pathology (+50pp gap) |
| F (ENT_NOT_FOUND+PROP) | 10 | 20% | 30% | Entity linking + knowledge gaps |

**Systemic bugs identified:**
1. Comparison guard bug — `sparql_rows` always empty for comparison queries (kills 7 correct answers across B/C)
2. dbp/dbo prefix asymmetry — false `ask_verified` routing (3/5 Scenario C failures)
3. Entity pollution — parent entities passed to all sub-questions without scoping (D/F failures)
4. RAG volume bias — no per-entity chunk cap; seed_boost=1.2 too weak (D failures)
