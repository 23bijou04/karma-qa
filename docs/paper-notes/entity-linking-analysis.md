# Entity Linking Analysis

**Sources:** `plans/reports/260320-entity-linking-failure-analysis.md`, `plans/reports/entity-linking-investigation.md`, `plans/reports/parent-entities-pollution-investigation.md`

---

## Coverage Statistics (1000q subset)

| Metric | Value | Notes |
|--------|-------|-------|
| Entity coverage | 93.6% (936/1000) | Verified live, confirmed |
| True entity linking failures | 6.4% (64/1000) | Entity URI absent in DBpedia |
| Generic/variant URIs (wrong entity) | 1.1% (11/1000) | Spotlight returns wrong disambiguation |
| Property gap failures | 54.7% of all failures | Dominant failure mode |
| Entity linking failures (of all failures) | 30.2% | Secondary failure mode |

---

## Failure Mode Breakdown (260320 analysis)

| Failure Mode | Share | Description |
|-------------|-------|-------------|
| Property gap | 54.7% | Entity found, but required property absent from DBpedia |
| Entity linking failure | 30.2% | Entity not found or wrong URI returned |
| Routing error | ~11% | Entity found, property present, but routed to wrong path |
| Other | ~4% | Annotation errors, transient failures |

---

## Entity Linking Mechanism

- **Primary:** DBpedia Spotlight API (`min_confidence=0.5`, `https://api.dbpedia-spotlight.org/en/annotate`)
- **Fallback:** SPARQL label lookup
- **Post-hoc validation:** Type hints via `TYPE_HINTS` and `RELATION_TYPE_HINTS` dictionaries
- **Limitation:** Spotlight returns a single URI per surface form; no disambiguation ranking exposed

---

## Thomas Jefferson / Ken Burns Pollution Bug

**Case:** Question about Ken Burns documentary → Spotlight links "Ken Burns" correctly, but "Thomas Jefferson" (subject of the documentary) also gets linked as a parent entity.

**Problem flow:**
1. Entity discovery links both Ken Burns + Thomas Jefferson as question entities
2. Both URIs passed as parent entities to ALL sub-questions
3. Sub-question about "who directed this film" receives Thomas Jefferson URI in context
4. SPARQL query attempts to find director of Thomas Jefferson (not the film)
5. Wrong answer or no answer returned

**Root cause:** `entity_discovery.py` returns all question entities; `logicrag_decomposition.py` passes full parent list to every sub-question without scoping.

**Fix needed:** Scope parent entities to the sub-question they are relevant to (only pass entities mentioned/needed in that sub-question).

---

## Type Validation Failures

**Issue:** Post-hoc entity type validation in `entity_linker.py` fails due to noisy `obj_phrase` extraction.

- `obj_phrase` may include stop words or partial matches
- `RELATION_TYPE_HINTS` requires clean surface form to fire correctly
- Result: type validation silently passes invalid entity URIs

**Impact:** ~11 of 1000 questions receive wrong disambiguation (generic variant URIs).

---

## Parent Entity Pollution (Sub-question Scoping)

**Location:** `logicrag_decomposition.py` — `_resolve_subquestion()` call

**Mechanism:**
- `_try_logicrag_decomposition()` decomposes question into N sub-questions
- Each sub-question call passes `entities=parent_entities` (full set from original question)
- Sub-question SPARQL and text queries include all parent entity URIs as context

**Impact observed:**
- Scenario D (BOTH_MISSING): 2/7 failures attributed to entity pollution
- Scenario F (ENT_NOT_FOUND+PROP): contributes to fixable failures
- Ken Burns case: Thomas Jefferson entity passed to sub-question about "director of documentary"

**Fix:** Build per-sub-question entity list from sub-question text only, not from full parent entity set.

---

## Key Numbers for Paper

- 93.6% entity coverage (DBpedia Spotlight + SPARQL fallback on 1000q)
- 54 questions with truly missing entities (entity URI absent from DBpedia)
- 11 questions with wrong disambiguation (generic variants)
- Property gap is 1.8x more common than entity linking failure as root cause of system failure
