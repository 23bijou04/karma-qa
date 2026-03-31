# DBpedia Coverage Verification on the 1000-Question Evaluation Subset

*Analysis Date: March 2026 | DBpedia Endpoint: https://dbpedia.org/sparql*
*Ground truth: `data/eval_1000q/hop_verification_results.json` (script: `scripts/verify_hop_coverage.py`)*

---

## 1. Context

The 2WikiMultiHopQA benchmark provides multi-hop questions where each answer requires traversing a chain of evidence triples: seed entity → hop-1 property → bridge entity → hop-2 property → answer. Our prior coverage analysis on the full dev set (12,576 questions) established entity coverage of 92.5% and per-pair property coverage of 60.1%, yielding a full-dataset answerability rate of 31.1%.

To obtain more rigorous numbers for thesis reporting, we used a 1000-question evaluation subset (from LinearRAG's evaluation set, Zly0523/linear-rag) and conducted three progressively more rigorous verification passes against live DBpedia SPARQL, culminating in a hop-by-hop chain verification that checks every evidence triple independently.

## 2. Methodology Progression

Three verification methodologies were applied, each more rigorous than the last:

### Pass 1: Original Property-Existence Analysis

**Script:** `archive/scripts/coverage_analysis.py`
**Output:** `data/eval_1000q/coverage_analysis_1000.json`

For each question, checks whether (a) all entities exist as DBpedia URIs and (b) all required properties exist as DBpedia predicates on those entities. Uses a hand-curated `PROPERTY_MAPPINGS` dictionary to translate benchmark property names (e.g., "country of origin") to DBpedia predicates (e.g., `dbo:country`).

**Finding:** The `PROPERTY_MAPPINGS` was incomplete --- five benchmark properties ("country of origin", "performer", "place of burial", "inception", "award received") were entirely absent, and four others had narrow predicate lists. The fallback heuristic (camelCase conversion, e.g., "country of origin" → `dbo:countryoforigin`) generated non-existent predicates.

| Metric | Value |
|--------|-------|
| Entity coverage | 93.6% (936/1000) |
| Property coverage | 38.5% (385/1000) |
| Answerable | 34.6% (346/1000) |

### Pass 2: Predicate-Expanded Correction

**Script:** `scripts/correct-property-coverage.py`
**Output:** `data/eval_1000q/coverage_analysis_1000_corrected.json`

Re-checked all 361 (entity, property) pairs previously marked "missing" for the 9 properties with >0% found rate in a prior sampled verification. Used an expanded predicate list (e.g., "country of origin" → `dbo:country`, `dbp:country`, `dbo:nationality`, `dbp:origin`, `dbp:countryOfOrigin`).

**Finding:** 80/361 "missing" pairs actually exist in live DBpedia. The biggest gap was "country of origin" (75% false-negative rate --- the property exists via `dbp:country` but the original mapping missed it entirely).

| Metric | Value | Delta from Pass 1 |
|--------|-------|--------------------|
| Entity coverage | 93.6% (936/1000) | unchanged |
| Property coverage | 41.6% (416/1000) | +31 questions |
| Answerable | 37.5% (375/1000) | +29 questions |

### Pass 3: Hop-by-Hop Chain Verification (Ground Truth)

**Script:** `scripts/verify_hop_coverage.py`
**Output:** `data/eval_1000q/hop_verification_results.json`

The most rigorous methodology. For each question, independently resolves every evidence triple's subject entity against DBpedia (fresh entity resolution, not reusing cached URIs), checks predicate existence, retrieves actual values, and verifies value match against the benchmark's expected object.

Two metrics are computed:

- **Property exists:** All required predicates exist at every hop
- **Strict value match:** All predicates exist AND all value strings match exactly

| Metric | Value |
|--------|-------|
| Entity coverage | 93.6% (936/1000) |
| Property exists (all hops) | **36.8% (368/1000)** |
| Chain-traversable (strict) | 11.9% (119/1000) |

**Why Pass 3 < Pass 2 (36.8% vs 41.6%):** Pass 3 resolves entities independently per evidence triple, catching cases where the entity resolves to the wrong DBpedia URI (e.g., "Thomas Jefferson" → the president, not the film). Pass 2 reused pre-resolved URIs from Pass 1, which masked entity disambiguation errors.

**Why strict is so low (11.9%):** The 249-question gap (368→119) arises from three compounding sources: (1) date format mismatch (dominant) — DBpedia stores `1923-04-09` while the benchmark expects `9 April 1923`; (2) entity disambiguation — DBpedia returns a valid but different entity than the benchmark expects; (3) label normalization — DBpedia returns a URI where the benchmark expects a label string. The strict metric is artificially low and should not be reported as the primary number.

## 3. Entity Coverage: 93.6% (Verified)

### Arithmetic

936/1000 questions have all entities resolved to valid DBpedia URIs. 64 questions have at least one missing entity, corresponding to 65 unique entity names.

### Live Verification

- **Existing entities:** 248/248 sampled entity URIs confirmed live in DBpedia (100% accuracy)
- **Missing entities:** 65/65 confirmed absent via direct URI check. Exhaustive variant search (redirects, title-case, stripped parentheticals, Wikipedia cross-check) found 11 partial matches, but these map to disambiguation pages (e.g., `Geoffrey III, Count of Perche` → generic `dbr:Geoffrey`), not the actual entities. True missing count: **54 after exhaustive search, 65 by direct URI.**

### Missing Entity Profile

The missing entities are predominantly obscure historical figures with non-standard naming conventions: nobility with ordinals ("John II, Duke of Cleves"), parenthetical dates ("Elene (1753–1786) was a Georgia"), titles with special characters ("Empress Dowager Từ Dụ"), and song/album titles with leading quote characters.

**Source:** `plans/reports/verify-260325-1302-entity-coverage.md`, `scripts/verify_entity_coverage.py`

## 4. Property Coverage: Three Perspectives

### 4.1 Per-Property Breakdown (from Pass 3)

Properties with **zero** found rate (confirmed truly missing in DBpedia):

| Property | Missing pairs | Notes |
|----------|:---:|-------|
| director | 86 | Films often lack `dbo:director` for obscure titles |
| father | 103 | DBpedia rarely models parentage for historical figures |
| publication date | 134 | Albums/songs often lack `dbo:releaseDate` |
| date of birth | 56 | Present for well-known figures, absent for obscure ones |
| date of death | 35 | Same pattern as date of birth |
| spouse | 33 | Inconsistent coverage in DBpedia |
| place of birth | 34 | Often present but with different entity resolution |
| place of death | 23 | Similar to place of birth |
| mother | 22 | Rarely modeled |

Properties with **significant false-negative rates** in original analysis (found via expanded predicates):

| Property | Missing (original) | Actually found | Found rate | Root cause |
|----------|:---:|:---:|:---:|-----------|
| country of origin | 52 | 39 | **75%** | Mapped to nonexistent `dbo:countryoforigin`; actual: `dbp:country` |
| performer | 27 | 15 | **55%** | Missed `dbo:artist` (songs use `artist`, not `performer`) |
| place of burial | 13 | 7 | **54%** | Missed `dbp:burialPlace`, `dbo:restingPlace` |
| inception | 11 | 5 | **46%** | Missed `dbo:foundingDate`, `dbp:founded` |
| award received | 7 | 2 | **29%** | Missed `dbo:award` |

**Source:** `data/eval_1000q/property_coverage_verification.json`, `scripts/verify_property_coverage.py`

### 4.2 The Property Mapping Gap

The original `PROPERTY_MAPPINGS` in `archive/scripts/coverage_analysis.py` contains 23 entries. The 2WikiMultiHopQA benchmark uses 27 distinct property names across the 1000-question subset. The 4 unmapped properties + 5 with narrow predicate lists account for the 38.5% → 41.6% correction. However, even with perfect predicate mapping, entity disambiguation errors reduce effective property coverage to 36.8% (Pass 3).

### 4.3 The Value Match Gap

Of the 368 questions where all predicates exist, only 119 have exact-string matching values. The 249-question gap (368→119) comes from:

- **Date format mismatch (dominant):** DBpedia stores `xsd:date` values like `1923-04-09` while the benchmark expects human-readable text like `9 April 1923`. These are informationally identical but produce zero string-match score.
- **Entity disambiguation:** DBpedia returns a valid but different entity than the benchmark expects (e.g., the entity exists but is the wrong "Thomas Jefferson")
- **Label normalization:** DBpedia returns a URI where the benchmark expects a label string, or vice versa
- **Multi-valued properties:** DBpedia has multiple values (e.g., multiple nationalities) and the benchmark expects a specific one

## 5. Answerable Subset: Which Number to Report?

| Metric | Value | What it measures | Use case |
|--------|-------|------------------|----------|
| Original answerable | 34.6% (346) | Predicate exists (narrow map) | Historical (superseded by hop-verified 368) |
| Predicate-corrected | 37.5% (375) | Predicate exists (expanded map) | Superseded |
| **Property-complete (hop)** | **36.8% (368)** | All predicates exist per hop | **Primary thesis number** |
| Strict match | 11.9% (119) | Exact value string match | Artificially low (date format artifact) |

**Recommendation:** Report **36.8% (368/1000, hop-verified predicate existence)** as the primary "KG-answerable" rate. It is the most honest and defensible number: it verifies that every evidence predicate exists at every hop via live SPARQL traversal, without baking in assumptions about value normalization. The gap between 36.8% (predicates exist) and 11.9% (strict match) quantifies date format mismatch, entity disambiguation, and label normalization costs — these are reported as data quality findings, not adjustments to the primary number.

**Note on evaluation subset:** As of 2026-03-30, both SPARQL and GRASP evaluations have been re-run on the full 368-question hop-verified set with the same judge model (gemma3:12b). Earlier evaluations on the 346-question set are superseded.

## 6. Comparison with Full Dev Set

| Metric | Full Dev (12,576q) | 1000q Subset | Notes |
|--------|:---:|:---:|-------|
| Entity coverage | 92.5% | 93.6% | Consistent |
| Property coverage | 60.1% (per-pair) | 36.8% (per-question, hop-verified) | Different granularity: per-pair vs per-question |
| Answerability | 31.1% | 36.8% (hop-verified, predicate existence) | 1000q is the more rigorous measurement |

The 1000q hop-verified rate (36.8%) is moderately higher than the full dev set answerability (31.1%). The full dev set numbers have **not** been hop-verified and likely suffer from the same property mapping gaps identified here; the 31.1% figure uses a narrower predicate mapping and may understate true predicate coverage.

## 7. Key Findings

1. **Entity coverage is solid (93.6%)** and confirmed against live DBpedia with 100% accuracy on sampled URIs. The 65 missing entities are genuinely absent --- obscure historical figures with non-standard naming.

2. **Property coverage depends heavily on methodology.** The original narrow PROPERTY_MAP (38.5%) understates coverage by ~3pp due to missing predicate mappings. But even with perfect mapping (41.6%), fresh entity resolution drops it to 36.8%.

3. **The property mapping gap is a known, fixable limitation** of the analysis pipeline, not of DBpedia itself. The five unmapped properties and four with narrow predicates could be corrected in the analysis code. However, for thesis reporting, the hop-verified numbers are more honest because they also capture entity disambiguation errors that the analysis pipeline masks.

4. **The 36.8% → 11.9% strict-match gap is dominated by date format mismatch** (DBpedia stores `xsd:date` values while the benchmark uses human-readable text), plus entity disambiguation and label normalization. The 249-question gap between predicate-complete and strict-match is a data quality finding, not a reason to adjust the primary number.

5. **36.8% (hop-verified) vs 31.1% (full-dataset):** The 1000q subset is moderately higher, consistent with the expanded predicate mapping applied to the 1000q analysis. The full-dataset figure is based on a narrower predicate set and likely understates true predicate coverage.

---

*Verification artifacts:*
- *Entity verification: `scripts/verify_entity_coverage.py` → `plans/reports/verify-260325-1302-entity-coverage.md`*
- *Property verification: `scripts/verify_property_coverage.py` → `data/eval_1000q/property_coverage_verification.json`*
- *Predicate correction: `scripts/correct-property-coverage.py` → `data/eval_1000q/coverage_analysis_1000_corrected.json`*
- *Hop verification: `scripts/verify_hop_coverage.py` → `data/eval_1000q/hop_verification_results.json`*
- *Combined entity+property: `scripts/verify_entity_property_coverage.py`*
- *Factcheck: `plans/reports/factcheck-260325-1507-coverage-claims.md`*
