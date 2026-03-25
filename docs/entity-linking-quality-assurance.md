# Entity Linking Quality Assurance Strategy

**Purpose**: Document the multi-layered QA approach for entity linking pipeline robustness.

**Scope**: `dbpedia/entity_linker.py`, `dbpedia/sparql_utils.py`, and related modules in entity discovery chain.

---

## Quality Assurance Layers

### Layer 1: Unit Tests (Functional Coverage)

**Files**: `tests/test_entity_linking_edge_cases.py` (main suite)

**Scope**: Core entity linking functions in isolation.

- Entity extraction from question text
- Spotlight API interaction and fallback
- Entity type validation and correction
- SPARQL search and retrieval

**Coverage**: ~45 core test cases covering happy paths and documented edge cases.

---

### Layer 2: Stress Tests (Edge Case Catalog)

**Files**: 5 stress test modules + centralized case database

**Scope**: Exhaustive edge-case testing based on critical analysis of code paths.

- **106 test cases** across 10 functional areas
- **Severity-based categorization** (P0: critical, P1: high, P2: medium, P3: low)
- **Single source of truth** (`tests/test_data/stress_test_cases.py`) for case metadata

**Execution**:
```bash
pytest -m stress
```

**Key Statistics**:
- P0 bugs discovered: 2 (both fixed)
- P1 bugs discovered: 6 (all fixed)
- P2/P3 cases: 98 (documented, tracked for future work)

---

### Layer 3: Integration Tests (Pipeline Behavior)

**Files**: `tests/test_routing_2hop.py`, `tests/test_logicrag_decomposition.py`

**Scope**: Entity linking as part of full question-answering pipeline.

- Multi-hop question routing decisions
- LogicRAG sub-question decomposition with entity linking
- Entity + relation extraction in context of schema projection

**Coverage**: ~150 integration test cases across routing and decomposition logic.

---

### Layer 4: Benchmark Evaluation (Real-World Performance)

**Scripts**: `scripts/hybrid_batch_eval.py`, evaluation metrics in `evaluation/metrics.py`

**Scope**: Entity linking impact on 2WikiMultiHopQA benchmark (1,000 question eval set).

- 1,000 question evaluation set (346 answerable)
- Entity coverage analysis: 93.6% of questions have resolvable entities
- Property coverage analysis: 38.5% of required relations present in DBpedia
- Answerability classification by KG coverage scenario

---

## Bug Discovery & Triage Process

### Discovery Phase

1. **Code Analysis**: Read entity_linker.py and related modules line-by-line
2. **Threat Modeling**: Identify potential edge cases per functional area
3. **Case Generation**: Create test cases capturing threats
4. **Verification**: Run cases against actual implementation
5. **Categorization**: Severity rating (P0-P3) based on impact

### Triage & Fix Decision

| Severity | Discovery | Fix Decision | Example |
|----------|-----------|--------------|---------|
| **P0** | Blocks benchmark questions | Fix immediately | #26 (connector chain), #50 (type maps) |
| **P1** | Real entities, measurable impact | Fix in current sprint | #80-81 (escaping), #42/#105 (quotes) |
| **P2** | Uncommon but valid cases | Track for next PR | #87-88 (entity name separators) |
| **P3** | Theoretical/unlikely | Document only | #1 (pure stopwords) |

### Current Status

**8 bugs fixed** (2 P0, 6 P1):
- All fixes verified by parametrized stress tests
- All fixes committed in `entity-linking-stress-test` branch
- All fixes documented in `docs/project-changelog.md`

**6 bugs deferred** (all P1, tracked separately):
- #9: Pure stopword entity search guard
- #25: Fallback on lowercase questions
- #58: Title keyword preservation in phrase cleaning
- #69/#72: Short entity disambiguation
- #95: Short entity visibility across extraction paths

---

## Test Data Management

### Stress Test Case Database

**Location**: `tests/test_data/stress_test_cases.py`

**Structure**: 106 cases as Python dictionaries with metadata:

```python
STRESS_CASES = [
    {
        'case_id': 26,
        'category': 'mention-extraction',
        'subcategory': 'title-extraction',
        'input': 'book Lord of the Rings',
        'expected_mention': 'Lord of the Rings',
        'actual_mention': 'Lord',  # Before fix
        'issue': 'Connector chain truncation',
        'severity': 'P0',
        'status': 'FIXED',
        'test_file': 'test_mention_extraction_stress.py',
        'test_function': 'test_connector_chain',
    },
    # ... 105 more cases
]
```

**Properties**:
- **case_id**: Unique identifier (1-106)
- **category**: One of 10 functional areas
- **input**: Question text or entity string
- **expected_mention**: What should be extracted
- **actual_mention**: What was extracted before fix (if applicable)
- **severity**: P0/P1/P2/P3
- **status**: FIXED / DEFERRED / DOCUMENTED
- **test_file**: Which test module verifies this case
- **test_function**: Which test function in that module

---

## Verification Protocol

### Before Deployment

Each bug fix must have:

1. ✅ **Failing test case** demonstrating the bug
2. ✅ **Code fix** in `dbpedia/entity_linker.py` or `dbpedia/sparql_utils.py`
3. ✅ **Passing test** confirming fix works
4. ✅ **Metadata entry** in stress_test_cases.py with test file + function reference
5. ✅ **Changelog entry** in `docs/project-changelog.md` with case ID and fix description

### CI/CD Integration

Stress tests run in automated pipeline:

```bash
pytest -m stress --junitxml=stress-test-results.xml
```

Failures block PR merge until resolved (P0 immediately, P1 within sprint).

---

## Coverage by Functional Area

### 1. Mention Extraction (47 cases)

**Core responsibility**: Extract entity mentions from raw question text.

**Sub-components**:
- Quoted entities (`"Hamlet"`)
- NER-detected entities (spaCy)
- Title-after-keyword extraction (`film Inception`)
- Noun chunk processing with stopword stripping
- Possessive handling (Shakespeare's → Shakespeare)
- Deduplication and filtering

**Top issues fixed**:
- #26 (P0): Connector chain truncation in titles
- #2-5 (P1): Strip words destroying entity names
- #42/#105 (P1): Curly quote invisibility

**Remaining defects** (#9, #25, others in deferred list)

---

### 2. Type System (17 cases)

**Core responsibility**: Infer entity type from question keywords and validate against DBpedia.

**Sub-components**:
- Type hint extraction (`film Inception` → dbo:Film)
- Type validation via entity label + DBpedia lookup
- URI probe disambiguation (`Lincoln` person vs film)
- `bif:contains` fallback with type constraint

**Top issues fixed**:
- #50 (P0): Missing type suffix mappings (Person, MusicalArtist, City)

**Key limitation**: Type hints can false-positive on non-film uses of "directed" (e.g., "directed surgery").

---

### 3. SPARQL Escaping (15 cases)

**Core responsibility**: Safely escape entity names and URIs for SPARQL queries.

**Sub-components**:
- String literal escaping (quotes, backslashes, newlines)
- URI percent-encoding (special characters)
- SPARQL variable injection prevention

**Top issues fixed**:
- #80 (P1): Apostrophe not percent-encoded (`'` → `%27`)
- #81 (P1): Question mark not percent-encoded (`?` → `%3F`)

**Security note**: Injection surface is small; quotes and URIs are primary risk vectors.

---

### 4. Unlinked Detection (13 cases)

**Core responsibility**: Identify entities not found by Spotlight and trigger fallback search.

**Sub-components**:
- Jaccard similarity threshold (0.5)
- Substring containment matching
- Comparison question entity extraction (`X or Y`)

**Known issues**: None fixed in stress test (core logic solid).

**Edge case limitation**: Multi-word entities with separators ("Truth or Dare") can confuse comparison regex.

---

### 5. Fallback & Cache (16 cases)

**Core responsibility**: Last-resort entity extraction and performance optimization.

**Sub-components**:
- SPARQL fallback capitalized-phrase extraction
- Cache key matching
- Cache invalidation on empty results
- spaCy NER limitations mitigation

**Known issues**:
- Cases #89-96: spaCy NER is upstream limitation, cannot fix in our code
- Mitigated by Spotlight + keyword extraction fallbacks

---

## Metrics & Health Checks

### Test Suite Health

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Stress test pass rate | 100% | 100% (108/108) | ✅ Green |
| Unit test pass rate | 100% | 100% (733/733) | ✅ Green |
| Integration test pass rate | 100% | 100% | ✅ Green |
| P0 defects open | 0 | 0 | ✅ Green |
| P1 defects open (current sprint) | <3 | 0 (6 deferred to next PR) | ✅ Green |

### Benchmark Performance

| Metric | Value | Status |
|--------|-------|--------|
| Entity coverage (1000q) | 93.6% | Good |
| Property coverage (1000q) | 38.5% | Acceptable (schema gap) |
| Answerability (1000q) | 34.6% (346/1000) | Expected (schema-limited) |

---

## Known Limitations

### Code-Level Limitations (Cannot Fix)
- **spaCy NER**: `en_core_web_sm` misses common-noun and location-named titles
  - Mitigation: Spotlight API + keyword extraction fallbacks
  - Impact: Cases #89-96 documented but unfixable

### Design Limitations (Require Redesign)
- **Short entity names**: Entities like "It", "A", "Her" ambiguous in `bif:contains` search
  - Current: Length filtering + type constraints
  - Future: Semantic disambiguation or entity ranking model

### Data Limitations (Benchmark-Related)
- **Wikidata accuracy**: 2WikiMultiHopQA built from Dec 2019 Wikidata snapshot
  - Some relations were incorrect at snapshot time (e.g., Margaretha van Mechelen spouse)
  - Relations later corrected in Wikidata but benchmark unchanged
  - Impact: Some questions unanswerable as stated (documented in `benchmark-errors-2wikimultihopqa.md`)

---

## Recommendations for Future Work

### High Priority (Next PR)
1. **#9**: Guard `bif:contains` against pure-stopword searches
2. **#25**: Fallback capitalized-phrase extraction for lowercase questions
3. **#58**: Preserve title keywords in `obj_phrase` cleaning

### Medium Priority (Future Sprints)
4. **#69/#72**: Semantic disambiguation for short entities
5. **#95**: Cross-path visibility for short entity names
6. **#87-88**: Handle entity names containing separators in comparison regex

### Investigation Needed
7. Evaluate if spaCy model upgrade (newer version or custom training) would help NER coverage
8. Analyze impact of entity ranking models on `bif:contains` selection
9. Consider knowledge graph completion techniques for property coverage gap

---

## Documentation References

| Document | Purpose | Location |
|-----------|---------|----------|
| **Stress Test Catalog** | Detailed case analysis (106 cases, root causes, impact) | `docs/entity-linking-stress-test-catalog.md` |
| **Stress Test Guide** | How to run tests, test structure, adding new cases | `docs/entity-linking-stress-test-guide.md` |
| **Project Changelog** | All bug fixes with test file + case ID references | `docs/project-changelog.md` |
| **Daily Log** | Timeline of work (2026-03-25 entry) | `docs/daily-log.md` |
| **Benchmark Errors** | Confirmed data errors in 2WikiMultiHopQA | `docs/benchmark-errors-2wikimultihopqa.md` |

---

## Glossary

- **P0/P1/P2/P3**: Severity levels (critical to low)
- **Stress test**: Parametrized test case from centralized case database
- **Integration test**: Full pipeline test (routing, decomposition)
- **Coverage scenario**: KG answerability classification (entity coverage, property coverage, relation coverage)
- **Unlinked entity**: Mention not resolved by Spotlight, triggers fallback search
- **bif:contains**: DBpedia full-text search predicate for entity label matching
- **Type hint**: Expected entity type inferred from question keywords

