# Entity Linking Documentation Index

Quick reference for navigating entity linking documentation and stress test resources.

---

## Quick Start

### I want to...

| Goal | Go to |
|------|-------|
| **Run stress tests** | [entity-linking-stress-test-guide.md](entity-linking-stress-test-guide.md) → "Running the Stress Tests" |
| **Understand what was fixed** | [project-changelog.md](project-changelog.md) → "2026-03-25 Entity Linking Stress Test Suite & Bug Fixes" |
| **See all 106 test cases** | [entity-linking-stress-test-catalog.md](entity-linking-stress-test-catalog.md) → Case tables by category |
| **Check QA strategy** | [entity-linking-quality-assurance.md](entity-linking-quality-assurance.md) → "Quality Assurance Layers" |
| **Track daily progress** | [daily-log.md](daily-log.md) → "2026-03-25 Entity Linking Stress Test Implementation" |
| **Add a new test case** | [entity-linking-stress-test-guide.md](entity-linking-stress-test-guide.md) → "Adding New Test Cases" |
| **Find a specific bug** | [project-changelog.md](project-changelog.md) → Search for case ID (e.g., "#26") |

---

## Documentation Files

### Primary References

#### 1. **entity-linking-stress-test-catalog.md** (298 lines)
**Most detailed**

106 test cases organized by functional area with:
- Case ID, input, expected vs actual behavior
- Root cause analysis
- Severity (P0-P3) classification
- Impact assessment
- Summary tables by severity

**When to use**: Need details on a specific case, understanding the scope of issues, research/publication context.

---

#### 2. **entity-linking-stress-test-guide.md** (320 lines)
**Most practical**

How to run, understand, and extend the stress tests:
- Pytest commands (copy-paste ready)
- Test file organization (5 modules, 108 cases)
- Test data structure
- Bug fix reference table
- Adding new test cases (3-step process)
- CI/CD integration

**When to use**: Running tests, setting up CI, adding cases, debugging failures.

---

#### 3. **entity-linking-quality-assurance.md** (340 lines)
**Most strategic**

Multi-layered QA approach and verification protocol:
- 4 QA layers (unit, stress, integration, benchmark)
- Bug discovery & triage process
- Coverage by functional area
- Health metrics and status
- Known limitations
- Future work recommendations

**When to use**: Understanding overall QA approach, making prioritization decisions, planning follow-up work.

---

#### 4. **project-changelog.md** (140 lines)
**Most concise & timely**

Comprehensive record of all changes:
- 2026-03-25: 8 bug fixes (2 P0, 6 P1) with test file references
- 2026-03-18: LogicRAG comparison verification
- 2026-03-17: System hardening
- 2026-02-12 to 03-17: Initial development

**When to use**: Quick overview of what changed, commit message context, timeline.

---

### Supporting References

#### 5. **daily-log.md** (Updated 2026-03-25)
Progress tracking with entry for entity linking work:
- Bug fixes and their P0/P1 severity
- Test suite results (733 passed, 4 skipped)
- Documentation updates

**When to use**: Day-by-day project progress, finding when work was completed.

---

## Bug Fix Quick Reference

### All 8 Bugs Fixed

| Case | Severity | Issue | Test File | Test Function |
|------|----------|-------|-----------|---------------|
| #2-5 | P1 | Strip words guard | mention_extraction_stress | test_strip_words_guard |
| #26 | P0 | Connector chain | mention_extraction_stress | test_connector_chain |
| #42 | P1 | Curly quotes (type) | mention_extraction_stress | test_quoted_curly_quotes |
| #50 | P0 | Type suffix maps | type_system_stress | test_missing_type_suffix_map |
| #80 | P1 | Apostrophe escape | sparql_escaping_stress | test_apostrophe_in_uri |
| #81 | P1 | Question mark escape | sparql_escaping_stress | test_question_mark_in_uri |
| #104 | P1 | Unicode normalization | mention_extraction_stress | test_unicode_normalization |
| #105 | P1 | Curly quotes (extract) | mention_extraction_stress | test_quoted_curly_quotes |

### 6 Deferred Bugs (Next PR)

| Case | Severity | Issue |
|------|----------|-------|
| #9 | P1 | Pure stopword entity search |
| #25 | P1 | Fallback on lowercase questions |
| #58 | P1 | Title keyword preservation |
| #69 | P1 | Short entity "A" disambiguation |
| #72 | P1 | Short entity "It" filtering |
| #95 | P1 | Short entity visibility |

See [entity-linking-stress-test-catalog.md](entity-linking-stress-test-catalog.md) for full case descriptions.

---

## Test Suite Overview

### 5 Test Modules (108 cases)

```
tests/test_mention_extraction_stress.py         47 tests
├─ Noun chunk stripping (5)
├─ Possessive splitting (7)
├─ NER processing (9)
├─ Title extraction (11)
├─ Dedup & filtering (6)
└─ Quoted entities (6)

tests/test_type_system_stress.py                17 tests
├─ Type hinting (2)
├─ Missing type maps (1)
└─ Validation paths (8)

tests/test_sparql_escaping_stress.py            15 tests
├─ String escaping (4)
├─ URI escaping (3)
└─ Edge cases (8)

tests/test_unlinked_comparison_stress.py        13 tests
├─ Jaccard similarity (3)
├─ Comparison patterns (7)
└─ Helpers (3)

tests/test_fallback_cache_spacy_stress.py       16 tests
├─ SPARQL fallback (3)
├─ Cache behavior (3)
└─ spaCy NER limits (10)
```

**Run all**: `pytest -m stress`

---

## Functional Areas Covered

| Area | Test Count | P0s | P1s | Issues |
|------|-----------|-----|-----|--------|
| Mention extraction | 47 | 1 | 5 | Stopword handling, connector chains, curly quotes |
| Type system | 17 | 1 | 0 | Missing type mappings |
| SPARQL escaping | 15 | 0 | 2 | Apostrophe, question mark encoding |
| Unlinked detection | 13 | 0 | 0 | Jaccard/substring logic solid |
| Fallback & cache | 16 | 0 | 0 | spaCy NER limitations (upstream) |
| **TOTAL** | **108** | **2** | **6** | |

---

## Test Execution

### Last Run Results
```
733 passed, 4 skipped, 3 xfailed, 0 failed
```

**Stress tests subset**: 108 passed, 0 failed

### Run Commands

```bash
# All stress tests
pytest -m stress

# Single test file
pytest tests/test_mention_extraction_stress.py -v

# Single test case
pytest tests/test_mention_extraction_stress.py::test_connector_chain -v

# With coverage
pytest -m stress --cov=dbpedia --cov-report=html
```

See [entity-linking-stress-test-guide.md](entity-linking-stress-test-guide.md) for more commands.

---

## Documentation Statistics

| File | Lines | Purpose |
|------|-------|---------|
| entity-linking-stress-test-catalog.md | 298 | Detailed case analysis (106 cases) |
| entity-linking-stress-test-guide.md | 320 | How to run tests + structure |
| entity-linking-quality-assurance.md | 340 | QA strategy + verification protocol |
| project-changelog.md | 140 | All changes with bug fix details |
| daily-log.md | 126 | Progress tracking (updated) |
| **Total** | **1,224** | |

---

## Knowledge Base Organization

### By Question Type

**Q: "How do I run the tests?"**
→ [entity-linking-stress-test-guide.md](entity-linking-stress-test-guide.md) → Running section

**Q: "What bugs were fixed?"**
→ [project-changelog.md](project-changelog.md) → 2026-03-25 section

**Q: "What's case #26 about?"**
→ [entity-linking-stress-test-catalog.md](entity-linking-stress-test-catalog.md) → Section 1.4, row for case #26

**Q: "How is entity linking tested?"**
→ [entity-linking-quality-assurance.md](entity-linking-quality-assurance.md) → Quality Assurance Layers

**Q: "Why isn't bug X fixed?"**
→ [entity-linking-quality-assurance.md](entity-linking-quality-assurance.md) → Known Limitations section

**Q: "How do I add a new test case?"**
→ [entity-linking-stress-test-guide.md](entity-linking-stress-test-guide.md) → Adding New Test Cases section

**Q: "What's the project status?"**
→ [daily-log.md](daily-log.md) → 2026-03-25 entry

---

## Integration with Other Docs

### Related Documentation (Cross-Referenced)

- **benchmark-errors-2wikimultihopqa.md** — Data errors in benchmark (referenced in QA strategy)
- **hyperparameters-and-magic-numbers.md** — Configuration magic numbers (separate from entity linking)
- **kg-coverage-scenarios.md** — Answerability classification (related to entity coverage)

---

## Version & Maintenance

**Last Updated**: 2026-03-25
**Documentation Version**: 1.0 (initial entity linking focus)
**Status**: Complete, ready for reference and CI/CD integration

**Maintenance Schedule**:
- Update daily-log.md on working days
- Update project-changelog.md on release/major fix
- Update stress-test-guide.md when adding new test cases
- Review QA strategy quarterly for trends/recommendations

---

## For More Information

### Tests (Code)
- `tests/test_mention_extraction_stress.py` — Actual test implementations
- `tests/test_data/stress_test_cases.py` — Case definitions (single source of truth)
- `tests/conftest.py` — Test infrastructure (markers, fixtures)

### Code Being Tested
- `dbpedia/entity_linker.py` — Main entity linking module
- `dbpedia/sparql_utils.py` — SPARQL utility functions
- `dbpedia/client.py` — SPARQL client interaction

### Reports
- `plans/reports/docs-manager-260325-1131-entity-linking-stress-test-documentation.md` — Full documentation update report

---

**Navigation**: Use this index to find exactly what you need. Each file is self-contained but cross-referenced for deeper exploration.

