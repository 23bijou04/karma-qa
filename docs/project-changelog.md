# Project Changelog

**DBpedia Hybrid QA System** — Comprehensive record of features, fixes, and improvements.

---

## [2026-03-25] Entity Linking Stress Test Suite & Bug Fixes

### Added
- **Stress test suite (106 cases, 5 test files)**:
  - `tests/test_mention_extraction_stress.py` — 47 tests covering noun chunk stripping, possessive handling, NER processing, title-after-keyword extraction, deduplication, and quoted entities
  - `tests/test_type_system_stress.py` — 17 tests for type hinting priority, missing type mappings, and entity type validation
  - `tests/test_sparql_escaping_stress.py` — 15 tests for URI/string escaping attack surface (apostrophes, question marks, injection attempts)
  - `tests/test_unlinked_comparison_stress.py` — 13 tests for Jaccard similarity, substring matching, and comparison question entity extraction
  - `tests/test_fallback_cache_spacy_stress.py` — 16 tests for SPARQL fallback capitalized-phrase extraction, cache behavior, and spaCy NER fundamental limitations
- **Test data centralization**: `tests/test_data/stress_test_cases.py` — 106 parametrized cases as single source of truth for all stress tests
- **Test infrastructure**: Extended `tests/conftest.py` with `@pytest.mark.stress` marker and shared fixtures for DRY test setup
- **Test summary**: 733 passed, 4 skipped, 3 xfailed, 0 failures

### Fixed
- **[P0] #26 — Connector chain truncation in title extraction**: Multi-word titles with lowercase connectors ("Lord of the Rings", "Wizard of Oz") extracted only first word. Root cause: lookahead breaks when next word is lowercase connector. Fixed by checking if current word is connector before advancing loop. Verified in `test_mention_extraction_stress.py::test_connector_chain` (case #26).

- **[P0] #50 — Missing type suffix mappings**: Type hints for `dbo:Person`, `dbo:MusicalArtist`, `dbo:City` had no URI suffix entries in `_TYPE_SUFFIX_MAP`, causing type validation to silently skip correction. Fixed by adding mappings: `'dbo:Person': 'person'`, `'dbo:MusicalArtist': 'musician'`, `'dbo:City': 'city'`. Verified in `test_type_system_stress.py::test_missing_type_suffix_map` (case #50).

- **[P1] #80 — Apostrophe not percent-encoded in SPARQL URIs**: Entities with apostrophes (e.g., "Aren't You Kind of Glad We Did?") not escaped in `escape_sparql_uri`, causing potential SPARQL syntax errors. Fixed by adding `'` → `%27` conversion. Verified in `test_sparql_escaping_stress.py::test_apostrophe_in_uri` (case #80).

- **[P1] #81 — Question mark not percent-encoded in SPARQL URIs**: Entities with question marks (e.g., "Did?") not escaped, causing `?` to be interpreted as SPARQL variable. Fixed by adding `?` → `%3F` conversion. Verified in `test_sparql_escaping_stress.py::test_question_mark_in_uri` (case #81).

- **[P1] #42/#105 — Curly quote extraction failure**: Quoted entity extraction regex `r'"([^"]+)"'` only matched ASCII double quotes, invisible to curly quotes `\u201C`/`\u201D` (common in formatted text). Fixed by extending regex in both `_extract_mentions` (case #105) and type extraction (case #42) to include `\u201C|\u201D|\u2018|\u2019`. Verified in `test_mention_extraction_stress.py::test_quoted_curly_quotes` (cases #42, #105).

- **[P1] #104 — Unicode normalization blindness in unlinked detection**: Mentions and surface forms using different Unicode normalizations (NFC vs NFD) treated as different strings in Jaccard similarity and deduplication. Fixed by normalizing both to NFC before comparison. Verified in `test_mention_extraction_stress.py::test_unicode_normalization` (case #104).

- **[P1] #2-5 — Strip words destroying entity names**: Front-stripping of `_STRIP_WORDS` too aggressive on entities like "The Who", "It", "The Band", "A Clockwork Orange", resulting in either empty strings or different entities. Fixed by checking stripped result is meaningful (preserves multi-word structure) before applying. Verified in `test_mention_extraction_stress.py::test_strip_words_guard` (cases #2-5).

### Deferred (Tracked for Follow-up PR)
- **[P1] #9**: `bif:contains` search for pure-stopword entities ("A", "The One") returns no results — needs stopword guard in `build_bif_expression`
- **[P1] #25**: Connector at end of title in title-after-keyword ("film Lord of") → only "Lord" extracted when lookahead fails
- **[P1] #58**: `obj_phrase` cleaning strips title keywords too aggressively ("wrote The Year 2000" → only "Year" searched)
- **[P1] #69**: Entity "A" (film) filtered by length guard — needs semantic awareness
- **[P1] #72**: Entity "It" (film) ambiguous in `bif:contains` — needs type-aware disambiguation
- **[P1] #95**: Short entity names ("Her", "Us", "Up", "A") invisible across extraction paths — structural issue spanning multiple components

### Documentation
- Updated `docs/entity-linking-stress-test-catalog.md` with verified case analysis and severity classification
- Created `docs/daily-log.md` entry for 2026-03-25 documenting full scope of testing and fixes
- Each bug fix includes test file, test function name, and case ID for traceability

---

## [2026-03-18] LogicRAG Comparison Report Verification

### Fixed
- **Documentation**: Corrected 2 major errors in `logicrag-vs-our-retrieval.md`:
  - LogicRAG's retrieval is **iterative LLM-driven**, not direct concatenation of all relevant text
  - Uses **3-round multi-step retrieval**, not single-path corpus traversal
- **Verification**: Confirmed all 9 technical claims about our hybrid system remain correct in comparison

### Analysis
- Identified axis reframing opportunity: comparison should focus on corpus source (single vs multi-document), reasoning architecture (decomposition strategy), and KG augmentation (entity-only vs full schema)
- Recommended controlled ablation with benchmark passage parity and LLM backbone matching (LogicRAG uses GPT-class vs our Llama 3.3 70B)

---

## [2026-03-17] System Hardening Across Core Modules

### Fixed
- **Routing**: Edge cases in bridge detection and SPARQL ASK verification (18 files touched)
- **Decomposition**: Improved LogicRAG sub-question resolution robustness
- **Schema**: Tightened schema discovery queries to reduce noise
- **LLM provider**: Added error handling for malformed LLM responses
- **Evaluation**: Updated answer source analysis with corrected classification logic

---

## [2026-02-12 to 2026-03-17] Initial Development (30 commits)

### Project Initialization
- **Core pipeline**: 5 major components (hybrid_qa, graphrag_wiki, routing, decomposition, solvers)
- **KG integration**: DBpedia SPARQL client, entity linker, schema discovery
- **Semantic retrieval**: FAISS-based RAG with cross-encoder reranking
- **Evaluation infrastructure**: Batch eval, comparison tables, statistical testing
- **Baselines**: SPARQL-only, GRASP (external), LinearRAG (self-contained)
- **Dataset**: 2WikiMultiHopQA benchmark (12,576 questions, 1,000-question eval subset)

### Key Features
- Three-stage routing: deterministic rules + SPARQL ASK + LLM fallback
- LogicRAG-style multi-hop sub-question decomposition
- Schema projection for question-aware entity filtering
- Wikipedia text + RAG fallback with semantic chunking
- LLM provider factory supporting Groq and GitHub Models

---

## Version Numbering

This project uses date-based versioning: `[YYYY-MM-DD]` marks significant release points or documentation snapshots. Breaking changes and major bug fixes are recorded in this changelog.

## Relationship to Test Suite

- Each bug fix entry includes reference to the stress test that verifies it (file, test function, case ID)
- Test suite (`tests/`) is the executable specification for entity linking behavior
- Cases in `tests/test_data/stress_test_cases.py` are authoritative record of known issues and expected behavior
