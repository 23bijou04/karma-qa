# Daily Log

Progress tracking for the DBpedia Hybrid QA project, updated at end of each working day.

## Project Structure Overview

**Core pipeline** (5,156 lines across 17 Python files at root):
- `hybrid_qa.py` (343L) — Main orchestrator: SPARQL + LogicRAG + text fallback
- `graphrag_wiki.py` (847L) — Text-only pipeline: Spotlight -> KG -> Wikipedia -> LLM
- `routing.py` (498L) — Three-stage routing: deterministic rules + SPARQL ASK + LLM fallback
- `logicrag_decomposition.py` (635L) — Multi-hop sub-question decomposition
- `sparql_solver.py` (172L) — SPARQL generation + execution with retry
- `text_solver.py` (215L) — Wikipedia retrieval + RAG context assembly
- `schema_projection.py` (232L) — Question-aware 4-layer schema projection
- `question_analysis.py` (281L) — Question type classification
- `entity_discovery.py` (197L) — Type-guided entity discovery
- `llm_provider.py` (102L) — Unified LLM factory (Groq, GitHub Models)

**Supporting modules:**
- `dbpedia/` — KG integration (SPARQL client, entity linker, schema discovery, cache)
- `rag/` — Semantic retrieval (chunker, FAISS vector store, cross-encoder reranker)
- `evaluation/` — Official EM/F1/containment metrics
- `scripts/` — Batch eval, LLM judge, analysis, comparison tools
- `ablation_study_v3/` — 5-level ablation (L0-L4)
- `baselines/LinearRAG/` — LinearRAG baseline (self-contained)
- `tests/` — Unit tests for routing, schema, decomposition, etc.

**Data:** 12,576 questions total, 1,000-question eval subset (318 KG-answerable via hop verification, date-tolerant; evaluations ran on original 346-question set).

**Baselines:** SPARQL-only, GRASP (external), LinearRAG, Hybrid (ours).

**Timeline:** 14 working days (2026-02-12 to 2026-03-17), 30 commits.

---

## 2026-03-25

### Entity Linking Stress Test Implementation & Bug Fixes

- **[testing]**: Created comprehensive 106-case entity linking stress test suite across 5 test files:
  - `tests/test_mention_extraction_stress.py` (47 tests covering noun chunks, possessives, NER, title extraction, dedup)
  - `tests/test_type_system_stress.py` (17 tests for type hinting and validation)
  - `tests/test_sparql_escaping_stress.py` (15 tests for URI/string escaping injection surface)
  - `tests/test_unlinked_comparison_stress.py` (13 tests for unlinked entity detection and comparison patterns)
  - `tests/test_fallback_cache_spacy_stress.py` (16 tests for SPARQL fallback, cache behavior, spaCy NER limits)
- **[testing]**: Extended `tests/conftest.py` with `@pytest.mark.stress` marker and shared fixtures
- **[testing]**: Created `tests/test_data/stress_test_cases.py` with 106 parametrized test cases as source of truth
- **[testing]**: Full suite: 733 passed, 4 skipped, 3 xfailed, 0 failures

### Bug Fixes Delivered

**P0 Severity (2 bugs):**
1. **#26 — Connector chain truncation** in title-after-keyword extraction: `"Lord of the Rings"` → only `"Lord"` extracted. Root cause: lookahead breaks when next word is lowercase connector. Fixed by checking if current word is connector before breaking loop. Affects all multi-word titles with lowercase connectors (Wizard of Oz, War of the Worlds, Bridge on the River Kwai, Lord of the Flies).
2. **#50 — Missing `_TYPE_SUFFIX_MAP` entries**: `dbo:Person`, `dbo:MusicalArtist`, `dbo:City` type hints had no suffix mapping → type validation silently skipped. Fixed by adding entries mapping types to their URI suffixes ('Person', 'musician', 'city').

**P1 Severity (6 bugs):**
3. **#80 — Apostrophe URI escaping**: Apostrophe `'` not percent-encoded in `escape_sparql_uri` → potential SPARQL syntax errors for entities like "Aren't You Kind of Glad We Did?". Fixed by adding `'` → `%27` to escaper.
4. **#81 — Question mark URI escaping**: `?` not percent-encoded → looks like SPARQL variable. Fixed by adding `?` → `%3F` to escaper.
5. **#42/#105 — Curly quote regex blindness**: Both `_extract_mentions` quoted-entity regex and comparison patterns only matched ASCII `"`. Curly quotes `\u201C`/`\u201D` (common in formatted text) invisible. Fixed by extending regex to include `\u201C|\u201D|\u2018|\u2019` (left/right double/single curly quotes).
6. **#104 — Unicode NFC/NFD normalization**: `"Beyoncé"` (NFC) vs `"Beyoncé"` (NFD) treated as different strings in Jaccard similarity and dedup. Fixed by normalizing both mention and surface form to NFC before comparison.
7. **#2-5 — `_STRIP_WORDS` destroying entity names**: Strip logic was too aggressive on special cases ("The Who", "It", "The Band", "A Clockwork Orange"). Fixed by checking stripped result length and preserving original if stripping too destructive.

### Deferred Bugs (Separate PR)

Six additional P1/P2 issues identified but deferred for follow-up:
- **#9 (P1)**: `bif:contains` for pure-stopword entities ("A", "The One") — needs guard
- **#25 (P1)**: `_SPARQL_FALLBACK` on all-lowercase questions — needs fallback logic
- **#58 (P1)**: `obj_phrase` cleaning strips title keywords (year, determiners) too aggressively
- **#69 (P1)**: Entity `"It"` (film) ambiguous in `bif:contains` search
- **#72 (P1)**: Entity `"A"` (film) filtered out by length guard
- **#95 (P1)**: Short common-word entities ("Her", "Us", "Up") invisible across extraction paths

### Documentation

- Updated `docs/entity-linking-stress-test-catalog.md` with corrected case verifications and severity mapping
- All 8 bugs now have verified test cases confirming the root cause and demonstrating the fix
- Test coverage spans 10 functional areas: mention extraction (noun chunks, possessives, NER, title extraction, dedup, quoted entities), type system (hinting, validation, URI probes), SPARQL security (escaping, injection), unlinked detection (Jaccard, substring matching, comparison patterns), fallback paths (capitalized phrases, caching, spaCy limitations)

## 2026-03-18

- **[docs]**: Factchecked LogicRAG vs Our Retrieval comparison report — found 2 major errors (LogicRAG's retrieval is iterative LLM-driven, not direct concatenation; uses 3-round retrieval, not single path)
- **[docs]**: Verified all 9 claims about our system are correct in the comparison report
- **[analysis]**: Recommended reframing comparison around 3 axes: corpus source, reasoning architecture, KG augmentation
- **[analysis]**: Identified need for controlled ablation with benchmark passages and LLM backbone parity (LogicRAG uses GPT-class vs our Llama 3.3 70B)

## 2026-03-17

- **[hardening]**: Hardened answer validation, routing, decomposition, and schema queries across 18 files
- **[routing]**: Fixed edge cases in bridge detection and SPARQL ASK verification
- **[decomposition]**: Improved LogicRAG sub-question resolution robustness
- **[schema]**: Tightened schema discovery queries to reduce noise
- **[llm]**: Added error handling for malformed LLM responses in provider
- **[eval]**: Updated answer source analysis with corrected classification logic
- **[infra]**: Set up daily progress logging (`/daily-log` command)
