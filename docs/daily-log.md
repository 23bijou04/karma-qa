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

## 2026-03-26

- **[metrics]**: Added `normalize_date()` to `evaluation/metrics.py` — converts ISO/DMY/MDY dates to canonical YYYY-MM-DD, closing 5pp gap between property coverage (36.8%) and strict-match answerability (11.9% → 31.8%)
- **[bugfix]**: Fixed RAG chunk unpacking in `text_solver.py` — `RAGIndex.retrieve()` returns `(WikiChunk, float)` tuples, metadata tracking now unpacks correctly
- **[data]**: Committed hop-verified ground truth (`hop_verification_results.json`, 318/1000 KG-answerable) and eval artifacts (corrected coverage, judged hybrid eval, LogicRAG traces)
- **[docs]**: Updated all coverage references across 7 docs from corrected estimates (42.2%/34.6%) to hop-verified numbers (36.8%/31.8%); added `coverage-verification-1000q.md` methodology doc
- **[scripts]**: Added LogicRAG eval infrastructure — `logicrag-instrumented.py` (trace-capturing subclass), `run-logicrag-1000q.py` (batch runner), `rerun_failed_text.py` (selective re-eval of failed text-path questions)
- **[cleanup]**: Removed 19 outdated verification/analysis scripts and docs superseded by hop-verified ground truth

## 2026-03-25

21 commits across 6 work streams. Busiest day on this project.

- **[entity-linking]**: Built 3-phase EL cascade — Phase 1: LLM candidate disambiguation, Phase 2: direct URI construction probe, Phase 3: Wikipedia search + redirect resolution. Unit tests for all phases. Evaluated on 100q sample (`scripts/eval_entity_linking_100q.py`, `scripts/el_eval_report.py`)
- **[entity-linking]**: Fixed URI validation (ucfirst, confidence gate), renamed `ddgs` → `duckduckgo-search` dep
- **[bugfix]**: Fixed 8 entity linker bugs (2×P0, 6×P1): connector chain truncation (#26), missing `_TYPE_SUFFIX_MAP` (#50), apostrophe/question-mark URI escaping (#80/#81), curly quote regex (#42/#105), Unicode NFC normalization (#104), aggressive `_STRIP_WORDS` (#2-5). 6 more P1/P2 deferred
- **[testing]**: Created 106-case stress test suite across 5 files (mention extraction, type system, SPARQL escaping, unlinked detection, fallback/cache). Extended conftest with `@pytest.mark.stress`. Full suite: 733 passed, 0 failures
- **[security]**: Blocked SPARQL `SERVICE` injection, added Spotlight circuit breaker, batch eval timeout
- **[coverage]**: Added live coverage verification scripts, ran against 1000q — results: 93.6% entity, property/answerability numbers fed into hop-verified ground truth pipeline
- **[infra]**: Added full pipeline traceability for 1000q batch eval; removed corpus chunking to match LogicRAG embedding format
- **[cleanup]**: Removed 19 outdated docs/scripts; removed obsolete eval artifacts from `eval_1000q/`
- **[docs]**: Entity linking QA guide, benchmark error analysis, stress test catalog

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
