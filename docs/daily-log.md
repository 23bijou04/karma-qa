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

**Data:** 12,576 questions total, 1,000-question eval subset (346 answerable).

**Baselines:** SPARQL-only, GRASP (external), LinearRAG, Hybrid (ours).

**Timeline:** 14 working days (2026-02-12 to 2026-03-17), 30 commits.

---

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
