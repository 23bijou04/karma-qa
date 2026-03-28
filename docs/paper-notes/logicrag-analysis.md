# LogicRAG Analysis

**Sources:** `plans/reports/Explore-260327-0919-logicrag-baseline.md`, `plans/reports/Explore-260327-0919-logicrag-readiness.md`, `plans/reports/researcher-260327-0919-logicrag-methodology.md`, `plans/reports/researcher-260327-1001-logicrag-paper-analysis.md`, `plans/reports/debug-260326-0000-logicrag-failure-analysis.md`, `plans/reports/logic-validation-260326-1040-hybrid-logicrag-analysis.md`

---

## CRITICAL: Naming Collision

Our `logicrag_decomposition.py` is NOT a reproduction of the LogicRAG paper (AAAI 2026, arXiv:2508.06105). They share terminology but are architecturally different:

| Aspect | LogicRAG paper | Our logicrag_decomposition.py |
|--------|---------------|-------------------------------|
| Knowledge source | Text passages only (Wikipedia corpora) | DBpedia SPARQL + Wikipedia text |
| Sub-Q resolution | Dense vector retrieval | SPARQL or text per sub-Q (routed) |
| Reasoning backend | LLM over retrieved passages | SPARQL queries + RAG + LLM |
| Graph type | Dynamically constructed retrieval DAG | Fixed KG (DBpedia) + entity profiles |
| Pre-built graphs | Explicitly NOT used (core thesis) | Uses pre-built DBpedia KG |

**Any comparison to LogicRAG paper must acknowledge this fundamental difference.**

---

## LogicRAG Paper (AAAI 2026) — Reference Baseline

**Paper:** "You Don't Need Pre-built Graphs for RAG: Retrieval Augmented Generation with Adaptive Reasoning Structures"
**ArXiv:** 2508.06105

### Methodology
- 3-stage pipeline: Query Decomposition → DAG Dependency Modeling → Topological Sort linearization
- Sub-question resolution via dense retrieval (all-MiniLM-L6-v2, top-k=3)
- DAG construction: LLM infers dependency edges between sub-questions
- Context pruning: removes redundant subproblems; LLM-based summarization

### LLM Configuration
- Model: GPT-4o-mini (all experiments, all baselines)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Temperature: not specified (likely 0 or default)
- Prompt templates: not disclosed (reproducibility risk)

### Results on 2WikiMultiHopQA (1000q validation subset)

| Metric | LogicRAG | vs HippoRAG2 (next best) |
|--------|----------|--------------------------|
| String-based accuracy | 64.7% | +14.7 pp |
| LLM-based accuracy | 62.5% | +12.5 pp |

### Evaluation Metrics
- String-based: lowercase normalization + gold answer substring match
- LLM-based: GPT-4o-mini judges semantic equivalence
- No EM/F1 reported

---

## Our LogicRAG Implementation Results (100q pilot)

**Traced run:** 100/1000 questions, full execution traces
**Infrastructure:** `logicrag_decomposition.py` + `sparql_solver.py` + `text_solver.py`

| Metric | Value |
|--------|-------|
| EM | 62% |
| F1 | 69.8% |
| Questions answered via logicrag_decomposed path | 47/100 |
| Wrong answers (of decomposed) | 24/47 = 48.9% error rate |
| Remaining 900q infrastructure | Ready (auto-resume) |

### Scenario breakdown (60-question scenario eval, complete)
- All 60 questions evaluated; per-scenario results in scenario failure reports

---

## LogicRAG Failure Analysis (100q debug)

**Primary failure cause: RAG `page_title` bug**
- Location: `text_solver.py:262`
- Bug: `page_title` variable not properly scoped; wrong Wikipedia page retrieved for sub-questions
- Impact: 19/24 wrong answers (79%) traced to this single bug
- Remaining 5 failures: genuine knowledge gaps or entity linking issues

**Secondary failures:**
- Entity pollution: parent entities passed to all sub-questions (see entity-linking-analysis.md)
- RAG volume bias: no per-entity chunk cap; dominant entity drowns out target entity

### Statistical Validity Note (from logic-validation report)
- Observed correlation: more sub-questions → lower accuracy (round-count degradation)
- This is correlation, NOT causation — harder questions both need more rounds AND are harder
- Do not report round-count degradation as a causal finding without controlling for question difficulty

---

## Our System vs LogicRAG Paper (Fair Comparison Framework)

### What makes comparison fair
- Same dataset: 2WikiMultiHopQA 1000q validation subset ✓
- Same LLM: **NOT SAME** -- LogicRAG paper uses GPT-4o-mini; our system uses llama-3.3-70b-versatile (Groq). Our LogicRAG reproduction also uses llama-3.3-70b for controlled comparison. ✗
- Same embedding model: all-MiniLM-L6-v2 ✓
- Same evaluation: string-based + LLM-based accuracy ✓

### What makes comparison unfair / must be disclosed
- LogicRAG: text-only (no KG); Our system: hybrid KG + text
- LogicRAG: evaluated on full Wikipedia corpora; Our system: DBpedia KG subset
- LogicRAG uses complete Wikipedia passages; we use Wikipedia API fetch per entity
- Our SPARQL baseline uses 7-rule prompt; hybrid uses 8-rule prompt (see sparql-pipeline-analysis.md)

### Hybrid advantages over LogicRAG's approach
1. SPARQL for structured facts (exact values, dates, counts)
2. Schema-aware routing (entity types guide retrieval)
3. Answerability verification via ASK queries
4. Mixed-source synthesis (KG + text)

### Hybrid risks vs LogicRAG's approach
1. SPARQL brittleness: KG topology gaps → wrong routing
2. Property coverage only 36.8% (LogicRAG: 100% via text fallback)
3. More failure modes (rules, ASK, LLM routing, schema projection)
4. RAG page_title bug affects 79% of wrong decomposed answers

---

## Key Numbers for Paper

- LogicRAG paper: 64.7% string / 62.5% LLM on 2WikiMQA 1000q
- Our logicrag_decomposed pilot (100q): 62% EM / 69.8% F1
- 19/24 wrong decomposed answers caused by single fixable bug (page_title)
- 47/100 questions answered via decomposition path; error rate 48.9% (pre-bugfix)
- Round-count correlation is confounded by question difficulty — not a causal claim
