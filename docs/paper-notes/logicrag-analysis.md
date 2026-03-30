# LogicRAG Analysis

**Sources:** Consolidated from investigation sessions (2026-03-26 through 2026-03-27). Original session reports no longer available; facts verified against code and data files directly.

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
| Questions with decomposition (rounds>=1) | 72/100 |
| Multi-round decomposition (rounds>=2) | 47/100 |
| Error rate (all decomposed, EM=0) | 28/72 = 38.9% |
| Error rate (multi-round only) | 22/47 = 46.8% |
| Remaining 900q infrastructure | Ready (auto-resume) |

### Scenario breakdown (60-question scenario eval, complete)
- All 60 questions evaluated; per-scenario results in scenario failure reports

---

## Full 1000q Results (current)

| Metric | Value |
|--------|-------|
| LLM judge accuracy | 74.5% (745/1000) |
| EM | 61.1% |
| F1 | 67.5% |
| Decomposed (rounds>=1) | 514/1000 |
| Decomposed error rate (EM) | 51.4% |

---

## LogicRAG Failure Analysis (100q debug)

**Primary failure cause: RAG `page_title` bug (FIXED)**
- Location: `text_solver.py:262`
- Bug: `page_title` variable not properly scoped; wrong Wikipedia page retrieved for sub-questions
- Impact: 19/24 wrong answers (79%) traced to this single bug
- Remaining 5 failures: genuine knowledge gaps or entity linking issues
- **Note:** This bug was identified and fixed. Line 262 is now metadata logging. The statistics below reflect the pre-fix state and are retained for historical context.

**Secondary failures:**
- Entity pollution: parent entities passed to all sub-questions (FIXED — scoping implemented at logicrag_decomposition.py:623-643) (see entity-linking-analysis.md)
- RAG volume bias: per-entity cap exists in assembled context (4,000 chars) but retrieval ranking still has volume bias

### Statistical Validity Note (from logic-validation report)
- Observed correlation: more sub-questions → lower accuracy (round-count degradation)
- This is correlation, NOT causation — harder questions both need more rounds AND are harder
- Do not report round-count degradation as a causal finding without controlling for question difficulty

---

## Our System vs LogicRAG Paper (Comparison Framework)

### What makes comparison fair
- Same dataset: 2WikiMultiHopQA 1000q validation subset ✓
- Same LLM: **NO** — LogicRAG paper uses GPT-4o-mini; our system uses llama-3.3-70b-versatile via Groq ✗
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
4. RAG page_title bug affects 79% of wrong decomposed answers (pre-fix; now resolved)

---

## Key Numbers for Paper

- LogicRAG paper: 64.7% string / 62.5% LLM on 2WikiMQA 1000q
- Our system full 1000q: 74.5% LLM judge / 61.1% EM / 67.5% F1
- Our logicrag_decomposed pilot (100q): 62% EM / 69.8% F1
- 100q pilot: 72/100 decomposed (rounds>=1), 47/100 multi-round (rounds>=2)
- 100q pilot error rates: 38.9% (all decomposed), 46.8% (multi-round only) — pre-bugfix
- 19/24 wrong decomposed answers caused by single fixable bug (page_title, now fixed)
- 1000q: 514/1000 decomposed; 51.4% decomposed error rate (EM)
- Round-count correlation is confounded by question difficulty — not a causal claim
