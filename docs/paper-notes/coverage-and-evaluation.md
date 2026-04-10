# Coverage and Evaluation

**Sources:** Consolidated from investigation sessions (2026-03-25 through 2026-03-27). Original session reports no longer available; facts verified against code and data files directly.

---

## KG Coverage Statistics (1000q, ground truth)

| Metric | Value | Method | Status |
|--------|-------|--------|--------|
| Entity coverage | 93.6% (936/1000) | Live DBpedia lookup | Confirmed |
| Property coverage (original) | 38.5% | Predicate existence check | Understated |
| Property coverage (expanded) | 41.6% | +expanded predicate set | Intermediate |
| Property coverage (hop-verified) | **36.8% (368/1000)** | All predicates exist per hop | **Ground truth** |
| Answerable (original) | 34.6% | Based on 38.5% property | Understated |
| Answerable (expanded) | 37.5% | Based on 41.6% property | Intermediate |
| Answerable (strict value match) | 11.9% (119/1000) | Exact value match | Low (date format + disambiguation mismatch) |

**Ground truth file:** `data/eval_1000q/hop_verification_results.json`
**Script:** `scripts/verify_hop_coverage.py`

### Entity Coverage Breakdown
- 936/1000 entities found in DBpedia
- 53 truly missing (entity URI absent)
- 11 wrong disambiguation (generic variant URIs returned by Spotlight)

### Methodology Progression Note
Original 38.5% → expanded 41.6% → hop-verified 36.8%. The hop-verified number is lower because it requires the full multi-hop chain to be traversable, not just individual predicate existence.

---

## System Evaluation Results

### Hybrid System — First 100 of 1000q (regression reference)
| Metric | Hybrid | LogicRAG |
|--------|--------|----------|
| LLM judge accuracy | 89% | 76% |
| EM | 67.0% | 61.0% |
| F1 | 78.6% | 69.5% |

**Source:** first 100 questions (by hybrid eval order) from `hybrid_eval_1000_judged.jsonl` / `logicrag_1000q_traced_judged.jsonl`.

### Hybrid vs LogicRAG — Full 1000q Comparison

| Metric | Hybrid | LogicRAG decomposed |
|--------|--------|---------------------|
| LLM judge accuracy | 89.4% (894/1000) | 74.5% (745/1000) |
| EM | 72.6% | 61.1% |
| F1 | 80.1% | 67.5% |

**Delta:** +14.9 pp judge, +11.5 pp EM, +12.6 pp F1

### By Question Type — Full 1000q

| Type | n | Hybrid judge | LogicRAG judge | Delta |
|------|---|-------------|----------------|-------|
| Comparison | 244 | 95.9% (234/244) | 92.6% | +3.3 pp |
| Bridge comparison | 235 | 91.9% (216/235) | 72.8% | +19.1 pp |
| Inference | 108 | 95.4% (103/108) | 86.1% | +9.3 pp |
| Compositional | 413 | 82.6% (341/413) | 61.7% | +20.9 pp |

Hybrid leads on all 4 types after failure recovery. Bridge comparison and compositional show the largest gaps (+19.1 pp, +20.9 pp).

### By Question Type — First 100 (regression reference)

| Type | n | Hybrid judge | LogicRAG judge | Delta |
|------|---|-------------|----------------|-------|
| Comparison | 20 | 95.0% | 90.0% | +5.0 pp |
| Bridge comparison | 19 | 94.7% | 84.2% | +10.5 pp |
| Inference | 15 | 80.0% | 73.3% | +6.7 pp |
| Compositional | 46 | 87.0% | 67.4% | +19.6 pp |

**100→1000q regression note:** After ASK fix (2026-03-31) and failure recovery (2026-04-01), Hybrid advantage on bridge_comparison widened substantially (+10.5→+19.1 pp at 1000q scale). Inference rises (+6.7→+9.3 pp). Compositional gap widens (+19.6→+20.9 pp). Comparison recovers from -1.2 pp (old 1000q) to +3.3 pp — failure recovery resolved previously failing comparison questions.

### KG-Answerable Segment (hop_verified, all_properties_exist=True)

**Full 1000q (n=368):**

| System | Judge accuracy |
|--------|---------------|
| Hybrid | 92.9% (342/368) |
| LogicRAG decomposed | 73.9% (272/368) |
| Delta | +19.0 pp |

**First 100 (n=38):**

| System | Judge accuracy |
|--------|---------------|
| Hybrid | 97.4% (37/38) |
| LogicRAG decomposed | 76.3% (29/38) |
| Delta | +21.1 pp |

**Key insight:** On KG-answerable questions, hybrid consistently outperforms text-only decomposition. Delta at 1000q scale is +19.0 pp (after failure recovery, up from +14.1 pp pre-recovery). Source: `hop_verification_results.json` (all_properties_exist=True).

---

## Hyperparameter Catalog (Verified against code 2026-03-30)

### RAG / Retrieval
| Parameter | Value | Source |
|-----------|-------|--------|
| Chunk size | 512 characters | `rag/chunker.py` |
| Chunk overlap | 64 characters | `rag/chunker.py` |
| Bi-encoder model | `all-MiniLM-L6-v2` | `rag/vector_store.py` |
| Cross-encoder model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `rag/reranker.py` |
| Retrieval top_k | 20 | `text_solver.py:259` (overrides `vector_store.py` default of 30) |
| Reranker top_k | 8 | `text_solver.py:271` (overrides `reranker.py` default of 15) |
| Max context chars | 12,000 | `text_solver.py` |
| Seed entity boost | 1.2 | `rag/vector_store.py` |

### Entity Linking
| Parameter | Value | Source |
|-----------|-------|--------|
| Spotlight min_confidence | 0.5 | `dbpedia/entity_linker.py` |
| Spotlight endpoint | `https://api.dbpedia-spotlight.org/en/annotate` | `dbpedia/entity_linker.py` |

### LLM
| Parameter | Value | Source |
|-----------|-------|--------|
| Default model | `llama-3.3-70b-versatile` | `hybrid_qa.py` |
| Temperature | 0 | `llm_provider.py` |
| Retry initial wait | 10s (doubles per attempt) | `llm_provider.py` |

### DBpedia
| Parameter | Value | Source |
|-----------|-------|--------|
| Cached class schemas | 7 (Person, Film, Company, Place, MusicalArtist, Book, MusicalWork) | `dbpedia/schema_discovery.py` |
| Wikipedia API delay | 0.5s between calls | `dbpedia/wiki_resolver.py` |

---

## RAG Volume Bias (Thomas Jefferson / Ken Burns Case)

**Problem:** Single flat FAISS index with no per-entity chunk cap at the retrieval stage. Thomas Jefferson (large Wikipedia article) produces far more chunks than Ken Burns → dominates retrieval results. Note: `text_solver.py` has a `max_per_entity=4_000` char cap in the assembled context, but this does not apply at the FAISS retrieval/ranking stage.

**3 identified defects:**
1. No per-entity chunk cap at retrieval stage: high-profile entities crowd out answer entity (assembly-level 4,000-char cap exists but does not fix ranking bias)
2. Seed boost too weak: `seed_boost=1.2` insufficient to prioritize answer entity chunks
3. Routing gap: some questions that need text retrieval fall through to SPARQL, then fail

**Impact:** Affects multi-entity questions where one entity has much larger Wikipedia presence than the other.

**Mitigation (not yet implemented):** Cap chunks per entity at `ceil(top_k / n_entities)`; increase seed_boost or use hard filtering.

---

## Architecture Claims Factcheck (48% error rate in technical docs)

**From factcheck-260326-1033-architecture-claims.md:**
- 48% error rate found in technical documentation (v1.0)
- Critical fabrications in SPARQL and LogicRAG sections
- Fixed in docs v1.1 (36 errors) and v1.2 (29 additional errors)
- Key fabrications: non-existent functions, wrong line numbers, invented parameter names

**Lesson:** Do not trust technical documentation written without code verification. All architecture claims should be verified against actual source files.

---

## Key Numbers for Paper

- Entity coverage: 93.6% (ground truth, verified)
- Property/hop coverage: 36.8% (368/1000, ground truth, hop-verified predicate existence)
- Strict value match: 11.9% (119/1000, lower bound due to date format + disambiguation mismatch)
- Hybrid LLM judge accuracy: 89.4% (894/1000), EM: 72.6%, F1: 80.1%, Containment: 82.8%
- LogicRAG LLM judge accuracy: 74.5% (1000q), EM: 61.1%, F1: 67.5%
- Delta (Hybrid vs LogicRAG): +14.9 pp judge, +11.5 pp EM, +12.6 pp F1
- Compositional (1000q): Hybrid 82.6% vs LogicRAG 61.7% (+20.9 pp) — largest gap
- Bridge comparison (1000q): Hybrid 91.9% vs LogicRAG 72.8% (+19.1 pp) — widened by ASK fix + recovery
- Comparison (1000q): Hybrid 95.9% vs LogicRAG 92.6% (+3.3 pp) — Hybrid leads all 4 types after recovery
- Inference (1000q): Hybrid 95.4% vs LogicRAG 86.1% (+9.3 pp)
- Answerable (1000q): Hybrid 93.0% (346/372), Non-answerable: 87.3% (548/628)
- KG-answerable (1000q, n=368): Hybrid 92.9% vs LogicRAG 73.9% (+19.0 pp)
- KG-answerable (first 100, n=38): Hybrid 97.4% vs LogicRAG 76.3% (+21.1 pp)
- **Within Hybrid pipeline**, method success rates (1000q): logicrag_decomposed 87.9% (616/701), sparql 92.9% (274/295), text 100% (4/4)

### Baseline Results (hop-verified 368q, updated 2026-03-30)

- Schema-Guided SPARQL — Hybrid's direct SPARQL component evaluated standalone (n=368): EM 68.5%, F1 81.3%, Judge EM 91.6%, Containment 86.1%
- GRASP (n=368): EM 32.1%, F1 47.3%, Judge EM 79.1%, Containment 85.6%
- Head-to-head (n=368): SPARQL EM 68.5% vs GRASP 32.1% (+36.4 pp), Judge 91.6% vs 79.1% (+12.5 pp)
- Judge model: gemma3:12b for both systems (consistent, no cross-judge bias)
