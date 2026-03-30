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

### Hybrid System — 100q Pilot (current-current, 2026-03-30)
| Metric | Value |
|--------|-------|
| LLM judge accuracy | 88% |
| Exact match (EM) | 63% |
| F1 | 73.3% |

### Hybrid vs LogicRAG — Full 1000q Comparison

| Metric | Hybrid | LogicRAG decomposed |
|--------|--------|---------------------|
| LLM judge accuracy | 84.1% (841/1000) | 74.5% (745/1000) |
| EM | 68.1% | 61.1% |
| F1 | 75.4% | 67.5% |

**Delta:** +9.6 pp judge, +7.0 pp EM, +7.9 pp F1

### By Question Type (100q pilot, current-current 2026-03-30)

| Type | n | Hybrid judge | LogicRAG judge | Delta |
|------|---|-------------|----------------|-------|
| Comparison | 24 | 95.8% | 87.5% | +8.3 pp |
| Bridge comparison | 20 | 90.0% | 75.0% | +15.0 pp |
| Inference | 13 | 100.0% | 92.3% | +7.7 pp |
| Compositional | 43 | 79.1% | 58.1% | +20.9 pp |

Hybrid leads on all types. Prior .bak_v1 data showed LogicRAG winning inference (100% vs 80%) — re-judging corrected this.

### KG-Answerable Segment (n=32, 100q subset, hop_verified)

| System | Judge accuracy |
|--------|---------------|
| Hybrid | 93.8% (30/32) |
| LogicRAG decomposed | 65.6% (21/32) |
| Delta | +28.1 pp |

**Key insight:** On KG-answerable questions (where DBpedia has the answer), hybrid system substantially outperforms text-only decomposition approach. Delta is even larger than previously reported (+28pp vs +14pp) due to corrected answerability flag and re-judging.

---

## Hyperparameter Catalog (Verified, 93.9% correct)

### RAG / Retrieval
| Parameter | Value | Source |
|-----------|-------|--------|
| Chunk size | 512 characters | `rag/chunker.py` |
| Chunk overlap | 64 tokens | `rag/chunker.py` |
| Bi-encoder model | `all-MiniLM-L6-v2` | `rag/vector_store.py` |
| Cross-encoder model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `rag/reranker.py` |
| Retrieval top_k | 30 | `rag/vector_store.py` |
| Reranker top_k | 15 | `rag/reranker.py` |
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
- Hybrid LLM judge accuracy: 84.1% (1000q), EM: 68.1%, F1: 75.4%
- LogicRAG LLM judge accuracy: 74.5% (1000q), EM: 61.1%, F1: 67.5%
- Delta (Hybrid vs LogicRAG): +9.6 pp judge, +7.0 pp EM, +7.9 pp F1
- KG-answerable segment (100q pilot): Hybrid 97.2% vs LogicRAG 83.3% (+13.9 pp)
- Bridge comparison (100q pilot): Hybrid 94.7% vs LogicRAG 73.7% (+21.0 pp)
