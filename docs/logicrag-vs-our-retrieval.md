# LogicRAG Baseline vs Our System: Retrieval Comparison

**Date:** 2026-03-17

---

## Benchmark Corpus Origin

2WikiMultiHopQA (Ho et al., COLING 2020) builds on HotpotQA's **October 2017 Wikipedia dump**. Questions are generated from **98,284 Wikidata entities across 29 relations**. The text corpus consists of Wikipedia **intro/lead paragraphs only** — not full articles.

### 1000q Eval Subset Corpus

1,000 questions × 10 context paragraphs = 10,000 entries. Deduplicated by title: **6,120 unique pages** (6,119 unique texts — one duplicate text across two titles).

| Role | Count | Notes |
|---|---|---|
| Supporting evidence pages | 2,339 | Referenced in `supporting_facts` for at least one question |
| Pure distractor pages | 3,781 | Never appear in any question's `supporting_facts` |
| Dual-role pages | 29 | Supporting in one question, distractor in another |
| **Total unique** | **6,120** | |

Passage size distribution (all 6,119 unique texts):

| Bucket | Count | % |
|---|---|---|
| < 100 chars | 833 | 13.6% |
| 100–250 chars | 2,176 | 35.6% |
| 250–500 chars | 1,660 | 27.1% |
| 500–1,000 chars | 892 | 14.6% |
| > 1,000 chars | 559 | 9.1% |

**Median: 254 chars, 2 sentences.** For comparison, a typical Wikipedia lead paragraph is 500–4,000 chars; a full article is 10,000–100,000+ chars. These are intro-paragraph snippets, not Wikipedia pages.

### Benchmark Structure (verified)

- Every question has exactly **10 context passages**
- 78.1% of questions: 2 gold + 8 distractors, 2 supporting facts
- 21.9% of questions (`bridge_comparison`): 4 gold + 6 distractors, 4 supporting facts
- ~85% of answers are directly extractable from supporting fact sentences; ~15% require inference (comparison → "yes"/"no")

---

## Retrieval Task Comparison

### What LogicRAG Indexes

LogicRAG pre-embeds the **6,119 benchmark-provided snippets** into a single FAISS index at startup (`baselines/LogicRAG/src/models/base_rag.py:42`). The retrieval task: rank 254-char median passages using `all-MiniLM-L6-v2`, retrieve top-k. The gold answer is **guaranteed** to exist in exactly 2 of the 6,119 passages by construction.

### What Our System Indexes

Our system fetches **full Wikipedia pages live** via the Wikipedia API (up to 50,000 chars each), chunks them into 512-char section-aware pieces (`rag/chunker.py`), builds a **per-question FAISS index**, retrieves top-20 with seed entity boost, then reranks with a cross-encoder (`ms-marco-MiniLM-L-6-v2`, top-8). No benchmark context data is ever accessed at inference time.

### Side-by-Side

| Dimension | LogicRAG Baseline | Our System |
|---|---|---|
| **Corpus** | 6,119 fixed benchmark snippets | Live Wikipedia pages, per-question |
| **Passage size** | Median 254 chars (~2 sentences) | 512-char chunks from full articles |
| **Index lifetime** | Built once at startup | Built fresh per question |
| **Embedding model** | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` (same) |
| **Reranker** | None | Cross-encoder `ms-marco-MiniLM-L-6-v2` |
| **Answer in corpus?** | Guaranteed by construction | Not guaranteed — pages may differ from 2017 dump, entity may not resolve |
| **Entity resolution** | Pre-resolved benchmark titles | DBpedia Spotlight + SPARQL type validation |
| **Multi-hop strategy** | LLM decompose → retrieve from same fixed corpus | LLM decompose → route sub-Qs to SPARQL or text → fetch new pages per hop |
| **Search space** | 6,119 homogeneous snippets | Hundreds of heterogeneous chunks (infoboxes, headers, prose, lists) |
| **Context assembly** | Direct concatenation | Reranked chunks within 12K char budget |

### Key Code Paths

| | LogicRAG Baseline | Our System |
|---|---|---|
| Corpus | `baselines/LogicRAG/dataset/2wikimultihopqa_corpus.json` | N/A (no pre-built corpus) |
| Embedding | `baselines/LogicRAG/config/config.py` | `rag/vector_store.py:15` |
| Retrieval | `baselines/LogicRAG/src/models/base_rag.py:132` | `rag/vector_store.py` → `RAGIndex.retrieve()` |
| Reranking | N/A | `rag/reranker.py` |
| Wikipedia fetch | N/A | `graphrag_wiki.py:583` → `retrieve_and_chunk_wikipedia()` |
| Chunking | N/A | `rag/chunker.py:24` (512-char, section-aware) |
| Sub-Q routing | Single retrieval path | `logicrag_decomposition.py` → SPARQL or `text_solver.solve()` |

---

## Case Study: "When did Lothair II's mother die?"

| | LogicRAG | Our System |
|---|---|---|
| **Hop 1** | Retrieve "Lothair II" passage (208 chars, 3 sentences) from 6,119. Sentence 1: *"He was the second son of Emperor Lothair I and Ermengarde of Tours."* | Fetch full Lothair II Wikipedia page live → chunk → retrieve relevant chunks |
| **Hop 2** | Retrieve "Ermengarde of Tours" passage (106-char lead sentence: *"d. 20 March 851"*) from same 6,119 | Fetch full Ermengarde of Tours Wikipedia page live → chunk → retrieve |
| **Difficulty** | Find 2 short passages among 6,119 where the answer is stated verbatim | Resolve "Lothair II" via Spotlight, fetch live page, extract mother's name, resolve "Ermengarde of Tours", fetch that page, find death date |

---

## Bottom Line

LogicRAG performs **closed-corpus passage ranking**: find 2 gold snippets among 6,119 short, benchmark-curated passages where the answer is guaranteed to exist. Our system performs **open-corpus live retrieval**: resolve entities via DBpedia, fetch full Wikipedia articles at inference time, chunk and rerank them, with no guarantee the answer appears. Same bi-encoder, fundamentally different retrieval task.
