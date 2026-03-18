# Text Retrieval & RAG Edge Case Analysis

**Date:** 2026-03-17
**Scope:** `text_solver.py`, `rag/vector_store.py`, `rag/reranker.py`, `rag/chunker.py`, `rag/cache.py`, `graphrag_wiki.py`
**Category:** Text Retrieval & RAG subsystem

---

## Edge Case #25: RAGIndex created per-call

**Location:** `text_solver.py:181`

**Verdict:** :warning: Partial — by design but wasteful for repeated entities

**Analysis:**

Line 181 creates `rag_index = RAGIndex()` fresh on every `_solve_rag()` call. The FAISS index itself is lightweight (flat index, no training step), so the *index creation* cost is negligible. The actual cost is in `rag_index.index_chunks(chunks)` at line 182, which calls `encoder.encode(texts, ...)` — the bi-encoder inference pass.

However, key mitigating factors:

1. **The bi-encoder model is a module-level singleton** (`vector_store.py:12-28`). The `_encoder` is loaded once and reused. Model loading (~1-2 seconds) only happens on the first call.

2. **The WikiRAGCache (`rag/cache.py`)** caches *chunks* per Wikipedia page title (not embeddings at the FAISS level). So the Wikipedia fetch + chunking step is cached, but the embedding step is NOT cached. `cache.get_embeddings()` and `cache.store_embeddings()` exist (lines 77-89) but are **never called** from `text_solver.py` or `retrieve_and_chunk_wikipedia()`.

3. **Each question typically involves different entities** in the batch eval pipeline, so index reuse across questions would be architecturally complex (the index contents change per question).

**What's missing:** The embedding cache in `WikiRAGCache` (`get_embeddings`/`store_embeddings`) is implemented but unused. If the same Wikipedia pages are fetched for different questions (common for popular entities like "Christopher Nolan"), the chunks are re-embedded every time. For 1000-question batch eval, this is measurable but not critical — bi-encoder inference on ~50-200 chunks takes ~0.1-0.5 seconds per question, dwarfed by LLM API latency.

---

## Edge Case #26: _assemble_from_chunks() budget overflow

**Location:** `text_solver.py:47-63`

**Verdict:** :warning: Partial — unlikely in practice but no safeguard

**Analysis:**

The function at lines 52-54 adds `kg_facts` to the budget unconditionally:
```python
if kg_facts:
    parts.append(kg_facts)
    total += len(kg_facts)
```

If `len(kg_facts) >= max_chars` (12,000), then `total` starts at 12K+ and the loop at line 58 (`if total + len(entry) > max_chars: break`) immediately breaks for every chunk. Result: the context contains ONLY `kg_facts` with zero Wikipedia text.

**How long can kg_facts be in practice?**

In `hybrid_qa.py:244-251`, `kg_facts` is built from `sparql_neighbor` entities:
```python
kg_lines.append(f"- {e.discovery_relation}: {e.label}")
```

Each line is ~30-60 chars (e.g., `"- director: Christopher Nolan"`). The `_sparql_neighbors()` function at `hybrid_qa.py:57` uses `limit_per_entity=10` and typically 2-4 seed entities, yielding at most ~20-40 neighbor entities. So `kg_facts` is typically 1-2 KB at most — well under 12K.

In `logicrag_decomposition.py:492-493`, the `kg_facts_block` is built from `known_facts` dict entries (sub-question answers). Each entry is a short answer string. With typically 2-4 sub-questions, this is also small (~200-500 chars).

**Risk is theoretical but real:** If a future change increases `kg_facts` size (e.g., including full triples, entity descriptions, or many sub-question answers), the function would silently drop all Wikipedia text. There is no warning or truncation of `kg_facts` itself.

---

## Edge Case #27: Double KG facts in prompt

**Location:** `text_solver.py:152-158` vs `text_solver.py:195`

**Verdict:** :white_check_mark: Handled — already fixed in current code

**Analysis:**

This edge case was identified as a potential duplication bug in the RAG path, but verification against the current codebase shows the issue does not exist:

1. `solve()` calls `_solve_rag()` at line 137-140, passing `known_facts`.
2. `_solve_rag()` calls `_assemble_from_chunks(reranked, "", max_chars=12_000)` at line 195. The second argument is an empty string `""`, NOT `known_facts`.
3. `_assemble_from_chunks()` has a guard at lines 52-54: `if kg_facts: parts.append(kg_facts)`. Since `""` is falsy, nothing is prepended.
4. Back in `solve()` at lines 152-158, the prompt template is filled:
   ```python
   prompt = TEXT_REASON_PROMPT.format(
       question=question, context=context, kg_facts=known_facts,
   )
   ```
5. KG facts appear exactly once via the `{kg_facts}` placeholder. The `{context}` contains only wiki chunks.

Both the RAG path and the keyword path now handle KG facts consistently — KG facts are injected only via the prompt template, not embedded in the context string. The test at `test_kg_provenance.py:54-64` verifies: `assert "Known facts" not in context`.

---

## Edge Case #28: RAG singleton memory

**Location:** `text_solver.py:29-44`, `rag/cache.py`, `rag/reranker.py:9-26`

**Verdict:** :white_check_mark: Handled — cache has eviction; models are appropriately sized

**Analysis:**

**WikiRAGCache (`rag/cache.py`):**
- Has a 7-day eviction policy at lines 13, 43-46:
  ```python
  _EVICTION_DAYS = 7
  def _evict_stale(self):
      cutoff = time.time() - _EVICTION_DAYS * 86400
      self._conn.execute("DELETE FROM wiki_pages WHERE fetched_at < ?", (cutoff,))
  ```
- Eviction runs once on construction (`__init__` at line 30).
- SQLite storage is disk-based, so memory impact is minimal (only the connection object is in-memory).
- For a 1000-question eval within a single session, no eviction will trigger (all entries are <7 days old). But SQLite handles this gracefully — disk storage scales linearly with unique Wikipedia pages (~500-2000 pages for 1000 questions), each stored as JSON text. Estimated size: 50-200 MB on disk, minimal RAM.

**Reranker (`rag/reranker.py:9-26`):**
- The cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is a lazy singleton at module level.
- Model size: ~80 MB in memory (MiniLM-L-6 has 22M parameters, float32).
- This is loaded once and stays in memory for the process lifetime. Appropriate for batch evaluation.

**Bi-encoder (`rag/vector_store.py:12-28`):**
- `all-MiniLM-L6-v2` is also a lazy singleton, ~80 MB in memory.
- Same lifecycle as the reranker — loaded once, never freed.

**Total additional memory for RAG path:** ~160 MB for the two models + minimal SQLite connection overhead. Reasonable for a Python data science workflow.

**Minor note:** Neither model singleton has a `close()` or `unload()` method. For long-running processes that switch between RAG and non-RAG modes, there is no way to reclaim the ~160 MB. Not an issue for the batch eval use case.

---

## Edge Case #29: retrieve_wikipedia() failure handling

**Location:** `graphrag_wiki.py:499-580`

**Verdict:** :white_check_mark: Handled — multi-layer fallback with graceful degradation

**Analysis:**

**Wikipedia API failures:**
- `get_wikipedia_text_by_title()` in `wiki_utils.py:69-86` uses a `_retry()` wrapper with 3 attempts and exponential backoff (lines 29-39).
- HTTP timeout is 15 seconds (line 76).
- Rate limiting: `_rate_limit_wiki()` enforces 0.5-second delay between calls (lines 21-26).
- Individual page failures are silently skipped — the loop continues to the next entity.

**Multi-layer fallback cascade:**
1. URI-based lookup (lines 511-512) — exact Wikipedia title from DBpedia URI
2. Web search fallback per entity (lines 529-543) — if URI lookup returns nothing, tries `web_search_wikipedia()` which itself has Wikipedia API primary + DuckDuckGo fallback with circuit breaker
3. Keyword-based web search (lines 545-561) — searches using question keywords
4. Full-question fallback (lines 563-578) — if `raw_texts` is still empty, searches with the full question text

**What if ALL pages fail?**
- `retrieve_wikipedia()` returns an empty dict `{}`.
- In the RAG path (`_solve_rag`, line 176): `if not chunks: return None`
- In the keyword path (`_solve_keyword`, line 205): `if not raw_texts: return None`
- `solve()` at line 147: `if context is None: return None`
- `hybrid_qa.py:287`: `if not answer: answer = "Could not find answer"`

The entire chain handles the empty case correctly — no crashes, just a graceful "no answer" response.

---

## Edge Case #30: retrieve_and_chunk_wikipedia() empty pages

**Location:** `graphrag_wiki.py:583-622`

**Verdict:** :white_check_mark: Handled — empty pages produce zero chunks

**Analysis:**

The function at lines 602-604:
```python
raw_texts = retrieve_wikipedia(entities, question, verbose=verbose)
if not raw_texts:
    return []
```

For each page in `raw_texts`, `chunk_wikipedia_page()` is called at line 615. In `chunker.py:48-49`:
```python
if not text or not text.strip():
    return []
```

So empty or whitespace-only pages produce zero chunks. These are simply not added to `all_chunks`.

**Chunk size and overlap:**
- Default chunk size: 512 characters (line 27)
- Default overlap: 64 characters (line 28)
- Section header is subtracted from effective chunk size (line 118-119): `effective_size = chunk_size - header_len`
- Minimum effective size is clamped to 100 chars (line 121): `if effective_size < 100: effective_size = 100`
- Sentences exceeding chunk size are truncated to `chunk_size` (line 138): `result_chunks.append(sentence[:chunk_size])`

**Edge case: very short pages.** A page with only 10 characters of text would produce a single chunk with that text. This is fine — the bi-encoder handles short texts.

**Cache interaction:** Cached chunks at lines 608-614 are returned directly. If a page was previously cached with content but the Wikipedia page has since been blanked, the cached (non-empty) chunks would be returned until the 7-day eviction. This is acceptable behavior.

---

## Edge Case #31: assemble_context() empty context

**Location:** `graphrag_wiki.py:629-717`

**Verdict:** :white_check_mark: Handled — returns non-empty string in all non-trivial cases

**Analysis:**

**Can it return an empty context string?**

Only if `ranked_pages` is empty AND no pages pass the keyword filter. Let's trace:

1. If `raw_texts` is empty, `assemble_context` would not normally be called (callers check for empty texts first).
2. If all pages have text but no keywords match, `extract_relevant_sections()` at line 397-398 falls back to `text[:char_budget]` — it always returns something if `text` is non-empty.
3. So `filtered_texts` will be non-empty as long as `raw_texts` is non-empty.
4. The budget loop at lines 706-714 will include at least the first page (since `total_chars` starts at `len(kg_facts)` which is typically 0-2 KB, well under 12K).

**Budget allocation specifics:**

When `seed_titles` is provided:
- Seed budget: 60% of 12K = 7,200 chars
- Bridge budget: 30% of 12K = 3,600 chars
- Per-page seed budget: `max(3000, 7200 // n_seed)` — at least 3K per seed page
- Per-page bridge budget: `max(1500, 3600 // n_bridge)` — at least 1.5K per bridge page

**The `kg_facts` budget reservation:**
Line 704: `total_chars = len(kg_facts) if kg_facts else 0` — this correctly reserves space in the final context for KG facts that the prompt template will add separately. The comment at line 702 confirms this is intentional: "Reserve space for kg_facts (injected by prompt template, not here)." The keyword path correctly does NOT include `kg_facts` in the returned context string (verified by test at `test_kg_provenance.py:62-63`).

**Minor issue:** If `kg_facts` exceeds `MAX_CONTEXT_CHARS` (12K), `total_chars` starts above the limit. The loop would try `remaining = MAX_CONTEXT_CHARS - total_chars` which would be negative, so `remaining > 200` would be false, and the context would be empty. This is the same theoretical risk as Edge Case #26. In practice, `kg_facts` is always small (see #26 analysis).

---

## Summary Table

| # | Edge Case | Verdict | Impact | Action Needed |
|---|-----------|---------|--------|---------------|
| 25 | RAGIndex per-call | Partial | Low - embedding cache exists but unused | Consider wiring `cache.get_embeddings()`/`store_embeddings()` |
| 26 | Budget overflow | Partial | Very Low - kg_facts always small in practice | Add defensive truncation of kg_facts |
| 27 | Double KG facts | Handled | None - already fixed in current code (`""` passed to `_assemble_from_chunks()`) | No action needed |
| 28 | Singleton memory | Handled | None | No action needed |
| 29 | Wikipedia failures | Handled | None | Multi-layer fallback works correctly |
| 30 | Empty page chunks | Handled | None | Chunker guards handle empty input |
| 31 | Empty context | Handled | None | Budget allocation always yields content |

### Recommended Fix Priority

1. **Edge Case #25 (Unused embedding cache)** — Low priority optimization. The `WikiRAGCache` already has `get_embeddings()`/`store_embeddings()` implemented. Wiring them into `_solve_rag()` would skip bi-encoder inference for cached pages, saving ~0.1-0.3 seconds per question for repeated entities.

2. **Edge Case #26 (Budget overflow)** — Defensive improvement. Add a guard in `_assemble_from_chunks()`: `kg_facts = kg_facts[:max_chars // 2]` to ensure at least half the budget goes to wiki text. Very low priority given current usage patterns.

*Note: Edge Case #27 (Double KG facts) was previously listed as the top priority bug, but verification confirmed the fix is already applied in the current code (`""` passed to `_assemble_from_chunks()` at line 195).*
