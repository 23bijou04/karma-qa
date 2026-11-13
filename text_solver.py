"""Unified TEXT solver: Wikipedia retrieval + context assembly + LLM reasoning.

Single entry point `solve()` used by both the main text fallback in hybrid_qa.py
and per-sub-question TEXT resolution in logicrag_decomposition.py.
"""

import logging

from models import Entity, extract_final_answer
from graphrag_wiki import assemble_context, retrieve_wikipedia
from llm_provider import call_llm
from wiki_utils import uri_to_wiki_title

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RAG import guard — falls back to keyword path if deps not installed
# ---------------------------------------------------------------------------
try:
    from rag.cache import WikiRAGCache
    from rag.reranker import Reranker
    from rag.infrastructure.vector_store import RAGIndex

    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

# Module-level singletons (lazy-initialized on first RAG call)
_rag_cache: "WikiRAGCache | None" = None
_reranker: "Reranker | None" = None
_corpus_index: "RAGIndex | None" = None
_last_solve_meta: dict = {}


def _get_rag_cache():
    global _rag_cache
    if _rag_cache is None:
        _rag_cache = WikiRAGCache()
    return _rag_cache


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def _build_corpus_index(benchmark_corpus: list[dict], verbose: bool = False) -> "RAGIndex":
    """Build or return cached FAISS index from benchmark corpus.

    Built once on first call, reused for all subsequent questions.
    Embeds whole entries with title prefix (matches LogicRAG format).
    Uses WikiRAGCache for embedding persistence across runs.

    NOTE: If switching from the old chunked format, delete data/wiki_cache.db
    first — short entries (<=512 chars) have matching chunk counts (1==1) so
    the cache won't auto-invalidate their stale embeddings.
    """
    global _corpus_index
    if _corpus_index is not None:
        return _corpus_index

    from rag.chunker import WikiChunk

    chunks = []
    for entry in benchmark_corpus:
        title, text = entry["title"], entry["text"]
        # Embed whole entry with title prefix (matches LogicRAG format)
        full_text = f"Title: {title}. Content: {text}"
        chunks.append(WikiChunk(
            text=full_text, page_title=title,
            section_header="(Lead)", chunk_index=0, char_offset=0,
        ))

    cache = _get_rag_cache()
    index = RAGIndex()
    index.index_chunks(chunks, cache=cache)
    _corpus_index = index

    if verbose:
        print(f"  Built corpus FAISS index: {len(chunks)} entries (no chunking)")

    return _corpus_index


def _assemble_from_chunks(chunks, kg_facts: str = "",
                          max_chars: int = 25_000,
                          max_per_entity: int = 4_000) -> str:
    """Build LLM context string from reranked WikiChunks within char budget.

    Args:
        max_per_entity: Max chars per page_title to prevent volume bias.
    """
    parts: list[str] = []
    total = 0
    entity_used: dict[str, int] = {}

    if kg_facts:
        parts.append(kg_facts)
        total += len(kg_facts)

    for chunk in chunks:
        entry = f"=== {chunk.page_title} / {chunk.section_header} ===\n{chunk.text}"
        if total + len(entry) > max_chars:
            break
        page = chunk.page_title
        used = entity_used.get(page, 0)
        if used + len(entry) > max_per_entity:
            continue  # skip, this entity already has enough context
        parts.append(entry)
        total += len(entry)
        entity_used[page] = used + len(entry)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TEXT_REASON_PROMPT = """Answer this question using the sources below.

Question: {question}

{kg_facts}Wikipedia text:
{context}

Instructions:
- Knowledge graph facts (if present) are verified structured data — trust them
- Use Wikipedia text to fill gaps not covered by KG facts
- Show brief step-by-step reasoning
- End with exactly: FINAL ANSWER: <answer>
"""

TEXT_SUBQ_PROMPT = """Answer this sub-question using the sources below.

Sub-question: {question}

{kg_facts}Wikipedia text:
{context}

Instructions:
- Knowledge graph facts (if present) are verified structured data -- trust them
- Use Wikipedia text to fill gaps not covered by KG facts
- Give a brief, factual answer
- End with exactly: FINAL ANSWER: <answer>
"""


def solve(
    question: str,
    entities: list[Entity],
    known_facts: str = "",
    relation_hints: dict[str, str] | None = None,
    llm_client=None,
    model: str = "",
    verbose: bool = False,
    is_subquestion: bool = False,
    seed_titles: set[str] | None = None,
    benchmark_corpus: list[dict] | None = None,
) -> str | None:
    """Retrieve Wikipedia text, assemble context, LLM reasoning.

    Uses RAG (bi-encoder + cross-encoder) if dependencies installed,
    otherwise falls back to legacy keyword-scored extraction.

    Args:
        question: The question to answer.
        entities: Entities to retrieve Wikipedia pages for.
        known_facts: Pre-formatted KG facts or accumulated sub-Q answers.
        relation_hints: Title->relation map for section boosting / query augmentation.
        llm_client: OpenAI-compatible LLM client.
        model: LLM model name.
        verbose: Print pipeline steps.
        is_subquestion: Use sub-question prompt (shorter, more direct).
        seed_titles: Override seed titles for budget allocation / score boost.

    Returns:
        Answer string if successful, None if resolution fails.
    """
    global _last_solve_meta
    _last_solve_meta = {}
    # Build seed titles if not provided
    if seed_titles is None:
        seed_titles = {
            uri_to_wiki_title(e.uri) for e in entities
            if e.uri and "/resource/" in e.uri
        }

    if _RAG_AVAILABLE:
        try:
            context = _solve_rag(
                question, entities, known_facts, relation_hints or {},
                seed_titles, verbose, benchmark_corpus=benchmark_corpus,
            )
            _last_solve_meta["retrieval_path"] = "rag"
            _last_solve_meta["context_chars"] = len(context) if context else 0
        except Exception as e:
            logger.warning("RAG retrieval failed, falling back to keyword: %s", e,
                          exc_info=True)
            _last_solve_meta["retrieval_path"] = "keyword_fallback_from_rag"
            _last_solve_meta["rag_error"] = str(e)
            context = _solve_keyword(
                question, entities, known_facts, relation_hints or {},
                seed_titles, verbose, benchmark_corpus=benchmark_corpus,
            )
    else:
        context = _solve_keyword(
            question, entities, known_facts, relation_hints or {},
            seed_titles, verbose, benchmark_corpus=benchmark_corpus,
        )
        _last_solve_meta["retrieval_path"] = "keyword"
        _last_solve_meta["context_chars"] = len(context) if context else 0

    if context is None:
        return None

    # Choose prompt
    if is_subquestion:
        prompt = TEXT_SUBQ_PROMPT.format(
            question=question, context=context, kg_facts=known_facts,
        )
    else:
        prompt = TEXT_REASON_PROMPT.format(
            question=question, context=context, kg_facts=known_facts,
        )

    _last_solve_meta["llm_prompt"] = prompt
    response = call_llm(llm_client, prompt, model)
    _last_solve_meta["llm_raw_response"] = response
    answer = extract_final_answer(response) if response else None

    if verbose and answer:
        print(f"  TEXT answer: {answer}")

    return answer


def _solve_rag(question, entities, known_facts, relation_hints,
               seed_titles, verbose, benchmark_corpus=None) -> str | None:
    """RAG path: bi-encoder retrieval → cross-encoder reranking → context assembly."""
    if benchmark_corpus:
        rag_index = _build_corpus_index(benchmark_corpus, verbose=verbose)
        if verbose:
            print("  Retrieving from shared corpus index (no live Wikipedia)")
    else:
        from graphrag_wiki import retrieve_and_chunk_wikipedia

        cache = _get_rag_cache()
        chunks = retrieve_and_chunk_wikipedia(entities, question, cache=cache, verbose=verbose)
        if not chunks:
            if verbose:
                print("  No Wikipedia chunks retrieved (RAG path).")
            return None
        rag_index = RAGIndex()
        rag_index.index_chunks(chunks, cache=cache)

    top_chunks = rag_index.retrieve(question, top_k=20, seed_titles=seed_titles)
    _last_solve_meta["rag_chunks_retrieved"] = len(top_chunks)
    # top_chunks is list[(WikiChunk, float)] from RAGIndex.retrieve()
    _last_solve_meta["rag_pages_in_top"] = list({c.page_title for c, _ in top_chunks})

    if not top_chunks:
        if verbose:
            print("  RAG retrieval returned no results.")
        return None

    reranker = _get_reranker()
    corpus_hints = {} if benchmark_corpus else relation_hints
    reranked = reranker.rerank(
        question, top_chunks, top_k=8, relation_hints=corpus_hints,
    )
    _last_solve_meta["rag_chunks_reranked"] = len(reranked)
    _last_solve_meta["rag_unique_pages"] = len({c.page_title for c in reranked})
    _last_solve_meta["rag_reranked_pages"] = list({c.page_title for c in reranked})

    context = _assemble_from_chunks(reranked, "", max_chars=12_000)
    if verbose:
        print(f"  RAG context: {len(reranked)} chunks, {len(context)} chars")
    return context


def _solve_keyword(question, entities, known_facts, relation_hints,
                   seed_titles, verbose, benchmark_corpus=None) -> str | None:
    """Legacy keyword path: keyword-scored paragraph extraction."""
    if benchmark_corpus:
        if verbose:
            print("  Keyword path skipped (corpus mode, RAG already attempted)")
        return None

    raw_texts = retrieve_wikipedia(entities, question, verbose=verbose)
    _last_solve_meta["pages_fetched"] = list(raw_texts.keys())
    if not raw_texts:
        if verbose:
            print("  No Wikipedia text retrieved.")
        return None

    context, _ = assemble_context(
        raw_texts, question, seed_titles=seed_titles,
        kg_facts=known_facts, relation_hints=relation_hints,
        verbose=verbose,
    )
    return context
