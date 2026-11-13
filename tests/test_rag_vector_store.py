"""Tests for rag/vector_store.py — FAISS index + retrieval."""

import pytest

from rag.chunker import WikiChunk

# Skip all tests if RAG deps not installed
faiss = pytest.importorskip("faiss")
st = pytest.importorskip("sentence_transformers")

from rag.infrastructure.vector_store import RAGIndex


def _make_chunks() -> list[WikiChunk]:
    """Create test chunks with distinct semantic content."""
    return [
        WikiChunk(
            text="Christopher Nolan directed the film Inception in 2010.",
            page_title="Inception (film)", section_header="(Lead)",
            chunk_index=0, char_offset=0,
        ),
        WikiChunk(
            text="The film grossed over $800 million worldwide at the box office.",
            page_title="Inception (film)", section_header="== Box office ==",
            chunk_index=1, char_offset=100,
        ),
        WikiChunk(
            text="Leonardo DiCaprio starred as Dom Cobb in the lead role.",
            page_title="Inception (film)", section_header="== Cast ==",
            chunk_index=2, char_offset=200,
        ),
        WikiChunk(
            text="Nolan was born on July 30, 1970 in London, England.",
            page_title="Christopher Nolan", section_header="(Lead)",
            chunk_index=0, char_offset=0,
        ),
        WikiChunk(
            text="He studied English literature at University College London.",
            page_title="Christopher Nolan", section_header="== Early life ==",
            chunk_index=1, char_offset=100,
        ),
    ]


class TestRAGIndex:

    def test_basic_retrieval(self):
        index = RAGIndex()
        chunks = _make_chunks()
        index.index_chunks(chunks)

        results = index.retrieve("Who directed Inception?", top_k=3)
        assert len(results) > 0
        # The chunk about Nolan directing should be highly ranked
        top_texts = [c.text for c, _ in results]
        assert any("Nolan directed" in t for t in top_texts)

    def test_empty_index(self):
        index = RAGIndex()
        index.index_chunks([])
        results = index.retrieve("test query")
        assert results == []

    def test_retrieve_empty_index(self):
        """Retrieve on never-indexed RAGIndex returns empty."""
        index = RAGIndex()
        assert index.retrieve("test") == []

    def test_seed_boost(self):
        """Chunks from seed titles should get higher scores."""
        index = RAGIndex()
        index.index_chunks(_make_chunks())

        # Query about Nolan's birth — relevant to both pages
        query = "When was the director born?"
        no_boost = index.retrieve(query, top_k=5, seed_titles=None)
        with_boost = index.retrieve(
            query, top_k=5,
            seed_titles={"Christopher Nolan"}, seed_boost=2.0,
        )

        # With high boost, Christopher Nolan chunks should rank higher
        boosted_nolan_ranks = [
            i for i, (c, _) in enumerate(with_boost)
            if c.page_title == "Christopher Nolan"
        ]
        unboosted_nolan_ranks = [
            i for i, (c, _) in enumerate(no_boost)
            if c.page_title == "Christopher Nolan"
        ]

        # At minimum, boost should not make results worse
        assert len(with_boost) > 0
        # With 2x boost, Nolan chunks should appear at or near top
        if boosted_nolan_ranks and unboosted_nolan_ranks:
            assert min(boosted_nolan_ranks) <= min(unboosted_nolan_ranks)

    def test_top_k_limits_results(self):
        index = RAGIndex()
        index.index_chunks(_make_chunks())
        results = index.retrieve("film", top_k=2)
        assert len(results) <= 2

    def test_clear_resets_index(self):
        index = RAGIndex()
        index.index_chunks(_make_chunks())
        assert len(index.retrieve("Nolan")) > 0

        index.clear()
        assert index.retrieve("Nolan") == []

    def test_scores_are_floats(self):
        index = RAGIndex()
        index.index_chunks(_make_chunks())
        results = index.retrieve("Inception")
        for _, score in results:
            assert isinstance(score, float)
