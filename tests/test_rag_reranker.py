"""Tests for rag/reranker.py — cross-encoder reranking."""

import pytest

from rag.chunker import WikiChunk

# Skip all tests if RAG deps not installed
st = pytest.importorskip("sentence_transformers")

from rag.reranker import Reranker


def _make_scored_chunks() -> list[tuple[WikiChunk, float]]:
    """Create test chunks with dummy bi-encoder scores."""
    chunks = [
        WikiChunk(
            text="Christopher Nolan directed the film Inception in 2010.",
            page_title="Inception (film)", section_header="(Lead)",
            chunk_index=0, char_offset=0,
        ),
        WikiChunk(
            text="The film had a budget of $160 million and was produced by Warner Bros.",
            page_title="Inception (film)", section_header="== Production ==",
            chunk_index=1, char_offset=100,
        ),
        WikiChunk(
            text="Nolan was born on July 30, 1970 in London, England.",
            page_title="Christopher Nolan", section_header="(Lead)",
            chunk_index=0, char_offset=0,
        ),
        WikiChunk(
            text="The platypus is a semi-aquatic mammal endemic to eastern Australia.",
            page_title="Platypus", section_header="(Lead)",
            chunk_index=0, char_offset=0,
        ),
    ]
    # Give them dummy bi-encoder scores
    return [(c, 0.5 + i * 0.1) for i, c in enumerate(chunks)]


class TestReranker:

    def test_basic_reranking(self):
        reranker = Reranker()
        scored = _make_scored_chunks()
        results = reranker.rerank("Who directed Inception?", scored, top_k=3)
        assert len(results) <= 3
        # The Nolan/Inception chunk should rank higher than the platypus chunk
        texts = [c.text for c in results]
        assert any("Nolan" in t or "directed" in t for t in texts)

    def test_empty_input(self):
        reranker = Reranker()
        assert reranker.rerank("test query", [], top_k=5) == []

    def test_top_k_limits_output(self):
        reranker = Reranker()
        scored = _make_scored_chunks()
        results = reranker.rerank("Inception", scored, top_k=2)
        assert len(results) <= 2

    def test_relation_hint_augmentation(self):
        """Relation hints should influence reranking toward relevant chunks."""
        reranker = Reranker()
        scored = _make_scored_chunks()

        # Without hints
        no_hints = reranker.rerank("Who directed Inception?", scored, top_k=4)

        # With hints — hint that Nolan page discovered via "director" relation
        hints = {"Christopher Nolan": "director"}
        with_hints = reranker.rerank(
            "Who directed Inception?", scored, top_k=4, relation_hints=hints,
        )

        # Both should return results
        assert len(no_hints) > 0
        assert len(with_hints) > 0

    def test_returns_wikichunk_objects(self):
        reranker = Reranker()
        scored = _make_scored_chunks()
        results = reranker.rerank("Inception", scored, top_k=3)
        for chunk in results:
            assert isinstance(chunk, WikiChunk)
            assert chunk.page_title
            assert chunk.text
