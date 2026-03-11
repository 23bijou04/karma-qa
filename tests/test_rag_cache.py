"""Tests for rag/cache.py — SQLite chunk/embedding persistence."""

import time

import numpy as np
import pytest

from rag.cache import _EVICTION_DAYS, WikiRAGCache
from rag.chunker import WikiChunk


@pytest.fixture
def cache(tmp_path):
    """Create a cache backed by a temp SQLite file."""
    db_path = tmp_path / "test_cache.db"
    c = WikiRAGCache(db_path=db_path)
    yield c
    c.close()


def _make_chunks(title: str, n: int = 3) -> list[WikiChunk]:
    return [
        WikiChunk(
            text=f"Chunk {i} of {title}",
            page_title=title,
            section_header=f"== Section {i} ==",
            chunk_index=i,
            char_offset=i * 100,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Chunks round-trip
# ---------------------------------------------------------------------------

def test_store_and_retrieve_chunks(cache):
    chunks = _make_chunks("Albert Einstein")
    cache.store_chunks("Albert Einstein", chunks)
    retrieved = cache.get_chunks("Albert Einstein")
    assert retrieved is not None
    assert len(retrieved) == 3
    assert retrieved[0].text == "Chunk 0 of Albert Einstein"
    assert retrieved[0].page_title == "Albert Einstein"
    assert retrieved[2].chunk_index == 2


def test_get_chunks_missing(cache):
    assert cache.get_chunks("Nonexistent Page") is None


def test_store_chunks_overwrite(cache):
    chunks1 = _make_chunks("Test", n=2)
    chunks2 = _make_chunks("Test", n=5)
    cache.store_chunks("Test", chunks1)
    cache.store_chunks("Test", chunks2)
    retrieved = cache.get_chunks("Test")
    assert len(retrieved) == 5


# ---------------------------------------------------------------------------
# Embeddings round-trip
# ---------------------------------------------------------------------------

def test_store_and_retrieve_embeddings(cache):
    # Must store chunks first (creates the row)
    cache.store_chunks("Test", _make_chunks("Test", n=3))
    emb = np.random.randn(3, 384).astype(np.float32)
    cache.store_embeddings("Test", emb)
    retrieved = cache.get_embeddings("Test")
    assert retrieved is not None
    np.testing.assert_array_almost_equal(retrieved, emb)


def test_get_embeddings_missing(cache):
    assert cache.get_embeddings("Nonexistent") is None


def test_get_embeddings_not_stored(cache):
    """Chunks stored but no embeddings yet."""
    cache.store_chunks("Test", _make_chunks("Test"))
    assert cache.get_embeddings("Test") is None


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

def test_eviction_stale_entries(tmp_path):
    db_path = tmp_path / "evict_test.db"
    cache = WikiRAGCache(db_path=db_path)
    cache.store_chunks("Old Page", _make_chunks("Old Page"))

    # Manually backdate the entry
    cache._conn.execute(
        "UPDATE wiki_pages SET fetched_at = ? WHERE title = ?",
        (time.time() - (_EVICTION_DAYS + 1) * 86400, "Old Page"),
    )
    cache._conn.commit()

    # Re-open cache (triggers eviction)
    cache.close()
    cache2 = WikiRAGCache(db_path=db_path)
    assert cache2.get_chunks("Old Page") is None
    cache2.close()


def test_fresh_entries_not_evicted(cache):
    cache.store_chunks("Fresh Page", _make_chunks("Fresh Page"))
    # Re-read should work
    assert cache.get_chunks("Fresh Page") is not None


# ---------------------------------------------------------------------------
# Concurrent reads (basic)
# ---------------------------------------------------------------------------

def test_concurrent_read_same_db(tmp_path):
    """Two cache instances reading the same DB should not conflict."""
    db_path = tmp_path / "shared.db"
    c1 = WikiRAGCache(db_path=db_path)
    c1.store_chunks("Test", _make_chunks("Test"))

    c2 = WikiRAGCache(db_path=db_path)
    retrieved = c2.get_chunks("Test")
    assert retrieved is not None
    assert len(retrieved) == 3

    c1.close()
    c2.close()
