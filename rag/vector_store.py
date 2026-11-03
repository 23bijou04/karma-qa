"""FAISS-based vector store with sentence-transformers bi-encoder."""

import logging

import numpy as np

from rag.chunker import WikiChunk

logger = logging.getLogger(__name__)

# Lazy-loaded singleton (same pattern as _nlp in graphrag_wiki.py)
_encoder = None


def _get_encoder(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the sentence-transformer model."""
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for RAG retrieval. "
                "Install with: pip install -e '.[rag]'"
            )
        logger.info("Loading bi-encoder model: %s", model_name)
        _encoder = SentenceTransformer(model_name)
    return _encoder


class RAGIndex:
    """FAISS flat index over WikiChunk embeddings.

    All pages go into ONE index for cross-page multi-hop retrieval.
    Seed entity chunks receive a configurable score boost.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._chunks: list[WikiChunk] = []
        self._index = None
        self._empty = True

    def index_chunks(self, chunks: list[WikiChunk], cache=None) -> None:
        """Encode and index all chunks into a FAISS flat index.

        If cache (WikiRAGCache) is provided, checks for cached embeddings
        per page before encoding and stores newly computed embeddings.
        """
        if not chunks:
            self._empty = True
            return

            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for RAG retrieval. "
                "Install with: pip install -e '.[rag]'"
            )

        encoder = _get_encoder(self._model_name)
        dim = encoder.get_sentence_embedding_dimension()

        if cache:
            embeddings = self._index_with_cache(chunks, encoder, cache, dim)
        else:
            texts = [c.text for c in chunks]
            embeddings = encoder.encode(
                texts, show_progress_bar=False, normalize_embeddings=True,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)

        self._index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine
        self._index.add(embeddings)
        self._chunks = list(chunks)
        self._empty = False

    def _index_with_cache(self, chunks, encoder, cache, dim):
        """Encode chunks with per-page embedding cache."""
        # Group chunk indices by page_title
        page_indices: dict[str, list[int]] = {}
        for i, chunk in enumerate(chunks):
            page_indices.setdefault(chunk.page_title, []).append(i)

        embeddings = np.zeros((len(chunks), dim), dtype=np.float32)
        to_encode_indices: list[int] = []
        to_encode_texts: list[str] = []

        for title, indices in page_indices.items():
            cached_emb = cache.get_embeddings(title)
            if cached_emb is not None and len(cached_emb) == len(indices):
                for j, idx in enumerate(indices):
                    embeddings[idx] = cached_emb[j]
            else:
                for idx in indices:
                    to_encode_indices.append(idx)
                    to_encode_texts.append(chunks[idx].text)

        if to_encode_texts:
            new_emb = encoder.encode(
                to_encode_texts, show_progress_bar=False, normalize_embeddings=True,
            )
            new_emb = np.asarray(new_emb, dtype=np.float32)
            for j, idx in enumerate(to_encode_indices):
                embeddings[idx] = new_emb[j]

            # Store newly computed embeddings per page
            for title, indices in page_indices.items():
                cached_emb = cache.get_embeddings(title)
                if cached_emb is None or len(cached_emb) != len(indices):
                    page_emb = np.array(
                        [embeddings[i] for i in indices], dtype=np.float32,
                    )
                    cache.store_embeddings(title, page_emb)

        return embeddings

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        seed_titles: set[str] | None = None,
        seed_boost: float = 1.2,
    ) -> list[tuple[WikiChunk, float]]:
        """Retrieve top-k chunks by cosine similarity with optional seed boost.

        Args:
            query: Question text to search for.
            top_k: Number of chunks to return.
            seed_titles: Page titles to boost (seed entities).
            seed_boost: Multiplicative boost for seed page chunks.

        Returns:
            List of (chunk, score) sorted by descending score.
        """
        if self._empty or self._index is None:
            return []

        encoder = _get_encoder(self._model_name)
        q_emb = encoder.encode([query], show_progress_bar=False, normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)

        # Retrieve more than needed to account for boost reordering
        fetch_k = min(top_k * 2, self._index.ntotal)
        scores, indices = self._index.search(q_emb, fetch_k)

        results: list[tuple[WikiChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            final_score = float(score)
            if seed_titles and chunk.page_title in seed_titles:
                final_score *= seed_boost
            results.append((chunk, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """Reset index for next question."""
        self._chunks = []
        self._index = None
        self._empty = True
