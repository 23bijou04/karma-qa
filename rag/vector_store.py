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

    def index_chunks(self, chunks: list[WikiChunk]) -> None:
        """Encode and index all chunks into a FAISS flat index."""
        if not chunks:
            self._empty = True
            return

        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for RAG retrieval. "
                "Install with: pip install -e '.[rag]'"
            )

        encoder = _get_encoder(self._model_name)
        texts = [c.text for c in chunks]
        embeddings = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine
        self._index.add(embeddings)
        self._chunks = list(chunks)
        self._empty = False

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
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
