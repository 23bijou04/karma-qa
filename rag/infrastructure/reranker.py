"""Cross-encoder reranker for RAG retrieval refinement."""

import logging

from rag.chunker import WikiChunk

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_cross_encoder = None


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy-load the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for RAG reranking. "
                "Install with: pip install -e '.[rag]'"
            )
        logger.info("Loading cross-encoder model: %s", model_name)
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


class Reranker:
    """Cross-encoder reranker with relation hint query augmentation.

    Takes top-N bi-encoder results and reranks to top-K using a cross-encoder.
    Supports query augmentation: if a chunk's page has a relation hint,
    the query is augmented with that context for more precise reranking.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        chunks: list[tuple[WikiChunk, float]],
        top_k: int = 15,
        relation_hints: dict[str, str] | None = None,
    ) -> list[WikiChunk]:
        """Rerank chunks using cross-encoder.

        Args:
            query: Original question.
            chunks: List of (chunk, bi-encoder score) from vector store.
            top_k: Number of chunks to return after reranking.
            relation_hints: {page_title: relation_name} for query augmentation.

        Returns:
            Top-K reranked WikiChunk objects (best first).
        """
        if not chunks:
            return []

        cross_encoder = _get_cross_encoder(self._model_name)

        # Build query-document pairs with optional relation augmentation
        pairs = []
        chunk_objects = []
        for chunk, _ in chunks:
            augmented_query = query
            if relation_hints and chunk.page_title in relation_hints:
                relation = relation_hints[chunk.page_title]
                augmented_query = f"{query} [context: discovered via {relation} relation]"
            pairs.append((augmented_query, chunk.text))
            chunk_objects.append(chunk)

        scores = cross_encoder.predict(pairs)

        # Sort by cross-encoder score descending
        scored = sorted(zip(chunk_objects, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[:top_k]]
