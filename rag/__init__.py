"""RAG (Retrieval-Augmented Generation) module for semantic Wikipedia retrieval.

Two-stage pipeline: bi-encoder retrieval (FAISS) → cross-encoder reranking.
Falls back gracefully to keyword path if dependencies not installed.
"""

from rag.cache import WikiRAGCache
from rag.chunker import WikiChunk, chunk_wikipedia_page

__all__ = [
    "WikiChunk",
    "chunk_wikipedia_page",
    "WikiRAGCache",
]

# Conditionally export vector store and reranker (need sentence-transformers + faiss)
try:
    from rag.reranker import Reranker
    from rag.infrastructure.vector_store import RAGIndex

    __all__ += ["RAGIndex", "Reranker"]
except ImportError:
    pass
