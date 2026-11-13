# rag/application/retrieval_service.py
from rag.application.ports import Reranker, VectorStore
from rag.domain.models import Chunk


class RetrievalService:
    def __init__(
        self,
        vector_store: VectorStore,
        reranker: Reranker,
        retrieval_top_k: int = 20,
        context_top_k: int = 5,
    ):
        if not 1 <= context_top_k <= retrieval_top_k:
            raise ValueError(
                "Il faut respecter : 1 <= context_top_k <= retrieval_top_k"
            )

        self._vector_store = vector_store
        self._reranker = reranker
        self._retrieval_top_k = retrieval_top_k
        self._context_top_k = context_top_k

    def retrieve(
        self,
        question: str,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Indexe les chunks et retourne les passages les plus pertinents."""
        question = question.strip()
        if not question:
            raise ValueError("La question ne peut pas être vide")

        if not chunks:
            return []

        self._vector_store.index(chunks)

        candidates = self._vector_store.search(
            question,
            top_k=self._retrieval_top_k,
        )

        if not candidates:
            return []

        return self._reranker.rerank(
            question,
            candidates,
            top_k=self._context_top_k,
        )