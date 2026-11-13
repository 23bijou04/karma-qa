# rag/infrastructure/cross_encoder_reranker.py
"""Reclassement des passages avec un cross-encoder."""

import logging

from rag.domain.models import Chunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self._model_name = model_name
        self._cross_encoder = None

    def _get_cross_encoder(self):
        """Charge le modèle une seule fois pour cette instance."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Chargement du cross-encoder : %s",
                self._model_name,
            )
            self._cross_encoder = CrossEncoder(self._model_name)

        return self._cross_encoder

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int,
    ) -> list[Chunk]:
        """Retourne les meilleurs chunks par pertinence décroissante."""
        if top_k < 1:
            raise ValueError("top_k doit être supérieur à 0")

        query = query.strip()
        if not query:
            raise ValueError("La recherche ne peut pas être vide")

        if not chunks:
            return []

        cross_encoder = self._get_cross_encoder()

        pairs = [
            (query, chunk.text)
            for chunk in chunks
        ]
        scores = cross_encoder.predict(pairs)

        ranked_chunks = sorted(
            zip(chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            chunk
            for chunk, _ in ranked_chunks[:top_k]
        ]