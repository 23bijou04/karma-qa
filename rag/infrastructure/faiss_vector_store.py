# rag/infrastructure/faiss_vector_store.py
"""Index vectoriel FAISS avec embeddings Sentence Transformers."""

import logging

import numpy as np

from rag.domain.models import Chunk

logger = logging.getLogger(__name__)


class FaissVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._encoder = None
        self._chunks: list[Chunk] = []
        self._index = None

    def _get_encoder(self):
        """Charge le modèle une seule fois pour cette instance."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Chargement du modèle : %s", self._model_name)
            self._encoder = SentenceTransformer(self._model_name)

        return self._encoder

    def index(self, chunks: list[Chunk]) -> None:
        """Remplace l’index existant par les chunks fournis."""
        if not chunks:
            self.clear()
            return

        import faiss

        encoder = self._get_encoder()
        embeddings = encoder.encode(
            [chunk.text for chunk in chunks],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        # Sur des vecteurs normalisés, le produit scalaire
        # correspond à la similarité cosinus.
        dimension = embeddings.shape[1]
        new_index = faiss.IndexFlatIP(dimension)
        new_index.add(embeddings)

        # Remplacer l’ancien état après construction du nouvel index.
        self._chunks = list(chunks)
        self._index = new_index

    def search(self, query: str, top_k: int) -> list[Chunk]:
        """Retourne les chunks classés par similarité décroissante."""
        if top_k < 1:
            raise ValueError("top_k doit être supérieur à 0")

        query = query.strip()
        if not query:
            raise ValueError("La recherche ne peut pas être vide")

        if self._index is None or not self._chunks:
            return []

        encoder = self._get_encoder()
        query_embedding = encoder.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        query_embedding = np.ascontiguousarray(
            query_embedding,
            dtype=np.float32,
        )

        limit = min(top_k, len(self._chunks))
        _, indices = self._index.search(query_embedding, limit)

        return [
            self._chunks[int(index)]
            for index in indices[0]
            if index >= 0
        ]

    def clear(self) -> None:
        """Vide l’index, mais conserve le modèle chargé."""
        self._chunks = []
        self._index = None