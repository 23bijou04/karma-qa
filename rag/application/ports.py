from typing import Protocol

from rag.domain.models import Chunk, Document


class DocumentSource(Protocol):
    def search_pages(self, question: str, limit: int) -> list[str]:
        """Retourne au maximum `limit` URL pertinentes."""
        ...

    def fetch_page(self, url: str) -> Document:
        """Retourne le texte, le titre et la source du document."""
        ...


class Chunker(Protocol):
    def chunk_document(self, document: Document) -> list[Chunk]:
        """Découpe le document en conservant ses informations de source."""
        ...


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Retourne un vecteur par texte, dans le même ordre."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Retourne un vecteur compatible avec ceux des documents."""
        ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int,
    ) -> list[Chunk]:
        """Retourne au maximum `top_k` chunks, du plus au moins pertinent."""
        ...


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[Chunk]:
        """Retourne au maximum `top_k` passages pertinents."""
        ...


class LLMProvider(Protocol):
    def generate(self, question: str, context: str) -> str:
        """Génère une réponse à partir de la question et du contexte."""
        ...