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

class VectorStore(Protocol):
    def index(self, chunks: list[Chunk]) -> None:
        """Remplace le contenu de l’index par les chunks fournis.

        Calcule et stocke leurs embeddings.
        """
        ...

    def search(self, query: str, top_k: int) -> list[Chunk]:
        """Retourne au maximum `top_k` chunks par similarité.

        Calcule l’embedding de la question avec un modèle
        compatible avec celui des chunks indexés.
        """
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




class LLMProvider(Protocol):
    def generate(self, question: str, context: str) -> str:
        """Génère une réponse à partir de la question et du contexte."""
        ...