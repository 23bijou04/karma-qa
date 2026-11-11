from typing import Protocol
from rag.chunker import WikiChunk

class DocumentSource(Protocol):
    def search_pages(self, question: str, limit: int) -> list[str]:
        """Retourne les URL des pages pertinentes."""
        ...

    def fetch_page(self, url: str) -> str:
        """Retourne le contenu textuel d’une page."""
        ...  

class Chunker(Protocol): 
    def chunk_page(self , page: str) -> list[WikiChunk]:
        """Retourne le contenu textuel d’une page découpée."""
        ...

class Embedder(Protocol): 
    def embed():
        ...        # Produire les vecteurs

class EncoderManager(Protocol):
    def get_encoder():
        ...

class Reranker(Protocol):
    def re_rank(self,
            query: str,
            chunks: list[tuple[WikiChunk, float]],
            top_k: int ,
            relation_hints: dict[str, str] | None = None,
        ) -> list[WikiChunk]:
        ...
         # reclasser les chunks après le retreive 

class Retriever(Protocol): 
    def retreive(self,
            query: str,
            top_k: int = 30,
            seed_titles: set[str] | None = None,
            seed_boost: float = 1.2,
        ) -> list[tuple[WikiChunk, float]]:
        ...       # Retrouver les passages pertinents

class LLMProvider(Protocol):
    def answer(self, question: str,  context: str) -> str:
        ...      # Générer la réponse