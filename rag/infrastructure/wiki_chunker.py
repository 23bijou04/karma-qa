"""Découpeur de pages Wikipédia pour le RAG (implémente Chunker).

Rôle : recevoir le texte d'UNE page (celui de WikipediaSearch.fetch_page) et le
       découper en chunks, chacun avec sa provenance (titre, url, version).

Utilisation depuis Python :
   from rag.wiki_chunker import WikiChunker
   chunker = WikiChunker()                      # réglages de rag/config.py
   chunks = chunker.chunk_page("Jane Doe", texte, url, revid)

Lancement terminal (nécessite internet, depuis la racine du projet) :
   python -m rag.wiki_chunker "Who was John Middleton Murry's wife?"

Le découpage lui-même est fait par rag/chunker.py (sections, puis paragraphes,
puis phrases, sans couper une phrase). Ce fichier-ci ajoute la provenance.
"""
import sys
from dataclasses import replace   # copie un chunk en changeant seulement certains champs

from rag.chunker import WikiChunk, chunk_wikipedia_page
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.protocols import Chunker


class WikiChunker(Chunker):
    """Découpe le texte d'une page Wikipédia en chunks (tailles en caractères)."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size    # taille visée d'un chunk, en caractères
        self.overlap = overlap          # caractères répétés entre deux chunks qui se suivent

    def chunk_page(self, title: str, text: str, url: str = "", revid: int | None = None) -> list[WikiChunk]:
        """Renvoie les chunks de la page ([] si le titre ou le texte est vide)."""
        if not title or not text:
            return []

        # Le vrai découpage (sections -> paragraphes -> phrases)
        chunks = chunk_wikipedia_page(title, text, chunk_size=self.chunk_size, overlap=self.overlap)

        # Ajout de la provenance : url, version de la page, identifiant unique
        return [
            replace(c, url=url, revid=revid, chunk_id=self._make_chunk_id(title, revid, c.chunk_index))
            for c in chunks
        ]

    def _make_chunk_id(self, title: str, revid: int | None, chunk_index: int) -> str:
        """Identifiant unique d'un chunk, par exemple "Jane Doe:900001:500/50:3".

        = titre de la page, version de la page, réglages de découpage, numéro du chunk.
        L'identifiant change si la page change de version ou si on change les réglages
        de découpage : deux découpages différents ne partagent jamais le même identifiant.
        """
        rev = revid if revid is not None else "norev"
        return f"{title}:{rev}:{self.chunk_size}/{self.overlap}:{chunk_index}"


if __name__ == "__main__":
    # Chaîne complète : question -> pages -> texte -> chunks
    from rag.wikipedia_search2 import WikipediaSearch

    question = " ".join(sys.argv[1:]) or input("Question : ")
    source = WikipediaSearch()
    chunker = WikiChunker()

    for url in source.search_pages(question):
        text = source.fetch_page(url)
        v = source.revisions.get(url, {})                    # titre réel et version lue
        chunks = chunker.chunk_page(v.get("title", ""), text, url, v.get("revid"))
        print(f"\n{v.get('title', url)} : {len(text)} caractères -> {len(chunks)} chunks")
        for c in chunks[:2]:                                 # aperçu des 2 premiers
            print(f"   [{c.chunk_id}] {c.section_header} | {c.text[:70]!r}")