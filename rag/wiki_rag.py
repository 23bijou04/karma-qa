"""Performe une recherche RAG sur 50 sous questtions de 2wiki."""
from rag import build_chunks
from rag.chunker import WikiChunk
from rag.vector_store import RAGIndex
import json
from pathlib import Path


def lire_pages_json(chemin: str | Path) -> list[dict]:
    """Lit le JSON et retourne les entrées contenant les pages par sous-question."""
    with open(chemin, "r", encoding="utf-8") as fichier:
        items = json.load(fichier)

    if not isinstance(items, list):
        raise ValueError("Le fichier JSON doit contenir une liste d'entrées.")

    return items


rag_index = RAGIndex()
items = lire_pages_json("results/wikipedia_pages_2wiki.json")
results, stats = build_chunks.chunk_pages(items)

for item in results:
    chunks = [WikiChunk(**c) for c in item["chunks"]]

    # Construire l’index avec les chunks de cette sous-question
    rag_index.clear()
    rag_index.index_chunks(chunks)

    # Retrouver les passages pertinents
    matches = rag_index.retrieve(
        query=item["sub_question"],
        top_k=5,
    )

    contexte = "\n\n".join(chunk.text for chunk, score in matches)

    # Puis :
    # 1. Envoyer la sous-question et le contexte à Ollama
    # 2. Récupérer la réponse
    # 3. L’enregistrer dans un fichier avec ton code Python