"""
build_chunks.py  -  Étape RAG de KARMA-Live : pages Wikipédia -> chunks

Rôle : lire les pages trouvées par wikipedia_search.py et les découper en chunks
       (petits passages), en gardant pour chacun d'où il vient : page, version
       de la page, rang dans la recherche. Le fichier produit est celui que lit
       la personne qui calcule les embeddings.

Entrée  : results/wikipedia_pages_2wiki.json
          une entrée par sous-question, avec la liste de ses pages
Sortie  : results/chunks_2wiki.json
          les mêmes entrées, avec "chunks" à la place de "pages"

Utilisation en console (depuis le dossier du projet) :
   python -m rag.build_chunks results/wikipedia_pages_2wiki.json
   python -m rag.build_chunks results/wikipedia_pages_2wiki.json -o results/chunks.json --chunk-size 400 --overlap 50

Utilisation depuis Python :
   from rag.build_chunks import chunk_pages_file
   items, bilan = chunk_pages_file("results/wikipedia_pages_2wiki.json")

Le découpage lui-même est fait par rag/chunker.py (par sections, puis paragraphes,
puis phrases, sans jamais couper une phrase). Ce fichier-ci ajoute la provenance.
"""

# ----------------------------------------------------------------------
# Imports : les bibliothèques dont on a besoin
# ----------------------------------------------------------------------
import argparse                           # lire les options tapées dans la console
import json                               # lire les pages, écrire les chunks
import time                               # mesure de la durée
from dataclasses import asdict, replace   # WikiChunk -> dictionnaire, copie avec champs changés
from pathlib import Path                  # chemins de fichiers

from rag.chunker import WikiChunk, chunk_wikipedia_page


# ----------------------------------------------------------------------
# Valeurs par défaut (celles de rag/chunker.py)
# ----------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 512   # taille visée d'un chunk, en caractères
DEFAULT_OVERLAP = 64       # caractères répétés entre deux chunks qui se suivent


# ======================================================================
# IDENTIFIANT D'UN CHUNK
# ======================================================================

def make_chunk_id(pageid, revid, chunk_size, overlap, chunk_index):
    """Construit l'identifiant unique d'un chunk.

    Exemple : "1001:900001:512/64:3"
        = page 1001, version 900001 de la page, découpage 512/64, 4e chunk de la page.

    L'identifiant change quand la page change de version (revid) ET quand on change
    les réglages de découpage : pour l'étude de la segmentation, deux découpages
    différents ne doivent jamais partager le même identifiant (cache, index FAISS).
    """
    rev = revid if revid is not None else "norev"
    page = pageid if pageid is not None else "nopage"
    return f"{page}:{rev}:{chunk_size}/{overlap}:{chunk_index}"


# ======================================================================
# UNE PAGE -> SES CHUNKS
# ======================================================================

def chunk_page(page, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Découpe UNE page (un dictionnaire de wikipedia_search.py) en chunks.

    Renvoie une liste de WikiChunk avec pageid, revid, url et chunk_id remplis.
    Renvoie [] si la page n'a pas de titre ou pas de texte.
    Le rang de la page (page_rank) est ajouté plus loin, par chunk_pages().
    """
    title = page.get("title")
    text = page.get("text")
    if not title or not text:
        return []

    chunks = chunk_wikipedia_page(title, text, chunk_size=chunk_size, overlap=overlap)
    pageid = page.get("pageid")
    revid = page.get("revid")
    url = page.get("url") or ""

    # replace() copie le chunk en changeant seulement les champs demandés
    return [
        replace(
            c,
            pageid=pageid,
            revid=revid,
            url=url,
            chunk_id=make_chunk_id(pageid, revid, chunk_size, overlap, c.chunk_index),
        )
        for c in chunks
    ]


# ======================================================================
# TOUTES LES SOUS-QUESTIONS -> LEURS CHUNKS
# ======================================================================

def chunk_pages(items, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Découpe les pages de toutes les sous-questions.

    items : le contenu du JSON de wikipedia_search.py (une liste d'entrées,
            chacune avec une clé "pages").
    Renvoie (résultats, bilan) :
      - résultats : une entrée par sous-question, avec "chunks" (liste de dictionnaires)
                    à la place de "pages", et "n_chunks"
      - bilan     : un dictionnaire de compteurs, affiché par la console
    """
    start = time.time()
    memo = {}   # (pageid, revid, titre) -> chunks : une page vue deux fois n'est découpée qu'une fois
    stats = {
        "sub_questions": 0,
        "sub_questions_without_chunks": 0,
        "pages_used": 0,
        "pages_skipped": 0,
        "chunks": 0,
    }
    total_chars = 0
    results = []

    for item in items:
        chunks = []
        for page in item.get("pages") or []:
            key = (page.get("pageid"), page.get("revid"), page.get("title"))
            if key not in memo:
                memo[key] = chunk_page(page, chunk_size, overlap)
            page_chunks = memo[key]

            if not page_chunks:
                stats["pages_skipped"] += 1      # page sans titre ou sans texte
                continue
            stats["pages_used"] += 1
            # Le rang dépend de la sous-question (la même page peut être 1re ici, 4e ailleurs)
            rank = page.get("rank")
            chunks.extend(replace(c, page_rank=rank) for c in page_chunks)

        # On garde tout ce que contenait l'entrée (question_id, sub_question...) sauf "pages"
        entry = {k: v for k, v in item.items() if k != "pages"}
        entry["n_chunks"] = len(chunks)
        entry["chunks"] = [asdict(c) for c in chunks]
        results.append(entry)

        stats["sub_questions"] += 1
        if not chunks:
            stats["sub_questions_without_chunks"] += 1
        stats["chunks"] += len(chunks)
        total_chars += sum(len(c.text) for c in chunks)

    stats["avg_chunk_chars"] = round(total_chars / stats["chunks"]) if stats["chunks"] else 0
    stats["seconds"] = round(time.time() - start, 2)
    return results, stats


# ======================================================================
# FICHIER -> FICHIER
# ======================================================================

def default_output_path(input_path):
    """results/wikipedia_pages_2wiki.json -> results/chunks_2wiki.json.

    Ne renvoie jamais le fichier d'entrée lui-même (on ne veut pas l'écraser).
    """
    input_path = Path(input_path)
    name = input_path.stem.replace("wikipedia_pages", "chunks") + ".json"
    output = input_path.with_name(name)
    if output == input_path:
        output = input_path.with_name(input_path.stem + "_chunks.json")
    return output


def chunk_pages_file(input_path, output_path=None,
                     chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Lit le JSON de pages, écrit le JSON de chunks. Renvoie (résultats, bilan).

    Lève ValueError si le fichier n'a pas la forme attendue (une liste d'entrées).
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else default_output_path(input_path)

    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(
            f"{input_path} : une liste d'entrées était attendue "
            f"(le fichier produit par wikipedia_search.py), pas un {type(items).__name__}"
        )

    results, stats = chunk_pages(items, chunk_size, overlap)

    output_path.parent.mkdir(parents=True, exist_ok=True)   # crée le dossier results/ si besoin
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    stats["output"] = str(output_path)
    return results, stats


# ======================================================================
# LANCEMENT EN CONSOLE
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Découpe en chunks les pages de wikipedia_search.py")
    parser.add_argument("input", help="JSON de pages (ex. results/wikipedia_pages_2wiki.json)")
    parser.add_argument("-o", "--output", default=None,
                        help="JSON de chunks à écrire (par défaut : results/chunks_2wiki.json)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"taille visée d'un chunk en caractères (défaut {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                        help=f"chevauchement entre chunks en caractères (défaut {DEFAULT_OVERLAP})")
    args = parser.parse_args()

    _, stats = chunk_pages_file(args.input, args.output, args.chunk_size, args.overlap)

    print("=" * 60)
    print("BILAN")
    print("=" * 60)
    print(f"Sous-questions traitées      : {stats['sub_questions']}")
    print(f"  sans aucun chunk           : {stats['sub_questions_without_chunks']}")
    print(f"Pages découpées              : {stats['pages_used']}")
    print(f"Pages ignorées (sans texte)  : {stats['pages_skipped']}")
    print(f"Chunks produits              : {stats['chunks']}  "
          f"(taille moyenne {stats['avg_chunk_chars']} caractères)")
    print(f"Réglages                     : chunk_size={args.chunk_size}, overlap={args.overlap}")
    print(f"Durée                        : {stats['seconds']} s")
    print(f"Résultats enregistrés dans   : {stats['output']}")


if __name__ == "__main__":
    main()
