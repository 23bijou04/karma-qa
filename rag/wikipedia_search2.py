"""Source de documents Wikipédia pour le RAG (implémente DocumentSource).

Installation (une seule fois) :
   pip install requests numpy              # nécessaire
   pip install sentence-transformers       # seulement pour le rerank

Lancement terminal (depuis la racine du projet) :
   python -m rag.wikipedia_search2 "Who was John Middleton Murry's wife?"

Ou exécuter le fichier avec le bouton Run et donner la question directement dans le terminal

RERANK (optionnel) : reclasser les 10 pages avant de garder les 5 meilleures. Deux versions :
   A : sur le snippet (l'extrait renvoyé par la recherche)
   B : sur l'introduction de la page (une requête de plus)
Désactivé par défaut. Pour l'activer, décommenter la méthode _rerank (RERANK 2/2)
et UNE des deux lignes RERANK A ou RERANK B (raccourci clavier : ctrl + :).

VERSIONNAGE : à chaque lecture d'une page (fetch_page), la classe note dans self.revisions
le numéro de version de la page (revid), sa date et un lien permanent vers cette version
"""
import sys
import time
from datetime import datetime, timezone
from urllib.parse import quote, unquote

import requests

from rag.config import MAX_PAGES               # nombre de pages cherchées (10)
from rag.protocols import DocumentSource

API_URL = "https://en.wikipedia.org/w/api.php"
PAGE_URL = "https://en.wikipedia.org/wiki/"
HEADERS = {"User-Agent": "KarmaLive/1.0 (https://github.com/GDD-Nantes/karma-qa)"}  # demandé par Wikipédia


class WikipediaSearch(DocumentSource):
    """Cherche les pages Wikipédia d'une question et lit leur texte."""

    def __init__(self):
        self.revisions = {}     # versionnage : {url: {"title", "revid", "timestamp", "retrieved_at", "permalink"}}

    def search_pages(self, question: str, limit: int = 5) -> list[str]:
        """Cherche MAX_PAGES pages (10) sur Wikipédia et renvoie les URL des `limit` premières (5)."""
        data = self._request({"list": "search", "srsearch": question,
                              "srlimit": MAX_PAGES, "srnamespace": 0})
        if not data:
            return []
        results = data["query"]["search"]           # [{"title": ..., "snippet": ...}, ...]
        for i, r in enumerate(results, start=1):
            r["wiki_rank"] = i                      # position donnée par Wikipédia
        self.before = list(results)                 # pour l'affichage : avant le rerank
        # RERANK (1/2) : décommenter UNE seule des deux lignes
        # results = self._rerank(question, results, use_intro=False)   # RERANK A : sur les snippets
        # results = self._rerank(question, results, use_intro=True)    # RERANK B : sur l'introduction
        self.after = results                        # pour l'affichage : après le rerank
        return [PAGE_URL + quote(r["title"].replace(" ", "_")) for r in results[:limit]]

    def fetch_page(self, url: str) -> str:
        """Renvoie le texte brut de la page désignée par son URL ("" si introuvable).

        Note aussi la version lue dans self.revisions[url] (versionnage).
        """
        title = unquote(url.split("/wiki/")[-1]).replace("_", " ")
        data = self._request({"prop": "extracts|revisions",   # le texte ET la version de la page
                              "explaintext": 1, "redirects": 1,
                              "rvprop": "ids|timestamp",       # numéro et date de la version
                              "titles": title})
        if not data:
            return ""
        page = data["query"]["pages"][0]
        if page.get("missing") or page.get("invalid"):
            return ""

        revision = page.get("revisions", [{}])[0]
        self.revisions[url] = {
            "title": page["title"],                              # vrai titre (après redirection)
            "revid": revision.get("revid"),                      # numéro de la version lue
            "timestamp": revision.get("timestamp"),              # date de cette version
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # date de lecture
            "permalink": f"https://en.wikipedia.org/w/index.php?oldid={revision.get('revid')}",  # lien vers CETTE version
        }
        return page.get("extract", "")

    # RERANK (2/2) : la méthode qui note et reclasse les pages
    # def _rerank(self, question, results, use_intro=False):
    #     """Note chaque page avec un cross-encoder et les trie de la meilleure à la moins bonne.
    #     use_intro=False : le modèle lit titre + snippet       (RERANK A)
    #     use_intro=True  : le modèle lit titre + introduction  (RERANK B)
    #     """
    #     import re
    #     from html import unescape
    #     from sentence_transformers import CrossEncoder
    #     if not hasattr(self, "model"):
    #         self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")   # même modèle que rag/reranker.py
    #     if use_intro:
    #         # une requête pour les introductions des 10 pages (3 premières phrases)
    #         data = self._request({"prop": "extracts", "exintro": 1, "explaintext": 1, "exsentences": 3,
    #                               "exlimit": len(results), "titles": "|".join(r["title"] for r in results)})
    #         intros = {p["title"]: p.get("extract", "") for p in (data or {}).get("query", {}).get("pages", [])}
    #         texts = [intros.get(r["title"], "") for r in results]
    #     else:
    #         texts = [unescape(re.sub(r"<[^>]+>", "", r["snippet"])) for r in results]
    #     pairs = [(question, r["title"] + ". " + text) for r, text in zip(results, texts)]
    #     for r, score in zip(results, self.model.predict(pairs)):
    #         r["score"] = float(score)               # la note de la page
    #     return sorted(results, key=lambda r: r["score"], reverse=True)

    def _request(self, params, retries=3):
        """Interroge l'API ; réessaie en cas d'erreur ; renvoie le JSON ou None."""
        params = {"action": "query", "format": "json", "formatversion": 2, **params}
        for attempt in range(retries):
            try:
                r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
                if r.status_code == 200:
                    return r.json()
            except requests.RequestException:
                pass                            # pas de réseau, délai dépassé...
            time.sleep(2 ** attempt)            # attente avant de réessayer : 1 s, 2 s, 4 s
        return None


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question : ")
    source = WikipediaSearch()
    urls = source.search_pages(question)
    reranked = any("score" in r for r in source.after)    # le rerank a-t-il été fait ?

    # les 10 pages trouvées par Wikipédia
    print(f"\n{len(source.before)} Les pages trouvées sur Wikipédia" + ("" if reranked else " (✓ = gardée)"))
    for r in source.before:
        kept = "✓" if not reranked and r["wiki_rank"] <= len(urls) else " "
        print(f"  {kept:2}{r['wiki_rank']:>2}. {r['title']}")

    # Avec rerank
    if reranked:
        print("\nRERANK (✓ = gardée)")
        print(f"  {'':2}{'rang':>4}  {'note':>6}  {'avant':>5}  page")
        for i, r in enumerate(source.after, start=1):
            kept = "✓" if i <= len(urls) else " "
            print(f"  {kept:2}{i:>4}  {r['score']:6.2f}  {r['wiki_rank']:>5}  {r['title']}")

    # URL + versionnage : on lit chaque page gardée et on affiche sa version
    print("\nURL :")
    for url in urls:
        source.fetch_page(url)
        v = source.revisions.get(url)
        if v:
            date = v["timestamp"].replace("T", " ").replace("Z", "")    # 2026-11-10T08:12:44Z -> 2026-11-10 08:12:44
            print(f"  {v['title']} (version {v['revid']} du {date})\n     {url}")
        else:
            print(f"  {url}")