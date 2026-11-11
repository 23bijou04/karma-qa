"""Source de documents Wikipédia pour le RAG (implémente DocumentSource).

Lancement : python -m rag.wikipedia_search2 "Who was John Middleton Murry's wife?"

"""
import re
import sys
import time
from html import unescape
from urllib.parse import quote, unquote

import requests

from rag.config import MAX_PAGES               # nombre de pages candidates (10)
from rag.protocols import DocumentSource

API_URL = "https://en.wikipedia.org/w/api.php"
PAGE_URL = "https://en.wikipedia.org/wiki/"
HEADERS = {"User-Agent": "KarmaLive/1.0 (https://github.com/GDD-Nantes/karma-qa)"}  # demandé par Wikipédia
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # même modèle que rag/reranker.py


class WikipediaSearch(DocumentSource):
    """Cherche les pages Wikipédia d'une question et lit leur texte."""

    def __init__(self, use_rerank=True):
        self.use_rerank = use_rerank
        self.model = None                       # chargé seulement au premier rerank

    def search_pages(self, question: str, limit: int = 5) -> list[str]:
        """Cherche MAX_PAGES pages, les reclasse sur leurs snippets, renvoie les URL des `limit` meilleures."""
        data = self._request({"list": "search", "srsearch": question,
                              "srlimit": MAX_PAGES, "srnamespace": 0})
        if not data:
            return []
        results = data["query"]["search"]       # [{"title": ..., "snippet": ...}, ...]

        if self.use_rerank and len(results) > 1:
            results = self._rerank(question, results)

        return [PAGE_URL + quote(r["title"].replace(" ", "_")) for r in results[:limit]]

    def fetch_page(self, url: str) -> str:
        """Renvoie le texte brut de la page désignée par son URL ("" si introuvable)."""
        title = unquote(url.split("/wiki/")[-1]).replace("_", " ")
        data = self._request({"prop": "extracts", "explaintext": 1,
                              "redirects": 1, "titles": title})
        if not data:
            return ""
        return data["query"]["pages"][0].get("extract", "")

    def _rerank(self, question, results):
        """Note chaque page (titre + snippet) avec un cross-encoder et les trie de la meilleure à la moins bonne."""
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(RERANK_MODEL)
        pairs = [(question, r["title"] + ". " + unescape(re.sub(r"<[^>]+>", "", r["snippet"])))
                 for r in results]
        scores = self.model.predict(pairs)
        return [r for _, r in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)]

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
    for url in WikipediaSearch().search_pages(question):
        print(url)
