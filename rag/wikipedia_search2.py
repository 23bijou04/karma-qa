"""Source de documents Wikipédia pour le RAG (implémente DocumentSource).

Installation (une seule fois) :
   pip install requests numpy

Lancement terminal (depuis la racine du projet) :
   python -m rag.wikipedia_search2 "Who was John Middleton Murry's wife?"

Ou exécuter le fichier avec le bouton Run et donner la question directement dans le terminal

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
        self.found = []                             # les pages trouvées (pour l'affichage)
        data = self._request({"list": "search", "srsearch": question,
                              "srlimit": max(limit, MAX_PAGES), "srnamespace": 0})
        if not data:
            return []
        self.found = data["query"]["search"]        # [{"title": ..., "snippet": ...}, ...]
        return [PAGE_URL + quote(r["title"].replace(" ", "_")) for r in self.found[:limit]]

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

    # les 10 pages trouvées par Wikipédia
    print(f"\n{len(source.found)} Les pages trouvées sur Wikipédia (✓ = gardée)")
    for i, r in enumerate(source.found, start=1):
        kept = "✓" if i <= len(urls) else " "
        print(f"  {kept:2}{i:>2}. {r['title']}")

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