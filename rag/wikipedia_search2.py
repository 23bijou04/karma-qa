"""Source de documents Wikipédia pour le RAG (implémente DocumentSource).

Lancement terminal (depuis la racine du projet) :
   python -m rag.wikipedia_search2 "Who was John Middleton Murry's wife?"

"""
import json
import sys
import time
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlencode
from urllib.request import Request, urlopen

from rag.config import MAX_PAGES               # nombre de pages par défaut
from rag.protocols import DocumentSource

API_URL = "https://en.wikipedia.org/w/api.php"
PAGE_URL = "https://en.wikipedia.org/wiki/"
HEADERS = {"User-Agent": "KarmaLive/1.0 (https://github.com/GDD-Nantes/karma-qa)"}  # demandé par Wikipédia


class WikipediaSearch(DocumentSource):
    """Cherche les pages Wikipédia d'une question et lit leur texte."""

    def __init__(self):
        self.revisions = {}     # versionnage : {url: {"title", "revid", "timestamp", "retrieved_at", "permalink"}}

    def search_pages(self, question: str, limit: int = MAX_PAGES) -> list[str]:
        """Cherche `limit` pages sur Wikipédia (MAX_PAGES par défaut) et renvoie leurs URL."""
        data = self._request({"list": "search", "srsearch": question,
                              "srlimit": limit, "srnamespace": 0})
        if not data:
            return []
        return [PAGE_URL + quote(r["title"].replace(" ", "_")) for r in data["query"]["search"]]

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
        url = API_URL + "?" + urlencode(params)          # construit l'adresse avec les paramètres
        for attempt in range(retries):
            try:
                with urlopen(Request(url, headers=HEADERS), timeout=15) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (OSError, ValueError):
                pass                            # pas de réseau, délai dépassé, erreur HTTP, réponse illisible...
            time.sleep(2 ** attempt)            # attente avant de réessayer : 1 s, 2 s, 4 s
        return None


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question : ")
    source = WikipediaSearch()
    urls = source.search_pages(question)

    # les pages trouvées par Wikipédia, avec leur version
    print(f"\nLes {len(urls)} pages trouvées sur Wikipédia\n")
    for i, url in enumerate(urls, start=1):
        source.fetch_page(url)
        v = source.revisions.get(url)
        if v:
            date = v["timestamp"].replace("T", " ").replace("Z", "")    # 2026-11-10T08:12:44Z -> 2026-11-10 08:12:44
            print(f"  {i:>2}. {v['title']} (version {v['revid']} du {date})\n      {url}\n")
        else:
            print(f"  {i:>2}. {url}\n")