"""Source de documents Wikipédia pour le RAG (implémente DocumentSource).

Lancement terminal (depuis la racine du projet) :
   python -m rag.infrastructure.wikipedia_search2 "Who was John Middleton Murry's wife?"
   python -m rag.infrastructure.wikipedia_search2 "Qui était la femme de John Middleton Murry ?"

La langue de la question est repérée par l'API de Wikimedia (langid) : on cherche dans la Wikipédia de cette langue.
"""
import json
import sys
import time
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlencode
from urllib.request import Request, urlopen

from rag.application.ports import DocumentSource   # l'interface implémentée
from rag.config import MAX_PAGES                    # nombre de pages par défaut
from rag.domain.models import Document              # ce que renvoie fetch_page

HEADERS = {"User-Agent": "KarmaLive/1.0 (https://github.com/GDD-Nantes/karma-qa)"}  # demandé par Wikipédia

LANGID_URL = "https://api.wikimedia.org/service/lw/inference/v1/models/langid:predict"  # repère la langue


class WikipediaSearch(DocumentSource):
    """Cherche les pages Wikipédia d'une question, dans la langue de la question, et lit leur texte."""

    def __init__(self):
        self.revisions = {}     # versionnage : {url: {"title", "revid", "timestamp", "retrieved_at", "permalink"}}
        self.ranks = {}         # rang de chaque page dans la recherche : {url: 1, 2, 3...}

    def search_pages(self, question: str, limit: int = MAX_PAGES) -> list[str]:
        """Repère la langue de la question, cherche `limit` pages dans cette Wikipédia et renvoie leurs URL."""
        lang = detect_language(question)                # "en", "fr", "ru"...
        data = self._request(lang, {"list": "search", "srsearch": question,
                                    "srlimit": limit, "srnamespace": 0})
        if not data:
            return []

        urls = []                                       # liste vide, remplie dans la boucle
        for result in data["query"]["search"]:          # chaque page trouvée par Wikipédia
            title = result["title"].replace(" ", "_")   # "John Middleton Murry" -> "John_Middleton_Murry"
            url = f"https://{lang}.wikipedia.org/wiki/" + quote(title)
            urls.append(url)                            # ajoute l'URL à la fin de la liste

        for rank, url in enumerate(urls, start=1):
            self.ranks[url] = rank              # mémorisé pour le page_rank du Document
        return urls

    def fetch_page(self, url: str) -> Document:
        """Renvoie la page désignée par son URL sous forme de Document (texte vide si introuvable).

        La langue est lue dans l'URL (fr.wikipedia.org -> "fr").
        Note aussi la version lue dans self.revisions[url] (versionnage).
        """
        lang = url.split("//")[-1].split(".")[0]        # "https://fr.wikipedia.org/wiki/..." -> "fr"
        end = url.split("/wiki/")[-1]                   # la fin de l'URL : "Fran%C3%A7ois_Mitterrand"
        end = unquote(end)                              # décode les caractères spéciaux : "François_Mitterrand"
        title = end.replace("_", " ")                   # remet les espaces : "François Mitterrand"

        data = self._request(lang, {"prop": "extracts|revisions",   # le texte ET la version de la page
                                    "explaintext": 1, "redirects": 1,
                                    "rvprop": "ids|timestamp",       # numéro et date de la version
                                    "titles": title})
        if not data:
            return Document(url=url, page_title=title, text="", page_rank=self.ranks.get(url))
        page = data["query"]["pages"][0]
        if page.get("missing") or page.get("invalid"):
            return Document(url=url, page_title=title, text="", page_rank=self.ranks.get(url))

        revision = page.get("revisions", [{}])[0]
        self.revisions[url] = {
            "title": page["title"],                              # vrai titre (après redirection)
            "revid": revision.get("revid"),                      # numéro de la version lue
            "timestamp": revision.get("timestamp"),              # date de cette version
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # date de lecture
            "permalink": f"https://{lang}.wikipedia.org/w/index.php?oldid={revision.get('revid')}",  # lien vers CETTE version
        }
        return Document(
            url=url,
            page_title=page["title"],                            # vrai titre (après redirection)
            text=page.get("extract", ""),                        # le texte -> pour le découpage
            page_id=page.get("pageid"),                          # identifiant de la page Wikipédia
            revision_id=revision.get("revid"),                   # numéro de la révision lue
            page_rank=self.ranks.get(url),                       # position dans la recherche
        )

    def _request(self, lang, params, retries=3):
        """Interroge l'API de la Wikipédia `lang` ; réessaie en cas d'erreur ; renvoie le JSON ou None."""
        params = {"action": "query", "format": "json", "formatversion": 2, **params}
        url = f"https://{lang}.wikipedia.org/w/api.php?" + urlencode(params)   # adresse avec les paramètres
        for attempt in range(retries):
            try:
                with urlopen(Request(url, headers=HEADERS), timeout=15) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (OSError, ValueError):
                pass                            # pas de réseau, délai dépassé, erreur HTTP, réponse illisible...
            time.sleep(2 ** attempt)            # attente avant de réessayer : 1 s, 2 s, 4 s
        return None


def detect_language(question):
    """Demande à Wikimedia la langue de la question ("en", "fr", "ru"...). Anglais en cas d'échec."""
    body = json.dumps({"text": question}).encode("utf-8")              # {"text": "Qui était ..."}
    headers = {**HEADERS, "Content-Type": "application/json"}
    try:
        with urlopen(Request(LANGID_URL, data=body, headers=headers), timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))["wikicode"]   # "fr"
    except (OSError, ValueError, KeyError):
        return "en"


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question : ")
    source = WikipediaSearch()
    urls = source.search_pages(question)

    print(f"\nLangue repérée : {detect_language(question)}")

    # les pages trouvées par Wikipédia, avec leur version
    print(f"\nLes {len(urls)} pages trouvées sur Wikipédia\n")
    for url in urls:
        doc = source.fetch_page(url)
        v = source.revisions.get(url)
        if v:
            date = v["timestamp"].replace("T", " ").replace("Z", "")    # 2026-11-10T08:12:44Z -> 2026-11-10 08:12:44
            print(f"  {doc.page_rank:>2}. {doc.page_title} (version {doc.revision_id} du {date})\n      {url}\n")
        else:
            print(f"  {doc.page_rank:>2}. {url}\n")