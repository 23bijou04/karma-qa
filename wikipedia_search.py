"""
wikipedia_search.py  -  Recherche de pages Wikipédia pour le RAG de KARMA-Live

Rôle : pour une sous-question, envoyer des requêtes à l'API Wikipédia
       et renvoyer la liste des pages trouvées (top 5), avec leur texte.

API utilisée : MediaWiki Action API  ->  https://en.wikipedia.org/w/api.php
   - Requête 1 (recherche) : list=search      -> donne les TITRES des pages
   - Requête 2 (lecture)   : prop=extracts    -> donne le TEXTE d'une page
                             prop=revisions   -> donne le NUMÉRO DE VERSION de la page

Utilisation en console (depuis le dossier du projet) :
   python wikipedia_search.py        -> teste les 50 premières sous-questions 2Wiki
   python wikipedia_search.py 5      -> teste seulement les 5 premières (plus rapide)

Avant de lancer, mets ton adresse mail (Wikipédia demande un contact) :
   export WIKI_UA_CONTACT="prenom.nom@etu.univ-nantes.fr"
"""

# ----------------------------------------------------------------------
# Imports : les bibliothèques dont on a besoin
# ----------------------------------------------------------------------
import json        # lire le fichier de sous-questions (format JSON)
import os          # lire la variable WIKI_UA_CONTACT
import re          # nettoyer les sous-questions, remplacer les [Q0]
import sys         # lire le nombre de sous-questions tapé dans la console
import time        # faire des pauses entre les requêtes
import requests    # envoyer les requêtes HTTP à Wikipédia (pip install requests)


# ----------------------------------------------------------------------
# Mots à retirer avant la recherche (mots interrogatifs et petits mots).
# Ils ne servent pas à trouver la page et faussent les résultats :
# "Why did Katherine Mansfield die?" trouvait l'album "Why Should the Fire Die?".
# ----------------------------------------------------------------------
STOPWORDS = {
    "who", "whom", "whose", "what", "when", "where", "which", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "has", "have", "had",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "and", "or", "that", "this", "it", "its", "as", "s",
}


# ======================================================================
# LA CLASSE
# ======================================================================

class WikipediaSearch:
    """Envoie des requêtes à l'API Wikipédia et renvoie la liste des pages."""

    # Adresse de l'API Wikipédia en anglais (2WikiMultiHopQA est en anglais).
    API_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, top_k=5, delay=1.0, timeout=15, max_retries=4):
        """Prépare le chercheur. Appelé automatiquement quand on écrit WikipediaSearch().

        top_k       : nombre de pages à renvoyer par sous-question (5 d'après le sujet)
        delay       : pause minimum (en secondes) entre deux requêtes
        timeout     : temps maximum (en secondes) d'attente d'une réponse
        max_retries : nombre de tentatives si une requête échoue
        """
        self.top_k = top_k
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries

        # Une "session" garde la connexion ouverte entre les requêtes (plus rapide).
        self.session = requests.Session()

        # CORRECTION : Wikipédia demande un User-Agent avec un CONTACT (mail ou site).
        # Sans contact, elle peut limiter ou refuser les requêtes (codes 403 / 429).
        contact = os.getenv("WIKI_UA_CONTACT", "projet-karma-live")
        self.session.headers["User-Agent"] = f"KarmaLive/1.0 ({contact}) python-requests"

        self.last_request = 0   # heure de la dernière requête envoyée

        # Compteurs pour l'évaluation (le sujet demande les appels HTTP et les échecs)
        self.n_requests = 0
        self.n_failures = 0

    # ------------------------------------------------------------------
    # MÉTHODE PRINCIPALE : celle que le reste du programme appellera
    # ------------------------------------------------------------------
    def search_pages(self, sub_question):
        """Renvoie la liste des top_k pages Wikipédia pour une sous-question.

        Étapes :
          1. chercher les titres des pages (requête 1), avec des mots-clés
          2. pour chaque titre, lire la page (requête 2)
          3. garder les pages qui ont un texte, et noter leur rang

        Renvoie une liste de dictionnaires, ou [] si rien n'est trouvé.
        """
        titles, search_query = self.search_titles(sub_question)

        pages = []
        for rank, title in enumerate(titles, start=1):
            page = self.get_page(title)
            if page is None:
                # CORRECTION : on affiche la page perdue au lieu de la cacher
                print(f"   [page perdue] {title}")
                continue
            if not page["text"]:
                print(f"   [page vide] {title}")
                continue
            page["rank"] = rank                    # position dans les résultats
            page["query"] = sub_question           # provenance : la sous-question d'origine
            page["search_query"] = search_query    # provenance : ce qu'on a vraiment cherché
            pages.append(page)
        return pages

    # ------------------------------------------------------------------
    # REQUÊTE 1 : la recherche
    # ------------------------------------------------------------------
    def search_titles(self, sub_question):
        """Recherche par mots-clés. Renvoie (liste des titres, requête utilisée).

        CORRECTION : on n'envoie plus la sous-question telle quelle.
        On essaie plusieurs requêtes, de la plus précise à la plus large,
        et on garde la première qui donne des résultats :
          1. les mots-clés         "Who directed The Devil'S Hairpin?" -> "directed Devil Hairpin"
          2. les noms propres      (mots qui commencent par une majuscule) -> "Devil Hairpin"
          3. la phrase entière     (en dernier recours)
        """
        if not sub_question.strip():
            return [], ""

        for query in candidate_queries(sub_question):
            data = self._request({
                "action": "query",
                "list": "search",
                "srsearch": query,        # ce qu'on cherche
                "srlimit": self.top_k,    # combien de résultats (5)
                "srnamespace": 0,         # 0 = articles seulement
            })
            if data is None:
                continue                  # requête échouée -> on essaie la suivante
            titles = [result["title"] for result in data["query"]["search"]]
            if titles:
                return titles, query
        return [], ""

    # ------------------------------------------------------------------
    # REQUÊTE 2 : la lecture d'une page
    # ------------------------------------------------------------------
    def get_page(self, title):
        """Lit une page à partir de son titre. Renvoie un dictionnaire, ou None."""
        data = self._request({
            "action": "query",
            "titles": title,                  # la page à lire
            "prop": "extracts|revisions",     # le texte ET les infos de version
            "explaintext": 1,                 # texte brut au lieu de HTML
            "redirects": 1,                   # suivre les redirections
            "rvprop": "ids|timestamp",        # numéro et date de la version
            # Pas de "exintro" : on veut tout le texte (exintro=0 l'activerait quand même).
        })
        if data is None:
            return None

        # Avec formatversion=2, "pages" est une LISTE. On n'a demandé qu'une page.
        page = data["query"]["pages"][0]

        # Page inexistante ou titre interdit
        if page.get("missing") or page.get("invalid"):
            return None

        revision = page.get("revisions", [{}])[0]
        return {
            "title": page["title"],                   # vrai titre (après redirection)
            "pageid": page["pageid"],                 # numéro unique de la page
            "text": page.get("extract", ""),          # le texte -> pour le découpage en passages
            "revid": revision.get("revid"),           # numéro de version -> cache et provenance
            "timestamp": revision.get("timestamp"),   # date de cette version
            "url": "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_"),
        }

    # ------------------------------------------------------------------
    # ENVOI D'UNE REQUÊTE (utilisé par les deux requêtes ci-dessus)
    # ------------------------------------------------------------------
    def _request(self, params):
        """Envoie une requête à l'API et renvoie la réponse JSON, ou None si tout échoue.

        Protections : pause entre les requêtes, délai maximum, nouvelles tentatives.
        CORRECTION : chaque échec est affiché (avant, il était caché).
        """
        params = {**params, "format": "json", "formatversion": 2}
        what = params.get("titles") or params.get("srsearch")   # pour les messages

        for attempt in range(1, self.max_retries + 1):          # 1, 2, 3, 4

            # 1) Pause si la dernière requête est trop récente
            wait = self.delay - (time.time() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.time()
            self.n_requests += 1

            # 2) Envoi de la requête
            retry_after = None
            try:
                response = self.session.get(self.API_URL, params=params, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    if "error" not in data:
                        return data                                   # succès
                    problem = "erreur API : " + data["error"].get("info", "?")
                else:
                    # 429 = trop de requêtes, 403 = refusé, 5xx = panne de Wikipédia
                    problem = f"code {response.status_code}"
                    retry_after = response.headers.get("Retry-After")
            except requests.RequestException as e:
                problem = f"pas de réponse ({type(e).__name__})"   # réseau, délai dépassé...
            except ValueError:
                problem = "réponse illisible (pas du JSON)"

            print(f"   [échec {attempt}/{self.max_retries}] {what} -> {problem}")

            # 3) Attente avant de réessayer : 2 s, 4 s, 8 s...
            #    (ou la durée demandée par Wikipédia dans Retry-After)
            if attempt < self.max_retries:
                pause = 2 ** attempt
                if retry_after and retry_after.isdigit():
                    pause = max(pause, int(retry_after))
                time.sleep(pause)

        self.n_failures += 1
        return None


# ======================================================================
# PRÉPARATION DE LA REQUÊTE DE RECHERCHE
# ======================================================================

def candidate_queries(sub_question):
    """Renvoie les requêtes à essayer, de la plus précise à la plus large (sans doublon).

    "Who directed The Devil'S Hairpin?" donne :
       1. "directed Devil Hairpin"          (mots-clés)
       2. "Devil Hairpin"                   (noms propres)
       3. "Who directed The Devil'S Hairpin?" (phrase entière)
    """
    # Enlever les possessifs ('s) puis tout ce qui n'est pas une lettre, un chiffre ou un tiret
    text = re.sub(r"['’][sS]\b", "", sub_question)
    words = re.findall(r"[\w-]+", text)

    keywords = [w for w in words if w.lower() not in STOPWORDS]
    names = [w for w in keywords if w[0].isupper() or w[0].isdigit()]

    queries = []
    for q in (" ".join(keywords), " ".join(names), sub_question.strip()):
        if q and q not in queries:
            queries.append(q)
    return queries


# ======================================================================
# TESTS EN CONSOLE (ne fait pas partie de la classe)
# ======================================================================

def load_sub_questions(path, only_text=True):
    """Lit les sous-questions déjà générées par KARMA à partir des questions 2Wiki.

    only_text=True : garder seulement celles résolues par le texte (le RAG).
    Les [Q0] sont remplacés par la réponse de la sous-question 0 :
       "Where was [Q0] born?" -> "Where was Ken Burns born?"
    """
    sub_questions = []
    with open(path) as f:
        for line in f:                                     # une ligne = une question 2Wiki
            question = json.loads(line)
            log = question.get("sub_question_log") or []   # ses sous-questions
            answers = [s.get("answer") or "" for s in log] # leurs réponses, dans l'ordre
            for s in log:
                if only_text and s["method"] != "text":
                    continue                                # résolue par DBpedia -> on saute
                text = re.sub(r"\[Q(\d+)\]",
                              lambda m: answers[int(m.group(1))],
                              s["sub_question"])
                sub_questions.append(text)
    return sub_questions


def show(searcher, sub_question):
    """Cherche les pages d'une sous-question et les affiche."""
    print(f"\n{sub_question}")
    pages = searcher.search_pages(sub_question)
    query = pages[0]["search_query"] if pages else "-"
    print(f"   recherche : \"{query}\"  ->  {len(pages)} pages")
    for page in pages:
        print(f"   {page['rank']}. {page['title']}  (révision {page['revid']})  {page['url']}")


if __name__ == "__main__":
    # Fichier du dépôt qui contient les questions 2Wiki déjà découpées en sous-questions
    PATH_2WIKI = "data/eval_1000q/hybrid/hybrid_eval_1000_run2_judged.jsonl"

    # Nombre de sous-questions à tester : 50 par défaut (python wikipedia_search.py 5 -> 5)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    searcher = WikipediaSearch()
    sub_questions = load_sub_questions(PATH_2WIKI)[:n]
    print(f"{len(sub_questions)} sous-questions 2Wiki")

    start = time.time()
    for sub_question in sub_questions:
        show(searcher, sub_question)

    # Bilan : utile pour le rapport (appels HTTP, échecs, durée)
    print(f"\nBilan : {searcher.n_requests} requêtes, {searcher.n_failures} échecs, "
          f"{time.time() - start:.0f} s")
