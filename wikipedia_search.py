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
"""

# ----------------------------------------------------------------------
# Imports : les bibliothèques dont on a besoin
# ----------------------------------------------------------------------
import json        # lire le fichier de sous-questions (format JSON)
import re          # remplacer les [Q0] dans les sous-questions
import sys         # lire le nombre de sous-questions tapé dans la console
import time        # faire des pauses entre les requêtes
import requests    # envoyer les requêtes HTTP à Wikipédia (pip install requests)


# ======================================================================
# LA CLASSE
# ======================================================================

class WikipediaSearch:
    """Envoie des requêtes à l'API Wikipédia et renvoie la liste des pages."""

    # Adresse de l'API Wikipédia en anglais (2WikiMultiHopQA est en anglais).
    # C'est la partie avant le "?" des liens qu'on a testés dans le navigateur.
    API_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, top_k=5, delay=1.0, timeout=15, max_retries=3):
        """Prépare le chercheur. Appelé automatiquement quand on écrit WikipediaSearch().

        top_k       : nombre de pages à renvoyer par sous-question (5 d'après le sujet)
        delay       : pause minimum (en secondes) entre deux requêtes,
                      pour ne pas être bloqué par Wikipédia
        timeout     : temps maximum (en secondes) d'attente d'une réponse,
                      pour ne pas rester bloqué si Wikipédia ne répond pas
        max_retries : nombre de tentatives si une requête échoue
        """
        self.top_k = top_k
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries

        # Une "session" garde la connexion ouverte entre les requêtes (plus rapide)
        # et permet de fixer des réglages communs à toutes les requêtes.
        self.session = requests.Session()

        # Wikipédia demande que chaque programme se présente (User-Agent).
        # Sans ça, elle peut refuser les requêtes (erreur 403).
        self.session.headers["User-Agent"] = "KarmaLive/1.0 (projet universitaire)"

        # Heure de la dernière requête envoyée (0 = aucune pour l'instant).
        self.last_request = 0

    # ------------------------------------------------------------------
    # MÉTHODE PRINCIPALE : celle que le reste du programme appellera
    # ------------------------------------------------------------------
    def search_pages(self, sub_question):
        """Renvoie la liste des top_k pages Wikipédia pour une sous-question.

        Étapes :
          1. chercher les titres des pages (requête 1)
          2. pour chaque titre, lire la page (requête 2)
          3. garder les pages qui ont un texte, et noter leur rang

        Renvoie une liste de dictionnaires, par exemple :
          [{"title": "Ken Burns", "text": "...", "revid": 123, "rank": 1, ...}, ...]
        Renvoie une liste vide [] si rien n'est trouvé ou si Wikipédia ne répond pas.
        """
        pages = []

        # enumerate(..., start=1) donne le rang (1, 2, 3...) en même temps que le titre
        for rank, title in enumerate(self.search_titles(sub_question), start=1):
            page = self.get_page(title)

            # On ignore la page si elle n'existe plus (None) ou si son texte est vide
            if page and page["text"]:
                page["rank"] = rank              # position dans les résultats de la recherche
                page["query"] = sub_question     # provenance : quelle sous-question l'a trouvée
                pages.append(page)

        return pages

    # ------------------------------------------------------------------
    # REQUÊTE 1 : la recherche
    # ------------------------------------------------------------------
    def search_titles(self, sub_question):
        """Recherche par mots-clés. Renvoie la liste des TITRES des pages trouvées.

        Équivaut au lien testé dans le navigateur :
        https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=...&srlimit=5&format=json
        """
        # Sous-question vide -> inutile d'interroger Wikipédia
        if not sub_question.strip():
            return []

        # Les paramètres de l'URL, écrits sous forme de dictionnaire.
        # requests construit l'adresse complète tout seul.
        data = self._request({
            "action": "query",          # on interroge Wikipédia
            "list": "search",           # on fait une recherche
            "srsearch": sub_question,   # ce qu'on cherche
            "srlimit": self.top_k,      # combien de résultats (5)
            "srnamespace": 0,           # 0 = articles seulement (pas les pages "Discussion:", etc.)
        })

        # La requête a échoué -> aucune page
        if data is None:
            return []

        # La liste des résultats est dans  query -> search  (vu à l'étape 1).
        # Chaque résultat contient ns, title, pageid, size, wordcount, snippet, timestamp :
        # ici on ne garde que le titre.
        return [result["title"] for result in data["query"]["search"]]

    # ------------------------------------------------------------------
    # REQUÊTE 2 : la lecture d'une page
    # ------------------------------------------------------------------
    def get_page(self, title):
        """Lit une page à partir de son titre. Renvoie un dictionnaire, ou None si la page n'existe pas.

        Équivaut au lien testé dans le navigateur :
        https://en.wikipedia.org/w/api.php?action=query&prop=extracts|revisions&explaintext=1
            &redirects=1&rvprop=ids|timestamp&titles=Ken Burns&format=json&formatversion=2
        """
        data = self._request({
            "action": "query",
            "titles": title,                  # la page à lire
            "prop": "extracts|revisions",     # le texte ET les infos de version (le | sépare les deux)
            "explaintext": 1,                 # texte brut au lieu de HTML
            "redirects": 1,                   # suivre les redirections ("Zzzzzz" -> vraie page)
            "rvprop": "ids|timestamp",        # pour la version : son numéro et sa date
            # Pas de "exintro" : on veut tout le texte.
            # (Rappel : exintro=0 l'activerait quand même, il faut l'enlever complètement.)
        })

        # La requête a échoué
        if data is None:
            return None

        # Avec formatversion=2, "pages" est une LISTE. On n'a demandé qu'une page -> la première.
        page = data["query"]["pages"][0]

        # Page inexistante ("missing") ou titre interdit ("invalid") -> rien
        if page.get("missing") or page.get("invalid"):
            return None

        # Les infos de version sont dans une liste "revisions" ; on prend la première.
        # [{}] = valeur par défaut si jamais "revisions" est absent (évite une erreur).
        revision = page.get("revisions", [{}])[0]

        # On ne garde que les informations utiles (voir le tableau des champs)
        return {
            "title": page["title"],                   # vrai titre (après redirection)
            "pageid": page["pageid"],                 # numéro unique de la page
            "text": page.get("extract", ""),          # le texte -> pour le découpage en passages
            "revid": revision.get("revid"),           # numéro de version -> cache et provenance
            "timestamp": revision.get("timestamp"),   # date de cette version -> provenance
            "url": "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_"),
        }

    # ------------------------------------------------------------------
    # ENVOI D'UNE REQUÊTE (utilisé par les deux requêtes ci-dessus)
    # Le "_" au début du nom signifie : méthode interne, pas à appeler de l'extérieur.
    # ------------------------------------------------------------------
    def _request(self, params):
        """Envoie une requête à l'API et renvoie la réponse JSON (un dictionnaire).

        Protections (demandées dans le chantier 1 du sujet) :
          - pause entre deux requêtes (limitation de débit)
          - délai maximum d'attente (timeout)
          - nouvelles tentatives en cas d'échec, avec une attente de plus en plus longue
        Renvoie None si toutes les tentatives échouent : le programme ne plante pas.
        """
        # Paramètres communs à toutes les requêtes :
        #   format=json       -> réponse en JSON
        #   formatversion=2   -> JSON plus simple ("pages" est une liste, "missing": true)
        params.update({"format": "json", "formatversion": 2})

        for attempt in range(self.max_retries):        # attempt = 0, 1, 2

            # 1) Pause si la dernière requête est trop récente
            wait = self.delay - (time.time() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.time()

            # 2) Envoi de la requête
            try:
                response = self.session.get(self.API_URL, params=params, timeout=self.timeout)

                # 200 = réponse correcte. On vérifie aussi que Wikipédia ne signale pas
                # d'erreur dans le JSON (paramètre mal écrit, par exemple).
                if response.status_code == 200 and "error" not in response.json():
                    return response.json()

                # Sinon (429 = trop de requêtes, 403 = refusé, 500 = panne...) : on réessaie

            except (requests.RequestException, ValueError):
                # RequestException : pas de réseau, délai dépassé...
                # ValueError       : la réponse n'est pas du JSON
                pass   # on ne fait rien ici : on passe à la tentative suivante

            # 3) Attente avant de réessayer : 1 s, puis 2 s, puis 4 s
            time.sleep(2 ** attempt)

        # Toutes les tentatives ont échoué
        return None


# ======================================================================
# TESTS EN CONSOLE (ne fait pas partie de la classe)
# ======================================================================

def load_sub_questions(path, only_text=True):
    """Lit les sous-questions déjà générées par KARMA à partir des questions 2Wiki.

    path      : chemin du fichier de résultats du dépôt
    only_text : True = garder seulement les sous-questions résolues par le texte (le RAG),
                c'est-à-dire celles qui passeront par notre classe

    Dans le fichier, une sous-question peut contenir [Q0] = "la réponse de la sous-question 0".
    On le remplace par cette réponse :  "Where was [Q0] born?" -> "Where was Ken Burns born?"
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

                # [Q0], [Q1]... -> réponse correspondante
                text = re.sub(r"\[Q(\d+)\]",
                              lambda m: answers[int(m.group(1))],
                              s["sub_question"])
                sub_questions.append(text)
    return sub_questions


def show(searcher, sub_question):
    """Cherche les pages d'une sous-question et les affiche."""
    pages = searcher.search_pages(sub_question)
    print(f"\n{sub_question}  ->  {len(pages)} pages")
    for page in pages:
        print(f"   {page['rank']}. {page['title']}  (révision {page['revid']})  {page['url']}")


# Ce bloc ne s'exécute que si on lance CE fichier directement (python wikipedia_search.py),
# pas quand un autre fichier fait "from wikipedia_search import WikipediaSearch".
if __name__ == "__main__":
    # Fichier du dépôt qui contient les questions 2Wiki déjà découpées en sous-questions
    PATH_2WIKI = "data/eval_1000q/hybrid/hybrid_eval_1000_run2_judged.jsonl"

    # Nombre de sous-questions à tester : 50 par défaut.
    # sys.argv = ce qu'on a tapé dans la console.
    # Exemple : python wikipedia_search.py 5  ->  sys.argv = ["wikipedia_search.py", "5"]  ->  n = 5
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    searcher = WikipediaSearch()
    sub_questions = load_sub_questions(PATH_2WIKI)[:n]     # [:n] = les n premières
    print(f"{len(sub_questions)} sous-questions 2Wiki")

    for sub_question in sub_questions:
        show(searcher, sub_question)
