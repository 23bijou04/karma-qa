"""
wikipedia_search.py  -  Recherche de pages Wikipédia pour le RAG de KARMA-Live

Rôle : pour une sous-question, envoyer des requêtes à l'API Wikipédia
       et renvoyer la liste des pages trouvées (top 5), avec leur texte.

API utilisée : MediaWiki Action API  ->  https://en.wikipedia.org/w/api.php
   - Requête 1 (recherche) : list=search      -> donne les TITRES des pages
   - Requête 2 (lecture)   : prop=extracts    -> donne le TEXTE d'une page
                             prop=revisions   -> donne le NUMÉRO DE VERSION de la page

Utilisation en console (depuis le dossier du projet) :
   python wikipedia_search.py        -> 50 premières sous-questions 2Wiki
   python wikipedia_search.py 5      -> 5 premières seulement (plus rapide)

Les pages trouvées sont enregistrées dans :  results/wikipedia_pages_2wiki.json
La console affiche seulement l'avancement et le bilan final.
"""

# ----------------------------------------------------------------------
# Imports : les bibliothèques dont on a besoin
# ----------------------------------------------------------------------
import json        # lire les sous-questions et écrire le fichier de résultats
import os          # créer le dossier de résultats
import re          # nettoyer les sous-questions, remplacer les [Q0]
import sys         # lire le nombre de sous-questions tapé dans la console
import time        # pauses entre les requêtes, mesure de la durée
from datetime import datetime, timezone   # date du téléchargement (provenance)
import requests    # envoyer les requêtes HTTP à Wikipédia (pip install requests)


# ----------------------------------------------------------------------
# Contact envoyé à Wikipédia dans le User-Agent.
# La règle de Wikimedia demande un moyen de contact : un mail OU l'adresse d'un site.
# On met l'adresse du projet sur GitHub : aucune donnée personnelle.
# Toutes les API de Wikipédia ont la même règle : changer d'API ne l'éviterait pas.
# ----------------------------------------------------------------------
PROJECT_URL = "https://github.com/GDD-Nantes/karma-qa"

# ----------------------------------------------------------------------
# Mots à retirer avant la recherche (mots interrogatifs et petits mots).
# Ils faussent les résultats : "Why did Katherine Mansfield die?"
# trouvait l'album "Why Should the Fire Die?".
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

        # Carte d'identité du programme : nom/version + contact (l'adresse du projet).
        self.session.headers["User-Agent"] = f"KarmaLive/1.0 ({PROJECT_URL}) python-requests"

        self.last_request = 0   # heure de la dernière requête envoyée

        # Compteurs pour l'évaluation (le sujet demande les appels HTTP et les échecs)
        self.n_requests = 0
        self.n_failures = 0
        self.errors = []        # messages d'erreur, pour les afficher et les enregistrer

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
                self.errors.append(f"page perdue : {title}")
                continue
            if not page["text"]:
                self.errors.append(f"page vide : {title}")
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

        On essaie plusieurs requêtes, de la plus précise à la plus large,
        et on garde la première qui donne des résultats :
          1. les mots-clés      "Who directed The Devil'S Hairpin?" -> "directed Devil Hairpin"
          2. les noms propres   (mots qui commencent par une majuscule) -> "Devil Hairpin"
          3. la phrase entière  (en dernier recours)
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
            "url": "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_"),
            "revid": revision.get("revid"),           # numéro de version -> cache et provenance
            "rev_timestamp": revision.get("timestamp"),   # date de cette version
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "text": page.get("extract", ""),          # le texte -> pour le découpage en passages
        }

    # ------------------------------------------------------------------
    # ENVOI D'UNE REQUÊTE (utilisé par les deux requêtes ci-dessus)
    # ------------------------------------------------------------------
    def _request(self, params):
        """Envoie une requête à l'API et renvoie la réponse JSON, ou None si tout échoue.

        Protections : pause entre les requêtes, délai maximum, nouvelles tentatives.
        Chaque échec est noté dans self.errors.
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

            self.errors.append(f"échec {attempt}/{self.max_retries} : {what} -> {problem}")

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
       1. "directed Devil Hairpin"             (mots-clés)
       2. "Devil Hairpin"                      (noms propres)
       3. "Who directed The Devil'S Hairpin?"  (phrase entière)
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
# LANCEMENT SUR LES SOUS-QUESTIONS 2WIKI (ne fait pas partie de la classe)
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
                sub_questions.append({
                    "question_id": question["question_id"],      # question 2Wiki d'origine
                    "question": question["question"],
                    "question_type": question["question_type"],
                    "sub_question": text,
                })
    return sub_questions


def save(path, results):
    """Enregistre les résultats dans un fichier JSON lisible (indenté)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    PATH_2WIKI = "data/eval_1000q/hybrid/hybrid_eval_1000_run2_judged.jsonl"   # entrée
    OUTPUT = "results/wikipedia_pages_2wiki.json"                               # sortie

    # Nombre de sous-questions : 50 par défaut (python wikipedia_search.py 5 -> 5)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)   # crée le dossier results/ si besoin
    searcher = WikipediaSearch()
    items = load_sub_questions(PATH_2WIKI)[:n]
    results = []

    print(f"Recherche Wikipédia pour {len(items)} sous-questions 2Wiki -> {OUTPUT}\n")
    start = time.time()

    for i, item in enumerate(items, start=1):
        t0 = time.time()
        errors_before = len(searcher.errors)

        pages = searcher.search_pages(item["sub_question"])

        # On garde la sous-question + ses pages dans les résultats
        results.append({
            **item,
            "search_query": pages[0]["search_query"] if pages else "",
            "n_pages": len(pages),
            "pages": [{k: v for k, v in p.items() if k not in ("query", "search_query")}
                      for p in pages],
        })
        # Enregistrement après CHAQUE sous-question : si le programme s'arrête,
        # le travail déjà fait n'est pas perdu.
        save(OUTPUT, results)

        # Avancement en console : une ligne par sous-question
        new_errors = len(searcher.errors) - errors_before
        status = "OK " if pages else "VIDE"
        warn = f"  ({new_errors} erreur(s))" if new_errors else ""
        print(f"[{i:>3}/{len(items)}] {status} {len(pages)} pages  "
              f"{time.time() - t0:4.1f} s  {item['sub_question'][:60]}{warn}")

    # ------------------------------------------------------------------
    # État final
    # ------------------------------------------------------------------
    duration = time.time() - start
    with_pages = sum(1 for r in results if r["n_pages"] > 0)
    total_pages = sum(r["n_pages"] for r in results)

    print("\n" + "=" * 60)
    print("BILAN")
    print("=" * 60)
    print(f"Sous-questions traitées     : {len(results)}")
    print(f"  avec au moins une page    : {with_pages}")
    print(f"  sans aucune page          : {len(results) - with_pages}")
    print(f"Pages récupérées            : {total_pages}  (max {len(results) * searcher.top_k})")
    print(f"Requêtes HTTP envoyées      : {searcher.n_requests}")
    print(f"Requêtes définitivement KO  : {searcher.n_failures}")
    print(f"Durée totale                : {duration:.0f} s  "
          f"({duration / max(len(results), 1):.1f} s par sous-question)")
    print(f"Résultats enregistrés dans  : {OUTPUT}")

    if searcher.errors:
        print(f"\nErreurs ({len(searcher.errors)}) :")
        for e in searcher.errors[:20]:          # les 20 premières seulement
            print("  -", e)
