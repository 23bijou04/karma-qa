"""Wikipedia search + redirect resolution for entity mention fallback.

Standalone module: only depends on httpx (no circular imports with dbpedia/).
Used by DBpediaEntityLinker as Layer 3 fallback after URI probe + bif:contains.
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_RATE_LIMIT_S = 0.5  # seconds between Wikipedia API calls

# Map dbo: type hints to Wikipedia disambiguation suffixes
_TYPE_SUFFIX_MAP: dict[str, str] = {
    "dbo:Film": "film",
    "dbo:Book": "book",
    "dbo:MusicalWork": "album",
    "dbo:Country": "country",
    "dbo:Company": "company",
    "dbo:Person": "person",
    "dbo:MusicalArtist": "musician",
    "dbo:City": "city",
}


class WikipediaResolver:
    """Resolve entity mentions via Wikipedia search + redirect API.

    Usage:
        resolver = WikipediaResolver()
        uris = resolver.resolve_to_dbpedia_uris("The Return of the Hero", "dbo:Film")
        # -> ["http://dbpedia.org/resource/Le_Retour_du_h%C3%A9ros", ...]
    """

    def __init__(self, cache: dict | None = None):
        self._cache: dict[str, list[str]] = cache if cache is not None else {}
        self._last_call: float = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _RATE_LIMIT_S:
            time.sleep(_RATE_LIMIT_S - elapsed)
        self._last_call = time.monotonic()

    def search_candidates(self, mention: str, type_suffix: str = "") -> list[str]:
        """Search Wikipedia for a mention, return up to 3 page titles.

        Args:
            mention: Entity mention string.
            type_suffix: Optional disambiguation word (e.g. "film", "book").

        Returns:
            List of Wikipedia page titles (may be empty on failure).
        """
        query = f"{mention} {type_suffix}".strip()
        if query in self._cache:
            return self._cache[query]

        try:
            self._rate_limit()
            resp = httpx.get(
                _WIKI_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 3,
                    "format": "json",
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
            titles = [r["title"] for r in results]
        except Exception as e:
            logger.warning(f"Wikipedia search failed for '{query}': {e}")
            titles = []

        self._cache[query] = titles
        return titles

    def resolve_redirect(self, title: str) -> str | None:
        """Follow Wikipedia redirects, return canonical page title.

        Args:
            title: Wikipedia page title to resolve.

        Returns:
            Canonical title, or None if the page does not exist.
        """
        try:
            self._rate_limit()
            resp = httpx.get(
                _WIKI_API,
                params={
                    "action": "query",
                    "titles": title,
                    "redirects": "true",
                    "format": "json",
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json().get("query", {})
            redirects = data.get("redirects", [])
            if redirects:
                return redirects[-1]["to"]
            pages = data.get("pages", {})
            for pid, page in pages.items():
                if pid != "-1":
                    return page.get("title")
        except Exception as e:
            logger.warning(f"Wikipedia redirect resolution failed for '{title}': {e}")
        return None

    def resolve_to_dbpedia_uris(
        self, mention: str, type_hint: str | None
    ) -> list[str]:
        """Search + redirect-resolve a mention to candidate DBpedia URIs.

        Performs at most 2 Wikipedia API calls (search + one redirect check
        per result uses the cache from search_candidates).

        Args:
            mention: Entity mention string.
            type_hint: dbo: type string for disambiguation (e.g. "dbo:Film").

        Returns:
            List of candidate DBpedia resource URIs (unvalidated).
        """
        type_suffix = _TYPE_SUFFIX_MAP.get(type_hint or "", "")
        titles = self.search_candidates(mention, type_suffix)

        uris: list[str] = []
        seen: set[str] = set()
        for title in titles:
            canonical = self.resolve_redirect(title) or title
            local = canonical.strip().replace(" ", "_")
            if not local:
                continue
            # Ensure first letter is upper-cased (DBpedia convention)
            if local[0].islower():
                local = local[0].upper() + local[1:]
            uri = f"http://dbpedia.org/resource/{local}"
            if uri not in seen:
                seen.add(uri)
                uris.append(uri)

        return uris
