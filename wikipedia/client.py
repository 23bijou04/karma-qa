"""Wikipedia API client for text retrieval."""

import logging
import re
from typing import Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..dbpedia.cache import CacheManager

logger = logging.getLogger(__name__)


class WikipediaClient:
    """Client for retrieving Wikipedia article content.

    Uses the Wikipedia API to fetch article text that contains
    the facts extracted into DBpedia. This is the source text
    used for QA context in 2WikiMultiHopQA format.
    """

    API_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        timeout: int = 30,
    ):
        """Initialize Wikipedia client.

        Args:
            cache_manager: Optional cache manager for results
            timeout: Request timeout in seconds
        """
        self.cache = cache_manager or CacheManager()
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()

    def dbpedia_uri_to_title(self, uri: str) -> str:
        """Convert DBpedia URI to Wikipedia article title.

        Args:
            uri: DBpedia resource URI

        Returns:
            Wikipedia article title
        """
        if "dbpedia.org/resource/" in uri:
            title = uri.split("/resource/")[-1]
        elif "/" in uri:
            title = uri.split("/")[-1]
        else:
            title = uri

        return title.replace("_", " ")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _api_request(self, params: dict) -> dict:
        """Make request to Wikipedia API.

        Args:
            params: API parameters

        Returns:
            API response as dict
        """
        params["format"] = "json"

        response = self._client.get(self.API_URL, params=params)
        response.raise_for_status()
        return response.json()

    def get_summary(self, title: str, sentences: int = 5) -> Optional[str]:
        """Get article summary (first N sentences).

        This is typically the text that contains key infobox-derived facts.

        Args:
            title: Wikipedia article title
            sentences: Number of sentences to retrieve

        Returns:
            Article summary text or None
        """
        cache_key = f"wiki_summary:{title}:{sentences}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exsentences": sentences,
            "redirects": 1,
        }

        try:
            result = self._api_request(params)
            pages = result.get("query", {}).get("pages", {})

            for page_id, page_data in pages.items():
                if page_id != "-1":
                    summary = page_data.get("extract", "")
                    if summary:
                        self.cache.set(cache_key, summary)
                        return summary

        except Exception as e:
            logger.warning(f"Failed to get summary for {title}: {e}")

        return None

    def get_full_content(self, title: str) -> Optional[str]:
        """Get full article content in plain text.

        Args:
            title: Wikipedia article title

        Returns:
            Full article text or None
        """
        cache_key = f"wiki_content:{title}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "redirects": 1,
        }

        try:
            result = self._api_request(params)
            pages = result.get("query", {}).get("pages", {})

            for page_id, page_data in pages.items():
                if page_id != "-1":
                    content = page_data.get("extract", "")
                    if content:
                        self.cache.set(cache_key, content)
                        return content

        except Exception as e:
            logger.warning(f"Failed to get content for {title}: {e}")

        return None

    def get_sentences(self, title: str, max_sentences: int = 20) -> list[str]:
        """Get article content as list of sentences.

        This is the primary method for getting Wikipedia text
        to use as QA context. Each sentence can be indexed
        as a supporting fact.

        Args:
            title: Wikipedia article title
            max_sentences: Maximum sentences to return

        Returns:
            List of sentences
        """
        cache_key = f"wiki_sentences:{title}:{max_sentences}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        content = self.get_full_content(title)
        if not content:
            return []

        # Split into sentences
        # Use regex to handle abbreviations and decimal numbers
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        raw_sentences = re.split(sentence_pattern, content)

        # Clean up sentences
        sentences = []
        for sent in raw_sentences:
            sent = sent.strip()
            # Filter out very short or very long fragments
            if 10 < len(sent) < 500:
                # Remove citations like [1], [2]
                sent = re.sub(r'\[\d+\]', '', sent)
                sentences.append(sent.strip())

            if len(sentences) >= max_sentences:
                break

        self.cache.set(cache_key, sentences)
        return sentences

    def get_intro_sentences(self, title: str, num_sentences: int = 5) -> list[str]:
        """Get only introduction sentences (most relevant for facts).

        The introduction typically contains the key facts that are
        also in DBpedia (from infoboxes).

        Args:
            title: Wikipedia article title
            num_sentences: Number of intro sentences

        Returns:
            List of intro sentences
        """
        summary = self.get_summary(title, sentences=num_sentences)
        if not summary:
            return []

        # Split summary into sentences
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, summary)

        return [s.strip() for s in sentences if s.strip()]

    def get_paragraph_containing(
        self, title: str, search_text: str
    ) -> Optional[tuple[str, int]]:
        """Find paragraph containing specific text.

        Useful for finding the text that supports a KG fact.

        Args:
            title: Wikipedia article title
            search_text: Text to search for

        Returns:
            Tuple of (paragraph_text, sentence_index) or None
        """
        sentences = self.get_sentences(title)
        search_lower = search_text.lower()

        for i, sent in enumerate(sentences):
            if search_lower in sent.lower():
                return (sent, i)

        return None

    def get_content_for_dbpedia_entity(
        self, dbpedia_uri: str, max_sentences: int = 10
    ) -> list[str]:
        """Get Wikipedia sentences for a DBpedia entity.

        This is the key method for getting source text for KG facts.

        Args:
            dbpedia_uri: DBpedia resource URI
            max_sentences: Max sentences to return

        Returns:
            List of Wikipedia sentences
        """
        title = self.dbpedia_uri_to_title(dbpedia_uri)
        return self.get_sentences(title, max_sentences)

    def search_articles(self, query: str, limit: int = 5) -> list[str]:
        """Search for Wikipedia articles.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of article titles
        """
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
        }

        try:
            result = self._api_request(params)
            if len(result) >= 2:
                return result[1]
        except Exception as e:
            logger.warning(f"Search failed for {query}: {e}")

        return []

    def find_sentence_with_value(
        self, title: str, value: str, relation_hint: Optional[str] = None
    ) -> Optional[tuple[str, int]]:
        """Find sentence containing a specific value (from KG triple).

        Given a KG triple's object value, find the Wikipedia sentence
        that contains this value - this is the supporting text.

        Args:
            title: Wikipedia article title (from triple's subject)
            value: Value to find (from triple's object)
            relation_hint: Optional relation type for better matching

        Returns:
            Tuple of (sentence, index) or None
        """
        sentences = self.get_sentences(title)
        value_lower = value.lower()

        # First pass: exact value match
        for i, sent in enumerate(sentences):
            if value_lower in sent.lower():
                return (sent, i)

        # Second pass: try variations (dates, names with titles, etc.)
        value_parts = value_lower.split()
        if len(value_parts) > 1:
            # Try last word (often surname)
            for i, sent in enumerate(sentences):
                if value_parts[-1] in sent.lower():
                    return (sent, i)

        return None
