"""Entity linking using DBpedia Spotlight and SPARQL fallback."""

import logging
from typing import Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models import Entity
from .client import DBpediaSPARQLClient
from .cache import CacheManager

logger = logging.getLogger(__name__)


class DBpediaEntityLinker:
    """Entity linker using DBpedia Spotlight with SPARQL fallback."""

    def __init__(
        self,
        spotlight_endpoint: str = "https://api.dbpedia-spotlight.org/en/annotate",
        sparql_client: Optional[DBpediaSPARQLClient] = None,
        cache_manager: Optional[CacheManager] = None,
        min_confidence: float = 0.35,
        min_support: int = 20,
    ):
        self.spotlight_endpoint = spotlight_endpoint
        self.sparql_client = sparql_client or DBpediaSPARQLClient()
        self.cache = cache_manager or CacheManager()
        self.min_confidence = min_confidence
        self.min_support = min_support
        self._http_client = httpx.Client(timeout=30.0)

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_http_client"):
            self._http_client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _call_spotlight(self, text: str) -> dict:
        """Call DBpedia Spotlight API."""
        headers = {"Accept": "application/json"}
        params = {
            "text": text,
            "confidence": self.min_confidence,
            "support": self.min_support,
        }

        try:
            response = self._http_client.get(
                self.spotlight_endpoint, params=params, headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning(f"Spotlight API error: {e}")
            raise

    def link_entities(self, text: str, use_cache: bool = True) -> list[Entity]:
        """Extract and link entities from text using Spotlight.

        Args:
            text: Input text to extract entities from
            use_cache: Whether to use cached results

        Returns:
            List of linked Entity objects
        """
        if use_cache:
            cache_key = f"spotlight:{text}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return [Entity(**e) for e in cached]

        entities = []

        try:
            result = self._call_spotlight(text)
            resources = result.get("Resources", [])

            for resource in resources:
                entity = Entity(
                    uri=resource["@URI"],
                    label=resource["@surfaceForm"],
                    confidence=float(resource.get("@similarityScore", 0.5)),
                    source="spotlight",
                )
                entities.append(entity)

        except Exception as e:
            logger.warning(f"Spotlight failed, falling back to SPARQL: {e}")
            entities = self._sparql_fallback(text)

        if use_cache and entities:
            self.cache.set(
                f"spotlight:{text}",
                [
                    {
                        "uri": e.uri,
                        "label": e.label,
                        "confidence": e.confidence,
                        "source": e.source,
                    }
                    for e in entities
                ],
            )

        return entities

    def _sparql_fallback(self, text: str) -> list[Entity]:
        """Fall back to SPARQL-based entity extraction.

        Uses simple noun phrase heuristics and SPARQL lookup.
        """
        entities = []
        words = text.split()

        # Try to find capitalized phrases (likely named entities)
        current_phrase = []
        for i, word in enumerate(words):
            # Check if word starts with capital (excluding sentence start)
            if word[0].isupper() and (i == 0 or words[i - 1][-1] not in ".!?"):
                current_phrase.append(word)
            else:
                if current_phrase:
                    phrase = " ".join(current_phrase)
                    # Skip common words
                    if phrase.lower() not in {
                        "the", "a", "an", "who", "what", "when", "where", "which", "how"
                    }:
                        entity = self.sparql_client.get_entity_by_label(phrase)
                        if entity:
                            entities.append(entity)
                    current_phrase = []

        # Handle trailing phrase
        if current_phrase:
            phrase = " ".join(current_phrase)
            if phrase.lower() not in {
                "the", "a", "an", "who", "what", "when", "where", "which", "how"
            }:
                entity = self.sparql_client.get_entity_by_label(phrase)
                if entity:
                    entities.append(entity)

        return entities

    def link_entity(
        self, surface_form: str, context: Optional[str] = None
    ) -> Optional[Entity]:
        """Link a single entity mention to DBpedia.

        Args:
            surface_form: The entity mention text
            context: Optional surrounding text for disambiguation

        Returns:
            Linked Entity or None if not found
        """
        # First try Spotlight with context
        if context:
            entities = self.link_entities(context)
            for entity in entities:
                if entity.label.lower() == surface_form.lower():
                    return entity

        # Try direct label match via SPARQL
        entity = self.sparql_client.get_entity_by_label(surface_form)
        if entity:
            return entity

        # Try fuzzy search
        candidates = self.sparql_client.search_entities(surface_form, limit=5)
        if candidates:
            return self._disambiguate(surface_form, candidates, context)

        return None

    def _disambiguate(
        self,
        surface_form: str,
        candidates: list[Entity],
        context: Optional[str] = None,
    ) -> Optional[Entity]:
        """Disambiguate between multiple entity candidates.

        Args:
            surface_form: Original entity mention
            candidates: List of candidate entities
            context: Optional context for disambiguation

        Returns:
            Best matching Entity or None
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        if not context:
            # Return first candidate (usually most popular)
            return candidates[0]

        # Score candidates based on context word overlap with abstract
        context_words = set(context.lower().split())
        best_candidate = candidates[0]
        best_score = 0

        for candidate in candidates:
            if candidate.abstract:
                abstract_words = set(candidate.abstract.lower().split())
                overlap = len(context_words & abstract_words)
                if overlap > best_score:
                    best_score = overlap
                    best_candidate = candidate

        return best_candidate

    def resolve_coreference(
        self, pronoun: str, entities: list[Entity], context: str
    ) -> Optional[Entity]:
        """Simple coreference resolution for pronouns.

        Args:
            pronoun: Pronoun to resolve (he, she, it, they)
            entities: List of previously mentioned entities
            context: Surrounding text

        Returns:
            Most likely referent Entity or None
        """
        if not entities:
            return None

        pronoun_lower = pronoun.lower()

        # Filter by pronoun type
        if pronoun_lower in {"he", "his", "him"}:
            # Look for male entities
            for entity in entities:
                if entity.types:
                    if any("Person" in t for t in entity.types):
                        return entity
        elif pronoun_lower in {"she", "her", "hers"}:
            # Look for female entities
            for entity in entities:
                if entity.types:
                    if any("Person" in t for t in entity.types):
                        return entity
        elif pronoun_lower in {"it", "its"}:
            # Look for non-person entities
            for entity in entities:
                if entity.types:
                    if not any("Person" in t for t in entity.types):
                        return entity

        # Default to most recent entity
        return entities[-1] if entities else None
