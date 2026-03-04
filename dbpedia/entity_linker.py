"""Entity linking using DBpedia Spotlight and SPARQL fallback."""

import logging
import re
from typing import Optional

import httpx
import spacy
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models import Entity
from .cache import CacheManager
from .client import DBpediaSPARQLClient

logger = logging.getLogger(__name__)

_nlp = spacy.load("en_core_web_sm")

# Words to strip from the front of noun chunks
_STRIP_WORDS = frozenset({
    "who", "what", "when", "where", "which", "how", "why",
    "the", "a", "an", "is", "are", "was", "were", "be",
    "did", "does", "do", "has", "have", "had", "both", "either",
    "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "and", "or", "not", "that", "this", "it", "its",
})

# Common nouns that appear as spaCy mentions but are NOT entity names.
# These would resolve to ontology/property URIs, not resources.
_MENTION_BLOCKLIST = frozenset({
    "film", "movie", "director", "book", "author", "novel", "song",
    "album", "band", "country", "city", "person", "actor", "actress",
    "writer", "singer", "politician", "company", "school", "university",
    "year", "date", "time", "place", "name", "age", "number",
    "first", "last", "born", "died", "older", "younger",
    "earlier", "later", "more", "fewer", "same", "different",
})

# Maps question keywords to the type of SUBJECT entities being compared.
TYPE_HINTS: dict[str, str] = {
    r'\bfilm\b': 'dbo:Film',
    r'\bmovie\b': 'dbo:Film',
    r'\bdirector\b': 'dbo:Film',
    r'\bbook\b': 'dbo:Book',
    r'\bauthor\b': 'dbo:Book',
    r'\bnovel\b': 'dbo:Book',
    r'\bsong\b': 'dbo:MusicalWork',
    r'\balbum\b': 'dbo:MusicalWork',
    r'\bband\b': 'dbo:MusicalArtist',
    r'\bcountry\b': 'dbo:Country',
    r'\bcity\b': 'dbo:City',
    r'\bperson\b': 'dbo:Person',
}

# Maps relation words to the type of the OBJECT entity (the thing being acted upon).
# "director of X" -> X is a Film; "author of X" -> X is a Book.
RELATION_TYPE_HINTS: dict[str, str] = {
    r'\bdirector\s+of\b': 'dbo:Film',
    r'\bdirected\b': 'dbo:Film',
    r'\bcast\s+of\b': 'dbo:Film',
    r'\bstarring\s+in\b': 'dbo:Film',
    r'\bauthor\s+of\b': 'dbo:Book',
    r'\bwrote\b': 'dbo:Book',
    r'\bwriter\s+of\b': 'dbo:Book',
    r'\bcomposer\s+of\b': 'dbo:MusicalWork',
    r'\bpresident\s+of\b': 'dbo:Country',
    r'\bfounder\s+of\b': 'dbo:Company',
    r'\bcapital\s+of\b': 'dbo:Country',
}

# Map dbo:type to readable suffix for Wikipedia search disambiguation
_TYPE_SUFFIX_MAP = {
    'dbo:Film': 'film',
    'dbo:Book': 'book',
    'dbo:MusicalWork': 'album',
    'dbo:Country': 'country',
    'dbo:Company': 'company',
}


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

    # ------------------------------------------------------------------
    # Mention extraction & type hinting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mentions(question: str) -> list[str]:
        """Extract entity mentions from question using spaCy NER + noun chunks.

        Returns raw mention strings (no title-casing — that's for Wikipedia).
        Deduplicates and removes substring mentions.
        """
        doc = _nlp(question)
        mentions: list[str] = []

        # Quoted strings
        mentions.extend(re.findall(r'"([^"]+)"', question))

        # Noun chunks with stopword stripping
        for chunk in doc.noun_chunks:
            words = chunk.text.split()
            while words and words[0].lower() in _STRIP_WORDS:
                words.pop(0)
            text = " ".join(words).strip()
            if text:
                mentions.append(text)

        # Named entities (skip numeric types)
        # Split on " or " for bridge_comparison patterns ("X or Y?")
        for ent in doc.ents:
            if ent.label_ not in ("CARDINAL", "ORDINAL", "QUANTITY", "PERCENT"):
                if " or " in ent.text:
                    for part in ent.text.split(" or "):
                        part = part.strip()
                        if part:
                            mentions.append(part)
                else:
                    mentions.append(ent.text)

        # Deduplicate case-insensitively, filter blocklisted words
        seen: set[str] = set()
        unique: list[str] = []
        for m in mentions:
            key = m.lower().strip()
            if (key not in seen
                    and len(key) > 1
                    and key not in _MENTION_BLOCKLIST):
                seen.add(key)
                unique.append(m)

        # Remove substring mentions
        final: list[str] = []
        for m in unique:
            if not any(
                m.lower() != o.lower() and m.lower() in o.lower()
                for o in unique
            ):
                final.append(m)
        return final

    @staticmethod
    def _extract_type_hint(question: str) -> str | None:
        """Extract a dbo: type hint from question keywords and relation context.

        Checks two sources:
          1. RELATION_TYPE_HINTS: relation-implied types ("director of X" -> Film)
          2. TYPE_HINTS: explicit type words ("film", "movie")
        Relation hints take priority (more specific).
        """
        q_lower = question.lower()
        # Check relation-based hints first (higher specificity)
        for pattern, dbo_type in RELATION_TYPE_HINTS.items():
            if re.search(pattern, q_lower):
                return dbo_type
        # Fall back to explicit keyword hints
        for pattern, dbo_type in TYPE_HINTS.items():
            if re.search(pattern, q_lower):
                return dbo_type
        return None

    def _validate_entity_types(
        self, entities: list[Entity], question: str, expected_type: str,
    ) -> list[Entity]:
        """Validate Spotlight entities against expected type from question context.

        If an entity's rdf:type doesn't include the expected type AND the entity
        label appears after a relation keyword (e.g., "director of [X]"), try
        bif:contains search with type constraint as disambiguation.
        """
        # Build obj_pattern dynamically from RELATION_TYPE_HINTS to stay DRY.
        # Extract all first words from patterns like r'\bdirector\s+of\b' -> "director"
        relation_words = []
        for pattern in RELATION_TYPE_HINTS:
            m = re.search(r"\\b(\w+)", pattern)
            if m:
                relation_words.append(m.group(1))
        if not relation_words:
            return entities

        words_alt = "|".join(relation_words)
        # Match both "X of Y" form and verb "Xed Y" / "X Y" form
        obj_match = re.search(
            rf'(?:{words_alt})\s+of\s+(.+?)\s*(?:[?!,]|$)', question, re.IGNORECASE,
        )
        if not obj_match:
            # Verb form: "directed Polish-Russian War", "wrote Harry Potter"
            obj_match = re.search(
                rf'(?:{words_alt})\s+(.+?)\s*(?:[?!,]|$)', question, re.IGNORECASE,
            )
        if not obj_match:
            return entities

        obj_phrase = obj_match.group(1).strip().rstrip("?.,!")
        obj_lower = obj_phrase.lower()

        suffix = _TYPE_SUFFIX_MAP.get(expected_type, '')
        if not suffix:
            return entities

        validated = []
        for entity in entities:
            # Check if this entity corresponds to the object phrase
            if obj_lower not in entity.label.lower() and entity.label.lower() not in obj_lower:
                validated.append(entity)  # not the target entity, keep as-is
                continue

            # Check entity types via SPARQL
            try:
                types = self.sparql_client.get_entity_types(entity.uri)
                type_short = expected_type.replace('dbo:', '')
                if any(type_short in t for t in types):
                    validated.append(entity)  # correct type, keep
                    continue
            except Exception:
                pass  # can't verify, try fallback

            # Type mismatch: search with type suffix for disambiguation
            search_query = f"{obj_phrase} {suffix}"
            logger.info(
                f"Type mismatch for '{entity.label}' (expected {expected_type}), "
                f"searching: '{search_query}'"
            )
            fallback = self.sparql_client.search_entity_by_name(
                search_query, entity_type=expected_type,
            )
            if fallback and fallback.uri != entity.uri:
                logger.info(f"Replaced '{entity.label}' -> '{fallback.label}' ({fallback.uri})")
                validated.append(fallback)
            else:
                validated.append(entity)  # keep original if no better candidate

        return validated

    @staticmethod
    def _find_unlinked(
        mentions: list[str], entities: list[Entity]
    ) -> list[str]:
        """Find mentions that Spotlight didn't link (or mislinked).

        Uses word-level Jaccard similarity (threshold 0.5) and substring
        containment to match spaCy mentions against Spotlight surface forms.
        """
        # Collect Spotlight surface forms (label field holds surfaceForm)
        surface_forms = [e.label for e in entities]
        unlinked: list[str] = []

        for mention in mentions:
            m_lower = mention.lower().strip()
            m_words = set(m_lower.split())
            matched = False
            for sf in surface_forms:
                sf_lower = sf.lower().strip()
                sf_words = set(sf_lower.split())
                # Jaccard similarity
                intersection = m_words & sf_words
                union = m_words | sf_words
                jaccard = len(intersection) / len(union) if union else 0
                # Substring: mention IN surface_form is a match,
                # but surface_form IN mention is a MISLINK (partial match)
                mention_in_sf = m_lower in sf_lower
                if jaccard >= 0.5 or mention_in_sf:
                    matched = True
                    break
            if not matched:
                unlinked.append(mention)

        return unlinked

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
        """Extract and link entities from text using Spotlight + bif:contains.

        Cascade:
          1. Spotlight API (primary — good for common entities)
          2. spaCy NER identifies ALL entity mentions
          3. For unlinked mentions -> bif:contains text search with type hints
          4. Disambiguation via type matching + exact label preference

        Args:
            text: Input text to extract entities from
            use_cache: Whether to use cached results

        Returns:
            List of linked Entity objects (every entity has a resolved URI)
        """
        if use_cache:
            cache_key = f"link_entities:{text}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return [Entity(**e) for e in cached]

        entities: list[Entity] = []

        # Step 1: Spotlight
        try:
            result = self._call_spotlight(text)
            resources = result.get("Resources", [])
            for resource in resources:
                entities.append(Entity(
                    uri=resource["@URI"],
                    label=resource["@surfaceForm"],
                    confidence=float(resource.get("@similarityScore", 0.5)),
                    source="spotlight",
                ))
        except Exception as e:
            logger.warning(f"Spotlight failed: {e}")

        # Post-hoc type validation: if we can infer the expected type from the
        # question context, check Spotlight entities against it. If a Spotlight
        # entity's DBpedia types don't match, try search with type appended.
        type_hint = self._extract_type_hint(text)
        if type_hint and entities:
            entities = self._validate_entity_types(entities, text, type_hint)

        # Step 2: spaCy mention extraction + unlinked detection
        mentions = self._extract_mentions(text)
        unlinked = self._find_unlinked(mentions, entities)

        if unlinked:
            if type_hint is None:
                type_hint = self._extract_type_hint(text)
            existing_uris = {e.uri for e in entities}

            for mention in unlinked:
                found = self.sparql_client.search_entity_by_name(
                    mention, entity_type=type_hint
                )
                if found and found.uri not in existing_uris:
                    logger.info(
                        f"bif:contains resolved '{mention}' -> {found.uri}"
                    )
                    entities.append(found)
                    existing_uris.add(found.uri)

        # Fallback: if nothing found at all, try old SPARQL label fallback
        if not entities:
            entities = self._sparql_fallback(text)

        if use_cache and entities:
            self.cache.set(
                cache_key,
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
