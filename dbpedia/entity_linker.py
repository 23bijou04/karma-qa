"""Entity linking using DBpedia Spotlight and SPARQL fallback."""

import logging
import re
import unicodedata
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
    "performer", "publisher", "composer", "musician",
})

# Keywords that precede entity titles in benchmark questions.
# Used to extract "song Aren'T You Kind Of Glad We Did?" → entity name.
_TITLE_KEYWORDS = frozenset({
    "song", "film", "movie", "book", "novel", "album",
    "series", "show", "play", "opera", "poem", "painting",
})

# Lowercase words allowed inside Title Cased entity names.
_TITLE_CONNECTORS = frozenset({
    "of", "the", "a", "an", "in", "on", "at", "to", "for",
    "and", "or", "du", "le", "la", "el", "de", "von", "van",
    "di", "des", "les", "un", "et", "vs", "as", "by", "no",
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
    r'\bperformer\b': 'dbo:MusicalWork',
    r'\bband\b': 'dbo:MusicalArtist',
    r'\bmusician\b': 'dbo:MusicalArtist',
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
    r'\bperformer\s+of\b': 'dbo:MusicalWork',
    r'\bperformed\b': 'dbo:MusicalWork',
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
    'dbo:Person': 'person',
    'dbo:MusicalArtist': 'musician',
    'dbo:City': 'city',
}


class DBpediaEntityLinker:
    """Entity linker using DBpedia Spotlight with SPARQL fallback."""

    def __init__(
        self,
        spotlight_endpoint: str = "https://api.dbpedia-spotlight.org/en/annotate",
        sparql_client: Optional[DBpediaSPARQLClient] = None,
        cache_manager: Optional[CacheManager] = None,
        min_confidence: float = 0.5,
        min_support: int = 5,
        # Entities below this threshold still enter the pipeline but don't
        # block the unlinked resolution loop from re-resolving the same mention.
        high_confidence_threshold: float = 0.7,
    ):
        self.spotlight_endpoint = spotlight_endpoint
        self.sparql_client = sparql_client or DBpediaSPARQLClient()
        self.cache = cache_manager or CacheManager()
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.high_confidence_threshold = high_confidence_threshold
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
        # Track possessor-derived mentions so they survive substring removal.
        # "Dr. Dre's real name" → possessor "Dr. Dre" is a better entity
        # candidate than the full noun chunk.
        possessor_derived: set[str] = set()

        # Quoted strings (ASCII and curly double quotes)
        mentions.extend(re.findall(r'["\u201C]([^"\u201D]+)["\u201D]', question))

        # Noun chunks with stopword stripping + possessive splitting
        for chunk in doc.noun_chunks:
            words = chunk.text.split()
            original_words = list(words)
            while words and words[0].lower() in _STRIP_WORDS:
                words.pop(0)
            text = " ".join(words).strip()
            stripping_occurred = len(words) < len(original_words)

            if not words:
                # Stripping removed all words — entity might start with strip word
                text_original = " ".join(original_words).strip()
                if text_original and len(text_original) > 1:
                    mentions.append(text_original)
                continue

            mentions.append(text)
            # When stripping removed leading words, also keep the original in case
            # the article/preposition is part of the entity name (e.g. "The Who",
            # "A Clockwork Orange", "The Band").
            if stripping_occurred:
                text_original = " ".join(original_words).strip()
                if text_original.lower() != text.lower() and len(text_original) > 1:
                    mentions.append(text_original)
            # Split at possessive "'s" to extract the possessor as a
            # separate mention (e.g., "Dr. Dre's real name" → "Dr. Dre")
            poss_match = re.match(r"^(.+?)[''\u2019]s\b", text)
            if poss_match:
                possessor = poss_match.group(1).strip()
                if possessor and len(possessor) > 1:
                    mentions.append(possessor)
                    possessor_derived.add(possessor.lower())

        # Named entities (skip numeric types)
        # Split on " or " for bridge_comparison patterns ("X or Y?")
        # NOTE: don't split on "and" — it's common inside titles
        # ("Angel And The Badman", "Joseph And The Amazing...").
        for ent in doc.ents:
            if ent.label_ not in ("CARDINAL", "ORDINAL", "QUANTITY", "PERCENT"):
                if " or " in ent.text:
                    parts = ent.text.split(" or ")
                else:
                    parts = [ent.text]
                for part in parts:
                    part = part.strip()
                    # Strip trailing "or" that spaCy sometimes leaves on
                    # entity spans in comparison questions.
                    # e.g. "The Singing Fool or" → "The Singing Fool"
                    # Only strip "or" — other connectors (and, in, the, of)
                    # can legitimately end titles ("The Girl In", "Joseph And").
                    ent_words = part.split()
                    while ent_words and ent_words[-1].lower() == "or":
                        ent_words.pop()
                    part = " ".join(ent_words).strip()
                    if part:
                        mentions.append(part)

        # Title-after-keyword extraction: "song Aren'T You Kind Of Glad We
        # Did?" → extract entity name by following capitalized words after
        # a title keyword (song, film, book, etc.). Handles entity names
        # with special chars that spaCy/Spotlight miss.
        words = question.split()
        for i, w in enumerate(words):
            if w.lower().rstrip(".,?!:;") in _TITLE_KEYWORDS and i + 1 < len(words):
                nxt = words[i + 1]
                first_char = nxt.lstrip("'\"(\u2018\u201C")[:1]
                if not first_char or not first_char.isupper():
                    continue
                # Collect Title Cased sequence. Connectors (of, the, and,
                # or...) are only included if the NEXT word is capitalized
                # — prevents "Inception or the" from leaking past the entity.
                title_words = []
                j = i + 1
                while j < len(words):
                    word = words[j]
                    clean = word.lstrip("'\"(\u2018\u201C").rstrip(".,;")
                    if not clean:
                        break
                    first = clean[0]
                    if first.isupper():
                        title_words.append(word)
                        j += 1
                    elif clean.lower() in _TITLE_CONNECTORS:
                        # Look past consecutive connectors for a capitalized word
                        k = j + 1
                        found_upper = False
                        while k < len(words):
                            peek_word = words[k].lstrip("'\"\u2018\u201C(")
                            peek_clean = peek_word.rstrip(".,;")
                            if not peek_clean:
                                break
                            if peek_clean[0].isupper():
                                # Found uppercase after connector chain → include all connectors
                                for ci in range(j, k):
                                    title_words.append(words[ci])
                                found_upper = True
                                j = k  # advance j to the capitalized word (next iter appends it)
                                break
                            elif peek_clean.lower() in _TITLE_CONNECTORS:
                                k += 1
                                continue
                            else:
                                break
                        if not found_upper:
                            break
                    else:
                        break
                if len(title_words) >= 1:
                    title = " ".join(title_words).rstrip(".,;")
                    mentions.append(title)

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

        # Remove substring mentions, but keep possessor-derived mentions
        # even if they're substrings (they're more precise entity candidates).
        final: list[str] = []
        for m in unique:
            m_lower = m.lower()
            if m_lower in possessor_derived:
                final.append(m)  # always keep possessor-derived
            elif not any(
                m_lower != o.lower() and m_lower in o.lower()
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

        # Clean obj_phrase for fallback search: strip determiners, years,
        # and generic descriptors that pollute bif:contains queries.
        _clean = re.sub(r'\b(?:the|a|an|this|that)\b', '', obj_phrase, flags=re.I)
        _clean = re.sub(r'\b\d{4}\b', '', _clean)  # years
        _clean = re.sub(
            r'\b(?:documentary|miniseries|series|biopic|telefilm)\b',
            '', _clean, flags=re.I,
        )
        clean_obj = re.sub(r'\s+', ' ', _clean).strip()

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

            # Direct URI probe: try dbr:Name_(type) pattern before text search.
            # DBpedia uses this for disambiguation (e.g., Thomas_Jefferson_(film)).
            probe_local = entity.label.replace(' ', '_')
            # DBpedia URIs always capitalize the first letter (MediaWiki convention)
            if probe_local and probe_local[0].islower():
                probe_local = probe_local[0].upper() + probe_local[1:]
            probe_uri = f"http://dbpedia.org/resource/{probe_local}_({suffix})"
            try:
                probe_types = self.sparql_client.get_entity_types(probe_uri)
                type_short = expected_type.replace('dbo:', '')
                if any(type_short in t for t in probe_types):
                    logger.info(
                        f"URI probe hit: '{entity.label}' -> {probe_uri}"
                    )
                    validated.append(Entity(
                        uri=probe_uri, label=entity.label,
                        source="type_corrected",
                    ))
                    continue
            except Exception:
                pass  # probe failed, fall through to search

            # Type mismatch: search with cleaned phrase + type suffix
            search_query = f"{clean_obj} {suffix}" if clean_obj else f"{obj_phrase} {suffix}"
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
            m_lower = unicodedata.normalize("NFC", mention.lower().strip())
            m_words = set(m_lower.split())
            matched = False
            for sf in surface_forms:
                sf_lower = unicodedata.normalize("NFC", sf.lower().strip())
                sf_words = set(sf_lower.split())
                # Jaccard similarity
                intersection = m_words & sf_words
                union = m_words | sf_words
                jaccard = len(intersection) / len(union) if union else 0
                # Substring: mention IN surface_form is a match,
                # but surface_form IN mention is a MISLINK (partial match)
                mention_in_sf = m_lower in sf_lower
                sf_in_mention = sf_lower in m_lower and sf_lower != m_lower
                if mention_in_sf:
                    matched = True
                    break
                if jaccard >= 0.5 and not sf_in_mention:
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
        # Split entities by confidence: only high-confidence Spotlight links
        # block the unlinked loop. Low-confidence links are kept as fallbacks
        # but their surface forms don't prevent re-resolution.
        mentions = self._extract_mentions(text)
        high_conf = [e for e in entities
                     if e.confidence >= self.high_confidence_threshold]
        low_conf = [e for e in entities
                    if e.confidence < self.high_confidence_threshold
                    and e.source == "spotlight"]
        unlinked = self._find_unlinked(mentions, high_conf)

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
                    # Replace low-confidence Spotlight entity for same mention
                    mention_lower = mention.lower().strip()
                    entities = [
                        e for e in entities
                        if not (e in low_conf
                                and e.label.lower().strip() == mention_lower)
                    ]
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

