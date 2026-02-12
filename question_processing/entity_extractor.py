"""Entity extraction from questions."""

import re
import logging
from typing import Optional

from ..models import Entity
from ..dbpedia.entity_linker import DBpediaEntityLinker

logger = logging.getLogger(__name__)


class QuestionEntityExtractor:
    """Extracts and links entities from questions.

    Uses DBpedia Spotlight for entity linking with fallbacks
    to pattern-based extraction and SPARQL lookup.
    """

    def __init__(
        self,
        entity_linker: Optional[DBpediaEntityLinker] = None,
    ):
        """Initialize extractor.

        Args:
            entity_linker: DBpedia entity linker instance
        """
        self.entity_linker = entity_linker or DBpediaEntityLinker()

        # Common words to filter out
        self.stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "who", "what", "when", "where", "which", "how", "why",
            "did", "does", "do", "has", "have", "had",
            "first", "last", "same", "both", "all", "any",
            "more", "less", "most", "least",
            "film", "movie", "person", "people", "country", "city",
        }

    def extract_entities(self, question: str) -> list[Entity]:
        """Extract and link entities from a question.

        Args:
            question: The question text

        Returns:
            List of linked Entity objects
        """
        # First try Spotlight
        entities = self.entity_linker.link_entities(question)

        if entities:
            return entities

        # Fall back to pattern-based extraction
        mentions = self._extract_entity_mentions(question)
        entities = []

        for mention in mentions:
            entity = self.entity_linker.link_entity(mention, context=question)
            if entity:
                entities.append(entity)

        return entities

    def _extract_entity_mentions(self, question: str) -> list[str]:
        """Extract potential entity mentions using patterns.

        Args:
            question: The question text

        Returns:
            List of potential entity mention strings
        """
        mentions = []

        # Pattern 1: Quoted strings
        quoted = re.findall(r'"([^"]+)"', question)
        mentions.extend(quoted)

        # Pattern 2: Capitalized phrases
        # Look for sequences of capitalized words
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        capitalized = re.findall(cap_pattern, question)

        for phrase in capitalized:
            if phrase.lower() not in self.stopwords:
                mentions.append(phrase)

        # Pattern 3: "X or Y" pattern (comparison questions)
        or_match = re.search(r'([A-Z][^,?]+?)\s+or\s+([A-Z][^,?]+?)(?:\?|$)', question)
        if or_match:
            for group in or_match.groups():
                clean = group.strip()
                if clean and clean.lower() not in self.stopwords:
                    mentions.append(clean)

        # Pattern 4: After "of" preposition
        of_pattern = r'of\s+([A-Z][A-Za-z\s]+?)(?:\?|,|\.|\s+(?:and|or|who|that|which))'
        of_matches = re.findall(of_pattern, question)
        for match in of_matches:
            clean = match.strip()
            if clean and clean.lower() not in self.stopwords:
                mentions.append(clean)

        # Deduplicate while preserving order
        seen = set()
        unique_mentions = []
        for mention in mentions:
            if mention not in seen:
                seen.add(mention)
                unique_mentions.append(mention)

        return unique_mentions

    def extract_entity_pairs(self, question: str) -> list[tuple[Entity, Entity]]:
        """Extract pairs of entities for comparison questions.

        Args:
            question: The question text

        Returns:
            List of entity pairs
        """
        entities = self.extract_entities(question)

        if len(entities) < 2:
            return []

        # For comparison questions, typically first two entities
        pairs = []
        if len(entities) >= 2:
            pairs.append((entities[0], entities[1]))

        return pairs

    def extract_main_entity(self, question: str) -> Optional[Entity]:
        """Extract the primary entity the question is about.

        Args:
            question: The question text

        Returns:
            Main Entity or None
        """
        entities = self.extract_entities(question)

        if not entities:
            return None

        # The main entity is typically:
        # 1. The one mentioned last (in "of X" patterns)
        # 2. The first proper noun

        # Check for "of [Entity]?" pattern
        of_match = re.search(r'of\s+([A-Z][^?]+?)\??$', question)
        if of_match:
            main_mention = of_match.group(1).strip()
            for entity in entities:
                if entity.label.lower() == main_mention.lower():
                    return entity

        # Default to first entity
        return entities[0] if entities else None

    def extract_bridge_entity_type(self, question: str) -> Optional[str]:
        """Identify the type of bridge entity needed.

        For questions like "Who is the director of the film that X starred in?"
        Returns the bridge entity type (e.g., "film", "company").

        Args:
            question: The question text

        Returns:
            Bridge entity type or None
        """
        question_lower = question.lower()

        type_patterns = {
            "film": ["film", "movie"],
            "company": ["company", "organization", "business"],
            "university": ["university", "college", "school"],
            "country": ["country", "nation"],
            "city": ["city", "town"],
            "album": ["album", "record"],
            "book": ["book", "novel"],
        }

        for entity_type, keywords in type_patterns.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return entity_type

        return None

    def extract_relation_mentions(self, question: str) -> list[str]:
        """Extract relation/property mentions from the question.

        Args:
            question: The question text

        Returns:
            List of relation names mentioned
        """
        question_lower = question.lower()

        # Common relations in 2WikiMultiHopQA
        relations = [
            "born", "birth", "died", "death", "lived",
            "director", "directed", "producer", "produced",
            "founder", "founded", "author", "wrote",
            "father", "mother", "spouse", "husband", "wife",
            "child", "son", "daughter", "sibling", "brother", "sister",
            "grandfather", "grandmother", "grandchild",
            "nationality", "country", "citizen",
            "occupation", "profession", "job",
            "educated", "studied", "graduated",
        ]

        found_relations = []
        for relation in relations:
            if relation in question_lower:
                found_relations.append(relation)

        return found_relations

    def resolve_pronouns(
        self, question: str, context_entities: list[Entity]
    ) -> list[Entity]:
        """Resolve pronouns in question to entities.

        Args:
            question: The question text
            context_entities: Previously mentioned entities

        Returns:
            Updated list of entities with resolved pronouns
        """
        if not context_entities:
            return self.extract_entities(question)

        pronouns = ["he", "she", "it", "they", "his", "her", "its", "their"]
        question_lower = question.lower()

        # Check if question contains pronouns
        has_pronouns = any(
            re.search(rf'\b{p}\b', question_lower) for p in pronouns
        )

        if not has_pronouns:
            return self.extract_entities(question)

        # Extract new entities and combine with context
        new_entities = self.extract_entities(question)
        all_entities = list(context_entities) + new_entities

        # Resolve pronouns to most recent matching entity
        for pronoun in pronouns:
            if re.search(rf'\b{pronoun}\b', question_lower):
                resolved = self.entity_linker.resolve_coreference(
                    pronoun, context_entities, question
                )
                if resolved and resolved not in all_entities:
                    all_entities.append(resolved)

        return all_entities
