"""Relevance scoring for graph traversal."""

import re
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class RelevanceScorer(ABC):
    """Abstract base class for relevance scoring."""

    @abstractmethod
    def score(
        self,
        question: str,
        entity: str,
        relation: str,
        entity_label: Optional[str] = None,
    ) -> float:
        """Score the relevance of an entity/relation to a question.

        Args:
            question: The question text
            entity: Entity URI or name
            relation: Relation URI or name
            entity_label: Optional human-readable entity label

        Returns:
            Relevance score between 0 and 1
        """
        pass


class KeywordRelevanceScorer(RelevanceScorer):
    """Keyword-based relevance scoring."""

    # Property name mappings to question keywords
    PROPERTY_KEYWORDS = {
        "birthDate": ["born", "birth", "birthday"],
        "deathDate": ["died", "death", "passed away"],
        "birthPlace": ["born", "birth place", "birthplace", "native"],
        "deathPlace": ["died", "death place"],
        "father": ["father", "dad", "paternal"],
        "mother": ["mother", "mom", "maternal"],
        "spouse": ["spouse", "wife", "husband", "married"],
        "child": ["child", "son", "daughter", "children"],
        "nationality": ["nationality", "citizen", "country"],
        "occupation": ["occupation", "job", "profession", "work"],
        "director": ["director", "directed"],
        "producer": ["producer", "produced"],
        "releaseDate": ["released", "release", "premiere", "publication"],
        "foundedBy": ["founded", "founder", "started"],
        "foundingDate": ["founded", "inception", "established"],
        "almaMater": ["educated", "studied", "graduated", "university"],
    }

    def __init__(self, boost_exact_match: float = 0.3):
        """Initialize scorer.

        Args:
            boost_exact_match: Bonus score for exact keyword matches
        """
        self.boost_exact_match = boost_exact_match
        self._build_reverse_mapping()

    def _build_reverse_mapping(self) -> None:
        """Build reverse mapping from keywords to properties."""
        self.keyword_to_properties: dict[str, list[str]] = {}
        for prop, keywords in self.PROPERTY_KEYWORDS.items():
            for keyword in keywords:
                if keyword not in self.keyword_to_properties:
                    self.keyword_to_properties[keyword] = []
                self.keyword_to_properties[keyword].append(prop)

    def score(
        self,
        question: str,
        entity: str,
        relation: str,
        entity_label: Optional[str] = None,
    ) -> float:
        """Score relevance using keyword matching.

        Args:
            question: The question text
            entity: Entity URI
            relation: Relation URI
            entity_label: Optional entity label

        Returns:
            Relevance score between 0 and 1
        """
        question_lower = question.lower()
        question_words = set(question_lower.split())

        # Extract local names from URIs
        relation_name = self._extract_local_name(relation)
        entity_name = entity_label or self._extract_local_name(entity)

        score = 0.0

        # 1. Check if entity name appears in question
        if entity_name.lower() in question_lower:
            score += 0.4

        # 2. Check if relation matches question keywords
        relation_score = self._score_relation_relevance(relation_name, question_words)
        score += relation_score * 0.4

        # 3. Check for property keyword matches
        for keyword, properties in self.keyword_to_properties.items():
            if keyword in question_lower:
                if any(prop.lower() in relation_name.lower() for prop in properties):
                    score += self.boost_exact_match

        # Normalize to [0, 1]
        return min(1.0, max(0.0, score))

    def _extract_local_name(self, uri: str) -> str:
        """Extract local name from URI."""
        if "/" in uri:
            local = uri.split("/")[-1]
        elif "#" in uri:
            local = uri.split("#")[-1]
        else:
            local = uri

        return local.replace("_", " ")

    def _score_relation_relevance(
        self, relation_name: str, question_words: set[str]
    ) -> float:
        """Score how relevant a relation is to the question."""
        relation_lower = relation_name.lower()
        relation_words = set(relation_lower.replace("_", " ").split())

        # Check for direct word overlap
        overlap = question_words & relation_words
        if overlap:
            return len(overlap) / len(relation_words)

        # Check for partial matches
        for qword in question_words:
            if qword in relation_lower or relation_lower in qword:
                return 0.5

        return 0.0

    def get_relevant_properties(self, question: str) -> list[str]:
        """Get list of potentially relevant DBpedia properties for a question.

        Args:
            question: The question text

        Returns:
            List of property names that might be relevant
        """
        question_lower = question.lower()
        relevant = []

        for keyword, properties in self.keyword_to_properties.items():
            if keyword in question_lower:
                relevant.extend(properties)

        return list(set(relevant))


class EmbeddingRelevanceScorer(RelevanceScorer):
    """Embedding-based relevance scoring (requires sentence-transformers)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with embedding model.

        Args:
            model_name: Sentence-transformers model name
        """
        self.model = None
        self.model_name = model_name
        self._load_model()

    def _load_model(self) -> None:
        """Lazy load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to keyword scoring."
            )

    def score(
        self,
        question: str,
        entity: str,
        relation: str,
        entity_label: Optional[str] = None,
    ) -> float:
        """Score relevance using embedding similarity.

        Args:
            question: The question text
            entity: Entity URI
            relation: Relation URI
            entity_label: Optional entity label

        Returns:
            Relevance score between 0 and 1
        """
        if self.model is None:
            # Fall back to keyword scoring
            fallback = KeywordRelevanceScorer()
            return fallback.score(question, entity, relation, entity_label)

        # Extract local names
        relation_name = self._extract_local_name(relation).replace("_", " ")
        entity_name = entity_label or self._extract_local_name(entity)

        # Create combined text for entity+relation
        combined = f"{entity_name} {relation_name}"

        # Compute embeddings
        embeddings = self.model.encode([question, combined])

        # Compute cosine similarity
        similarity = self._cosine_similarity(embeddings[0], embeddings[1])

        # Normalize to [0, 1]
        return (similarity + 1) / 2

    def _extract_local_name(self, uri: str) -> str:
        """Extract local name from URI."""
        if "/" in uri:
            return uri.split("/")[-1]
        elif "#" in uri:
            return uri.split("#")[-1]
        return uri

    def _cosine_similarity(self, a, b) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


class CombinedRelevanceScorer(RelevanceScorer):
    """Combines multiple scoring strategies."""

    def __init__(
        self,
        keyword_weight: float = 0.6,
        embedding_weight: float = 0.4,
        use_embeddings: bool = False,
    ):
        """Initialize combined scorer.

        Args:
            keyword_weight: Weight for keyword-based scoring
            embedding_weight: Weight for embedding-based scoring
            use_embeddings: Whether to use embedding scorer
        """
        self.keyword_scorer = KeywordRelevanceScorer()
        self.keyword_weight = keyword_weight

        if use_embeddings:
            self.embedding_scorer = EmbeddingRelevanceScorer()
            self.embedding_weight = embedding_weight
        else:
            self.embedding_scorer = None
            self.embedding_weight = 0.0
            self.keyword_weight = 1.0

    def score(
        self,
        question: str,
        entity: str,
        relation: str,
        entity_label: Optional[str] = None,
    ) -> float:
        """Compute combined relevance score.

        Args:
            question: The question text
            entity: Entity URI
            relation: Relation URI
            entity_label: Optional entity label

        Returns:
            Combined relevance score between 0 and 1
        """
        keyword_score = self.keyword_scorer.score(
            question, entity, relation, entity_label
        )

        if self.embedding_scorer:
            embedding_score = self.embedding_scorer.score(
                question, entity, relation, entity_label
            )
            return (
                self.keyword_weight * keyword_score
                + self.embedding_weight * embedding_score
            )

        return keyword_score
