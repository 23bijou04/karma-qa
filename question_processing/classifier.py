"""Question type classifier for 2WikiMultiHopQA question types."""

import re
import logging
from typing import Optional

from ..models import QuestionType
from ..config import QUESTION_TYPE_PATTERNS

logger = logging.getLogger(__name__)


class QuestionClassifier:
    """Classifies questions into 2WikiMultiHopQA types.

    Question types:
    - Comparison: Compare properties of two entities
    - Inference: Use logical rules (e.g., grandfather = father's father)
    - Compositional: Chain of relations without inference
    - Bridge-Comparison: Find bridge entities then compare
    """

    def __init__(self, llm_client: Optional[object] = None):
        """Initialize classifier.

        Args:
            llm_client: Optional LLM client for fallback classification
        """
        self.llm_client = llm_client
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for classification."""
        self.compiled_patterns: dict[QuestionType, list[re.Pattern]] = {}

        for type_name, patterns in QUESTION_TYPE_PATTERNS.items():
            question_type = QuestionType(type_name)
            self.compiled_patterns[question_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def classify(self, question: str) -> QuestionType:
        """Classify a question into one of the 2WikiMultiHopQA types.

        Args:
            question: The question text

        Returns:
            QuestionType enum value
        """
        # Try regex-based classification first
        question_type = self._classify_by_patterns(question)
        if question_type != QuestionType.UNKNOWN:
            return question_type

        # Try heuristic-based classification
        question_type = self._classify_by_heuristics(question)
        if question_type != QuestionType.UNKNOWN:
            return question_type

        # Fall back to LLM if available
        if self.llm_client:
            return self._classify_by_llm(question)

        return QuestionType.UNKNOWN

    def _classify_by_patterns(self, question: str) -> QuestionType:
        """Classify using compiled regex patterns."""
        # Check patterns in priority order
        priority_order = [
            QuestionType.BRIDGE_COMPARISON,  # Most specific
            QuestionType.INFERENCE,
            QuestionType.COMPARISON,
            QuestionType.COMPOSITIONAL,  # Most general
        ]

        for question_type in priority_order:
            patterns = self.compiled_patterns.get(question_type, [])
            for pattern in patterns:
                if pattern.search(question):
                    logger.debug(f"Pattern match: {question_type.value} for '{question}'")
                    return question_type

        return QuestionType.UNKNOWN

    def _classify_by_heuristics(self, question: str) -> QuestionType:
        """Classify using structural heuristics."""
        question_lower = question.lower()

        # Comparison indicators
        comparison_words = {
            "both", "same", "more", "less", "older", "younger",
            "first", "earlier", "later", "longer", "shorter",
            "compare", "versus", "vs", "or"
        }

        # Inference indicators (family relations)
        inference_relations = {
            "grandfather", "grandmother", "grandchild", "grandson", "granddaughter",
            "father-in-law", "mother-in-law", "brother-in-law", "sister-in-law",
            "uncle", "aunt", "nephew", "niece", "cousin",
            "paternal", "maternal"
        }

        # Bridge question indicators
        bridge_phrases = [
            "the director of",
            "the producer of",
            "the founder of",
            "the author of",
            "the composer of",
            "who directed",
            "who produced",
            "who founded",
        ]

        # Check for inference relations
        for relation in inference_relations:
            if relation in question_lower:
                # Check if it's also a comparison
                if any(word in question_lower for word in comparison_words):
                    return QuestionType.BRIDGE_COMPARISON
                return QuestionType.INFERENCE

        # Check for bridge phrases with comparison
        has_bridge = any(phrase in question_lower for phrase in bridge_phrases)
        has_comparison = any(word in question_lower for word in comparison_words)

        if has_bridge and has_comparison:
            return QuestionType.BRIDGE_COMPARISON
        elif has_comparison and not has_bridge:
            return QuestionType.COMPARISON
        elif has_bridge:
            return QuestionType.COMPOSITIONAL

        # Check for "of the" chains (compositional)
        if question_lower.count(" of ") >= 2:
            return QuestionType.COMPOSITIONAL
        elif " of the " in question_lower or " of " in question_lower:
            if "who" in question_lower or "what" in question_lower:
                return QuestionType.COMPOSITIONAL

        return QuestionType.UNKNOWN

    def _classify_by_llm(self, question: str) -> QuestionType:
        """Classify using LLM as fallback."""
        if not self.llm_client:
            return QuestionType.UNKNOWN

        prompt = f"""Classify this multi-hop question into one of these categories:
1. comparison - Compares properties of two entities (e.g., "Who was born first, A or B?")
2. inference - Uses logical rules like family relations (e.g., "Who is the grandfather of X?")
3. compositional - Chains relations without inference (e.g., "Who is the director of the film that X starred in?")
4. bridge_comparison - Finds bridge entities then compares (e.g., "Which film has the director born first?")

Question: {question}

Answer with just the category name (comparison, inference, compositional, or bridge_comparison):"""

        try:
            response = self.llm_client.generate(prompt)
            response_lower = response.strip().lower()

            for qtype in QuestionType:
                if qtype.value in response_lower:
                    return qtype
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

        return QuestionType.UNKNOWN

    def get_classification_confidence(self, question: str) -> tuple[QuestionType, float]:
        """Classify question and return confidence score.

        Args:
            question: The question text

        Returns:
            Tuple of (QuestionType, confidence score 0-1)
        """
        # Pattern match = high confidence
        question_type = self._classify_by_patterns(question)
        if question_type != QuestionType.UNKNOWN:
            return question_type, 0.9

        # Heuristic match = medium confidence
        question_type = self._classify_by_heuristics(question)
        if question_type != QuestionType.UNKNOWN:
            return question_type, 0.7

        # LLM classification = lower confidence
        if self.llm_client:
            question_type = self._classify_by_llm(question)
            if question_type != QuestionType.UNKNOWN:
                return question_type, 0.6

        return QuestionType.UNKNOWN, 0.0

    def extract_comparison_entities(self, question: str) -> list[str]:
        """Extract entity names being compared in a comparison question.

        Args:
            question: The question text

        Returns:
            List of entity names (usually 2)
        """
        entities = []

        # Pattern: "X or Y"
        or_match = re.search(r"([A-Z][^,?]+?)\s+or\s+([A-Z][^,?]+?)(?:\?|$)", question)
        if or_match:
            entities.extend([or_match.group(1).strip(), or_match.group(2).strip()])
            return entities

        # Pattern: "both X and Y"
        both_match = re.search(
            r"both\s+([A-Z][^,]+?)\s+and\s+([A-Z][^,?]+?)(?:\s|,|\?|$)",
            question,
            re.IGNORECASE,
        )
        if both_match:
            entities.extend([both_match.group(1).strip(), both_match.group(2).strip()])
            return entities

        # Pattern: "X, Y, and Z"
        comma_match = re.findall(r"([A-Z][A-Za-z\s]+?)(?:,|and|\?)", question)
        if len(comma_match) >= 2:
            return [e.strip() for e in comma_match[:2]]

        return entities

    def extract_inference_relation(self, question: str) -> Optional[str]:
        """Extract the inference relation from an inference question.

        Args:
            question: The question text

        Returns:
            Relation name or None
        """
        question_lower = question.lower()

        inference_relations = {
            "paternal grandfather": "paternal_grandfather",
            "paternal grandmother": "paternal_grandmother",
            "maternal grandfather": "maternal_grandfather",
            "maternal grandmother": "maternal_grandmother",
            "grandfather": "grandfather",
            "grandmother": "grandmother",
            "grandchild": "grandchild",
            "grandson": "grandchild",
            "granddaughter": "grandchild",
            "father-in-law": "father-in-law",
            "mother-in-law": "mother-in-law",
            "brother-in-law": "sibling-in-law",
            "sister-in-law": "sibling-in-law",
            "son-in-law": "child-in-law",
            "daughter-in-law": "child-in-law",
            "uncle": "uncle",
            "aunt": "aunt",
        }

        for phrase, relation in inference_relations.items():
            if phrase in question_lower:
                return relation

        return None
