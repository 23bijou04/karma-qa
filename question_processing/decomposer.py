"""Question decomposition for multi-hop reasoning."""

import re
import logging
from typing import Optional

from ..models import QuestionType, SubQuestion
from ..config import INFERENCE_RULES

logger = logging.getLogger(__name__)


class QuestionDecomposer:
    """Decomposes multi-hop questions into sub-questions.

    Handles the four question types from 2WikiMultiHopQA:
    - Comparison: Split into parallel property retrieval
    - Inference: Chain based on logical rules
    - Compositional: Sequential relation following
    - Bridge-Comparison: Find bridges then compare
    """

    def __init__(self, llm_client: Optional[object] = None):
        """Initialize decomposer.

        Args:
            llm_client: Optional LLM client for complex decomposition
        """
        self.llm_client = llm_client
        self.inference_rules = INFERENCE_RULES

    def decompose(
        self, question: str, question_type: QuestionType, entities: list[str] = None
    ) -> list[SubQuestion]:
        """Decompose a question into sub-questions based on its type.

        Args:
            question: The original question text
            question_type: The classified question type
            entities: Optional list of extracted entities

        Returns:
            List of SubQuestion objects in execution order
        """
        entities = entities or []

        if question_type == QuestionType.COMPARISON:
            return self._decompose_comparison(question, entities)
        elif question_type == QuestionType.INFERENCE:
            return self._decompose_inference(question, entities)
        elif question_type == QuestionType.COMPOSITIONAL:
            return self._decompose_compositional(question, entities)
        elif question_type == QuestionType.BRIDGE_COMPARISON:
            return self._decompose_bridge_comparison(question, entities)
        else:
            # Unknown type - try generic decomposition
            return self._decompose_generic(question, entities)

    def _decompose_comparison(
        self, question: str, entities: list[str]
    ) -> list[SubQuestion]:
        """Decompose comparison question into parallel property queries.

        Example: "Who was born first, Einstein or Lincoln?"
        -> Q1: "When was Einstein born?"
        -> Q2: "When was Lincoln born?"
        -> Q3: "Compare the birth dates"
        """
        sub_questions = []

        # Extract the property being compared
        property_name = self._extract_comparison_property(question)

        # Extract entities to compare
        if len(entities) < 2:
            entities = self._extract_comparison_entities(question)

        if len(entities) >= 2:
            # Create parallel sub-questions for each entity
            for i, entity in enumerate(entities[:2]):
                sub_q = SubQuestion(
                    text=f"What is the {property_name} of {entity}?",
                    order=i,
                    depends_on=[],
                )
                sub_questions.append(sub_q)

            # Create comparison sub-question
            compare_q = SubQuestion(
                text=f"Compare the {property_name} of {entities[0]} and {entities[1]}",
                order=2,
                depends_on=[0, 1],
            )
            sub_questions.append(compare_q)
        else:
            # Fallback: single sub-question
            sub_questions.append(SubQuestion(text=question, order=0, depends_on=[]))

        return sub_questions

    def _decompose_inference(
        self, question: str, entities: list[str]
    ) -> list[SubQuestion]:
        """Decompose inference question using logical rules.

        Example: "Who is the paternal grandfather of X?"
        -> Q1: "Who is the father of X?"
        -> Q2: "Who is the father of [answer from Q1]?"
        """
        sub_questions = []

        # Extract the inference relation
        relation = self._extract_inference_relation(question)
        if not relation:
            return [SubQuestion(text=question, order=0, depends_on=[])]

        # Get the rule for this relation
        rule = self.inference_rules.get(relation)
        if not rule:
            return [SubQuestion(text=question, order=0, depends_on=[])]

        rel1, rel2 = rule

        # Extract the main entity
        main_entity = entities[0] if entities else self._extract_main_entity(question)

        if main_entity:
            # First hop
            sub_q1 = SubQuestion(
                text=f"Who is the {rel1} of {main_entity}?",
                order=0,
                depends_on=[],
            )
            sub_questions.append(sub_q1)

            # Second hop (depends on first)
            sub_q2 = SubQuestion(
                text=f"Who is the {rel2} of [answer from previous question]?",
                order=1,
                depends_on=[0],
            )
            sub_questions.append(sub_q2)
        else:
            sub_questions.append(SubQuestion(text=question, order=0, depends_on=[]))

        return sub_questions

    def _decompose_compositional(
        self, question: str, entities: list[str]
    ) -> list[SubQuestion]:
        """Decompose compositional question into sequential relation queries.

        Example: "Who is the founder of the company that distributed La La Land?"
        -> Q1: "What company distributed La La Land?"
        -> Q2: "Who founded [answer from Q1]?"
        """
        sub_questions = []

        # Try to identify the chain of relations
        # Pattern: "What/Who is the [rel2] of the [type] that [rel1] [entity]?"
        pattern = r"(?:who|what)\s+is\s+the\s+(\w+)\s+of\s+the\s+(\w+)\s+that\s+(\w+)\s+(.+?)\??"
        match = re.search(pattern, question, re.IGNORECASE)

        if match:
            rel2 = match.group(1)  # founder
            entity_type = match.group(2)  # company
            rel1 = match.group(3)  # distributed
            entity = match.group(4)  # La La Land

            # First sub-question: find the bridge entity
            sub_q1 = SubQuestion(
                text=f"What {entity_type} {rel1} {entity}?",
                order=0,
                depends_on=[],
            )
            sub_questions.append(sub_q1)

            # Second sub-question: find the target
            sub_q2 = SubQuestion(
                text=f"Who is the {rel2} of [answer from previous question]?",
                order=1,
                depends_on=[0],
            )
            sub_questions.append(sub_q2)
        else:
            # Try simpler "of the" decomposition
            sub_questions = self._decompose_of_chain(question, entities)

        if not sub_questions:
            sub_questions.append(SubQuestion(text=question, order=0, depends_on=[]))

        return sub_questions

    def _decompose_bridge_comparison(
        self, question: str, entities: list[str]
    ) -> list[SubQuestion]:
        """Decompose bridge-comparison question.

        Example: "Which film has the director born first, La La Land or Tenet?"
        -> Q1: "Who directed La La Land?"
        -> Q2: "Who directed Tenet?"
        -> Q3: "When was [answer from Q1] born?"
        -> Q4: "When was [answer from Q2] born?"
        -> Q5: "Compare birth dates"
        """
        sub_questions = []

        # Extract entities and bridge relation
        if len(entities) < 2:
            entities = self._extract_comparison_entities(question)

        bridge_relation = self._extract_bridge_relation(question)
        comparison_property = self._extract_comparison_property(question)

        if len(entities) >= 2 and bridge_relation:
            # Find bridge entities for both
            for i, entity in enumerate(entities[:2]):
                bridge_q = SubQuestion(
                    text=f"Who is the {bridge_relation} of {entity}?",
                    order=i,
                    depends_on=[],
                )
                sub_questions.append(bridge_q)

            # Get comparison property for bridge entities
            for i, entity in enumerate(entities[:2]):
                prop_q = SubQuestion(
                    text=f"What is the {comparison_property} of the {bridge_relation} of {entity}?",
                    order=i + 2,
                    depends_on=[i],
                )
                sub_questions.append(prop_q)

            # Final comparison
            compare_q = SubQuestion(
                text=f"Compare the {comparison_property} of the directors",
                order=4,
                depends_on=[2, 3],
            )
            sub_questions.append(compare_q)
        else:
            sub_questions.append(SubQuestion(text=question, order=0, depends_on=[]))

        return sub_questions

    def _decompose_generic(
        self, question: str, entities: list[str]
    ) -> list[SubQuestion]:
        """Generic decomposition using LLM or simple heuristics."""
        if self.llm_client:
            return self._decompose_with_llm(question, entities)

        # Simple single-question fallback
        return [SubQuestion(text=question, order=0, depends_on=[])]

    def _decompose_with_llm(
        self, question: str, entities: list[str]
    ) -> list[SubQuestion]:
        """Use LLM to decompose complex questions."""
        prompt = f"""Break down this multi-hop question into simpler sub-questions.
Each sub-question should be answerable from a knowledge graph using 1-2 hops.
Number them in dependency order.

Question: {question}
Known entities: {', '.join(entities) if entities else 'None'}

Sub-questions (numbered, one per line):"""

        try:
            response = self.llm_client.generate(prompt)
            lines = response.strip().split("\n")

            sub_questions = []
            for i, line in enumerate(lines):
                # Remove numbering
                clean_line = re.sub(r"^\d+[.)]\s*", "", line.strip())
                if clean_line:
                    sub_q = SubQuestion(
                        text=clean_line,
                        order=i,
                        depends_on=[i - 1] if i > 0 else [],
                    )
                    sub_questions.append(sub_q)

            return sub_questions if sub_questions else [
                SubQuestion(text=question, order=0, depends_on=[])
            ]
        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}")
            return [SubQuestion(text=question, order=0, depends_on=[])]

    def _decompose_of_chain(
        self, question: str, entities: list[str]
    ) -> list[SubQuestion]:
        """Decompose questions with 'of the' chains."""
        sub_questions = []

        # Find "of the" or "of" patterns
        parts = re.split(r"\s+of\s+(?:the\s+)?", question, flags=re.IGNORECASE)

        if len(parts) >= 2:
            # Work backwards from the known entity
            for i in range(len(parts) - 1, 0, -1):
                sub_q = SubQuestion(
                    text=f"What is {parts[i-1].strip()} of {parts[i].strip()}?",
                    order=len(parts) - 1 - i,
                    depends_on=[len(parts) - 2 - i] if i < len(parts) - 1 else [],
                )
                sub_questions.append(sub_q)

        return sub_questions

    def _extract_comparison_property(self, question: str) -> str:
        """Extract the property being compared."""
        question_lower = question.lower()

        property_patterns = {
            r"born\s+(first|earlier|later)": "birth date",
            r"died\s+(first|earlier|later)": "death date",
            r"lived\s+longer": "lifespan",
            r"(older|younger)": "age",
            r"from\s+the\s+same\s+country": "country",
            r"same\s+nationality": "nationality",
            r"released\s+(first|earlier|later)": "release date",
        }

        for pattern, prop in property_patterns.items():
            if re.search(pattern, question_lower):
                return prop

        return "property"

    def _extract_comparison_entities(self, question: str) -> list[str]:
        """Extract entities being compared."""
        entities = []

        # Pattern: "X or Y?"
        or_match = re.search(r"([A-Z][^,?]+?)\s+or\s+([A-Z][^,?]+?)(?:\?|$)", question)
        if or_match:
            return [or_match.group(1).strip(), or_match.group(2).strip()]

        return entities

    def _extract_inference_relation(self, question: str) -> Optional[str]:
        """Extract inference relation from question."""
        question_lower = question.lower()

        relations = {
            "paternal grandfather": "paternal_grandfather",
            "paternal grandmother": "paternal_grandmother",
            "maternal grandfather": "maternal_grandfather",
            "maternal grandmother": "maternal_grandmother",
            "grandfather": "grandfather",
            "grandmother": "grandmother",
            "grandchild": "grandchild",
            "father-in-law": "father-in-law",
            "mother-in-law": "mother-in-law",
        }

        for phrase, relation in relations.items():
            if phrase in question_lower:
                return relation

        return None

    def _extract_main_entity(self, question: str) -> Optional[str]:
        """Extract the main entity from a question."""
        # Pattern: "of [Entity]?"
        match = re.search(r"of\s+([A-Z][^?]+?)\??$", question)
        if match:
            return match.group(1).strip()
        return None

    def _extract_bridge_relation(self, question: str) -> Optional[str]:
        """Extract the bridge relation from a bridge-comparison question."""
        question_lower = question.lower()

        relations = {
            "director": "director",
            "producer": "producer",
            "founder": "founder",
            "author": "author",
            "composer": "composer",
        }

        for word, relation in relations.items():
            if word in question_lower:
                return relation

        return None
