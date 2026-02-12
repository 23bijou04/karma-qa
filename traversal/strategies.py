"""Question-type specific traversal strategies."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..models import QuestionType, Entity, Triple, Subgraph, SubQuestion
from ..dbpedia.client import DBpediaSPARQLClient
from ..dbpedia.property_mapper import PropertyMapper
from .ranked_bfs import RankedBFS
from .relevance_scorer import KeywordRelevanceScorer

logger = logging.getLogger(__name__)


class TraversalStrategy(ABC):
    """Abstract base class for traversal strategies."""

    def __init__(
        self,
        sparql_client: DBpediaSPARQLClient,
        max_hops: int = 3,
        beam_width: int = 10,
    ):
        self.client = sparql_client
        self.max_hops = max_hops
        self.beam_width = beam_width
        self.property_mapper = PropertyMapper()
        self.bfs = RankedBFS(
            sparql_client,
            relevance_scorer=KeywordRelevanceScorer(),
            max_hops=max_hops,
            beam_width=beam_width,
        )

    @abstractmethod
    def traverse(
        self,
        question: str,
        entities: list[Entity],
        sub_questions: list[SubQuestion],
    ) -> Subgraph:
        """Execute traversal strategy.

        Args:
            question: Original question text
            entities: Extracted entities
            sub_questions: Decomposed sub-questions

        Returns:
            Subgraph with retrieved facts
        """
        pass


class ComparisonStrategy(TraversalStrategy):
    """Strategy for comparison questions.

    Comparison questions compare properties of two entities.
    Example: "Who was born first, Einstein or Lincoln?"

    Traversal: Get the comparison property for both entities in parallel.
    """

    def traverse(
        self,
        question: str,
        entities: list[Entity],
        sub_questions: list[SubQuestion],
    ) -> Subgraph:
        """Execute comparison traversal."""
        subgraph = Subgraph()

        if len(entities) < 2:
            logger.warning("Comparison requires at least 2 entities")
            return subgraph

        # Identify the property being compared
        comparison_property = self._extract_comparison_property(question)

        # Get DBpedia properties
        dbp_properties = self.property_mapper.get_property_from_name(comparison_property)

        # Query both entities for the property
        for entity in entities[:2]:
            subgraph.add_entity(entity)
            subgraph.seed_entities.append(entity.uri)

            for prop in dbp_properties:
                value = self.client.get_property_value(entity.uri, prop)
                if value:
                    triple = Triple(
                        subject=entity.uri,
                        subject_label=entity.label,
                        predicate=prop,
                        predicate_label=comparison_property,
                        obj=value,
                        confidence=1.0,
                    )
                    subgraph.add_triple(triple)
                    break

        subgraph.hops = 1
        return subgraph

    def _extract_comparison_property(self, question: str) -> str:
        """Extract what property is being compared."""
        question_lower = question.lower()

        property_patterns = {
            "birth": "birth date",
            "born": "birth date",
            "death": "death date",
            "died": "death date",
            "lived longer": "death date",
            "older": "birth date",
            "younger": "birth date",
            "nationality": "nationality",
            "country": "nationality",
            "released": "release date",
        }

        for keyword, prop in property_patterns.items():
            if keyword in question_lower:
                return prop

        return "birth date"  # Default for person comparisons


class InferenceStrategy(TraversalStrategy):
    """Strategy for inference questions.

    Inference questions use logical rules (e.g., grandfather = father's father).
    Example: "Who is the paternal grandfather of X?"

    Traversal: Follow the inference rule chain.
    """

    def traverse(
        self,
        question: str,
        entities: list[Entity],
        sub_questions: list[SubQuestion],
    ) -> Subgraph:
        """Execute inference traversal."""
        subgraph = Subgraph()

        if not entities:
            logger.warning("Inference requires at least 1 entity")
            return subgraph

        main_entity = entities[0]
        subgraph.add_entity(main_entity)
        subgraph.seed_entities.append(main_entity.uri)

        # Identify the inference relation
        inference_relation = self._extract_inference_relation(question)

        if not inference_relation:
            # Fall back to generic BFS
            return self.bfs.traverse(question, entities)

        # Get the rule chain
        rule = self.property_mapper.get_inference_rule(inference_relation)
        if not rule:
            return self.bfs.traverse(question, entities)

        rel1, rel2 = rule

        # First hop
        props1 = self.property_mapper.get_property_from_name(rel1)
        intermediate_uri = None

        for prop in props1:
            value = self.client.get_property_value(main_entity.uri, prop)
            if value:
                # Check if value is an entity URI
                if value.startswith("http://dbpedia.org/resource/"):
                    intermediate_uri = value
                    triple = Triple(
                        subject=main_entity.uri,
                        subject_label=main_entity.label,
                        predicate=prop,
                        predicate_label=rel1,
                        obj=value,
                        confidence=1.0,
                    )
                    subgraph.add_triple(triple)

                    # Get intermediate entity details
                    intermediate_entity = self.client.get_entity_by_uri(value)
                    if intermediate_entity:
                        subgraph.add_entity(intermediate_entity)
                    break

        # Second hop
        if intermediate_uri:
            props2 = self.property_mapper.get_property_from_name(rel2)

            for prop in props2:
                value = self.client.get_property_value(intermediate_uri, prop)
                if value:
                    triple = Triple(
                        subject=intermediate_uri,
                        predicate=prop,
                        predicate_label=rel2,
                        obj=value,
                        confidence=1.0,
                    )
                    subgraph.add_triple(triple)

                    # Get final entity if it's a resource
                    if value.startswith("http://dbpedia.org/resource/"):
                        final_entity = self.client.get_entity_by_uri(value)
                        if final_entity:
                            subgraph.add_entity(final_entity)
                    break

        subgraph.hops = 2
        return subgraph

    def _extract_inference_relation(self, question: str) -> Optional[str]:
        """Extract inference relation from question."""
        question_lower = question.lower()

        relations = {
            "paternal grandfather": "paternal_grandfather",
            "paternal grandmother": "paternal_grandmother",
            "maternal grandfather": "maternal_grandfather",
            "maternal grandmother": "maternal_grandmother",
            "grandfather": "paternal_grandfather",  # Default to paternal
            "grandmother": "paternal_grandmother",
            "grandchild": "grandchild",
            "father-in-law": "father-in-law",
            "mother-in-law": "mother-in-law",
        }

        for phrase, relation in relations.items():
            if phrase in question_lower:
                return relation

        return None


class CompositionalStrategy(TraversalStrategy):
    """Strategy for compositional questions.

    Compositional questions chain relations without inference rules.
    Example: "Who founded the company that distributed La La Land?"

    Traversal: Find bridge entity, then target property.
    """

    def traverse(
        self,
        question: str,
        entities: list[Entity],
        sub_questions: list[SubQuestion],
    ) -> Subgraph:
        """Execute compositional traversal."""
        subgraph = Subgraph()

        if not entities:
            logger.warning("Compositional requires at least 1 entity")
            return subgraph

        # Add seed entity
        main_entity = entities[0]
        subgraph.add_entity(main_entity)
        subgraph.seed_entities.append(main_entity.uri)

        # Use sub-questions if available
        if sub_questions and len(sub_questions) >= 2:
            return self._traverse_with_subquestions(
                question, main_entity, sub_questions, subgraph
            )

        # Otherwise, do generic 2-hop traversal
        return self.bfs.traverse(question, entities)

    def _traverse_with_subquestions(
        self,
        question: str,
        main_entity: Entity,
        sub_questions: list[SubQuestion],
        subgraph: Subgraph,
    ) -> Subgraph:
        """Traverse following decomposed sub-questions."""
        current_uri = main_entity.uri

        for i, sub_q in enumerate(sub_questions[:-1]):  # Skip comparison step
            # Extract relation from sub-question
            relation = self._extract_relation_from_subq(sub_q.text)

            if relation:
                props = self.property_mapper.get_property_from_name(relation)

                for prop in props:
                    value = self.client.get_property_value(current_uri, prop)
                    if value:
                        triple = Triple(
                            subject=current_uri,
                            predicate=prop,
                            predicate_label=relation,
                            obj=value,
                            confidence=1.0,
                        )
                        subgraph.add_triple(triple)

                        # Update sub-question answer
                        sub_q.answer = value

                        if value.startswith("http://dbpedia.org/resource/"):
                            entity = self.client.get_entity_by_uri(value)
                            if entity:
                                subgraph.add_entity(entity)
                            current_uri = value
                        break

        subgraph.hops = len(sub_questions) - 1
        return subgraph

    def _extract_relation_from_subq(self, sub_question: str) -> Optional[str]:
        """Extract relation from sub-question text."""
        sub_q_lower = sub_question.lower()

        relation_patterns = {
            "director": "director",
            "directed": "director",
            "producer": "producer",
            "produced": "producer",
            "founder": "founded by",
            "founded": "founded by",
            "distributor": "distributor",
            "distributed": "distributor",
            "composer": "composer",
            "composed": "composer",
        }

        for keyword, relation in relation_patterns.items():
            if keyword in sub_q_lower:
                return relation

        return None


class BridgeComparisonStrategy(TraversalStrategy):
    """Strategy for bridge-comparison questions.

    Bridge-comparison combines bridge entity finding with comparison.
    Example: "Which film has the director born first, La La Land or Tenet?"

    Traversal: Find bridge entities for both, then compare property.
    """

    def traverse(
        self,
        question: str,
        entities: list[Entity],
        sub_questions: list[SubQuestion],
    ) -> Subgraph:
        """Execute bridge-comparison traversal."""
        subgraph = Subgraph()

        if len(entities) < 2:
            logger.warning("Bridge-comparison requires at least 2 entities")
            return subgraph

        # Add seed entities
        for entity in entities[:2]:
            subgraph.add_entity(entity)
            subgraph.seed_entities.append(entity.uri)

        # Identify bridge relation and comparison property
        bridge_relation = self._extract_bridge_relation(question)
        comparison_property = self._extract_comparison_property(question)

        bridge_props = self.property_mapper.get_property_from_name(bridge_relation)
        comparison_props = self.property_mapper.get_property_from_name(comparison_property)

        # For each entity, find bridge then comparison property
        for entity in entities[:2]:
            # First hop: find bridge entity
            bridge_uri = None

            for prop in bridge_props:
                value = self.client.get_property_value(entity.uri, prop)
                if value and value.startswith("http://dbpedia.org/resource/"):
                    bridge_uri = value
                    triple = Triple(
                        subject=entity.uri,
                        subject_label=entity.label,
                        predicate=prop,
                        predicate_label=bridge_relation,
                        obj=value,
                        confidence=1.0,
                    )
                    subgraph.add_triple(triple)

                    bridge_entity = self.client.get_entity_by_uri(value)
                    if bridge_entity:
                        subgraph.add_entity(bridge_entity)
                    break

            # Second hop: get comparison property of bridge
            if bridge_uri:
                for prop in comparison_props:
                    value = self.client.get_property_value(bridge_uri, prop)
                    if value:
                        triple = Triple(
                            subject=bridge_uri,
                            predicate=prop,
                            predicate_label=comparison_property,
                            obj=value,
                            confidence=1.0,
                        )
                        subgraph.add_triple(triple)
                        break

        subgraph.hops = 2
        return subgraph

    def _extract_bridge_relation(self, question: str) -> str:
        """Extract bridge relation from question."""
        question_lower = question.lower()

        if "director" in question_lower:
            return "director"
        elif "producer" in question_lower:
            return "producer"
        elif "founder" in question_lower:
            return "founder"
        elif "author" in question_lower:
            return "author"

        return "director"  # Default

    def _extract_comparison_property(self, question: str) -> str:
        """Extract comparison property from question."""
        question_lower = question.lower()

        if "born" in question_lower:
            return "birth date"
        elif "died" in question_lower:
            return "death date"
        elif "country" in question_lower or "nationality" in question_lower:
            return "nationality"

        return "birth date"  # Default


def get_strategy_for_question_type(
    question_type: QuestionType,
    sparql_client: DBpediaSPARQLClient,
    max_hops: int = 3,
    beam_width: int = 10,
) -> TraversalStrategy:
    """Get the appropriate traversal strategy for a question type.

    Args:
        question_type: The classified question type
        sparql_client: DBpedia SPARQL client
        max_hops: Maximum traversal depth
        beam_width: Beam width for BFS

    Returns:
        TraversalStrategy instance
    """
    strategies = {
        QuestionType.COMPARISON: ComparisonStrategy,
        QuestionType.INFERENCE: InferenceStrategy,
        QuestionType.COMPOSITIONAL: CompositionalStrategy,
        QuestionType.BRIDGE_COMPARISON: BridgeComparisonStrategy,
    }

    strategy_class = strategies.get(question_type, CompositionalStrategy)
    return strategy_class(sparql_client, max_hops, beam_width)
