"""Ranked Breadth-First Search for graph traversal."""

import logging
from heapq import heappush, heappop
from typing import Optional, Set

from ..models import Entity, Triple, Subgraph
from ..dbpedia.client import DBpediaSPARQLClient
from .relevance_scorer import RelevanceScorer, KeywordRelevanceScorer

logger = logging.getLogger(__name__)


class RankedBFS:
    """Relevance-ranked Breadth-First Search for knowledge graph traversal.

    Explores the knowledge graph from seed entities, prioritizing paths
    that are most relevant to the question being answered.
    """

    def __init__(
        self,
        sparql_client: DBpediaSPARQLClient,
        relevance_scorer: Optional[RelevanceScorer] = None,
        max_hops: int = 3,
        beam_width: int = 10,
        max_nodes: int = 100,
        relevance_threshold: float = 0.1,
    ):
        """Initialize Ranked BFS.

        Args:
            sparql_client: DBpedia SPARQL client
            relevance_scorer: Scorer for ranking neighbors
            max_hops: Maximum traversal depth
            beam_width: Number of neighbors to keep per hop
            max_nodes: Maximum total nodes to explore
            relevance_threshold: Minimum relevance score to include
        """
        self.client = sparql_client
        self.scorer = relevance_scorer or KeywordRelevanceScorer()
        self.max_hops = max_hops
        self.beam_width = beam_width
        self.max_nodes = max_nodes
        self.relevance_threshold = relevance_threshold

    def traverse(
        self,
        question: str,
        seed_entities: list[Entity],
        relation_filter: Optional[Set[str]] = None,
    ) -> Subgraph:
        """Traverse the knowledge graph from seed entities.

        Args:
            question: The question to answer
            seed_entities: Starting entities for traversal
            relation_filter: Optional set of relations to follow

        Returns:
            Subgraph containing traversed entities and triples
        """
        subgraph = Subgraph(seed_entities=[e.uri for e in seed_entities])

        # Add seed entities
        for entity in seed_entities:
            subgraph.add_entity(entity)

        # Priority queue: (-score, hop_depth, entity_uri)
        # Negative score for max-heap behavior
        priority_queue = []
        for entity in seed_entities:
            heappush(priority_queue, (-1.0, 0, entity.uri))

        # Track visited entities and their best scores
        visited: dict[str, float] = {e.uri: 1.0 for e in seed_entities}

        while priority_queue and len(subgraph.entities) < self.max_nodes:
            neg_score, hop_depth, current_uri = heappop(priority_queue)
            current_score = -neg_score

            if hop_depth >= self.max_hops:
                continue

            # Get neighbors from DBpedia
            try:
                neighbors = self.client.get_neighbors(
                    current_uri,
                    relation_types=relation_filter,
                    direction="both",
                    limit=50,
                )
            except Exception as e:
                logger.warning(f"Failed to get neighbors for {current_uri}: {e}")
                continue

            # Score and rank neighbors
            scored_neighbors = []
            for relation, neighbor_uri, _ in neighbors:
                # Skip already visited with better score
                if neighbor_uri in visited and visited[neighbor_uri] >= current_score:
                    continue

                # Get neighbor label for better scoring
                neighbor_entity = self.client.get_entity_by_uri(neighbor_uri)
                neighbor_label = neighbor_entity.label if neighbor_entity else None

                # Score relevance
                relevance = self.scorer.score(
                    question, neighbor_uri, relation, neighbor_label
                )

                if relevance >= self.relevance_threshold:
                    combined_score = current_score * relevance
                    scored_neighbors.append(
                        (combined_score, relation, neighbor_uri, neighbor_entity)
                    )

            # Sort by score and keep top beam_width
            scored_neighbors.sort(reverse=True, key=lambda x: x[0])
            top_neighbors = scored_neighbors[: self.beam_width]

            # Add to subgraph and queue
            for combined_score, relation, neighbor_uri, neighbor_entity in top_neighbors:
                # Add entity
                if neighbor_entity:
                    subgraph.add_entity(neighbor_entity)
                else:
                    subgraph.add_entity(
                        Entity(uri=neighbor_uri, label=neighbor_uri.split("/")[-1])
                    )

                # Add triple
                triple = Triple(
                    subject=current_uri,
                    predicate=relation,
                    obj=neighbor_uri,
                    confidence=combined_score,
                    source="dbpedia",
                )
                subgraph.add_triple(triple)

                # Update visited and queue
                if (
                    neighbor_uri not in visited
                    or visited[neighbor_uri] < combined_score
                ):
                    visited[neighbor_uri] = combined_score
                    heappush(
                        priority_queue,
                        (-combined_score, hop_depth + 1, neighbor_uri),
                    )

        subgraph.hops = self.max_hops
        return subgraph

    def find_path(
        self,
        question: str,
        start_entity: Entity,
        target_properties: list[str],
        max_depth: int = 2,
    ) -> list[Triple]:
        """Find a path through the graph following specific properties.

        Used for inference questions where we need to follow a specific
        chain of relations (e.g., father -> father for grandfather).

        Args:
            question: The question text
            start_entity: Starting entity
            target_properties: Properties to follow in order
            max_depth: Maximum path length

        Returns:
            List of triples forming the path
        """
        path = []
        current_uri = start_entity.uri

        for depth, properties in enumerate(target_properties[:max_depth]):
            if not isinstance(properties, list):
                properties = [properties]

            # Try each property variant
            found = False
            for prop in properties:
                # Query for this property
                try:
                    neighbors = self.client.get_neighbors(
                        current_uri,
                        relation_types={prop},
                        direction="out",
                        limit=10,
                    )
                except Exception as e:
                    logger.warning(f"Failed to query property {prop}: {e}")
                    continue

                if neighbors:
                    # Take the most relevant neighbor
                    best_neighbor = None
                    best_score = -1

                    for relation, neighbor_uri, _ in neighbors:
                        score = self.scorer.score(question, neighbor_uri, relation)
                        if score > best_score:
                            best_score = score
                            best_neighbor = (relation, neighbor_uri)

                    if best_neighbor:
                        relation, neighbor_uri = best_neighbor
                        triple = Triple(
                            subject=current_uri,
                            predicate=relation,
                            obj=neighbor_uri,
                            confidence=best_score,
                        )
                        path.append(triple)
                        current_uri = neighbor_uri
                        found = True
                        break

            if not found:
                logger.warning(
                    f"Could not find path at depth {depth} for properties {properties}"
                )
                break

        return path

    def get_entity_context(
        self,
        entity: Entity,
        properties: Optional[list[str]] = None,
    ) -> Subgraph:
        """Get immediate context for an entity (1-hop neighbors).

        Args:
            entity: The entity to get context for
            properties: Optional specific properties to retrieve

        Returns:
            Subgraph with entity's immediate neighborhood
        """
        subgraph = Subgraph(seed_entities=[entity.uri])
        subgraph.add_entity(entity)

        try:
            # Get properties
            triples = self.client.get_entity_properties(entity.uri, properties)

            for triple in triples:
                subgraph.add_triple(triple)

                # Try to get entity details for object
                if triple.obj.startswith("http://dbpedia.org/resource/"):
                    obj_entity = self.client.get_entity_by_uri(triple.obj)
                    if obj_entity:
                        subgraph.add_entity(obj_entity)

        except Exception as e:
            logger.warning(f"Failed to get context for {entity.uri}: {e}")

        return subgraph

    def traverse_for_comparison(
        self,
        question: str,
        entity1: Entity,
        entity2: Entity,
        comparison_property: str,
    ) -> Subgraph:
        """Specialized traversal for comparison questions.

        Gets the comparison property for both entities.

        Args:
            question: The question text
            entity1: First entity to compare
            entity2: Second entity to compare
            comparison_property: Property to compare

        Returns:
            Subgraph with both entities and their property values
        """
        from ..dbpedia.property_mapper import PropertyMapper

        subgraph = Subgraph(seed_entities=[entity1.uri, entity2.uri])
        subgraph.add_entity(entity1)
        subgraph.add_entity(entity2)

        # Get DBpedia property URIs
        mapper = PropertyMapper()
        dbp_properties = mapper.get_property_from_name(comparison_property)

        for entity in [entity1, entity2]:
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
                    break  # Found value, no need to try other property variants

        return subgraph
