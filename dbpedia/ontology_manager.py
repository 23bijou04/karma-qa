"""DBpedia Ontology Manager for local schema access and entity-specific partitioning.

This module provides:
1. Local ontology loading and caching
2. Entity-specific schema partition extraction
3. Class hierarchy traversal
4. Property domain/range extraction
"""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from functools import lru_cache

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

logger = logging.getLogger(__name__)

# DBpedia namespaces
DBO = Namespace("http://dbpedia.org/ontology/")
DBP = Namespace("http://dbpedia.org/property/")
DBR = Namespace("http://dbpedia.org/resource/")


@dataclass
class OntologyClass:
    """Represents a class in the ontology."""
    uri: str
    label: Optional[str] = None
    comment: Optional[str] = None
    parent_classes: list[str] = field(default_factory=list)
    equivalent_classes: list[str] = field(default_factory=list)


@dataclass
class OntologyProperty:
    """Represents a property in the ontology."""
    uri: str
    label: Optional[str] = None
    comment: Optional[str] = None
    domain: list[str] = field(default_factory=list)
    range: list[str] = field(default_factory=list)
    property_type: str = "object"  # object, datatype, annotation


@dataclass
class EntitySchemaPartition:
    """Schema partition relevant to a specific entity."""
    entity_uri: str
    entity_types: list[str]
    applicable_properties: list[OntologyProperty]
    class_hierarchy: list[OntologyClass]

    def to_prompt_context(self, max_properties: int = 50) -> str:
        """Convert to LLM-friendly prompt context."""
        lines = []

        # Entity types
        lines.append("## Entity Types")
        for t in self.entity_types[:10]:
            short_type = t.split("/")[-1]
            lines.append(f"- {short_type}")

        # Class hierarchy
        if self.class_hierarchy:
            lines.append("\n## Class Hierarchy")
            for cls in self.class_hierarchy[:5]:
                label = cls.label or cls.uri.split("/")[-1]
                lines.append(f"- {label}")
                if cls.comment:
                    lines.append(f"  Description: {cls.comment[:100]}")

        # Applicable properties
        lines.append("\n## Available Properties")
        for prop in self.applicable_properties[:max_properties]:
            prop_name = prop.uri.split("/")[-1]
            prop_type = "→ Object" if prop.property_type == "object" else "→ Literal"

            domain_str = ""
            if prop.domain:
                domain_str = f" (domain: {', '.join(d.split('/')[-1] for d in prop.domain[:2])})"

            range_str = ""
            if prop.range:
                range_items = [r.split("/")[-1] for r in prop.range[:2]]
                range_str = f" (range: {', '.join(range_items)})"

            label = prop.label or prop_name
            lines.append(f"- dbo:{prop_name} {prop_type}{domain_str}{range_str}")
            if prop.label and prop.label != prop_name:
                lines.append(f"  Label: {prop.label}")

        return "\n".join(lines)


class OntologyManager:
    """Manages local DBpedia ontology for schema queries."""

    DEFAULT_ONTOLOGY_PATH = Path(__file__).parent.parent / "data" / "ontology" / "dbpedia-ontology.owl"

    def __init__(self, ontology_path: Optional[Path] = None):
        self.ontology_path = ontology_path or self.DEFAULT_ONTOLOGY_PATH
        self._graph: Optional[Graph] = None
        self._classes: dict[str, OntologyClass] = {}
        self._properties: dict[str, OntologyProperty] = {}
        self._domain_index: dict[str, list[str]] = {}  # class_uri -> [property_uris]
        self._loaded = False

    def load(self) -> None:
        """Load ontology from file."""
        if self._loaded:
            return

        if not self.ontology_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {self.ontology_path}")

        logger.info(f"Loading ontology from {self.ontology_path}")
        self._graph = Graph()
        self._graph.parse(str(self.ontology_path), format="xml")
        logger.info(f"Loaded {len(self._graph)} triples")

        self._index_classes()
        self._index_properties()
        self._build_domain_index()
        self._loaded = True

    def _index_classes(self) -> None:
        """Index all classes in the ontology."""
        for class_uri in self._graph.subjects(RDF.type, OWL.Class):
            if isinstance(class_uri, URIRef):
                cls = OntologyClass(uri=str(class_uri))

                # Get label
                for label in self._graph.objects(class_uri, RDFS.label):
                    if isinstance(label, Literal) and label.language == "en":
                        cls.label = str(label)
                        break

                # Get comment
                for comment in self._graph.objects(class_uri, RDFS.comment):
                    if isinstance(comment, Literal) and comment.language == "en":
                        cls.comment = str(comment)
                        break

                # Get parent classes
                for parent in self._graph.objects(class_uri, RDFS.subClassOf):
                    if isinstance(parent, URIRef):
                        cls.parent_classes.append(str(parent))

                self._classes[str(class_uri)] = cls

        logger.info(f"Indexed {len(self._classes)} classes")

    def _index_properties(self) -> None:
        """Index all properties in the ontology."""
        # Object properties
        for prop_uri in self._graph.subjects(RDF.type, OWL.ObjectProperty):
            self._index_single_property(prop_uri, "object")

        # Datatype properties
        for prop_uri in self._graph.subjects(RDF.type, OWL.DatatypeProperty):
            self._index_single_property(prop_uri, "datatype")

        # Generic rdf:Property
        for prop_uri in self._graph.subjects(RDF.type, RDF.Property):
            if str(prop_uri) not in self._properties:
                self._index_single_property(prop_uri, "generic")

        logger.info(f"Indexed {len(self._properties)} properties")

    def _index_single_property(self, prop_uri: URIRef, prop_type: str) -> None:
        """Index a single property."""
        if not isinstance(prop_uri, URIRef):
            return

        prop = OntologyProperty(uri=str(prop_uri), property_type=prop_type)

        # Get label
        for label in self._graph.objects(prop_uri, RDFS.label):
            if isinstance(label, Literal) and label.language == "en":
                prop.label = str(label)
                break

        # Get comment
        for comment in self._graph.objects(prop_uri, RDFS.comment):
            if isinstance(comment, Literal) and comment.language == "en":
                prop.comment = str(comment)
                break

        # Get domain
        for domain in self._graph.objects(prop_uri, RDFS.domain):
            if isinstance(domain, URIRef):
                prop.domain.append(str(domain))

        # Get range
        for range_val in self._graph.objects(prop_uri, RDFS.range):
            if isinstance(range_val, URIRef):
                prop.range.append(str(range_val))

        self._properties[str(prop_uri)] = prop

    def _build_domain_index(self) -> None:
        """Build index from class URIs to applicable properties."""
        for prop_uri, prop in self._properties.items():
            for domain_uri in prop.domain:
                if domain_uri not in self._domain_index:
                    self._domain_index[domain_uri] = []
                self._domain_index[domain_uri].append(prop_uri)

        logger.info(f"Built domain index for {len(self._domain_index)} classes")

    def get_class(self, class_uri: str) -> Optional[OntologyClass]:
        """Get class by URI."""
        self.load()
        return self._classes.get(class_uri)

    def get_property(self, prop_uri: str) -> Optional[OntologyProperty]:
        """Get property by URI."""
        self.load()
        return self._properties.get(prop_uri)

    @lru_cache(maxsize=1000)
    def get_class_hierarchy(self, class_uri: str, max_depth: int = 5) -> list[OntologyClass]:
        """Get class hierarchy (ancestors) for a class."""
        self.load()

        hierarchy = []
        visited = set()
        to_visit = [class_uri]
        depth = 0

        while to_visit and depth < max_depth:
            current_uri = to_visit.pop(0)
            if current_uri in visited:
                continue
            visited.add(current_uri)

            cls = self._classes.get(current_uri)
            if cls:
                hierarchy.append(cls)
                to_visit.extend(cls.parent_classes)

            depth += 1

        return hierarchy

    def get_properties_for_class(self, class_uri: str, include_inherited: bool = True) -> list[OntologyProperty]:
        """Get all properties applicable to a class."""
        self.load()

        property_uris = set()

        # Direct properties
        if class_uri in self._domain_index:
            property_uris.update(self._domain_index[class_uri])

        # Inherited properties
        if include_inherited:
            hierarchy = self.get_class_hierarchy(class_uri)
            for cls in hierarchy:
                if cls.uri in self._domain_index:
                    property_uris.update(self._domain_index[cls.uri])

        return [self._properties[uri] for uri in property_uris if uri in self._properties]

    def get_all_properties(self) -> list[OntologyProperty]:
        """Get all properties in the ontology."""
        self.load()
        return list(self._properties.values())

    def get_entity_schema_partition(
        self,
        entity_types: list[str],
        include_general_properties: bool = True,
        max_properties: int = 100
    ) -> EntitySchemaPartition:
        """Extract schema partition relevant to an entity based on its types.

        Args:
            entity_types: List of rdf:type URIs for the entity
            include_general_properties: Include properties without domain restrictions
            max_properties: Maximum properties to include

        Returns:
            EntitySchemaPartition with relevant schema information
        """
        self.load()

        # Collect applicable properties
        property_set = set()
        class_hierarchy = []

        for type_uri in entity_types:
            # Get hierarchy for this type
            hierarchy = self.get_class_hierarchy(type_uri)
            class_hierarchy.extend(hierarchy)

            # Get properties for this type
            props = self.get_properties_for_class(type_uri)
            for prop in props:
                property_set.add(prop.uri)

        # Include general properties (no domain restriction)
        if include_general_properties:
            for prop_uri, prop in self._properties.items():
                if not prop.domain:
                    property_set.add(prop_uri)

        # Deduplicate class hierarchy
        seen_classes = set()
        unique_hierarchy = []
        for cls in class_hierarchy:
            if cls.uri not in seen_classes:
                seen_classes.add(cls.uri)
                unique_hierarchy.append(cls)

        # Get property objects
        properties = [
            self._properties[uri]
            for uri in list(property_set)[:max_properties]
            if uri in self._properties
        ]

        # Sort by whether they have labels (more useful first)
        properties.sort(key=lambda p: (p.label is None, p.uri))

        return EntitySchemaPartition(
            entity_uri="",
            entity_types=entity_types,
            applicable_properties=properties,
            class_hierarchy=unique_hierarchy
        )

    def search_properties(self, keyword: str, limit: int = 20) -> list[OntologyProperty]:
        """Search properties by keyword in label or URI."""
        self.load()

        keyword_lower = keyword.lower()
        results = []

        for prop in self._properties.values():
            # Check URI
            if keyword_lower in prop.uri.lower():
                results.append(prop)
                continue

            # Check label
            if prop.label and keyword_lower in prop.label.lower():
                results.append(prop)
                continue

            # Check comment
            if prop.comment and keyword_lower in prop.comment.lower():
                results.append(prop)

        return results[:limit]

    def search_classes(self, keyword: str, limit: int = 20) -> list[OntologyClass]:
        """Search classes by keyword in label or URI."""
        self.load()

        keyword_lower = keyword.lower()
        results = []

        for cls in self._classes.values():
            if keyword_lower in cls.uri.lower():
                results.append(cls)
                continue

            if cls.label and keyword_lower in cls.label.lower():
                results.append(cls)
                continue

        return results[:limit]

    def get_stats(self) -> dict:
        """Get ontology statistics."""
        self.load()

        object_props = sum(1 for p in self._properties.values() if p.property_type == "object")
        datatype_props = sum(1 for p in self._properties.values() if p.property_type == "datatype")

        return {
            "total_triples": len(self._graph),
            "total_classes": len(self._classes),
            "total_properties": len(self._properties),
            "object_properties": object_props,
            "datatype_properties": datatype_props,
            "classes_with_properties": len(self._domain_index),
        }


# Singleton instance
_ontology_manager: Optional[OntologyManager] = None


def get_ontology_manager() -> OntologyManager:
    """Get or create the singleton ontology manager."""
    global _ontology_manager
    if _ontology_manager is None:
        _ontology_manager = OntologyManager()
    return _ontology_manager
