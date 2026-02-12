"""DBpedia schema explorer for discovering entity classes and properties.

Given an entity, finds its class and available properties within N hops,
enabling LLM to understand valid traversal paths for multi-hop reasoning.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .client import DBpediaSPARQLClient
from .cache import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class PropertyInfo:
    """Information about a property."""
    uri: str
    label: str
    domain: Optional[str] = None
    range: Optional[str] = None
    range_class: Optional[str] = None  # Class name if range is a class

    @property
    def local_name(self) -> str:
        """Get property local name."""
        if "#" in self.uri:
            return self.uri.split("#")[-1]
        return self.uri.split("/")[-1]

    @property
    def is_object_property(self) -> bool:
        """Check if property links to another entity (vs literal)."""
        if not self.range:
            return False
        return "dbpedia.org" in self.range or "Thing" in self.range


@dataclass
class ClassSchema:
    """Schema for a DBpedia class."""
    uri: str
    label: str
    parent_classes: list[str] = field(default_factory=list)
    properties: list[PropertyInfo] = field(default_factory=list)

    @property
    def local_name(self) -> str:
        """Get class local name."""
        if "#" in self.uri:
            return self.uri.split("#")[-1]
        return self.uri.split("/")[-1]

    def get_object_properties(self) -> list[PropertyInfo]:
        """Get properties that link to other entities."""
        return [p for p in self.properties if p.is_object_property]

    def get_literal_properties(self) -> list[PropertyInfo]:
        """Get properties with literal values (dates, strings, etc.)."""
        return [p for p in self.properties if not p.is_object_property]


@dataclass
class HopSchema:
    """Schema for multi-hop traversal."""
    entry_class: ClassSchema
    hop1_classes: dict[str, ClassSchema] = field(default_factory=dict)  # property -> class
    hop2_classes: dict[str, ClassSchema] = field(default_factory=dict)  # property path -> class

    def to_prompt_context(self) -> str:
        """Format schema for LLM prompt."""
        lines = []

        # Entry class
        lines.append(f"Entry Class: {self.entry_class.local_name}")
        lines.append("Available properties:")

        for prop in self.entry_class.properties:
            range_str = f" → {prop.range_class}" if prop.range_class else ""
            lines.append(f"  - {prop.local_name}{range_str}")

        # Hop 1 classes
        if self.hop1_classes:
            lines.append("\nReachable via 1 hop:")
            for prop, cls in self.hop1_classes.items():
                lines.append(f"  {prop} → {cls.local_name}")
                for p in cls.get_object_properties()[:5]:  # Limit for prompt size
                    lines.append(f"    - {p.local_name}")

        # Hop 2 classes
        if self.hop2_classes:
            lines.append("\nReachable via 2 hops:")
            for path, cls in self.hop2_classes.items():
                lines.append(f"  {path} → {cls.local_name}")

        return "\n".join(lines)


class SchemaExplorer:
    """Explores DBpedia schema starting from an entity or class."""

    def __init__(
        self,
        sparql_client: Optional[DBpediaSPARQLClient] = None,
        cache_manager: Optional[CacheManager] = None,
    ):
        """Initialize schema explorer.

        Args:
            sparql_client: SPARQL client for DBpedia
            cache_manager: Cache for schema queries
        """
        self.sparql = sparql_client or DBpediaSPARQLClient()
        self.cache = cache_manager or CacheManager()

        # Common class mappings for faster lookup
        self._class_cache: dict[str, ClassSchema] = {}

    def get_entity_class(self, entity_uri: str) -> Optional[str]:
        """Get the most specific class for an entity.

        Args:
            entity_uri: DBpedia entity URI

        Returns:
            Class URI or None
        """
        cache_key = f"entity_class:{entity_uri}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        query = f"""
        SELECT ?class WHERE {{
            <{entity_uri}> a ?class .
            FILTER(STRSTARTS(STR(?class), "http://dbpedia.org/ontology/"))
        }}
        ORDER BY ?class
        LIMIT 1
        """

        results = self.sparql.query(query)

        if results:
            class_uri = results[0].get("class", {}).get("value")
            if class_uri:
                self.cache.set(cache_key, class_uri)
                return class_uri

        return None

    def get_entity_classes(self, entity_uri: str) -> list[str]:
        """Get all classes for an entity.

        Args:
            entity_uri: DBpedia entity URI

        Returns:
            List of class URIs
        """
        query = f"""
        SELECT DISTINCT ?class WHERE {{
            <{entity_uri}> a ?class .
            FILTER(STRSTARTS(STR(?class), "http://dbpedia.org/ontology/"))
        }}
        """

        results = self.sparql.query(query)
        return [r["class"]["value"] for r in results if "class" in r]

    def get_class_properties(self, class_uri: str) -> list[PropertyInfo]:
        """Get properties where this class is the domain.

        Args:
            class_uri: DBpedia class URI

        Returns:
            List of PropertyInfo objects
        """
        cache_key = f"class_props:{class_uri}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        query = f"""
        SELECT DISTINCT ?prop ?label ?range WHERE {{
            ?prop rdfs:domain <{class_uri}> .
            OPTIONAL {{ ?prop rdfs:label ?label . FILTER(LANG(?label) = "en") }}
            OPTIONAL {{ ?prop rdfs:range ?range }}
        }}
        """

        results = self.sparql.query(query)
        properties = []

        for r in results:
            prop_uri = r.get("prop", {}).get("value", "")
            if not prop_uri:
                continue

            label = r.get("label", {}).get("value", "")
            range_uri = r.get("range", {}).get("value", "")

            # Determine range class name
            range_class = None
            if range_uri and "dbpedia.org/ontology/" in range_uri:
                range_class = range_uri.split("/")[-1]

            properties.append(PropertyInfo(
                uri=prop_uri,
                label=label or prop_uri.split("/")[-1],
                domain=class_uri,
                range=range_uri,
                range_class=range_class,
            ))

        self.cache.set(cache_key, properties)
        return properties

    def get_class_schema(self, class_uri: str) -> ClassSchema:
        """Get full schema for a class.

        Args:
            class_uri: DBpedia class URI

        Returns:
            ClassSchema object
        """
        if class_uri in self._class_cache:
            return self._class_cache[class_uri]

        # Get class label
        label_query = f"""
        SELECT ?label WHERE {{
            <{class_uri}> rdfs:label ?label .
            FILTER(LANG(?label) = "en")
        }}
        LIMIT 1
        """
        label_results = self.sparql.query(label_query)
        label = class_uri.split("/")[-1]
        if label_results:
            label = label_results[0].get("label", {}).get("value", label)

        # Get parent classes
        parent_query = f"""
        SELECT ?parent WHERE {{
            <{class_uri}> rdfs:subClassOf ?parent .
            FILTER(STRSTARTS(STR(?parent), "http://dbpedia.org/ontology/"))
        }}
        """
        parent_results = self.sparql.query(parent_query)
        parents = [r["parent"]["value"] for r in parent_results if "parent" in r]

        # Get properties
        properties = self.get_class_properties(class_uri)

        # Also include inherited properties from parent classes
        for parent_uri in parents:
            parent_props = self.get_class_properties(parent_uri)
            existing_uris = {p.uri for p in properties}
            for prop in parent_props:
                if prop.uri not in existing_uris:
                    properties.append(prop)

        schema = ClassSchema(
            uri=class_uri,
            label=label,
            parent_classes=parents,
            properties=properties,
        )

        self._class_cache[class_uri] = schema
        return schema

    def get_schema_for_entity(
        self,
        entity_uri: str,
        max_hops: int = 2,
    ) -> Optional[HopSchema]:
        """Get schema for multi-hop traversal starting from an entity.

        Args:
            entity_uri: Starting entity URI
            max_hops: Maximum hops to explore (1 or 2)

        Returns:
            HopSchema with available traversal paths
        """
        # Get entity's class
        class_uri = self.get_entity_class(entity_uri)
        if not class_uri:
            logger.warning(f"Could not find class for {entity_uri}")
            return None

        # Get entry class schema
        entry_schema = self.get_class_schema(class_uri)

        hop_schema = HopSchema(entry_class=entry_schema)

        if max_hops < 1:
            return hop_schema

        # Explore hop 1: classes reachable via object properties
        for prop in entry_schema.get_object_properties():
            if prop.range_class and prop.range:
                try:
                    hop1_schema = self.get_class_schema(prop.range)
                    hop_schema.hop1_classes[prop.local_name] = hop1_schema
                except Exception as e:
                    logger.debug(f"Could not get schema for {prop.range}: {e}")

        if max_hops < 2:
            return hop_schema

        # Explore hop 2: classes reachable via 2 object properties
        for prop1_name, hop1_class in hop_schema.hop1_classes.items():
            for prop2 in hop1_class.get_object_properties():
                if prop2.range_class and prop2.range:
                    path = f"{prop1_name} → {prop2.local_name}"
                    try:
                        hop2_schema = self.get_class_schema(prop2.range)
                        hop_schema.hop2_classes[path] = hop2_schema
                    except Exception as e:
                        logger.debug(f"Could not get schema for {prop2.range}: {e}")

        return hop_schema

    def get_traversal_paths(
        self,
        entity_uri: str,
        target_property: Optional[str] = None,
        max_hops: int = 2,
    ) -> list[list[str]]:
        """Find possible paths from entity to target property.

        Args:
            entity_uri: Starting entity
            target_property: Property to reach (e.g., "birthDate")
            max_hops: Maximum path length

        Returns:
            List of paths (each path is list of property names)
        """
        hop_schema = self.get_schema_for_entity(entity_uri, max_hops)
        if not hop_schema:
            return []

        paths = []

        # Check hop 0 (direct property)
        for prop in hop_schema.entry_class.properties:
            if target_property is None or target_property in prop.local_name.lower():
                paths.append([prop.local_name])

        # Check hop 1
        for prop1_name, hop1_class in hop_schema.hop1_classes.items():
            for prop in hop1_class.properties:
                if target_property is None or target_property in prop.local_name.lower():
                    paths.append([prop1_name, prop.local_name])

        # Check hop 2
        for path_str, hop2_class in hop_schema.hop2_classes.items():
            prop1, prop2 = path_str.split(" → ")
            for prop in hop2_class.properties:
                if target_property is None or target_property in prop.local_name.lower():
                    paths.append([prop1, prop2, prop.local_name])

        return paths

    def format_schema_for_prompt(
        self,
        entity_uri: str,
        max_hops: int = 2,
        max_properties: int = 15,
    ) -> str:
        """Format entity schema as context for LLM prompt.

        Args:
            entity_uri: Entity to explore
            max_hops: Hops to include
            max_properties: Max properties per class

        Returns:
            Formatted schema string for prompt
        """
        hop_schema = self.get_schema_for_entity(entity_uri, max_hops)
        if not hop_schema:
            return f"No schema found for {entity_uri}"

        lines = []
        entity_name = entity_uri.split("/")[-1].replace("_", " ")

        lines.append(f"Schema for: {entity_name}")
        lines.append(f"Type: {hop_schema.entry_class.local_name}")
        lines.append("")

        # Direct properties (hop 0)
        lines.append("Direct properties:")
        obj_props = hop_schema.entry_class.get_object_properties()[:max_properties]
        lit_props = hop_schema.entry_class.get_literal_properties()[:max_properties]

        for prop in lit_props:
            lines.append(f"  {prop.local_name} (value)")
        for prop in obj_props:
            target = prop.range_class or "Entity"
            lines.append(f"  {prop.local_name} → {target}")

        # Hop 1 connections
        if hop_schema.hop1_classes:
            lines.append("")
            lines.append("1-hop connections:")
            for prop_name, cls in list(hop_schema.hop1_classes.items())[:5]:
                lines.append(f"  {prop_name} → {cls.local_name}")
                # Show key properties of connected class
                key_props = cls.properties[:5]
                for p in key_props:
                    if p.is_object_property:
                        lines.append(f"    .{p.local_name} → {p.range_class or 'Entity'}")
                    else:
                        lines.append(f"    .{p.local_name}")

        # Hop 2 connections (summarized)
        if hop_schema.hop2_classes:
            lines.append("")
            lines.append("2-hop paths:")
            for path, cls in list(hop_schema.hop2_classes.items())[:10]:
                lines.append(f"  {path} → {cls.local_name}")

        return "\n".join(lines)


# Common inference patterns based on ontology
MULTI_HOP_PATTERNS = {
    "paternal_grandfather": {
        "path": ["father", "father"],
        "description": "Father's father",
    },
    "maternal_grandfather": {
        "path": ["mother", "father"],
        "description": "Mother's father",
    },
    "paternal_grandmother": {
        "path": ["father", "mother"],
        "description": "Father's mother",
    },
    "maternal_grandmother": {
        "path": ["mother", "mother"],
        "description": "Mother's mother",
    },
    "director_birthplace": {
        "path": ["director", "birthPlace"],
        "description": "Where the director was born",
    },
    "author_nationality": {
        "path": ["author", "nationality"],
        "description": "Nationality of the author",
    },
    "founder_birthdate": {
        "path": ["foundedBy", "birthDate"],
        "description": "When the founder was born",
    },
    "spouse_occupation": {
        "path": ["spouse", "occupation"],
        "description": "What the spouse does",
    },
}
