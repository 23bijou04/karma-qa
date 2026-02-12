"""Wikidata to DBpedia property mapping for 2WikiMultiHopQA compatibility."""

import logging
from typing import Optional

from ..config import WIKIDATA_TO_DBPEDIA_PROPERTIES, INFERENCE_RULES

logger = logging.getLogger(__name__)


class PropertyMapper:
    """Maps Wikidata properties to DBpedia equivalents.

    The 2WikiMultiHopQA benchmark uses Wikidata property codes (P-codes).
    This mapper translates them to DBpedia ontology/property URIs.
    """

    # Human-readable property names to DBpedia URIs
    PROPERTY_NAME_TO_DBPEDIA: dict[str, list[str]] = {
        # Person properties
        "date of birth": ["dbo:birthDate", "dbp:birthDate"],
        "birth date": ["dbo:birthDate", "dbp:birthDate"],
        "birthDate": ["dbo:birthDate", "dbp:birthDate"],
        "date of death": ["dbo:deathDate", "dbp:deathDate"],
        "death date": ["dbo:deathDate", "dbp:deathDate"],
        "deathDate": ["dbo:deathDate", "dbp:deathDate"],
        "place of birth": ["dbo:birthPlace", "dbp:birthPlace"],
        "birthPlace": ["dbo:birthPlace", "dbp:birthPlace"],
        "place of death": ["dbo:deathPlace", "dbp:deathPlace"],
        "deathPlace": ["dbo:deathPlace", "dbp:deathPlace"],
        "father": ["dbo:father", "dbp:father"],
        "mother": ["dbo:mother", "dbp:mother"],
        "spouse": ["dbo:spouse", "dbp:spouse"],
        "child": ["dbo:child", "dbp:children"],
        "children": ["dbo:child", "dbp:children"],
        "sibling": ["dbo:relative", "dbp:siblings"],
        "nationality": ["dbo:nationality", "dbp:nationality"],
        "country of citizenship": ["dbo:nationality", "dbp:nationality"],
        "occupation": ["dbo:occupation", "dbp:occupation"],
        "educated at": ["dbo:almaMater", "dbp:almaMater"],
        "alma mater": ["dbo:almaMater", "dbp:almaMater"],
        "employer": ["dbo:employer", "dbp:employer"],
        "doctoral advisor": ["dbo:doctoralAdvisor", "dbp:doctoralAdvisor"],
        "doctoral student": ["dbo:doctoralStudent", "dbp:doctoralStudents"],
        "field of work": ["dbo:field", "dbp:field"],
        # Film properties
        "director": ["dbo:director", "dbp:director"],
        "producer": ["dbo:producer", "dbp:producer"],
        "publication date": ["dbo:releaseDate", "dbp:releaseDate"],
        "release date": ["dbo:releaseDate", "dbp:releaseDate"],
        "releaseDate": ["dbo:releaseDate", "dbp:releaseDate"],
        "country of origin": ["dbo:country", "dbp:country"],
        "country": ["dbo:country", "dbp:country"],
        "composer": ["dbo:musicComposer", "dbp:music"],
        "screenwriter": ["dbo:writer", "dbp:writer"],
        "writer": ["dbo:writer", "dbp:writer"],
        # Organization properties
        "founded by": ["dbo:foundedBy", "dbp:founder"],
        "founder": ["dbo:foundedBy", "dbp:founder"],
        "inception": ["dbo:foundingDate", "dbp:founded"],
        "founding date": ["dbo:foundingDate", "dbp:founded"],
        # Music properties
        "performer": ["dbo:artist", "dbp:artist"],
        "artist": ["dbo:artist", "dbp:artist"],
        "record label": ["dbo:recordLabel", "dbp:label"],
    }

    def __init__(self):
        self.wikidata_mapping = WIKIDATA_TO_DBPEDIA_PROPERTIES
        self.inference_rules = INFERENCE_RULES
        self._build_reverse_mapping()

    def _build_reverse_mapping(self) -> None:
        """Build reverse mapping from DBpedia to Wikidata."""
        self.dbpedia_to_wikidata: dict[str, str] = {}
        for wikidata_id, dbpedia_props in self.wikidata_mapping.items():
            for dbpedia_prop in dbpedia_props:
                self.dbpedia_to_wikidata[dbpedia_prop] = wikidata_id

    def get_dbpedia_properties(self, wikidata_property: str) -> list[str]:
        """Get DBpedia property URIs for a Wikidata property.

        Args:
            wikidata_property: Wikidata property ID (e.g., "P569") or name

        Returns:
            List of DBpedia property URIs to try
        """
        # Check if it's a P-code
        if wikidata_property.startswith("P") and wikidata_property[1:].isdigit():
            return self.wikidata_mapping.get(wikidata_property, [])

        # Check by property name
        prop_lower = wikidata_property.lower().strip()
        return self.PROPERTY_NAME_TO_DBPEDIA.get(prop_lower, [])

    def get_property_from_name(self, property_name: str) -> list[str]:
        """Get DBpedia properties from human-readable property name.

        Args:
            property_name: Human-readable property name (e.g., "date of birth")

        Returns:
            List of DBpedia property URIs
        """
        prop_lower = property_name.lower().strip()

        # Direct match
        if prop_lower in self.PROPERTY_NAME_TO_DBPEDIA:
            return self.PROPERTY_NAME_TO_DBPEDIA[prop_lower]

        # Partial match
        for key, value in self.PROPERTY_NAME_TO_DBPEDIA.items():
            if prop_lower in key or key in prop_lower:
                return value

        return []

    def get_wikidata_property(self, dbpedia_property: str) -> Optional[str]:
        """Get Wikidata property ID for a DBpedia property.

        Args:
            dbpedia_property: DBpedia property URI

        Returns:
            Wikidata property ID or None
        """
        return self.dbpedia_to_wikidata.get(dbpedia_property)

    def get_inference_rule(self, relation: str) -> Optional[tuple[str, str]]:
        """Get the inference rule (premise relations) for a derived relation.

        Args:
            relation: Derived relation name (e.g., "paternal_grandfather")

        Returns:
            Tuple of (relation1, relation2) that derive this relation
        """
        return self.inference_rules.get(relation)

    def get_inference_chain(self, relation: str) -> list[list[str]]:
        """Get the chain of DBpedia properties for an inference relation.

        For example, "paternal_grandfather" requires:
        1. First hop: father properties
        2. Second hop: father properties

        Args:
            relation: Inference relation name

        Returns:
            List of property lists for each hop
        """
        rule = self.get_inference_rule(relation)
        if not rule:
            return []

        rel1, rel2 = rule
        props1 = self.get_property_from_name(rel1)
        props2 = self.get_property_from_name(rel2)

        return [props1, props2] if props1 and props2 else []

    def expand_property_for_query(self, property_name: str) -> str:
        """Expand a property name to SPARQL UNION clause.

        Args:
            property_name: Human-readable property name

        Returns:
            SPARQL clause with UNION of all possible property URIs
        """
        properties = self.get_property_from_name(property_name)

        if not properties:
            # Try as-is (might be a DBpedia property already)
            return f"?subject <{property_name}> ?object ."

        if len(properties) == 1:
            return f"?subject {properties[0]} ?object ."

        # Create UNION clause
        clauses = [f"{{ ?subject {prop} ?object . }}" for prop in properties]
        return " UNION ".join(clauses)

    def get_all_family_properties(self) -> dict[str, list[str]]:
        """Get all family-related properties (used in inference questions).

        Returns:
            Dictionary mapping relation names to DBpedia properties
        """
        family_relations = [
            "father", "mother", "spouse", "child", "sibling",
            "grandfather", "grandmother", "grandchild"
        ]
        return {
            rel: self.get_property_from_name(rel)
            for rel in family_relations
            if self.get_property_from_name(rel)
        }

    def get_comparison_properties(self, entity_type: str) -> dict[str, list[str]]:
        """Get properties commonly used in comparison questions.

        Args:
            entity_type: Type of entity (person, film, organization, etc.)

        Returns:
            Dictionary mapping property names to DBpedia URIs
        """
        if entity_type.lower() in {"person", "human"}:
            return {
                "birth date": self.get_property_from_name("date of birth"),
                "death date": self.get_property_from_name("date of death"),
                "nationality": self.get_property_from_name("nationality"),
                "occupation": self.get_property_from_name("occupation"),
                "birth place": self.get_property_from_name("place of birth"),
            }
        elif entity_type.lower() in {"film", "movie"}:
            return {
                "release date": self.get_property_from_name("release date"),
                "director": self.get_property_from_name("director"),
                "country": self.get_property_from_name("country"),
            }
        elif entity_type.lower() in {"organization", "company"}:
            return {
                "inception": self.get_property_from_name("inception"),
                "founder": self.get_property_from_name("founded by"),
                "country": self.get_property_from_name("country"),
            }
        else:
            return {}
