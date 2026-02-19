"""Hybrid schema discovery for DBpedia entities.

Combines cached ontology schemas with dynamic property discovery
to provide complete schema information to LLM for SPARQL generation.
"""

import logging
from typing import Optional
from SPARQLWrapper import SPARQLWrapper, JSON

from .sparql_utils import escape_sparql_string, escape_sparql_uri

logger = logging.getLogger(__name__)

# Cached ontology schemas for common DBpedia classes
# Maps class name -> (dbo: properties, dbp: properties, property descriptions)
CACHED_CLASS_SCHEMAS = {
    "Person": {
        "dbo": [
            "birthDate", "deathDate", "birthPlace", "deathPlace",
            "parent", "child", "spouse", "nationality", "occupation",
            "almaMater", "residence", "religion", "ethnicity"
        ],
        "dbp": [
            "father", "mother", "birthDate", "birthPlace", "deathDate",
            "deathPlace", "spouse", "children", "nationality", "occupation",
            "education", "residence"
        ],
        "descriptions": {
            "father": "father of the person (use for paternal relations)",
            "mother": "mother of the person (use for maternal relations)",
            "parent": "generic parent (could be father or mother)",
            "birthDate": "date of birth (xsd:date format)",
            "deathDate": "date of death (xsd:date format)",
            "birthPlace": "place of birth (returns Place with country, isPartOf)",
            "spouse": "married partner",
            "child": "children of the person",
            "nationality": "country of citizenship"
        },
        "dbo_2hop": {
            "parent": ("Person", ["birthDate", "deathDate", "birthPlace"]),
            "spouse": ("Person", ["birthDate", "deathDate", "birthPlace", "nationality"]),
            "child": ("Person", ["birthDate", "deathDate", "birthPlace"]),
            "birthPlace": ("Place", ["country", "isPartOf", "populationTotal"]),
            "deathPlace": ("Place", ["country", "isPartOf"]),
        },
    },
    "Film": {
        "dbo": [
            "director", "producer", "starring", "releaseDate", "runtime",
            "distributor", "country", "budget", "gross", "cinematography",
            "musicComposer", "editing", "writer"
        ],
        "dbp": [
            "director", "producer", "starring", "released", "runtime",
            "studio", "distributor", "country", "budget", "gross",
            "music", "cinematography", "editing", "writer"
        ],
        "descriptions": {
            "director": "film director (returns Person with birthDate, deathDate, birthPlace)",
            "producer": "film producer (returns Person with birthDate, deathDate)",
            "starring": "lead actors (returns Person with birthDate, deathDate)",
            "releaseDate": "release date (xsd:date)",
            "released": "release date (alternative property)",
            "distributor": "distribution company",
            "country": "country of origin"
        },
        "dbo_2hop": {
            "director": ("Person", ["birthDate", "deathDate", "birthPlace", "nationality"]),
            "producer": ("Person", ["birthDate", "deathDate", "birthPlace"]),
            "starring": ("Person", ["birthDate", "deathDate", "birthPlace", "nationality"]),
            "writer": ("Person", ["birthDate", "deathDate", "birthPlace"]),
            "musicComposer": ("Person", ["birthDate", "deathDate", "birthPlace"]),
            "distributor": ("Company", ["foundingDate", "locationCity", "locationCountry"]),
        },
    },
    "Company": {
        "dbo": [
            "foundedBy", "foundingDate", "locationCity", "locationCountry",
            "industry", "numberOfEmployees", "revenue", "netIncome",
            "parentCompany", "subsidiary", "keyPerson"
        ],
        "dbp": [
            "founder", "founded", "location", "headquarters", "industry",
            "employees", "revenue", "netIncome", "parent", "subsidiaries",
            "keyPeople", "products", "services"
        ],
        "descriptions": {
            "foundedBy": "founder of the company (returns Person with birthDate, deathDate)",
            "founder": "founder (alternative property)",
            "foundingDate": "date company was founded",
            "founded": "founding date (alternative)",
            "locationCity": "city where headquartered",
            "industry": "business sector"
        },
        "dbo_2hop": {
            "foundedBy": ("Person", ["birthDate", "deathDate", "birthPlace"]),
            "keyPerson": ("Person", ["birthDate", "deathDate", "birthPlace"]),
            "parentCompany": ("Company", ["foundingDate", "locationCity"]),
            "locationCity": ("Place", ["country", "isPartOf", "populationTotal"]),
            "locationCountry": ("Place", ["capital", "populationTotal"]),
        },
    },
    "Place": {
        "dbo": [
            "country", "isPartOf", "populationTotal", "areaTotal",
            "leader", "leaderName", "capital", "largestCity"
        ],
        "dbp": [
            "country", "subdivision", "population", "area",
            "leader", "capital", "largestCity", "timezone"
        ],
        "descriptions": {
            "country": "country the place is in",
            "isPartOf": "parent administrative region",
            "populationTotal": "total population count",
            "capital": "capital city (for countries/states)"
        },
        "dbo_2hop": {
            "country": ("Place", ["capital", "populationTotal", "leader"]),
            "isPartOf": ("Place", ["country", "capital", "populationTotal"]),
            "capital": ("Place", ["populationTotal", "country"]),
        },
    },
    "MusicalArtist": {
        "dbo": [
            "genre", "recordLabel", "associatedMusicalArtist", "instrument",
            "birthDate", "birthPlace", "activeYearsStartYear"
        ],
        "dbp": [
            "genre", "label", "associatedActs", "instrument",
            "birthDate", "birthPlace", "yearsActive", "origin"
        ],
        "descriptions": {
            "genre": "music genre",
            "recordLabel": "record label company",
            "associatedMusicalArtist": "collaborating artists"
        },
        "dbo_2hop": {
            "associatedMusicalArtist": ("MusicalArtist", ["genre", "recordLabel", "birthDate"]),
            "birthPlace": ("Place", ["country", "isPartOf", "populationTotal"]),
        },
    },
    "Book": {
        "dbo": [
            "author", "publisher", "publicationDate", "numberOfPages",
            "isbn", "literaryGenre", "language"
        ],
        "dbp": [
            "author", "publisher", "published", "pages",
            "isbn", "genre", "language", "country"
        ],
        "descriptions": {
            "author": "book author (returns Person with birthDate, deathDate, birthPlace)",
            "publisher": "publishing company",
            "publicationDate": "date of publication"
        },
        "dbo_2hop": {
            "author": ("Person", ["birthDate", "deathDate", "birthPlace", "nationality"]),
            "publisher": ("Company", ["foundingDate", "locationCity"]),
        },
    }
}

# Property -> target class mappings for dbo: properties that return entities.
# Only includes properties whose range is one of the 6 cached classes.
PROPERTY_RANGES: dict[str, str] = {
    # Film -> Person
    "dbo:director": "Person",
    "dbo:producer": "Person",
    "dbo:starring": "Person",
    "dbo:writer": "Person",
    "dbo:musicComposer": "Person",
    "dbo:cinematography": "Person",
    "dbo:editing": "Person",
    # Film -> Company
    "dbo:distributor": "Company",
    # Book -> Person
    "dbo:author": "Person",
    # Book -> Company
    "dbo:publisher": "Company",
    # Person -> Place
    "dbo:birthPlace": "Place",
    "dbo:deathPlace": "Place",
    "dbo:residence": "Place",
    "dbo:almaMater": "Place",
    # Person -> Person
    "dbo:parent": "Person",
    "dbo:spouse": "Person",
    "dbo:child": "Person",
    # Company -> Person
    "dbo:foundedBy": "Person",
    "dbo:keyPerson": "Person",
    # Company -> Company
    "dbo:parentCompany": "Company",
    "dbo:subsidiary": "Company",
    # Company -> Place
    "dbo:locationCity": "Place",
    "dbo:locationCountry": "Place",
    # Place -> Place
    "dbo:country": "Place",
    "dbo:isPartOf": "Place",
    "dbo:capital": "Place",
    # MusicalArtist -> MusicalArtist
    "dbo:associatedMusicalArtist": "MusicalArtist",
}


def format_2hop_lines(cached: dict, max_chains: int = 5, max_props: int = 4) -> list[str]:
    """Format 2-hop chain info from a cached class schema entry.

    Shared by _build_schema_text() and sparql_evaluation.get_schema_for_uris().
    Returns list of lines (without trailing newline).
    """
    dbo_2hop = cached.get("dbo_2hop")
    if not dbo_2hop:
        return []
    lines = ["  2-Hop chains:"]
    for prop, (target_cls, target_props) in list(dbo_2hop.items())[:max_chains]:
        lines.append(
            f"    - dbo:{prop} -> {target_cls}: "
            f"{', '.join(target_props[:max_props])}"
        )
    return lines


# Common entity name -> DBpedia URI mappings (for entities with special URIs)
ENTITY_URI_OVERRIDES = {
    "prince william": "http://dbpedia.org/resource/William,_Prince_of_Wales",
    "william prince of wales": "http://dbpedia.org/resource/William,_Prince_of_Wales",
    "king charles": "http://dbpedia.org/resource/Charles_III",
    "charles iii": "http://dbpedia.org/resource/Charles_III",
    "queen elizabeth": "http://dbpedia.org/resource/Elizabeth_II",
    "elizabeth ii": "http://dbpedia.org/resource/Elizabeth_II",
}


class SchemaDiscovery:
    """Hybrid schema discovery combining cached ontology with dynamic lookup."""

    def __init__(self, sparql_endpoint: str = "https://dbpedia.org/sparql"):
        """Initialize schema discovery.

        Args:
            sparql_endpoint: DBpedia SPARQL endpoint URL
        """
        self.endpoint = sparql_endpoint
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.addCustomHttpHeader("User-Agent", "DBpediaQA/1.0 (research)")

        # Cache for discovered entity URIs and schemas
        self._uri_cache: dict[str, str] = {}
        self._schema_cache: dict[str, dict] = {}

    def get_entity_uri(self, entity_name: str) -> Optional[str]:
        """Find DBpedia URI for an entity name.

        Args:
            entity_name: Entity name (e.g., "Albert Einstein")

        Returns:
            DBpedia URI or None if not found
        """
        # Check cache
        cache_key = entity_name.lower().strip()
        if cache_key in self._uri_cache:
            return self._uri_cache[cache_key]

        # Check overrides for special entities
        if cache_key in ENTITY_URI_OVERRIDES:
            uri = ENTITY_URI_OVERRIDES[cache_key]
            self._uri_cache[cache_key] = uri
            return uri

        # Try direct URI construction
        uri_name = entity_name.strip().replace(" ", "_")
        direct_uri = f"http://dbpedia.org/resource/{uri_name}"

        # Verify it exists
        if self._entity_exists(direct_uri):
            self._uri_cache[cache_key] = direct_uri
            return direct_uri

        # Try lowercase variation (e.g., "Film" -> "film")
        uri_name_lower = uri_name.replace("_Film)", "_film)")
        direct_uri_lower = f"http://dbpedia.org/resource/{uri_name_lower}"
        if self._entity_exists(direct_uri_lower):
            self._uri_cache[cache_key] = direct_uri_lower
            return direct_uri_lower

        # Search by label (fuzzy match)
        uri = self._search_entity_by_label(entity_name)
        if uri:
            self._uri_cache[cache_key] = uri
            return uri

        # Try searching with CONTAINS for partial match
        uri = self._search_entity_fuzzy(entity_name)
        if uri:
            self._uri_cache[cache_key] = uri
        return uri

    def _search_entity_fuzzy(self, entity_name: str) -> Optional[str]:
        """Search for entity using fuzzy matching."""
        # Extract key terms for search
        search_term = entity_name.replace("'", "").replace("(", "").replace(")", "")

        query = f"""
        SELECT ?entity WHERE {{
            ?entity rdfs:label ?label .
            FILTER(LANG(?label) = "en")
            FILTER(CONTAINS(LCASE(?label), "{escape_sparql_string(search_term.lower()[:30])}"))
        }}
        LIMIT 1
        """
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
                return bindings[0]["entity"]["value"]
        except Exception as e:
            logger.warning(f"Error in fuzzy search: {e}")

        # Try with year variations for films
        import re
        year_match = re.search(r'\((\d{4})\s*(?:Film|film)?\)', entity_name)
        if year_match:
            year = year_match.group(1)
            base_name = re.sub(r'\s*\(\d{4}.*?\)', '', entity_name).strip()
            # Try "_film)" suffix
            uri_name = base_name.replace(" ", "_").replace("'", "%27")
            variations = [
                f"http://dbpedia.org/resource/{uri_name}_({year}_film)",
                f"http://dbpedia.org/resource/{uri_name}_({year}_Film)",
                f"http://dbpedia.org/resource/{uri_name}_(film)",
                f"http://dbpedia.org/resource/{uri_name}",
            ]
            for uri in variations:
                if self._entity_exists(uri):
                    return uri

        return None

    def _entity_exists(self, uri: str) -> bool:
        """Check if an entity URI exists in DBpedia."""
        safe_uri = escape_sparql_uri(uri)
        query = f"""
        ASK WHERE {{
            <{safe_uri}> ?p ?o .
        }}
        """
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            return results.get("boolean", False)
        except Exception as e:
            logger.warning(f"Error checking entity existence: {e}")
            return False

    def _search_entity_by_label(self, entity_name: str) -> Optional[str]:
        """Search for entity by rdfs:label."""
        query = f"""
        SELECT ?entity WHERE {{
            ?entity rdfs:label "{escape_sparql_string(entity_name)}"@en .
            ?entity a ?type .
            FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
        }}
        LIMIT 1
        """
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
                return bindings[0]["entity"]["value"]
        except Exception as e:
            logger.warning(f"Error searching entity by label: {e}")
        return None

    def get_entity_class(self, entity_uri: str) -> Optional[str]:
        """Get the primary DBpedia class for an entity.

        Args:
            entity_uri: Full DBpedia URI

        Returns:
            Class name (e.g., "Person", "Film") or None
        """
        safe_uri = escape_sparql_uri(entity_uri)
        query = f"""
        SELECT ?type WHERE {{
            <{safe_uri}> a ?type .
            FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
        }}
        """
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])

            # Find the most specific class that we have a schema for
            for row in bindings:
                type_uri = row["type"]["value"]
                class_name = type_uri.split("/")[-1]
                if class_name in CACHED_CLASS_SCHEMAS:
                    return class_name

            # Return first class if no cached schema match
            if bindings:
                return bindings[0]["type"]["value"].split("/")[-1]
        except Exception as e:
            logger.warning(f"Error getting entity class: {e}")
        return None

    def get_actual_properties(self, entity_uri: str,
                              property_filter: Optional[list[str]] = None) -> dict[str, list[str]]:
        """Query DBpedia for actual properties an entity has.

        Args:
            entity_uri: Full DBpedia URI
            property_filter: Optional list of property name substrings to filter by

        Returns:
            Dict with 'dbo' and 'dbp' keys containing property lists
        """
        filter_clause = ""
        if property_filter:
            conditions = " || ".join([
                f'CONTAINS(LCASE(STR(?property)), "{escape_sparql_string(p.lower())}")'
                for p in property_filter
            ])
            filter_clause = f"FILTER({conditions})"

        safe_uri = escape_sparql_uri(entity_uri)
        query = f"""
        SELECT DISTINCT ?property ?value WHERE {{
            <{safe_uri}> ?property ?value .
            FILTER(
                STRSTARTS(STR(?property), "http://dbpedia.org/ontology/") ||
                STRSTARTS(STR(?property), "http://dbpedia.org/property/")
            )
            FILTER(!CONTAINS(STR(?property), "wikiPage"))
            FILTER(!CONTAINS(STR(?property), "abstract"))
            FILTER(!CONTAINS(STR(?property), "thumbnail"))
            FILTER(!CONTAINS(STR(?property), "isPrimaryTopicOf"))
            {filter_clause}
        }}
        LIMIT 50
        """

        result = {"dbo": [], "dbp": [], "values": {}}

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            for row in results.get("results", {}).get("bindings", []):
                prop_uri = row["property"]["value"]
                prop_name = prop_uri.split("/")[-1]
                value = row["value"]["value"]

                # Format value
                if row["value"]["type"] == "uri":
                    value = value.split("/")[-1].replace("_", " ")

                if "/ontology/" in prop_uri:
                    if prop_name not in result["dbo"]:
                        result["dbo"].append(prop_name)
                        result["values"][f"dbo:{prop_name}"] = value
                else:
                    if prop_name not in result["dbp"]:
                        result["dbp"].append(prop_name)
                        result["values"][f"dbp:{prop_name}"] = value

        except Exception as e:
            logger.warning(f"Error getting actual properties: {e}")

        return result

    def get_schema_for_entity(self, entity_name: str) -> dict:
        """Get complete hybrid schema for an entity.

        This is the main method that combines:
        1. Entity URI resolution
        2. Class detection
        3. Cached ontology schema
        4. Actual properties from DBpedia

        Args:
            entity_name: Entity name (e.g., "Prince William")

        Returns:
            Complete schema dict with all information for LLM
        """
        # Check cache
        cache_key = entity_name.lower().strip()
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        result = {
            "entity_name": entity_name,
            "entity_uri": None,
            "class": None,
            "cached_schema": None,
            "actual_properties": {"dbo": [], "dbp": [], "values": {}},
            "schema_text": ""
        }

        # Step 1: Get entity URI
        entity_uri = self.get_entity_uri(entity_name)
        if not entity_uri:
            logger.warning(f"Could not find URI for entity: {entity_name}")
            result["schema_text"] = f"Entity '{entity_name}' not found in DBpedia."
            return result

        result["entity_uri"] = entity_uri

        # Step 2: Get entity class
        entity_class = self.get_entity_class(entity_uri)
        result["class"] = entity_class

        # Step 3: Get cached schema for class
        if entity_class and entity_class in CACHED_CLASS_SCHEMAS:
            result["cached_schema"] = CACHED_CLASS_SCHEMAS[entity_class]

        # Step 4: Get actual properties from DBpedia
        actual = self.get_actual_properties(entity_uri)
        result["actual_properties"] = actual

        # Step 5: Build schema text for LLM
        result["schema_text"] = self._build_schema_text(result)

        # Cache result
        self._schema_cache[cache_key] = result
        return result

    def _build_schema_text(self, schema: dict) -> str:
        """Build human-readable schema text for LLM prompt."""
        lines = []

        lines.append(f"Entity: {schema['entity_name']}")
        lines.append(f"URI: <{schema['entity_uri']}>")

        if schema["class"]:
            lines.append(f"Class: dbo:{schema['class']}")

        lines.append("")

        # Show cached ontology properties
        if schema["cached_schema"]:
            lines.append("Ontology properties (from schema):")
            cached = schema["cached_schema"]

            # Show dbo properties
            if cached.get("dbo"):
                lines.append(f"  dbo: {', '.join(cached['dbo'][:10])}")

            # Show dbp properties
            if cached.get("dbp"):
                lines.append(f"  dbp: {', '.join(cached['dbp'][:10])}")

            # Show relevant descriptions
            if cached.get("descriptions"):
                lines.append("  Property hints:")
                for prop, desc in list(cached["descriptions"].items())[:5]:
                    lines.append(f"    - {prop}: {desc}")

            lines.extend(format_2hop_lines(cached))

        lines.append("")

        # Show actual properties with sample values
        actual = schema["actual_properties"]
        if actual["dbo"] or actual["dbp"]:
            lines.append("Actual properties found for this entity:")

            values = actual.get("values", {})
            for prop in actual["dbo"][:8]:
                val = values.get(f"dbo:{prop}", "")
                lines.append(f"  dbo:{prop} -> {val}")

            for prop in actual["dbp"][:8]:
                val = values.get(f"dbp:{prop}", "")
                lines.append(f"  dbp:{prop} -> {val}")

        return "\n".join(lines)

    def get_schema_for_question(self, question: str, entities: list[str]) -> str:
        """Get combined schema for all entities in a question.

        Args:
            question: The question text
            entities: List of entity names extracted from question

        Returns:
            Combined schema text for all entities
        """
        schemas = []

        for entity in entities:
            schema = self.get_schema_for_entity(entity)
            if schema["entity_uri"]:
                schemas.append(schema["schema_text"])

        if not schemas:
            return "No entities found in DBpedia."

        combined = "\n\n---\n\n".join(schemas)

        # Add general prefixes
        prefix_text = """
SPARQL Prefixes:
  PREFIX dbr: <http://dbpedia.org/resource/>
  PREFIX dbo: <http://dbpedia.org/ontology/>
  PREFIX dbp: <http://dbpedia.org/property/>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

IMPORTANT: Use the EXACT URIs shown above. For properties, prefer dbp:
if both dbo: and dbp: are available, as dbp: often has better coverage.
"""

        return prefix_text + "\n" + combined
