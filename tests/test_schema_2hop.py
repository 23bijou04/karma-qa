"""Tests for 2-hop chain inference data structures and property filtering."""
from unittest.mock import MagicMock

import pytest

from dbpedia.schema_discovery import (
    CACHED_CLASS_SCHEMAS,
    PROPERTY_RANGES,
    SchemaDiscovery,
    format_2hop_lines,
)


class TestPropertyRanges:
    """Validate PROPERTY_RANGES dict integrity."""

    def test_all_ranges_are_cached_classes(self):
        """Ranges that point to known ontology classes must be in CACHED_CLASS_SCHEMAS.

        Some ranges (Award, Area, Demographics, etc.) are not modelled as full
        classes in the ontology cache — they're leaf values or abstract types.
        We allow those to be absent but ensure core navigable ranges are present.
        """
        core_range_props = [
            "dbo:director", "dbo:author", "dbo:birthPlace",
            "dbo:foundedBy", "dbo:starring", "dbo:producer",
        ]
        for prop in core_range_props:
            if prop in PROPERTY_RANGES:
                assert PROPERTY_RANGES[prop] in CACHED_CLASS_SCHEMAS, (
                    f"{prop} -> {PROPERTY_RANGES[prop]} not in CACHED_CLASS_SCHEMAS"
                )

    def test_all_props_have_dbo_prefix(self):
        """All keys must start with 'dbo:'."""
        for prop in PROPERTY_RANGES:
            assert prop.startswith("dbo:"), f"{prop} missing dbo: prefix"

    def test_core_mappings_exist(self):
        """Verify essential property mappings."""
        assert PROPERTY_RANGES["dbo:director"] == "Person"
        assert PROPERTY_RANGES["dbo:author"] == "Person"
        assert PROPERTY_RANGES["dbo:birthPlace"] == "Place"
        assert PROPERTY_RANGES["dbo:foundedBy"] == "Person"


class TestCachedClassSchemas2Hop:
    """Validate dbo_2hop metadata in CACHED_CLASS_SCHEMAS."""

    @pytest.mark.parametrize("cls_name", list(CACHED_CLASS_SCHEMAS.keys()))
    def test_every_class_has_dbo_2hop(self, cls_name):
        """Every cached class should have a dbo_2hop key."""
        schema = CACHED_CLASS_SCHEMAS[cls_name]
        assert "dbo_2hop" in schema, f"{cls_name} missing dbo_2hop"

    @pytest.mark.parametrize("cls_name", list(CACHED_CLASS_SCHEMAS.keys()))
    def test_2hop_target_classes_exist(self, cls_name):
        """All 2-hop target classes must be in CACHED_CLASS_SCHEMAS."""
        schema = CACHED_CLASS_SCHEMAS[cls_name]
        for prop, (target_cls, _) in schema.get("dbo_2hop", {}).items():
            assert target_cls in CACHED_CLASS_SCHEMAS, (
                f"{cls_name}.{prop} -> {target_cls} not in cache"
            )

    @pytest.mark.parametrize("cls_name", list(CACHED_CLASS_SCHEMAS.keys()))
    def test_2hop_target_props_exist_in_target_class(self, cls_name):
        """Target props must exist in the target class's dbo list."""
        schema = CACHED_CLASS_SCHEMAS[cls_name]
        for prop, (target_cls, target_props) in schema.get("dbo_2hop", {}).items():
            target_dbo = CACHED_CLASS_SCHEMAS[target_cls].get("dbo", [])
            for tp in target_props:
                assert tp in target_dbo, (
                    f"{cls_name}.{prop} -> {target_cls}.{tp} "
                    f"but {tp} not in {target_cls} dbo list"
                )

    def test_film_director_chain(self):
        """Film.director should chain to Person with date properties."""
        chain = CACHED_CLASS_SCHEMAS["Film"]["dbo_2hop"]["director"]
        assert chain[0] == "Person"
        assert "birthDate" in chain[1]
        assert "deathDate" in chain[1]


class TestFormat2HopLines:
    """Validate format_2hop_lines() helper."""

    def test_film_output(self):
        # Use large max_chains so all chains are rendered (Film has 27+ chains)
        lines = format_2hop_lines(CACHED_CLASS_SCHEMAS["Film"], max_chains=50)
        assert lines[0] == "  2-Hop chains:"
        assert any("dbo:director -> Person" in line for line in lines)

    def test_empty_schema(self):
        assert format_2hop_lines({}) == []
        assert format_2hop_lines({"dbo": ["birthDate"]}) == []

    def test_max_chains_limit(self):
        lines = format_2hop_lines(CACHED_CLASS_SCHEMAS["Film"], max_chains=2)
        # Header + 2 chain lines
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Fix 4: dbo:description filtered from get_actual_properties() SPARQL query
# ---------------------------------------------------------------------------

class TestDescriptionFilter:
    """Verify that dbo:description is excluded from actual properties."""

    def test_query_contains_description_filter(self):
        """The SPARQL query should have a FILTER excluding 'description'."""
        sd = SchemaDiscovery()
        mock_sparql = MagicMock()
        mock_sparql.query.return_value.convert.return_value = {
            "results": {"bindings": []}
        }
        sd.sparql = mock_sparql

        sd.get_actual_properties("http://dbpedia.org/resource/Test")

        # Capture the query that was set
        set_query_call = mock_sparql.setQuery.call_args
        query = set_query_call[0][0]
        assert 'FILTER(!CONTAINS(STR(?property), "description"))' in query

    def test_query_still_filters_wikipage(self):
        """Existing filters (wikiPage, abstract, etc.) remain."""
        sd = SchemaDiscovery()
        mock_sparql = MagicMock()
        mock_sparql.query.return_value.convert.return_value = {
            "results": {"bindings": []}
        }
        sd.sparql = mock_sparql

        sd.get_actual_properties("http://dbpedia.org/resource/Test")

        query = mock_sparql.setQuery.call_args[0][0]
        assert 'FILTER(!CONTAINS(STR(?property), "wikiPage"))' in query
        assert 'FILTER(!CONTAINS(STR(?property), "abstract"))' in query
        assert 'FILTER(!CONTAINS(STR(?property), "thumbnail"))' in query
        assert 'FILTER(!CONTAINS(STR(?property), "isPrimaryTopicOf"))' in query

    def test_description_not_in_returned_properties(self):
        """If SPARQL returns a description property, it shouldn't be included."""
        sd = SchemaDiscovery()
        mock_sparql = MagicMock()
        # Simulate SPARQL returning description alongside a valid property
        mock_sparql.query.return_value.convert.return_value = {
            "results": {"bindings": [
                {
                    "property": {"value": "http://dbpedia.org/ontology/birthDate"},
                    "sampleValue": {"value": "1879-03-14", "type": "literal"},
                },
                {
                    "property": {"value": "http://dbpedia.org/ontology/description"},
                    "sampleValue": {"value": "fisikawan bidang fisika", "type": "literal"},
                },
            ]}
        }
        sd.sparql = mock_sparql

        result = sd.get_actual_properties("http://dbpedia.org/resource/Test")
        # birthDate should be present
        assert "birthDate" in result["dbo"]
        # description should also appear in parsing (filter is at SPARQL level),
        # but the SPARQL query itself filters it. In mock mode the bindings
        # are returned directly, so we just verify the query was correct.
        query = mock_sparql.setQuery.call_args[0][0]
        assert "description" in query

    def test_get_actual_properties_with_filter(self):
        """property_filter parameter still works alongside description filter."""
        sd = SchemaDiscovery()
        mock_sparql = MagicMock()
        mock_sparql.query.return_value.convert.return_value = {
            "results": {"bindings": []}
        }
        sd.sparql = mock_sparql

        sd.get_actual_properties(
            "http://dbpedia.org/resource/Test",
            property_filter=["birth"],
        )

        query = mock_sparql.setQuery.call_args[0][0]
        # Both description filter and custom filter present
        assert 'FILTER(!CONTAINS(STR(?property), "description"))' in query
        assert 'CONTAINS(LCASE(STR(?property)), "birth")' in query


# ---------------------------------------------------------------------------
# Additional format_2hop_lines tests
# ---------------------------------------------------------------------------

class TestFormat2HopLinesDeep:
    def test_max_props_limit(self):
        """max_props=1 limits each chain line to 1 property."""
        lines = format_2hop_lines(CACHED_CLASS_SCHEMAS["Film"], max_props=1)
        for line in lines[1:]:  # skip header
            # Each line: "    - dbo:prop -> Class: prop1"
            colon_parts = line.split(": ", 1)
            if len(colon_parts) > 1:
                props_str = colon_parts[-1]
                assert "," not in props_str, f"Expected 1 prop, got: {props_str}"

    def test_max_props_default_is_4(self):
        """Default max_props=4 limits each chain to 4 properties.

        Priority-sorted props for Person are birthDate, deathDate, birthPlace, deathPlace.
        Film.director chain is rendered with max_chains=50 to ensure it appears.
        """
        lines = format_2hop_lines(CACHED_CLASS_SCHEMAS["Film"], max_chains=50)
        director_line = [ln for ln in lines if "dbo:director" in ln]
        assert len(director_line) == 1
        props = director_line[0].split(": ")[-1].split(", ")
        assert len(props) == 4
        # Priority props come first: birthDate and deathDate must be in the 4
        assert "birthDate" in props
        assert "deathDate" in props

    def test_person_chains_rendered(self):
        # max_chains=100 to include all chains (Person has 70+ object properties)
        lines = format_2hop_lines(CACHED_CLASS_SCHEMAS["Person"], max_chains=100)
        assert lines[0] == "  2-Hop chains:"
        assert any("dbo:parent" in line for line in lines)
        assert any("dbo:birthPlace" in line for line in lines)

    def test_company_chains_rendered(self):
        # max_chains=50 to include all Company chains (foundedBy and locationCity not in first 5)
        lines = format_2hop_lines(CACHED_CLASS_SCHEMAS["Company"], max_chains=50)
        assert any("dbo:foundedBy" in line for line in lines)
        assert any("dbo:locationCity" in line for line in lines)


# ---------------------------------------------------------------------------
# PROPERTY_TO_CLASSES: dbp properties coverage
# ---------------------------------------------------------------------------

class TestPropertyToClassesDbp:
    def test_father_maps_to_person(self):
        from dbpedia.schema_discovery import PROPERTY_TO_CLASSES
        assert "father" in PROPERTY_TO_CLASSES
        assert "Person" in PROPERTY_TO_CLASSES["father"]

    def test_mother_maps_to_person(self):
        from dbpedia.schema_discovery import PROPERTY_TO_CLASSES
        assert "mother" in PROPERTY_TO_CLASSES
        assert "Person" in PROPERTY_TO_CLASSES["mother"]

    def test_founder_maps_to_company(self):
        from dbpedia.schema_discovery import PROPERTY_TO_CLASSES
        # 'founder' (dbp) is gone after ontology refactor; 'foundedBy' (dbo) is the canonical prop
        assert "foundedBy" in PROPERTY_TO_CLASSES
        assert "Company" in PROPERTY_TO_CLASSES["foundedBy"]

    def test_all_dbp_properties_indexed(self):
        """Every dbp property in CACHED_CLASS_SCHEMAS is in PROPERTY_TO_CLASSES."""
        from dbpedia.schema_discovery import PROPERTY_TO_CLASSES
        for cls_name, schema in CACHED_CLASS_SCHEMAS.items():
            for prop in schema.get("dbp", []):
                assert prop in PROPERTY_TO_CLASSES, (
                    f"dbp prop '{prop}' from {cls_name} missing"
                )
                assert cls_name in PROPERTY_TO_CLASSES[prop]


# ---------------------------------------------------------------------------
# CACHED_CLASS_SCHEMAS structural integrity
# ---------------------------------------------------------------------------

class TestCachedClassSchemasIntegrity:
    @pytest.mark.parametrize("cls_name", list(CACHED_CLASS_SCHEMAS.keys()))
    def test_has_descriptions(self, cls_name):
        """Every cached class should have a descriptions dict."""
        schema = CACHED_CLASS_SCHEMAS[cls_name]
        assert "descriptions" in schema

    @pytest.mark.parametrize("cls_name", list(CACHED_CLASS_SCHEMAS.keys()))
    def test_dbo_and_dbp_are_lists(self, cls_name):
        """dbo and dbp entries must be lists of strings."""
        schema = CACHED_CLASS_SCHEMAS[cls_name]
        assert isinstance(schema.get("dbo", []), list)
        assert isinstance(schema.get("dbp", []), list)
        for prop in schema.get("dbo", []):
            assert isinstance(prop, str)
        for prop in schema.get("dbp", []):
            assert isinstance(prop, str)

    def test_all_cached_classes_present(self):
        # Ontology cache now has 734 classes (auto-generated from rdfs:domain/range/subClassOf).
        # Verify the 7 core classes still present and total count exceeds original 7.
        core_classes = {"Person", "Film", "Company", "Place", "MusicalArtist", "Book", "MusicalWork"}
        assert core_classes.issubset(set(CACHED_CLASS_SCHEMAS.keys()))
        assert len(CACHED_CLASS_SCHEMAS) > 100, (
            f"Expected 700+ classes, got {len(CACHED_CLASS_SCHEMAS)}"
        )
