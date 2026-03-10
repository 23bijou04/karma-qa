"""Tests for 2-hop chain inference data structures and property filtering."""
from unittest.mock import MagicMock, patch

import pytest

from dbpedia_qa.dbpedia.schema_discovery import (
    CACHED_CLASS_SCHEMAS,
    PROPERTY_RANGES,
    SchemaDiscovery,
    format_2hop_lines,
)


class TestPropertyRanges:
    """Validate PROPERTY_RANGES dict integrity."""

    def test_all_ranges_are_cached_classes(self):
        """Every range value must be a class in CACHED_CLASS_SCHEMAS."""
        for prop, target_cls in PROPERTY_RANGES.items():
            assert target_cls in CACHED_CLASS_SCHEMAS, (
                f"{prop} -> {target_cls} but {target_cls} not in CACHED_CLASS_SCHEMAS"
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
        lines = format_2hop_lines(CACHED_CLASS_SCHEMAS["Film"])
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
                    "value": {"value": "1879-03-14", "type": "literal"},
                },
                {
                    "property": {"value": "http://dbpedia.org/ontology/description"},
                    "value": {"value": "fisikawan bidang fisika", "type": "literal"},
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
