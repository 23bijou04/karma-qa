"""Tests for SPARQL read-only guard (C1 fix)."""

from sparql_solver import _is_readonly_sparql


class TestSparqlReadonly:
    """Verify _is_readonly_sparql rejects write operations."""

    def test_select_allowed(self):
        assert _is_readonly_sparql("SELECT ?x WHERE { ?x a dbo:Film }") is True

    def test_ask_allowed(self):
        assert _is_readonly_sparql("ASK { <http://dbpedia.org/resource/X> a dbo:Film }") is True

    def test_construct_allowed(self):
        assert _is_readonly_sparql("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }") is True

    def test_describe_allowed(self):
        assert _is_readonly_sparql("DESCRIBE <http://dbpedia.org/resource/X>") is True

    def test_insert_rejected(self):
        assert _is_readonly_sparql("INSERT DATA { <x> <y> <z> }") is False

    def test_delete_rejected(self):
        assert _is_readonly_sparql("DELETE WHERE { <x> <y> <z> }") is False

    def test_drop_rejected(self):
        assert _is_readonly_sparql("DROP GRAPH <http://example.org/g>") is False

    def test_clear_rejected(self):
        assert _is_readonly_sparql("CLEAR DEFAULT") is False

    def test_case_insensitive(self):
        assert _is_readonly_sparql("insert data { <x> <y> <z> }") is False
        assert _is_readonly_sparql("Delete Where { <x> <y> <z> }") is False

    def test_insert_inside_select_rejected(self):
        """Even if embedded in a larger query, write ops are rejected."""
        q = "SELECT ?x WHERE { ?x a dbo:Film } ; INSERT DATA { <x> <y> <z> }"
        assert _is_readonly_sparql(q) is False

    def test_delete_substring_in_uri_ok(self):
        """'Delete' inside a URI like Delete_scene is not a SPARQL keyword."""
        # \b doesn't match between 'e' and '_' (both word chars), so no false positive
        q = 'SELECT ?x WHERE { <http://dbpedia.org/resource/Delete_scene> ?p ?x }'
        assert _is_readonly_sparql(q) is True
