"""Tests for Case 12: URI encoding safe characters."""

from urllib.parse import quote

# The safe chars used in _sparql_results_to_entities and entity_to_uri
SAFE_CHARS = "/:@!$&'()*+,;=-._~%"


class TestURISafeChars:
    def test_apostrophe_preserved(self):
        """O'Brien stays unencoded in DBpedia URI."""
        local_name = "O'Brien"
        encoded = quote(local_name, safe=SAFE_CHARS)
        assert encoded == "O'Brien"

    def test_comma_preserved(self):
        """St._Louis,_Missouri stays unencoded."""
        local_name = "St._Louis,_Missouri"
        encoded = quote(local_name, safe=SAFE_CHARS)
        assert encoded == "St._Louis,_Missouri"

    def test_parentheses_preserved(self):
        """Parentheses stay unencoded (common in DBpedia disambiguation)."""
        local_name = "Polish-Russian_War_(film)"
        encoded = quote(local_name, safe=SAFE_CHARS)
        assert encoded == "Polish-Russian_War_(film)"

    def test_spaces_encoded(self):
        """Spaces should be percent-encoded."""
        local_name = "Albert Einstein"
        encoded = quote(local_name, safe=SAFE_CHARS)
        assert " " not in encoded
        assert "Albert%20Einstein" == encoded


class TestSparqlResultsToEntities:
    def test_apostrophe_roundtrip(self):
        """Entity with apostrophe survives SPARQL result -> Entity roundtrip."""
        from routing import _sparql_results_to_entities
        results = [{"entity": "O'Brien", "label": "O'Brien"}]
        entities = _sparql_results_to_entities(results, set())
        assert len(entities) == 1
        assert "O'Brien" in entities[0].uri
        assert "%27" not in entities[0].uri

    def test_full_uri_preserved(self):
        """Full DBpedia URI passes through without re-encoding."""
        from routing import _sparql_results_to_entities
        full_uri = "http://dbpedia.org/resource/O'Brien"
        results = [{"entity": full_uri, "label": "O'Brien"}]
        entities = _sparql_results_to_entities(results, set())
        assert entities[0].uri == full_uri

    def test_comma_roundtrip(self):
        """Entity with comma survives roundtrip."""
        from routing import _sparql_results_to_entities
        results = [{"entity": "St. Louis, Missouri", "label": "St. Louis, Missouri"}]
        entities = _sparql_results_to_entities(results, set())
        assert len(entities) == 1
        assert "%2C" not in entities[0].uri


    def test_arbitrary_variable_name(self):
        """LLM-generated variable names (not in hardcoded list) still extract entities."""
        from routing import _sparql_results_to_entities
        uri = "http://dbpedia.org/resource/Christopher_Nolan"
        results = [{"director": uri}]
        entities = _sparql_results_to_entities(results, seed_uris=set())
        assert len(entities) == 1
        assert entities[0].uri == uri

    def test_multiple_resource_uris_picks_first(self):
        """When row has multiple dbr: URIs, first one (by insertion order) is picked."""
        from routing import _sparql_results_to_entities
        results = [{
            "film": "http://dbpedia.org/resource/Inception",
            "director": "http://dbpedia.org/resource/Christopher_Nolan",
        }]
        entities = _sparql_results_to_entities(results, seed_uris=set())
        assert len(entities) == 1
        # First URI in insertion order wins
        assert entities[0].uri == "http://dbpedia.org/resource/Inception"

    def test_ontology_uri_not_extracted(self):
        """dbo: URIs are not extracted as entity URIs."""
        from routing import _sparql_results_to_entities
        results = [{"p": "http://dbpedia.org/ontology/director"}]
        entities = _sparql_results_to_entities(results, seed_uris=set())
        assert len(entities) == 0


class TestExecuteSparqlPreserveURIs:
    def test_preserve_uris_default_false(self):
        """Default preserve_uris=False still works (backward compatibility)."""
        import inspect

        from sparql_evaluation import execute_sparql
        sig = inspect.signature(execute_sparql)
        assert sig.parameters["preserve_uris"].default is False
