"""Tests for type-guided entity discovery."""

from unittest.mock import MagicMock

from models import Entity
from entity_discovery import (
    _score_candidates,
    discover_context_entities,
)


def _make_bindings(*entities):
    """Helper: create SPARQL-style bindings."""
    return [
        {"entity": {"value": uri}, "label": {"value": label}}
        for uri, label in entities
    ]


def _mock_kg_client(bindings=None, ask_results=None):
    """Create a mock kg_client with configurable query responses."""
    client = MagicMock()
    ask_results = ask_results or {}

    def query_side_effect(q):
        if q.strip().startswith("ASK"):
            for uri_fragment, result in ask_results.items():
                if uri_fragment in q:
                    return {"boolean": result}
            return {"boolean": False}
        return {"results": {"bindings": bindings or []}}

    client.query.side_effect = query_side_effect
    return client


def _mock_projection(expected_type="Person", bridge_property="director"):
    """Create a mock SchemaProjection."""
    proj = MagicMock()
    proj.question_analysis.expected_target_type = expected_type
    proj.question_analysis.bridge_property = bridge_property
    proj.entity_profile.actual_properties = {"values": {}}
    return proj


class TestScoring:
    def test_label_overlap_adds_score(self):
        bindings = _make_bindings(
            ("http://dbpedia.org/resource/Einstein", "Albert Einstein"),
        )
        scored = _score_candidates(
            bindings, "http://dbpedia.org/resource/X",
            "When was Einstein born?", None, set(),
            _mock_kg_client(),
        )
        assert scored[0].score >= 1
        assert any("label_overlap" in r for r in scored[0].reasons)

    def test_already_known_penalized(self):
        uri = "http://dbpedia.org/resource/Known"
        bindings = _make_bindings((uri, "Known Entity"))
        scored = _score_candidates(
            bindings, "http://dbpedia.org/resource/X",
            "test question", None, {uri},
            _mock_kg_client(),
        )
        assert scored[0].score <= 0
        assert any("already_known" in r for r in scored[0].reasons)

    def test_backlink_bonus(self):
        seed = "http://dbpedia.org/resource/Inception"
        cand = "http://dbpedia.org/resource/Nolan"
        bindings = _make_bindings((cand, "Christopher Nolan"))
        client = _mock_kg_client(ask_results={cand: True})

        scored = _score_candidates(
            bindings, seed, "test", None, set(), client,
        )
        assert scored[0].score >= 3
        assert any("backlink" in r for r in scored[0].reasons)


class TestDiscoverContextEntities:
    def test_type_filtered_discovery(self):
        """Returns entities when type-filtered query has results."""
        bindings = _make_bindings(
            ("http://dbpedia.org/resource/Nolan", "Christopher Nolan"),
        )
        client = _mock_kg_client(bindings=bindings)
        seed = Entity(uri="http://dbpedia.org/resource/Inception", label="Inception")
        proj = _mock_projection(expected_type="Person")

        result = discover_context_entities([seed], [proj], client, question="test")
        assert len(result) >= 2  # seed + discovered
        discovered = [e for e in result if e.source == "sparql_neighbor"]
        assert len(discovered) >= 1

    def test_fallback_to_unfiltered(self):
        """When type-filtered returns empty, tries unfiltered."""
        call_count = {"n": 0}

        def query_side_effect(q):
            call_count["n"] += 1
            if q.strip().startswith("ASK"):
                return {"boolean": False}
            # First call (type-filtered) returns empty, second (unfiltered) returns data
            if call_count["n"] == 1:
                return {"results": {"bindings": []}}
            return {"results": {"bindings": _make_bindings(
                ("http://dbpedia.org/resource/X", "Entity X"),
            )}}

        client = MagicMock()
        client.query.side_effect = query_side_effect
        seed = Entity(uri="http://dbpedia.org/resource/Y", label="Y")
        proj = _mock_projection(expected_type="Person")

        result = discover_context_entities([seed], [proj], client, question="test")
        discovered = [e for e in result if e.source == "sparql_neighbor"]
        assert len(discovered) >= 1

    def test_empty_discovery_returns_seeds(self):
        """No wikiPageWikiLink at all -> returns seed entities only."""
        client = _mock_kg_client(bindings=[])
        seed = Entity(uri="http://dbpedia.org/resource/Lonely", label="Lonely")
        proj = _mock_projection()

        result = discover_context_entities([seed], [proj], client, question="test")
        assert len(result) == 1
        assert result[0].uri == seed.uri

    def test_deduplication(self):
        """Same URI from multiple seeds appears once."""
        bindings = _make_bindings(
            ("http://dbpedia.org/resource/Shared", "Shared Entity"),
        )
        client = _mock_kg_client(bindings=bindings)
        seed1 = Entity(uri="http://dbpedia.org/resource/A", label="A")
        seed2 = Entity(uri="http://dbpedia.org/resource/B", label="B")
        proj = _mock_projection()

        result = discover_context_entities([seed1, seed2], [proj], client, question="test")
        shared = [e for e in result if "Shared" in e.label]
        assert len(shared) == 1

    def test_no_projections_returns_seeds(self):
        """No projections -> returns seed entities only."""
        seed = Entity(uri="http://dbpedia.org/resource/X", label="X")
        result = discover_context_entities([seed], [], MagicMock(), question="test")
        assert len(result) == 1

    def test_source_is_sparql_neighbor(self):
        """Discovered entities have source='sparql_neighbor'."""
        bindings = _make_bindings(
            ("http://dbpedia.org/resource/Found", "Found Entity"),
        )
        client = _mock_kg_client(bindings=bindings)
        seed = Entity(uri="http://dbpedia.org/resource/Seed", label="Seed")
        proj = _mock_projection()

        result = discover_context_entities([seed], [proj], client, question="test")
        discovered = [e for e in result if e.uri != seed.uri]
        for e in discovered:
            assert e.source == "sparql_neighbor"
