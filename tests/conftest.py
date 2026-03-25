"""Test configuration — ensure project root is on sys.path."""
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Custom markers for stress test severity filtering
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "stress: stress test from edge-case catalog")
    config.addinivalue_line("markers", "p0: P0 severity (critical)")
    config.addinivalue_line("markers", "p1: P1 severity (high)")
    config.addinivalue_line("markers", "p2: P2 severity (medium)")
    config.addinivalue_line("markers", "p3: P3 severity (low)")
    config.addinivalue_line("markers", "integration: integration test (needs network)")


# ---------------------------------------------------------------------------
# Shared fixtures for stress tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spacy_nlp():
    """Load spaCy model once per session."""
    import spacy
    return spacy.load("en_core_web_sm")


@pytest.fixture
def mock_sparql_client():
    """Mock DBpediaSPARQLClient with configurable responses."""
    from dbpedia_qa.dbpedia.client import DBpediaSPARQLClient
    client = create_autospec(DBpediaSPARQLClient, instance=True)
    client.get_entity_types.return_value = []
    client.search_entity_by_name.return_value = None
    client.get_entity_by_label.return_value = None
    return client


@pytest.fixture
def linker_no_network(mock_sparql_client):
    """DBpediaEntityLinker that never hits the network."""
    from dbpedia_qa.dbpedia.entity_linker import DBpediaEntityLinker
    linker = DBpediaEntityLinker.__new__(DBpediaEntityLinker)
    linker.sparql_client = mock_sparql_client
    linker.spotlight_endpoint = "http://localhost:0/annotate"
    linker.cache = None
    linker.min_confidence = 0.5
    linker.min_support = 5
    return linker


# ---------------------------------------------------------------------------
# Helper assertion functions (importable by stress test files)
# ---------------------------------------------------------------------------

def assert_mention_found(mentions, expected_fragment):
    """Assert at least one mention contains the expected fragment (case-insensitive)."""
    assert any(expected_fragment.lower() in m.lower() for m in mentions), \
        f"Expected '{expected_fragment}' in mentions, got: {mentions}"


def assert_mention_missing(mentions, unexpected_exact):
    """Assert no mention exactly matches the unexpected string (case-insensitive)."""
    assert not any(unexpected_exact.lower() == m.lower() for m in mentions), \
        f"Unexpected '{unexpected_exact}' found as exact mention in: {mentions}"


def assert_mentions_exact(mentions, expected_set):
    """Assert the mention set matches exactly (case-insensitive, order-independent)."""
    actual = {m.lower() for m in mentions}
    expected = {e.lower() for e in expected_set}
    assert actual == expected, f"Expected mentions {expected}, got {actual}"
