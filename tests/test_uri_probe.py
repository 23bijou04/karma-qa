"""Tests for _construct_dbpedia_uri, _probe_single_uri, _probe_uri_variants."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from dbpedia.entity_linker import DBpediaEntityLinker  # noqa: E402


def _make_linker():
    linker = DBpediaEntityLinker.__new__(DBpediaEntityLinker)
    linker.sparql_client = MagicMock()
    linker.sparql_client.ask_entity_exists.return_value = False
    linker.sparql_client.validate_entity_uri.return_value = {
        "exists": True, "redirect_target": None, "is_disambiguation": False,
    }
    linker.sparql_client.get_entity_types.return_value = []
    return linker


# ---------------------------------------------------------------------------
# _construct_dbpedia_uri — static method, no mocking required
# ---------------------------------------------------------------------------

def test_construct_plain_title():
    uri = DBpediaEntityLinker._construct_dbpedia_uri("Return of the Hero")
    assert uri == "http://dbpedia.org/resource/Return_of_the_Hero"


def test_construct_strips_trailing_punctuation():
    uri = DBpediaEntityLinker._construct_dbpedia_uri("lincoln.")
    assert uri == "http://dbpedia.org/resource/Lincoln"


def test_construct_ucfirst():
    uri = DBpediaEntityLinker._construct_dbpedia_uri("the Who")
    assert uri == "http://dbpedia.org/resource/The_Who"


def test_construct_preserves_internal_caps():
    uri = DBpediaEntityLinker._construct_dbpedia_uri("A Couch in New York")
    assert uri == "http://dbpedia.org/resource/A_Couch_in_New_York"


# ---------------------------------------------------------------------------
# _probe_single_uri
# ---------------------------------------------------------------------------

def test_probe_single_uri_exists_correct_type():
    linker = _make_linker()
    linker.sparql_client.ask_entity_exists.return_value = True
    linker.sparql_client.get_entity_types.return_value = ["Film"]
    result = linker._probe_single_uri(
        "http://dbpedia.org/resource/Inception", "dbo:Film"
    )
    assert result is not None
    assert result.uri == "http://dbpedia.org/resource/Inception"
    assert result.source == "uri_probe"


def test_probe_single_uri_not_exists():
    linker = _make_linker()
    linker.sparql_client.ask_entity_exists.return_value = False
    result = linker._probe_single_uri(
        "http://dbpedia.org/resource/Nonexistent", "dbo:Film"
    )
    assert result is None


def test_probe_single_uri_disambiguation_skipped():
    linker = _make_linker()
    linker.sparql_client.ask_entity_exists.return_value = True
    linker.sparql_client.validate_entity_uri.return_value = {
        "exists": True, "redirect_target": None, "is_disambiguation": True,
    }
    result = linker._probe_single_uri(
        "http://dbpedia.org/resource/Lincoln", None
    )
    assert result is None


def test_probe_single_uri_follows_redirect():
    linker = _make_linker()
    linker.sparql_client.ask_entity_exists.return_value = True
    linker.sparql_client.validate_entity_uri.return_value = {
        "exists": True,
        "redirect_target": "http://dbpedia.org/resource/Abraham_Lincoln",
        "is_disambiguation": False,
    }
    linker.sparql_client.get_entity_types.return_value = ["Person"]
    result = linker._probe_single_uri(
        "http://dbpedia.org/resource/Lincoln", "dbo:Person"
    )
    assert result is not None
    assert result.uri == "http://dbpedia.org/resource/Abraham_Lincoln"


# ---------------------------------------------------------------------------
# _probe_uri_variants
# ---------------------------------------------------------------------------

def test_probe_variants_plain_uri_exists():
    linker = _make_linker()
    linker.sparql_client.ask_entity_exists.return_value = True
    linker.sparql_client.get_entity_types.return_value = ["Film"]
    result = linker._probe_uri_variants("Inception", "dbo:Film")
    assert result is not None
    assert result.label == "Inception"
    assert result.source == "uri_probe"


def test_probe_variants_plain_missing_suffix_hit():
    """Plain URI absent; _(film) suffix URI exists."""
    linker = _make_linker()
    call_count = {"n": 0}

    def ask_side_effect(uri):
        call_count["n"] += 1
        return "_(film)" in uri  # only suffixed URI exists

    linker.sparql_client.ask_entity_exists.side_effect = ask_side_effect
    linker.sparql_client.get_entity_types.return_value = ["Film"]
    result = linker._probe_uri_variants("Being Human", "dbo:Film")
    assert result is not None
    assert "_(film)" in result.uri


def test_probe_variants_all_fail():
    linker = _make_linker()
    linker.sparql_client.ask_entity_exists.return_value = False
    result = linker._probe_uri_variants("Nonexistent Title", "dbo:Film")
    assert result is None


def test_probe_variants_no_type_hint_only_plain_probed():
    """Without type_hint, only the plain URI is probed (no suffix variants)."""
    linker = _make_linker()
    linker.sparql_client.ask_entity_exists.return_value = False
    linker._probe_uri_variants("Some Entity", None)
    assert linker.sparql_client.ask_entity_exists.call_count == 1
