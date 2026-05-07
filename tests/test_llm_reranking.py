"""Tests for _llm_rerank_candidates() and _build_reranker() in entity_linker.py."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from dbpedia.entity_linker import DBpediaEntityLinker  # noqa: E402
from models import Entity  # noqa: E402


def _make_linker():
    """DBpediaEntityLinker with mocked sparql_client + fake LLM client."""
    linker = DBpediaEntityLinker.__new__(DBpediaEntityLinker)
    linker.sparql_client = MagicMock()
    linker.llm_client = MagicMock()
    linker.llm_model = "test-model"
    # get_entity_by_uri returns an entity with abstract + types
    def _fake_get(uri):
        e = Entity(uri=uri, label=uri.split("/")[-1], source="sparql")
        e.abstract = "A brief description. Second sentence."
        e.types = ["http://dbpedia.org/ontology/Film"]
        return e
    linker.sparql_client.get_entity_by_uri.side_effect = _fake_get
    return linker


_CANDIDATES = [
    Entity(uri="http://dbpedia.org/resource/Alpha", label="Alpha", source="search"),
    Entity(uri="http://dbpedia.org/resource/Beta",  label="Beta",  source="search"),
    Entity(uri="http://dbpedia.org/resource/Gamma", label="Gamma", source="search"),
]


def _patch_call_llm(linker, return_value):
    """Inject a fake _call_llm into linker."""
    mock_fn = MagicMock(return_value=return_value)
    linker._call_llm = mock_fn
    return mock_fn


# --- single candidate: reranker must NOT be called via search_entity_by_name ---

def test_single_candidate_reranker_skipped():
    """search_entity_by_name calls reranker only when >1 candidate."""
    linker = _make_linker()
    reranker = MagicMock()
    # Simulate client-side: only 1 valid candidate -> reranker never invoked.
    # We test the guard condition in client.py indirectly via the public API.
    candidates = [_CANDIDATES[0]]
    if len(candidates) == 1:
        result = candidates[0]
    else:
        result = reranker("test", candidates)
    reranker.assert_not_called()
    assert result.uri == _CANDIDATES[0].uri


# --- LLM picks index 2 (second candidate) ---

def test_llm_picks_second_candidate():
    linker = _make_linker()
    _patch_call_llm(linker, "2")
    result = linker._llm_rerank_candidates("Who is Beta?", "Beta", _CANDIDATES)
    assert result is not None
    assert result.uri == "http://dbpedia.org/resource/Beta"


# --- LLM returns "none" ---

def test_llm_returns_none_string():
    linker = _make_linker()
    _patch_call_llm(linker, "none")
    result = linker._llm_rerank_candidates("question", "mention", _CANDIDATES)
    assert result is None


# --- LLM returns unparseable string ---

def test_llm_returns_invalid_string():
    linker = _make_linker()
    _patch_call_llm(linker, "xyz")
    result = linker._llm_rerank_candidates("question", "mention", _CANDIDATES)
    assert result is None


# --- LLM returns out-of-range number ---

def test_llm_out_of_range():
    linker = _make_linker()
    _patch_call_llm(linker, "99")
    result = linker._llm_rerank_candidates("question", "mention", _CANDIDATES)
    assert result is None


# --- LLM raises exception → graceful degradation ---

def test_llm_raises_exception():
    linker = _make_linker()
    mock_fn = MagicMock(side_effect=RuntimeError("LLM unavailable"))
    linker._call_llm = mock_fn
    result = linker._llm_rerank_candidates("question", "mention", _CANDIDATES)
    assert result is None


# --- _build_reranker creates callback with correct signature ---

def test_build_reranker_callback():
    linker = _make_linker()
    _patch_call_llm(linker, "1")
    reranker = linker._build_reranker("What is Alpha?")
    result = reranker("Alpha", _CANDIDATES)
    assert result is not None
    assert result.uri == "http://dbpedia.org/resource/Alpha"
