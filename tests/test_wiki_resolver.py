"""Tests for WikipediaResolver: search_candidates, resolve_redirect, resolve_to_dbpedia_uris."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from dbpedia_qa.dbpedia.wiki_resolver import WikipediaResolver  # noqa: E402


def _mock_resp(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _search_payload(*titles: str) -> dict:
    return {"query": {"search": [{"title": t} for t in titles]}}


def _redirect_payload(canonical: str | None, page_id: str = "123") -> dict:
    data: dict = {"query": {}}
    if canonical:
        data["query"]["redirects"] = [{"to": canonical}]
        data["query"]["pages"] = {page_id: {"title": canonical}}
    else:
        data["query"]["redirects"] = []
        data["query"]["pages"] = {page_id: {"title": "Some Page"}}
    return data


def _missing_page_payload() -> dict:
    return {"query": {"redirects": [], "pages": {"-1": {"title": "Missing"}}}}


# ---------------------------------------------------------------------------
# search_candidates
# ---------------------------------------------------------------------------

def test_search_candidates_returns_titles():
    resolver = WikipediaResolver()
    payload = _search_payload("Inception (film)", "Inception", "Inception 2010")
    with patch("httpx.get", return_value=_mock_resp(payload)):
        titles = resolver.search_candidates("Inception", "film")
    assert titles == ["Inception (film)", "Inception", "Inception 2010"]


def test_search_candidates_cache_hit_skips_api():
    resolver = WikipediaResolver(cache={"Inception film": ["Inception (film)"]})
    with patch("httpx.get") as mock_get:
        titles = resolver.search_candidates("Inception", "film")
    mock_get.assert_not_called()
    assert titles == ["Inception (film)"]


def test_search_candidates_api_error_returns_empty():
    resolver = WikipediaResolver()
    with patch("httpx.get", side_effect=Exception("network error")):
        titles = resolver.search_candidates("Broken Query", "")
    assert titles == []


# ---------------------------------------------------------------------------
# resolve_redirect
# ---------------------------------------------------------------------------

def test_resolve_redirect_follows_redirect():
    resolver = WikipediaResolver()
    payload = _redirect_payload("Abraham Lincoln")
    with patch("httpx.get", return_value=_mock_resp(payload)):
        title = resolver.resolve_redirect("Lincoln")
    assert title == "Abraham Lincoln"


def test_resolve_redirect_no_redirect_returns_page_title():
    resolver = WikipediaResolver()
    payload = _redirect_payload(None)
    payload["query"]["pages"] = {"42": {"title": "Some Page"}}
    with patch("httpx.get", return_value=_mock_resp(payload)):
        title = resolver.resolve_redirect("Some Page")
    assert title == "Some Page"


def test_resolve_redirect_missing_page_returns_none():
    resolver = WikipediaResolver()
    with patch("httpx.get", return_value=_mock_resp(_missing_page_payload())):
        title = resolver.resolve_redirect("Completely Missing Article")
    assert title is None


# ---------------------------------------------------------------------------
# resolve_to_dbpedia_uris — end-to-end
# ---------------------------------------------------------------------------

def test_resolve_to_dbpedia_uris_end_to_end():
    resolver = WikipediaResolver()
    search_payload = _search_payload("Return of the Hero (film)")
    # redirect resolves to canonical French title
    redirect_payload = _redirect_payload("Le Retour du héros")

    responses = iter([
        _mock_resp(search_payload),
        _mock_resp(redirect_payload),
    ])
    with patch("httpx.get", side_effect=lambda *a, **kw: next(responses)):
        uris = resolver.resolve_to_dbpedia_uris("Return of the Hero", "dbo:Film")

    assert len(uris) == 1
    assert uris[0] == "http://dbpedia.org/resource/Le_Retour_du_héros"


def test_resolve_to_dbpedia_uris_deduplicates():
    """Two search results that both redirect to the same canonical title produce one URI."""
    resolver = WikipediaResolver()
    search_payload = _search_payload("Lincoln (film)", "Lincoln (2012 film)")
    redirect_payload = _redirect_payload("Lincoln (film)")

    responses = [_mock_resp(redirect_payload)] * 2
    it = iter(responses)
    with patch("httpx.get", side_effect=lambda *a, **kw: next(it)):
        resolver._cache["Lincoln film"] = ["Lincoln (film)", "Lincoln (2012 film)"]
        uris = resolver.resolve_to_dbpedia_uris("Lincoln", "dbo:Film")

    # Both resolve to same URI — should be deduplicated
    assert len(set(uris)) == len(uris)
