"""Tests for Case 14: Wikipedia native search API + circuit breaker."""

from unittest.mock import MagicMock, patch

from graphrag_wiki import _CircuitBreaker, web_search_wikipedia


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = _CircuitBreaker(threshold=3, cooldown=60.0)
        assert not cb.is_open()

    def test_trips_after_threshold_failures(self):
        cb = _CircuitBreaker(threshold=3, cooldown=60.0)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open()
        cb.record_failure()  # threshold reached
        assert cb.is_open()

    def test_resets_on_success(self):
        cb = _CircuitBreaker(threshold=2, cooldown=60.0)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert not cb.is_open()  # reset after success

    def test_cooldown_resets(self):
        """Circuit breaker resets after cooldown period."""
        import time
        cb = _CircuitBreaker(threshold=1, cooldown=0.1)
        cb.record_failure()
        assert cb.is_open()
        time.sleep(0.15)
        assert not cb.is_open()  # cooldown expired


class TestWebSearchWikipedia:
    def test_returns_titles_from_wiki_api(self):
        """Wikipedia native search returns titles on success."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "query": {"search": [
                {"title": "Albert Einstein"},
                {"title": "Einstein (film)"},
            ]}
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("graphrag_wiki.requests.get", return_value=mock_resp):
            titles = web_search_wikipedia("Albert Einstein", limit=5)
        assert "Albert Einstein" in titles
        assert len(titles) == 2

    def test_falls_back_to_ddg_on_wiki_failure(self):
        """Falls back to DDG when Wikipedia API fails."""
        # Make Wikipedia API fail
        with patch("graphrag_wiki.requests.get", side_effect=Exception("API down")):
            # DDG fallback also mocked to avoid real network calls
            with patch("graphrag_wiki.DDGS") as mock_ddgs:
                mock_ddgs.return_value.text.return_value = [
                    {"href": "https://en.wikipedia.org/wiki/Albert_Einstein"}
                ]
                titles = web_search_wikipedia("Albert Einstein", limit=3)
        assert "Albert Einstein" in titles

    def test_empty_on_both_fail(self):
        """Returns empty list when both Wikipedia API and DDG fail."""
        with patch("graphrag_wiki.requests.get", side_effect=Exception("fail")):
            with patch("graphrag_wiki.DDGS") as mock_ddgs:
                mock_ddgs.return_value.text.side_effect = Exception("DDG fail")
                titles = web_search_wikipedia("Nonexistent", limit=3)
        assert titles == []
