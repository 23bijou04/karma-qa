"""Tests for edge case fixes identified in codebase review (2026-03-25).

C1: SERVICE keyword must be blocked by SPARQL read-only guard
L3: Spotlight should have a circuit breaker (reuse _CircuitBreaker)
C3: Pipeline-level timeout should exist in batch eval

These tests are written RED-first: they document the expected behavior.
Before the fix, some will FAIL (proving the bug exists).
After the fix, all should PASS.
"""

import re
import time

# ---------------------------------------------------------------------------
# C1: SERVICE keyword blocked by SPARQL read-only guard
# ---------------------------------------------------------------------------
from sparql_solver import _is_readonly_sparql


class TestServiceKeywordBlocked:
    """C1: _is_readonly_sparql must reject SERVICE federation queries.

    WHY: An LLM could generate SERVICE <http://attacker.com/sparql> { ... }
    causing the DBpedia SPARQL endpoint to make outbound requests to arbitrary
    URLs (SSRF). The current guard blocks INSERT/DELETE/DROP etc but not SERVICE.
    """

    def test_service_federation_rejected(self):
        """SERVICE federation should be blocked — it enables SSRF."""
        query = (
            "SELECT ?x WHERE { "
            "SERVICE <http://attacker.com/sparql> { ?x a dbo:Film } "
            "}"
        )
        # Before fix: this PASSES _is_readonly_sparql (bug!)
        # After fix: this should return False (blocked)
        assert _is_readonly_sparql(query) is False

    def test_service_case_insensitive(self):
        """SERVICE detection must be case-insensitive."""
        query = "SELECT ?x WHERE { service <http://evil.com/sparql> { ?x ?p ?o } }"
        assert _is_readonly_sparql(query) is False

    def test_service_in_uri_not_blocked(self):
        """The word 'service' inside a URI should NOT trigger the guard.

        Common pitfall: 'Service' is a valid word in entity names like
        http://dbpedia.org/resource/National_Park_Service
        The regex uses word boundaries (\b) so this should be safe.
        """
        query = (
            "SELECT ?x WHERE { "
            "<http://dbpedia.org/resource/National_Park_Service> ?p ?x "
            "}"
        )
        assert _is_readonly_sparql(query) is True

    def test_normal_select_still_allowed(self):
        """Regression: normal SELECT queries must still pass."""
        query = "SELECT ?x WHERE { ?x a dbo:Film } LIMIT 100"
        assert _is_readonly_sparql(query) is True

    def test_normal_ask_still_allowed(self):
        """Regression: ASK queries must still pass."""
        query = "ASK { <http://dbpedia.org/resource/Inception> dbo:director ?x }"
        assert _is_readonly_sparql(query) is True


# ---------------------------------------------------------------------------
# L3: Spotlight circuit breaker
# ---------------------------------------------------------------------------


class TestSpotlightCircuitBreaker:
    """L3: Spotlight should skip quickly after repeated failures.

    WHY: When Spotlight is down, every question wastes ~93s (3 retries × 30s
    timeout) on doomed requests before falling back to spaCy. The
    _CircuitBreaker class already exists in graphrag_wiki.py for DuckDuckGo.
    Spotlight should use the same pattern.

    We test the _CircuitBreaker class directly since it's the mechanism.
    """

    def test_circuit_breaker_trips_after_threshold(self):
        """After N failures, circuit should open (skip calls)."""
        from graphrag_wiki import _CircuitBreaker

        cb = _CircuitBreaker(threshold=3, cooldown=60.0)

        # Not open initially
        assert cb.is_open() is False

        # Record failures below threshold
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open() is False

        # Third failure trips it
        cb.record_failure()
        assert cb.is_open() is True

    def test_circuit_breaker_resets_on_success(self):
        """A success should reset the failure counter."""
        from graphrag_wiki import _CircuitBreaker

        cb = _CircuitBreaker(threshold=3, cooldown=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # reset
        cb.record_failure()
        # Only 1 failure after reset, should not be open
        assert cb.is_open() is False

    def test_circuit_breaker_cooldown_expires(self):
        """After cooldown period, circuit should close (allow retries)."""
        from graphrag_wiki import _CircuitBreaker

        cb = _CircuitBreaker(threshold=2, cooldown=0.1)  # 100ms cooldown
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open() is True

        time.sleep(0.15)  # Wait past cooldown
        assert cb.is_open() is False

    def test_spotlight_has_circuit_breaker(self):
        """After fix: entity_linker module should have a circuit breaker.

        This test verifies the fix is applied — the entity linker module
        should have a _spotlight_cb variable that is used to skip Spotlight
        when it's repeatedly down.

        We check the module-level variable (not instance) because the
        circuit breaker must be shared across all linker instances to
        track Spotlight health across the batch run.
        """
        from dbpedia import entity_linker as el_module

        # Before fix: _spotlight_cb won't exist in the module
        # After fix: _spotlight_cb should be a circuit breaker instance
        assert hasattr(el_module, '_spotlight_cb'), (
            "entity_linker module should have a _spotlight_cb circuit breaker"
        )
        cb = el_module._spotlight_cb
        assert hasattr(cb, 'is_open'), "_spotlight_cb must have is_open() method"
        assert hasattr(cb, 'record_failure'), "_spotlight_cb must have record_failure()"
        assert hasattr(cb, 'record_success'), "_spotlight_cb must have record_success()"


# ---------------------------------------------------------------------------
# C3: Pipeline-level timeout
# ---------------------------------------------------------------------------


class TestPipelineTimeout:
    """C3: Batch eval should have a per-question timeout.

    WHY: Without a timeout, a single degraded-service question can block
    the entire 1000-question evaluation run indefinitely. The batch eval
    script at scripts/hybrid_batch_eval.py wraps hybrid_answer() in
    try/except but has no time limit.

    We test the timeout mechanism, not the full pipeline.
    """

    def test_timeout_mechanism_exists(self):
        """After fix: hybrid_batch_eval should define a timeout constant
        or use signal.alarm / threading timeout for per-question capping.

        We verify this by checking the source for a timeout pattern.
        """
        import os

        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scripts", "hybrid_batch_eval.py",
        )
        with open(script_path) as f:
            source = f.read()

        # Look for timeout patterns: SIGALRM, alarm, TimeoutError,
        # threading.Timer, or a QUESTION_TIMEOUT constant
        timeout_patterns = [
            r"QUESTION_TIMEOUT",
            r"signal\.alarm",
            r"SIGALRM",
            r"TimeoutError",
            r"threading\.Timer",
            r"timeout.*=.*\d+",
        ]

        found = any(re.search(p, source) for p in timeout_patterns)
        assert found, (
            "hybrid_batch_eval.py should have a per-question timeout mechanism "
            "(QUESTION_TIMEOUT constant, signal.alarm, or threading.Timer)"
        )
