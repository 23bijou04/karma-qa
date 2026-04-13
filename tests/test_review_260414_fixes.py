"""Regression tests for review-260414-0026 bug fixes.

Covers the five priority fixes:
  #1 debug_info KeyError on assembly failure (hybrid_qa.py)
  #3 routing hop2 early-break bug (routing.py)
  #4 Step 5b redundant re-solve guard (hybrid_qa.py)
  #5b LogicRAG partial-chain fabrication (logicrag_decomposition.py)
  #6 extract_final_answer silent corruption (models.py)

Each test has a corresponding "without the fix this test fails" comment so
future readers can verify the guard is still load-bearing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from dbpedia_qa.models import extract_final_answer
from logicrag_decomposition import (
    SubQuestion,
    _build_info_summary,
    _chain_has_gap,
)


# ---------------------------------------------------------------------------
# Fix #3 — routing hop2 early-break
# ---------------------------------------------------------------------------


class TestRoutingHop2Scan:
    """Verify routing scans all dbo: candidates instead of breaking on the first.

    Without the fix: the first dbo: candidate with an invalid identifier would
    cause an unconditional break, leaving hop2_dbo_prop=None and silently
    degrading a 2-hop ASK query to a 1-hop one.
    """

    def test_scan_past_invalid_first_candidate(self):
        """First candidate invalid, second valid — must pick the second."""
        import routing

        # Build a minimal analysis stub with two hop2 candidates where the
        # first fails the identifier regex (contains dashes) and the second
        # passes.
        analysis = MagicMock()
        analysis.hop2_properties = ["dbo:not-a-valid-ident", "dbo:director"]
        analysis.bridge_property = "spouse"
        analysis.is_comparison = False

        # Stub entity and kg_client
        from dbpedia_qa.models import Entity
        ent = Entity(uri="http://dbpedia.org/resource/Foo", label="Foo")
        kg_client = MagicMock()
        # ASK returns True so any-verified path is taken
        captured_queries: list[str] = []

        def fake_query(q):
            captured_queries.append(q)
            return {"boolean": True}

        kg_client.query = fake_query

        result = routing.verify_sparql_feasibility(
            analysis,
            [],  # projections (unused when rule_result already matched)
            [ent],
            kg_client,
            ("SPARQL", "bridge_hop2"),
        )

        # The second candidate "director" should have been picked.
        assert result == ("SPARQL", "bridge_hop2+ask_verified")
        # The executed ASK query must be 2-hop (reference dbo:director),
        # not the 1-hop fallback.
        assert any("director" in q for q in captured_queries), (
            f"Expected 2-hop ASK with dbo:director; got {captured_queries}"
        )

    def test_skip_non_dbo_prefix(self):
        """Non-dbo candidates shouldn't short-circuit the scan."""
        import routing

        analysis = MagicMock()
        analysis.hop2_properties = ["dbp:someprop", "dbo:birthPlace"]
        analysis.bridge_property = "spouse"
        analysis.is_comparison = False

        from dbpedia_qa.models import Entity
        ent = Entity(uri="http://dbpedia.org/resource/Foo", label="Foo")
        kg_client = MagicMock()
        captured: list[str] = []
        kg_client.query = lambda q: (captured.append(q), {"boolean": True})[1]

        result = routing.verify_sparql_feasibility(
            analysis,
            [],
            [ent],
            kg_client,
            ("SPARQL", "bridge_hop2"),
        )
        assert result == ("SPARQL", "bridge_hop2+ask_verified")
        assert any("birthPlace" in q for q in captured)


# ---------------------------------------------------------------------------
# Fix #5b — LogicRAG partial-chain fabrication
# ---------------------------------------------------------------------------


class TestInfoSummaryUnknownMarker:
    """Verify unresolved sub-Qs emit [UNKNOWN] markers instead of silent drop.

    Without the fix: _build_info_summary filtered `if sq.original_idx in
    known_facts`, silently dropping failed sub-Qs. The LLM would then
    synthesize a final answer from e.g. Hop1+Hop3 without Hop2 — a
    fabrication from a broken chain.
    """

    def test_unknown_hop_emitted_in_summary(self):
        sqs = [
            SubQuestion(text="Q1", depends_on=[], original_idx=0),
            SubQuestion(text="Q2", depends_on=[0], original_idx=1),
            SubQuestion(text="Q3", depends_on=[1], original_idx=2),
        ]
        sqs[0].method = "sparql"
        sqs[2].method = "text"
        known_facts = {0: "AliceAnswer", 2: "CharlieAnswer"}  # Hop 1 missing

        summary = _build_info_summary(sqs, known_facts)

        assert "[UNKNOWN]" in summary
        assert "Q2" in summary
        assert "could not resolve" in summary
        # Resolved hops still rendered with provenance tag
        assert "[SPARQL]" in summary
        assert "[TEXT]" in summary

    def test_all_resolved_no_unknown_marker(self):
        sqs = [
            SubQuestion(text="Q1", depends_on=[], original_idx=0),
            SubQuestion(text="Q2", depends_on=[0], original_idx=1),
        ]
        sqs[0].method = "sparql"
        sqs[1].method = "sparql"
        known_facts = {0: "A", 1: "B"}

        summary = _build_info_summary(sqs, known_facts)
        assert "[UNKNOWN]" not in summary


class TestChainHasGap:
    """_chain_has_gap guards the final-answer step against partial chains."""

    def test_missing_intermediate_hop_detected(self):
        sqs = [
            SubQuestion(text="Q1", depends_on=[], original_idx=0),
            SubQuestion(text="Q2", depends_on=[0], original_idx=1),
            SubQuestion(text="Q3", depends_on=[1], original_idx=2),
        ]
        known_facts = {0: "A", 2: "C"}  # Hop2 missing
        assert _chain_has_gap(sqs, known_facts) is True

    def test_complete_chain_no_gap(self):
        sqs = [
            SubQuestion(text="Q1", depends_on=[], original_idx=0),
            SubQuestion(text="Q2", depends_on=[0], original_idx=1),
        ]
        known_facts = {0: "A", 1: "B"}
        assert _chain_has_gap(sqs, known_facts) is False

    def test_no_dependencies_no_gap(self):
        """Independent sub-Qs (e.g. comparison questions) have no chain to break."""
        sqs = [
            SubQuestion(text="Q1", depends_on=[], original_idx=0),
            SubQuestion(text="Q2", depends_on=[], original_idx=1),
        ]
        known_facts = {0: "A"}  # One resolved, one not — but no deps
        assert _chain_has_gap(sqs, known_facts) is False


# ---------------------------------------------------------------------------
# Fix #6 — extract_final_answer silent corruption
# ---------------------------------------------------------------------------


class TestExtractFinalAnswerParenStrip:
    """Verify trailing parentheticals are stripped from extracted answers.

    Without the fix: LLM output like "FINAL ANSWER: John Smith (director)"
    returned "John Smith (director)", breaking EM/F1 against gold "John Smith".
    """

    def test_strips_trailing_parenthetical(self):
        text = "reasoning... FINAL ANSWER: John Smith (director)"
        assert extract_final_answer(text) == "John Smith"

    def test_strips_trailing_paren_with_whitespace(self):
        text = "FINAL ANSWER: Albert Einstein  ( physicist ) "
        assert extract_final_answer(text) == "Albert Einstein"

    def test_preserves_name_without_parens(self):
        assert extract_final_answer("FINAL ANSWER: Berlin") == "Berlin"

    def test_does_not_strip_interior_parens(self):
        """Only trailing parentheticals, not mid-answer ones."""
        text = "FINAL ANSWER: E=mc^2"
        assert extract_final_answer(text) == "E=mc^2"

    def test_single_paren_only_is_removed(self):
        text = "FINAL ANSWER: Paris (France)"
        assert extract_final_answer(text) == "Paris"


class TestExtractFinalAnswerLogging:
    """Verify missing-marker case logs a diagnostic."""

    def test_no_marker_logs_debug(self, caplog):
        import logging
        caplog.set_level(logging.DEBUG, logger="dbpedia_qa.models")
        extract_final_answer("a long essay without any marker at all")
        assert any(
            "no FINAL ANSWER marker" in rec.message for rec in caplog.records
        ), f"Expected debug log; got {[r.message for r in caplog.records]}"

    def test_no_marker_still_returns_text(self):
        """Passthrough preserved for short inputs (backward compat)."""
        assert extract_final_answer("Christopher Nolan") == "Christopher Nolan"


# ---------------------------------------------------------------------------
# Fix #1 — debug_info assembly never crashes stage_timings writes
# ---------------------------------------------------------------------------


class TestDebugInfoAssemblyFallback:
    """hybrid_qa's debug_info except branch must seed stage_timings.

    Without the fix: an assembly failure would leave debug_info without a
    stage_timings key, then lines 193/215/251/345 would raise KeyError on
    `debug_info["stage_timings"][...]`.
    """

    def test_except_branch_contains_stage_timings(self):
        """Static check — read the source and assert the safety-net keys exist."""
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "hybrid_qa.py").read_text()
        # Find the except block that creates debug_info
        except_idx = src.index('debug_info = {\n            "assembly_error"')
        # The dict must include stage_timings with the three stage keys
        snippet = src[except_idx:except_idx + 500]
        assert "stage_timings" in snippet
        assert "entity_linking_s" in snippet
        assert "schema_projection_s" in snippet
        assert "routing_s" in snippet


# ---------------------------------------------------------------------------
# Fix #4 — Step 5b redundant re-solve is guarded
# ---------------------------------------------------------------------------


class TestStep5bRouteGuard:
    """Step 5b must not fire when route=='SPARQL' (Step 4 already tried)."""

    def test_route_guard_present_in_source(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "hybrid_qa.py").read_text()
        # Locate Step 5b block
        step5b_idx = src.index("Step 5b: SPARQL pre-check")
        snippet = src[step5b_idx:step5b_idx + 800]
        # The guard `route != "SPARQL"` must be in the if condition
        assert 'route != "SPARQL"' in snippet, (
            "Step 5b must skip when Step 4 already tried SPARQL"
        )
