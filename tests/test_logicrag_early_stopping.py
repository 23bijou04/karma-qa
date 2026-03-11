"""Tests for LogicRAG early stopping and FINAL_ANSWER_PROMPT features.

Tests:
  - _fix_json_response: robust JSON parsing
  - _check_can_answer: early stopping check (mocked LLM)
  - _generate_final_answer: concise answer generation (mocked LLM)
  - Prompt template formatting
  - Early stopping guards: comparison skip, last sub-Q skip
"""

import json
from unittest.mock import MagicMock, patch

from logicrag_decomposition import (
    _check_can_answer,
    _fix_json_response,
    _generate_final_answer,
)
from logicrag_prompts import DEPENDENCY_RAG_PROMPT, FINAL_ANSWER_PROMPT

# ---------------------------------------------------------------------------
# _fix_json_response
# ---------------------------------------------------------------------------


class TestFixJsonResponse:
    def test_valid_json(self):
        result = _fix_json_response('{"can_answer": true, "text": "hello"}')
        assert result == {"can_answer": True, "text": "hello"}

    def test_markdown_code_block(self):
        result = _fix_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_trailing_comma_object(self):
        result = _fix_json_response('{"a": 1, "b": 2,}')
        assert result == {"a": 1, "b": 2}

    def test_trailing_comma_array(self):
        result = _fix_json_response('{"items": [1, 2, 3,]}')
        assert result == {"items": [1, 2, 3]}

    def test_malformed_returns_empty_dict(self):
        result = _fix_json_response("not json at all")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        assert _fix_json_response("") == {}

    def test_none_returns_empty_dict(self):
        assert _fix_json_response(None) == {}

    def test_nested_json(self):
        text = '{"can_answer": false, "data": {"nested": true}}'
        result = _fix_json_response(text)
        assert result["data"]["nested"] is True


# ---------------------------------------------------------------------------
# _check_can_answer
# ---------------------------------------------------------------------------


class TestCheckCanAnswer:
    def test_can_answer_true(self):
        def fake_llm(client, prompt, model):
            return json.dumps({
                "can_answer": True,
                "current_understanding": "We know the director and birth date.",
            })

        with patch("logicrag_decomposition.call_llm", side_effect=fake_llm):
            result = _check_can_answer(
                "When was the director born?",
                "- Who directed X? -> Christopher Nolan",
                "1. Who directed X?\n2. When was director born?",
                "Who directed X?",
                MagicMock(), "model",
            )
        assert result["can_answer"] is True
        assert "director" in result["current_understanding"].lower()

    def test_can_answer_false(self):
        def fake_llm(client, prompt, model):
            return json.dumps({
                "can_answer": False,
                "current_understanding": "We know the director but not birth date.",
            })

        with patch("logicrag_decomposition.call_llm", side_effect=fake_llm):
            result = _check_can_answer(
                "When was the director born?",
                "- Who directed X? -> Nolan",
                "1. Q1\n2. Q2", "Q1",
                MagicMock(), "model",
            )
        assert result["can_answer"] is False

    def test_malformed_response_returns_false(self):
        """Parse failure should conservatively return can_answer=False."""
        def fake_llm(client, prompt, model):
            return "I think yes but this is not JSON"

        with patch("logicrag_decomposition.call_llm", side_effect=fake_llm):
            result = _check_can_answer(
                "Q?", "info", "deps", "current",
                MagicMock(), "model",
            )
        assert result["can_answer"] is False
        assert result["current_understanding"] == ""

    def test_string_false_treated_as_false(self):
        """LLM returning can_answer as string 'false' should be treated as False."""
        def fake_llm(client, prompt, model):
            return json.dumps({
                "can_answer": "false",
                "current_understanding": "not enough",
            })

        with patch("logicrag_decomposition.call_llm", side_effect=fake_llm):
            result = _check_can_answer(
                "Q?", "info", "deps", "current",
                MagicMock(), "model",
            )
        assert result["can_answer"] is False

    def test_llm_exception_returns_false(self):
        """LLM exception should return can_answer=False (conservative)."""
        with patch("logicrag_decomposition.call_llm",
                   side_effect=RuntimeError("API error")):
            result = _check_can_answer(
                "Q?", "info", "deps", "current",
                MagicMock(), "model",
            )
        assert result["can_answer"] is False

    def test_string_true_treated_as_true(self):
        """LLM returning can_answer as string 'True' should be treated as True."""
        def fake_llm(client, prompt, model):
            return json.dumps({
                "can_answer": "True",
                "current_understanding": "sufficient",
            })

        with patch("logicrag_decomposition.call_llm", side_effect=fake_llm):
            result = _check_can_answer(
                "Q?", "info", "deps", "current",
                MagicMock(), "model",
            )
        assert result["can_answer"] is True


# ---------------------------------------------------------------------------
# _generate_final_answer
# ---------------------------------------------------------------------------


class TestGenerateFinalAnswer:
    def test_extracts_final_answer(self):
        def fake_llm(client, prompt, model):
            return "FINAL ANSWER: Christopher Nolan"

        with patch("logicrag_decomposition.call_llm", side_effect=fake_llm):
            result = _generate_final_answer(
                "Who directed Inception?",
                "- director -> Christopher Nolan",
                MagicMock(), "model",
            )
        assert result == "Christopher Nolan"

    def test_concise_date_answer(self):
        def fake_llm(client, prompt, model):
            return "FINAL ANSWER: 1970-07-30"

        with patch("logicrag_decomposition.call_llm", side_effect=fake_llm):
            result = _generate_final_answer(
                "When was born?", "info", MagicMock(), "model",
            )
        assert result == "1970-07-30"

    def test_none_response_returns_none(self):
        def fake_llm(client, prompt, model):
            return None

        with patch("logicrag_decomposition.call_llm", side_effect=fake_llm):
            result = _generate_final_answer(
                "Q?", "info", MagicMock(), "model",
            )
        assert result is None

    def test_llm_exception_returns_none(self):
        """LLM exception should return None (graceful degradation)."""
        with patch("logicrag_decomposition.call_llm",
                   side_effect=RuntimeError("API error")):
            result = _generate_final_answer(
                "Q?", "info", MagicMock(), "model",
            )
        assert result is None


# ---------------------------------------------------------------------------
# Prompt template formatting
# ---------------------------------------------------------------------------


class TestPromptFormatting:
    def test_dependency_rag_prompt_renders(self):
        result = DEPENDENCY_RAG_PROMPT.format(
            question="Who directed X?",
            info_summary="- Q1 -> Nolan",
            dependencies="1. Q1\n2. Q2",
            current_dependency="Q1",
        )
        assert "Who directed X?" in result
        assert "Q1 -> Nolan" in result
        assert "can_answer" in result

    def test_final_answer_prompt_renders(self):
        result = FINAL_ANSWER_PROMPT.format(
            question="When was Nolan born?",
            info_summary="- director -> Nolan\n- birth -> 1970",
        )
        assert "When was Nolan born?" in result
        assert "FINAL ANSWER:" in result
        assert "concise" in result.lower()


# ---------------------------------------------------------------------------
# Early stopping guards (orchestration logic)
# ---------------------------------------------------------------------------


class TestEarlyStoppingGuards:
    """Verify guards prevent premature early stopping."""

    def _run_decomposition(self, is_comparison: bool, num_sub_qs: int = 3):
        """Run _try_logicrag_decomposition with controlled sub-Qs."""
        from logicrag_decomposition import _try_logicrag_decomposition

        early_stop_checks = []

        def fake_call_llm(client, prompt, model):
            # Check decompose first — prompt starts with "Decompose this"
            if prompt.startswith("Decompose this"):
                sub_qs = [f"Sub-Q {i}" for i in range(num_sub_qs)]
                return json.dumps({
                    "sub_questions": sub_qs,
                    "dependencies": [],
                })
            if "We pre-parsed" in prompt:
                early_stop_checks.append(True)
                return json.dumps({
                    "can_answer": False,
                    "current_understanding": "need more",
                })
            if "concise" in prompt.lower():
                return "FINAL ANSWER: test answer"
            return "FINAL ANSWER: sub-answer"

        mock_entity = MagicMock()
        mock_entity.uri = "http://dbpedia.org/resource/X"
        mock_entity.label = "X"
        mock_entity.__eq__ = lambda self, other: False
        mock_entity.__hash__ = lambda self: id(self)

        mock_profile = MagicMock()
        mock_profile.actual_property_set = frozenset()
        mock_projection = MagicMock()
        mock_projection.render_for_sparql.return_value = "mock"
        mock_projection.entity_profile = mock_profile
        mock_projection.completeness = "complete"

        with patch("logicrag_decomposition.call_llm", side_effect=fake_call_llm), \
             patch("sparql_solver.solve",
                   return_value=("answer", "SELECT", [])), \
             patch("logicrag_decomposition.build_entity_profile",
                   return_value=mock_profile), \
             patch("logicrag_decomposition.qa_analyze", return_value=MagicMock(
                 target_properties=frozenset(), is_comparison=False,
                 bridge_keyword=None, bridge_property=None,
                 hop2_keyword=None, hop2_properties=None,
                 target_keywords=[], target_keywords_set=frozenset(),
                 hop_structure="direct",
             )), \
             patch("logicrag_decomposition.sp_project",
                   return_value=mock_projection), \
             patch("logicrag_decomposition._rule_based_route",
                   return_value=("SPARQL", "test")), \
             patch("logicrag_decomposition._is_comparison_question",
                   return_value=is_comparison):
            result = _try_logicrag_decomposition(
                "Test question?",
                [mock_entity], "schema", MagicMock(), "model",
                MagicMock(), MagicMock(), MagicMock(),
            )
        return result, early_stop_checks

    def test_comparison_skips_early_stopping(self):
        """Comparison questions should never trigger early stopping checks."""
        _, checks = self._run_decomposition(is_comparison=True, num_sub_qs=3)
        assert len(checks) == 0

    def test_non_comparison_checks_early_stopping(self):
        """Non-comparison with 3 sub-Qs should check early stopping twice (skip last)."""
        _, checks = self._run_decomposition(is_comparison=False, num_sub_qs=3)
        # 3 sub-Qs: check after sub-Q 0 and 1, skip sub-Q 2 (last)
        assert len(checks) == 2

    def test_two_subq_checks_once(self):
        """2 sub-Qs: check after first only (last is skipped)."""
        _, checks = self._run_decomposition(is_comparison=False, num_sub_qs=2)
        assert len(checks) == 1

    def test_early_exit_when_can_answer_true(self):
        """When can_answer=True, should exit early and return answer."""
        from logicrag_decomposition import _try_logicrag_decomposition

        resolve_count = {"n": 0}

        def fake_call_llm(client, prompt, model):
            if prompt.startswith("Decompose this"):
                return json.dumps({
                    "sub_questions": ["Q1?", "Q2?", "Q3?"],
                    "dependencies": [],
                })
            if "We pre-parsed" in prompt:
                return json.dumps({
                    "can_answer": True,
                    "current_understanding": "sufficient",
                })
            if "concise" in prompt.lower():
                return "FINAL ANSWER: early answer"
            resolve_count["n"] += 1
            return "FINAL ANSWER: sub-answer"

        mock_entity = MagicMock()
        mock_entity.uri = "http://dbpedia.org/resource/X"
        mock_entity.label = "X"
        mock_entity.__eq__ = lambda self, other: False
        mock_entity.__hash__ = lambda self: id(self)

        mock_profile = MagicMock()
        mock_profile.actual_property_set = frozenset()
        mock_projection = MagicMock()
        mock_projection.render_for_sparql.return_value = "mock"
        mock_projection.entity_profile = mock_profile
        mock_projection.completeness = "complete"

        with patch("logicrag_decomposition.call_llm", side_effect=fake_call_llm), \
             patch("sparql_solver.solve",
                   return_value=("answer", "SELECT", [])), \
             patch("logicrag_decomposition.build_entity_profile",
                   return_value=mock_profile), \
             patch("logicrag_decomposition.qa_analyze", return_value=MagicMock(
                 target_properties=frozenset(), is_comparison=False,
                 bridge_keyword=None, bridge_property=None,
                 hop2_keyword=None, hop2_properties=None,
                 target_keywords=[], target_keywords_set=frozenset(),
                 hop_structure="direct",
             )), \
             patch("logicrag_decomposition.sp_project",
                   return_value=mock_projection), \
             patch("logicrag_decomposition._rule_based_route",
                   return_value=("SPARQL", "test")), \
             patch("logicrag_decomposition._is_comparison_question",
                   return_value=False):
            result = _try_logicrag_decomposition(
                "Test?", [mock_entity], "schema", MagicMock(), "model",
                MagicMock(), MagicMock(), MagicMock(),
            )

        assert result == "early answer"
