"""Tests for LogicRAG sub-question decomposition and answer chaining."""

import json
import os

import pytest

from dbpedia_qa.models import Entity
from logicrag_decomposition import (
    SubQuestion,
    _substitute_answers,
    _topological_sort_indices,
)

# ---------------------------------------------------------------------------
# _topological_sort_indices
# ---------------------------------------------------------------------------


class TestTopologicalSortIndices:
    def test_linear_chain(self):
        """A -> B -> C should produce [A, B, C]."""
        order = _topological_sort_indices(3, [[1, 0], [2, 1]])
        assert order == [0, 1, 2]

    def test_diamond_dependency(self):
        """A -> B, A -> C, B -> D, C -> D should produce valid order."""
        order = _topological_sort_indices(4, [[1, 0], [2, 0], [3, 1], [3, 2]])
        assert order[0] == 0  # A must be first
        assert order[-1] == 3  # D must be last
        assert order.index(1) < order.index(3)
        assert order.index(2) < order.index(3)

    def test_parallel_independent(self):
        """No edges -> all nodes present (order is valid but arbitrary)."""
        order = _topological_sort_indices(3, [])
        assert set(order) == {0, 1, 2}

    def test_cycle_fallback(self):
        """Cycle should fall back to sequential 0..n-1."""
        order = _topological_sort_indices(2, [[0, 1], [1, 0]])
        assert order == [0, 1]  # Kahn's detects cycle, falls back to sequential

    def test_single_node(self):
        order = _topological_sort_indices(1, [])
        assert order == [0]

    def test_out_of_range_edges_ignored(self):
        """Edges referencing non-existent nodes should be ignored."""
        order = _topological_sort_indices(2, [[5, 0], [1, 0]])
        assert 0 in order and 1 in order
        assert len(order) == 2

    def test_empty(self):
        order = _topological_sort_indices(0, [])
        assert order == []


# ---------------------------------------------------------------------------
# _substitute_answers
# ---------------------------------------------------------------------------


class TestSubstituteAnswers:
    def test_single_placeholder(self):
        result = _substitute_answers(
            "When was [director] born?",
            {0: "Christopher Nolan"},
        )
        assert "Christopher Nolan" in result
        assert "[director]" not in result

    def test_multiple_placeholders(self):
        result = _substitute_answers(
            "Did [person A] and [person B] work together?",
            {0: "Alice", 1: "Bob"},
        )
        assert "Alice" in result
        assert "Bob" in result

    def test_no_placeholders(self):
        """Text without brackets should pass through unchanged."""
        text = "Who directed Inception?"
        result = _substitute_answers(text, {0: "Christopher Nolan"})
        assert "directed" in result

    def test_empty_known_facts(self):
        text = "When was [director] born?"
        result = _substitute_answers(text, {})
        assert result == text


# ---------------------------------------------------------------------------
# SubQuestion dataclass
# ---------------------------------------------------------------------------


class TestSubQuestion:
    def test_defaults(self):
        sq = SubQuestion(text="Who directed Inception?")
        assert sq.text == "Who directed Inception?"
        assert sq.depends_on == []
        assert sq.answer is None
        assert sq.method is None
        assert sq.entities is None

    def test_with_dependencies(self):
        sq = SubQuestion(
            text="When was [director] born?",
            depends_on=[0],
            original_idx=1,
        )
        assert sq.depends_on == [0]
        assert sq.original_idx == 1


# ---------------------------------------------------------------------------
# JSON parsing logic (extracted from _decompose_subquestions)
# ---------------------------------------------------------------------------


class TestDecomposeJsonParsing:
    """Test the JSON parsing logic used inside _decompose_subquestions.
    Does NOT call LLM -- tests the parsing path only."""

    @staticmethod
    def _parse_decomposition_response(resp: str):
        """Extract the JSON parsing logic from _decompose_subquestions."""
        clean = resp.strip()
        clean = clean.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(clean)
        sub_qs = data.get("sub_questions", [])
        deps = data.get("dependencies", [])
        return sub_qs, deps

    def test_valid_json(self):
        resp = '{"sub_questions": ["Q1", "Q2"], "dependencies": [[1, 0]]}'
        sub_qs, deps = self._parse_decomposition_response(resp)
        assert sub_qs == ["Q1", "Q2"]
        assert deps == [[1, 0]]

    def test_json_with_code_fences(self):
        resp = '```json\n{"sub_questions": ["Q1", "Q2"], "dependencies": []}\n```'
        sub_qs, deps = self._parse_decomposition_response(resp)
        assert len(sub_qs) == 2

    def test_single_subquestion_returns_empty(self):
        resp = '{"sub_questions": ["Q1"], "dependencies": []}'
        sub_qs, _ = self._parse_decomposition_response(resp)
        assert len(sub_qs) < 2  # _decompose_subquestions returns None for < 2

    def test_malformed_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            self._parse_decomposition_response("not json at all")

    def test_missing_keys(self):
        resp = '{"other_field": true}'
        sub_qs, deps = self._parse_decomposition_response(resp)
        assert sub_qs == []
        assert deps == []


# ---------------------------------------------------------------------------
# Comparison decomposition
# ---------------------------------------------------------------------------


class TestComparisonDecomposition:
    """Test that comparison questions decompose into parallel sub-Qs."""

    def test_comparison_produces_parallel_subqs(self):
        """Comparison JSON should have independent entity queries."""
        sub_qs = [
            "When was Albert Einstein born?",
            "When was Isaac Newton born?",
        ]
        deps = []  # No dependencies (parallel)
        order = _topological_sort_indices(len(sub_qs), deps)
        assert set(order) == {0, 1}

    def test_bridge_comparison_has_dependencies(self):
        """Bridge-comparison: director sub-Qs depend on director lookup."""
        # "Who was born first, director of X or director of Y?"
        sub_qs = [
            "Who directed X?",
            "Who directed Y?",
            "When was [director of X] born?",
            "When was [director of Y] born?",
        ]
        deps = [[2, 0], [3, 1]]
        order = _topological_sort_indices(len(sub_qs), deps)
        assert order.index(0) < order.index(2)
        assert order.index(1) < order.index(3)


# ---------------------------------------------------------------------------
# Integration tests (require GROQ_API_KEY)
# ---------------------------------------------------------------------------

HAS_API_KEY = bool(os.environ.get("GROQ_API_KEY"))


@pytest.mark.skipif(not HAS_API_KEY, reason="GROQ_API_KEY not set")
class TestLogicragIntegration:
    """Live integration tests against Groq API + DBpedia."""

    def test_bridge_question_decomposes(self):
        """Multi-hop bridge question should decompose into 2+ sub-Qs."""
        from groq import Groq

        from logicrag_decomposition import _decompose_subquestions

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        entities = [Entity(
            uri="http://dbpedia.org/resource/Inception",
            label="Inception", source="test",
        )]
        schema = "Entity: Inception\nClass: dbo:Film\n  dbo: director, releaseDate"

        result = _decompose_subquestions(
            "When was the director of Inception born?",
            entities, schema, client, "llama-3.3-70b-versatile",
        )
        assert result is not None
        assert len(result) >= 2

    def test_single_hop_returns_none(self):
        """Single-hop question should NOT decompose."""
        from groq import Groq

        from logicrag_decomposition import _decompose_subquestions

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        entities = [Entity(
            uri="http://dbpedia.org/resource/Inception",
            label="Inception", source="test",
        )]
        schema = "Entity: Inception\nClass: dbo:Film\n  dbo: director"

        result = _decompose_subquestions(
            "Who directed Inception?",
            entities, schema, client, "llama-3.3-70b-versatile",
        )
        assert result is None

    def test_full_logicrag_bridge(self):
        """Full end-to-end LogicRAG on a bridge question."""
        from hybrid_qa import hybrid_answer

        result = hybrid_answer(
            "When was the director of Inception born?",
            api_key=os.environ["GROQ_API_KEY"],
            verbose=False,
        )
        assert result.answer
        assert result.method in ("sparql", "logicrag_decomposed", "text")

    def test_comparison_question_decomposes(self):
        """Comparison question should decompose into parallel sub-Qs."""
        from groq import Groq

        from logicrag_decomposition import _decompose_subquestions

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        entities = [
            Entity(uri="http://dbpedia.org/resource/Albert_Einstein",
                   label="Albert Einstein", source="test"),
            Entity(uri="http://dbpedia.org/resource/Isaac_Newton",
                   label="Isaac Newton", source="test"),
        ]
        schema = ("Entity: Albert Einstein\nClass: dbo:Person\n"
                  "  dbo: birthDate, deathDate, birthPlace\n\n"
                  "Entity: Isaac Newton\nClass: dbo:Person\n"
                  "  dbo: birthDate, deathDate, birthPlace")

        result = _decompose_subquestions(
            "Who was born first, Albert Einstein or Isaac Newton?",
            entities, schema, client, "llama-3.3-70b-versatile",
            is_comparison=True,
        )
        assert result is not None
        assert len(result) >= 2
