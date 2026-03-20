"""Tests for LogicRAG sub-question decomposition and answer chaining."""

import json
import os

import pytest

from dbpedia_qa.models import Entity
from logicrag_decomposition import (
    SubQuestion,
    _make_fallback_entity,
    _resolve_answer_entities,
    _split_compound_answer,
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

    # -- C2 regression: out-of-order brackets must not swap answers --

    def test_indexed_placeholders_order_safe(self):
        """[Q0] and [Q1] are matched by index, not position (C2 fix)."""
        result = _substitute_answers(
            "Compare [Q1] birth year with [Q0] birth year",
            {0: "Einstein", 1: "Newton"},
        )
        # Q1 should be Newton, Q0 should be Einstein regardless of position
        assert "Newton" in result
        assert "Einstein" in result
        parts = result.split("birth year")
        # "Compare Newton ... with Einstein ..."
        assert "Newton" in parts[0]
        assert "Einstein" in parts[1]

    def test_indexed_beats_positional(self):
        """Indexed [Q{idx}] takes priority over positional [bracket] matching."""
        result = _substitute_answers(
            "Was [Q1] born before [Q0]?",
            {0: "Alice", 1: "Bob"},
        )
        assert result == "Was Bob born before Alice?"

    # -- C3 regression: backslash/backreference in answers --

    def test_answer_with_backslash(self):
        r"""Answer containing backslashes must not be interpreted (C3 fix)."""
        result = _substitute_answers(
            "Who is [person]?",
            {0: r"AC\DC"},
        )
        assert r"AC\DC" in result

    def test_answer_with_backreference_pattern(self):
        r"""Answer containing \1 or \n must be treated as literal text."""
        result = _substitute_answers(
            "What is [thing]?",
            {0: r"item\1 and line\n"},
        )
        assert r"item\1" in result
        assert r"line\n" in result


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
# Bridge entity resolution (helpers — no network)
# ---------------------------------------------------------------------------


class TestSplitCompoundAnswer:
    def test_two_names(self):
        assert _split_compound_answer("Alice and Bob") == ["Alice", "Bob"]

    def test_three_names(self):
        parts = _split_compound_answer("A and B and C")
        assert len(parts) == 3

    def test_no_and(self):
        assert _split_compound_answer("Christopher Nolan") == ["Christopher Nolan"]

    def test_empty(self):
        assert _split_compound_answer("") == []


class TestMakeFallbackEntity:
    def test_creates_uri(self):
        e = _make_fallback_entity("Christopher Nolan")
        assert e.uri == "http://dbpedia.org/resource/Christopher_Nolan"
        assert e.source == "logicrag_guessed"
        assert e.label == "Christopher Nolan"


# ---------------------------------------------------------------------------
# Bridge entity resolution (live DBpedia SPARQL)
# ---------------------------------------------------------------------------

HAS_NETWORK = True
try:
    from dbpedia_qa.dbpedia.client import DBpediaSPARQLClient
    _test_client = DBpediaSPARQLClient()
    _test_client.search_entity_by_name("Christopher Nolan")
except Exception:
    HAS_NETWORK = False


@pytest.mark.skipif(not HAS_NETWORK, reason="DBpedia SPARQL unreachable")
class TestBridgeEntityResolution:
    """Live tests for _resolve_answer_entities against DBpedia SPARQL."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from dbpedia_qa.dbpedia.client import DBpediaSPARQLClient
        self.kg_client = DBpediaSPARQLClient()

    def test_single_entity(self):
        """'Christopher Nolan' should resolve to a single validated entity."""
        entities = _resolve_answer_entities("Christopher Nolan", self.kg_client)
        assert len(entities) == 1
        assert "Christopher_Nolan" in entities[0].uri
        assert entities[0].source != "logicrag_guessed"

    def test_compound_answer_splits_and_resolves(self):
        """'Fridrikh Ermler and Eduard Ioganson' should split and resolve at least one."""
        entities = _resolve_answer_entities(
            "Fridrikh Ermler and Eduard Ioganson", self.kg_client,
        )
        assert len(entities) >= 1
        uris = [e.uri for e in entities]
        assert any("Ermler" in u for u in uris)

    def test_numeric_returns_empty(self):
        """Dates and numbers should return empty list (no SPARQL call)."""
        assert _resolve_answer_entities("1926-05-13", self.kg_client) == []
        assert _resolve_answer_entities("42", self.kg_client) == []

    def test_unsplit_entity_with_and(self):
        """'Simon and Garfunkel' should resolve as single entity (try-unsplit-first)."""
        entities = _resolve_answer_entities("Simon and Garfunkel", self.kg_client)
        assert len(entities) == 1
        assert "Simon_and_Garfunkel" in entities[0].uri

    def test_nonexistent_returns_empty(self):
        """Completely unknown entity should return empty (URI validated via ASK)."""
        entities = _resolve_answer_entities(
            "Xyzzy Nonexistent Entity 12345", self.kg_client,
        )
        assert len(entities) == 0

    def test_trailing_period_stripped(self):
        """Answer with trailing period should still resolve."""
        entities = _resolve_answer_entities("Christopher Nolan.", self.kg_client)
        assert len(entities) == 1
        assert "Christopher_Nolan" in entities[0].uri


# ---------------------------------------------------------------------------
# Phase 1: Composition synthesis vs bridge single-fact
# ---------------------------------------------------------------------------


class TestLogicragSynthesis:
    """Verify final answer generation: composition uses FINAL_ANSWER_PROMPT."""

    def test_composition_uses_final_answer_prompt(self):
        """When 2+ sub-Qs resolve, FINAL_ANSWER_PROMPT should be called."""
        from unittest.mock import MagicMock, patch

        from logicrag_decomposition import _try_logicrag_decomposition

        llm_calls = []

        def fake_call_llm(client, prompt, model):
            llm_calls.append(prompt[:300])
            if prompt.startswith("Decompose this"):
                return json.dumps({
                    "sub_questions": [
                        "Who directed Film X?",
                        "Who directed Film Y?",
                    ],
                    "dependencies": [],
                })
            if "We pre-parsed" in prompt:
                return json.dumps({
                    "can_answer": False,
                    "current_understanding": "need more info",
                })
            if "concise" in prompt.lower():
                return "FINAL ANSWER: Alice directed X and Bob directed Y"
            return "FINAL ANSWER: test"

        mock_entity = Entity(
            uri="http://dbpedia.org/resource/X", label="X", source="test",
        )

        mock_profile = MagicMock()
        mock_profile.actual_property_set = frozenset()
        mock_projection = MagicMock()
        mock_projection.render_for_sparql.return_value = "mock schema"
        mock_projection.entity_profile = mock_profile
        mock_projection.completeness = "complete"

        with patch("logicrag_decomposition.call_llm", side_effect=fake_call_llm), \
             patch("sparql_solver.solve",
                   return_value=("Alice", "SELECT ...", [])), \
             patch("logicrag_decomposition.build_entity_profile", return_value=mock_profile), \
             patch("logicrag_decomposition.qa_analyze", return_value=MagicMock(
                 target_properties=frozenset(), is_comparison=False, bridge_keyword=None,
                 bridge_property=None, hop2_keyword=None, hop2_properties=None,
                 target_keywords=[], target_keywords_set=frozenset(),
                 hop_structure="direct",
             )), \
             patch("logicrag_decomposition.sp_project", return_value=mock_projection), \
             patch("logicrag_decomposition._rule_based_route",
                   return_value=("SPARQL", "test")), \
             patch("logicrag_decomposition._is_comparison_question", return_value=False):
            result = _try_logicrag_decomposition(
                "Who directed Film X and who directed Film Y?",
                [mock_entity], "schema", MagicMock(), "model",
                MagicMock(), MagicMock(), MagicMock(),
            )

        assert result is not None
        # FINAL_ANSWER_PROMPT should have been called (contains "concise")
        assert any("concise" in c.lower() for c in llm_calls)

    def test_bridge_single_fact_returns_directly(self):
        """When only 1 sub-Q resolves, return directly without synthesis."""
        from unittest.mock import MagicMock, patch

        from logicrag_decomposition import _try_logicrag_decomposition

        llm_calls = []

        def fake_call_llm(client, prompt, model):
            llm_calls.append(prompt[:200])
            if "Decompose" in prompt:
                return json.dumps({
                    "sub_questions": [
                        "Who directed Inception?",
                        "When was [director] born?",
                    ],
                    "dependencies": [[1, 0]],
                })
            return "FINAL ANSWER: test"

        mock_entity = Entity(
            uri="http://dbpedia.org/resource/Inception",
            label="Inception", source="test",
        )

        solve_call_count = {"n": 0}

        def fake_solve(question, schema, entities, client, model, **kwargs):
            solve_call_count["n"] += 1
            if solve_call_count["n"] == 1:
                return "Christopher Nolan", "SELECT ...", []
            return None, "", []

        mock_profile = MagicMock()
        mock_profile.actual_property_set = frozenset()
        mock_projection = MagicMock()
        mock_projection.render_for_sparql.return_value = "mock schema"
        mock_projection.entity_profile = mock_profile
        mock_projection.completeness = "complete"

        with patch("logicrag_decomposition.call_llm", side_effect=fake_call_llm), \
             patch("sparql_solver.solve", side_effect=fake_solve), \
             patch("text_solver.solve", return_value=None), \
             patch("logicrag_decomposition.build_entity_profile", return_value=mock_profile), \
             patch("logicrag_decomposition.qa_analyze", return_value=MagicMock(
                 target_properties=frozenset(), is_comparison=False, bridge_keyword=None,
                 bridge_property=None, hop2_keyword=None, hop2_properties=None,
                 target_keywords=[], target_keywords_set=frozenset(),
                 hop_structure="direct",
             )), \
             patch("logicrag_decomposition.sp_project", return_value=mock_projection), \
             patch("logicrag_decomposition._rule_based_route",
                   return_value=("SPARQL", "test")), \
             patch("logicrag_decomposition._is_comparison_question", return_value=False):
            result = _try_logicrag_decomposition(
                "When was the director of Inception born?",
                [mock_entity], "schema", MagicMock(), "model",
                MagicMock(), MagicMock(), MagicMock(),
            )

        assert result is not None
        # Single resolved fact routes through LLM synthesis to handle
        # intermediate bridge answers (e.g., entity name vs. final answer).
        # The mock LLM returns "test" via extract_final_answer("FINAL ANSWER: test").
        assert result == "test"


# ---------------------------------------------------------------------------
# Integration tests (require GROQ_API_KEY)
# ---------------------------------------------------------------------------

HAS_API_KEY = bool(os.environ.get("GROQ_API_KEY"))


@pytest.mark.skipif(not HAS_API_KEY, reason="GROQ_API_KEY not set")
class TestLogicragIntegration:
    """Live integration tests against LLM API + DBpedia."""

    def test_bridge_question_decomposes(self):
        """Multi-hop bridge question should decompose into 2+ sub-Qs."""
        from llm_provider import create_llm_client
        from logicrag_decomposition import _decompose_subquestions

        client = create_llm_client("groq", api_key=os.environ["GROQ_API_KEY"])
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
        from llm_provider import create_llm_client
        from logicrag_decomposition import _decompose_subquestions

        client = create_llm_client("groq", api_key=os.environ["GROQ_API_KEY"])
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
        from llm_provider import create_llm_client
        from logicrag_decomposition import _decompose_subquestions

        client = create_llm_client("groq", api_key=os.environ["GROQ_API_KEY"])
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


# ---------------------------------------------------------------------------
# Bridge-entity scoping (fix for entity pollution bug)
# ---------------------------------------------------------------------------


class TestBridgeEntityScoping:
    """Verify that dependent sub-Qs get only bridge entities, not full parent set."""

    def test_dependent_subq_gets_only_bridge_entities(self):
        """Sub-Q with depends_on should receive only known_entities from deps."""
        from unittest.mock import MagicMock, patch

        from logicrag_decomposition import _try_logicrag_decomposition

        resolve_calls = []

        def fake_resolve(sub_q, idx, known_facts, known_entities,
                         parent_entities, *args, **kwargs):
            resolve_calls.append({
                "idx": idx,
                "parent_labels": sorted(e.label for e in parent_entities),
            })
            if idx == 0:
                return "Ken Burns", "sparql", [
                    Entity(uri="http://dbpedia.org/resource/Ken_Burns",
                           label="Ken Burns", source="sparql"),
                ]
            return "Brooklyn", "text", []

        parent_entities = [
            Entity(uri="http://dbpedia.org/resource/Thomas_Jefferson",
                   label="Thomas Jefferson", source="spotlight"),
            Entity(uri="http://dbpedia.org/resource/Ken_Burns",
                   label="Ken Burns", source="spotlight"),
        ]

        def fake_decompose(*args, **kwargs):
            return [
                SubQuestion(text="Who directed Thomas Jefferson?",
                            depends_on=[], original_idx=0),
                SubQuestion(text="Where was [director] born?",
                            depends_on=[0], original_idx=1),
            ]

        def fake_call_llm(client, prompt, model):
            # Early stopping check and final answer
            if "can_answer" in prompt.lower() or "pre-parsed" in prompt.lower():
                return '{"can_answer": false, "current_understanding": "need more"}'
            return "FINAL ANSWER: Brooklyn"

        with patch("logicrag_decomposition._decompose_subquestions",
                    side_effect=fake_decompose), \
             patch("logicrag_decomposition._resolve_subquestion",
                    side_effect=fake_resolve), \
             patch("logicrag_decomposition._is_comparison_question",
                    return_value=False), \
             patch("logicrag_decomposition.call_llm",
                    side_effect=fake_call_llm):
            _try_logicrag_decomposition(
                "Where was the director of Thomas Jefferson born?",
                parent_entities, "schema", MagicMock(), "model",
                MagicMock(), MagicMock(), MagicMock(),
            )

        assert len(resolve_calls) == 2
        # Sub-Q 0 (root): should get full parent set
        assert "Thomas Jefferson" in resolve_calls[0]["parent_labels"]
        assert "Ken Burns" in resolve_calls[0]["parent_labels"]
        # Sub-Q 1 (dependent): should get ONLY the bridge entity from sub-Q 0
        assert resolve_calls[1]["parent_labels"] == ["Ken Burns"]
        assert "Thomas Jefferson" not in resolve_calls[1]["parent_labels"]

    def test_root_subq_gets_full_entities(self):
        """Sub-Q with no depends_on should receive the full entity list."""
        from unittest.mock import MagicMock, patch

        from logicrag_decomposition import _try_logicrag_decomposition

        resolve_calls = []

        def fake_resolve(sub_q, idx, known_facts, known_entities,
                         parent_entities, *args, **kwargs):
            resolve_calls.append({
                "idx": idx,
                "parent_labels": sorted(e.label for e in parent_entities),
            })
            return "answer", "sparql", []

        entities = [
            Entity(uri="http://dbpedia.org/resource/A", label="A", source="t"),
            Entity(uri="http://dbpedia.org/resource/B", label="B", source="t"),
        ]

        def fake_decompose(*args, **kwargs):
            return [
                SubQuestion(text="Q about A?", depends_on=[], original_idx=0),
                SubQuestion(text="Q about B?", depends_on=[], original_idx=1),
            ]

        def fake_call_llm(client, prompt, model):
            if "can_answer" in prompt.lower() or "pre-parsed" in prompt.lower():
                return '{"can_answer": false, "current_understanding": "need more"}'
            return "FINAL ANSWER: answer"

        with patch("logicrag_decomposition._decompose_subquestions",
                    side_effect=fake_decompose), \
             patch("logicrag_decomposition._resolve_subquestion",
                    side_effect=fake_resolve), \
             patch("logicrag_decomposition._is_comparison_question",
                    return_value=True), \
             patch("logicrag_decomposition.call_llm",
                    side_effect=fake_call_llm):
            _try_logicrag_decomposition(
                "Compare A and B?",
                entities, "schema", MagicMock(), "model",
                MagicMock(), MagicMock(), MagicMock(),
            )

        # Both root sub-Qs get full entity list
        for call in resolve_calls:
            assert "A" in call["parent_labels"]
            assert "B" in call["parent_labels"]

    def test_no_fallback_when_no_bridge_entities(self):
        """When known_entities is empty for deps, effective_parents is empty (H1 fix).

        Previously this fell back to the full root entity set, causing entity
        pollution across sub-questions. The corrected behaviour passes an empty
        list so _resolve_subquestion works only with bridge entities it discovers
        from answers — not with unrelated root entities.
        """
        from unittest.mock import MagicMock, patch

        from logicrag_decomposition import _try_logicrag_decomposition

        resolve_calls = []

        def fake_resolve(sub_q, idx, known_facts, known_entities,
                         parent_entities, *args, **kwargs):
            resolve_calls.append({
                "idx": idx,
                "parent_labels": sorted(e.label for e in parent_entities),
            })
            if idx == 0:
                # Return answer but NO discovered entities (simulates bridge miss)
                return "some answer", "text", []
            return "final", "text", []

        entities = [
            Entity(uri="http://dbpedia.org/resource/X", label="X", source="t"),
        ]

        def fake_decompose(*args, **kwargs):
            return [
                SubQuestion(text="Q0?", depends_on=[], original_idx=0),
                SubQuestion(text="Q1?", depends_on=[0], original_idx=1),
            ]

        def fake_call_llm(client, prompt, model):
            if "can_answer" in prompt.lower() or "pre-parsed" in prompt.lower():
                return '{"can_answer": false, "current_understanding": "need more"}'
            return "FINAL ANSWER: final"

        with patch("logicrag_decomposition._decompose_subquestions",
                    side_effect=fake_decompose), \
             patch("logicrag_decomposition._resolve_subquestion",
                    side_effect=fake_resolve), \
             patch("logicrag_decomposition._is_comparison_question",
                    return_value=False), \
             patch("logicrag_decomposition.call_llm",
                    side_effect=fake_call_llm):
            _try_logicrag_decomposition(
                "Q about X?",
                entities, "schema", MagicMock(), "model",
                MagicMock(), MagicMock(), MagicMock(),
            )

        # Sub-Q 1 depends on Q0 but Q0 returned no entities -> empty parent list,
        # NOT the full root set (that would pollute the sub-Q with irrelevant entities).
        assert resolve_calls[1]["parent_labels"] == []
