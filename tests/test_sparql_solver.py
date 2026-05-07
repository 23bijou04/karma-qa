"""Unit tests for sparql_solver.solve() — SPARQL generation, execution, retry, answer extraction."""

from unittest.mock import MagicMock, patch

from models import Entity
from sparql_solver import solve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm_client():
    """Minimal OpenAI-compatible mock client (not used directly — call_llm is patched)."""
    return MagicMock()


def _entity(uri="http://dbpedia.org/resource/Inception", label="Inception"):
    return Entity(uri=uri, label=label)


def _make_solve_args(**overrides):
    defaults = dict(
        question="Who directed Inception?",
        schema_ctx="Entity: Inception (Film)\nProperties: dbo:director",
        entities=[_entity()],
        llm_client=_mock_llm_client(),
        model="test-model",
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestSolveHappyPath:
    @patch("sparql_solver._sparql_results_to_entities", return_value=[])
    @patch("sparql_solver.execute_sparql")
    @patch("sparql_solver.call_llm")
    def test_solve_returns_sparql_answer_on_success(
        self, mock_call_llm, mock_execute, mock_discover
    ):
        """Successful SPARQL path returns extracted answer."""
        mock_call_llm.side_effect = [
            "```sparql\nSELECT ?director WHERE { <http://dbpedia.org/resource/Inception> dbo:director ?director }\n```",
            "The director is Christopher Nolan.\nFINAL ANSWER: Christopher Nolan",
        ]
        mock_execute.return_value = ([{"director": "Christopher Nolan"}], None)

        answer, query, discovered = solve(**_make_solve_args())

        assert answer == "Christopher Nolan"
        assert "SELECT" in query.upper()
        assert discovered == []

    @patch("sparql_solver._sparql_results_to_entities", return_value=[_entity()])
    @patch("sparql_solver.execute_sparql")
    @patch("sparql_solver.call_llm")
    def test_solve_returns_discovered_entities(
        self, mock_call_llm, mock_execute, mock_discover
    ):
        """Discovered entities from SPARQL results are returned in third element."""
        mock_call_llm.side_effect = [
            "SELECT ?p WHERE { <http://dbpedia.org/resource/Inception> dbo:director ?p }",
            "FINAL ANSWER: Christopher Nolan",
        ]
        mock_execute.return_value = ([{"p": "http://dbpedia.org/resource/Christopher_Nolan"}], None)

        _, _, discovered = solve(**_make_solve_args())

        assert len(discovered) == 1
        assert discovered[0].label == "Inception"

    @patch("sparql_solver._sparql_results_to_entities", return_value=[])
    @patch("sparql_solver.execute_sparql")
    @patch("sparql_solver.call_llm")
    def test_solve_with_known_facts_block(self, mock_call_llm, mock_execute, mock_discover):
        """Known facts are included in SPARQL generation prompt."""
        mock_call_llm.side_effect = [
            "SELECT ?birth WHERE { <http://dbpedia.org/resource/Christopher_Nolan> dbo:birthDate ?birth }",
            "FINAL ANSWER: 1970-07-30",
        ]
        mock_execute.return_value = ([{"birth": "1970-07-30"}], None)

        args = _make_solve_args(
            question="When was the director of Inception born?",
            known_facts="director = Christopher Nolan",
        )
        answer, _, _ = solve(**args)

        assert answer is not None
        # Verify known_facts appeared in the generation prompt
        gen_prompt = mock_call_llm.call_args_list[0][0][1]
        assert "Christopher Nolan" in gen_prompt


# ---------------------------------------------------------------------------
# Retry on SPARQL execution error
# ---------------------------------------------------------------------------

class TestSolveRetry:
    @patch("sparql_solver._sparql_results_to_entities", return_value=[])
    @patch("sparql_solver.execute_sparql")
    @patch("sparql_solver.call_llm")
    def test_solve_retries_on_execution_error(self, mock_call_llm, mock_execute, mock_discover):
        """First execution fails; retry succeeds and returns answer."""
        mock_call_llm.side_effect = [
            "SELECT ?d WHERE { <http://dbpedia.org/resource/Inception> dbo:director ?d }",  # gen
            "SELECT ?d WHERE { ?film dbo:director ?d FILTER(?film = <http://dbpedia.org/resource/Inception>) }",  # retry gen
            "FINAL ANSWER: Christopher Nolan",  # answer extraction
        ]
        mock_execute.side_effect = [
            ([], "Virtuoso: syntax error"),         # first attempt fails
            ([{"d": "Christopher Nolan"}], None),   # retry succeeds
        ]

        answer, _, _ = solve(**_make_solve_args())

        assert answer == "Christopher Nolan"
        assert mock_execute.call_count == 2

    @patch("sparql_solver._sparql_results_to_entities", return_value=[])
    @patch("sparql_solver.execute_sparql")
    @patch("sparql_solver.call_llm")
    def test_solve_returns_none_when_both_attempts_fail(
        self, mock_call_llm, mock_execute, mock_discover
    ):
        """Both initial and retry execution fail → returns None."""
        mock_call_llm.side_effect = [
            "SELECT ?d WHERE { <http://dbpedia.org/resource/Inception> dbo:director ?d }",
            "SELECT ?d WHERE { ?d a dbo:Person }",  # retry gen
        ]
        mock_execute.return_value = ([], "endpoint unreachable")

        answer, _, _ = solve(**_make_solve_args())

        assert answer is None
        assert mock_execute.call_count == 2

    @patch("sparql_solver._sparql_results_to_entities", return_value=[])
    @patch("sparql_solver.execute_sparql")
    @patch("sparql_solver.call_llm")
    def test_solve_retries_on_empty_results(self, mock_call_llm, mock_execute, mock_discover):
        """Empty result set (no error) triggers retry path."""
        mock_call_llm.side_effect = [
            "SELECT ?d WHERE { <http://dbpedia.org/resource/Inception> dbo:director ?d }",
            "SELECT ?d WHERE { ?film dbo:director ?d }",
            "FINAL ANSWER: Christopher Nolan",
        ]
        mock_execute.side_effect = [
            ([], None),                              # empty → triggers retry
            ([{"d": "Christopher Nolan"}], None),
        ]

        answer, _, _ = solve(**_make_solve_args())

        assert answer == "Christopher Nolan"


# ---------------------------------------------------------------------------
# Null / empty answer handling
# ---------------------------------------------------------------------------

class TestSolveNullAnswer:
    @patch("sparql_solver._sparql_results_to_entities", return_value=[])
    @patch("sparql_solver.execute_sparql")
    @patch("sparql_solver.call_llm")
    def test_null_answer_phrase_returns_none(self, mock_call_llm, mock_execute, mock_discover):
        """LLM returning null-answer phrases like 'unknown' resolves to None."""
        mock_call_llm.side_effect = [
            "SELECT ?d WHERE { ?d a dbo:Film }",
            "FINAL ANSWER: unknown",
        ]
        mock_execute.return_value = ([{"d": "some_result"}], None)

        answer, _, _ = solve(**_make_solve_args())

        assert answer is None

    @patch("sparql_solver._sparql_results_to_entities", return_value=[])
    @patch("sparql_solver.execute_sparql")
    @patch("sparql_solver.call_llm")
    def test_not_found_answer_returns_none(self, mock_call_llm, mock_execute, mock_discover):
        """'not found' answer phrase is treated as None."""
        mock_call_llm.side_effect = [
            "SELECT ?d WHERE { ?d a dbo:Film }",
            "FINAL ANSWER: not found",
        ]
        mock_execute.return_value = ([{"d": "some_result"}], None)

        answer, _, _ = solve(**_make_solve_args())

        assert answer is None

    @patch("sparql_solver._sparql_results_to_entities", return_value=[])
    @patch("sparql_solver.execute_sparql")
    @patch("sparql_solver.call_llm")
    def test_list_article_answer_returns_none(self, mock_call_llm, mock_execute, mock_discover):
        """Answer matching list-article pattern (e.g. 'List of films') is rejected."""
        mock_call_llm.side_effect = [
            "SELECT ?d WHERE { ?d a dbo:Film }",
            "FINAL ANSWER: List of films directed by Christopher Nolan",
        ]
        mock_execute.return_value = ([{"d": "some_result"}], None)

        answer, _, _ = solve(**_make_solve_args())

        assert answer is None


# ---------------------------------------------------------------------------
# SPARQL read-only guard (already tested in test_sparql_readonly.py, extended here
# to verify solve() itself returns None for write queries from LLM)
# ---------------------------------------------------------------------------

class TestSolveReadonlyGuard:
    @patch("sparql_solver.execute_sparql")
    @patch("sparql_solver.call_llm")
    def test_write_query_from_llm_is_rejected(self, mock_call_llm, mock_execute):
        """solve() returns None without calling execute_sparql for INSERT queries."""
        mock_call_llm.return_value = "INSERT DATA { <http://example.org/x> <y> <z> }"

        answer, query, discovered = solve(**_make_solve_args())

        assert answer is None
        assert discovered == []
        mock_execute.assert_not_called()

    @patch("sparql_solver.execute_sparql")
    @patch("sparql_solver.call_llm")
    def test_write_retry_query_from_llm_is_rejected(self, mock_call_llm, mock_execute):
        """Even if initial query is valid, a write retry query is blocked."""
        mock_call_llm.side_effect = [
            "SELECT ?d WHERE { <http://dbpedia.org/resource/Inception> dbo:director ?d }",  # gen OK
            "DELETE WHERE { ?s ?p ?o }",  # retry returns write op
        ]
        mock_execute.side_effect = [
            ([], "error to trigger retry"),
        ]

        answer, _, _ = solve(**_make_solve_args())

        # execute_sparql only called once (for the initial valid query)
        assert mock_execute.call_count == 1
        assert answer is None
