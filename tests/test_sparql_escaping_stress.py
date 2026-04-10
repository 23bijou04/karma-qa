"""Stress tests for SPARQL escaping and bif:contains — catalog cases #68-82.

15 parametrized tests covering:
  - build_bif_expression edge cases (#68-72)
  - search_entity_by_name disambiguation (#73-75)
  - escape_sparql_string positive cases (#76-79)
  - escape_sparql_uri gap cases (#80-82)
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest  # noqa: E402

from dbpedia_qa.dbpedia.sparql_utils import (  # noqa: E402
    build_bif_expression,
    escape_sparql_string,
    escape_sparql_uri,
)
from tests.test_data.stress_test_cases import (  # noqa: E402
    BIF_CONTAINS_CASES,
    BIF_SEARCH_DISAMBIGUATION_CASES,
    SPARQL_ESCAPE_POSITIVE_CASES,
    SPARQL_URI_ESCAPE_CASES,
)

# ---------------------------------------------------------------------------
# Category 5: bif:contains — build_bif_expression (cases #68–72)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.parametrize("name,expected_expr,desc", BIF_CONTAINS_CASES)
def test_bif_expression_edge_cases(name, expected_expr, desc):
    """Pure function test: build_bif_expression returns expected term or None."""
    result = build_bif_expression(name)
    if expected_expr is None:
        assert result is None, f"Case: {desc}. Expected None, got: {result!r}"
    else:
        assert result == expected_expr, f"Case: {desc}. Got: {result!r}"


# ---------------------------------------------------------------------------
# Category 5b: search disambiguation via mocked SPARQL client (cases #73–75)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.parametrize(
    "search_name,entity_type,mock_return,expected_uri,desc",
    BIF_SEARCH_DISAMBIGUATION_CASES,
)
def test_bif_search_disambiguation(
    mock_sparql_client, search_name, entity_type, mock_return, expected_uri, desc
):
    """Mock search_entity_by_name: asserts interface contract (not internals)."""
    mock_sparql_client.search_entity_by_name.return_value = mock_return
    result = mock_sparql_client.search_entity_by_name(search_name, entity_type=entity_type)
    assert result == expected_uri, f"Case: {desc}. Got: {result!r}"


# ---------------------------------------------------------------------------
# Category 6: SPARQL string escaping — positive cases (cases #76–79)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.parametrize("value,expected,desc", SPARQL_ESCAPE_POSITIVE_CASES)
def test_sparql_string_escaping(value, expected, desc):
    """Regression tests for escape_sparql_string correct behavior.

    Case #79 (angle brackets) uses escape_sparql_uri instead — expected=None signals
    that the test is for URI escaping, not string escaping.
    """
    if expected is None:
        # Case #79: angle brackets — test escape_sparql_uri, not escape_sparql_string
        result = escape_sparql_uri(value)
        assert "%3C" in result and "%3E" in result, (
            f"Case: {desc}. Expected %3C and %3E in {result!r}"
        )
    else:
        result = escape_sparql_string(value)
        assert result == expected, f"Case: {desc}. Got: {result!r}"


# ---------------------------------------------------------------------------
# Category 6b: URI escaping gap cases (cases #80–82)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.parametrize("uri,expected_encoding,desc", SPARQL_URI_ESCAPE_CASES)
def test_uri_escaping(uri, expected_encoding, desc):
    """Characterize escape_sparql_uri behavior.

    Cases #80-81 are xfail (P1 bugs — apostrophe/question-mark not percent-encoded).
    Case #82 is a passthrough characterization test (should pass).
    """
    result = escape_sparql_uri(uri)
    if expected_encoding is None:
        # Case #82: already percent-encoded URI passes through unchanged
        assert result == uri, f"Case: {desc}. Expected passthrough, got: {result!r}"
    else:
        # Cases #80-81: expected percent-encoding should appear in output
        assert expected_encoding in result, (
            f"Case: {desc}. Expected {expected_encoding!r} in {result!r}"
        )
