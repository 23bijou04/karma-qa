"""Stress tests for type hinting and type validation — 15+ parametrized cases.

Covers:
  - _extract_type_hint: 6 cases (cases 46–51)
  - _validate_entity_types: 9 cases (cases 52–60)
  - _TYPE_SUFFIX_MAP completeness: 1 standalone P0 xfail
  - Year-strip P1 bug: 1 standalone xfail verifying search query corruption (#58)
"""

import pytest

from dbpedia_qa.dbpedia.entity_linker import _TYPE_SUFFIX_MAP, DBpediaEntityLinker
from dbpedia_qa.models import Entity
from tests.test_data.stress_test_cases import TYPE_HINT_CASES, TYPE_VALIDATION_CASES

# ---------------------------------------------------------------------------
# Type hint extraction (cases 46–51) — pure function, no mocks needed
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.parametrize("question,expected_type,desc", TYPE_HINT_CASES)
def test_type_hint_extraction(question, expected_type, desc):
    """_extract_type_hint returns the correct dbo: type or None."""
    result = DBpediaEntityLinker._extract_type_hint(question)
    assert result == expected_type, f"Case: {desc}. Got {result!r}, expected {expected_type!r}"


# ---------------------------------------------------------------------------
# Type validation (cases 52–60) — requires linker_no_network + mock client
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.parametrize(
    "entities_data,question,expected_type,mock_types,expected_replacement,desc",
    TYPE_VALIDATION_CASES,
)
def test_type_validation(
    linker_no_network,
    mock_sparql_client,
    entities_data,
    question,
    expected_type,
    mock_types,
    expected_replacement,
    desc,
):
    """_validate_entity_types replaces or keeps entities per type match logic.

    expected_replacement=False → entity URI must be unchanged (correct type or no
                                  relation context triggered).
    expected_replacement=True  → replacement path exercised; with search returning
                                  None the original entity is kept, but get_entity_types
                                  must have been called (when suffix is present).
    """
    entities = [Entity(uri=uri, label=lbl, source=src) for uri, lbl, src in entities_data]
    mock_sparql_client.get_entity_types.return_value = mock_types
    mock_sparql_client.search_entity_by_name.return_value = None

    result = linker_no_network._validate_entity_types(entities, question, expected_type)

    assert isinstance(result, list), f"Case: {desc}. Expected list, got {type(result)}"
    assert len(result) == len(entities), (
        f"Case: {desc}. Entity count changed: {len(entities)} → {len(result)}"
    )

    if not expected_replacement:
        for orig, kept in zip(entities, result):
            assert kept.uri == orig.uri, (
                f"Case: {desc}. Expected entity kept at {orig.uri!r}, got {kept.uri!r}"
            )
    else:
        # Replacement path: suffix present means get_entity_types was invoked.
        if _TYPE_SUFFIX_MAP.get(expected_type):
            assert mock_sparql_client.get_entity_types.called, (
                f"Case: {desc}. Expected get_entity_types to be called when "
                f"suffix is present for {expected_type!r}."
            )


# ---------------------------------------------------------------------------
# P1 #58 — _clean strips year from entity title → wrong search query
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.p1
@pytest.mark.xfail(
    reason="P1 #58: _clean strips year from 'The Year 2000' → searches 'Year book' not 'The Year 2000 book'",
    strict=False,
)
def test_type_validation_year_title_clean_p1(linker_no_network, mock_sparql_client):
    """_validate_entity_types should search with the full title, not a year-stripped fragment.

    Bug: _clean removes 'The' and '2000' from 'The Year 2000', leaving just 'Year'.
    The search call becomes 'Year book' instead of 'The Year 2000 book'.
    """
    mock_sparql_client.get_entity_types.return_value = []  # no Book type → triggers search
    mock_sparql_client.search_entity_by_name.return_value = None

    entity = Entity(
        uri="http://dbpedia.org/resource/The_Year_2000",
        label="The Year 2000",
        source="spotlight",
    )
    linker_no_network._validate_entity_types([entity], "wrote The Year 2000", "dbo:Book")

    # The correct search query preserves the full entity title.
    # Buggy code produces 'Year book'; correct code should produce 'The Year 2000 book'.
    call_args = mock_sparql_client.search_entity_by_name.call_args
    assert call_args is not None, "search_entity_by_name was never called"
    actual_query = call_args[0][0]  # first positional arg
    assert "Year 2000" in actual_query, (
        f"Search query should preserve title year, got: {actual_query!r}"
    )


# ---------------------------------------------------------------------------
# _TYPE_SUFFIX_MAP completeness — P0 standalone xfail
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.p0
def test_type_suffix_map_completeness():
    """All types in TYPE_HINTS and RELATION_TYPE_HINTS must have _TYPE_SUFFIX_MAP entries."""
    from dbpedia_qa.dbpedia.entity_linker import (
        _TYPE_SUFFIX_MAP,
        RELATION_TYPE_HINTS,
        TYPE_HINTS,
    )
    all_types = set(TYPE_HINTS.values()) | set(RELATION_TYPE_HINTS.values())
    missing = sorted(t for t in all_types if t not in _TYPE_SUFFIX_MAP)
    assert missing == [], f"Missing from _TYPE_SUFFIX_MAP: {missing}"
