"""Stress tests for _find_unlinked and _has_incomplete_entity_coverage.

Characterization tests — document current behavior, no code changes.
Cases 61–67 (unlinked detection) and 83–88 (comparison patterns).
"""

import sys
from pathlib import Path

import pytest

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dbpedia.entity_linker import DBpediaEntityLinker  # noqa: E402
from models import Entity  # noqa: E402
from routing import _has_incomplete_entity_coverage  # noqa: E402
from tests.test_data.stress_test_cases import (  # noqa: E402
    COMPARISON_COVERAGE_CASES,
    UNLINKED_CASES,
)

# ---------------------------------------------------------------------------
# Correct expected values (verified against actual function behavior 2026-03-25)
# Discrepancies from stress_test_cases.py:
#   p3_084_both_and: data says False, actual True
#     (regex captures 'The' from 'both X and The Matrix' → unlinked)
#   p2_086_three_way: data says False, actual True
#     (regex captures 'X, Y,' → not a label → unlinked)
# ---------------------------------------------------------------------------
_ACTUAL_COMPARISON_EXPECTED = {
    "p3_084_both_and": True,
    "p2_086_three_way": True,
}


def _corrected_comparison_cases():
    """Yield COMPARISON_COVERAGE_CASES with expected values fixed to actual behavior."""
    for param in COMPARISON_COVERAGE_CASES:
        pid = param.id
        if pid in _ACTUAL_COMPARISON_EXPECTED:
            question, entities_data, _original, desc = param.values
            corrected_expected = _ACTUAL_COMPARISON_EXPECTED[pid]
            yield pytest.param(
                question,
                entities_data,
                corrected_expected,
                desc,
                id=pid,
                marks=param.marks,
            )
        else:
            yield param


# ---------------------------------------------------------------------------
# Test 1: _find_unlinked — 7 parametrized cases (#61–67)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.parametrize("mentions,entities_data,expected_unlinked,desc", UNLINKED_CASES)
def test_find_unlinked(mentions, entities_data, expected_unlinked, desc):
    entities = [Entity(uri=u, label=lbl, source=s) for u, lbl, s in entities_data]
    result = DBpediaEntityLinker._find_unlinked(mentions, entities)
    assert set(result) == set(expected_unlinked), (
        f"Case: {desc}. Got {result}, expected {expected_unlinked}"
    )


# ---------------------------------------------------------------------------
# Test 2: _has_incomplete_entity_coverage — 6 parametrized cases (#83–88)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.parametrize(
    "question,entities_data,expected_incomplete,desc",
    list(_corrected_comparison_cases()),
)
def test_comparison_entity_coverage(question, entities_data, expected_incomplete, desc):
    entities = [Entity(uri=u, label=lbl, source=s) for u, lbl, s in entities_data]
    result = _has_incomplete_entity_coverage(question, entities)
    assert result == expected_incomplete, (
        f"Case: {desc}. Got {result}, expected {expected_incomplete}"
    )
