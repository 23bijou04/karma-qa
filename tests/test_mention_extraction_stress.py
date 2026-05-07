"""Stress tests for _extract_mentions: 45 cases from catalog sections 1.1-1.6.

Parametrized by severity (P0-P3):
  P0/P1 — known bugs, marked xfail (assert desired behavior, currently broken)
  P2/P3 — characterization tests (assert current behavior, must all pass)

No network calls. _extract_mentions uses spaCy only.
"""

import sys
from pathlib import Path

# Ensure project root (parent of tests/) is importable as 'dbpedia_qa'.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest  # noqa: E402

from dbpedia.entity_linker import DBpediaEntityLinker  # noqa: E402
from tests.test_data.stress_test_cases import (  # noqa: E402
    MENTION_ADDITIONAL_CASES,
    MENTION_DEDUP_CASES,
    MENTION_NER_CASES,
    MENTION_POSSESSIVE_CASES,
    MENTION_QUOTED_CASES,
    MENTION_STOPWORD_CASES,
    MENTION_TITLE_KEYWORD_CASES,
)

# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def _check(mentions, expected_fragments, expected_missing, desc):
    """Central assertion: fragment-in-any-mention + exact-not-in-any-mention."""
    for frag in expected_fragments:
        assert any(frag.lower() in m.lower() for m in mentions), (
            f"Case: {desc!r}. Expected fragment {frag!r} not found in {mentions!r}"
        )
    for frag in expected_missing:
        assert not any(frag.lower() == m.lower() for m in mentions), (
            f"Case: {desc!r}. Unexpected exact mention {frag!r} found in {mentions!r}"
        )


# ---------------------------------------------------------------------------
# 1.1  Stopword stripping (cases 1–5)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.parametrize("question,expected_fragments,expected_missing,desc", MENTION_STOPWORD_CASES)
def test_stopword_stripping(question, expected_fragments, expected_missing, desc):
    mentions = DBpediaEntityLinker._extract_mentions(question)
    _check(mentions, expected_fragments, expected_missing, desc)


# ---------------------------------------------------------------------------
# 1.2  Possessive splitting (cases 6–12)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.parametrize("question,expected_fragments,expected_missing,desc", MENTION_POSSESSIVE_CASES)
def test_possessive_splitting(question, expected_fragments, expected_missing, desc):
    mentions = DBpediaEntityLinker._extract_mentions(question)
    _check(mentions, expected_fragments, expected_missing, desc)


# ---------------------------------------------------------------------------
# 1.3  NER entity processing (cases 13–21, plus case 106)
# ---------------------------------------------------------------------------

# Case 106 lives here: it tests the interaction between title-after-keyword
# and NER or-splitting ("song Truth or Dare?" — title path wins).
_NER_CASES_EXTENDED = MENTION_NER_CASES + [MENTION_ADDITIONAL_CASES[1]]


@pytest.mark.stress
@pytest.mark.parametrize("question,expected_fragments,expected_missing,desc", _NER_CASES_EXTENDED)
def test_ner_processing(question, expected_fragments, expected_missing, desc):
    mentions = DBpediaEntityLinker._extract_mentions(question)
    _check(mentions, expected_fragments, expected_missing, desc)


# ---------------------------------------------------------------------------
# 1.4  Title-after-keyword extraction (cases 22–32)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.parametrize("question,expected_fragments,expected_missing,desc", MENTION_TITLE_KEYWORD_CASES)
def test_title_after_keyword(question, expected_fragments, expected_missing, desc):
    mentions = DBpediaEntityLinker._extract_mentions(question)
    _check(mentions, expected_fragments, expected_missing, desc)


# ---------------------------------------------------------------------------
# 1.5  Deduplication & filtering (cases 33–39)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.parametrize("question,expected_fragments,expected_missing,desc", MENTION_DEDUP_CASES)
def test_dedup_filtering(question, expected_fragments, expected_missing, desc):
    mentions = DBpediaEntityLinker._extract_mentions(question)
    _check(mentions, expected_fragments, expected_missing, desc)


# ---------------------------------------------------------------------------
# 1.6  Quoted entities (cases 40–45, plus case 105)
# ---------------------------------------------------------------------------

# Case 105 tests the same curly-quote extraction path as case 42.
_QUOTED_CASES_EXTENDED = MENTION_QUOTED_CASES + [MENTION_ADDITIONAL_CASES[0]]


@pytest.mark.stress
@pytest.mark.parametrize("question,expected_fragments,expected_missing,desc", _QUOTED_CASES_EXTENDED)
def test_quoted_entities(question, expected_fragments, expected_missing, desc):
    mentions = DBpediaEntityLinker._extract_mentions(question)
    _check(mentions, expected_fragments, expected_missing, desc)
