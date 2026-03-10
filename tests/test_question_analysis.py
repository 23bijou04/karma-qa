"""Tests for QuestionAnalysis and KEYWORD_TO_PROPERTIES."""

from question_analysis import (
    KEYWORD_TO_PROPERTIES,
    analyze_question,
)
from routing import BRIDGE_WORD_TO_PROP


class TestKeywordToProperties:
    def test_director_present(self):
        assert "director" in KEYWORD_TO_PROPERTIES
        assert "director" in KEYWORD_TO_PROPERTIES["director"]

    def test_born_expands_to_date_and_place(self):
        props = KEYWORD_TO_PROPERTIES["born"]
        assert "birthDate" in props
        assert "birthPlace" in props

    def test_all_bridge_words_covered(self):
        """Every key in routing.py BRIDGE_WORD_TO_PROP must be in KEYWORD_TO_PROPERTIES."""
        for word in BRIDGE_WORD_TO_PROP:
            assert word in KEYWORD_TO_PROPERTIES, f"Missing bridge word: {word}"


class TestAnalyzeQuestion:
    def test_direct_single_hop(self):
        result = analyze_question("When was Albert Einstein born?")
        assert "born" in result.target_keywords
        assert "dbo:birthDate" in result.target_properties
        assert result.hop_structure == "direct"

    def test_bridge_2hop(self):
        result = analyze_question("Where was the director of Inception born?")
        assert result.bridge_keyword == "director" or result.bridge_keyword == "directed"
        assert result.hop2_keyword == "born"
        assert result.hop_structure == "bridge"
        assert result.hop2_properties is not None
        assert "dbo:birthDate" in result.hop2_properties

    def test_comparison(self):
        result = analyze_question("Who was born first, Einstein or Bohr?")
        assert result.is_comparison is True
        assert "dbo:birthDate" in result.target_properties

    def test_no_keyword_match(self):
        result = analyze_question("What is the meaning of life?")
        assert result.target_keywords == []
        assert result.target_properties == frozenset()
        assert result.hop_structure == "direct"

    def test_expected_target_type_for_director(self):
        result = analyze_question("Who directed Inception?")
        # director is in Film class
        assert result.expected_target_type == "Film"

    def test_bridge_property_set(self):
        result = analyze_question("When was the director of Inception born?")
        assert result.bridge_property == "director"

    def test_compositional_multiple_keywords(self):
        """Multiple keywords without clear bridge -> compositional."""
        result = analyze_question("What is the capital of the country where Einstein was born?")
        # Has "capital" and "born" -- could be bridge or compositional
        assert len(result.target_keywords) >= 2
        # Both born and capital matched
        assert "dbo:capital" in result.target_properties
        assert "dbo:birthDate" in result.target_properties

    def test_target_keywords_set(self):
        result = analyze_question("Who directed Inception?")
        assert isinstance(result.target_keywords_set, frozenset)
        assert "directed" in result.target_keywords_set or "director" in result.target_keywords_set
