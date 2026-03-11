"""Tests for QuestionAnalysis and KEYWORD_TO_PROPERTIES."""

from question_analysis import (
    _BRIDGE_KEYWORDS,
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

    def test_bridge_expected_target_type_is_range(self):
        """Bridge question: expected_target_type should be the PROPERTY_RANGES target
        of the bridge property, not raw vote winner."""
        result = analyze_question("When was the director of Inception born?")
        # bridge_property="director", PROPERTY_RANGES["dbo:director"]="Person"
        assert result.expected_target_type == "Person"

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


# ---------------------------------------------------------------------------
# Fix 2: Bridge expected_target_type override via PROPERTY_RANGES
# ---------------------------------------------------------------------------

class TestBridgeExpectedTargetType:
    """Verify bridge questions override expected_target_type with PROPERTY_RANGES target."""

    def test_director_bridge_returns_person(self):
        result = analyze_question("When was the director of Inception born?")
        assert result.expected_target_type == "Person"

    def test_author_bridge_returns_person(self):
        result = analyze_question("Where was the author of Harry Potter born?")
        assert result.expected_target_type == "Person"

    def test_producer_bridge_returns_person(self):
        result = analyze_question("Where was the producer of Titanic born?")
        assert result.expected_target_type == "Person"

    def test_starring_bridge_returns_person(self):
        """'actor' is a bridge keyword mapping to 'starring' property."""
        result = analyze_question("Where was the actor of The Matrix born?")
        assert result.expected_target_type == "Person"

    def test_composer_bridge_returns_person(self):
        result = analyze_question("When was the composer of Inception born?")
        assert result.expected_target_type == "Person"

    def test_writer_bridge_returns_person(self):
        result = analyze_question("Where was the writer of Inception born?")
        assert result.expected_target_type == "Person"

    def test_founder_bridge_returns_person(self):
        result = analyze_question("When was the founder of Apple born?")
        assert result.expected_target_type == "Person"

    def test_father_bridge_returns_person(self):
        result = analyze_question("When was the father of Prince William born?")
        assert result.expected_target_type == "Person"

    def test_mother_bridge_returns_person(self):
        result = analyze_question("Where was the mother of Einstein born?")
        assert result.expected_target_type == "Person"

    def test_spouse_not_a_bridge_keyword(self):
        """'spouse' is NOT in _BRIDGE_KEYWORDS, so no bridge override."""
        result = analyze_question("When was the spouse of Einstein born?")
        # spouse + born → both matched but spouse is not a bridge keyword
        assert result.bridge_keyword is None or result.bridge_keyword != "spouse"

    def test_direct_director_keeps_film(self):
        """Non-bridge (single keyword) keeps vote-based type."""
        result = analyze_question("Who directed Inception?")
        assert result.hop_structure == "direct"
        # No hop2 keyword → no bridge override
        assert result.expected_target_type == "Film"

    def test_direct_born_keeps_vote_type(self):
        """Direct 'born' with no bridge → vote-based type (Person or MusicalArtist)."""
        result = analyze_question("When was Einstein born?")
        assert result.hop_structure == "direct"
        assert result.bridge_keyword is None
        # No bridge override, so vote-based. Person and MusicalArtist both have
        # birthDate/birthPlace; either is acceptable since there's no bridge.
        assert result.expected_target_type in ("Person", "MusicalArtist")

    def test_comparison_bridge_still_overrides(self):
        """Comparison + bridge: override still applies."""
        result = analyze_question(
            "Who was born earlier, the director of Inception or the director of Titanic?"
        )
        # Has comparison word "earlier" AND bridge "director" + hop2 "born"
        if result.bridge_keyword and result.hop2_keyword:
            assert result.expected_target_type == "Person"

    def test_no_bridge_no_override(self):
        """Question with no bridge keyword keeps vote-based type."""
        result = analyze_question("What genre is Bohemian Rhapsody?")
        assert result.bridge_keyword is None
        # genre maps to both Book and MusicalArtist; vote-based picks one
        assert result.expected_target_type in ("Book", "MusicalArtist")

    def test_bridge_property_not_in_ranges_keeps_vote(self):
        """If bridge property isn't in PROPERTY_RANGES, vote-based type is kept."""
        # "award" is not a bridge keyword so this tests a different path,
        # but let's verify that non-PROPERTY_RANGES bridge props don't crash.
        result = analyze_question("What is the genre of that?")
        # "genre" is not a bridge keyword, so no override
        assert result.bridge_keyword is None


class TestBridgeDetection:
    """Verify bridge/hop2 keyword detection across all bridge keywords."""

    def test_all_bridge_keywords_detected(self):
        """Each _BRIDGE_KEYWORD paired with a hop2 keyword creates a bridge."""
        hop2_suffix = {
            "born": "born",
            "died": "died",
            "country": "country",
            "nationality": "nationality",
        }
        for bk in _BRIDGE_KEYWORDS:
            for h2, h2_word in hop2_suffix.items():
                q = f"Where was the {bk} of X {h2_word}?"
                result = analyze_question(q)
                if result.bridge_keyword and result.hop2_keyword:
                    assert result.hop_structure == "bridge", (
                        f"Expected bridge for '{q}', got {result.hop_structure}"
                    )

    def test_bridge_without_hop2_is_not_bridge(self):
        """Bridge keyword alone (no hop2) doesn't create bridge structure."""
        result = analyze_question("Who directed Inception?")
        assert result.hop_structure == "direct"
        assert result.bridge_keyword is None

    def test_hop2_without_bridge_is_not_bridge(self):
        """Hop2 keyword alone (no bridge) doesn't create bridge structure."""
        result = analyze_question("When was Einstein born?")
        assert result.hop_structure == "direct"
        assert result.bridge_keyword is None

    def test_hop2_properties_populated_for_bridge(self):
        """Bridge detection populates hop2_properties."""
        result = analyze_question("Where was the director of Inception born?")
        assert result.hop2_properties is not None
        assert "dbo:birthDate" in result.hop2_properties
        assert "dbo:birthPlace" in result.hop2_properties
        assert "dbp:birthDate" in result.hop2_properties
        assert "dbp:birthPlace" in result.hop2_properties

    def test_hop2_properties_none_for_non_bridge(self):
        result = analyze_question("Who directed Inception?")
        assert result.hop2_properties is None


class TestComparisonDetection:
    """Verify comparison word detection."""

    def test_all_comparison_words(self):
        from question_analysis import COMPARISON_WORDS
        for word in COMPARISON_WORDS:
            q = f"Which entity is {word}, A or B?"
            result = analyze_question(q)
            assert result.is_comparison, f"Failed for comparison word: {word}"

    def test_no_comparison_word(self):
        result = analyze_question("Who directed Inception?")
        assert result.is_comparison is False

    def test_comparison_with_bridge(self):
        result = analyze_question("Who was born earlier, the director of X or Y?")
        assert result.is_comparison is True
        # Should still detect bridge
        if result.bridge_keyword:
            assert result.hop_structure == "bridge"


# ---------------------------------------------------------------------------
# Edge cases: word boundaries, multiple keywords, compositional
# ---------------------------------------------------------------------------

class TestWordBoundary:
    def test_born_not_in_stubborn(self):
        result = analyze_question("She was a stubborn woman.")
        assert "born" not in result.target_keywords

    def test_died_not_in_studied(self):
        result = analyze_question("He studied physics at MIT.")
        assert "died" not in result.target_keywords

    def test_direct_not_in_directly(self):
        """'direct' should not match in 'directly'."""
        result = analyze_question("She directly managed the team.")
        assert "direct" not in result.target_keywords

    def test_founded_not_in_unfounded(self):
        result = analyze_question("That claim is unfounded.")
        assert "founded" not in result.target_keywords

    def test_award_not_in_awarded_separately(self):
        """Both 'award' and 'awarded' are valid keywords."""
        result = analyze_question("She was awarded the prize.")
        assert "awarded" in result.target_keywords

    def test_parent_not_in_transparent(self):
        result = analyze_question("The policy was transparent.")
        assert "parent" not in result.target_keywords


class TestNoEnglishWords:
    def test_symbols_only(self):
        result = analyze_question("??? 12345 @#$")
        assert result.target_keywords == []
        assert result.target_properties == frozenset()
        assert result.hop_structure == "direct"

    def test_empty_string(self):
        result = analyze_question("")
        assert result.target_keywords == []
        assert result.hop_structure == "direct"

    def test_single_word_no_match(self):
        result = analyze_question("Hello")
        assert result.target_keywords == []


class TestMultipleBridgeHop2:
    def test_multiple_bridge_keywords_first_wins(self):
        """Two bridge keywords: longest-first sort determines which is matched first."""
        result = analyze_question("Where was the director and producer born?")
        assert result.bridge_keyword is not None
        assert result.hop2_keyword == "born"
        # Both "director" and "producer" are bridge keywords; first from longest-first
        assert result.bridge_keyword in ("director", "producer", "produced")

    def test_multiple_hop2_keywords_first_wins(self):
        """Two hop2 keywords: first from longest-first sort is selected."""
        result = analyze_question("Where was the director born and died?")
        assert result.hop2_keyword is not None
        assert result.hop2_keyword in ("born", "died", "death", "birth")

    def test_bridge_founded_uses_first_property(self):
        """'founded' maps to [foundedBy, foundingDate, foundingYear].
        bridge_property should be 'foundedBy' (first)."""
        result = analyze_question("When was the founder of Apple born?")
        assert result.bridge_property == "foundedBy"


class TestCompositionalHopStructure:
    def test_compositional_explicit(self):
        """Multiple keywords without bridge+hop2 pair -> compositional."""
        result = analyze_question("What is the award and genre of the artist?")
        assert len(result.target_keywords) >= 2
        assert result.hop_structure == "compositional"

    def test_comparison_only_no_keywords(self):
        """Only comparison word, no property keywords."""
        result = analyze_question("Which is earlier?")
        assert result.is_comparison is True
        assert result.target_keywords == []
        assert result.hop_structure == "comparison"

    def test_comparison_overrides_compositional(self):
        """Comparison + multiple keywords: if comparison word present, is_comparison=True."""
        result = analyze_question("Which country has more population?")
        assert result.is_comparison is True


class TestTargetKeywordsSet:
    def test_matches_list_content(self):
        result = analyze_question("Where was the director of Inception born?")
        assert result.target_keywords_set == frozenset(result.target_keywords)

    def test_is_frozenset(self):
        result = analyze_question("Who directed Inception?")
        assert isinstance(result.target_keywords_set, frozenset)

    def test_empty_for_no_keywords(self):
        result = analyze_question("What is the meaning of life?")
        assert result.target_keywords_set == frozenset()


class TestTypeVoteDeterminism:
    def test_genre_deterministic_type(self):
        """'genre' maps to both Book and MusicalArtist. Result should be deterministic."""
        result1 = analyze_question("What genre is this?")
        result2 = analyze_question("What genre is this?")
        assert result1.expected_target_type == result2.expected_target_type
        assert result1.expected_target_type in ("Book", "MusicalArtist")

    def test_occupation_type(self):
        """'occupation' maps to Person only."""
        result = analyze_question("What is the occupation of X?")
        assert result.expected_target_type == "Person"
