"""Stress test for entity linking with special characters and edge cases.

Tests the full chain: mention extraction → SPARQL escaping → bif:contains →
Spotlight → URI handling. Covers apostrophes, question marks, parentheses,
ampersands, accented characters, and other tricky entity names.
"""

import pytest

from dbpedia_qa.dbpedia.entity_linker import DBpediaEntityLinker
from dbpedia_qa.dbpedia.sparql_utils import (
    build_bif_expression,
    escape_bif_term,
    escape_sparql_string,
    escape_sparql_uri,
)

# ── Unit tests: escape functions (no network) ──────────────────────────


class TestEscapeSparqlString:
    """Test SPARQL string literal escaping."""

    def test_apostrophe(self):
        assert escape_sparql_string("Aren't") == "Aren't"  # single quotes OK in ""

    def test_double_quote(self):
        assert escape_sparql_string('Say "Hello"') == 'Say \\"Hello\\"'

    def test_backslash(self):
        assert escape_sparql_string("C:\\path") == "C:\\\\path"

    def test_newlines(self):
        assert escape_sparql_string("line1\nline2") == "line1\\nline2"

    def test_tabs(self):
        assert escape_sparql_string("col1\tcol2") == "col1\\tcol2"

    def test_combined_special(self):
        val = 'He said "don\'t"\nOK'
        escaped = escape_sparql_string(val)
        assert '\\"' in escaped
        assert "\\n" in escaped
        assert "don't" in escaped  # apostrophe preserved

    def test_empty_string(self):
        assert escape_sparql_string("") == ""

    def test_only_special_chars(self):
        assert escape_sparql_string('"""') == '\\"\\"\\"\\"'[:-2]  # three escaped quotes


class TestEscapeBifTerm:
    """Test bif:contains term escaping."""

    def test_apostrophe_doubled(self):
        assert escape_bif_term("Aren't") == "Aren''t"

    def test_double_quotes_stripped(self):
        assert escape_bif_term('"Hello"') == "Hello"

    def test_parentheses_stripped(self):
        assert escape_bif_term("(film)") == "film"

    def test_brackets_stripped(self):
        assert escape_bif_term("[disambiguation]") == "disambiguation"

    def test_backslash_stripped(self):
        assert escape_bif_term("C\\sharp") == "Csharp"

    def test_angle_brackets_stripped(self):
        assert escape_bif_term("<tag>") == "tag"

    def test_semicolons_stripped(self):
        assert escape_bif_term("foo;bar") == "foobar"

    def test_question_mark_preserved(self):
        # ? is NOT in the strip set — passes through
        assert escape_bif_term("Did?") == "Did?"

    def test_exclamation_preserved(self):
        assert escape_bif_term("Yahoo!") == "Yahoo!"

    def test_hyphen_preserved(self):
        assert escape_bif_term("Spider-Man") == "Spider-Man"

    def test_ampersand_preserved(self):
        assert escape_bif_term("Simon&Garfunkel") == "Simon&Garfunkel"

    def test_period_preserved(self):
        assert escape_bif_term("Dr.") == "Dr."

    def test_colon_preserved(self):
        assert escape_bif_term("Star:") == "Star:"

    def test_plus_sign_preserved(self):
        assert escape_bif_term("C++") == "C++"

    def test_accented_chars_preserved(self):
        assert escape_bif_term("Beyoncé") == "Beyoncé"

    def test_empty_after_strip(self):
        assert escape_bif_term('()""') == ""

    def test_obrien_apostrophe(self):
        assert escape_bif_term("O'Brien") == "O''Brien"


class TestBuildBifExpression:
    """Test full bif:contains expression builder."""

    def test_simple_name(self):
        assert build_bif_expression("Inception") == "'Inception'"

    def test_multi_word(self):
        result = build_bif_expression("Spider Man")
        assert result == "'Spider' AND 'Man'"

    def test_stopword_removal(self):
        result = build_bif_expression("The Lord of the Rings")
        assert result == "'Lord' AND 'Rings'"

    def test_short_word_removal(self):
        # Single char words are stripped
        result = build_bif_expression("I Robot")
        assert result == "'Robot'"

    def test_all_stopwords_returns_none(self):
        assert build_bif_expression("a the of") is None

    def test_apostrophe_in_word(self):
        result = build_bif_expression("Aren't You Kind Of Glad We Did?")
        # "Aren't" -> "Aren''t" (doubled apostrophe)
        # Stopwords removed: "of"
        # Short words removed: none
        assert "'Aren''t'" in result
        assert "'You'" in result
        assert "'Kind'" in result
        assert "'Glad'" in result
        assert "'We'" in result
        assert "'Did?'" in result

    def test_hyphenated_name(self):
        result = build_bif_expression("Spider-Man")
        assert result == "'Spider-Man'"

    def test_ampersand_name(self):
        result = build_bif_expression("Simon & Garfunkel")
        # "&" is single char, gets stripped
        assert "'Simon'" in result
        assert "'Garfunkel'" in result

    def test_accented_name(self):
        result = build_bif_expression("Beyoncé Knowles")
        assert "'Beyoncé'" in result
        assert "'Knowles'" in result

    def test_parenthesized_suffix(self):
        result = build_bif_expression("Do the Right Thing (film)")
        # Parentheses stripped by escape_bif_term
        assert "'Right'" in result
        assert "'Thing'" in result
        assert "'film'" in result
        # No leftover parens
        assert "(" not in result
        assert ")" not in result

    def test_colon_in_title(self):
        result = build_bif_expression("Star Wars: Episode IV")
        assert "'Star'" in result
        assert "'Wars:'" in result  # colon preserved
        assert "'Episode'" in result
        assert "'IV'" in result

    def test_period_in_name(self):
        result = build_bif_expression("Dr. Dre")
        assert "'Dr.'" in result
        assert "'Dre'" in result

    def test_exclamation_mark(self):
        result = build_bif_expression("Yahoo! Inc")
        assert "'Yahoo!'" in result
        assert "'Inc'" in result

    def test_number_name(self):
        result = build_bif_expression("50 Cent")
        assert "'50'" in result
        assert "'Cent'" in result

    def test_all_parens_returns_none(self):
        # After stripping parens and filtering, nothing left
        assert build_bif_expression("() ()") is None

    def test_plus_sign(self):
        result = build_bif_expression("C++ Programming")
        assert "'C++'" in result
        assert "'Programming'" in result


class TestEscapeSparqlUri:
    """Test URI escaping for <uri> syntax."""

    def test_clean_uri(self):
        uri = "http://dbpedia.org/resource/Inception"
        assert escape_sparql_uri(uri) == uri

    def test_space_in_uri(self):
        uri = "http://dbpedia.org/resource/New York"
        assert escape_sparql_uri(uri) == "http://dbpedia.org/resource/New%20York"

    def test_angle_brackets(self):
        uri = "http://example.org/<test>"
        assert "%3C" in escape_sparql_uri(uri)
        assert "%3E" in escape_sparql_uri(uri)

    def test_double_quote_in_uri(self):
        uri = 'http://example.org/"test"'
        assert "%22" in escape_sparql_uri(uri)

    def test_curly_braces(self):
        uri = "http://example.org/{test}"
        assert "%7B" in escape_sparql_uri(uri)
        assert "%7D" in escape_sparql_uri(uri)

    def test_apostrophe_not_escaped(self):
        # apostrophe (') IS now escaped to %27 by escape_sparql_uri
        uri = "http://dbpedia.org/resource/Aren't_You_Kind_of_Glad_We_Did?"
        escaped = escape_sparql_uri(uri)
        assert "'" not in escaped  # apostrophe is percent-encoded
        assert "%27" in escaped

    def test_question_mark_not_escaped(self):
        # ? IS now escaped to %3F by escape_sparql_uri
        uri = "http://dbpedia.org/resource/Aren't_You_Kind_of_Glad_We_Did?"
        escaped = escape_sparql_uri(uri)
        assert "?" not in escaped  # question mark is percent-encoded
        assert "%3F" in escaped

    def test_percent_encoded_uri_passthrough(self):
        # Already percent-encoded URIs should pass through unchanged
        uri = "http://dbpedia.org/resource/Aren%27t_You_Kind_of_Glad_We_Did%3F"
        assert escape_sparql_uri(uri) == uri


# ── Unit tests: mention extraction (no network) ────────────────────────


class TestExtractMentions:
    """Test spaCy-based mention extraction with special character edge cases."""

    def test_apostrophe_in_title_without_keyword(self):
        """Without a title keyword, unquoted contraction entities are missed
        by spaCy. This is a known limitation — spaCy can't distinguish
        'Aren't You Kind Of Glad We Did?' from regular English."""
        q = "Who wrote Aren't You Kind Of Glad We Did?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        # No keyword ("song", "film", etc.) → spaCy sees no entity
        assert isinstance(mentions, list)

    def test_apostrophe_title_after_keyword(self):
        """With a title keyword, the entity is extracted via capitalization."""
        q = "Where did the composer of song Aren'T You Kind Of Glad We Did? die?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        assert any("glad" in m.lower() for m in mentions), \
               f"Expected song title in mentions, got: {mentions}"

    def test_quoted_entity(self):
        q = 'Who directed "Aren\'t You Kind Of Glad We Did?"'
        mentions = DBpediaEntityLinker._extract_mentions(q)
        # Quoted strings are captured by the regex
        assert any("Aren't" in m for m in mentions), \
               f"Expected quoted entity in mentions, got: {mentions}"

    def test_hyphenated_entity(self):
        q = "Who is Spider-Man's alter ego?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        assert any("spider" in m.lower() for m in mentions), \
               f"Expected Spider-Man in mentions, got: {mentions}"

    def test_ampersand_entity(self):
        q = "When did Simon & Garfunkel first perform?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        mention_text = " ".join(mentions).lower()
        assert "simon" in mention_text or "garfunkel" in mention_text, \
               f"Expected Simon/Garfunkel in mentions, got: {mentions}"

    def test_accented_name(self):
        q = "Where was Beyoncé born?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        assert any("beyoncé" in m.lower() or "beyonce" in m.lower()
                    for m in mentions), \
               f"Expected Beyoncé in mentions, got: {mentions}"

    def test_period_in_name(self):
        q = "What is Dr. Dre's real name?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        assert any("dr" in m.lower() for m in mentions), \
               f"Expected Dr. Dre in mentions, got: {mentions}"

    def test_exclamation_in_name(self):
        q = "When was Yahoo! founded?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        assert any("yahoo" in m.lower() for m in mentions), \
               f"Expected Yahoo! in mentions, got: {mentions}"

    def test_colon_in_title(self):
        q = "Who directed Star Wars: Episode IV?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        mention_text = " ".join(mentions).lower()
        assert "star wars" in mention_text or "star" in mention_text, \
               f"Expected Star Wars in mentions, got: {mentions}"

    def test_parenthesized_disambiguation(self):
        q = "Who is the main character in 1984 (novel)?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        assert any("1984" in m for m in mentions), \
               f"Expected 1984 in mentions, got: {mentions}"

    def test_roman_numerals(self):
        q = "When was Henry VIII born?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        assert any("henry" in m.lower() for m in mentions), \
               f"Expected Henry VIII in mentions, got: {mentions}"

    def test_number_entity(self):
        q = "Where is 50 Cent from?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        assert any("50 cent" in m.lower() or "cent" in m.lower()
                    for m in mentions), \
               f"Expected 50 Cent in mentions, got: {mentions}"

    def test_multiple_special_entities(self):
        q = 'Are "Aren\'t You Kind Of Glad We Did?" and "Let\'s Call the Whole Thing Off" by the same composer?'
        mentions = DBpediaEntityLinker._extract_mentions(q)
        # Both quoted entities should be captured
        assert len(mentions) >= 2, \
               f"Expected >= 2 mentions, got {len(mentions)}: {mentions}"

    def test_comma_in_location(self):
        q = "What is the population of Austin, Texas?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        mention_text = " ".join(mentions).lower()
        assert "austin" in mention_text or "texas" in mention_text, \
               f"Expected Austin/Texas in mentions, got: {mentions}"

    def test_slash_in_name(self):
        q = "Who created AC/DC?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        assert any("ac/dc" in m.lower() or "ac" in m.lower()
                    for m in mentions), \
               f"Expected AC/DC in mentions, got: {mentions}"

    def test_very_long_entity_name(self):
        q = "Who wrote The Hitchhiker's Guide to the Galaxy?"
        mentions = DBpediaEntityLinker._extract_mentions(q)
        mention_text = " ".join(mentions).lower()
        assert "hitchhiker" in mention_text or "galaxy" in mention_text, \
               f"Expected Hitchhiker's Guide in mentions, got: {mentions}"

    def test_empty_question(self):
        mentions = DBpediaEntityLinker._extract_mentions("")
        assert mentions == []

    def test_no_entities(self):
        mentions = DBpediaEntityLinker._extract_mentions("What is the meaning of life?")
        # Should not crash, may return empty or "life"
        assert isinstance(mentions, list)


# ── Integration tests: full pipeline (needs network) ───────────────────


@pytest.mark.integration
class TestEntityLinkingIntegration:
    """Full pipeline tests that hit Spotlight + SPARQL endpoints.

    Run with: pytest -m integration tests/test_entity_linking_edge_cases.py
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.linker = DBpediaEntityLinker()

    def _assert_any_linked(self, entities, expected_fragment):
        """Assert at least one entity URI/label contains the expected fragment."""
        found = any(
            expected_fragment.lower() in e.uri.lower()
            or expected_fragment.lower() in e.label.lower()
            for e in entities
        )
        assert found, (
            f"Expected '{expected_fragment}' in entities, got: "
            f"{[(e.label, e.uri) for e in entities]}"
        )

    def test_apostrophe_entity_without_keyword(self):
        """Without title keyword, unquoted contraction entity is NOT linked.
        This documents a known limitation — no keyword means no extraction."""
        entities = self.linker.link_entities(
            "Who wrote Aren't You Kind Of Glad We Did?",
            use_cache=False,
        )
        # Without "song" keyword, this links garbage (You, We) or nothing.
        # The fix requires either quoting or a title keyword.
        assert isinstance(entities, list)

    def test_apostrophe_entity_with_keyword(self):
        """Actual benchmark question: 'song' keyword enables extraction."""
        entities = self.linker.link_entities(
            "Where did the composer of song Aren'T You Kind Of Glad We Did? die?",
            use_cache=False,
        )
        self._assert_any_linked(entities, "Glad")

    def test_simple_entity(self):
        """Baseline: common entity should always work."""
        entities = self.linker.link_entities(
            "Who directed Inception?",
            use_cache=False,
        )
        self._assert_any_linked(entities, "Inception")

    def test_hyphenated_entity(self):
        """Spider-Man — hyphen in entity name."""
        entities = self.linker.link_entities(
            "When was Spider-Man first published?",
            use_cache=False,
        )
        self._assert_any_linked(entities, "Spider")

    def test_accented_entity(self):
        """Beyoncé — accented character."""
        entities = self.linker.link_entities(
            "Where was Beyoncé born?",
            use_cache=False,
        )
        self._assert_any_linked(entities, "Beyonc")

    def test_ampersand_entity(self):
        """Simon & Garfunkel — ampersand."""
        entities = self.linker.link_entities(
            "When did Simon & Garfunkel first perform?",
            use_cache=False,
        )
        assert len(entities) > 0, "No entities linked for ampersand name"

    def test_colon_in_title(self):
        """Star Wars: A New Hope — colon."""
        entities = self.linker.link_entities(
            "Who directed Star Wars: A New Hope?",
            use_cache=False,
        )
        self._assert_any_linked(entities, "Star_Wars")

    def test_exclamation_mark(self):
        """Yahoo! — exclamation."""
        entities = self.linker.link_entities(
            "When was Yahoo! founded?",
            use_cache=False,
        )
        self._assert_any_linked(entities, "Yahoo")

    def test_period_in_name(self):
        """Dr. Dre — period."""
        entities = self.linker.link_entities(
            "What is Dr. Dre's real name?",
            use_cache=False,
        )
        self._assert_any_linked(entities, "Dre")

    def test_quoted_song_title(self):
        """Double-quoted entity with apostrophe."""
        entities = self.linker.link_entities(
            'Who composed "Aren\'t You Kind Of Glad We Did?"',
            use_cache=False,
        )
        assert len(entities) > 0, "No entities linked for quoted title"

    def test_multiple_tricky_entities(self):
        """Two entities, both with special characters."""
        entities = self.linker.link_entities(
            "Did Beyoncé perform at the O2 Arena?",
            use_cache=False,
        )
        assert len(entities) >= 1, f"Expected >= 1 entity, got: {entities}"

    def test_very_long_entity(self):
        """Long multi-word title."""
        entities = self.linker.link_entities(
            "Who wrote The Hitchhiker's Guide to the Galaxy?",
            use_cache=False,
        )
        assert len(entities) > 0, "No entities linked for long title"

    def test_numeric_entity(self):
        """Entity starting with a number."""
        entities = self.linker.link_entities(
            "What genre is the film 2001: A Space Odyssey?",
            use_cache=False,
        )
        self._assert_any_linked(entities, "2001")

    def test_non_latin_chars_in_entity(self):
        """Entity with non-Latin characters."""
        entities = self.linker.link_entities(
            "Where is São Paulo located?",
            use_cache=False,
        )
        self._assert_any_linked(entities, "Paulo")

    def test_parenthesized_entity_in_question(self):
        """Entity with disambiguation parentheses."""
        entities = self.linker.link_entities(
            "Who stars in the film Avatar (2009 film)?",
            use_cache=False,
        )
        self._assert_any_linked(entities, "Avatar")

    def test_possessive_entity(self):
        """Entity followed by 's."""
        entities = self.linker.link_entities(
            "What was Shakespeare's birthplace?",
            use_cache=False,
        )
        self._assert_any_linked(entities, "Shakespeare")

    def test_comparison_with_special_chars(self):
        """Comparison question with special character entities."""
        entities = self.linker.link_entities(
            "Are Spider-Man and Dr. Strange in the same universe?",
            use_cache=False,
        )
        assert len(entities) >= 2, (
            f"Expected >= 2 entities for comparison, got: "
            f"{[(e.label, e.uri) for e in entities]}"
        )
