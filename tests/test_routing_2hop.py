"""Tests for 2-hop chain routing rules."""

from hybrid_qa import _parse_2hop_chains, _rule_based_route

SAMPLE_FILM_SCHEMA = """Entity: Inception
URI: <http://dbpedia.org/resource/Inception>
Class: dbo:Film

  dbo: director, producer, starring, releaseDate, runtime
  dbp: director, producer, starring, released
  Property hints:
    - director: film director (returns Person with birthDate, deathDate, birthPlace)
  2-Hop chains:
    - dbo:director -> Person: birthDate, deathDate, birthPlace, nationality
    - dbo:producer -> Person: birthDate, deathDate, birthPlace
    - dbo:starring -> Person: birthDate, deathDate, birthPlace, nationality

Actual properties found:
  dbo:director -> Christopher Nolan
  dbo:releaseDate -> 2010-07-16"""

SAMPLE_PERSON_SCHEMA = """Entity: Albert Einstein
URI: <http://dbpedia.org/resource/Albert_Einstein>
Class: dbo:Person

  dbo: birthDate, deathDate, birthPlace, deathPlace, parent, spouse
  2-Hop chains:
    - dbo:birthPlace -> Place: country, isPartOf, populationTotal

Actual properties found:
  dbo:birthDate -> 1879-03-14
  dbo:birthPlace -> Ulm"""


class TestParse2HopChains:
    def test_parses_film_chains(self):
        chains = _parse_2hop_chains(SAMPLE_FILM_SCHEMA)
        assert "director" in chains
        assert chains["director"][0] == "Person"
        assert "birthDate" in chains["director"][1]

    def test_does_not_match_actual_properties(self):
        """Must NOT parse 'dbo:director -> Christopher Nolan' as a chain."""
        chains = _parse_2hop_chains(SAMPLE_FILM_SCHEMA)
        for prop, (cls, props) in chains.items():
            assert cls != "Christopher", (
                f"Parser matched 'Actual properties' line: {prop} -> {cls}"
            )
        assert len(chains) == 3
        assert set(chains.keys()) == {"director", "producer", "starring"}

    def test_empty_schema(self):
        assert _parse_2hop_chains("") == {}

    def test_schema_without_chains(self):
        assert _parse_2hop_chains("Entity: Foo\nClass: dbo:Thing") == {}


class TestRuleBasedRoute:
    def test_bridge_comparison_director_born(self):
        """'Which director was born earlier' + Film schema -> SPARQL."""
        result = _rule_based_route(
            "Which director of Inception was born earlier?",
            SAMPLE_FILM_SCHEMA,
        )
        assert result == "SPARQL"

    def test_bridge_non_comparison_director_born(self):
        """'When was the director born' + Film schema -> SPARQL."""
        result = _rule_based_route(
            "When was the director of Inception born?",
            SAMPLE_FILM_SCHEMA,
        )
        assert result == "SPARQL"

    def test_direct_date_comparison(self):
        """'Who was born earlier' + Person with birthDate -> SPARQL."""
        result = _rule_based_route(
            "Who was born earlier, Einstein or Newton?",
            SAMPLE_PERSON_SCHEMA,
        )
        assert result == "SPARQL"

    def test_no_relevant_props_returns_none(self):
        """Question about unrelated topic -> None (LLM decides)."""
        result = _rule_based_route(
            "What is the capital of France?",
            SAMPLE_FILM_SCHEMA,
        )
        assert result is None

    def test_bridge_author_comparison(self):
        """'Which author was born earlier' with Book schema -> SPARQL."""
        book_schema = """Entity: Harry Potter
Class: dbo:Book

  dbo: author, publisher, publicationDate
  2-Hop chains:
    - dbo:author -> Person: birthDate, deathDate, birthPlace, nationality

Actual properties found:
  dbo:author -> J. K. Rowling"""
        result = _rule_based_route(
            "Which author was born earlier, the author of Harry Potter or LOTR?",
            book_schema,
        )
        assert result == "SPARQL"
