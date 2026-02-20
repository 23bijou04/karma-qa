"""Tests for 2-hop chain routing rules and comparison question detection."""

from hybrid_qa import _parse_2hop_chains, _rule_based_route, _is_comparison_question
from dbpedia_qa.models import Entity

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


def _make_entities(*names):
    """Helper: create Entity objects with URIs from names."""
    return [
        Entity(
            uri=f"http://dbpedia.org/resource/{n.replace(' ', '_')}",
            label=n,
            source="test",
        )
        for n in names
    ]


class TestIsComparisonQuestion:
    def test_earlier_with_two_entities(self):
        entities = _make_entities("Film A", "Film B")
        assert _is_comparison_question("Which film's director died earlier, Film A or Film B?", entities)

    def test_older_with_two_entities(self):
        entities = _make_entities("Person A", "Person B")
        assert _is_comparison_question("Who is older, Person A or Person B?", entities)

    def test_same_with_two_entities(self):
        entities = _make_entities("Film A", "Film B")
        assert _is_comparison_question("Are both Film A and Film B from the same country?", entities)

    def test_both_with_two_entities(self):
        entities = _make_entities("X", "Y")
        assert _is_comparison_question("Do both X and Y share the same director?", entities)

    def test_more_recently_with_two_entities(self):
        entities = _make_entities("Film A", "Film B")
        assert _is_comparison_question("Which was released more recently, Film A or Film B?", entities)

    def test_single_entity_returns_false(self):
        entities = _make_entities("Film A")
        assert not _is_comparison_question("Which film was released earlier?", entities)

    def test_no_comparison_words_returns_false(self):
        entities = _make_entities("Film A", "Film B")
        assert not _is_comparison_question("Who directed Film A?", entities)

    def test_empty_entities_returns_false(self):
        assert not _is_comparison_question("Who was born earlier?", [])
