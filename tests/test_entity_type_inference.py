"""Tests for Case 9: Spotlight entity disambiguation via relation-inferred type hints."""

from unittest.mock import MagicMock

from dbpedia_qa.dbpedia.entity_linker import (
    DBpediaEntityLinker,
)
from dbpedia_qa.models import Entity


class TestExtractTypeHint:
    def test_relation_type_hint_director_of(self):
        """'director of X' infers Film type."""
        hint = DBpediaEntityLinker._extract_type_hint(
            "Who is the mother of the director of Polish-Russian War?"
        )
        assert hint == "dbo:Film"

    def test_explicit_type_hint_film(self):
        """'film' keyword infers Film type."""
        hint = DBpediaEntityLinker._extract_type_hint("What film won the Oscar?")
        assert hint == "dbo:Film"

    def test_relation_hint_priority_over_explicit(self):
        """Relation hint ('director of') takes priority over explicit ('country')."""
        hint = DBpediaEntityLinker._extract_type_hint(
            "In which country was the director of Inception born?"
        )
        assert hint == "dbo:Film"  # 'director of' -> Film, not 'country'

    def test_author_of_infers_book(self):
        """'author of X' infers Book type."""
        hint = DBpediaEntityLinker._extract_type_hint(
            "When was the author of Harry Potter born?"
        )
        assert hint == "dbo:Book"

    def test_no_hint_returns_none(self):
        """No relevant keywords returns None."""
        hint = DBpediaEntityLinker._extract_type_hint(
            "What is the population of Earth?"
        )
        assert hint is None


class TestValidateEntityTypes:
    def test_replacement_on_type_mismatch(self):
        """Mismatched entity type triggers search fallback and replacement."""
        linker = DBpediaEntityLinker.__new__(DBpediaEntityLinker)
        linker.sparql_client = MagicMock()

        # Spotlight returned historical war (wrong type)
        war_entity = Entity(
            uri="http://dbpedia.org/resource/Polish-Russian_War_(historical)",
            label="Polish-Russian War",
            source="spotlight",
        )

        # Type check returns no Film type
        linker.sparql_client.get_entity_types.return_value = [
            "Event", "MilitaryConflict",
        ]

        # bif:contains fallback finds the film
        film_entity = Entity(
            uri="http://dbpedia.org/resource/Polish-Russian_War_(film)",
            label="Polish-Russian War (film)",
            source="sparql",
        )
        linker.sparql_client.search_entity_by_name.return_value = film_entity

        result = linker._validate_entity_types(
            [war_entity],
            "Who is the mother of the director of Polish-Russian War?",
            "dbo:Film",
        )
        assert len(result) == 1
        assert "film" in result[0].uri.lower()

    def test_keeps_correct_type(self):
        """Entity with correct type is kept as-is."""
        linker = DBpediaEntityLinker.__new__(DBpediaEntityLinker)
        linker.sparql_client = MagicMock()

        film_entity = Entity(
            uri="http://dbpedia.org/resource/Inception",
            label="Inception",
            source="spotlight",
        )

        linker.sparql_client.get_entity_types.return_value = ["Film", "Work"]

        result = linker._validate_entity_types(
            [film_entity],
            "Who is the director of Inception?",
            "dbo:Film",
        )
        assert len(result) == 1
        assert result[0].uri == film_entity.uri

    def test_non_target_entities_preserved(self):
        """Entities not matching the object phrase are kept unchanged."""
        linker = DBpediaEntityLinker.__new__(DBpediaEntityLinker)
        linker.sparql_client = MagicMock()

        person = Entity(
            uri="http://dbpedia.org/resource/Albert_Einstein",
            label="Albert Einstein",
            source="spotlight",
        )
        film = Entity(
            uri="http://dbpedia.org/resource/Polish-Russian_War_(historical)",
            label="Polish-Russian War",
            source="spotlight",
        )

        linker.sparql_client.get_entity_types.return_value = ["Event"]
        linker.sparql_client.search_entity_by_name.return_value = Entity(
            uri="http://dbpedia.org/resource/Polish-Russian_War_(film)",
            label="Polish-Russian War (film)",
            source="sparql",
        )

        result = linker._validate_entity_types(
            [person, film],
            "Who is the director of Polish-Russian War?",
            "dbo:Film",
        )
        assert len(result) == 2
        # Person entity unchanged
        assert result[0].uri == person.uri
        # Film entity replaced
        assert "film" in result[1].uri.lower()


class TestCleanObjPhrase:
    """Tests for Phase 4: noisy obj_phrase cleaning before fallback search."""

    def test_noisy_obj_phrase_cleaned_for_search(self):
        """Long obj_phrase with year/determiner/descriptor is cleaned before search."""
        linker = DBpediaEntityLinker.__new__(DBpediaEntityLinker)
        linker.sparql_client = MagicMock()

        president = Entity(
            uri="http://dbpedia.org/resource/Thomas_Jefferson",
            label="Thomas Jefferson",
            source="spotlight",
        )

        # First call: type check returns Person (not Film)
        # Second call: probe URI type check returns [] (no match)
        linker.sparql_client.get_entity_types.side_effect = [
            ["Person", "Agent"],  # original entity
            [],                    # probe URI miss
        ]

        film_entity = Entity(
            uri="http://dbpedia.org/resource/Thomas_Jefferson_(film)",
            label="Thomas Jefferson (film)",
            source="sparql",
        )
        linker.sparql_client.search_entity_by_name.return_value = film_entity

        linker._validate_entity_types(
            [president],
            "Who directed the 1995 Ken Burns documentary Thomas Jefferson?",
            "dbo:Film",
        )

        # Verify search was called with cleaned query
        call_args = linker.sparql_client.search_entity_by_name.call_args
        search_query = call_args[0][0]
        assert "1995" not in search_query
        assert "documentary" not in search_query.lower()
        assert "Thomas Jefferson" in search_query

    def test_clean_obj_keeps_entity_name(self):
        """Cleaning should preserve the entity name itself."""
        linker = DBpediaEntityLinker.__new__(DBpediaEntityLinker)
        linker.sparql_client = MagicMock()

        entity = Entity(
            uri="http://dbpedia.org/resource/Harry_Potter",
            label="Harry Potter",
            source="spotlight",
        )

        linker.sparql_client.get_entity_types.side_effect = [
            ["Person"],  # original
            [],           # probe miss
        ]
        linker.sparql_client.search_entity_by_name.return_value = Entity(
            uri="http://dbpedia.org/resource/Harry_Potter_(book)",
            label="Harry Potter (book)",
            source="sparql",
        )

        linker._validate_entity_types(
            [entity],
            "Who wrote the Harry Potter series?",
            "dbo:Book",
        )

        call_args = linker.sparql_client.search_entity_by_name.call_args
        search_query = call_args[0][0]
        assert "Harry Potter" in search_query
        assert "series" not in search_query.lower()


class TestDirectUriProbe:
    """Tests for Phase 5: direct URI probe dbr:Name_(type) before text search."""

    def test_direct_uri_probe_finds_film(self):
        """URI probe dbr:Name_(film) should resolve before bif:contains search."""
        linker = DBpediaEntityLinker.__new__(DBpediaEntityLinker)
        linker.sparql_client = MagicMock()

        president = Entity(
            uri="http://dbpedia.org/resource/Thomas_Jefferson",
            label="Thomas Jefferson",
            source="spotlight",
        )

        # First call: type check for president -> Person (not Film)
        # Second call: probe URI type check -> Film (match!)
        linker.sparql_client.get_entity_types.side_effect = [
            ["Person", "Agent"],  # original entity types
            ["Film", "Work"],     # probe URI types -- HIT
        ]
        # _probe_single_uri also checks existence and disambiguation
        linker.sparql_client.ask_entity_exists.return_value = True
        linker.sparql_client.validate_entity_uri.return_value = {
            "is_disambiguation": False,
            "redirect_target": None,
        }

        result = linker._validate_entity_types(
            [president],
            "Who directed Thomas Jefferson?",
            "dbo:Film",
        )

        assert len(result) == 1
        assert result[0].uri == "http://dbpedia.org/resource/Thomas_Jefferson_(film)"
        assert result[0].source == "type_corrected"
        # search_entity_by_name should NOT have been called (probe hit first)
        linker.sparql_client.search_entity_by_name.assert_not_called()

    def test_direct_uri_probe_miss_falls_to_search(self):
        """When probe URI has wrong type, falls through to bif:contains search."""
        linker = DBpediaEntityLinker.__new__(DBpediaEntityLinker)
        linker.sparql_client = MagicMock()

        entity = Entity(
            uri="http://dbpedia.org/resource/Some_Name",
            label="Some Name",
            source="spotlight",
        )

        # First call: type check -> Person; Second: probe -> empty
        linker.sparql_client.get_entity_types.side_effect = [
            ["Person"],  # original entity
            [],          # probe URI doesn't exist
        ]

        fallback = Entity(
            uri="http://dbpedia.org/resource/Some_Name_(film)",
            label="Some Name (film)",
            source="sparql",
        )
        linker.sparql_client.search_entity_by_name.return_value = fallback

        result = linker._validate_entity_types(
            [entity],
            "Who directed Some Name?",
            "dbo:Film",
        )

        # Probe missed, should have fallen through to search
        linker.sparql_client.search_entity_by_name.assert_called_once()
        assert result[0].uri == fallback.uri
