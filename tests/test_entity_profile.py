"""Tests for EntityProfile and PROPERTY_TO_CLASSES."""

from unittest.mock import MagicMock

from dbpedia_qa.dbpedia.schema_discovery import CACHED_CLASS_SCHEMAS, PROPERTY_TO_CLASSES
from entity_profile import EntityProfile, build_entity_profile


class TestPropertyToClasses:
    def test_genre_maps_to_musical_artist(self):
        assert "MusicalArtist" in PROPERTY_TO_CLASSES["genre"]

    def test_birth_date_maps_to_person_and_musical_artist(self):
        classes = PROPERTY_TO_CLASSES["birthDate"]
        assert "Person" in classes
        assert "MusicalArtist" in classes

    def test_director_maps_to_film(self):
        assert "Film" in PROPERTY_TO_CLASSES["director"]

    def test_author_maps_to_book(self):
        assert "Book" in PROPERTY_TO_CLASSES["author"]

    def test_all_cached_class_properties_covered(self):
        """Every property in CACHED_CLASS_SCHEMAS appears in PROPERTY_TO_CLASSES."""
        for cls_name, schema in CACHED_CLASS_SCHEMAS.items():
            for prop in schema.get("dbo", []):
                assert prop in PROPERTY_TO_CLASSES, (
                    f"dbo prop '{prop}' from {cls_name} missing"
                )
                assert cls_name in PROPERTY_TO_CLASSES[prop]


class TestEntityProfileConstruction:
    def test_multi_class_entity(self):
        """Entity with types [Person, MusicalArtist] -> both in cached_classes."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/David_Bowie",
            label="David Bowie",
            all_classes=["Person", "MusicalArtist", "Agent"],
            cached_classes=["Person", "MusicalArtist"],
            actual_properties={"dbo": ["birthDate", "genre"], "dbp": ["birthDate"], "values": {}},
            actual_property_set=frozenset({"dbo:birthDate", "dbo:genre", "dbp:birthDate"}),
        )
        assert "Person" in profile.cached_classes
        assert "MusicalArtist" in profile.cached_classes
        assert len(profile.cached_classes) == 2

    def test_actual_property_set(self):
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/Test",
            label="Test",
            actual_properties={"dbo": ["birthDate"], "dbp": ["genre"], "values": {}},
            actual_property_set=frozenset({"dbo:birthDate", "dbp:genre"}),
        )
        assert "dbo:birthDate" in profile.actual_property_set
        assert "dbp:genre" in profile.actual_property_set

    def test_no_cached_class(self):
        """Entity with types not in CACHED_CLASS_SCHEMAS."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/FooThing",
            label="FooThing",
            all_classes=["SomethingRare"],
            cached_classes=[],
        )
        assert profile.all_classes == ["SomethingRare"]
        assert profile.cached_classes == []
        assert profile.all_2hop_chains == {}


class TestBuildEntityProfile:
    def test_builds_correct_profile(self):
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = (
            ["Person", "Agent"],
            ["Person"],
        )
        mock_disc.get_actual_properties.return_value = {
            "dbo": ["birthDate", "birthPlace"],
            "dbp": ["birthDate"],
            "values": {"dbo:birthDate": "1879-03-14"},
        }

        profile = build_entity_profile(
            "http://dbpedia.org/resource/Albert_Einstein",
            "Albert Einstein",
            mock_disc,
        )

        assert profile.uri == "http://dbpedia.org/resource/Albert_Einstein"
        assert profile.label == "Albert Einstein"
        assert profile.cached_classes == ["Person"]
        assert "dbo:birthDate" in profile.actual_property_set
        assert "dbo:birthPlace" in profile.actual_property_set
        assert "dbp:birthDate" in profile.actual_property_set
        # Person has 2-hop chains
        assert "parent" in profile.all_2hop_chains
        assert "birthPlace" in profile.all_2hop_chains

    def test_merged_2hop_chains(self):
        """Person + MusicalArtist both have birthPlace chain -> merged target props."""
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = (
            ["Person", "MusicalArtist"],
            ["Person", "MusicalArtist"],
        )
        mock_disc.get_actual_properties.return_value = {
            "dbo": ["birthDate", "birthPlace", "genre"],
            "dbp": [],
            "values": {},
        }

        profile = build_entity_profile("http://dbpedia.org/resource/X", "X", mock_disc)

        assert "birthPlace" in profile.all_2hop_chains
        target, props = profile.all_2hop_chains["birthPlace"]
        assert target == "Place"
        # Person birthPlace has: country, isPartOf, populationTotal
        # MusicalArtist birthPlace has: country, isPartOf, populationTotal
        # Union should contain all
        assert "country" in props
        assert "isPartOf" in props
        assert "populationTotal" in props

    def test_no_classes_entity(self):
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = ([], [])
        mock_disc.get_actual_properties.return_value = {
            "dbo": [], "dbp": [], "values": {},
        }

        profile = build_entity_profile("http://dbpedia.org/resource/Unknown", "Unknown", mock_disc)
        assert profile.cached_classes == []
        assert profile.all_2hop_chains == {}
        assert profile.actual_property_set == frozenset()
