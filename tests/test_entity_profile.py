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
            "dbo": ["birthDate", "birthPlace", "parent"],
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
        # Instance-filtered: only chains for properties entity actually has
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
        mock_disc.discover_property_ranges.return_value = {}

        profile = build_entity_profile("http://dbpedia.org/resource/Unknown", "Unknown", mock_disc)
        assert profile.cached_classes == []
        assert profile.all_2hop_chains == {}
        assert profile.actual_property_set == frozenset()

    def test_dynamic_schema_fallback_for_uncached_type(self):
        """Uncached types get synthetic 2-hop chains via PROPERTY_RANGES lookup."""
        mock_disc = MagicMock()
        # Simulate entity with no cached classes (truly unknown type)
        mock_disc.get_all_cached_classes.return_value = (
            ["SoccerClub", "SportsTeam", "Organisation", "Agent"],
            [],  # no cached classes
        )
        mock_disc.get_actual_properties.return_value = {
            "dbo": ["manager", "ground", "league"],
            "dbp": ["manager", "ground"],
            "values": {"dbo:manager": "Carlo Ancelotti"},
        }

        profile = build_entity_profile(
            "http://dbpedia.org/resource/Real_Madrid_CF", "Real Madrid CF", mock_disc,
        )
        assert profile.cached_classes == []
        # manager -> Person chain built from PROPERTY_RANGES (Person is cached)
        assert "manager" in profile.all_2hop_chains
        target_cls, target_props = profile.all_2hop_chains["manager"]
        assert target_cls == "Person"
        assert "birthDate" in target_props
        assert "nationality" in target_props
        assert "spouse" in target_props
        # ground -> Place chain built from PROPERTY_RANGES
        assert "ground" in profile.all_2hop_chains
        assert profile.all_2hop_chains["ground"][0] == "Place"
        # league -> SportsLeague: not in PROPERTY_RANGES → no chain
        assert "league" not in profile.all_2hop_chains
        # Dynamic ranges stored on profile
        assert "dbo:manager" in profile.dynamic_ranges
        assert profile.dynamic_ranges["dbo:manager"] == "Person"
        assert "dbo:ground" in profile.dynamic_ranges
        assert profile.dynamic_ranges["dbo:ground"] == "Place"
        # league not in PROPERTY_RANGES → not in dynamic_ranges
        assert "dbo:league" not in profile.dynamic_ranges

    def test_dynamic_ranges_not_called_for_cached_types(self):
        """Entities with cached classes should NOT call discover_property_ranges."""
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = (
            ["Person", "Agent"], ["Person"],
        )
        mock_disc.get_actual_properties.return_value = {
            "dbo": ["birthDate"], "dbp": [], "values": {},
        }

        profile = build_entity_profile("http://dbpedia.org/resource/X", "X", mock_disc)
        # discover_property_ranges should NOT be called for cached types
        mock_disc.discover_property_ranges.assert_not_called()
        # dynamic_ranges should be empty for cached types
        assert profile.dynamic_ranges == {}

    def test_chain_merge_sorts_properties(self):
        """Merged chain properties must be sorted alphabetically."""
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = (
            ["Person", "MusicalArtist"],
            ["Person", "MusicalArtist"],
        )
        mock_disc.get_actual_properties.return_value = {
            "dbo": ["birthPlace"], "dbp": [], "values": {},
        }
        profile = build_entity_profile("http://dbpedia.org/resource/X", "X", mock_disc)
        _, props = profile.all_2hop_chains["birthPlace"]
        assert props == sorted(props)

    def test_chain_merge_preserves_first_target_class(self):
        """When two classes define the same chain, target class from first wins."""
        mock_disc = MagicMock()
        # Person comes first in cached_classes
        mock_disc.get_all_cached_classes.return_value = (
            ["Person", "MusicalArtist"],
            ["Person", "MusicalArtist"],
        )
        mock_disc.get_actual_properties.return_value = {
            "dbo": ["birthPlace"], "dbp": [], "values": {},
        }
        profile = build_entity_profile("http://dbpedia.org/resource/X", "X", mock_disc)
        target, _ = profile.all_2hop_chains["birthPlace"]
        # Both Person and MusicalArtist define birthPlace -> Place
        assert target == "Place"

    def test_build_profile_missing_dbo_key(self):
        """Actual properties dict missing 'dbo'/'dbp' keys should not crash."""
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = ([], [])
        mock_disc.get_actual_properties.return_value = {"values": {}}
        mock_disc.discover_property_ranges.return_value = {}

        profile = build_entity_profile("http://dbpedia.org/resource/X", "X", mock_disc)
        assert profile.actual_property_set == frozenset()

    def test_build_profile_dbp_only(self):
        """Profile with only dbp properties, no dbo."""
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = (["Person"], ["Person"])
        mock_disc.get_actual_properties.return_value = {
            "dbo": [],
            "dbp": ["father", "mother", "birthDate"],
            "values": {"dbp:father": "John", "dbp:mother": "Jane"},
        }
        profile = build_entity_profile("http://dbpedia.org/resource/X", "X", mock_disc)
        assert "dbp:father" in profile.actual_property_set
        assert "dbp:mother" in profile.actual_property_set
        assert "dbp:birthDate" in profile.actual_property_set
        # No dbo properties
        assert not any(p.startswith("dbo:") for p in profile.actual_property_set)

    def test_film_and_book_author_chain_merge(self):
        """Film + Book both define author -> Person chain. Merge union."""
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = (
            ["Film", "Book"],
            ["Film", "Book"],
        )
        mock_disc.get_actual_properties.return_value = {
            "dbo": ["author"], "dbp": [], "values": {},
        }
        profile = build_entity_profile("http://dbpedia.org/resource/X", "X", mock_disc)
        assert "author" in profile.all_2hop_chains
        target, props = profile.all_2hop_chains["author"]
        assert target == "Person"
        # Both Film and Book define: birthDate, deathDate, birthPlace, nationality
        assert "birthDate" in props
        assert "nationality" in props

    def test_person_only_chains_not_duplicated(self):
        """Single cached class should have chains without duplication."""
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = (["Person"], ["Person"])
        mock_disc.get_actual_properties.return_value = {
            "dbo": ["parent"], "dbp": [], "values": {},
        }
        profile = build_entity_profile("http://dbpedia.org/resource/X", "X", mock_disc)
        _, props = profile.all_2hop_chains["parent"]
        assert len(props) == len(set(props)), "Chain props should have no duplicates"


class TestPropertyToClassesDbp:
    """dbo: equivalents of former dbp properties are indexed."""

    def test_father_maps_to_person(self):
        """father (orphan dbo property) appears in PROPERTY_TO_CLASSES."""
        assert "father" in PROPERTY_TO_CLASSES
        assert "Person" in PROPERTY_TO_CLASSES["father"]

    def test_mother_maps_to_person(self):
        assert "mother" in PROPERTY_TO_CLASSES
        assert "Person" in PROPERTY_TO_CLASSES["mother"]

    def test_foundedBy_maps_to_company(self):
        """foundedBy (dbo orphan) is the dbo equivalent of dbp:founder."""
        assert "foundedBy" in PROPERTY_TO_CLASSES
        assert "Company" in PROPERTY_TO_CLASSES["foundedBy"]
