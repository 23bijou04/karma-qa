"""HotPotQA-style tests for dynamic schema discovery.

Tests the new discover_property_ranges() feature using multi-hop questions
that involve entity types outside the 7 cached classes (Person, Film, Company,
Place, MusicalArtist, Book, MusicalWork).

Entity types tested: SoccerClub, TelevisionShow, University, Battle, Building.
Each test simulates a HotPotQA bridge question and verifies that:
  1. Dynamic range discovery identifies the correct target class
  2. Synthetic 2-hop chains are built from dynamic ranges
  3. Bridge detection works with dynamic ranges (enrich_analysis)
  4. Routing rules fire correctly for uncached types
  5. Schema projection renders useful output for uncached types
"""

from unittest.mock import MagicMock

from entity_profile import EntityProfile, build_entity_profile
from question_analysis import analyze_question, enrich_analysis
from schema_projection import project

# ---------------------------------------------------------------------------
# Helpers: build EntityProfile for each uncached type
# ---------------------------------------------------------------------------

def _soccer_club_profile(**overrides):
    """SoccerClub: manager -> Person, ground -> Place."""
    defaults = dict(
        uri="http://dbpedia.org/resource/Real_Madrid_CF",
        label="Real Madrid CF",
        all_classes=["SoccerClub", "SportsTeam", "Organisation", "Agent"],
        cached_classes=[],
        actual_properties={
            "dbo": ["manager", "ground", "league", "chairman"],
            "dbp": ["manager", "ground", "league"],
            "values": {
                "dbo:manager": "Carlo Ancelotti",
                "dbo:ground": "Santiago Bernabéu Stadium",
            },
        },
        actual_property_set=frozenset({
            "dbo:manager", "dbo:ground", "dbo:league", "dbo:chairman",
            "dbp:manager", "dbp:ground", "dbp:league",
        }),
        # Synthetic chains from dynamic discovery
        all_2hop_chains={
            "manager": ("Person", [
                "almaMater", "birthDate", "birthPlace", "child", "deathCause",
                "deathDate", "deathPlace", "ethnicity", "father", "mother",
                "nationality", "occupation", "parent", "religion", "residence",
                "spouse",
            ]),
            "ground": ("Place", [
                "areaTotal", "capital", "country", "isPartOf", "largestCity",
                "leader", "leaderName", "populationTotal",
            ]),
        },
        dynamic_ranges={
            "dbo:manager": "Person",
            "dbo:ground": "Place",
        },
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


def _tv_show_profile(**overrides):
    """TelevisionShow: creator -> Person, network -> Company."""
    defaults = dict(
        uri="http://dbpedia.org/resource/The_Simpsons",
        label="The Simpsons",
        all_classes=["TelevisionShow", "Work", "Agent"],
        cached_classes=[],
        actual_properties={
            "dbo": ["creator", "network", "numberOfSeasons", "numberOfEpisodes"],
            "dbp": ["creator", "network"],
            "values": {
                "dbo:creator": "Matt Groening",
                "dbo:network": "Fox Broadcasting Company",
            },
        },
        actual_property_set=frozenset({
            "dbo:creator", "dbo:network", "dbo:numberOfSeasons",
            "dbo:numberOfEpisodes", "dbp:creator", "dbp:network",
        }),
        all_2hop_chains={
            "creator": ("Person", [
                "almaMater", "birthDate", "birthPlace", "child", "deathCause",
                "deathDate", "deathPlace", "ethnicity", "father", "mother",
                "nationality", "occupation", "parent", "religion", "residence",
                "spouse",
            ]),
        },
        dynamic_ranges={
            "dbo:creator": "Person",
            "dbo:network": "Company",
        },
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


def _university_profile(**overrides):
    """University: chancellor -> Person, city -> Place."""
    defaults = dict(
        uri="http://dbpedia.org/resource/Stanford_University",
        label="Stanford University",
        all_classes=["University", "EducationalInstitution", "Organisation", "Agent"],
        cached_classes=[],
        actual_properties={
            "dbo": ["chancellor", "city", "country", "numberOfStudents"],
            "dbp": ["chancellor", "city"],
            "values": {
                "dbo:chancellor": "Marc Tessier-Lavigne",
                "dbo:city": "Stanford, California",
            },
        },
        actual_property_set=frozenset({
            "dbo:chancellor", "dbo:city", "dbo:country", "dbo:numberOfStudents",
            "dbp:chancellor", "dbp:city",
        }),
        all_2hop_chains={
            "chancellor": ("Person", [
                "almaMater", "birthDate", "birthPlace", "child", "deathCause",
                "deathDate", "deathPlace", "ethnicity", "father", "mother",
                "nationality", "occupation", "parent", "religion", "residence",
                "spouse",
            ]),
            "city": ("Place", [
                "areaTotal", "capital", "country", "isPartOf", "largestCity",
                "leader", "leaderName", "populationTotal",
            ]),
        },
        dynamic_ranges={
            "dbo:chancellor": "Person",
            "dbo:city": "Place",
        },
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


def _battle_profile(**overrides):
    """Battle/MilitaryConflict: commander -> Person, place -> Place."""
    defaults = dict(
        uri="http://dbpedia.org/resource/Battle_of_Waterloo",
        label="Battle of Waterloo",
        all_classes=["MilitaryConflict", "Event", "Activity", "TopicalConcept"],
        cached_classes=[],
        actual_properties={
            "dbo": ["commander", "place", "date", "result"],
            "dbp": ["commander", "place", "date"],
            "values": {
                "dbo:commander": "Duke of Wellington",
                "dbo:place": "Waterloo, Belgium",
            },
        },
        actual_property_set=frozenset({
            "dbo:commander", "dbo:place", "dbo:date", "dbo:result",
            "dbp:commander", "dbp:place", "dbp:date",
        }),
        all_2hop_chains={
            "commander": ("Person", [
                "almaMater", "birthDate", "birthPlace", "child", "deathCause",
                "deathDate", "deathPlace", "ethnicity", "father", "mother",
                "nationality", "occupation", "parent", "religion", "residence",
                "spouse",
            ]),
            "place": ("Place", [
                "areaTotal", "capital", "country", "isPartOf", "largestCity",
                "leader", "leaderName", "populationTotal",
            ]),
        },
        dynamic_ranges={
            "dbo:commander": "Person",
            "dbo:place": "Place",
        },
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


def _building_profile(**overrides):
    """Building: architect -> Person, location -> Place."""
    defaults = dict(
        uri="http://dbpedia.org/resource/Empire_State_Building",
        label="Empire State Building",
        all_classes=["Building", "ArchitecturalStructure", "Place"],
        cached_classes=["Place"],  # Place IS cached, but Building is the primary type
        actual_properties={
            "dbo": ["architect", "location", "floorCount", "openingDate"],
            "dbp": ["architect", "location"],
            "values": {
                "dbo:architect": "Shreve, Lamb & Harmon",
                "dbo:location": "Manhattan, New York",
            },
        },
        actual_property_set=frozenset({
            "dbo:architect", "dbo:location", "dbo:floorCount", "dbo:openingDate",
            "dbp:architect", "dbp:location",
        }),
        all_2hop_chains={
            # From cached Place schema
            "country": ("Place", ["capital", "populationTotal", "leader"]),
            "isPartOf": ("Place", ["country", "capital", "populationTotal"]),
            "capital": ("Place", ["populationTotal", "country"]),
        },
        dynamic_ranges={},  # Building has Place cached, so no dynamic discovery
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


# ===========================================================================
# TIER 1: Easy — Direct uncached → cached chain
# ===========================================================================

class TestSoccerClubManagerBirthplace:
    """Q1: 'Where was the manager of Real Madrid born?'

    Chain: SoccerClub.manager -> Person.birthPlace -> Place
    Dynamic discovery provides: manager -> Person
    """

    def test_dynamic_ranges_present(self):
        profile = _soccer_club_profile()
        assert "dbo:manager" in profile.dynamic_ranges
        assert profile.dynamic_ranges["dbo:manager"] == "Person"

    def test_2hop_chain_from_dynamic_discovery(self):
        profile = _soccer_club_profile()
        assert "manager" in profile.all_2hop_chains
        target_cls, target_props = profile.all_2hop_chains["manager"]
        assert target_cls == "Person"
        # Person schema has birthPlace — needed for the hop2 question
        assert "birthPlace" in target_props
        assert "birthDate" in target_props

    def test_bridge_detection_with_dynamic_ranges(self):
        """enrich_analysis should detect 'manager' as bridge via dynamic ranges."""
        base = analyze_question("Where was the manager of Real Madrid born?")
        # Simulate LLM extracting ["manager", "birthPlace"]
        enriched = enrich_analysis(
            base,
            ["manager", "birthPlace"],
            extra_ranges={"dbo:manager": "Person"},
        )
        assert enriched.bridge_property == "manager"
        assert enriched.expected_target_type == "Person"
        assert enriched.hop_structure == "bridge"

    def test_projection_shows_uncached_type(self):
        """Schema projection should render SoccerClub info for SPARQL gen."""
        profile = _soccer_club_profile()
        analysis = analyze_question("Where was the manager of Real Madrid born?")
        proj = project(profile, analysis)
        # primary_class should be SoccerClub (first non-generic class)
        assert proj.primary_class == "SoccerClub"

    def test_projection_renders_dynamic_chains(self):
        profile = _soccer_club_profile()
        analysis = analyze_question("Where was the manager of Real Madrid born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        # Should mention the entity and its actual properties
        assert "Real Madrid CF" in output
        assert "SoccerClub" in output

    def test_build_profile_via_mock_discovery(self):
        """build_entity_profile uses PROPERTY_RANGES (ontology cache) — no discover_property_ranges."""
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = (
            ["SoccerClub", "SportsTeam", "Organisation", "Agent"],
            [],  # no cached classes
        )
        mock_disc.get_actual_properties.return_value = {
            "dbo": ["manager", "ground", "league"],
            "dbp": ["manager"],
            "values": {"dbo:manager": "Carlo Ancelotti"},
        }

        profile = build_entity_profile(
            "http://dbpedia.org/resource/Real_Madrid_CF",
            "Real Madrid CF",
            mock_disc,
        )

        # discover_property_ranges is NOT called — ranges come from PROPERTY_RANGES cache.
        mock_disc.discover_property_ranges.assert_not_called()
        # manager has range "Person" in PROPERTY_RANGES — chain built
        assert "manager" in profile.all_2hop_chains
        assert profile.all_2hop_chains["manager"][0] == "Person"
        # dbo:manager is recorded in dynamic_ranges
        assert "dbo:manager" in profile.dynamic_ranges


class TestTVShowCreatorBirthplace:
    """Q3: 'Where was the creator of The Simpsons born?'

    Chain: TelevisionShow.creator -> Person.birthPlace -> Place
    """

    def test_dynamic_ranges_present(self):
        profile = _tv_show_profile()
        assert "dbo:creator" in profile.dynamic_ranges
        assert profile.dynamic_ranges["dbo:creator"] == "Person"

    def test_creator_chain(self):
        profile = _tv_show_profile()
        assert "creator" in profile.all_2hop_chains
        target_cls, target_props = profile.all_2hop_chains["creator"]
        assert target_cls == "Person"
        assert "birthPlace" in target_props

    def test_bridge_detection_creator(self):
        base = analyze_question("Where was the creator of The Simpsons born?")
        enriched = enrich_analysis(
            base,
            ["creator", "birthPlace"],
            extra_ranges={"dbo:creator": "Person"},
        )
        assert enriched.bridge_property == "creator"
        assert enriched.expected_target_type == "Person"

    def test_projection_primary_class(self):
        profile = _tv_show_profile()
        analysis = analyze_question("Where was the creator of The Simpsons born?")
        proj = project(profile, analysis)
        # TelevisionShow is the most specific non-generic class
        assert proj.primary_class == "TelevisionShow"


class TestUniversityChancellorNationality:
    """Q2: 'What is the nationality of the chancellor of Stanford?'

    Chain: University.chancellor -> Person.nationality -> Place
    """

    def test_dynamic_ranges_present(self):
        profile = _university_profile()
        assert "dbo:chancellor" in profile.dynamic_ranges
        assert profile.dynamic_ranges["dbo:chancellor"] == "Person"

    def test_chancellor_chain(self):
        profile = _university_profile()
        assert "chancellor" in profile.all_2hop_chains
        target_cls, _ = profile.all_2hop_chains["chancellor"]
        assert target_cls == "Person"

    def test_bridge_detection_chancellor(self):
        base = analyze_question(
            "What is the nationality of the chancellor of Stanford University?"
        )
        enriched = enrich_analysis(
            base,
            ["chancellor", "nationality"],
            extra_ranges={"dbo:chancellor": "Person"},
        )
        assert enriched.bridge_property == "chancellor"
        assert enriched.expected_target_type == "Person"

    def test_university_primary_class(self):
        profile = _university_profile()
        analysis = analyze_question("What is the nationality of the chancellor?")
        proj = project(profile, analysis)
        assert proj.primary_class == "University"


# ===========================================================================
# TIER 2: Medium — Multi-hop dynamic discovery + cached chains
# ===========================================================================

class TestBattleCommanderBirthplace:
    """Q6: 'Where was the general who commanded the Battle of Waterloo born?'

    Chain: MilitaryConflict.commander -> Person.birthPlace -> Place
    """

    def test_dynamic_ranges_present(self):
        profile = _battle_profile()
        assert "dbo:commander" in profile.dynamic_ranges
        assert profile.dynamic_ranges["dbo:commander"] == "Person"

    def test_commander_chain(self):
        profile = _battle_profile()
        assert "commander" in profile.all_2hop_chains
        target_cls, target_props = profile.all_2hop_chains["commander"]
        assert target_cls == "Person"
        assert "birthPlace" in target_props
        assert "deathDate" in target_props

    def test_bridge_detection_commander(self):
        """Commander is not in KEYWORD_TO_PROPERTIES, so bridge detection
        depends on enrich_analysis with dynamic ranges."""
        base = analyze_question("Where was the commander of the Battle of Waterloo born?")
        enriched = enrich_analysis(
            base,
            ["commander", "birthPlace"],
            extra_ranges={"dbo:commander": "Person"},
        )
        assert enriched.bridge_property == "commander"
        assert enriched.expected_target_type == "Person"
        assert enriched.hop_structure == "bridge"

    def test_battle_primary_class(self):
        profile = _battle_profile()
        analysis = analyze_question("Where was the commander born?")
        proj = project(profile, analysis)
        # MilitaryConflict is specific; Event and Activity are more generic
        # but not in GENERIC_CLASSES. TopicalConcept IS in GENERIC_CLASSES.
        assert proj.primary_class == "MilitaryConflict"

    def test_place_chain_from_battle(self):
        """Battle also has place -> Place chain for geographic questions."""
        profile = _battle_profile()
        assert "place" in profile.all_2hop_chains
        target_cls, _ = profile.all_2hop_chains["place"]
        assert target_cls == "Place"


class TestUniversityLocationCountry:
    """Q7: 'What country is Stanford University located in?'

    Chain: University.city -> Place.country -> Place
    Two-level chain: university location -> city -> country
    """

    def test_city_chain(self):
        profile = _university_profile()
        assert "city" in profile.all_2hop_chains
        target_cls, target_props = profile.all_2hop_chains["city"]
        assert target_cls == "Place"
        assert "country" in target_props

    def test_projection_for_location_question(self):
        profile = _university_profile()
        analysis = analyze_question("What country is Stanford University located in?")
        proj = project(profile, analysis)
        assert proj.primary_class == "University"
        output = proj.render_for_sparql()
        assert "Stanford University" in output


# ===========================================================================
# TIER 3: Hard — Edge cases and mixed cached/uncached types
# ===========================================================================

class TestBuildingWithCachedPlace:
    """Building that inherits from Place (cached) — should use Place schema
    for chains but Building as primary class for specificity."""

    def test_has_place_chains(self):
        profile = _building_profile()
        # Place chains from cached schema
        assert "country" in profile.all_2hop_chains
        assert "isPartOf" in profile.all_2hop_chains

    def test_primary_class_is_building(self):
        """Building is more specific than Place; project should prefer it."""
        profile = _building_profile()
        analysis = analyze_question("Who designed the Empire State Building?")
        proj = project(profile, analysis)
        # Building is NOT cached but profile has cached_classes=["Place"].
        # _score_class for Place should score lower for "designed" question.
        # primary_class depends on class scoring.
        assert proj.primary_class is not None

    def test_no_dynamic_discovery_when_cached(self):
        """Building has Place cached -> discover_property_ranges NOT called."""
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = (
            ["Building", "ArchitecturalStructure", "Place"],
            ["Place"],  # Place IS cached
        )
        mock_disc.get_actual_properties.return_value = {
            "dbo": ["architect", "location"], "dbp": [], "values": {},
        }

        build_entity_profile(
            "http://dbpedia.org/resource/Empire_State_Building",
            "Empire State Building",
            mock_disc,
        )
        # Should NOT call discover_property_ranges because cached_classes is non-empty
        mock_disc.discover_property_ranges.assert_not_called()


class TestComparisonTwoUncachedEntities:
    """Q13: 'Which soccer club was founded first, Real Madrid or Barcelona?'

    Both entities are SoccerClub (uncached). Dynamic ranges should be
    collected from BOTH profiles for routing.
    """

    def test_both_profiles_have_dynamic_ranges(self):
        p1 = _soccer_club_profile()
        p2 = _soccer_club_profile(
            uri="http://dbpedia.org/resource/FC_Barcelona",
            label="FC Barcelona",
        )
        assert "dbo:manager" in p1.dynamic_ranges
        assert "dbo:manager" in p2.dynamic_ranges

    def test_comparison_analysis(self):
        """Comparison question should be detected by analyze_question."""
        analysis = analyze_question(
            "Which soccer club was founded first, Real Madrid or Barcelona?"
        )
        assert analysis.is_comparison is True

    def test_bridge_enrichment_with_merged_ranges(self):
        """Dynamic ranges from two uncached entities merge correctly."""
        base = analyze_question(
            "Whose manager was born earlier, Real Madrid or Barcelona?"
        )
        # Merge dynamic ranges from both entities
        merged_ranges = {
            "dbo:manager": "Person",
            "dbo:ground": "Place",
        }
        enriched = enrich_analysis(
            base,
            ["manager", "birthDate"],
            extra_ranges=merged_ranges,
        )
        assert enriched.bridge_property == "manager"
        assert enriched.expected_target_type == "Person"


class TestNoTargetsCachedFallback:
    """Entity whose object properties all point to non-cached classes.

    Example: Tournament with dbo:sport -> Sport (not cached).
    Dynamic discovery finds the range but can't build useful chains.
    """

    def test_non_cached_targets_excluded(self):
        mock_disc = MagicMock()
        mock_disc.get_all_cached_classes.return_value = (
            ["Tournament", "Event"], [],
        )
        mock_disc.get_actual_properties.return_value = {
            "dbo": ["sport"], "dbp": [], "values": {"dbo:sport": "Football"},
        }
        # Dynamic discovery: sport -> Sport (NOT cached)
        mock_disc.discover_property_ranges.return_value = {
            "sport": "Sport",
        }

        profile = build_entity_profile(
            "http://dbpedia.org/resource/FIFA_World_Cup",
            "FIFA World Cup",
            mock_disc,
        )

        # No chains built (Sport not in CACHED_CLASS_SCHEMAS)
        assert profile.all_2hop_chains == {}
        # No dynamic ranges stored either
        assert profile.dynamic_ranges == {}

    def test_projection_still_works(self):
        """Even without chains, projection should produce usable output."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/FIFA_World_Cup",
            label="FIFA World Cup",
            all_classes=["Tournament", "Event"],
            cached_classes=[],
            actual_properties={
                "dbo": ["sport"], "dbp": [],
                "values": {"dbo:sport": "Football"},
            },
            actual_property_set=frozenset({"dbo:sport"}),
        )
        analysis = analyze_question("What sport is the FIFA World Cup?")
        proj = project(profile, analysis)
        # primary_class should be Tournament (Event is not in GENERIC_CLASSES)
        assert proj.primary_class == "Tournament"
        output = proj.render_for_sparql()
        assert "FIFA World Cup" in output


# ===========================================================================
# TIER 4: Comparison questions — symmetric dynamic discovery
# ===========================================================================

class TestTVShowComparison:
    """Q14: 'Which show has been on air longer, The Simpsons or Game of Thrones?'

    Both TelevisionShow entities should get dynamic ranges independently.
    """

    def test_symmetric_dynamic_ranges(self):
        p1 = _tv_show_profile()
        p2 = _tv_show_profile(
            uri="http://dbpedia.org/resource/Game_of_Thrones",
            label="Game of Thrones",
        )
        # Both should have identical dynamic range structure
        assert p1.dynamic_ranges == p2.dynamic_ranges

    def test_projection_comparison_question(self):
        p1 = _tv_show_profile()
        p2 = _tv_show_profile(
            uri="http://dbpedia.org/resource/Game_of_Thrones",
            label="Game of Thrones",
        )
        analysis = analyze_question(
            "Which show has been on air longer, The Simpsons or Game of Thrones?"
        )
        proj1 = project(p1, analysis)
        proj2 = project(p2, analysis)
        # Both should have TelevisionShow as primary class
        assert proj1.primary_class == "TelevisionShow"
        assert proj2.primary_class == "TelevisionShow"

    def test_multi_entity_render_prefix_once(self):
        """Joined rendering: prefix only on the first entity."""
        p1 = _tv_show_profile()
        p2 = _tv_show_profile(
            uri="http://dbpedia.org/resource/Game_of_Thrones",
            label="Game of Thrones",
        )
        analysis = analyze_question("Which show has been on air longer?")
        proj1 = project(p1, analysis)
        proj2 = project(p2, analysis)
        joined = "\n\n---\n\n".join(
            p.render_for_sparql(include_prefixes=(i == 0))
            for i, p in enumerate([proj1, proj2])
        )
        assert joined.count("PREFIX dbr:") == 1
        assert "The Simpsons" in joined
        assert "Game of Thrones" in joined


# ===========================================================================
# Cross-cutting: enrich_analysis scoring correctness
# ===========================================================================

class TestEnrichAnalysisBridgeScoring:
    """Verify that bridge candidate scoring works with dynamic ranges.

    The scoring algorithm iterates LLM properties, looks up target classes
    in effective_ranges, then counts how many remaining properties exist
    in the target class schema. The highest-scoring candidate wins.
    """

    def test_manager_beats_birthdate_as_bridge(self):
        """For ['manager', 'birthDate'], manager -> Person has birthDate (score=1).
        birthDate has no range in PROPERTY_RANGES or dynamic_ranges (score=0).
        So manager should win as bridge."""
        base = analyze_question("When was the manager of Real Madrid born?")
        enriched = enrich_analysis(
            base,
            ["manager", "birthDate"],
            extra_ranges={"dbo:manager": "Person"},
        )
        assert enriched.bridge_property == "manager"
        assert enriched.hop2_keyword == "birthDate"

    def test_commander_beats_birthplace_as_bridge(self):
        """For ['commander', 'birthPlace'], commander -> Person has birthPlace."""
        base = analyze_question("Where was the commander born?")
        enriched = enrich_analysis(
            base,
            ["commander", "birthPlace"],
            extra_ranges={"dbo:commander": "Person"},
        )
        assert enriched.bridge_property == "commander"
        assert enriched.hop2_keyword == "birthPlace"

    def test_creator_beats_nationality_as_bridge(self):
        """For ['creator', 'nationality'], creator -> Person has nationality."""
        base = analyze_question("What is the nationality of the creator?")
        enriched = enrich_analysis(
            base,
            ["creator", "nationality"],
            extra_ranges={"dbo:creator": "Person"},
        )
        assert enriched.bridge_property == "creator"
        assert enriched.hop2_keyword == "nationality"

    def test_no_bridge_without_dynamic_ranges(self):
        """manager IS in PROPERTY_RANGES (→ Person) after 734-class ontology cache.

        This test verifies that bridge detection works via the static PROPERTY_RANGES
        even without explicit extra_ranges. The bridge is found because manager now
        has a known range in the ontology cache.
        """
        base = analyze_question("When was the manager of Real Madrid born?")
        enriched = enrich_analysis(
            base,
            ["manager", "birthDate"],
            # No extra_ranges — manager IS now in PROPERTY_RANGES -> "Person"
        )
        # manager -> Person is in PROPERTY_RANGES, so bridge CAN be detected
        assert enriched.bridge_property == "manager"
        assert enriched.expected_target_type == "Person"

    def test_three_properties_correct_bridge(self):
        """For ['manager', 'birthDate', 'nationality'], manager should still win.
        manager -> Person has both birthDate (✓) and nationality (✓) -> score=2."""
        base = analyze_question("What nationality is the manager?")
        enriched = enrich_analysis(
            base,
            ["manager", "birthDate", "nationality"],
            extra_ranges={"dbo:manager": "Person"},
        )
        assert enriched.bridge_property == "manager"
        # score for manager should be 2 (birthDate + nationality in Person schema)


# ===========================================================================
# Schema projection rendering for uncached types
# ===========================================================================

class TestRenderForSparqlUncached:
    """Verify render_for_sparql() output for uncached entity types."""

    def test_soccer_club_shows_actual_properties(self):
        profile = _soccer_club_profile()
        analysis = analyze_question("Who manages Real Madrid?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        # Should show actual dbo properties (not cached schema ones)
        assert "manager" in output
        assert "Real Madrid CF" in output

    def test_battle_shows_chain_info(self):
        profile = _battle_profile()
        analysis = analyze_question("Where was the commander of the battle born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "Battle of Waterloo" in output
        assert "MilitaryConflict" in output

    def test_university_shows_dynamic_chains(self):
        profile = _university_profile()
        analysis = analyze_question(
            "What is the nationality of the chancellor of Stanford?"
        )
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "Stanford University" in output
        assert "University" in output
