"""Tests for SchemaProjection and consumer rendering."""

from entity_profile import EntityProfile
from question_analysis import analyze_question
from schema_projection import _score_class, project


def _film_profile(**overrides):
    """Helper: build a Film EntityProfile."""
    defaults = dict(
        uri="http://dbpedia.org/resource/Inception",
        label="Inception",
        all_classes=["Film"],
        cached_classes=["Film"],
        actual_properties={
            "dbo": ["director", "releaseDate"],
            "dbp": ["director"],
            "values": {"dbo:director": "Christopher Nolan", "dbo:releaseDate": "2010-07-16"},
        },
        actual_property_set=frozenset({"dbo:director", "dbo:releaseDate", "dbp:director"}),
        all_2hop_chains={
            "director": ("Person", ["birthDate", "deathDate", "birthPlace", "nationality"]),
        },
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


def _person_profile(**overrides):
    """Helper: build a Person EntityProfile."""
    defaults = dict(
        uri="http://dbpedia.org/resource/Albert_Einstein",
        label="Albert Einstein",
        all_classes=["Person"],
        cached_classes=["Person"],
        actual_properties={
            "dbo": ["birthDate", "birthPlace", "deathDate"],
            "dbp": ["birthDate"],
            "values": {"dbo:birthDate": "1879-03-14", "dbo:birthPlace": "Ulm"},
        },
        actual_property_set=frozenset(
            {"dbo:birthDate", "dbo:birthPlace", "dbo:deathDate", "dbp:birthDate"}
        ),
        all_2hop_chains={
            "birthPlace": ("Place", ["country", "isPartOf", "populationTotal"]),
        },
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


def _book_profile(**overrides):
    """Helper: build a Book EntityProfile."""
    defaults = dict(
        uri="http://dbpedia.org/resource/Harry_Potter_and_the_Philosopher's_Stone",
        label="Harry Potter",
        all_classes=["Book"],
        cached_classes=["Book"],
        actual_properties={
            "dbo": ["author", "publicationDate"],
            "dbp": ["author"],
            "values": {"dbo:author": "J. K. Rowling", "dbo:publicationDate": "1997-06-26"},
        },
        actual_property_set=frozenset({"dbo:author", "dbo:publicationDate", "dbp:author"}),
        all_2hop_chains={
            "author": ("Person", ["birthDate", "deathDate", "birthPlace", "nationality"]),
        },
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


def _place_profile(**overrides):
    """Helper: build a Place EntityProfile."""
    defaults = dict(
        uri="http://dbpedia.org/resource/France",
        label="France",
        all_classes=["Place"],
        cached_classes=["Place"],
        actual_properties={
            "dbo": ["capital", "populationTotal"],
            "dbp": ["capital"],
            "values": {"dbo:capital": "Paris", "dbo:populationTotal": "67390000"},
        },
        actual_property_set=frozenset({"dbo:capital", "dbo:populationTotal", "dbp:capital"}),
        all_2hop_chains={
            "capital": ("Place", ["populationTotal", "country"]),
            "country": ("Place", ["capital", "populationTotal", "leader"]),
        },
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


class TestClassScoring:
    def test_film_scores_higher_for_director(self):
        analysis = analyze_question("Where was the director of Inception born?")
        film_score = _score_class("Film", analysis)
        person_score = _score_class("Person", analysis)
        assert film_score >= person_score

    def test_person_scores_for_birth(self):
        analysis = analyze_question("When was Albert Einstein born?")
        score = _score_class("Person", analysis)
        assert score >= 1

    def test_unknown_class_scores_zero(self):
        analysis = analyze_question("Who directed Inception?")
        score = _score_class("UnknownClass", analysis)
        assert score == 0

    def test_no_keywords_all_zero(self):
        analysis = analyze_question("What is the meaning of life?")
        for cls in ("Person", "Film", "Place"):
            assert _score_class(cls, analysis) == 0


class TestCompleteness:
    def test_complete_prefix_independent(self):
        """Entity has dbo:birthDate (but NOT dbp:birthDate), question asks 'born'
        -> completeness='complete' (prefix-independent)."""
        profile = _person_profile(
            actual_property_set=frozenset({"dbo:birthDate", "dbo:birthPlace"}),
        )
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        assert proj.completeness == "complete"

    def test_missing_bridge(self):
        """Film without director in actual -> completeness='missing'."""
        profile = _film_profile(
            actual_properties={"dbo": ["releaseDate"], "dbp": [], "values": {}},
            actual_property_set=frozenset({"dbo:releaseDate"}),
        )
        analysis = analyze_question("Where was the director of Inception born?")
        proj = project(profile, analysis)
        assert proj.completeness == "missing"
        assert any("director" in h for h in proj.missing_hops)

    def test_partial(self):
        """Has birthDate but not birthPlace -> partial."""
        profile = _person_profile(
            actual_property_set=frozenset({"dbo:birthDate"}),
        )
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        # "born" maps to birthDate AND birthPlace; has birthDate, missing birthPlace
        assert proj.completeness == "partial"

    def test_no_keywords_complete(self):
        """Question with no keyword matches -> completeness='complete' (nothing needed)."""
        profile = _film_profile()
        analysis = analyze_question("What is the meaning of life?")
        proj = project(profile, analysis)
        assert proj.completeness == "complete"

    def test_completeness_detail_populated(self):
        profile = _person_profile()
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        assert isinstance(proj.completeness_detail, dict)
        assert "birthDate" in proj.completeness_detail
        assert proj.completeness_detail["birthDate"] is True


# ---------------------------------------------------------------------------
# Fix 1: Prefix deduplication in render_for_sparql()
# ---------------------------------------------------------------------------

class TestPrefixDedup:
    def test_default_includes_prefixes(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "PREFIX dbr:" in output
        assert "PREFIX dbo:" in output
        assert "PREFIX dbp:" in output
        assert "PREFIX rdfs:" in output

    def test_include_prefixes_false_excludes_all(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql(include_prefixes=False)
        assert "PREFIX" not in output
        assert "SPARQL Prefixes:" not in output

    def test_no_prefix_still_has_entity_info(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql(include_prefixes=False)
        assert "Entity: Inception" in output
        assert "URI: <http://dbpedia.org/resource/Inception>" in output
        assert "Class: dbo:Film" in output

    def test_no_prefix_still_has_properties(self):
        profile = _person_profile()
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql(include_prefixes=False)
        assert "dbo: birthDate" in output
        assert "Actual properties found:" in output

    def test_multi_entity_join_prefix_once(self):
        """When joining multiple projections, prefix appears only once."""
        p1 = _person_profile()
        p2 = _person_profile(
            uri="http://dbpedia.org/resource/Nikola_Tesla",
            label="Nikola Tesla",
        )
        analysis = analyze_question("Who was born first, Einstein or Tesla?")
        proj1 = project(p1, analysis)
        proj2 = project(p2, analysis)
        joined = "\n\n---\n\n".join(
            p.render_for_sparql(include_prefixes=(i == 0))
            for i, p in enumerate([proj1, proj2])
        )
        assert joined.count("PREFIX dbr:") == 1
        assert joined.count("PREFIX dbo:") == 1
        assert joined.count("PREFIX dbp:") == 1

    def test_multi_entity_join_both_entities_present(self):
        p1 = _person_profile()
        p2 = _person_profile(
            uri="http://dbpedia.org/resource/Nikola_Tesla",
            label="Nikola Tesla",
        )
        analysis = analyze_question("Who was born first?")
        proj1 = project(p1, analysis)
        proj2 = project(p2, analysis)
        joined = "\n\n---\n\n".join(
            p.render_for_sparql(include_prefixes=(i == 0))
            for i, p in enumerate([proj1, proj2])
        )
        assert "Entity: Albert Einstein" in joined
        assert "Entity: Nikola Tesla" in joined

    def test_single_entity_has_prefix_by_default(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert output.startswith("SPARQL Prefixes:")

    def test_three_entity_join_prefix_once(self):
        """Three entities: prefix only on the first."""
        profiles = [
            _person_profile(uri="http://dbpedia.org/resource/A", label="A"),
            _person_profile(uri="http://dbpedia.org/resource/B", label="B"),
            _person_profile(uri="http://dbpedia.org/resource/C", label="C"),
        ]
        analysis = analyze_question("Who was born first?")
        projs = [project(p, analysis) for p in profiles]
        joined = "\n\n---\n\n".join(
            p.render_for_sparql(include_prefixes=(i == 0))
            for i, p in enumerate(projs)
        )
        assert joined.count("PREFIX dbr:") == 1
        assert "Entity: A" in joined
        assert "Entity: B" in joined
        assert "Entity: C" in joined


# ---------------------------------------------------------------------------
# Fix 3: Filtered property hints
# ---------------------------------------------------------------------------

class TestFilteredHints:
    def test_born_question_shows_birthdate_hint(self):
        profile = _person_profile()
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "birthDate: date of birth" in output

    def test_born_question_hides_father_hint(self):
        profile = _person_profile()
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "father of the person" not in output

    def test_born_question_hides_mother_hint(self):
        profile = _person_profile()
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "mother of the person" not in output

    def test_born_question_hides_parent_hint(self):
        profile = _person_profile()
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "generic parent" not in output

    def test_born_question_hides_death_hint(self):
        profile = _person_profile()
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "date of death" not in output

    def test_director_question_shows_director_hint(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "director: film director" in output

    def test_director_question_hides_producer_hint(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "film producer" not in output

    def test_director_question_hides_starring_hint(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "lead actors" not in output

    def test_bridge_director_born_shows_director_hint(self):
        """Bridge question: director + born → show director hint."""
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "director: film director" in output

    def test_bridge_director_born_hides_irrelevant_hints(self):
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "film producer" not in output
        assert "lead actors" not in output

    def test_no_keywords_fallback_shows_hints(self):
        """No keyword match → fallback to first 5 hints."""
        profile = _film_profile()
        analysis = analyze_question("What is the meaning of life?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        # With no relevant keywords, should fall back to showing some hints
        assert "Property hints:" in output

    def test_no_keywords_fallback_limit_5(self):
        """Fallback mode shows at most 5 hints."""
        profile = _person_profile()
        analysis = analyze_question("What is the meaning of life?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        hint_lines = [line for line in output.splitlines() if line.strip().startswith("- ")]
        assert len(hint_lines) <= 5

    def test_death_question_shows_death_hint(self):
        """Must use 'died' keyword (not 'die') for keyword matching."""
        profile = _person_profile()
        analysis = analyze_question("When did Albert Einstein died?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "deathDate: date of death" in output

    def test_death_question_hides_birth_hint(self):
        profile = _person_profile()
        analysis = analyze_question("When did Albert Einstein died?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "date of birth" not in output

    def test_father_question_shows_parent_hint(self):
        """'father' keyword → parent property → parent hint shown."""
        profile = _person_profile()
        analysis = analyze_question("Who is the father of Prince William?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        # "father" maps to "parent" property, check parent-related hints
        assert "parent" in output.lower()

    def test_author_shows_author_hint(self):
        profile = _book_profile()
        analysis = analyze_question("Who wrote Harry Potter?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "author: book author" in output

    def test_author_hides_publisher_hint(self):
        profile = _book_profile()
        analysis = analyze_question("Who wrote Harry Potter?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "publishing company" not in output

    def test_capital_question_shows_capital_hint(self):
        profile = _place_profile()
        analysis = analyze_question("What is the capital of France?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "capital" in output

    def test_full_cached_property_lists_always_shown(self):
        """Even with filtered hints, full dbo/dbp property lists remain."""
        profile = _person_profile()
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        # Full property list still present
        assert "dbo: birthDate, deathDate, birthPlace" in output
        assert "dbp: father, mother, birthDate" in output


# ---------------------------------------------------------------------------
# _relevant_property_names() unit tests
# ---------------------------------------------------------------------------

class TestRelevantPropertyNames:
    def test_direct_question(self):
        """Direct 'born' → {birthDate, birthPlace}."""
        profile = _person_profile()
        analysis = analyze_question("When was Einstein born?")
        proj = project(profile, analysis)
        names = proj._relevant_property_names()
        assert "birthDate" in names
        assert "birthPlace" in names

    def test_bridge_includes_bridge_property(self):
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        names = proj._relevant_property_names()
        assert "director" in names

    def test_bridge_includes_chain_target_properties(self):
        """Bridge chains like director → Person: birthDate, deathDate..."""
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        names = proj._relevant_property_names()
        assert "birthDate" in names
        assert "deathDate" in names
        assert "birthPlace" in names

    def test_bridge_includes_hop2_properties(self):
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        names = proj._relevant_property_names()
        # hop2 "born" → birthDate, birthPlace
        assert "birthDate" in names
        assert "birthPlace" in names

    def test_no_keywords_returns_empty(self):
        profile = _film_profile()
        analysis = analyze_question("What is the meaning of life?")
        proj = project(profile, analysis)
        names = proj._relevant_property_names()
        assert names == set()

    def test_direct_director_includes_director(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        names = proj._relevant_property_names()
        assert "director" in names

    def test_no_irrelevant_properties_in_names(self):
        """Direct 'born' should NOT include director, producer, etc."""
        profile = _person_profile()
        analysis = analyze_question("When was Einstein born?")
        proj = project(profile, analysis)
        names = proj._relevant_property_names()
        assert "director" not in names
        assert "producer" not in names
        assert "starring" not in names


# ---------------------------------------------------------------------------
# Standard rendering tests
# ---------------------------------------------------------------------------

class TestRendering:
    def test_render_for_routing_token_count(self):
        profile = _film_profile()
        analysis = analyze_question("Where was the director of Inception born?")
        proj = project(profile, analysis)
        output = proj.render_for_routing()
        # Rough token estimate: word count / 0.75
        word_count = len(output.split())
        assert word_count < 150, f"Routing output too long: {word_count} words"

    def test_render_for_sparql_includes_prefix(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "PREFIX dbr:" in output
        assert "PREFIX dbo:" in output

    def test_render_for_sparql_no_prefix(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql(include_prefixes=False)
        assert "PREFIX" not in output
        assert "Entity: Inception" in output

    def test_render_for_sparql_hints_filtered(self):
        """Property hints should only show question-relevant properties."""
        profile = _person_profile()
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        # birthDate is relevant to "born", should be in hints
        assert "birthDate" in output
        # father/mother are NOT relevant to "born", should be filtered out of hints
        assert "father of the person" not in output

    def test_render_for_text_includes_type(self):
        profile = _film_profile()
        analysis = analyze_question("Where was the director of Inception born?")
        proj = project(profile, analysis)
        output = proj.render_for_text()
        assert "Expect:" in output
        assert "Discovery:" in output

    def test_render_for_sparql_includes_uri(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "http://dbpedia.org/resource/Inception" in output

    def test_render_for_sparql_includes_class(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "Class: dbo:Film" in output

    def test_render_for_sparql_includes_actual_values(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "dbo:director -> Christopher Nolan" in output

    def test_render_for_sparql_includes_chains(self):
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "2-Hop chains:" in output
        assert "dbo:director -> Person" in output

    def test_render_for_routing_shows_completeness(self):
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        output = proj.render_for_routing()
        assert "Completeness:" in output

    def test_render_for_routing_shows_chains(self):
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        output = proj.render_for_routing()
        assert "Chains:" in output

    def test_render_for_routing_shows_missing(self):
        profile = _film_profile(
            actual_properties={"dbo": ["releaseDate"], "dbp": [], "values": {}},
            actual_property_set=frozenset({"dbo:releaseDate"}),
        )
        analysis = analyze_question("Where was the director of Inception born?")
        proj = project(profile, analysis)
        output = proj.render_for_routing()
        assert "Missing:" in output


# ---------------------------------------------------------------------------
# Fix 2 through rendering: TEXT view bridge target type
# ---------------------------------------------------------------------------

class TestTextViewTargetType:
    def test_bridge_text_view_shows_person(self):
        """Bridge 'director + born' → TEXT view should show Person, not MusicalArtist."""
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        output = proj.render_for_text()
        assert "Person" in output
        assert "MusicalArtist" not in output

    def test_bridge_text_view_shows_via_bridge(self):
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        output = proj.render_for_text()
        assert "via director" in output

    def test_bridge_text_view_type_filter(self):
        """TEXT view should use expected_target_type in discovery filter."""
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        output = proj.render_for_text()
        assert "dbo:Person" in output

    def test_author_bridge_text_view_shows_person(self):
        profile = _book_profile()
        analysis = analyze_question("Where was the author of Harry Potter born?")
        proj = project(profile, analysis)
        output = proj.render_for_text()
        assert "Person" in output

    def test_no_bridge_text_view_no_via(self):
        """Direct question → no 'via' in TEXT view."""
        profile = _person_profile()
        analysis = analyze_question("When was Einstein born?")
        proj = project(profile, analysis)
        output = proj.render_for_text()
        assert "via" not in output

    def test_no_keywords_text_view_unknown(self):
        profile = _film_profile()
        analysis = analyze_question("What is the meaning of life?")
        proj = project(profile, analysis)
        output = proj.render_for_text()
        assert "unknown type" in output
        assert "unfiltered" in output


# ---------------------------------------------------------------------------
# No class / edge cases
# ---------------------------------------------------------------------------

class TestNoClassEntity:
    def test_no_cached_class_no_hints(self):
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/SomeThing",
            label="SomeThing",
            all_classes=["SomethingElse"],
            cached_classes=[],
            actual_properties={"dbo": ["foo"], "dbp": [], "values": {"dbo:foo": "bar"}},
            actual_property_set=frozenset({"dbo:foo"}),
            all_2hop_chains={},
        )
        analysis = analyze_question("What is foo?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        # No cached class → no hints section
        assert "Property hints:" not in output
        # But actual properties still shown
        assert "dbo:foo -> bar" in output
        # Dynamic fallback assigns primary_class from all_classes
        assert proj.primary_class == "SomethingElse"

    def test_no_cached_class_no_class_line(self):
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/X",
            label="X",
            all_classes=[],
            cached_classes=[],
            actual_properties={"dbo": [], "dbp": [], "values": {}},
            actual_property_set=frozenset(),
            all_2hop_chains={},
        )
        analysis = analyze_question("What is X?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "Class:" not in output

    def test_uncached_type_dynamic_rendering(self):
        """Uncached types show actual properties and dynamic chains in SPARQL view."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/SomeFilmStudio",
            label="Some Film Studio",
            all_classes=["FilmStudio", "Organisation", "Agent"],
            cached_classes=[],
            actual_properties={
                "dbo": ["foundedBy", "locationCity", "foundingDate"],
                "dbp": ["founder"],
                "values": {
                    "dbo:foundedBy": "Jane Doe",
                    "dbo:locationCity": "Los Angeles",
                },
            },
            actual_property_set=frozenset({
                "dbo:foundedBy", "dbo:locationCity", "dbo:foundingDate",
                "dbp:founder",
            }),
            # Dynamic chain: foundedBy -> Person (matches KEYWORD_TO_PROPERTIES)
            all_2hop_chains={
                "foundedBy": ("Person", ["birthDate", "deathDate", "birthPlace"]),
            },
        )
        # "founder" maps to "foundedBy" in KEYWORD_TO_PROPERTIES
        analysis = analyze_question("When was the founder of this studio born?")
        proj = project(profile, analysis)

        # primary_class should be FilmStudio (first specific, filtering Agent)
        assert proj.primary_class == "FilmStudio"
        # Dynamic rendering should show actual properties
        output = proj.render_for_sparql(include_prefixes=False)
        assert "Class: dbo:FilmStudio" in output
        assert "dbo: foundedBy" in output
        # Dynamic chains appear (foundedBy matches bridge_property from "founder" keyword)
        assert "foundedBy -> Person" in output

    def test_generic_classes_filtered_for_primary(self):
        """Generic classes (Agent, Thing) are skipped when selecting primary_class."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/X",
            label="X",
            all_classes=["Agent", "Thing", "SoccerClub"],
            cached_classes=[],
            actual_properties={"dbo": [], "dbp": [], "values": {}},
            actual_property_set=frozenset(),
            all_2hop_chains={},
        )
        analysis = analyze_question("What is X?")
        proj = project(profile, analysis)
        assert proj.primary_class == "SoccerClub"

    def test_empty_actual_properties(self):
        profile = _film_profile(
            actual_properties={"dbo": [], "dbp": [], "values": {}},
            actual_property_set=frozenset(),
        )
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "Actual properties found:" not in output


class TestMultiClassEntity:
    def test_musical_artist_preferred_for_genre(self):
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/David_Bowie",
            label="David Bowie",
            all_classes=["Person", "MusicalArtist"],
            cached_classes=["Person", "MusicalArtist"],
            actual_properties={"dbo": ["birthDate", "genre"], "dbp": [], "values": {}},
            actual_property_set=frozenset({"dbo:birthDate", "dbo:genre"}),
            all_2hop_chains={},
        )
        analysis = analyze_question("What genre does David Bowie play?")
        proj = project(profile, analysis)
        assert proj.primary_class == "MusicalArtist"


# ---------------------------------------------------------------------------
# Deep: relevant_properties deduplication
# ---------------------------------------------------------------------------

class TestRelevantPropertiesDedup:
    def test_bridge_dedup_target_and_hop2(self):
        """Bridge 'director + born': birthDate appears in both target_properties
        and hop2_properties. relevant_properties should have no duplicates."""
        profile = _person_profile()
        analysis = analyze_question("When was the director of X born?")
        proj = project(profile, analysis)
        assert len(proj.relevant_properties) == len(set(proj.relevant_properties))

    def test_direct_no_duplicates(self):
        """Direct question: relevant_properties should never have dupes."""
        profile = _person_profile()
        analysis = analyze_question("When was Einstein born?")
        proj = project(profile, analysis)
        assert len(proj.relevant_properties) == len(set(proj.relevant_properties))


# ---------------------------------------------------------------------------
# Deep: chain filtering by target_names (not bridge_property)
# ---------------------------------------------------------------------------

class TestChainFilteringByTargetName:
    def test_direct_birthplace_matches_chain(self):
        """Direct 'born' question → birthPlace in target_names → birthPlace chain kept."""
        profile = _person_profile()
        analysis = analyze_question("Where was Einstein born?")
        proj = project(profile, analysis)
        # "born" → birthDate, birthPlace. birthPlace is a chain in Person profile.
        assert "birthPlace" in proj.relevant_chains

    def test_director_question_matches_director_chain(self):
        """Direct 'directed' → director in target_names → director chain kept."""
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        assert "director" in proj.relevant_chains

    def test_unrelated_chain_excluded(self):
        """Question about director should NOT include producer chain."""
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        assert "producer" not in proj.relevant_chains if hasattr(proj, 'relevant_chains') else True

    def test_capital_matches_capital_chain(self):
        """'capital' question → capital chain in Place profile kept."""
        profile = _place_profile()
        analysis = analyze_question("What is the capital of France?")
        proj = project(profile, analysis)
        assert "capital" in proj.relevant_chains


# ---------------------------------------------------------------------------
# Deep: completeness for comparison and compositional questions
# ---------------------------------------------------------------------------

class TestCompletenessComparison:
    def test_comparison_complete(self):
        """Comparison with all needed properties → complete."""
        profile = _person_profile()
        analysis = analyze_question("Who was born earlier, Einstein or Newton?")
        proj = project(profile, analysis)
        assert proj.completeness == "complete"

    def test_comparison_partial(self):
        """Comparison question with only birthDate (not birthPlace) → partial."""
        profile = _person_profile(
            actual_property_set=frozenset({"dbo:birthDate"}),
        )
        analysis = analyze_question("Who was born earlier, Einstein or Newton?")
        proj = project(profile, analysis)
        assert proj.completeness == "partial"

    def test_comparison_missing_all(self):
        """Comparison question with NO birth properties → missing."""
        profile = _person_profile(
            actual_property_set=frozenset({"dbo:occupation"}),
        )
        analysis = analyze_question("Who was born earlier, Einstein or Newton?")
        proj = project(profile, analysis)
        assert proj.completeness == "missing"


class TestCompletenessCompositional:
    def test_compositional_partial(self):
        """Question with 2 keywords, only 1 property present → partial."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/X",
            label="X",
            all_classes=["Person"],
            cached_classes=["Person"],
            actual_properties={"dbo": ["birthDate"], "dbp": [], "values": {}},
            actual_property_set=frozenset({"dbo:birthDate"}),
            all_2hop_chains={},
        )
        # "born" (birthDate, birthPlace) + "award" (award) → compositional
        # Has birthDate but not birthPlace, not award
        analysis = analyze_question("When was this person born and what award did they win?")
        proj = project(profile, analysis)
        assert proj.completeness == "partial"

    def test_compositional_complete(self):
        """All needed properties present → complete."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/X",
            label="X",
            all_classes=["MusicalArtist"],
            cached_classes=["MusicalArtist"],
            actual_properties={
                "dbo": ["genre", "birthDate", "birthPlace"],
                "dbp": ["genre", "birthDate", "birthPlace"],
                "values": {},
            },
            actual_property_set=frozenset({
                "dbo:genre", "dbo:birthDate", "dbo:birthPlace",
                "dbp:genre", "dbp:birthDate", "dbp:birthPlace",
            }),
            all_2hop_chains={},
        )
        analysis = analyze_question("What genre and when was this artist born?")
        proj = project(profile, analysis)
        assert proj.completeness == "complete"


# ---------------------------------------------------------------------------
# Deep: class scoring tie-breaking
# ---------------------------------------------------------------------------

class TestClassScoringTiebreak:
    def test_longer_name_wins_on_tie(self):
        """Equal score → longer class name wins (MusicalArtist > Person)."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/X",
            label="X",
            all_classes=["Person", "MusicalArtist"],
            cached_classes=["Person", "MusicalArtist"],
            actual_properties={
                "dbo": ["birthDate", "birthPlace"],
                "dbp": ["birthDate"],
                "values": {},
            },
            actual_property_set=frozenset({"dbo:birthDate", "dbo:birthPlace", "dbp:birthDate"}),
            all_2hop_chains={},
        )
        # "born" → birthDate, birthPlace — both Person and MusicalArtist have these
        analysis = analyze_question("When was X born?")
        proj = project(profile, analysis)
        # Both score the same; MusicalArtist is longer name → wins tie
        assert proj.primary_class == "MusicalArtist"


# ---------------------------------------------------------------------------
# Deep: rendering truncation limits
# ---------------------------------------------------------------------------

class TestRenderingTruncation:
    def test_routing_truncates_properties_at_5(self):
        """render_for_routing shows max 5 properties."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/X",
            label="X",
            all_classes=["Person"],
            cached_classes=["Person"],
            actual_properties={"dbo": [], "dbp": [], "values": {}},
            actual_property_set=frozenset({
                "dbo:birthDate", "dbo:birthPlace", "dbo:deathDate",
                "dbo:deathPlace", "dbo:nationality", "dbo:occupation",
                "dbo:parent", "dbo:spouse",
            }),
            all_2hop_chains={},
        )
        analysis = analyze_question("When was X born and died and what nationality and occupation?")
        proj = project(profile, analysis)
        output = proj.render_for_routing()
        if "Props:" in output:
            props_line = [ln for ln in output.splitlines() if ln.startswith("Props:")][0]
            prop_count = len(props_line.split(","))
            assert prop_count <= 5

    def test_routing_truncates_chains_at_2(self):
        """render_for_routing shows max 2 chains."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/X",
            label="X",
            all_classes=["Person"],
            cached_classes=["Person"],
            actual_properties={"dbo": [], "dbp": [], "values": {}},
            actual_property_set=frozenset({
                "dbo:parent", "dbo:spouse", "dbo:birthPlace",
            }),
            all_2hop_chains={
                "parent": ("Person", ["birthDate", "deathDate"]),
                "spouse": ("Person", ["birthDate", "deathDate"]),
                "birthPlace": ("Place", ["country", "isPartOf"]),
            },
        )
        analysis = analyze_question("Who is the parent and spouse born where?")
        proj = project(profile, analysis)
        output = proj.render_for_routing()
        if "Chains:" in output:
            chains_line = [ln for ln in output.splitlines() if ln.startswith("Chains:")][0]
            chain_count = len(chains_line.split(";"))
            assert chain_count <= 2


# ---------------------------------------------------------------------------
# Deep: render_for_sparql dbp properties and edge cases
# ---------------------------------------------------------------------------

class TestRenderSparqlDbp:
    def test_shows_dbp_actual_properties(self):
        """Both dbo and dbp actual properties should render."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/X",
            label="X",
            all_classes=["Person"],
            cached_classes=["Person"],
            actual_properties={
                "dbo": ["birthDate"],
                "dbp": ["father", "mother"],
                "values": {
                    "dbo:birthDate": "1990-01-01",
                    "dbp:father": "John",
                    "dbp:mother": "Jane",
                },
            },
            actual_property_set=frozenset({"dbo:birthDate", "dbp:father", "dbp:mother"}),
            all_2hop_chains={},
        )
        analysis = analyze_question("Who is the father of X?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "dbp:father -> John" in output
        assert "dbp:mother -> Jane" in output

    def test_no_class_shows_actual_values(self):
        """Entity with no cached class but actual values → renders values."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/X",
            label="X",
            all_classes=["RareThing"],
            cached_classes=[],
            actual_properties={
                "dbo": ["foo"],
                "dbp": [],
                "values": {"dbo:foo": "bar_value"},
            },
            actual_property_set=frozenset({"dbo:foo"}),
            all_2hop_chains={},
        )
        analysis = analyze_question("What is foo?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "dbo:foo -> bar_value" in output
        assert "Actual properties found:" in output


# ---------------------------------------------------------------------------
# Deep: render_for_text edge cases
# ---------------------------------------------------------------------------

class TestRenderTextEdgeCases:
    def test_direct_born_has_target_type(self):
        """Direct 'born' question → Person expected_target_type in text view."""
        profile = _person_profile()
        analysis = analyze_question("When was Einstein born?")
        proj = project(profile, analysis)
        output = proj.render_for_text()
        assert "Expect:" in output
        # Direct born → vote-based type (Person or MusicalArtist)
        assert "Person" in output or "MusicalArtist" in output

    def test_capital_question_place_type(self):
        """'capital' question → Place in text view."""
        profile = _place_profile()
        analysis = analyze_question("What is the capital of France?")
        proj = project(profile, analysis)
        output = proj.render_for_text()
        assert "Place" in output


# ---------------------------------------------------------------------------
# Deep: SchemaProjection dataclass attributes
# ---------------------------------------------------------------------------

class TestProjectionDataIntegrity:
    def test_projection_references_profile(self):
        """projection.entity_profile is the same profile object."""
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        assert proj.entity_profile is profile

    def test_projection_references_analysis(self):
        """projection.question_analysis is the same analysis object."""
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        assert proj.question_analysis is analysis

    def test_relevant_chains_subset_of_profile(self):
        """relevant_chains keys are always a subset of profile.all_2hop_chains keys."""
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        assert set(proj.relevant_chains.keys()) <= set(profile.all_2hop_chains.keys())

    def test_relevant_properties_all_in_actual(self):
        """relevant_properties are all in profile.actual_property_set."""
        profile = _person_profile()
        analysis = analyze_question("When was Einstein born?")
        proj = project(profile, analysis)
        for prop in proj.relevant_properties:
            assert prop in profile.actual_property_set

    def test_completeness_detail_keys_match_needed(self):
        """completeness_detail keys = needed property names (without prefix)."""
        profile = _person_profile()
        analysis = analyze_question("When was Einstein born?")
        proj = project(profile, analysis)
        needed = {p.split(":")[-1] for p in analysis.target_properties}
        assert set(proj.completeness_detail.keys()) == needed
