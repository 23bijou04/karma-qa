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
        # expected_target_type derived from property votes; should be some valid class
        assert "Expect:" in output
        assert "Discovery:" in output

    def test_render_for_sparql_includes_uri(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        output = proj.render_for_sparql()
        assert "http://dbpedia.org/resource/Inception" in output


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
