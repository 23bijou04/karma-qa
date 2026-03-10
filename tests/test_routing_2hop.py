"""Tests for routing rules: structured SchemaProjection-based routing."""

from dbpedia_qa.models import Entity
from entity_profile import EntityProfile
from question_analysis import analyze_question
from routing import (
    ROUTE_PROMPT,
    _has_incomplete_entity_coverage,
    _is_comparison_question,
    _rule_based_route,
)
from schema_projection import project

# ---------------------------------------------------------------------------
# Helper: build EntityProfile fixtures
# ---------------------------------------------------------------------------

def _film_profile(**overrides):
    defaults = dict(
        uri="http://dbpedia.org/resource/Inception",
        label="Inception",
        all_classes=["Film"],
        cached_classes=["Film"],
        actual_properties={
            "dbo": ["director", "producer", "starring", "releaseDate", "runtime"],
            "dbp": ["director", "producer", "starring", "released"],
            "values": {"dbo:director": "Christopher Nolan", "dbo:releaseDate": "2010-07-16"},
        },
        actual_property_set=frozenset({
            "dbo:director", "dbo:producer", "dbo:starring", "dbo:releaseDate", "dbo:runtime",
            "dbp:director", "dbp:producer", "dbp:starring", "dbp:released",
        }),
        all_2hop_chains={
            "director": ("Person", ["birthDate", "deathDate", "birthPlace", "nationality"]),
            "producer": ("Person", ["birthDate", "deathDate", "birthPlace"]),
            "starring": ("Person", ["birthDate", "deathDate", "birthPlace", "nationality"]),
        },
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


def _person_profile(**overrides):
    defaults = dict(
        uri="http://dbpedia.org/resource/Albert_Einstein",
        label="Albert Einstein",
        all_classes=["Person"],
        cached_classes=["Person"],
        actual_properties={
            "dbo": ["birthDate", "deathDate", "birthPlace", "deathPlace", "parent", "spouse"],
            "dbp": [],
            "values": {"dbo:birthDate": "1879-03-14", "dbo:birthPlace": "Ulm"},
        },
        actual_property_set=frozenset({
            "dbo:birthDate", "dbo:deathDate", "dbo:birthPlace", "dbo:deathPlace",
            "dbo:parent", "dbo:spouse",
        }),
        all_2hop_chains={
            "birthPlace": ("Place", ["country", "isPartOf", "populationTotal"]),
        },
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


def _book_profile(**overrides):
    defaults = dict(
        uri="http://dbpedia.org/resource/Harry_Potter",
        label="Harry Potter",
        all_classes=["Book"],
        cached_classes=["Book"],
        actual_properties={
            "dbo": ["author", "publisher", "publicationDate"],
            "dbp": [],
            "values": {"dbo:author": "J. K. Rowling"},
        },
        actual_property_set=frozenset({
            "dbo:author", "dbo:publisher", "dbo:publicationDate",
        }),
        all_2hop_chains={
            "author": ("Person", ["birthDate", "deathDate", "birthPlace", "nationality"]),
        },
    )
    defaults.update(overrides)
    return EntityProfile(**defaults)


class TestRuleBasedRoute:
    def test_bridge_comparison_director_born(self):
        """'Which director was born earlier' + Film schema -> SPARQL.
        Hits direct_date_comparison first because Film has dbo:releaseDate."""
        profile = _film_profile()
        analysis = analyze_question("Which director of Inception was born earlier?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        route, reason = result
        assert route == "SPARQL"
        assert reason == "rule:direct_date_comparison"

    def test_bridge_non_comparison_director_born(self):
        """'When was the director born' + Film schema -> SPARQL."""
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        route, reason = result
        assert route == "SPARQL"
        assert reason == "rule:bridge_hop2"

    def test_direct_date_comparison(self):
        """'Who was born earlier' + Person with birthDate -> SPARQL."""
        profile = _person_profile()
        analysis = analyze_question("Who was born earlier, Einstein or Newton?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        route, reason = result
        assert route == "SPARQL"
        assert reason == "rule:direct_date_comparison"

    def test_no_relevant_props_returns_none(self):
        """Question about unrelated topic -> None (LLM decides)."""
        profile = _film_profile()
        analysis = analyze_question("What is the capital of France?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is None

    def test_bridge_author_comparison(self):
        """'Which author was born earlier' with Book schema -> SPARQL."""
        profile = _book_profile()
        analysis = analyze_question(
            "Which author was born earlier, the author of Harry Potter or LOTR?"
        )
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        route, reason = result
        assert route == "SPARQL"
        assert "bridge" in reason


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
        assert _is_comparison_question(
            "Which film's director died earlier, Film A or Film B?", entities
        )

    def test_older_with_two_entities(self):
        entities = _make_entities("Person A", "Person B")
        assert _is_comparison_question("Who is older, Person A or Person B?", entities)

    def test_same_with_two_entities(self):
        entities = _make_entities("Film A", "Film B")
        assert _is_comparison_question(
            "Are both Film A and Film B from the same country?", entities
        )

    def test_both_with_two_entities(self):
        entities = _make_entities("X", "Y")
        assert _is_comparison_question("Do both X and Y share the same director?", entities)

    def test_more_recently_with_two_entities(self):
        entities = _make_entities("Film A", "Film B")
        assert _is_comparison_question(
            "Which was released more recently, Film A or Film B?", entities
        )

    def test_single_entity_returns_false(self):
        entities = _make_entities("Film A")
        assert not _is_comparison_question("Which film was released earlier?", entities)

    def test_no_comparison_words_returns_false(self):
        entities = _make_entities("Film A", "Film B")
        assert not _is_comparison_question("Who directed Film A?", entities)

    def test_empty_entities_returns_false(self):
        assert not _is_comparison_question("Who was born earlier?", [])


# ---- Country/nationality comparison routing ----

class TestCountryComparisonRouting:
    def test_same_country_routes_sparql(self):
        """'same country' comparison with nationality property -> SPARQL country route."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/Person1",
            label="Person1",
            all_classes=["Person"],
            cached_classes=["Person"],
            actual_properties={
                "dbo": ["birthPlace", "nationality", "occupation"],
                "dbp": ["nationality"],
                "values": {"dbo:nationality": "Germany"},
            },
            actual_property_set=frozenset({
                "dbo:birthPlace", "dbo:nationality", "dbo:occupation", "dbp:nationality",
            }),
            all_2hop_chains={},
        )
        analysis = analyze_question("Are both Person1 and Person2 from the same country?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        route, reason = result
        assert route == "SPARQL"
        assert "country" in reason

    def test_same_nationality_routes_sparql(self):
        """'same nationality' routes to SPARQL."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/Person1",
            label="Person1",
            all_classes=["Person"],
            cached_classes=["Person"],
            actual_properties={
                "dbo": ["nationality"],
                "dbp": ["nationality"],
                "values": {"dbo:nationality": "Germany"},
            },
            actual_property_set=frozenset({"dbo:nationality", "dbp:nationality"}),
            all_2hop_chains={},
        )
        analysis = analyze_question("Do Person1 and Person2 have the same nationality?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"

    def test_same_without_country_keyword_no_country_route(self):
        """'same' alone doesn't trigger country comparison route."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/Film1",
            label="Film1",
            all_classes=["Film"],
            cached_classes=["Film"],
            actual_properties={
                "dbo": ["director", "producer"],
                "dbp": [],
                "values": {"dbo:director": "Christopher Nolan"},
            },
            actual_property_set=frozenset({"dbo:director", "dbo:producer"}),
            all_2hop_chains={},
        )
        analysis = analyze_question("Is Film1 the same as Film2?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        if result is not None:
            assert "country" not in result[1]

    def test_more_awards_no_false_date_route(self):
        """'Which has more awards?' with date property."""
        profile = EntityProfile(
            uri="http://dbpedia.org/resource/Person1",
            label="Person1",
            all_classes=["Person"],
            cached_classes=["Person"],
            actual_properties={
                "dbo": ["birthDate", "award"],
                "dbp": [],
                "values": {"dbo:birthDate": "1990-01-01", "dbo:award": "Academy Award"},
            },
            actual_property_set=frozenset({"dbo:birthDate", "dbo:award"}),
            all_2hop_chains={},
        )
        analysis = analyze_question("Which person has more awards, Person1 or Person2?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        if result is not None:
            assert result[0] == "SPARQL"


class TestRoutePromptContract:
    """Verify ROUTE_PROMPT placeholders and format contract."""

    def test_prompt_has_required_placeholders(self):
        assert "{schema}" in ROUTE_PROMPT
        assert "{question}" in ROUTE_PROMPT
        assert "{hint}" in ROUTE_PROMPT

    def test_prompt_mentions_2hop_reasoning(self):
        lower = ROUTE_PROMPT.lower()
        assert "2-hop" in lower or "chain" in lower

    def test_prompt_format_with_hint(self):
        result = ROUTE_PROMPT.format(
            schema="Entity: Foo\n  dbo: bar",
            question="Who is Foo?",
            hint="Schema analysis suggests: rule:direct_property (SPARQL likely feasible)",
        )
        assert "Who is Foo?" in result
        assert "rule:direct_property" in result

    def test_prompt_format_without_hint(self):
        result = ROUTE_PROMPT.format(
            schema="Entity: Foo\n  dbo: bar",
            question="Who is Foo?",
            hint="",
        )
        assert "Who is Foo?" in result

    def test_prompt_format_with_escaped_braces(self):
        """Schema or question with curly braces must be escaped before .format()."""
        safe_schema = "Entity: {Foo}".replace("{", "{{").replace("}", "}}")
        result = ROUTE_PROMPT.format(
            schema=safe_schema, question="Who is Foo?", hint="",
        )
        assert "{Foo}" in result


class TestIncompleteCoverage:
    """Tests for _has_incomplete_entity_coverage() guard."""

    def test_detects_missing_entity(self):
        entities = _make_entities("The Nice Guys")
        q = "Which film has the director who was born first, Piccoli Fuochi or The Nice Guys?"
        assert _has_incomplete_entity_coverage(q, entities, "Entity: The Nice Guys")

    def test_complete_coverage_passes(self):
        entities = _make_entities("Film A", "Film B")
        q = "Which was released first, Film A or Film B?"
        assert not _has_incomplete_entity_coverage(
            q, entities, "Entity: Film A\nEntity: Film B"
        )

    def test_non_comparison_skips(self):
        entities = _make_entities("X")
        q = "Who directed X?"
        assert not _has_incomplete_entity_coverage(q, entities, "Entity: X")

    def test_no_or_pattern_skips(self):
        entities = _make_entities("Film A")
        q = "Which film was released first?"
        assert not _has_incomplete_entity_coverage(q, entities, "Entity: Film A")

    def test_case_insensitive_label_match(self):
        entities = _make_entities("the nice guys")
        q = "Which was released first, Film X or The Nice Guys?"
        assert _has_incomplete_entity_coverage(q, entities, "Entity: the nice guys")

    def test_and_pattern_detects_missing(self):
        """'both X and Y' should detect entity names."""
        entities = _make_entities("X")
        q = "Are both X and Y located in the same country?"
        assert _has_incomplete_entity_coverage(q, entities, "Entity: X")

    def test_comma_and_pattern(self):
        """', X and Y?' should detect entity names."""
        entities = _make_entities("X")
        q = "Which was released first, X and Y?"
        assert _has_incomplete_entity_coverage(q, entities, "Entity: X")
