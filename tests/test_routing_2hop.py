"""Tests for routing rules: structured SchemaProjection-based routing."""

from models import Entity
from entity_profile import EntityProfile
from question_analysis import analyze_question
from routing import (
    PROPERTY_EXTRACTION_PROMPT,
    _build_property_schema,
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
        """'Which author was born earlier' with Book schema -> SPARQL.

        May fire as direct_date_comparison (publicationDate in _DATE_PROPERTIES)
        or bridge_comparison (author -> Person -> birthDate). Both are correct
        SPARQL routes.
        """
        profile = _book_profile()
        analysis = analyze_question(
            "Which author was born earlier, the author of Harry Potter or LOTR?"
        )
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        route, reason = result
        assert route == "SPARQL"
        assert "bridge" in reason or "date_comparison" in reason


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


class TestPropertyExtractionPrompt:
    """Verify PROPERTY_EXTRACTION_PROMPT placeholders and format."""

    def test_prompt_has_required_placeholders(self):
        assert "{schema}" in PROPERTY_EXTRACTION_PROMPT
        assert "{question}" in PROPERTY_EXTRACTION_PROMPT

    def test_prompt_mentions_2hop(self):
        lower = PROPERTY_EXTRACTION_PROMPT.lower()
        assert "2-hop" in lower or "chain" in lower

    def test_prompt_format_basic(self):
        result = PROPERTY_EXTRACTION_PROMPT.format(
            schema="Entity: Foo\n  dbo properties: bar, baz",
            question="Who is Foo?",
        )
        assert "Who is Foo?" in result
        assert "bar, baz" in result

    def test_prompt_format_with_escaped_braces(self):
        safe_schema = "Entity: {Foo}".replace("{", "{{").replace("}", "}}")
        result = PROPERTY_EXTRACTION_PROMPT.format(
            schema=safe_schema, question="Who is Foo?",
        )
        assert "{Foo}" in result


class TestBuildPropertySchema:
    """Verify _build_property_schema produces correct schema summary."""

    def test_film_schema_includes_class_and_props(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        schema = _build_property_schema([proj])
        assert "Inception" in schema
        assert "Film" in schema
        assert "director" in schema

    def test_film_schema_includes_chains(self):
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        schema = _build_property_schema([proj])
        assert "Chain:" in schema
        assert "Person" in schema

    def test_multi_entity_schema(self):
        p1 = _film_profile()
        p2 = _person_profile()
        a = analyze_question("Who was born earlier?")
        schema = _build_property_schema([project(p1, a), project(p2, a)])
        assert "Inception" in schema
        assert "Albert Einstein" in schema


# ---------------------------------------------------------------------------
# Rule 4: parent_chain
# ---------------------------------------------------------------------------

class TestParentChainRule:
    def test_father_routes_sparql(self):
        """Father keyword + dbo:parent actual → SPARQL (via bridge_hop2 or parent_chain).
        bridge_hop2 fires first because father is in _BRIDGE_KEYWORDS + hop2 'born'."""
        profile = _person_profile()
        analysis = analyze_question("When was the father of Prince William born?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"
        # bridge_hop2 fires before parent_chain (rule 3 before rule 4)
        assert "bridge" in result[1] or "parent" in result[1]

    def test_mother_routes_sparql(self):
        """Mother keyword + dbo:parent actual → SPARQL."""
        profile = _person_profile()
        analysis = analyze_question("When was the mother of Einstein born?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"

    def test_parent_no_hop2_hits_direct_property(self):
        """Parent keyword without hop2 → bridge_keyword=None → hits direct_property."""
        profile = _person_profile()
        analysis = analyze_question("Who is the father of Prince William?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"
        # Without hop2, bridge_keyword is None → rules 3,4 skip → rule 5 fires
        assert "direct_property" in result[1]

    def test_parent_without_prop_returns_none(self):
        """Parent keyword but no parent property in actual → None."""
        profile = _person_profile(
            actual_property_set=frozenset({"dbo:occupation"}),
        )
        analysis = analyze_question("Who is the father of X?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        # "father" → target_properties = {dbo:father, dbp:father, dbo:parent, dbp:parent}
        # occupation doesn't match → None
        assert result is None


# ---------------------------------------------------------------------------
# Rule 5: direct_property
# ---------------------------------------------------------------------------

class TestDirectPropertyRule:
    def test_director_direct(self):
        """Direct 'Who directed X?' with dbo:director → SPARQL direct_property."""
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"
        assert "direct_property" in result[1]

    def test_birth_date_direct(self):
        """'When was X born?' with dbo:birthDate → SPARQL direct_property."""
        profile = _person_profile()
        analysis = analyze_question("When was Albert Einstein born?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"
        assert "direct_property" in result[1]

    def test_author_direct(self):
        """'Who wrote X?' with dbo:author → SPARQL direct_property."""
        profile = _book_profile()
        analysis = analyze_question("Who wrote Harry Potter?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"
        assert "direct_property" in result[1]

    def test_no_matching_property_returns_none(self):
        """No target property in actual → None."""
        profile = _film_profile(
            actual_property_set=frozenset({"dbo:runtime"}),
        )
        analysis = analyze_question("What is the capital of France?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is None


# ---------------------------------------------------------------------------
# Multi-projection routing (multiple entities)
# ---------------------------------------------------------------------------

class TestMultiProjectionRouting:
    def test_asymmetric_date_properties_does_not_fire_date_comparison(self):
        """Two entities: A has birthDate, B has only deathDate (different properties).
        Intersection is empty → direct_date_comparison must NOT fire.
        SPARQL would only return data for one entity; LLM cannot compare fairly.
        Falls through to a later rule (direct_property or None)."""
        p1 = _person_profile(
            uri="http://dbpedia.org/resource/A", label="A",
            actual_property_set=frozenset({"dbo:birthDate"}),
        )
        p2 = _person_profile(
            uri="http://dbpedia.org/resource/B", label="B",
            actual_property_set=frozenset({"dbo:deathDate"}),
        )
        analysis = analyze_question("Who was born earlier, A or B?")
        proj1 = project(p1, analysis)
        proj2 = project(p2, analysis)
        result = _rule_based_route(analysis, [proj1, proj2])
        # direct_date_comparison must not fire when properties differ across entities
        assert result is None or "date_comparison" not in result[1]

    def test_symmetric_date_properties_fires_date_comparison(self):
        """Both entities have birthDate → intersection non-empty → date comparison fires."""
        p1 = _person_profile(
            uri="http://dbpedia.org/resource/A", label="A",
            actual_property_set=frozenset({"dbo:birthDate"}),
        )
        p2 = _person_profile(
            uri="http://dbpedia.org/resource/B", label="B",
            actual_property_set=frozenset({"dbo:birthDate"}),
        )
        analysis = analyze_question("Who was born earlier, A or B?")
        proj1 = project(p1, analysis)
        proj2 = project(p2, analysis)
        result = _rule_based_route(analysis, [proj1, proj2])
        assert result is not None
        assert result[0] == "SPARQL"
        assert "date_comparison" in result[1]

    def test_multi_entity_no_overlap_returns_none(self):
        """Two entities with no relevant properties → None."""
        p1 = EntityProfile(
            uri="http://dbpedia.org/resource/A", label="A",
            all_classes=[], cached_classes=[],
            actual_properties={"dbo": [], "dbp": [], "values": {}},
            actual_property_set=frozenset({"dbo:runtime"}),
            all_2hop_chains={},
        )
        p2 = EntityProfile(
            uri="http://dbpedia.org/resource/B", label="B",
            all_classes=[], cached_classes=[],
            actual_properties={"dbo": [], "dbp": [], "values": {}},
            actual_property_set=frozenset({"dbo:budget"}),
            all_2hop_chains={},
        )
        analysis = analyze_question("What is the capital of France?")
        proj1 = project(p1, analysis)
        proj2 = project(p2, analysis)
        result = _rule_based_route(analysis, [proj1, proj2])
        assert result is None


# ---------------------------------------------------------------------------
# Bridge comparison via PROPERTY_RANGES fallback
# ---------------------------------------------------------------------------

class TestBridgeComparisonFallback:
    def test_property_ranges_fallback_no_chains(self):
        """Film with empty chains + no date properties but bridge 'director' in PROPERTY_RANGES
        → SPARQL via bridge_comparison fallback."""
        # Remove date properties to avoid direct_date_comparison firing first
        profile = _film_profile(
            actual_property_set=frozenset({"dbo:director", "dbo:producer"}),
            all_2hop_chains={},
        )
        analysis = analyze_question(
            "Which director was born earlier, the director of Inception or Titanic?"
        )
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"
        assert "bridge" in result[1]

    def test_direct_date_fires_before_fallback(self):
        """Film with releaseDate (a date property) → direct_date_comparison fires first."""
        profile = _film_profile()  # has dbo:releaseDate
        analysis = analyze_question(
            "Which director was born earlier, the director of Inception or Titanic?"
        )
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"
        # direct_date_comparison fires BEFORE bridge_comparison
        assert "direct_date_comparison" in result[1]


# ---------------------------------------------------------------------------
# verify_sparql_feasibility
# ---------------------------------------------------------------------------

class TestVerifySparqlFeasibility:
    def test_none_rule_returns_none(self):
        from routing import verify_sparql_feasibility
        profile = _person_profile()
        analysis = analyze_question("What is the meaning of life?")
        proj = project(profile, analysis)
        result = verify_sparql_feasibility(
            analysis, [proj], [], None, rule_result=None,
        )
        assert result is None

    def test_direct_property_passes_through(self):
        from routing import verify_sparql_feasibility
        profile = _person_profile()
        analysis = analyze_question("When was Einstein born?")
        proj = project(profile, analysis)
        rule = ("SPARQL", "rule:direct_property")
        result = verify_sparql_feasibility(
            analysis, [proj], [], None, rule_result=rule,
        )
        assert result == rule

    def test_direct_date_comparison_passes_through(self):
        from routing import verify_sparql_feasibility
        profile = _person_profile()
        analysis = analyze_question("Who was born earlier?")
        proj = project(profile, analysis)
        rule = ("SPARQL", "rule:direct_date_comparison")
        result = verify_sparql_feasibility(
            analysis, [proj], [], None, rule_result=rule,
        )
        assert result == rule

    def test_direct_country_comparison_passes_through(self):
        from routing import verify_sparql_feasibility
        rule = ("SPARQL", "rule:direct_country_comparison")
        analysis = analyze_question("Same country?")
        profile = _person_profile()
        proj = project(profile, analysis)
        result = verify_sparql_feasibility(
            analysis, [proj], [], None, rule_result=rule,
        )
        assert result == rule

    def test_bridge_complete_projection_verified(self):
        """Bridge rule + complete projection → SPARQL+ask_verified."""
        from unittest.mock import MagicMock

        from routing import verify_sparql_feasibility
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        mock_kg = MagicMock()
        mock_kg.query.return_value = {"boolean": True}
        entities = _make_entities("Inception")
        rule = ("SPARQL", "rule:bridge_hop2")
        result = verify_sparql_feasibility(
            analysis, [proj], entities, mock_kg, rule_result=rule,
        )
        assert result is not None
        assert result[0] == "SPARQL"
        assert "ask_verified" in result[1]

    def test_bridge_incomplete_ask_verified(self):
        """Bridge + incomplete projection + ASK returns True → SPARQL+ask_verified."""
        from unittest.mock import MagicMock

        from routing import verify_sparql_feasibility
        # Film without director in actual → incomplete
        profile = _film_profile(
            actual_properties={"dbo": ["releaseDate"], "dbp": [], "values": {}},
            actual_property_set=frozenset({"dbo:releaseDate"}),
        )
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        mock_kg = MagicMock()
        mock_kg.query.return_value = {"boolean": True}
        entities = _make_entities("Inception")
        rule = ("SPARQL", "rule:bridge_hop2")
        result = verify_sparql_feasibility(
            analysis, [proj], entities, mock_kg, rule_result=rule,
        )
        assert result is not None
        assert result[0] == "SPARQL"
        assert "ask_verified" in result[1]

    def test_bridge_incomplete_ask_missing(self):
        """Bridge + incomplete projection + ASK returns False → DECOMPOSE."""
        from unittest.mock import MagicMock

        from routing import verify_sparql_feasibility
        profile = _film_profile(
            actual_properties={"dbo": ["releaseDate"], "dbp": [], "values": {}},
            actual_property_set=frozenset({"dbo:releaseDate"}),
        )
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        mock_kg = MagicMock()
        mock_kg.query.return_value = {"boolean": False}
        entities = _make_entities("Inception")
        rule = ("SPARQL", "rule:bridge_hop2")
        result = verify_sparql_feasibility(
            analysis, [proj], entities, mock_kg, rule_result=rule,
        )
        assert result is not None
        assert result[0] == "DECOMPOSE"
        assert "ask_missing" in result[1]

    def test_bridge_empty_entities_passthrough(self):
        """Bridge with empty entities list → returns original rule."""
        from routing import verify_sparql_feasibility
        profile = _film_profile(
            actual_properties={"dbo": ["releaseDate"], "dbp": [], "values": {}},
            actual_property_set=frozenset({"dbo:releaseDate"}),
        )
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        rule = ("SPARQL", "rule:bridge_hop2")
        result = verify_sparql_feasibility(
            analysis, [proj], [], None, rule_result=rule,
        )
        assert result is not None
        assert result[0] == "SPARQL"  # passes through unchanged

    def test_bridge_ask_exception_continues(self):
        """ASK query raises Exception → continues to next entity or DECOMPOSE."""
        from unittest.mock import MagicMock

        from routing import verify_sparql_feasibility
        profile = _film_profile(
            actual_properties={"dbo": ["releaseDate"], "dbp": [], "values": {}},
            actual_property_set=frozenset({"dbo:releaseDate"}),
        )
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        mock_kg = MagicMock()
        mock_kg.query.side_effect = Exception("SPARQL timeout")
        entities = _make_entities("Inception")
        rule = ("SPARQL", "rule:bridge_hop2")
        result = verify_sparql_feasibility(
            analysis, [proj], entities, mock_kg, rule_result=rule,
        )
        # Exception → can't verify → DECOMPOSE
        assert result is not None
        assert result[0] == "DECOMPOSE"

    # --- Comparison-aware ASK verification (all vs any) ---

    def test_comparison_both_verified_routes_sparql(self):
        """Comparison + bridge: both entities pass ASK → SPARQL+ask_verified."""
        from unittest.mock import MagicMock

        from routing import verify_sparql_feasibility
        profile = _film_profile(
            actual_property_set=frozenset({"dbo:director", "dbo:producer"}),
            all_2hop_chains={},
        )
        analysis = analyze_question(
            "Which director was born earlier, the director of Inception or Titanic?"
        )
        proj = project(profile, analysis)
        mock_kg = MagicMock()
        # Both entities return True
        mock_kg.query.return_value = {"boolean": True}
        entities = _make_entities("Inception", "Titanic")
        rule = ("SPARQL", "rule:bridge_comparison")
        result = verify_sparql_feasibility(
            analysis, [proj], entities, mock_kg, rule_result=rule,
        )
        assert result is not None
        assert result[0] == "SPARQL"
        assert "ask_verified" in result[1]
        # Both entities must be checked
        assert mock_kg.query.call_count == 2

    def test_comparison_one_fails_routes_decompose(self):
        """Comparison + bridge: first entity True, second False → DECOMPOSE+ask_partial."""
        from unittest.mock import MagicMock

        from routing import verify_sparql_feasibility
        profile = _film_profile(
            actual_property_set=frozenset({"dbo:director", "dbo:producer"}),
            all_2hop_chains={},
        )
        analysis = analyze_question(
            "Which director was born earlier, the director of Inception or Titanic?"
        )
        proj = project(profile, analysis)
        mock_kg = MagicMock()
        # First entity True, second False → partial coverage
        mock_kg.query.side_effect = [{"boolean": True}, {"boolean": False}]
        entities = _make_entities("Inception", "Titanic")
        rule = ("SPARQL", "rule:bridge_comparison")
        result = verify_sparql_feasibility(
            analysis, [proj], entities, mock_kg, rule_result=rule,
        )
        assert result is not None
        assert result[0] == "DECOMPOSE"
        assert "ask_partial" in result[1]

    def test_comparison_exception_counts_as_failure(self):
        """Comparison + bridge: ASK exception for one entity → DECOMPOSE."""
        from unittest.mock import MagicMock

        from routing import verify_sparql_feasibility
        profile = _film_profile(
            actual_property_set=frozenset({"dbo:director", "dbo:producer"}),
            all_2hop_chains={},
        )
        analysis = analyze_question(
            "Which director was born earlier, the director of Inception or Titanic?"
        )
        proj = project(profile, analysis)
        mock_kg = MagicMock()
        # First entity ok, second throws → partial
        mock_kg.query.side_effect = [{"boolean": True}, Exception("timeout")]
        entities = _make_entities("Inception", "Titanic")
        rule = ("SPARQL", "rule:bridge_comparison")
        result = verify_sparql_feasibility(
            analysis, [proj], entities, mock_kg, rule_result=rule,
        )
        assert result is not None
        assert result[0] == "DECOMPOSE"

    def test_comparison_no_uri_entities_passthrough(self):
        """Comparison + bridge: all entities have uri=None → passthrough (no ASK possible)."""
        from routing import verify_sparql_feasibility
        profile = _film_profile(
            actual_property_set=frozenset({"dbo:director"}),
            all_2hop_chains={},
        )
        analysis = analyze_question(
            "Which director was born earlier, the director of Inception or Titanic?"
        )
        proj = project(profile, analysis)
        entities = [
            Entity(uri=None, label="Inception", source="test"),
            Entity(uri=None, label="Titanic", source="test"),
        ]
        rule = ("SPARQL", "rule:bridge_comparison")
        result = verify_sparql_feasibility(
            analysis, [proj], entities, None, rule_result=rule,
        )
        assert result is not None
        # No URI entities → passthrough (original rule, no ASK attempted)
        assert result[0] == "SPARQL"

    def test_non_comparison_any_logic_preserved(self):
        """Non-comparison bridge: first entity True → SPARQL (any suffices, no need to check all)."""
        from unittest.mock import MagicMock

        from routing import verify_sparql_feasibility
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        mock_kg = MagicMock()
        mock_kg.query.return_value = {"boolean": True}
        # Two entities but NOT a comparison question
        entities = _make_entities("Inception", "Titanic")
        rule = ("SPARQL", "rule:bridge_hop2")
        result = verify_sparql_feasibility(
            analysis, [proj], entities, mock_kg, rule_result=rule,
        )
        assert result is not None
        assert result[0] == "SPARQL"
        assert "ask_verified" in result[1]
        # any() short-circuits: only first entity checked
        assert mock_kg.query.call_count == 1


# ---------------------------------------------------------------------------
# entities_to_schema_dicts
# ---------------------------------------------------------------------------

class TestEntitiesToSchemaDicts:
    def test_basic(self):
        from routing import entities_to_schema_dicts
        entities = _make_entities("Albert Einstein", "Nikola Tesla")
        dicts = entities_to_schema_dicts(entities)
        assert len(dicts) == 2
        assert dicts[0]["name"] == "Albert Einstein"
        assert "Albert_Einstein" in dicts[0]["uri"]

    def test_filters_no_uri(self):
        from routing import entities_to_schema_dicts
        entities = [Entity(uri=None, label="Ghost", source="test")]
        dicts = entities_to_schema_dicts(entities)
        assert len(dicts) == 0

    def test_empty_list(self):
        from routing import entities_to_schema_dicts
        assert entities_to_schema_dicts([]) == []


# ---------------------------------------------------------------------------
# _sparql_results_to_entities
# ---------------------------------------------------------------------------

class TestSparqlResultsToEntities:
    def test_seed_source(self):
        from routing import _sparql_results_to_entities
        uri = "http://dbpedia.org/resource/Albert_Einstein"
        results = [{"entity": uri}]
        entities = _sparql_results_to_entities(results, seed_uris={uri})
        assert len(entities) == 1
        assert entities[0].source == "seed"

    def test_context_source(self):
        from routing import _sparql_results_to_entities
        uri = "http://dbpedia.org/resource/Albert_Einstein"
        results = [{"entity": uri}]
        entities = _sparql_results_to_entities(results, seed_uris=set())
        assert len(entities) == 1
        assert entities[0].source == "sparql_context"

    def test_label_fallback_from_uri(self):
        from routing import _sparql_results_to_entities
        uri = "http://dbpedia.org/resource/Albert_Einstein"
        results = [{"entity": uri}]  # no label key
        entities = _sparql_results_to_entities(results, seed_uris=set())
        assert entities[0].label == "Albert Einstein"

    def test_explicit_label_used(self):
        from routing import _sparql_results_to_entities
        uri = "http://dbpedia.org/resource/Albert_Einstein"
        results = [{"entity": uri, "label": "Einstein, A."}]
        entities = _sparql_results_to_entities(results, seed_uris=set())
        assert entities[0].label == "Einstein, A."

    def test_relation_extraction(self):
        from routing import _sparql_results_to_entities
        uri = "http://dbpedia.org/resource/Albert_Einstein"
        results = [{"entity": uri, "p": "http://dbpedia.org/ontology/birthPlace"}]
        entities = _sparql_results_to_entities(results, seed_uris=set())
        assert entities[0].discovery_relation == "birthPlace"

    def test_dedup_same_uri(self):
        from routing import _sparql_results_to_entities
        uri = "http://dbpedia.org/resource/Albert_Einstein"
        results = [{"entity": uri}, {"entity": uri}]
        entities = _sparql_results_to_entities(results, seed_uris=set())
        assert len(entities) == 1

    def test_local_name_to_uri(self):
        from routing import _sparql_results_to_entities
        results = [{"entity": "Albert Einstein"}]
        entities = _sparql_results_to_entities(results, seed_uris=set())
        assert entities[0].uri == "http://dbpedia.org/resource/Albert_Einstein"

    def test_empty_value_skipped(self):
        from routing import _sparql_results_to_entities
        results = [{"entity": "", "related": ""}]
        entities = _sparql_results_to_entities(results, seed_uris=set())
        assert len(entities) == 0

    def test_empty_results(self):
        from routing import _sparql_results_to_entities
        entities = _sparql_results_to_entities([], seed_uris=set())
        assert len(entities) == 0


# ---------------------------------------------------------------------------
# Cross-module integration: question → analysis → projection → routing
# ---------------------------------------------------------------------------

class TestIntegrationPipeline:
    def test_full_pipeline_direct_sparql(self):
        """'Who directed Inception?' → analysis → Film projection → SPARQL."""
        profile = _film_profile()
        analysis = analyze_question("Who directed Inception?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"

    def test_full_pipeline_bridge_sparql(self):
        """'When was the director born?' → bridge analysis → Film projection → SPARQL."""
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"
        assert "bridge" in result[1] or "hop2" in result[1]

    def test_full_pipeline_bridge_missing_hop1(self):
        """Film without director → bridge should NOT match direct_property.
        May still match via PROPERTY_RANGES fallback."""
        profile = _film_profile(
            actual_properties={"dbo": ["releaseDate"], "dbp": [], "values": {}},
            actual_property_set=frozenset({"dbo:releaseDate"}),
            all_2hop_chains={},
        )
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        if result is not None:
            # Should not be direct_property (no director in actual)
            assert "direct_property" not in result[1]

    def test_full_pipeline_comparison_two_entities(self):
        """Two Person entities with birthDate → date comparison → SPARQL."""
        p1 = _person_profile(uri="http://dbpedia.org/resource/A", label="A")
        p2 = _person_profile(uri="http://dbpedia.org/resource/B", label="B")
        analysis = analyze_question("Who was born earlier, A or B?")
        proj1 = project(p1, analysis)
        proj2 = project(p2, analysis)
        result = _rule_based_route(analysis, [proj1, proj2])
        assert result is not None
        assert result[0] == "SPARQL"
        assert "date_comparison" in result[1]

    def test_full_pipeline_no_keywords(self):
        """No keyword match → None (LLM decides)."""
        profile = _film_profile()
        analysis = analyze_question("What is the meaning of life?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is None

    def test_full_pipeline_book_author_birth(self):
        """Book: 'Where was the author born?' → bridge → SPARQL."""
        profile = _book_profile()
        analysis = analyze_question("Where was the author of Harry Potter born?")
        proj = project(profile, analysis)
        result = _rule_based_route(analysis, [proj])
        assert result is not None
        assert result[0] == "SPARQL"

    def test_rendering_roundtrip_bridge(self):
        """Bridge question: all renderers produce consistent output."""
        profile = _book_profile()
        analysis = analyze_question("Where was the author of Harry Potter born?")
        proj = project(profile, analysis)

        routing_out = proj.render_for_routing()
        sparql_out = proj.render_for_sparql()
        text_out = proj.render_for_text()

        assert "Completeness:" in routing_out
        assert "Entity: Harry Potter" in sparql_out
        assert "Person" in text_out
        assert "via" in text_out  # bridge shows "via author"


class TestIncompleteCoverage:
    """Tests for _has_incomplete_entity_coverage() guard."""

    def test_detects_missing_entity(self):
        entities = _make_entities("The Nice Guys")
        q = "Which film has the director who was born first, Piccoli Fuochi or The Nice Guys?"
        assert _has_incomplete_entity_coverage(q, entities)

    def test_complete_coverage_passes(self):
        entities = _make_entities("Film A", "Film B")
        q = "Which was released first, Film A or Film B?"
        assert not _has_incomplete_entity_coverage(q, entities)

    def test_non_comparison_skips(self):
        entities = _make_entities("X")
        q = "Who directed X?"
        assert not _has_incomplete_entity_coverage(q, entities)

    def test_no_or_pattern_skips(self):
        entities = _make_entities("Film A")
        q = "Which film was released first?"
        assert not _has_incomplete_entity_coverage(q, entities)

    def test_case_insensitive_label_match(self):
        entities = _make_entities("the nice guys")
        q = "Which was released first, Film X or The Nice Guys?"
        assert _has_incomplete_entity_coverage(q, entities)

    def test_and_pattern_detects_missing(self):
        """'both X and Y' should detect entity names."""
        entities = _make_entities("X")
        q = "Are both X and Y located in the same country?"
        assert _has_incomplete_entity_coverage(q, entities)

    def test_comma_and_pattern(self):
        """', X and Y?' should detect entity names."""
        entities = _make_entities("X")
        q = "Which was released first, X and Y?"
        assert _has_incomplete_entity_coverage(q, entities)


# ---------------------------------------------------------------------------
# enrich_analysis
# ---------------------------------------------------------------------------

class TestEnrichAnalysis:
    """Tests for LLM property enrichment of QuestionAnalysis."""

    def test_empty_properties_returns_base(self):
        from question_analysis import enrich_analysis
        base = analyze_question("What is the capital of France?")
        enriched = enrich_analysis(base, [])
        assert enriched is base

    def test_adds_prefixed_properties(self):
        from question_analysis import enrich_analysis
        base = analyze_question("What is the meaning of life?")
        enriched = enrich_analysis(base, ["director"])
        assert "dbo:director" in enriched.target_properties
        assert "dbp:director" in enriched.target_properties

    def test_detects_bridge_structure(self):
        """Two properties where first is in PROPERTY_RANGES → bridge detected."""
        from question_analysis import enrich_analysis
        base = analyze_question("What is the meaning of life?")
        enriched = enrich_analysis(base, ["director", "birthDate"])
        assert enriched.bridge_property == "director"
        assert enriched.hop2_keyword == "birthDate"
        assert enriched.hop_structure == "bridge"
        assert enriched.expected_target_type == "Person"

    def test_preserves_existing_bridge(self):
        """If base already has bridge, don't override."""
        from question_analysis import enrich_analysis
        base = analyze_question("When was the director of Inception born?")
        assert base.bridge_property == "director"
        enriched = enrich_analysis(base, ["starring"])
        assert enriched.bridge_property == "director"  # preserved

    def test_single_property_no_bridge(self):
        """Single LLM property → adds to targets, no bridge."""
        from question_analysis import enrich_analysis
        base = analyze_question("What is the meaning of life?")
        enriched = enrich_analysis(base, ["birthDate"])
        assert enriched.bridge_property is None
        assert "dbo:birthDate" in enriched.target_properties

    def test_enriched_enables_routing(self):
        """LLM-enriched analysis enables rule-based routing that keyword missed."""
        from question_analysis import enrich_analysis
        # "helmed" is not in KEYWORD_TO_PROPERTIES → no target_properties
        base = analyze_question("Who helmed Inception?")
        assert not base.target_properties
        # LLM extracts "director"
        enriched = enrich_analysis(base, ["director"])
        # Now projection should match
        profile = _film_profile()
        proj = project(profile, enriched)
        result = _rule_based_route(enriched, [proj])
        assert result is not None
        assert result[0] == "SPARQL"
        assert "direct_property" in result[1]


# ---------------------------------------------------------------------------
# _extract_properties_llm
# ---------------------------------------------------------------------------

class TestExtractPropertiesLlm:
    """Tests for LLM property extraction parsing."""

    def test_parses_comma_separated(self):
        from unittest.mock import MagicMock, patch

        from routing import _extract_properties_llm
        mock_client = MagicMock()
        with patch("llm_provider.call_llm", return_value="director, birthDate"):
            result = _extract_properties_llm("q", "schema", mock_client, "model")
        assert result == ["director", "birthDate"]

    def test_strips_prefixes(self):
        from unittest.mock import MagicMock, patch

        from routing import _extract_properties_llm
        mock_client = MagicMock()
        with patch("llm_provider.call_llm", return_value="dbo:director, dbp:birthDate"):
            result = _extract_properties_llm("q", "schema", mock_client, "model")
        assert result == ["director", "birthDate"]

    def test_none_response(self):
        from unittest.mock import MagicMock, patch

        from routing import _extract_properties_llm
        mock_client = MagicMock()
        with patch("llm_provider.call_llm", return_value="NONE"):
            result = _extract_properties_llm("q", "schema", mock_client, "model")
        assert result == []

    def test_empty_response(self):
        from unittest.mock import MagicMock, patch

        from routing import _extract_properties_llm
        mock_client = MagicMock()
        with patch("llm_provider.call_llm", return_value=""):
            result = _extract_properties_llm("q", "schema", mock_client, "model")
        assert result == []

    def test_single_property(self):
        from unittest.mock import MagicMock, patch

        from routing import _extract_properties_llm
        mock_client = MagicMock()
        with patch("llm_provider.call_llm", return_value="director"):
            result = _extract_properties_llm("q", "schema", mock_client, "model")
        assert result == ["director"]


# ---------------------------------------------------------------------------
# H1: route_question Stage 2 LLM failure → TEXT fallback
# ---------------------------------------------------------------------------

class TestRouteQuestionLlmFailure:
    """Stage 2 LLM crash must fall through to TEXT, not raise."""

    def test_llm_exception_falls_to_text(self):
        from unittest.mock import MagicMock, patch

        from routing import route_question
        mock_kg = MagicMock()
        mock_llm = MagicMock()
        entities = _make_entities("Inception")
        profile = _film_profile()
        analysis = analyze_question("What is the meaning of life?")
        proj = project(profile, analysis)

        with patch("llm_provider.call_llm", side_effect=RuntimeError("LLM down")):
            route, source = route_question(
                "What is the meaning of life?",
                entities, mock_kg, mock_llm, "model",
                projections=[proj],
            )
        assert route == "TEXT"
        assert "no_match" in source


# ---------------------------------------------------------------------------
# H2: Order-invariant bridge detection in enrich_analysis
# ---------------------------------------------------------------------------

class TestEnrichBridgeOrdering:
    """Bridge detection must pick the navigational property regardless of order."""

    def test_director_birthdate_order(self):
        """["director", "birthDate"] → director is bridge (Person has birthDate)."""
        from question_analysis import enrich_analysis
        base = analyze_question("What is the meaning of life?")
        enriched = enrich_analysis(base, ["director", "birthDate"])
        assert enriched.bridge_property == "director"
        assert enriched.hop2_keyword == "birthDate"

    def test_birthdate_director_reversed(self):
        """["birthDate", "director"] → director still wins (Person has birthDate)."""
        from question_analysis import enrich_analysis
        base = analyze_question("What is the meaning of life?")
        enriched = enrich_analysis(base, ["birthDate", "director"])
        assert enriched.bridge_property == "director"
        assert enriched.hop2_keyword == "birthDate"

    def test_birthplace_country(self):
        """["birthPlace", "country"] → birthPlace is bridge (Place has country)."""
        from question_analysis import enrich_analysis
        base = analyze_question("What is the meaning of life?")
        enriched = enrich_analysis(base, ["birthPlace", "country"])
        assert enriched.bridge_property == "birthPlace"
        assert enriched.hop2_keyword == "country"

    def test_country_birthplace_reversed(self):
        """["country", "birthPlace"] → birthPlace still wins (Place has country)."""
        from question_analysis import enrich_analysis
        base = analyze_question("What is the meaning of life?")
        enriched = enrich_analysis(base, ["country", "birthPlace"])
        assert enriched.bridge_property == "birthPlace"
        assert enriched.hop2_keyword == "country"

    def test_two_leaf_properties_no_bridge(self):
        """["birthDate", "deathDate"] → no bridge (neither covers the other)."""
        from question_analysis import enrich_analysis
        base = analyze_question("What is the meaning of life?")
        enriched = enrich_analysis(base, ["birthDate", "deathDate"])
        assert enriched.bridge_property is None
        assert enriched.hop_structure != "bridge"

    def test_author_nationality(self):
        """["author", "nationality"] → author is bridge (Person has nationality)."""
        from question_analysis import enrich_analysis
        base = analyze_question("What is the meaning of life?")
        enriched = enrich_analysis(base, ["author", "nationality"])
        assert enriched.bridge_property == "author"
        assert enriched.hop2_keyword == "nationality"


# ---------------------------------------------------------------------------
# route_source substring match verification
# ---------------------------------------------------------------------------

class TestRouteSourceSubstringMatch:
    """Verify bridge_patterns substring check handles all route_source forms."""

    def test_plain_bridge_hop2(self):
        bridge_patterns = ("bridge_comparison", "bridge_hop2", "parent_chain")
        assert any(p in "rule:bridge_hop2" for p in bridge_patterns)

    def test_ask_verified_bridge(self):
        bridge_patterns = ("bridge_comparison", "bridge_hop2", "parent_chain")
        assert any(p in "rule:bridge_hop2+ask_verified" for p in bridge_patterns)

    def test_llm_prop_bridge(self):
        bridge_patterns = ("bridge_comparison", "bridge_hop2", "parent_chain")
        assert any(p in "llm_prop:rule:bridge_hop2+ask_verified" for p in bridge_patterns)

    def test_llm_prop_parent(self):
        bridge_patterns = ("bridge_comparison", "bridge_hop2", "parent_chain")
        assert any(p in "llm_prop:rule:parent_chain+ask_verified" for p in bridge_patterns)

    def test_direct_property_no_match(self):
        bridge_patterns = ("bridge_comparison", "bridge_hop2", "parent_chain")
        assert not any(p in "rule:direct_property" for p in bridge_patterns)

    def test_no_schema_no_match(self):
        bridge_patterns = ("bridge_comparison", "bridge_hop2", "parent_chain")
        assert not any(p in "no_schema_fallback" for p in bridge_patterns)


# ---------------------------------------------------------------------------
# No-URI guard in route_question
# ---------------------------------------------------------------------------

class TestNoEntityUriGuard:
    """route_question returns TEXT when no entities have URIs."""

    def test_all_entities_no_uri(self):
        from unittest.mock import MagicMock

        from routing import route_question
        entities = [Entity(uri=None, label="Albert Einstein")]
        route, reason = route_question(
            "When was Einstein born?", entities,
            kg_client=MagicMock(), llm_client=MagicMock(), model="test",
        )
        assert route == "TEXT"
        assert "no_entity_uris" in reason

    def test_empty_entities_returns_text(self):
        from unittest.mock import MagicMock

        from routing import route_question
        route, reason = route_question(
            "When was Einstein born?", [],
            kg_client=MagicMock(), llm_client=MagicMock(), model="test",
        )
        assert route == "TEXT"


# ---------------------------------------------------------------------------
# 2-hop ASK verification
# ---------------------------------------------------------------------------

class TestTwoHopAskVerification:
    """verify_sparql_feasibility checks both hops for bridge routes."""

    def test_ask_includes_hop2_property(self):
        """ASK query must contain hop2 property when bridge has hop2_properties."""
        from unittest.mock import MagicMock

        from routing import verify_sparql_feasibility
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        mock_kg = MagicMock()
        mock_kg.query.return_value = {"boolean": True}
        entities = _make_entities("Inception")
        rule = ("SPARQL", "rule:bridge_hop2")
        verify_sparql_feasibility(
            analysis, [proj], entities, mock_kg, rule_result=rule,
        )
        # Inspect the actual ASK query sent to kg_client
        call_args = mock_kg.query.call_args[0][0]
        assert "director" in call_args
        assert "birthDate" in call_args or "birthPlace" in call_args

    def test_hop2_missing_returns_decompose(self):
        """When 2-hop ASK returns False (hop2 missing), route is DECOMPOSE."""
        from unittest.mock import MagicMock

        from routing import verify_sparql_feasibility
        profile = _film_profile()
        analysis = analyze_question("When was the director of Inception born?")
        proj = project(profile, analysis)
        mock_kg = MagicMock()
        mock_kg.query.return_value = {"boolean": False}
        entities = _make_entities("Inception")
        rule = ("SPARQL", "rule:bridge_hop2")
        result = verify_sparql_feasibility(
            analysis, [proj], entities, mock_kg, rule_result=rule,
        )
        assert result[0] == "DECOMPOSE"
        assert "ask_missing" in result[1]
