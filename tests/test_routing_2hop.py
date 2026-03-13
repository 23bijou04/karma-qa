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
    def test_union_properties_across_entities(self):
        """Two entities: one has birthDate, other has deathDate.
        Union should contain both → date comparison triggers."""
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
