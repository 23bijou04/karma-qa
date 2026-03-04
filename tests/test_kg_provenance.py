"""Tests for Phase 1+3: KG provenance, kg_facts injection, relation-guided section boosting."""

from dbpedia_qa.models import Entity, HybridResult


class TestEntityDiscoveryRelation:
    def test_entity_default_none(self):
        e = Entity(uri="http://dbpedia.org/resource/X", label="X")
        assert e.discovery_relation is None

    def test_entity_with_relation(self):
        e = Entity(
            uri="http://dbpedia.org/resource/X", label="X",
            discovery_relation="director",
        )
        assert e.discovery_relation == "director"

    def test_entity_hash_ignores_relation(self):
        """discovery_relation must NOT affect hash/equality (hash uses uri only)."""
        e1 = Entity(uri="http://dbpedia.org/resource/X", label="X")
        e2 = Entity(
            uri="http://dbpedia.org/resource/X", label="X",
            discovery_relation="director",
        )
        assert e1 == e2
        assert hash(e1) == hash(e2)


class TestHybridResultProvenance:
    def test_default_fields(self):
        r = HybridResult("answer", "sparql")
        assert r.sparql_query is None
        assert r.kg_context == ""

    def test_with_provenance(self):
        r = HybridResult(
            "London", "sparql", "rule:bridge_hop2",
            sparql_query="SELECT ?x WHERE { ... }",
        )
        assert r.sparql_query == "SELECT ?x WHERE { ... }"
        assert r.method == "sparql"

    def test_with_kg_context(self):
        r = HybridResult(
            "London", "text", "llm",
            sparql_fallback=True,
            kg_context="Known facts from knowledge graph:\n- director: Christopher Nolan\n\n",
        )
        assert "Christopher Nolan" in r.kg_context
        assert r.sparql_fallback is True


class TestAssembleContextKgFacts:
    def test_kg_facts_reserves_budget_not_prepended(self):
        """kg_facts reserves budget space but is NOT prepended (prompt handles it)."""
        from graphrag_wiki import assemble_context
        raw_texts = {"Page A": "Content about topic with keywords."}
        kg_facts = "Known facts from knowledge graph:\n- director: Nolan\n\n"
        context, _ = assemble_context(
            raw_texts, "Who directed the film?", kg_facts=kg_facts,
        )
        # kg_facts should NOT appear in context (prompt template injects it)
        assert "Known facts" not in context
        assert "Page A" in context

    def test_empty_kg_facts_backward_compat(self):
        from graphrag_wiki import assemble_context
        raw_texts = {"Page A": "Content about topic."}
        context, _ = assemble_context(raw_texts, "test question")
        assert len(context) > 0
        assert "Known facts" not in context

    def test_kg_facts_counted_against_budget(self):
        """Large kg_facts reduces wiki text room (budget reservation)."""
        from graphrag_wiki import assemble_context
        big_facts = "x" * 8000  # 8K reserved for kg_facts
        raw_texts = {"Page A": "y " * 10000}  # 20K of wiki text
        context_with, _ = assemble_context(
            raw_texts, "test", kg_facts=big_facts,
        )
        context_without, _ = assemble_context(
            raw_texts, "test",
        )
        # With big kg_facts, wiki text should be significantly shorter
        assert len(context_with) < len(context_without)

    def test_relation_hints_param_accepted(self):
        """relation_hints param is accepted without error."""
        from graphrag_wiki import assemble_context
        raw_texts = {"Page A": "Content about topic."}
        context, _ = assemble_context(
            raw_texts, "test", relation_hints={"Page A": "director"},
        )
        assert len(context) > 0


class TestRelationSectionBoosting:
    def test_boost_early_life_for_director(self):
        """Paragraphs mentioning 'early life' get 2x boost with director hint."""
        from graphrag_wiki import extract_relevant_sections
        text = (
            "== Filmography ==\nDirected many films about London.\n\n"
            "== Early life ==\nBorn in London in 1970. Grew up in the city.\n\n"
            "== Awards ==\nWon Academy Award in London ceremony."
        )
        keywords = ["born", "london"]
        with_hint = extract_relevant_sections(
            text, keywords, 500, relation_hint="director",
        )
        assert "Born in London" in with_hint

    def test_no_boost_without_hint(self):
        """Backward compat: no relation_hint produces same behavior."""
        from graphrag_wiki import extract_relevant_sections
        text = "Some content about the topic with relevant keywords."
        result = extract_relevant_sections(text, ["topic", "relevant"], 500)
        assert len(result) > 0

    def test_boost_does_not_affect_unmatched_relation(self):
        """Unknown relation_hint has no boost effect."""
        from graphrag_wiki import extract_relevant_sections
        text = (
            "== Early life ==\nBorn in London.\n\n"
            "== Career ==\nWorked in film industry."
        )
        keywords = ["born", "london"]
        no_hint = extract_relevant_sections(text, keywords, 500)
        unknown_hint = extract_relevant_sections(
            text, keywords, 500, relation_hint="unknownRelation",
        )
        assert no_hint == unknown_hint

    def test_relation_section_boost_dict_has_key_relations(self):
        """RELATION_SECTION_BOOST must cover key predicates."""
        from graphrag_wiki import RELATION_SECTION_BOOST
        for rel in ["director", "author", "birthPlace", "deathDate", "spouse"]:
            assert rel in RELATION_SECTION_BOOST

    def test_assemble_context_passes_relation_hints(self):
        """relation_hints reach extract_relevant_sections via assemble_context."""
        from graphrag_wiki import assemble_context
        text = (
            "== Filmography ==\nDirected many films about London.\n\n"
            "== Early life ==\nBorn in London in 1970. Grew up in the city.\n\n"
            "== Awards ==\nWon Academy Award in London ceremony."
        )
        raw_texts = {"Christopher Nolan": text}
        context, _ = assemble_context(
            raw_texts, "Where was born in London?",
            relation_hints={"Christopher Nolan": "director"},
        )
        assert "Born in London" in context
