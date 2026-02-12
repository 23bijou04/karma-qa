"""Basic tests for DBpedia GraphRAG components."""

import pytest

from dbpedia_qa.models import Entity, Triple, QuestionType, Subgraph
from dbpedia_qa.config import Config, WIKIDATA_TO_DBPEDIA_PROPERTIES
from dbpedia_qa.question_processing.classifier import QuestionClassifier
from dbpedia_qa.question_processing.decomposer import QuestionDecomposer
from dbpedia_qa.dbpedia.property_mapper import PropertyMapper
from dbpedia_qa.context.textualizer import TripleTextualizer
from dbpedia_qa.evaluation.metrics import (
    normalize_answer,
    compute_em,
    compute_f1,
    compute_supporting_facts_f1,
)


class TestConfig:
    """Test configuration."""

    def test_default_config(self):
        config = Config()
        assert config.dbpedia.sparql_endpoint == "https://dbpedia.org/sparql"
        assert config.traversal.max_hops == 3
        assert config.traversal.beam_width == 10

    def test_property_mapping_exists(self):
        assert "P569" in WIKIDATA_TO_DBPEDIA_PROPERTIES  # birth date
        assert "P570" in WIKIDATA_TO_DBPEDIA_PROPERTIES  # death date
        assert "P57" in WIKIDATA_TO_DBPEDIA_PROPERTIES   # director


class TestModels:
    """Test data models."""

    def test_entity_creation(self):
        entity = Entity(
            uri="http://dbpedia.org/resource/Albert_Einstein",
            label="Albert Einstein",
        )
        assert entity.label == "Albert Einstein"
        assert entity.local_name == "Albert Einstein"

    def test_triple_creation(self):
        triple = Triple(
            subject="http://dbpedia.org/resource/Albert_Einstein",
            predicate="http://dbpedia.org/ontology/birthDate",
            obj="1879-03-14",
        )
        assert "1879-03-14" in triple.obj

    def test_triple_to_natural_language(self):
        triple = Triple(
            subject="http://dbpedia.org/resource/Einstein",
            subject_label="Albert Einstein",
            predicate="http://dbpedia.org/ontology/birthDate",
            predicate_label="birth date",
            obj="1879-03-14",
        )
        text = triple.to_natural_language()
        assert "Albert Einstein" in text
        assert "1879" in text

    def test_subgraph(self):
        subgraph = Subgraph()
        entity = Entity(uri="http://dbpedia.org/resource/Test", label="Test")
        subgraph.add_entity(entity)
        assert len(subgraph.entities) == 1

        # Adding same entity shouldn't duplicate
        subgraph.add_entity(entity)
        assert len(subgraph.entities) == 1


class TestQuestionClassifier:
    """Test question classification."""

    def test_comparison_question(self):
        classifier = QuestionClassifier()
        qtype = classifier.classify("Who was born first, Einstein or Lincoln?")
        assert qtype == QuestionType.COMPARISON

    def test_inference_question(self):
        classifier = QuestionClassifier()
        qtype = classifier.classify("Who is the paternal grandfather of John?")
        assert qtype == QuestionType.INFERENCE

    def test_compositional_question(self):
        classifier = QuestionClassifier()
        qtype = classifier.classify(
            "Who is the founder of the company that distributed La La Land?"
        )
        assert qtype == QuestionType.COMPOSITIONAL

    def test_bridge_comparison_question(self):
        classifier = QuestionClassifier()
        qtype = classifier.classify(
            "Which film has the director born first, La La Land or Tenet?"
        )
        assert qtype == QuestionType.BRIDGE_COMPARISON

    def test_extract_comparison_entities(self):
        classifier = QuestionClassifier()
        entities = classifier.extract_comparison_entities(
            "Who was born first, Einstein or Lincoln?"
        )
        assert "Einstein" in entities
        assert "Lincoln" in entities


class TestQuestionDecomposer:
    """Test question decomposition."""

    def test_comparison_decomposition(self):
        decomposer = QuestionDecomposer()
        subs = decomposer.decompose(
            "Who was born first, Einstein or Lincoln?",
            QuestionType.COMPARISON,
            ["Einstein", "Lincoln"],
        )
        assert len(subs) >= 2

    def test_inference_decomposition(self):
        decomposer = QuestionDecomposer()
        subs = decomposer.decompose(
            "Who is the paternal grandfather of John?",
            QuestionType.INFERENCE,
            ["John"],
        )
        assert len(subs) >= 2
        # Should decompose into father -> father chain


class TestPropertyMapper:
    """Test property mapping."""

    def test_wikidata_to_dbpedia(self):
        mapper = PropertyMapper()
        props = mapper.get_dbpedia_properties("P569")  # birth date
        assert len(props) > 0
        assert any("birthDate" in p for p in props)

    def test_property_name_to_dbpedia(self):
        mapper = PropertyMapper()
        props = mapper.get_property_from_name("date of birth")
        assert len(props) > 0

    def test_inference_rule(self):
        mapper = PropertyMapper()
        rule = mapper.get_inference_rule("paternal_grandfather")
        assert rule is not None
        assert rule == ("father", "father")


class TestTextualizer:
    """Test triple textualization."""

    def test_natural_format(self):
        textualizer = TripleTextualizer()
        triple = Triple(
            subject="http://dbpedia.org/resource/Einstein",
            subject_label="Albert Einstein",
            predicate="http://dbpedia.org/ontology/birthDate",
            predicate_label="birth date",
            obj="1879-03-14",
            obj_label="March 14, 1879",
        )
        text = textualizer.textualize(triple, format="natural")
        assert "Albert Einstein" in text

    def test_structured_format(self):
        textualizer = TripleTextualizer()
        triple = Triple(
            subject="Test",
            predicate="testProp",
            obj="Value",
        )
        text = textualizer.textualize(triple, format="structured")
        assert "Subject:" in text
        assert "Predicate:" in text
        assert "Object:" in text


class TestMetrics:
    """Test evaluation metrics."""

    def test_normalize_answer(self):
        assert normalize_answer("The Answer") == "answer"
        assert normalize_answer("an example") == "example"
        assert normalize_answer("Hello, World!") == "hello world"

    def test_exact_match(self):
        assert compute_em("Albert Einstein", "albert einstein") == 1.0
        assert compute_em("Einstein", "Lincoln") == 0.0

    def test_f1_score(self):
        assert compute_f1("Albert Einstein", "Albert Einstein") == 1.0
        assert compute_f1("Albert", "Albert Einstein") > 0
        assert compute_f1("Einstein", "Lincoln") == 0.0

    def test_supporting_facts_f1(self):
        pred = [("Title1", 0), ("Title1", 1)]
        gold = [("Title1", 0), ("Title2", 0)]
        prec, rec, f1 = compute_supporting_facts_f1(pred, gold)
        assert 0 < f1 < 1  # Partial overlap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
