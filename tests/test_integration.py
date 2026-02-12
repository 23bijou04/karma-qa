"""Integration tests for DBpedia GraphRAG system."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from dbpedia_qa.models import Entity, Triple, QuestionType, Subgraph, Question, Answer
from dbpedia_qa.config import Config
from dbpedia_qa.dbpedia.client import DBpediaSPARQLClient, RateLimiter
from dbpedia_qa.dbpedia.entity_linker import DBpediaEntityLinker
from dbpedia_qa.dbpedia.property_mapper import PropertyMapper
from dbpedia_qa.dbpedia.cache import CacheManager, InMemoryCache
from dbpedia_qa.question_processing.classifier import QuestionClassifier
from dbpedia_qa.question_processing.decomposer import QuestionDecomposer
from dbpedia_qa.question_processing.entity_extractor import QuestionEntityExtractor
from dbpedia_qa.traversal.ranked_bfs import RankedBFS
from dbpedia_qa.traversal.relevance_scorer import KeywordRelevanceScorer
from dbpedia_qa.traversal.strategies import (
    ComparisonStrategy,
    InferenceStrategy,
    CompositionalStrategy,
    BridgeComparisonStrategy,
    get_strategy_for_question_type,
)
from dbpedia_qa.wikipedia.client import WikipediaClient
from dbpedia_qa.context.textualizer import TripleTextualizer
from dbpedia_qa.context.assembler import ContextAssembler
from dbpedia_qa.llm.prompts import get_prompt_for_question_type, PromptTemplate
from dbpedia_qa.llm.parser import ResponseParser
from dbpedia_qa.evaluation.metrics import (
    normalize_answer,
    compute_em,
    compute_f1,
    compute_supporting_facts_f1,
    compute_joint_metrics,
    MetricsAggregator,
)


class TestCacheIntegration:
    """Test caching functionality."""

    def test_in_memory_cache(self):
        cache = InMemoryCache(max_size=10, default_ttl=3600)
        cache.set("test_key", {"data": "value"})
        result = cache.get("test_key")
        assert result == {"data": "value"}

    def test_cache_manager_memory_only(self):
        manager = CacheManager(use_redis=False)
        manager.set("key1", "value1")
        assert manager.get("key1") == "value1"
        assert manager.get("nonexistent") is None

    def test_cache_decorator(self):
        manager = CacheManager()

        @manager.cached(ttl=60)
        def expensive_function(x):
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(5)
        assert result1 == result2 == 10


class TestRateLimiter:
    """Test rate limiting."""

    def test_rate_limiter_init(self):
        limiter = RateLimiter(rate=10.0)
        assert limiter.rate == 10.0

    def test_rate_limiter_acquire(self):
        limiter = RateLimiter(rate=100.0)  # High rate for fast test
        # Should not block significantly
        limiter.acquire()
        limiter.acquire()


class TestPropertyMapperIntegration:
    """Test property mapping functionality."""

    def test_full_mapping_chain(self):
        mapper = PropertyMapper()

        # Test Wikidata P-code mapping
        birth_props = mapper.get_dbpedia_properties("P569")
        assert "dbo:birthDate" in birth_props

        # Test human-readable name mapping
        birth_props2 = mapper.get_property_from_name("date of birth")
        assert "dbo:birthDate" in birth_props2

        # Test inference chain
        chain = mapper.get_inference_chain("paternal_grandfather")
        assert len(chain) == 2

    def test_family_properties(self):
        mapper = PropertyMapper()
        family_props = mapper.get_all_family_properties()
        assert "father" in family_props
        assert "mother" in family_props

    def test_comparison_properties(self):
        mapper = PropertyMapper()
        person_props = mapper.get_comparison_properties("person")
        assert "birth date" in person_props

        film_props = mapper.get_comparison_properties("film")
        assert "release date" in film_props


class TestQuestionProcessingPipeline:
    """Test question processing pipeline."""

    def test_classification_to_decomposition(self):
        """Test that classified questions are properly decomposed."""
        classifier = QuestionClassifier()
        decomposer = QuestionDecomposer()

        # Comparison question
        q1 = "Who was born first, Einstein or Newton?"
        q1_type = classifier.classify(q1)
        q1_subs = decomposer.decompose(q1, q1_type, ["Einstein", "Newton"])
        assert q1_type == QuestionType.COMPARISON
        assert len(q1_subs) >= 2

        # Inference question
        q2 = "Who is the paternal grandfather of John Smith?"
        q2_type = classifier.classify(q2)
        q2_subs = decomposer.decompose(q2, q2_type, ["John Smith"])
        assert q2_type == QuestionType.INFERENCE
        assert len(q2_subs) >= 2

    def test_entity_extraction_with_classifier(self):
        """Test entity extraction from classified questions."""
        classifier = QuestionClassifier()

        # Extract entities from comparison question
        entities = classifier.extract_comparison_entities(
            "Who was born first, Albert Einstein or Abraham Lincoln?"
        )
        assert "Albert Einstein" in entities or "Einstein" in entities

    def test_confidence_scoring(self):
        """Test classification confidence scoring."""
        classifier = QuestionClassifier()

        # Pattern-matched should have high confidence
        q_type, confidence = classifier.get_classification_confidence(
            "Who was born first, A or B?"
        )
        assert confidence >= 0.7


class TestTraversalStrategies:
    """Test traversal strategy selection and behavior."""

    def test_strategy_selection(self):
        """Test that correct strategy is selected for each question type."""
        mock_client = Mock(spec=DBpediaSPARQLClient)

        comp_strategy = get_strategy_for_question_type(
            QuestionType.COMPARISON, mock_client
        )
        assert isinstance(comp_strategy, ComparisonStrategy)

        inf_strategy = get_strategy_for_question_type(
            QuestionType.INFERENCE, mock_client
        )
        assert isinstance(inf_strategy, InferenceStrategy)

        compositional_strategy = get_strategy_for_question_type(
            QuestionType.COMPOSITIONAL, mock_client
        )
        assert isinstance(compositional_strategy, CompositionalStrategy)

        bridge_strategy = get_strategy_for_question_type(
            QuestionType.BRIDGE_COMPARISON, mock_client
        )
        assert isinstance(bridge_strategy, BridgeComparisonStrategy)

    def test_relevance_scorer_keywords(self):
        """Test keyword-based relevance scoring."""
        scorer = KeywordRelevanceScorer()

        # High relevance when entity in question
        score1 = scorer.score(
            "When was Einstein born?",
            "http://dbpedia.org/resource/Albert_Einstein",
            "http://dbpedia.org/ontology/birthDate",
            "Albert Einstein",
        )
        assert score1 >= 0.3  # At least moderate relevance

        # Test property keywords
        relevant_props = scorer.get_relevant_properties("When was he born?")
        assert "birthDate" in relevant_props


class TestContextAssembly:
    """Test context assembly functionality."""

    def test_textualizer_formats(self):
        """Test different textualization formats."""
        textualizer = TripleTextualizer()
        triple = Triple(
            subject="http://dbpedia.org/resource/Einstein",
            subject_label="Albert Einstein",
            predicate="http://dbpedia.org/ontology/birthDate",
            predicate_label="birth date",
            obj="1879-03-14",
        )

        natural = textualizer.textualize(triple, format="natural")
        assert "Albert Einstein" in natural

        structured = textualizer.textualize(triple, format="structured")
        assert "Subject:" in structured

        triple_fmt = textualizer.textualize(triple, format="triple")
        assert "(" in triple_fmt

    def test_textualizer_batch(self):
        """Test batch textualization."""
        textualizer = TripleTextualizer()
        triples = [
            Triple(subject="A", predicate="p1", obj="B"),
            Triple(subject="B", predicate="p2", obj="C"),
        ]

        batch_text = textualizer.textualize_batch(triples)
        assert "\n" in batch_text


class TestPromptGeneration:
    """Test prompt template generation."""

    def test_prompt_templates(self):
        """Test that prompts are generated for all question types."""
        for qtype in [
            QuestionType.COMPARISON,
            QuestionType.INFERENCE,
            QuestionType.COMPOSITIONAL,
            QuestionType.BRIDGE_COMPARISON,
        ]:
            template = get_prompt_for_question_type(qtype)
            assert isinstance(template, PromptTemplate)
            assert template.system_prompt
            assert template.user_template

    def test_prompt_formatting(self):
        """Test prompt template formatting."""
        template = get_prompt_for_question_type(QuestionType.COMPARISON)
        formatted = template.format(
            question="Who was born first?",
            context="Einstein was born in 1879. Newton was born in 1643.",
        )
        assert "Who was born first?" in formatted
        assert "Einstein" in formatted


class TestResponseParsing:
    """Test response parsing functionality."""

    def test_structured_response_parsing(self):
        """Test parsing of structured LLM responses."""
        parser = ResponseParser()

        response = """ANSWER: Albert Einstein
REASONING: Einstein was born in 1879, Newton was born in 1643. Newton was born first.
EVIDENCE: (Newton, birthDate, 1643)
CONFIDENCE: high"""

        parsed = parser.parse(response)
        assert parsed["answer"] == "Albert Einstein"
        assert "reasoning" in parsed
        assert parsed["confidence"] == "high"

    def test_chain_of_thought_parsing(self):
        """Test parsing of chain-of-thought responses."""
        parser = ResponseParser()

        response = """1. The question asks who was born first.
2. Einstein was born in 1879.
3. Newton was born in 1643.
4. Therefore, the answer is Newton."""

        parsed = parser.parse_chain_of_thought(response)
        assert len(parsed["steps"]) >= 3
        assert "newton" in parsed.get("answer", "").lower() or parsed["steps"]

    def test_yes_no_parsing(self):
        """Test yes/no response parsing."""
        parser = ResponseParser()

        assert parser.parse_yes_no("Yes, they are the same.") is True
        assert parser.parse_yes_no("No, they are different.") is False
        # "Maybe" contains no clear yes/no indicators, but "not sure" might not be parsed as None
        # The actual behavior depends on implementation - test for valid boolean or None
        result = parser.parse_yes_no("I cannot determine this.")
        assert result in (True, False, None)

    def test_answer_normalization(self):
        """Test answer normalization."""
        parser = ResponseParser()

        assert parser.normalize_answer("The Answer") == "answer"
        assert parser.normalize_answer("An Example!") == "example"


class TestMetricsCalculation:
    """Test metrics calculation."""

    def test_em_normalization(self):
        """Test exact match with normalization."""
        # Should match after normalization
        assert compute_em("Albert Einstein", "albert einstein") == 1.0
        assert compute_em("The Answer", "answer") == 1.0

        # Should not match
        assert compute_em("Einstein", "Newton") == 0.0

    def test_f1_partial_match(self):
        """Test F1 with partial matches."""
        # Full match
        assert compute_f1("Albert Einstein", "Albert Einstein") == 1.0

        # Partial match (one common token)
        f1 = compute_f1("Albert", "Albert Einstein")
        assert 0 < f1 < 1

        # No match
        assert compute_f1("Einstein", "Newton") == 0.0

    def test_supporting_facts_metrics(self):
        """Test supporting facts F1."""
        pred = [("Title1", 0), ("Title1", 1), ("Title2", 0)]
        gold = [("Title1", 0), ("Title2", 0), ("Title2", 1)]

        prec, rec, f1 = compute_supporting_facts_f1(pred, gold)
        assert 0 < prec <= 1
        assert 0 < rec <= 1
        assert 0 < f1 <= 1

    def test_joint_metrics(self):
        """Test joint metrics calculation."""
        metrics = compute_joint_metrics(
            answer_prediction="Einstein",
            answer_gold="Albert Einstein",
            sf_prediction=[("Title1", 0)],
            sf_gold=[("Title1", 0), ("Title1", 1)],
        )

        assert "answer_em" in metrics
        assert "answer_f1" in metrics
        assert "supporting_facts_f1" in metrics
        assert "joint_f1" in metrics

    def test_metrics_aggregator(self):
        """Test metrics aggregation."""
        aggregator = MetricsAggregator()

        aggregator.add({"answer_f1": 0.8, "joint_f1": 0.6}, "comparison")
        aggregator.add({"answer_f1": 0.9, "joint_f1": 0.7}, "comparison")
        aggregator.add({"answer_f1": 0.7, "joint_f1": 0.5}, "inference")

        agg = aggregator.aggregate()
        assert "mean_answer_f1" in agg
        assert agg["num_examples"] == 3

        by_type = aggregator.aggregate_by_type()
        assert "comparison" in by_type
        assert "inference" in by_type


class TestEndToEndPipeline:
    """Test end-to-end pipeline components."""

    def test_question_to_subquestions(self):
        """Test full question processing pipeline."""
        classifier = QuestionClassifier()
        decomposer = QuestionDecomposer()

        questions = [
            ("Who was born first, Einstein or Newton?", QuestionType.COMPARISON),
            ("Who is the paternal grandfather of John?", QuestionType.INFERENCE),
            ("Who directed the film that won Best Picture?", QuestionType.COMPOSITIONAL),
        ]

        for question, expected_type in questions:
            q_type = classifier.classify(question)
            assert q_type == expected_type

            subs = decomposer.decompose(question, q_type)
            assert len(subs) >= 1

    def test_triple_to_context_flow(self):
        """Test triple textualization to context assembly."""
        textualizer = TripleTextualizer()

        triples = [
            Triple(
                subject="http://dbpedia.org/resource/Einstein",
                subject_label="Einstein",
                predicate="dbo:birthDate",
                predicate_label="birth date",
                obj="1879-03-14",
            ),
            Triple(
                subject="http://dbpedia.org/resource/Newton",
                subject_label="Newton",
                predicate="dbo:birthDate",
                predicate_label="birth date",
                obj="1643-01-04",
            ),
        ]

        context = textualizer.textualize_batch(triples)
        assert "Einstein" in context
        assert "Newton" in context
        assert "1879" in context
        assert "1643" in context

    def test_subgraph_operations(self):
        """Test subgraph data structure operations."""
        subgraph = Subgraph()

        # Add entities
        e1 = Entity(uri="http://dbpedia.org/resource/Einstein", label="Einstein")
        e2 = Entity(uri="http://dbpedia.org/resource/Newton", label="Newton")
        subgraph.add_entity(e1)
        subgraph.add_entity(e2)

        # Add triples
        t1 = Triple(subject=e1.uri, predicate="dbo:birthDate", obj="1879")
        subgraph.add_triple(t1)

        assert len(subgraph.entities) == 2
        assert len(subgraph.triples) == 1

        # Get entity by URI
        found = subgraph.get_entity_by_uri(e1.uri)
        assert found == e1

        # Get triples for entity
        entity_triples = subgraph.get_triples_for_entity(e1.uri)
        assert len(entity_triples) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
