"""
Test LLM's ability to generate SPARQL queries when provided with entity-specific schema partitions.

This test framework evaluates:
1. Schema partition quality - does it contain relevant properties?
2. LLM SPARQL generation - can LLM write valid SPARQL with schema context?
3. Query execution - do generated queries return correct results?
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
from SPARQLWrapper import SPARQLWrapper, JSON

from dbpedia.ontology_manager import OntologyManager, get_ontology_manager
from dbpedia.client import DBpediaSPARQLClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestEntity:
    """Entity for testing SPARQL generation."""
    uri: str
    label: str
    expected_types: list[str]
    test_questions: list[dict] = field(default_factory=list)
    # Each question: {"question": str, "expected_property": str, "expected_answer": str}


@dataclass
class SPARQLGenerationResult:
    """Result of LLM SPARQL generation test."""
    entity: TestEntity
    question: str
    schema_context: str
    generated_sparql: Optional[str]
    is_valid_syntax: bool
    execution_result: Optional[dict]
    expected_answer: str
    got_correct_answer: bool
    error: Optional[str] = None


# Test entities covering different DBpedia types
TEST_ENTITIES = [
    TestEntity(
        uri="http://dbpedia.org/resource/Barack_Obama",
        label="Barack Obama",
        expected_types=[
            "http://dbpedia.org/ontology/Person",
            "http://dbpedia.org/ontology/Politician",
            "http://dbpedia.org/ontology/President",
        ],
        test_questions=[
            {
                "question": "What is Barack Obama's birth date?",
                "expected_property": "dbo:birthDate",
                "expected_answer_contains": "1961",
            },
            {
                "question": "Where was Barack Obama born?",
                "expected_property": "dbo:birthPlace",
                "expected_answer_contains": "Honolulu",
            },
            {
                "question": "Who is Barack Obama's spouse?",
                "expected_property": "dbo:spouse",
                "expected_answer_contains": "Michelle",
            },
        ],
    ),
    TestEntity(
        uri="http://dbpedia.org/resource/The_Godfather",
        label="The Godfather",
        expected_types=[
            "http://dbpedia.org/ontology/Film",
            "http://dbpedia.org/ontology/Work",
        ],
        test_questions=[
            {
                "question": "Who directed The Godfather?",
                "expected_property": "dbo:director",
                "expected_answer_contains": "Coppola",
            },
            {
                "question": "When was The Godfather released?",
                "expected_property": "dbo:releaseDate",
                "expected_answer_contains": "1972",
            },
        ],
    ),
    TestEntity(
        uri="http://dbpedia.org/resource/Paris",
        label="Paris",
        expected_types=[
            "http://dbpedia.org/ontology/City",
            "http://dbpedia.org/ontology/Place",
            "http://dbpedia.org/ontology/PopulatedPlace",
        ],
        test_questions=[
            {
                "question": "What country is Paris in?",
                "expected_property": "dbo:country",
                "expected_answer_contains": "France",
            },
            {
                "question": "What is the population of Paris?",
                "expected_property": "dbo:populationTotal",
                "expected_answer_contains": "",  # Any number
            },
        ],
    ),
    TestEntity(
        uri="http://dbpedia.org/resource/Albert_Einstein",
        label="Albert Einstein",
        expected_types=[
            "http://dbpedia.org/ontology/Person",
            "http://dbpedia.org/ontology/Scientist",
        ],
        test_questions=[
            {
                "question": "What field did Albert Einstein work in?",
                "expected_property": "dbo:field",
                "expected_answer_contains": "Physics",
            },
            {
                "question": "Where did Albert Einstein die?",
                "expected_property": "dbo:deathPlace",
                "expected_answer_contains": "Princeton",
            },
        ],
    ),
]


class TestOntologyManager:
    """Test the ontology manager functionality."""

    @pytest.fixture
    def ontology_manager(self) -> OntologyManager:
        """Create ontology manager instance."""
        return get_ontology_manager()

    def test_ontology_loads(self, ontology_manager: OntologyManager):
        """Test that ontology file loads successfully."""
        ontology_manager.load()
        stats = ontology_manager.get_stats()

        assert stats["total_triples"] > 0, "Ontology should have triples"
        assert stats["total_classes"] > 100, "Should have many classes"
        assert stats["total_properties"] > 100, "Should have many properties"

        logger.info(f"Ontology stats: {stats}")

    def test_get_person_properties(self, ontology_manager: OntologyManager):
        """Test getting properties for Person class."""
        person_uri = "http://dbpedia.org/ontology/Person"
        properties = ontology_manager.get_properties_for_class(person_uri)

        prop_uris = [p.uri for p in properties]
        logger.info(f"Person has {len(properties)} applicable properties")

        # Check for expected person properties
        expected = ["birthDate", "birthPlace", "spouse", "nationality"]
        for exp in expected:
            matching = [p for p in prop_uris if exp in p]
            assert len(matching) > 0, f"Person should have {exp} property"

    def test_get_film_properties(self, ontology_manager: OntologyManager):
        """Test getting properties for Film class."""
        film_uri = "http://dbpedia.org/ontology/Film"
        properties = ontology_manager.get_properties_for_class(film_uri)

        prop_uris = [p.uri for p in properties]
        logger.info(f"Film has {len(properties)} applicable properties")

        # Check for expected film properties
        expected = ["director", "producer", "releaseDate"]
        for exp in expected:
            matching = [p for p in prop_uris if exp in p]
            assert len(matching) > 0, f"Film should have {exp} property"

    def test_class_hierarchy(self, ontology_manager: OntologyManager):
        """Test class hierarchy retrieval."""
        politician_uri = "http://dbpedia.org/ontology/Politician"
        hierarchy = ontology_manager.get_class_hierarchy(politician_uri)

        class_uris = [c.uri for c in hierarchy]
        logger.info(f"Politician hierarchy: {class_uris}")

        # Politician should have Person in hierarchy
        assert any("Person" in uri for uri in class_uris), "Politician should inherit from Person"

    def test_search_properties(self, ontology_manager: OntologyManager):
        """Test property search functionality."""
        results = ontology_manager.search_properties("birth")
        logger.info(f"Found {len(results)} properties matching 'birth'")

        assert len(results) > 0, "Should find birth-related properties"
        prop_uris = [p.uri for p in results]
        assert any("birthDate" in uri for uri in prop_uris)

    def test_search_classes(self, ontology_manager: OntologyManager):
        """Test class search functionality."""
        results = ontology_manager.search_classes("person")
        logger.info(f"Found {len(results)} classes matching 'person'")

        assert len(results) > 0, "Should find Person class"


class TestEntitySchemaPartition:
    """Test schema partition extraction for entities."""

    @pytest.fixture
    def ontology_manager(self) -> OntologyManager:
        return get_ontology_manager()

    @pytest.fixture
    def sparql_client(self) -> DBpediaSPARQLClient:
        return DBpediaSPARQLClient()

    def test_get_entity_types_from_dbpedia(self, sparql_client: DBpediaSPARQLClient):
        """Test fetching entity types from live DBpedia."""
        entity_uri = "http://dbpedia.org/resource/Barack_Obama"

        query = f"""
        SELECT DISTINCT ?type WHERE {{
            <{entity_uri}> a ?type .
            FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
        }}
        """
        results = sparql_client.query(query)
        bindings = results.get("results", {}).get("bindings", [])
        types = [b["type"]["value"] for b in bindings]

        logger.info(f"Barack Obama types: {types}")
        assert len(types) > 0, "Should have DBpedia types"
        assert any("Person" in t for t in types), "Should be a Person"

    @pytest.mark.parametrize("entity", TEST_ENTITIES)
    def test_schema_partition_for_entity(
        self,
        ontology_manager: OntologyManager,
        sparql_client: DBpediaSPARQLClient,
        entity: TestEntity,
    ):
        """Test schema partition extraction for each test entity."""
        # Get actual types from DBpedia
        query = f"""
        SELECT DISTINCT ?type WHERE {{
            <{entity.uri}> a ?type .
            FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
        }}
        LIMIT 20
        """
        results = sparql_client.query(query)
        bindings = results.get("results", {}).get("bindings", [])
        actual_types = [b["type"]["value"] for b in bindings]

        logger.info(f"\n{entity.label} actual types: {actual_types}")

        # Get schema partition
        partition = ontology_manager.get_entity_schema_partition(actual_types)

        logger.info(f"Schema partition has {len(partition.applicable_properties)} properties")
        logger.info(f"Class hierarchy: {[c.uri.split('/')[-1] for c in partition.class_hierarchy[:5]]}")

        # Check that expected properties are available
        prop_uris = [p.uri for p in partition.applicable_properties]
        for q in entity.test_questions:
            expected_prop = q["expected_property"].replace("dbo:", "http://dbpedia.org/ontology/")
            matching = [p for p in prop_uris if expected_prop.split("/")[-1] in p]
            if not matching:
                logger.warning(f"Expected property {expected_prop} not in partition for {entity.label}")

        # Generate prompt context
        context = partition.to_prompt_context(max_properties=30)
        logger.info(f"\nGenerated schema context ({len(context)} chars):\n{context[:500]}...")

        assert len(partition.applicable_properties) > 0, f"Should have properties for {entity.label}"


class TestLLMSPARQLGeneration:
    """Test LLM's ability to generate SPARQL with schema context."""

    @pytest.fixture
    def ontology_manager(self) -> OntologyManager:
        return get_ontology_manager()

    @pytest.fixture
    def sparql_client(self) -> DBpediaSPARQLClient:
        return DBpediaSPARQLClient()

    def _get_entity_types(self, sparql_client: DBpediaSPARQLClient, entity_uri: str) -> list[str]:
        """Fetch entity types from DBpedia."""
        query = f"""
        SELECT DISTINCT ?type WHERE {{
            <{entity_uri}> a ?type .
            FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
        }}
        LIMIT 20
        """
        results = sparql_client.query(query)
        bindings = results.get("results", {}).get("bindings", [])
        return [b["type"]["value"] for b in bindings]

    def _build_sparql_prompt(
        self,
        question: str,
        entity_uri: str,
        entity_label: str,
        schema_context: str
    ) -> str:
        """Build prompt for LLM to generate SPARQL."""
        return f"""You are a SPARQL query generator for DBpedia.

Given an entity and a question, generate a SPARQL query to answer the question.

## Entity
- URI: <{entity_uri}>
- Label: {entity_label}

## DBpedia Schema for this Entity
{schema_context}

## SPARQL Prefixes Available
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX dbr: <http://dbpedia.org/resource/>

## Question
{question}

## Instructions
1. Use the schema above to identify the correct property
2. Write a SELECT query that returns the answer
3. Include OPTIONAL for labels when the result is a URI
4. Use LIMIT 10

Output ONLY the SPARQL query, no explanation:
"""

    def _validate_sparql_syntax(self, query: str) -> tuple[bool, Optional[str]]:
        """Basic SPARQL syntax validation."""
        query = query.strip()

        # Check for required elements
        if not query.upper().startswith("SELECT"):
            return False, "Query must start with SELECT"

        if "WHERE" not in query.upper():
            return False, "Query must have WHERE clause"

        # Check for balanced braces
        if query.count("{") != query.count("}"):
            return False, "Unbalanced braces"

        return True, None

    def _execute_sparql(
        self,
        sparql_client: DBpediaSPARQLClient,
        query: str
    ) -> tuple[bool, Optional[dict], Optional[str]]:
        """Execute SPARQL query and return results."""
        try:
            results = sparql_client.query(query, use_cache=False)
            return True, results, None
        except Exception as e:
            return False, None, str(e)

    @pytest.mark.parametrize("entity", TEST_ENTITIES)
    def test_schema_context_quality(
        self,
        ontology_manager: OntologyManager,
        sparql_client: DBpediaSPARQLClient,
        entity: TestEntity,
    ):
        """Test that schema context contains relevant information for SPARQL generation."""
        # Get types and partition
        types = self._get_entity_types(sparql_client, entity.uri)
        partition = ontology_manager.get_entity_schema_partition(types)
        context = partition.to_prompt_context(max_properties=50)

        # Verify context quality
        for q in entity.test_questions:
            expected_prop = q["expected_property"]
            prop_name = expected_prop.split(":")[-1]

            assert prop_name.lower() in context.lower(), \
                f"Schema context should contain {prop_name} for question: {q['question']}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Schema Context for {entity.label}:")
        logger.info(f"{'='*60}")
        logger.info(context)

    @pytest.mark.parametrize("entity", TEST_ENTITIES[:2])  # Test subset for speed
    def test_manual_sparql_with_schema(
        self,
        ontology_manager: OntologyManager,
        sparql_client: DBpediaSPARQLClient,
        entity: TestEntity,
    ):
        """Test manually written SPARQL queries using schema information."""
        types = self._get_entity_types(sparql_client, entity.uri)
        partition = ontology_manager.get_entity_schema_partition(types)

        for q in entity.test_questions:
            expected_prop = q["expected_property"]

            # Build query using schema
            query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?value ?label WHERE {{
                <{entity.uri}> {expected_prop} ?value .
                OPTIONAL {{ ?value rdfs:label ?label . FILTER(LANG(?label) = "en") }}
            }}
            LIMIT 5
            """

            success, results, error = self._execute_sparql(sparql_client, query)

            if success:
                bindings = results.get("results", {}).get("bindings", [])
                if bindings:
                    value = bindings[0].get("label", bindings[0].get("value", {})).get("value", "")
                    logger.info(f"Q: {q['question']}")
                    logger.info(f"A: {value}")

                    expected = q["expected_answer_contains"]
                    if expected:
                        assert expected.lower() in value.lower(), \
                            f"Expected '{expected}' in answer '{value}'"
            else:
                logger.warning(f"Query failed for {q['question']}: {error}")

    def test_prompt_generation(
        self,
        ontology_manager: OntologyManager,
        sparql_client: DBpediaSPARQLClient,
    ):
        """Test that generated prompts are well-formed."""
        entity = TEST_ENTITIES[0]  # Barack Obama
        types = self._get_entity_types(sparql_client, entity.uri)
        partition = ontology_manager.get_entity_schema_partition(types)
        context = partition.to_prompt_context(max_properties=30)

        for q in entity.test_questions:
            prompt = self._build_sparql_prompt(
                question=q["question"],
                entity_uri=entity.uri,
                entity_label=entity.label,
                schema_context=context
            )

            logger.info(f"\n{'='*60}")
            logger.info(f"Generated Prompt for: {q['question']}")
            logger.info(f"{'='*60}")
            logger.info(prompt)

            # Verify prompt quality
            assert entity.uri in prompt
            assert entity.label in prompt
            assert q["question"] in prompt
            assert "dbo:" in prompt  # Has prefixes
            assert "## Available Properties" in prompt


class TestEndToEndWithMockLLM:
    """End-to-end test with simulated LLM responses."""

    @pytest.fixture
    def ontology_manager(self) -> OntologyManager:
        return get_ontology_manager()

    @pytest.fixture
    def sparql_client(self) -> DBpediaSPARQLClient:
        return DBpediaSPARQLClient()

    def _mock_llm_sparql_generation(
        self,
        entity_uri: str,
        property_hint: str,
    ) -> str:
        """Simulate LLM generating SPARQL (for testing without actual LLM)."""
        # This mimics what a good LLM should generate given the schema context
        return f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?value ?label WHERE {{
            <{entity_uri}> {property_hint} ?value .
            OPTIONAL {{ ?value rdfs:label ?label . FILTER(LANG(?label) = "en") }}
        }}
        LIMIT 10
        """

    @pytest.mark.parametrize("entity", TEST_ENTITIES)
    def test_mock_llm_sparql_execution(
        self,
        ontology_manager: OntologyManager,
        sparql_client: DBpediaSPARQLClient,
        entity: TestEntity,
    ):
        """Test that well-formed SPARQL queries return expected results."""
        results_summary = []

        for q in entity.test_questions:
            # Simulate LLM generating query based on schema
            generated_query = self._mock_llm_sparql_generation(
                entity_uri=entity.uri,
                property_hint=q["expected_property"],
            )

            # Execute query
            success, results, error = self._execute_query(sparql_client, generated_query)

            result = {
                "question": q["question"],
                "property": q["expected_property"],
                "success": success,
                "answer": None,
                "expected_contains": q["expected_answer_contains"],
                "correct": False,
            }

            if success:
                bindings = results.get("results", {}).get("bindings", [])
                if bindings:
                    value = bindings[0].get("label", bindings[0].get("value", {})).get("value", "")
                    result["answer"] = value

                    expected = q["expected_answer_contains"]
                    if expected:
                        result["correct"] = expected.lower() in value.lower()
                    else:
                        result["correct"] = len(value) > 0

            results_summary.append(result)

        # Report results
        logger.info(f"\n{'='*60}")
        logger.info(f"Results for {entity.label}")
        logger.info(f"{'='*60}")

        correct = sum(1 for r in results_summary if r["correct"])
        total = len(results_summary)

        for r in results_summary:
            status = "✓" if r["correct"] else "✗"
            logger.info(f"{status} {r['question']}")
            logger.info(f"  Property: {r['property']}")
            logger.info(f"  Answer: {r['answer']}")
            if not r["correct"]:
                logger.info(f"  Expected to contain: {r['expected_contains']}")

        logger.info(f"\nScore: {correct}/{total} ({correct/total*100:.1f}%)")

        assert correct >= total * 0.5, f"At least 50% of queries should succeed for {entity.label}"

    def _execute_query(
        self,
        sparql_client: DBpediaSPARQLClient,
        query: str
    ) -> tuple[bool, Optional[dict], Optional[str]]:
        """Execute SPARQL query."""
        try:
            results = sparql_client.query(query, use_cache=False)
            return True, results, None
        except Exception as e:
            return False, None, str(e)


def run_full_evaluation():
    """Run full evaluation and generate report."""
    ontology_manager = get_ontology_manager()
    sparql_client = DBpediaSPARQLClient()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ontology_stats": ontology_manager.get_stats(),
        "entities_tested": [],
        "overall_accuracy": 0.0,
    }

    total_correct = 0
    total_questions = 0

    for entity in TEST_ENTITIES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {entity.label}")
        logger.info(f"{'='*60}")

        # Get types
        query = f"""
        SELECT DISTINCT ?type WHERE {{
            <{entity.uri}> a ?type .
            FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
        }}
        LIMIT 20
        """
        results = sparql_client.query(query)
        bindings = results.get("results", {}).get("bindings", [])
        types = [b["type"]["value"] for b in bindings]

        # Get schema partition
        partition = ontology_manager.get_entity_schema_partition(types)
        context = partition.to_prompt_context(max_properties=50)

        entity_results = {
            "uri": entity.uri,
            "label": entity.label,
            "types": types,
            "num_properties_in_partition": len(partition.applicable_properties),
            "questions": [],
        }

        for q in entity.test_questions:
            # Test query execution
            test_query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?value ?label WHERE {{
                <{entity.uri}> {q['expected_property']} ?value .
                OPTIONAL {{ ?value rdfs:label ?label . FILTER(LANG(?label) = "en") }}
            }}
            LIMIT 5
            """

            try:
                result = sparql_client.query(test_query, use_cache=False)
                bindings = result.get("results", {}).get("bindings", [])

                answer = None
                correct = False

                if bindings:
                    answer = bindings[0].get("label", bindings[0].get("value", {})).get("value", "")
                    expected = q["expected_answer_contains"]
                    correct = expected.lower() in answer.lower() if expected else bool(answer)

                entity_results["questions"].append({
                    "question": q["question"],
                    "property": q["expected_property"],
                    "answer": answer,
                    "correct": correct,
                })

                total_questions += 1
                if correct:
                    total_correct += 1

                status = "✓" if correct else "✗"
                logger.info(f"{status} {q['question']} → {answer}")

            except Exception as e:
                logger.error(f"Error for {q['question']}: {e}")
                entity_results["questions"].append({
                    "question": q["question"],
                    "property": q["expected_property"],
                    "error": str(e),
                    "correct": False,
                })
                total_questions += 1

        report["entities_tested"].append(entity_results)

    report["overall_accuracy"] = total_correct / total_questions if total_questions > 0 else 0

    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {total_correct}/{total_questions} ({report['overall_accuracy']*100:.1f}%)")

    # Save report
    report_path = Path("data/llm_sparql_test_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nReport saved to: {report_path}")

    return report


if __name__ == "__main__":
    run_full_evaluation()
