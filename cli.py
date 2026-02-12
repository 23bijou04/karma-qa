#!/usr/bin/env python3
"""Command-line interface for DBpedia GraphRAG system."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from dbpedia_qa.config import Config
from dbpedia_qa.system import DBpediaGraphRAGSystem, create_system
from dbpedia_qa.evaluation.runner import BenchmarkRunner
from dbpedia_qa.models import QuestionType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def answer_question(args):
    """Answer a single question."""
    config = Config.from_env()

    # Override config if arguments provided
    if args.llm_provider:
        config.llm.provider = args.llm_provider
    if args.llm_model:
        config.llm.model = args.llm_model

    logger.info("Initializing GraphRAG system...")
    system = create_system(config)

    logger.info(f"Processing question: {args.question}")

    answer, supporting_facts, evidence = system.answer_question(args.question)

    print("\n" + "=" * 60)
    print("ANSWER:", answer.text)
    print("=" * 60)

    if answer.reasoning:
        print("\nREASONING:")
        print(answer.reasoning)

    if supporting_facts:
        print("\nSUPPORTING FACTS:")
        for title, idx in supporting_facts:
            print(f"  - {title}: sentence {idx}")

    if evidence and evidence.triples:
        print("\nEVIDENCE TRIPLES:")
        for triple in evidence.triples[:5]:
            print(f"  - {triple.to_natural_language()}")

    print(f"\nConfidence: {answer.confidence:.2%}")
    print()


def run_benchmark(args):
    """Run benchmark evaluation."""
    config = Config.from_env()

    if args.llm_provider:
        config.llm.provider = args.llm_provider
    if args.llm_model:
        config.llm.model = args.llm_model

    logger.info("Initializing GraphRAG system...")
    system = create_system(config)

    logger.info(f"Loading benchmark from: {args.dataset}")
    runner = BenchmarkRunner(system, args.dataset)

    if args.gold_context:
        logger.info("Running evaluation with gold context...")
        results = runner.run_with_gold_context(
            path=args.dataset,
            limit=args.limit,
            verbose=args.verbose,
        )
    else:
        logger.info("Running full pipeline evaluation...")
        results = runner.run(
            path=args.dataset,
            limit=args.limit,
            verbose=args.verbose,
        )

    # Print report
    print("\n" + runner.report())

    # Save results if output path provided
    if args.output:
        runner.save_results(args.output)
        logger.info(f"Results saved to: {args.output}")

    return results


def demo_mode(args):
    """Run interactive demo mode."""
    config = Config.from_env()
    config.llm.provider = args.llm_provider or "ollama"
    config.llm.model = args.llm_model or "llama3.1:8b"

    print("\n" + "=" * 60)
    print("DBpedia GraphRAG System - Demo Mode")
    print("=" * 60)
    print("\nExample questions:")
    print("  - Who was born first, Einstein or Newton?")
    print("  - Who is the paternal grandfather of Prince William?")
    print("  - Who directed the film that won Best Picture in 2017?")
    print("\nType 'quit' to exit.\n")

    system = None

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not question:
                continue

            # Lazy initialize system
            if system is None:
                print("Initializing system (this may take a moment)...")
                system = create_system(config)

            print("\nProcessing...")
            answer, sf, evidence = system.answer_question(question)

            print(f"\nAnswer: {answer.text}")
            if answer.reasoning:
                print(f"Reasoning: {answer.reasoning}")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error processing question: {e}\n")


def test_components(args):
    """Test individual components."""
    print("\n" + "=" * 60)
    print("Component Tests")
    print("=" * 60)

    # Test imports
    print("\n1. Testing imports...")
    try:
        from dbpedia_qa import Config, Entity, Triple
        from dbpedia_qa.system import DBpediaGraphRAGSystem
        from dbpedia_qa.dbpedia import DBpediaSPARQLClient, DBpediaEntityLinker
        from dbpedia_qa.question_processing import QuestionClassifier, QuestionDecomposer
        from dbpedia_qa.traversal import RankedBFS, get_strategy_for_question_type
        from dbpedia_qa.evaluation import compute_em, compute_f1
        print("   ✓ All imports successful")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return

    # Test question classification
    print("\n2. Testing question classifier...")
    classifier = QuestionClassifier()
    test_questions = [
        ("Who was born first, Einstein or Newton?", QuestionType.COMPARISON),
        ("Who is the paternal grandfather of John?", QuestionType.INFERENCE),
        ("Who founded the company that made the iPhone?", QuestionType.COMPOSITIONAL),
    ]
    for q, expected in test_questions:
        result = classifier.classify(q)
        status = "✓" if result == expected else "✗"
        print(f"   {status} '{q[:40]}...' -> {result.value}")

    # Test property mapping
    print("\n3. Testing property mapper...")
    from dbpedia_qa.dbpedia import PropertyMapper
    mapper = PropertyMapper()
    props = mapper.get_dbpedia_properties("P569")  # birth date
    print(f"   ✓ P569 (birthDate) maps to: {props[:2]}")

    # Test metrics
    print("\n4. Testing evaluation metrics...")
    from dbpedia_qa.evaluation import compute_em, compute_f1
    em = compute_em("Albert Einstein", "albert einstein")
    f1 = compute_f1("Albert", "Albert Einstein")
    print(f"   ✓ EM('Albert Einstein', 'albert einstein') = {em}")
    print(f"   ✓ F1('Albert', 'Albert Einstein') = {f1:.2f}")

    # Test SPARQL client (if not dry-run)
    if not args.dry_run:
        print("\n5. Testing DBpedia SPARQL client...")
        try:
            from dbpedia_qa.dbpedia import DBpediaSPARQLClient
            client = DBpediaSPARQLClient()
            entity = client.get_entity_by_label("Albert Einstein")
            if entity:
                print(f"   ✓ Found entity: {entity.label} ({entity.uri})")
            else:
                print("   ✗ Entity not found (DBpedia may be unavailable)")
        except Exception as e:
            print(f"   ✗ SPARQL test failed: {e}")

    print("\n" + "=" * 60)
    print("All component tests completed!")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="DBpedia GraphRAG System for Multi-Hop Question Answering"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Answer command
    answer_parser = subparsers.add_parser("answer", help="Answer a single question")
    answer_parser.add_argument("question", type=str, help="The question to answer")
    answer_parser.add_argument("--llm-provider", type=str, help="LLM provider (ollama, openai)")
    answer_parser.add_argument("--llm-model", type=str, help="LLM model name")
    answer_parser.set_defaults(func=answer_question)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark evaluation")
    bench_parser.add_argument("dataset", type=str, help="Path to 2WikiMultiHopQA dataset")
    bench_parser.add_argument("--limit", type=int, help="Limit number of questions")
    bench_parser.add_argument("--output", "-o", type=str, help="Output file for results")
    bench_parser.add_argument("--gold-context", action="store_true", help="Use gold context")
    bench_parser.add_argument("--verbose", "-v", action="store_true", default=True)
    bench_parser.add_argument("--llm-provider", type=str, help="LLM provider")
    bench_parser.add_argument("--llm-model", type=str, help="LLM model name")
    bench_parser.set_defaults(func=run_benchmark)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Interactive demo mode")
    demo_parser.add_argument("--llm-provider", type=str, help="LLM provider")
    demo_parser.add_argument("--llm-model", type=str, help="LLM model name")
    demo_parser.set_defaults(func=demo_mode)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test system components")
    test_parser.add_argument("--dry-run", action="store_true", help="Skip network tests")
    test_parser.set_defaults(func=test_components)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
