"""Benchmark runner for 2WikiMultiHopQA evaluation."""

import json
import logging
from pathlib import Path
from typing import Optional, Iterator
from tqdm import tqdm

from ..models import Question, QuestionType, Answer, Triple
from .metrics import compute_joint_metrics, MetricsAggregator

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs evaluation on 2WikiMultiHopQA benchmark.

    Loads the dataset, runs the QA system, and computes metrics.
    """

    def __init__(
        self,
        qa_system,
        dataset_path: Optional[str] = None,
    ):
        """Initialize benchmark runner.

        Args:
            qa_system: The GraphRAG QA system to evaluate
            dataset_path: Path to 2WikiMultiHopQA dataset file
        """
        self.qa_system = qa_system
        self.dataset_path = dataset_path
        self.aggregator = MetricsAggregator()

    def load_dataset(
        self,
        path: Optional[str] = None,
        split: str = "dev",
        limit: Optional[int] = None,
    ) -> Iterator[Question]:
        """Load questions from 2WikiMultiHopQA dataset.

        Args:
            path: Path to dataset file (JSON)
            split: Dataset split (train, dev, test)
            limit: Optional limit on number of questions

        Yields:
            Question objects
        """
        path = path or self.dataset_path

        if path is None:
            raise ValueError("No dataset path provided")

        with open(path, "r") as f:
            data = json.load(f)

        count = 0
        for item in data:
            if limit and count >= limit:
                break

            question = self._parse_question(item)
            yield question
            count += 1

    def _parse_question(self, item: dict) -> Question:
        """Parse a question from dataset format.

        Args:
            item: Raw question dict from dataset

        Returns:
            Question object
        """
        # Map question type
        type_mapping = {
            "comparison": QuestionType.COMPARISON,
            "inference": QuestionType.INFERENCE,
            "compositional": QuestionType.COMPOSITIONAL,
            "bridge_comparison": QuestionType.BRIDGE_COMPARISON,
        }

        raw_type = item.get("type", "unknown")
        question_type = type_mapping.get(raw_type, QuestionType.UNKNOWN)

        # Parse supporting facts
        supporting_facts = []
        for sf in item.get("supporting_facts", []):
            if isinstance(sf, list) and len(sf) >= 2:
                supporting_facts.append((sf[0], sf[1]))

        # Parse evidence triples
        evidence = []
        for evi in item.get("evidences", []):
            if isinstance(evi, list) and len(evi) >= 3:
                evidence.append(Triple(
                    subject=evi[0],
                    predicate=evi[1],
                    obj=evi[2],
                ))

        # Parse context paragraphs
        context = []
        for ctx in item.get("context", []):
            if isinstance(ctx, list) and len(ctx) >= 2:
                context.append({
                    "title": ctx[0],
                    "sentences": ctx[1] if isinstance(ctx[1], list) else [ctx[1]],
                })

        return Question(
            id=item.get("_id", ""),
            text=item.get("question", ""),
            question_type=question_type,
            context_paragraphs=context,
            gold_answer=item.get("answer", ""),
            gold_supporting_facts=supporting_facts,
            gold_evidence=evidence,
        )

    def run(
        self,
        path: Optional[str] = None,
        split: str = "dev",
        limit: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """Run evaluation on dataset.

        Args:
            path: Dataset path
            split: Dataset split
            limit: Optional limit on examples
            verbose: Whether to show progress

        Returns:
            Aggregate metrics dict
        """
        questions = list(self.load_dataset(path, split, limit))

        if verbose:
            questions = tqdm(questions, desc="Evaluating")

        for question in questions:
            try:
                result = self._evaluate_single(question)
                self.aggregator.add(result, question.question_type.value)
            except Exception as e:
                logger.error(f"Error evaluating question {question.id}: {e}")
                # Add zero metrics for failed questions
                self.aggregator.add({
                    "answer_em": 0.0,
                    "answer_f1": 0.0,
                    "supporting_facts_em": 0.0,
                    "supporting_facts_f1": 0.0,
                    "joint_em": 0.0,
                    "joint_f1": 0.0,
                }, question.question_type.value)

        return self.aggregator.aggregate()

    def _evaluate_single(self, question: Question) -> dict:
        """Evaluate a single question.

        Args:
            question: Question to evaluate

        Returns:
            Metrics dict for this question
        """
        # Run QA system
        answer, supporting_facts, evidence = self.qa_system.answer_question(
            question.text,
            context_paragraphs=question.context_paragraphs,
        )

        # Extract predictions
        predicted_answer = answer.text if isinstance(answer, Answer) else str(answer)
        predicted_sf = supporting_facts or []

        # Extract evidence as triples
        predicted_evidence = None
        gold_evidence = None

        if evidence and question.gold_evidence:
            predicted_evidence = [
                (t.subject, t.predicate, t.obj)
                for t in (evidence.triples if hasattr(evidence, 'triples') else [])
            ]
            gold_evidence = [
                (t.subject, t.predicate, t.obj)
                for t in question.gold_evidence
            ]

        # Compute metrics
        metrics = compute_joint_metrics(
            answer_prediction=predicted_answer,
            answer_gold=question.gold_answer,
            sf_prediction=predicted_sf,
            sf_gold=question.gold_supporting_facts,
            evidence_prediction=predicted_evidence,
            evidence_gold=gold_evidence,
        )

        return metrics

    def run_with_gold_context(
        self,
        path: Optional[str] = None,
        limit: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """Run evaluation using gold context paragraphs.

        This tests the LLM's ability to answer questions when
        given the correct Wikipedia paragraphs.

        Args:
            path: Dataset path
            limit: Optional limit
            verbose: Show progress

        Returns:
            Aggregate metrics dict
        """
        questions = list(self.load_dataset(path, limit=limit))

        if verbose:
            questions = tqdm(questions, desc="Evaluating with gold context")

        for question in questions:
            try:
                # Format gold context
                context = self._format_gold_context(question)

                # Run QA with gold context
                answer, sf = self.qa_system.answer_with_context(
                    question.text,
                    context,
                    question.question_type,
                )

                predicted_answer = answer.text if isinstance(answer, Answer) else str(answer)
                predicted_sf = sf or []

                metrics = compute_joint_metrics(
                    answer_prediction=predicted_answer,
                    answer_gold=question.gold_answer,
                    sf_prediction=predicted_sf,
                    sf_gold=question.gold_supporting_facts,
                )

                self.aggregator.add(metrics, question.question_type.value)

            except Exception as e:
                logger.error(f"Error: {e}")
                self.aggregator.add({
                    "answer_em": 0.0,
                    "answer_f1": 0.0,
                    "joint_em": 0.0,
                    "joint_f1": 0.0,
                }, question.question_type.value)

        return self.aggregator.aggregate()

    def _format_gold_context(self, question: Question) -> str:
        """Format gold context paragraphs.

        Args:
            question: Question with gold context

        Returns:
            Formatted context string
        """
        parts = []
        for para in question.context_paragraphs:
            title = para.get("title", "")
            sentences = para.get("sentences", [])

            parts.append(f"Paragraph: {title}")
            for i, sent in enumerate(sentences):
                parts.append(f"[{i}] {sent}")
            parts.append("")

        return "\n".join(parts)

    def report(self) -> str:
        """Generate evaluation report.

        Returns:
            Formatted report string
        """
        return self.aggregator.report()

    def save_results(self, output_path: str):
        """Save evaluation results to file.

        Args:
            output_path: Path to save results
        """
        results = {
            "aggregate": self.aggregator.aggregate(),
            "by_type": self.aggregator.aggregate_by_type(),
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")
