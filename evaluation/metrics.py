"""Evaluation metrics for 2WikiMultiHopQA benchmark.

Implements EM and F1 metrics for:
- Answer prediction
- Supporting facts prediction
- Evidence generation
- Joint metrics
"""

import re
import string
from collections import Counter
from typing import Optional


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    Follows the official 2WikiMultiHopQA normalization:
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Remove extra whitespace

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def normalize_whitespace(text: str) -> str:
        return ' '.join(text.split())

    answer = answer.lower()
    answer = remove_punctuation(answer)
    answer = remove_articles(answer)
    answer = normalize_whitespace(answer)

    return answer.strip()


def compute_em(prediction: str, ground_truth: str) -> float:
    """Compute Exact Match score.

    Args:
        prediction: Predicted answer
        ground_truth: Gold answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score based on token overlap.

    Args:
        prediction: Predicted answer
        ground_truth: Gold answer

    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_containment(prediction: str, ground_truth: str) -> float:
    """Check if gold answer appears anywhere in the prediction (upper bound)."""
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))


def compute_supporting_facts_em(
    prediction: list[tuple[str, int]],
    ground_truth: list[tuple[str, int]],
) -> float:
    """Compute EM for supporting facts.

    Args:
        prediction: List of (title, sentence_idx) tuples
        ground_truth: Gold supporting facts

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_set = set(prediction)
    gold_set = set(ground_truth)
    return float(pred_set == gold_set)


def compute_supporting_facts_f1(
    prediction: list[tuple[str, int]],
    ground_truth: list[tuple[str, int]],
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for supporting facts.

    Args:
        prediction: List of (title, sentence_idx) tuples
        ground_truth: Gold supporting facts

    Returns:
        Tuple of (precision, recall, f1)
    """
    pred_set = set(prediction)
    gold_set = set(ground_truth)

    if len(pred_set) == 0 and len(gold_set) == 0:
        return 1.0, 1.0, 1.0

    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    if len(gold_set) == 0:
        return 0.0, 0.0, 0.0

    correct = len(pred_set & gold_set)
    precision = correct / len(pred_set)
    recall = correct / len(gold_set)

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_evidence_em(
    prediction: list[tuple[str, str, str]],
    ground_truth: list[tuple[str, str, str]],
) -> float:
    """Compute EM for evidence triples.

    Args:
        prediction: List of (subject, predicate, object) tuples
        ground_truth: Gold evidence triples

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    def normalize_triple(t: tuple) -> tuple:
        return tuple(normalize_answer(str(x)) for x in t)

    pred_set = set(normalize_triple(t) for t in prediction)
    gold_set = set(normalize_triple(t) for t in ground_truth)

    return float(pred_set == gold_set)


def compute_evidence_f1(
    prediction: list[tuple[str, str, str]],
    ground_truth: list[tuple[str, str, str]],
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for evidence triples.

    Args:
        prediction: List of (subject, predicate, object) tuples
        ground_truth: Gold evidence triples

    Returns:
        Tuple of (precision, recall, f1)
    """
    def normalize_triple(t: tuple) -> tuple:
        return tuple(normalize_answer(str(x)) for x in t)

    pred_set = set(normalize_triple(t) for t in prediction)
    gold_set = set(normalize_triple(t) for t in ground_truth)

    if len(pred_set) == 0 and len(gold_set) == 0:
        return 1.0, 1.0, 1.0

    if len(pred_set) == 0 or len(gold_set) == 0:
        return 0.0, 0.0, 0.0

    correct = len(pred_set & gold_set)
    precision = correct / len(pred_set)
    recall = correct / len(gold_set)

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_joint_metrics(
    answer_prediction: str,
    answer_gold: str,
    sf_prediction: list[tuple[str, int]],
    sf_gold: list[tuple[str, int]],
    evidence_prediction: Optional[list[tuple[str, str, str]]] = None,
    evidence_gold: Optional[list[tuple[str, str, str]]] = None,
) -> dict:
    """Compute all metrics including joint metrics.

    Following 2WikiMultiHopQA official evaluation:
    Joint F1 = 2 * P_joint * R_joint / (P_joint + R_joint)
    where P_joint = P_ans * P_sup * P_evi
    and R_joint = R_ans * R_sup * R_evi

    Args:
        answer_prediction: Predicted answer
        answer_gold: Gold answer
        sf_prediction: Predicted supporting facts
        sf_gold: Gold supporting facts
        evidence_prediction: Optional predicted evidence
        evidence_gold: Optional gold evidence

    Returns:
        Dict with all metrics
    """
    # Answer metrics
    answer_em = compute_em(answer_prediction, answer_gold)
    answer_f1 = compute_f1(answer_prediction, answer_gold)

    # Convert F1 to precision/recall for joint calculation
    # For answer, we use F1 as both P and R approximation
    ans_prec = answer_f1
    ans_recall = answer_f1

    # Supporting facts metrics
    sf_em = compute_supporting_facts_em(sf_prediction, sf_gold)
    sf_prec, sf_recall, sf_f1 = compute_supporting_facts_f1(sf_prediction, sf_gold)

    results = {
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "supporting_facts_em": sf_em,
        "supporting_facts_f1": sf_f1,
        "supporting_facts_precision": sf_prec,
        "supporting_facts_recall": sf_recall,
    }

    # Evidence metrics (if provided)
    if evidence_prediction is not None and evidence_gold is not None:
        evi_em = compute_evidence_em(evidence_prediction, evidence_gold)
        evi_prec, evi_recall, evi_f1 = compute_evidence_f1(
            evidence_prediction, evidence_gold
        )

        results["evidence_em"] = evi_em
        results["evidence_f1"] = evi_f1
        results["evidence_precision"] = evi_prec
        results["evidence_recall"] = evi_recall

        # Joint metrics with evidence
        joint_prec = ans_prec * sf_prec * evi_prec
        joint_recall = ans_recall * sf_recall * evi_recall
    else:
        # Joint metrics without evidence
        joint_prec = ans_prec * sf_prec
        joint_recall = ans_recall * sf_recall

    # Compute joint F1
    if joint_prec + joint_recall > 0:
        joint_f1 = (2 * joint_prec * joint_recall) / (joint_prec + joint_recall)
    else:
        joint_f1 = 0.0

    # Joint EM is 1 only if all tasks have exact match
    if evidence_prediction is not None and evidence_gold is not None:
        joint_em = float(
            answer_em == 1.0
            and sf_em == 1.0
            and results.get("evidence_em", 0) == 1.0
        )
    else:
        joint_em = float(answer_em == 1.0 and sf_em == 1.0)

    results["joint_em"] = joint_em
    results["joint_f1"] = joint_f1

    return results


class MetricsAggregator:
    """Aggregates metrics across multiple examples."""

    def __init__(self):
        self.results = []
        self.by_type = {}

    def add(self, metrics: dict, question_type: Optional[str] = None):
        """Add metrics for one example.

        Args:
            metrics: Dict of metrics
            question_type: Optional question type for breakdown
        """
        self.results.append(metrics)

        if question_type:
            if question_type not in self.by_type:
                self.by_type[question_type] = []
            self.by_type[question_type].append(metrics)

    def aggregate(self) -> dict:
        """Compute aggregate metrics.

        Returns:
            Dict with mean metrics
        """
        if not self.results:
            return {}

        # Compute means for each metric
        all_keys = set()
        for r in self.results:
            all_keys.update(r.keys())

        aggregated = {}
        for key in all_keys:
            values = [r.get(key, 0) for r in self.results]
            aggregated[f"mean_{key}"] = sum(values) / len(values)

        aggregated["num_examples"] = len(self.results)
        return aggregated

    def aggregate_by_type(self) -> dict:
        """Compute aggregate metrics by question type.

        Returns:
            Dict mapping type to aggregate metrics
        """
        result = {}
        for qtype, type_results in self.by_type.items():
            if not type_results:
                continue

            all_keys = set()
            for r in type_results:
                all_keys.update(r.keys())

            type_agg = {}
            for key in all_keys:
                values = [r.get(key, 0) for r in type_results]
                type_agg[f"mean_{key}"] = sum(values) / len(values)

            type_agg["num_examples"] = len(type_results)
            result[qtype] = type_agg

        return result

    def report(self) -> str:
        """Generate human-readable report.

        Returns:
            Formatted report string
        """
        lines = ["=" * 60, "Evaluation Results", "=" * 60]

        agg = self.aggregate()
        lines.append(f"\nOverall ({agg.get('num_examples', 0)} examples):")
        lines.append(f"  Answer EM:           {agg.get('mean_answer_em', 0):.2%}")
        lines.append(f"  Answer F1:           {agg.get('mean_answer_f1', 0):.2%}")
        lines.append(f"  Supporting Facts EM: {agg.get('mean_supporting_facts_em', 0):.2%}")
        lines.append(f"  Supporting Facts F1: {agg.get('mean_supporting_facts_f1', 0):.2%}")

        if "mean_evidence_f1" in agg:
            lines.append(f"  Evidence EM:         {agg.get('mean_evidence_em', 0):.2%}")
            lines.append(f"  Evidence F1:         {agg.get('mean_evidence_f1', 0):.2%}")

        lines.append(f"  Joint EM:            {agg.get('mean_joint_em', 0):.2%}")
        lines.append(f"  Joint F1:            {agg.get('mean_joint_f1', 0):.2%}")

        # By type breakdown
        by_type = self.aggregate_by_type()
        if by_type:
            lines.append("\n" + "-" * 60)
            lines.append("Results by Question Type:")

            for qtype, type_agg in sorted(by_type.items()):
                lines.append(f"\n  {qtype} ({type_agg.get('num_examples', 0)} examples):")
                lines.append(f"    Answer F1: {type_agg.get('mean_answer_f1', 0):.2%}")
                lines.append(f"    SF F1:     {type_agg.get('mean_supporting_facts_f1', 0):.2%}")
                lines.append(f"    Joint F1:  {type_agg.get('mean_joint_f1', 0):.2%}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
