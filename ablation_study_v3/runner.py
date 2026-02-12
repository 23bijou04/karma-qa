"""Main runner for Ablation Study V3: Controlled KG Detail Levels."""

import argparse
import json
import os
import time
import importlib
import logging
from dataclasses import asdict

from groq import Groq
from SPARQLWrapper import SPARQLWrapper, JSON

# Import evaluation.metrics directly to avoid evaluation/__init__.py side effects
_metrics_spec = importlib.util.spec_from_file_location(
    "evaluation.metrics",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation", "metrics.py"),
)
_metrics = importlib.util.module_from_spec(_metrics_spec)
_metrics_spec.loader.exec_module(_metrics)
compute_em = _metrics.compute_em
compute_f1 = _metrics.compute_f1

from .sampling import create_stratified_sample
from .levels_base import answer_level_0, answer_level_1, answer_level_4, LevelResult
from .levels_kg import answer_level_2, answer_level_3
from .reporting import generate_report

logger = logging.getLogger(__name__)

LEVEL_NAMES = {
    0: "No KG (Entity Names Only)",
    1: "Entity-Only KG",
    2: "Entity + Important Relations",
    3: "Full KG + Wikipedia Fallback",
    4: "Oracle KG",
}

LEVEL_FUNCTIONS = {
    0: answer_level_0,
    1: answer_level_1,
    2: answer_level_2,
    3: answer_level_3,
    4: answer_level_4,
}

LLM_DELAY = 4.0  # seconds between questions (Groq rate limit)


class AblationV3Runner:
    """Runs the multi-level ablation study."""

    DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"

    def __init__(
        self, levels=None, n_per_type=50, seed=42,
        groq_api_key=None, output_dir="data/ablation_v3", resume=True,
    ):
        self.levels = levels or [0, 1, 2, 3, 4]
        self.n_per_type = n_per_type
        self.seed = seed
        self.output_dir = output_dir
        self.resume = resume

        os.makedirs(output_dir, exist_ok=True)

        self.client = Groq(api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))

        self.sparql = SPARQLWrapper(self.DBPEDIA_ENDPOINT)
        self.sparql.setReturnFormat(JSON)
        self.sparql.addCustomHttpHeader("User-Agent", "AblationStudyV3/1.0")
        self.sparql.setTimeout(30)

    def _get_or_create_sample(self) -> list[dict]:
        sample_path = os.path.join(self.output_dir, f"sample_{self.n_per_type * 4}.json")
        if os.path.exists(sample_path):
            with open(sample_path, "r") as f:
                return json.load(f)

        sample = create_stratified_sample(
            "data/dev_full.json",
            n_per_type=self.n_per_type,
            seed=self.seed,
            output_file=sample_path,
        )
        print(f"  Created sample: {len(sample)} questions -> {sample_path}")
        return sample

    def _load_checkpoint(self, level: int) -> set[str]:
        """Load completed question IDs from checkpoint file."""
        path = os.path.join(self.output_dir, f"level_{level}_results.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        done.add(r["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        return done

    def _append_result(self, level: int, result: LevelResult):
        path = os.path.join(self.output_dir, f"level_{level}_results.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    def run(self):
        sample = self._get_or_create_sample()

        print("=" * 70)
        print("  ABLATION STUDY V3: Controlled KG Detail Levels")
        print(f"  Sample: {len(sample)} questions ({self.n_per_type} per type)")
        print(f"  Levels: {[LEVEL_NAMES[l] for l in self.levels]}")
        print("=" * 70)

        for level in self.levels:
            self.run_level(level, sample)

        generate_report(self.output_dir, self.levels)

    def run_level(self, level: int, sample: list[dict]):
        name = LEVEL_NAMES[level]
        func = LEVEL_FUNCTIONS[level]
        done_ids = self._load_checkpoint(level) if self.resume else set()

        remaining = [q for q in sample if q["_id"] not in done_ids]

        print(f"\n{'=' * 70}")
        print(f"  Level {level}: {name}")
        print(f"  Questions: {len(remaining)} remaining ({len(done_ids)} checkpointed)")
        print(f"{'=' * 70}")

        if not remaining:
            print("  All questions already completed.")
            return

        correct = 0
        total = 0

        for i, q in enumerate(remaining, 1):
            result = func(question=q, client=self.client, sparql=self.sparql)

            # Compute official metrics
            if result.predicted_answer:
                result.em = compute_em(result.predicted_answer, result.gold_answer)
                result.f1 = compute_f1(result.predicted_answer, result.gold_answer)

            self._append_result(level, result)

            total += 1
            if result.em > 0:
                correct += 1

            # Progress
            status = "+" if result.em > 0 else "-"
            pct = correct / total * 100
            q_text = q['question'][:80]
            print(
                f"  [{i}/{len(remaining)}] {status} "
                f"EM={result.em:.0f} F1={result.f1:.2f} "
                f"src={result.answer_source:9s} "
                f"| {q['type']:18s} "
                f"| Q: {q_text}"
            )
            print(
                f"              "
                f"pred='{result.predicted_answer[:50]}' "
                f"gold='{result.gold_answer[:50]}' "
                f"({pct:.0f}% so far)"
            )

            time.sleep(LLM_DELAY)

        print(f"\n  Level {level} done: {correct}/{total} EM ({correct/total*100:.1f}%)")


def main():
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Ablation Study V3: Controlled KG Detail Levels"
    )
    parser.add_argument("--api-key", "-k", help="Groq API key (or GROQ_API_KEY env var)")
    parser.add_argument(
        "--levels", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="Which levels to run (e.g., --levels 0 1 2)",
    )
    parser.add_argument("--n-per-type", type=int, default=50, help="Questions per type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="data/ablation_v3", help="Output directory")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--report-only", action="store_true", help="Only generate report")
    args = parser.parse_args()

    if args.report_only:
        generate_report(args.output_dir, args.levels)
        return

    runner = AblationV3Runner(
        levels=args.levels,
        n_per_type=args.n_per_type,
        seed=args.seed,
        groq_api_key=args.api_key,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
    runner.run()


if __name__ == "__main__":
    main()
