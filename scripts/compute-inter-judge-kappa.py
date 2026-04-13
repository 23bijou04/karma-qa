"""
Compute Cohen's kappa between Gemma3 12B and Claude Sonnet judges.
Both judges' decisions are available in Claude Sonnet files:
  - judge_correct       → Gemma3's decision (carried over)
  - judge_correct_claude → Claude Sonnet's decision
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "2wikimhqa": {
        "hybrid":    ROOT / "data/eval_1000q/hybrid/hybrid_eval_1000_judged_claude_sonnet.jsonl",
        "linearrag": ROOT / "data/eval_1000q/linearrag/linearrag_1000q_judged_claude_sonnet.jsonl",
        "logicrag":  ROOT / "data/eval_1000q/logicrag/logicrag_1000q_traced_judged_claude_sonnet.jsonl",
    },
    "hotpotqa": {
        "hybrid":    ROOT / "data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged_claude_sonnet.jsonl",
        "linearrag": ROOT / "data/cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_judged_claude_sonnet.jsonl",
        "logicrag":  ROOT / "data/cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_judged_claude_sonnet.jsonl",
    },
    "musique": {
        "hybrid":    ROOT / "data/cross_benchmark/musique/hybrid_musique_1000q_judged_claude_sonnet.jsonl",
        "linearrag": ROOT / "data/cross_benchmark/musique/linearrag_musique_1000q_judged_claude_sonnet.jsonl",
        "logicrag":  ROOT / "data/cross_benchmark/musique/logicrag_musique_1000q_judged_claude_sonnet.jsonl",
    },
}


def cohen_kappa(gemma_labels: list[bool], claude_labels: list[bool]) -> dict:
    """Compute Cohen's kappa from two binary label lists."""
    n = len(gemma_labels)
    if n == 0:
        return {"kappa": None, "n": 0, "agreement_pct": None}

    # Observed agreement
    both_correct   = sum(g and c for g, c in zip(gemma_labels, claude_labels))
    both_incorrect = sum(not g and not c for g, c in zip(gemma_labels, claude_labels))
    po = (both_correct + both_incorrect) / n

    # Expected agreement by chance
    p_gemma_correct  = sum(gemma_labels) / n
    p_claude_correct = sum(claude_labels) / n
    pe = (p_gemma_correct * p_claude_correct) + ((1 - p_gemma_correct) * (1 - p_claude_correct))

    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 1.0

    # Agreement matrix
    gemma_only  = sum(g and not c for g, c in zip(gemma_labels, claude_labels))
    claude_only = sum(not g and c for g, c in zip(gemma_labels, claude_labels))

    return {
        "kappa": round(kappa, 4),
        "n": n,
        "agreement_pct": round(po, 4),
        "agreement_matrix": {
            "both_correct":   both_correct,
            "both_incorrect": both_incorrect,
            "gemma_only":     gemma_only,
            "claude_only":    claude_only,
        },
    }


def load_pair(path: Path) -> tuple[list[bool], list[bool]]:
    """Load (gemma_labels, claude_labels) from a Claude Sonnet JSONL file."""
    gemma, claude = [], []
    skipped = 0
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            g = rec.get("judge_correct")
            c = rec.get("judge_correct_claude")
            if g is None or c is None:
                skipped += 1
                continue
            gemma.append(bool(g))
            claude.append(bool(c))
    if skipped:
        print(f"  [warn] {path.name}: {skipped} entries skipped (missing fields)")
    return gemma, claude


def interpret(kappa: float | None) -> str:
    if kappa is None:
        return "n/a"
    if kappa >= 0.81:
        return "almost perfect"
    if kappa >= 0.61:
        return "substantial"
    if kappa >= 0.41:
        return "moderate"
    return "fair/poor"


def main():
    results = {"per_file": {}, "per_benchmark": {}, "per_system": {}, "overall": {}}

    system_pairs: dict[str, tuple[list, list]] = {}
    all_g, all_c = [], []

    for benchmark, systems in FILES.items():
        bm_g, bm_c = [], []
        for system, path in systems.items():
            if not path.exists():
                print(f"[ERROR] Missing: {path}", file=sys.stderr)
                continue
            print(f"Loading {benchmark}/{system} ...")
            g_labels, c_labels = load_pair(path)
            stats = cohen_kappa(g_labels, c_labels)
            key = f"{benchmark}/{system}"
            results["per_file"][key] = stats
            print(f"  kappa={stats['kappa']}  agree={stats['agreement_pct']}  n={stats['n']}  [{interpret(stats['kappa'])}]")

            bm_g.extend(g_labels)
            bm_c.extend(c_labels)
            system_pairs.setdefault(system, ([], []))
            system_pairs[system][0].extend(g_labels)
            system_pairs[system][1].extend(c_labels)
            all_g.extend(g_labels)
            all_c.extend(c_labels)

        bm_stats = cohen_kappa(bm_g, bm_c)
        results["per_benchmark"][benchmark] = bm_stats
        print(f"  >> {benchmark} aggregate: kappa={bm_stats['kappa']}  [{interpret(bm_stats['kappa'])}]\n")

    for system, (g_labels, c_labels) in system_pairs.items():
        stats = cohen_kappa(g_labels, c_labels)
        results["per_system"][system] = stats
        print(f"System {system}: kappa={stats['kappa']}  n={stats['n']}  [{interpret(stats['kappa'])}]")

    overall = cohen_kappa(all_g, all_c)
    results["overall"] = overall
    print(f"\nOVERALL: kappa={overall['kappa']}  agree={overall['agreement_pct']}  n={overall['n']}  [{interpret(overall['kappa'])}]")

    out_path = ROOT / "data/eval_1000q/ablation/inter_judge_kappa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
