"""Shared utilities for MuSiQue analysis scripts.

Provides data loaders, metrics imports, and path constants used by all 5 analysis
scripts (per-hop audit, cross-system overlap, SPARQL disambiguation, decomposition
quality, DBpedia grounding).
"""
import importlib.util
import json
from pathlib import Path

# ── Path constants ──────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent
MUSIQUE_GT = BASE / "baselines/LogicRAG/dataset/musique.json"
HYBRID_JSONL = BASE / "data/cross_benchmark/musique/hybrid_musique_1000q_judged.jsonl"
LINEARRAG_JSONL = BASE / "data/cross_benchmark/musique/linearrag_musique_1000q_judged.jsonl"
LOGICRAG_JSONL = BASE / "data/cross_benchmark/musique/logicrag_musique_1000q_judged.jsonl"
ABLATION_DIR = BASE / "data/cross_benchmark/musique/ablation"

# ── Metrics import (avoids evaluation/__init__.py) ──────────────────────────
_spec = importlib.util.spec_from_file_location("metrics", BASE / "evaluation/metrics.py")
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
normalize_answer = _metrics.normalize_answer
compute_em = _metrics.compute_em
compute_f1 = _metrics.compute_f1
compute_containment = _metrics.compute_containment


# ── Data loaders ────────────────────────────────────────────────────────────
def load_musique_gt() -> dict[str, dict]:
    """Load MuSiQue GT as {question_id: record}."""
    with open(MUSIQUE_GT) as f:
        data = json.load(f)
    return {str(q["id"]): q for q in data}


def load_jsonl(path: Path) -> dict[str, dict]:
    """Load JSONL evaluation file as {question_id: record}."""
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                results[str(d["question_id"])] = d
    return results


def load_hybrid() -> dict[str, dict]:
    return load_jsonl(HYBRID_JSONL)


def load_linearrag() -> dict[str, dict]:
    return load_jsonl(LINEARRAG_JSONL)


def load_logicrag() -> dict[str, dict]:
    return load_jsonl(LOGICRAG_JSONL)


def ensure_ablation_dir():
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, filename: str) -> Path:
    """Save dict to ABLATION_DIR/filename. Returns path."""
    ensure_ablation_dir()
    out = ABLATION_DIR / filename
    with open(out, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out
