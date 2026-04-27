"""
Unified loader + file inventory for the 18 canonical judged JSONL files.

Conventions carried into downstream phases:
- CSV float columns use float_format="%.6f"
- question_id is the join key across all files; collisions within a file are fatal
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

# Repo root: scripts/analytics/load_judged.py → scripts/analytics → scripts → repo
_REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Canonical file registry — 18 entries
# ---------------------------------------------------------------------------
FILE_INVENTORY: dict[tuple[str, str, str], Path] = {
    # 2Wiki R1
    ("2wiki", "hybrid", "r1"): _REPO / "data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl",
    ("2wiki", "linearrag", "r1"): _REPO / "data/eval_1000q/linearrag/linearrag_1000q_judged.jsonl",
    ("2wiki", "logicrag", "r1"): _REPO / "data/eval_1000q/logicrag/logicrag_1000q_traced_judged.jsonl",
    # 2Wiki R2
    ("2wiki", "hybrid", "r2"): _REPO / "data/eval_1000q/hybrid/hybrid_eval_1000_run2_judged.jsonl",
    ("2wiki", "linearrag", "r2"): _REPO / "data/eval_1000q/linearrag/linearrag_1000q_run2_judged.jsonl",
    ("2wiki", "logicrag", "r2"): _REPO / "data/eval_1000q/logicrag/logicrag_1000q_traced_run2_judged.jsonl",
    # HotPotQA R1
    ("hotpotqa", "hybrid", "r1"): _REPO / "data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged.jsonl",
    ("hotpotqa", "linearrag", "r1"): _REPO / "data/cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_judged.jsonl",
    ("hotpotqa", "logicrag", "r1"): _REPO / "data/cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_judged.jsonl",
    # HotPotQA R2
    ("hotpotqa", "hybrid", "r2"): _REPO / "data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_run2_judged.jsonl",
    ("hotpotqa", "linearrag", "r2"): _REPO / "data/cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_run2_judged.jsonl",
    ("hotpotqa", "logicrag", "r2"): _REPO / "data/cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_run2_judged.jsonl",
    # MuSiQue R1
    ("musique", "hybrid", "r1"): _REPO / "data/cross_benchmark/musique/hybrid_musique_1000q_judged.jsonl",
    ("musique", "linearrag", "r1"): _REPO / "data/cross_benchmark/musique/linearrag_musique_1000q_judged.jsonl",
    ("musique", "logicrag", "r1"): _REPO / "data/cross_benchmark/musique/logicrag_musique_1000q_judged.jsonl",
    # MuSiQue R2
    ("musique", "hybrid", "r2"): _REPO / "data/cross_benchmark/musique/hybrid_musique_1000q_run2_judged.jsonl",
    ("musique", "linearrag", "r2"): _REPO / "data/cross_benchmark/musique/linearrag_musique_1000q_run2_judged.jsonl",
    ("musique", "logicrag", "r2"): _REPO / "data/cross_benchmark/musique/logicrag_musique_1000q_run2_judged.jsonl",
}

# ---------------------------------------------------------------------------
# Minimum expected fields per system
# ---------------------------------------------------------------------------
EXPECTED_FIELDS: dict[str, set[str]] = {
    "hybrid": {
        "question_id",
        "judge_correct",
        "question_type",
        "method",
        "route_source",
        "sub_question_log",
        "latency_seconds",
    },
    "linearrag": {
        "question_id",
        "judge_correct",
        "question_type",
        "predicted_answer",
        "gold_answer",
    },
    "logicrag": {
        "question_id",
        "judge_correct",
        "question_type",
        "predicted_answer",
        "gold_answer",
        "method",
    },
}


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> dict[str, dict]:
    """Read a JSONL file, key by question_id. Fails loud on duplicate qids."""
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{lineno} JSON parse error: {exc}") from exc
            qid = row.get("question_id")
            if qid is None:
                raise RuntimeError(f"{path}:{lineno} missing question_id")
            if qid in rows:
                raise RuntimeError(f"{path}:{lineno} duplicate question_id={qid!r}")
            rows[qid] = row
    return rows


def load_judged(bench: str, sys_: str, run: str) -> dict[str, dict]:
    """
    Load canonical judged JSONL for (bench, sys_, run).

    Returns: dict[question_id -> row_dict]
    """
    key = (bench, sys_, run)
    if key not in FILE_INVENTORY:
        raise KeyError(f"Unknown (bench, sys, run): {key!r}. Check FILE_INVENTORY.")

    return _read_jsonl(FILE_INVENTORY[key])


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
def _sha256_prefix(path: Path, n: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def inventory() -> list[dict]:
    """
    Return metadata for each of the 18 canonical files.
    Each dict: bench, sys, run, path, row_count, sha256_prefix, mtime_iso.
    """
    records = []
    for (bench, sys_, run), path in FILE_INVENTORY.items():
        if not path.exists():
            records.append(
                {
                    "bench": bench,
                    "sys": sys_,
                    "run": run,
                    "path": str(path),
                    "row_count": None,
                    "sha256_prefix": None,
                    "mtime_iso": None,
                    "missing": True,
                }
            )
            continue
        row_count = sum(1 for ln in open(path, encoding="utf-8", errors="replace") if ln.strip())
        sha = _sha256_prefix(path)
        mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        records.append(
            {
                "bench": bench,
                "sys": sys_,
                "run": run,
                "path": str(path.relative_to(_REPO)),
                "row_count": row_count,
                "sha256_prefix": sha,
                "mtime_iso": mtime,
                "missing": False,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def validate_schema(rows: dict[str, dict], sys_: str) -> list[str]:
    """
    Check that every row contains the minimum expected fields for sys_.
    Returns list of warning strings (empty = clean).
    """
    expected = EXPECTED_FIELDS.get(sys_, set())
    if not expected:
        return [f"No EXPECTED_FIELDS defined for system={sys_!r}"]

    warnings: list[str] = []
    missing_by_field: dict[str, int] = {}
    for row in rows.values():
        for field in expected:
            if field not in row:
                missing_by_field[field] = missing_by_field.get(field, 0) + 1

    for field, count in missing_by_field.items():
        warnings.append(f"sys={sys_!r} field={field!r} missing in {count}/{len(rows)} rows")
    return warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_inventory_table(records: list[dict]) -> None:
    header = f"{'bench':<10} {'sys':<10} {'run':<4} {'rows':>6} {'sha256':>8}  {'mtime':<20}  path"
    print(header)
    print("-" * len(header))
    for r in records:
        rows_str = str(r["row_count"]) if r["row_count"] is not None else "MISSING"
        sha_str = r["sha256_prefix"] or "—"
        mtime_str = r["mtime_iso"] or "—"
        print(
            f"{r['bench']:<10} {r['sys']:<10} {r['run']:<4} {rows_str:>6} {sha_str:>8}  "
            f"{mtime_str:<20}  {r['path']}"
        )


def _run_validate() -> None:
    load_errors: list[str] = []
    # field_missing[(sys, field)] -> list of (bench, run) where it is absent
    field_missing: dict[tuple[str, str], list[str]] = {}

    for (bench, sys_, run) in FILE_INVENTORY:
        try:
            rows = load_judged(bench, sys_, run)
        except Exception as exc:
            load_errors.append(f"LOAD ERROR ({bench},{sys_},{run}): {exc}")
            continue
        for w in validate_schema(rows, sys_):
            # w format: "sys=X field=Y missing in N/1000 rows"
            key = (sys_, w.split("field=")[1].split(" missing")[0] if "field=" in w else w)
            field_missing.setdefault(key, []).append(f"{bench}/{run}")

    # Emit aggregated warnings
    all_warnings: list[str] = list(load_errors)
    for (sys_, field), locations in field_missing.items():
        all_warnings.append(
            f"sys={sys_!r} field={field} absent in {len(locations)} file(s): {', '.join(locations)}"
        )

    if all_warnings:
        print(f"{len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(f"  WARN: {w}")
    else:
        print("All 18 files validated — no warnings.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judged JSONL inventory and validation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inventory", action="store_true", help="Print inventory table")
    group.add_argument("--validate", action="store_true", help="Validate all 18 files")
    args = parser.parse_args()

    if args.inventory:
        _print_inventory_table(inventory())
    elif args.validate:
        _run_validate()


if __name__ == "__main__":
    main()
