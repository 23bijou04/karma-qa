"""Phase 05: Companion Analytics Report Builder.

Produces 4 markdown reports (3 per-benchmark + 1 cross-benchmark).
Usage: python -m scripts.analytics.build_report [--benchmark all|2wiki|hotpotqa|musique] [--skip-compute]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.analytics.analytics_render_helpers import (
    CSV_MAP,
    DIM_META,
    DIM_NOTES,
    KEY_FINDINGS,
    collect_em_rows,
    collect_flip_rows,
    collect_sig_rows,
    load_csv,
    sha256_file,
    source_footer,
    to_markdown,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_REPO = Path(__file__).resolve().parents[2]
_DATA = _REPO / "plans/260422-1027-companion-analytics-18-judged-jsonls/reports/data"
_OUT = _DATA.parent

BENCHMARKS = ["2wiki", "hotpotqa", "musique"]

_DIM_MODULES = [
    "route_x_type", "decomp_stats", "fallback_chains", "subq_metrics",
    "failure_modes", "nonjudge_metrics", "latency", "flip_rate", "significance",
]

_BENCH_TITLE = {"2wiki": "2WikiMultiHopQA", "hotpotqa": "HotPotQA", "musique": "MuSiQue"}

_ORDERED_DIMS = [
    "route_x_type", "decomp_stats", "failure_modes", "nonjudge_metrics",
    "fallback_chains", "latency", "subq_metrics", "flip_rate", "significance",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        log.warning("git rev-parse failed — using 'unknown'")
        return "unknown"


def run_all_dims(bench: str) -> None:  # noqa: ARG001
    """Invoke each dimension module via subprocess. Aborts on non-zero exit."""
    for module in _DIM_MODULES:
        cmd = [sys.executable, "-m", f"scripts.analytics.{module}", "--all"]
        log.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=_REPO)
        if result.returncode != 0:
            sys.exit(f"ERROR: {module} exited with code {result.returncode}. Aborting.")


def render_inventory_header(bench: str) -> str:
    from scripts.analytics.load_judged import inventory
    records = inventory()
    if bench != "all":
        records = [r for r in records if r["bench"] == bench]
    df = pd.DataFrame(records)[["bench", "sys", "run", "row_count", "sha256_prefix", "mtime_iso", "path"]]
    df = df.sort_values(["bench", "sys", "run"]).reset_index(drop=True)
    return f"## Inputs\n\n{to_markdown(df)}\n"


def render_dim_section(dim_key: str, bench: str) -> str:
    dim_id, title = DIM_META[dim_key]
    lines: list[str] = [f"## Dim {dim_id} — {title}\n"]
    if dim_key in DIM_NOTES:
        lines.append(DIM_NOTES[dim_key] + "\n")

    for tmpl in CSV_MAP[dim_key]:
        filename = tmpl.replace("{bench}", bench)
        csv_path = _DATA / filename
        run_label = " (R1)" if "_r1" in filename else (" (R2)" if "_r2" in filename else "")

        if not csv_path.exists():
            lines.append(f"_{filename}: **MISSING**_\n")
            continue

        df = load_csv(csv_path)
        if "benchmark" in df.columns and bench != "all":
            df = df[df["benchmark"] == bench].reset_index(drop=True)
        if df.empty:
            lines.append(f"_{filename}{run_label}: no rows for benchmark={bench}_\n")
            continue

        df = df.sort_values(list(df.columns)).reset_index(drop=True)
        if run_label:
            lines.append(f"### {run_label.strip()}\n")
        lines.append(to_markdown(df, max_rows=30, csv_path=csv_path))
        lines.append("")
        lines.append(source_footer([csv_path], min(30, len(df))))
        lines.append("")

    return "\n".join(lines)


def render_benchmark_report(bench: str, git_sha: str, ts: str) -> str:
    parts = [
        f"# Companion Analytics — {_BENCH_TITLE[bench]}",
        "",
        f"Generated: {ts}",
        f"Git SHA: {git_sha}",
        f"Repro: `python -m scripts.analytics.build_report --benchmark {bench} --skip-compute`",
        "",
        render_inventory_header(bench),
        "",
    ]
    for key in _ORDERED_DIMS:
        parts += [render_dim_section(key, bench), ""]

    # Appendix
    from scripts.analytics.load_judged import inventory
    parts.append("## Appendix: File SHA256\n")
    rows = [
        {"file": r["path"], "sha256[:16]": sha256_file(_REPO / r["path"]) if (_REPO / r["path"]).exists() else "MISSING"}
        for r in sorted([r for r in inventory() if r["bench"] == bench], key=lambda x: (x["sys"], x["run"]))
    ]
    parts += [to_markdown(pd.DataFrame(rows)), ""]
    return "\n".join(parts)


def render_cross_report(git_sha: str, ts: str) -> str:
    parts = [
        "# Companion Analytics — Cross-Benchmark Summary", "",
        f"Generated: {ts}", f"Git SHA: {git_sha}",
        "Repro: `python -m scripts.analytics.build_report --benchmark all --skip-compute`",
        "", "## Per-System Judge EM with Wilson 95% CI\n",
    ]
    em_df = collect_em_rows(_DATA, BENCHMARKS)
    if not em_df.empty:
        parts += [to_markdown(em_df), ""]

    parts.append("## R1↔R2 Flip-Rate Summary\n")
    flip_df = collect_flip_rows(_DATA, BENCHMARKS)
    if not flip_df.empty:
        parts += [to_markdown(flip_df), ""]

    parts.append("## Between-System Significance (Bonferroni-adjusted p)\n")
    sig_df = collect_sig_rows(_DATA, BENCHMARKS)
    if not sig_df.empty:
        parts += [to_markdown(sig_df), ""]

    parts += ["## Key Findings\n", KEY_FINDINGS]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build companion analytics markdown reports")
    parser.add_argument("--benchmark", default="all", choices=["all", "2wiki", "hotpotqa", "musique"])
    parser.add_argument("--skip-compute", action="store_true",
                        help="Reuse existing CSVs; skip running dimension scripts")
    args = parser.parse_args()

    if not args.skip_compute:
        run_all_dims(args.benchmark)

    git_sha = _git_sha()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    benches = BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    for bench in benches:
        content = render_benchmark_report(bench, git_sha, ts)
        out_path = _OUT / f"companion-analytics-{bench}.md"
        out_path.write_text(content, encoding="utf-8")
        log.info("Wrote %s (%d lines)", out_path.relative_to(_REPO), content.count("\n"))

    if args.benchmark == "all":
        content = render_cross_report(git_sha, ts)
        out_path = _OUT / "companion-analytics-cross.md"
        out_path.write_text(content, encoding="utf-8")
        log.info("Wrote %s (%d lines)", out_path.relative_to(_REPO), content.count("\n"))


if __name__ == "__main__":
    main()
