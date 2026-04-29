"""Dataset characteristics table (paper Table tab:dataset-characteristics).

Per-benchmark stats for HotpotQA / 2WikiMHQA / MuSiQue:
  #Q, Avg Len, Type Dist (native labels), SPARQL-Answerable %, Avg Hops, Entity Density

Outputs (both under thirag-paper/):
  figures/tables/dataset-stats.tex   - booktabs fragment consumed via \\input
  figures/tables/dataset-stats.json  - same numbers, JSON-keyed by benchmark

Rules (see plan phase-02):
  * is_answerable: if absent/None on all records -> cell="---" with caption note
  * entities_linked (debug_info.entities_linked): missing -> 0
  * Avg Hops: HotpotQA=2.0 (by design); 2WikiMHQA=2.0 (hop_structure uniform);
    MuSiQue parsed from label (2hop/3hop*/4hop*).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path("~/Documents/2026/2026_02/thirag-paper/figures/tables").expanduser()

HYBRID_SOURCES = {
    "HotpotQA":  REPO / "data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged.jsonl",
    "2WikiMHQA": REPO / "data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl",
    "MuSiQue":   REPO / "data/cross_benchmark/musique/hybrid_musique_1000q_judged.jsonl",
}

# Display-order for Type Dist so tables are stable.
NATIVE_TYPE_ORDER = {
    "HotpotQA":  ["bridge", "comparison"],
    "2WikiMHQA": ["compositional", "comparison", "bridge_comparison", "inference"],
    "MuSiQue":   ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"],
}

HOTPOT_HOPS = 2.0   # Hotpot is 2-hop by design
WIKI_HOPS = 2.0     # 2Wiki hop_structure uniformly 2


def load_records(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def musique_hops_from_label(label: str) -> int:
    """`2hop` -> 2, `3hop1`/`3hop2` -> 3, `4hop*` -> 4. Fallback None."""
    if not label:
        return 0
    try:
        return int(label[0])
    except ValueError:
        return 0


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


@dataclass
class StatsRow:
    benchmark: str
    n_questions: int
    avg_len: float
    type_dist: list        # [(label, count, pct)]
    sparql_answerable_pct: float | None
    avg_hops: float
    entity_density: float


def compute(benchmark: str, path: Path) -> StatsRow:
    records = list(load_records(path))
    n = len(records)

    # Avg question length (whitespace tokens).
    lens = [len((r.get("question") or "").split()) for r in records]
    avg_len = sum(lens) / n if n else 0.0

    # Type distribution (native labels, display order; unknown-types appended).
    qtype_counts = Counter(r.get("question_type") for r in records)
    order = list(NATIVE_TYPE_ORDER[benchmark])
    for qt in sorted(qtype_counts):
        if qt not in order:
            order.append(qt)
    type_dist = []
    for qt in order:
        c = qtype_counts.get(qt, 0)
        if c == 0:
            continue
        type_dist.append((qt, c, 100.0 * c / n))

    # SPARQL-answerable %.
    ans_vals = [r.get("is_answerable") for r in records]
    ans_bool = [v for v in ans_vals if isinstance(v, bool)]
    sparql_answerable_pct = (
        100.0 * sum(ans_bool) / len(ans_bool) if ans_bool else None
    )

    # Avg hops.
    if benchmark == "HotpotQA":
        avg_hops = HOTPOT_HOPS
    elif benchmark == "2WikiMHQA":
        avg_hops = WIKI_HOPS
    elif benchmark == "MuSiQue":
        hops = [musique_hops_from_label(r.get("question_type") or "") for r in records]
        hops = [h for h in hops if h]
        avg_hops = sum(hops) / len(hops) if hops else 0.0
    else:
        avg_hops = 0.0

    # Entity density = question-side entities_linked length; missing -> 0.
    ent_counts = []
    for r in records:
        di = r.get("debug_info") or {}
        el = di.get("entities_linked")
        ent_counts.append(len(el) if isinstance(el, list) else 0)
    entity_density = sum(ent_counts) / n if n else 0.0

    return StatsRow(
        benchmark=benchmark,
        n_questions=n,
        avg_len=avg_len,
        type_dist=type_dist,
        sparql_answerable_pct=sparql_answerable_pct,
        avg_hops=avg_hops,
        entity_density=entity_density,
    )


def fmt_type_dist(dist) -> str:
    """Render Type Dist as `\\makecell`: 'label: XX\\%, ...'.

    Keep top entries; collapse the tail (beyond 4 entries) into `(rest)`.
    """
    top = dist[:4]
    tail_pct = sum(p for _, _, p in dist[4:])
    parts = [f"{tex_escape(l)}: {int(round(p))}\\%" for l, _, p in top]
    if tail_pct > 0:
        parts.append(f"(rest): {int(round(tail_pct))}\\%")
    # \\ for newlines inside makecell.
    return r"\makecell[l]{" + r"\\ ".join(parts) + "}"


def fmt_sparql_ans(pct) -> str:
    return f"{pct:.1f}" if pct is not None else "---"


def render_latex(rows: list[StatsRow]) -> str:
    lines: list[str] = []
    lines.append(r"% Auto-generated by dbpedia_qa/scripts/compute-dataset-stats.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(
        r"  \caption{Dataset characteristics. \#Q = evaluation subset size; "
        r"Avg Len = mean whitespace-tokenised question length; "
        r"Type Dist = native question-type labels with \%; "
        r"SPARQL-Ans.\% = share of questions with full entity+property linking coverage "
        r"over DBpedia (available for 2WikiMHQA only; cf.~Appendix~\ref{subsec:coverage}); "
        r"Avg Hops = coarse hop count (caveat: Type Dist preserves structural distinctions); "
        r"Ent.\ Dens.\ = mean question-side linked entities (missing counted as 0).}"
    )
    lines.append(r"  \label{tab:dataset-characteristics}")
    lines.append(r"  \begin{tabular}{l r r l r r r}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Dataset & \#Q & Avg Len & Type Dist & SPARQL-Ans.\% "
        r"& Avg Hops & Ent.\ Dens. \\"
    )
    lines.append(r"    \midrule")
    for r in rows:
        lines.append(
            f"    {tex_escape(r.benchmark)} & {r.n_questions} & "
            f"{r.avg_len:.1f} & {fmt_type_dist(r.type_dist)} & "
            f"{fmt_sparql_ans(r.sparql_answerable_pct)} & "
            f"{r.avg_hops:.2f} & {r.entity_density:.2f} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def row_as_json(r: StatsRow) -> dict:
    d = asdict(r)
    d["type_dist"] = [
        {"label": lbl, "count": c, "pct": round(p, 2)} for lbl, c, p in r.type_dist
    ]
    if r.sparql_answerable_pct is not None:
        d["sparql_answerable_pct"] = round(r.sparql_answerable_pct, 2)
    d["avg_len"] = round(r.avg_len, 2)
    d["avg_hops"] = round(r.avg_hops, 3)
    d["entity_density"] = round(r.entity_density, 3)
    return d


def probe_schema(path: Path, limit: int = 100) -> set[str]:
    fields: set[str] = set()
    for i, r in enumerate(load_records(path)):
        if i >= limit:
            break
        fields.update(r.keys())
        di = r.get("debug_info") or {}
        fields.update(f"debug_info.{k}" for k in di.keys())
    return fields


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[StatsRow] = []
    for bench, path in HYBRID_SOURCES.items():
        print(f"== {bench} ({path.name})")
        fields = probe_schema(path)
        has_ans = "is_answerable" in fields
        has_ent = "debug_info.entities_linked" in fields
        print(f"  schema probe: is_answerable={has_ans}, "
              f"debug_info.entities_linked={has_ent}")
        row = compute(bench, path)
        rows.append(row)
        print(f"  n={row.n_questions} avg_len={row.avg_len:.2f} "
              f"sparql_ans={fmt_sparql_ans(row.sparql_answerable_pct)} "
              f"avg_hops={row.avg_hops:.2f} ent_dens={row.entity_density:.2f}")
        for lbl, c, p in row.type_dist:
            print(f"    - {lbl:<22} n={c:4d}  {p:5.1f}%")

    tex = render_latex(rows)
    (OUT_DIR / "dataset-stats.tex").write_text(tex)

    json_rows = {r.benchmark: row_as_json(r) for r in rows}
    (OUT_DIR / "dataset-stats.json").write_text(
        json.dumps({"schema_version": 1, "benchmarks": json_rows}, indent=2)
    )

    print(f"\nwrote {OUT_DIR/'dataset-stats.tex'}")
    print(f"wrote {OUT_DIR/'dataset-stats.json'}")


if __name__ == "__main__":
    main()
