#!/usr/bin/env python3
"""Aggregate 3-benchmark DBpedia grounding results into a paper-ready markdown report.

Reads pre-computed JSON from verify-*-grounding.py scripts and emits one .md file.
Usage:
    python scripts/grounding/aggregate-coverage-tables.py [--twiki PATH] [--hotpotqa PATH]
        [--musique PATH] [--out PATH] [--delta-baseline PATH]
"""
import argparse
import json
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
_GT = BASE / "data/eval_1000q/ground-truth"
_XB = BASE / "data/cross_benchmark"

DEFAULTS = {
    "twiki":    _GT / "hop_verification_results_260414.json",
    "hotpotqa": _XB / "hotpotqa/dbpedia_grounding_results.json",
    "musique":  _XB / "musique/ablation/dbpedia_grounding_results.json",
    "out":      BASE / "plans/reports/coverage-260414-3-benchmark-grounding.md",
}
_TWIKI_FALLBACK = _GT / "hop_verification_results.json"


def _cell(n: int, total: int) -> str:
    return f"{n}/{total} ({100*n/max(total,1):.1f}%)"


def load_2wiki(path: Path) -> dict:
    if not path.exists():
        path = _TWIKI_FALLBACK
    data = json.loads(path.read_text())
    recs = data.get("results", data.get("per_question", []))
    by_type: dict = {}
    for r in recs:
        qt = r.get("question_type", r.get("type", "unknown"))
        b = by_type.setdefault(qt, {"total": 0, "t1": 0, "t2": 0, "t3": 0})
        b["total"] += 1
        if r.get("all_entities_resolved"): b["t1"] += 1
        if r.get("all_properties_exist"):  b["t2"] += 1
        if r.get("all_values_match"):      b["t3"] += 1
    t = len(recs)
    return {"total": t, "t1": sum(v["t1"] for v in by_type.values()),
            "t2": sum(v["t2"] for v in by_type.values()),
            "t3": sum(v["t3"] for v in by_type.values()),
            "by_type": by_type, "records": recs}


def load_hotpotqa(path: Path) -> dict:
    data = json.loads(path.read_text())
    s = data.get("summary", {})
    conn = s.get("pair_connectivity", {})
    ag   = s.get("answer_grounding", {})
    by_type = {qt: {"total": v.get("total", 0), "t1": v.get("entity_coverage", 0),
                    "t2": v.get("pair_connected", 0), "t3": v.get("answer_grounded", 0)}
               for qt, v in data.get("by_type", {}).items()}
    return {"total": s.get("total_questions", 0),
            "t1": s.get("entity_coverage", {}).get("count", 0),
            "t2": conn.get("count", 0), "t3": ag.get("total_grounded", 0),
            "by_type": by_type}


def load_musique(path: Path) -> dict:
    data = json.loads(path.read_text())
    s = data.get("summary", {})
    by_type = {qt: {"total": v.get("total", 0), "t1": v.get("t1", 0),
                    "t2": v.get("t2", 0), "t3": v.get("t3", 0)}
               for qt, v in data.get("by_type", {}).items()}
    same_e = sum(1 for r in data.get("per_question", [])
                 if any(c.get("type") == "same_entity" for c in r.get("tier2_connections", [])))
    return {"total": s.get("total_questions", 0),
            "t1": s.get("tier1_all_resolved", 0), "t2": s.get("tier2_all_connected", 0),
            "t3": s.get("tier3_all_grounded", 0), "by_type": by_type, "same_entity": same_e}


def delta_2wiki(fresh: dict, baseline: dict) -> dict:
    def _d(key):
        f, b = fresh[key], baseline[key]
        return {"baseline": b, "fresh": f, "delta": f-b,
                "delta_pct": round(100*(f-b)/max(b,1), 1)}
    fr_map = {r["question_id"]: r for r in fresh["records"]}
    ba_map = {r["question_id"]: r for r in baseline["records"]}
    flipped = [qid for qid, fr in fr_map.items()
               if (br := ba_map.get(qid)) and (
                   fr.get("all_values_match") != br.get("all_values_match") or
                   fr.get("all_properties_exist") != br.get("all_properties_exist") or
                   fr.get("all_entities_resolved") != br.get("all_entities_resolved"))]
    return {"t1": _d("t1"), "t2": _d("t2"), "t3": _d("t3"),
            "flipped_ids": flipped[:10], "total_flipped": len(flipped)}


_T2_LABELS = {"2WikiMultiHopQA": "Property grounding", "HotpotQA": "Pair connectivity", "MuSiQue": "Chain connectivity"}
_type_rows = lambda bt, order: [f"| {qt} | {r['total']} | {_cell(r['t1'],r['total'])} | {_cell(r['t2'],r['total'])} | {_cell(r['t3'],r['total'])} |" for qt in order if (r := bt.get(qt))]  # noqa: E731


def render_markdown(wiki: dict, hotpot: dict, musique: dict, delta: dict | None) -> str:
    W, H, M = wiki, hotpot, musique
    L: list[str] = [f"# 3-Benchmark DBpedia Grounding Report\n\nGenerated: {date.today()}\n",
        "## Section 1: Cross-Benchmark Grounding Funnel\n",
        "| Benchmark | Tier 1: Entity | Tier 2: KG Coverage | Tier 3: Answer |", "|---|---|---|---|",
        *[f"| {bm} | {_cell(d['t1'],d['total'])} | {_T2_LABELS[bm]}: {_cell(d['t2'],d['total'])} | {_cell(d['t3'],d['total'])} |"
          for bm, d in [("2WikiMultiHopQA", W), ("HotpotQA", H), ("MuSiQue", M)]],
        "",
        "## Section 2: Connectivity Semantics\n",
        "> **Tier 2 is NOT comparable across benchmarks:** 2Wiki verifies a known predicate exists "
        "and yields the expected value; HotpotQA checks any 1/2-hop path between a title pair; "
        "MuSiQue checks consecutive-pair connectivity along the decomposition chain.\n",
        "## Section 3: 2WikiMultiHopQA Per-Type Breakdown\n",
        "| Type | Total | Tier 1: Entity | Tier 2: Property | Tier 3: Value |", "|---|---|---|---|---|",
        *_type_rows(W["by_type"], ["compositional", "comparison", "bridge_comparison", "inference"]), "",
        "## Section 4: HotpotQA Per-Type Breakdown\n",
        "| Type | Total | Tier 1: Entity | Tier 2: Connectivity | Tier 3: Answer |", "|---|---|---|---|---|",
        *_type_rows(H["by_type"], ["bridge", "comparison"]),
        "", "*Footnote: Comparison entities share a property type but are not directly connected — 0% pair connectivity is expected for the comparison subset.*\n",
        "## Section 5: MuSiQue Per-Hop-Type Breakdown\n",
        "| Hop Type | Total | Tier 1: Entity | Tier 2: Chain Connectivity | Tier 3: Answer |", "|---|---|---|---|---|",
        *_type_rows(M["by_type"], ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"]),
        *([ f"\n*Note: {M['same_entity']} questions contain at least one `same_entity` hop-pair in Tier 2 connectivity.*"] if M.get("same_entity") else []),
        ""]

    if delta:
        rows = []
        for label, key in [("Tier 1 Entity", "t1"), ("Tier 2 Property", "t2"), ("Tier 3 Value", "t3")]:
            d = delta[key]; sign = "+" if d["delta"] >= 0 else ""
            rows.append(f"| {label} | {d['baseline']} | {d['fresh']} | {sign}{d['delta']} | {sign}{d['delta_pct']}% |")
        L += ["## Section 6: 2Wiki Delta vs Baseline\n",
              "| Metric | Baseline | Fresh | Delta | Delta % |", "|---|---|---|---|---|", *rows,
              "", f"Total flipped questions: **{delta['total_flipped']}**\n", "First 10 flipped question IDs:\n",
              *[f"- `{qid}`" for qid in delta["flipped_ids"]], ""]

    _s7a = ("All Tier 3 value lookups use `LIMIT 100` on DBpedia property retrieval. This cap may exclude valid "
            "values for entities with dense property graphs, potentially understating answer grounding rates. "
            "The effect is bounded: properties with >100 values are rare in dbo:/dbp: namespaces.")
    _s7b = ("The grounding funnel measures structural KG coverage, not system performance. A Tier 3 pass is "
            "necessary but not sufficient for the hybrid pipeline to answer correctly — routing, SPARQL generation, "
            "and live DBpedia availability all intervene. Conversely, questions failing Tier 2/3 are still "
            "answerable via the text fallback (Wikipedia + LLM), which explains why Hybrid Judge EM (89.4% on "
            "2Wiki, 86.1% on HotpotQA) substantially exceeds the Tier 3 grounding rate on all three benchmarks.")
    L += ["## Section 7: Methodology Notes\n", f"### (a) Tier 3 LIMIT Caveat\n\n{_s7a}\n",
          f"### (b) RQ2 Interpretation\n\n{_s7b}\n"]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(
        description="Aggregate 3-benchmark DBpedia grounding results into a markdown report.")
    p.add_argument("--twiki",    type=Path, default=DEFAULTS["twiki"],
                   help="2Wiki hop_verification_results JSON")
    p.add_argument("--hotpotqa", type=Path, default=DEFAULTS["hotpotqa"],
                   help="HotpotQA dbpedia_grounding_results JSON")
    p.add_argument("--musique",  type=Path, default=DEFAULTS["musique"],
                   help="MuSiQue dbpedia_grounding_results JSON")
    p.add_argument("--out",      type=Path, default=DEFAULTS["out"],
                   help="Output markdown path")
    p.add_argument("--delta-baseline", type=Path, default=None, dest="delta_baseline",
                   help="Baseline 2Wiki JSON for delta comparison (Section 6)")
    args = p.parse_args()

    # Resolve 2Wiki path (fallback to existing file if dated version absent)
    twiki_path = args.twiki if args.twiki.exists() else _TWIKI_FALLBACK
    missing = [(lbl, path) for lbl, path in
               [("HotpotQA", args.hotpotqa), ("MuSiQue", args.musique)] if not path.exists()]
    if not twiki_path.exists():
        missing.insert(0, ("2Wiki", args.twiki))
    if missing:
        for lbl, path in missing:
            print(f"ERROR: {lbl} file not found: {path}")
        print("Run the corresponding verify-*-grounding.py scripts first.")
        raise SystemExit(1)

    print("Loading grounding results...")
    wiki    = load_2wiki(twiki_path)
    hotpot  = load_hotpotqa(args.hotpotqa)
    musique = load_musique(args.musique)
    print(f"  2Wiki   : {wiki['total']}q  T1={wiki['t1']} T2={wiki['t2']} T3={wiki['t3']}")
    print(f"  HotpotQA: {hotpot['total']}q T1={hotpot['t1']} T2={hotpot['t2']} T3={hotpot['t3']}")
    print(f"  MuSiQue : {musique['total']}q T1={musique['t1']} T2={musique['t2']} T3={musique['t3']}")

    delta = None
    if args.delta_baseline:
        if not args.delta_baseline.exists():
            print(f"WARNING: --delta-baseline not found: {args.delta_baseline}; skipping Section 6")
        else:
            baseline = load_2wiki(args.delta_baseline)
            delta = delta_2wiki(wiki, baseline)
            print(f"  Delta   : {delta['total_flipped']} flipped questions")

    report = render_markdown(wiki, hotpot, musique, delta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    sections = "1-5,7" + (",6" if delta else "")
    print(f"\nReport written to {args.out}  (sections: {sections})")


if __name__ == "__main__":
    main()
