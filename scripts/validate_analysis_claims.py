"""
Validate every numerical claim in hybrid-vs-logicrag-performance-analysis.md
against the actual JSONL data.

Usage: python3 scripts/validate_analysis_claims.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
HYBRID_FILE = ROOT / "data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl"
LOGICRAG_FILE = ROOT / "data/eval_1000q/logicrag/logicrag_1000q_traced_judged.jsonl"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


hybrid = load_jsonl(HYBRID_FILE)
logicrag = load_jsonl(LOGICRAG_FILE)

# Build lookup by question_id
h_by_id = {r["question_id"]: r for r in hybrid}
l_by_id = {r["question_id"]: r for r in logicrag}

# Align: keep only shared IDs in original order
shared_ids = [r["question_id"] for r in hybrid if r["question_id"] in l_by_id]
print(f"Total records — Hybrid: {len(hybrid)}, LogicRAG: {len(logicrag)}, Shared IDs: {len(shared_ids)}\n")

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(claim_id, description, actual, expected, note=""):
    ok = actual == expected
    status = PASS if ok else FAIL
    results.append((claim_id, status, description, expected, actual, note))
    marker = "✓" if ok else "✗"
    print(f"  [{marker}] {claim_id}: {description}")
    if not ok:
        print(f"       Expected: {expected}  |  Actual: {actual}  {note}")
    elif note:
        print(f"       Note: {note}")


def check_approx(claim_id, description, actual, expected, tol=0.1, note=""):
    ok = abs(actual - expected) <= tol
    status = PASS if ok else FAIL
    results.append((claim_id, status, description, expected, actual, note))
    marker = "✓" if ok else "✗"
    print(f"  [{marker}] {claim_id}: {description}")
    if not ok:
        print(f"       Expected: {expected:.1f}%  |  Actual: {actual:.1f}%  {note}")
    elif note:
        print(f"       Note: {note}")


# ---------------------------------------------------------------------------
# Helper: correctness
# ---------------------------------------------------------------------------
def is_correct(record):
    """Prefer judge_correct if present, else em."""
    jc = record.get("judge_correct")
    if jc is not None:
        return bool(jc)
    return bool(record.get("em", 0))


def route_source(record):
    rs = record.get("route_source", "")
    method = record.get("method", "")
    return rs, method


# ===========================================================================
# SECTION 2 — Compositional questions
# ===========================================================================
print("=" * 70)
print("SECTION 2 — Compositional questions")
print("=" * 70)

comp_ids = [qid for qid in shared_ids if h_by_id[qid]["question_type"] == "compositional"]
n_comp = len(comp_ids)
check("S2-count", "# compositional questions in first 100", n_comp, 46)

h_comp_correct = [qid for qid in comp_ids if is_correct(h_by_id[qid])]
l_comp_correct = [qid for qid in comp_ids if is_correct(l_by_id[qid])]

h_comp_acc = 100 * len(h_comp_correct) / n_comp if n_comp else 0
l_comp_acc = 100 * len(l_comp_correct) / n_comp if n_comp else 0

check_approx("S2-hybrid-acc", "Hybrid compositional accuracy", h_comp_acc, 87.0, tol=0.15)
check_approx("S2-logicrag-acc", "LogicRAG compositional accuracy", l_comp_acc, 69.6, tol=0.15)

# Hybrid-only wins (compositional)
hybrid_only_wins = [
    qid for qid in comp_ids
    if is_correct(h_by_id[qid]) and not is_correct(l_by_id[qid])
]
n_how = len(hybrid_only_wins)
check("S2-hybrid-only-wins", "# hybrid-only wins (compositional)", n_how, 13)

# LogicRAG returned Unknown in 10/13
unknown_count = sum(
    1 for qid in hybrid_only_wins
    if "unknown" in l_by_id[qid].get("predicted_answer", "").lower()
)
check("S2-unknown-in-hybrid-wins", "LogicRAG returned Unknown in hybrid-only wins", unknown_count, 10)

# Breakdown: 7 sparql + 6 logicrag_decomposed
sparql_wins = [
    qid for qid in hybrid_only_wins
    if h_by_id[qid].get("method", "") == "sparql"
]
decomp_wins = [
    qid for qid in hybrid_only_wins
    if h_by_id[qid].get("method", "") == "logicrag_decomposed"
]
check("S2-sparql-wins", "Hybrid-only wins via SPARQL method", len(sparql_wins), 7)
check("S2-decomp-wins", "Hybrid-only wins via logicrag_decomposed method", len(decomp_wins), 6)

print("\n  [INFO] Hybrid-only win method breakdown:")
method_counts = defaultdict(int)
for qid in hybrid_only_wins:
    method_counts[h_by_id[qid].get("method", "unknown")] += 1
for m, c in sorted(method_counts.items()):
    print(f"         {m}: {c}")

print("\n  [INFO] LogicRAG answers in hybrid-only wins:")
for qid in hybrid_only_wins:
    ans = l_by_id[qid].get("predicted_answer", "")
    q = h_by_id[qid]["question"][:60]
    print(f"         [{qid[:8]}] '{ans[:40]}' | q: {q}")


# ===========================================================================
# SECTION 3 — Bridge comparison
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 3 — Bridge comparison")
print("=" * 70)

bridge_ids = [qid for qid in shared_ids if h_by_id[qid]["question_type"] == "bridge_comparison"]
n_bridge = len(bridge_ids)
check("S3-count", "# bridge_comparison questions in first 100", n_bridge, 19)

h_bridge_acc = 100 * sum(1 for qid in bridge_ids if is_correct(h_by_id[qid])) / n_bridge if n_bridge else 0
l_bridge_acc = 100 * sum(1 for qid in bridge_ids if is_correct(l_by_id[qid])) / n_bridge if n_bridge else 0

check_approx("S3-hybrid-acc", "Hybrid bridge_comparison accuracy", h_bridge_acc, 94.7, tol=0.15)
check_approx("S3-logicrag-acc", "LogicRAG bridge_comparison accuracy", l_bridge_acc, 73.7, tol=0.15)


# ===========================================================================
# SECTION 4 — Inference
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 4 — Inference")
print("=" * 70)

inf_ids = [qid for qid in shared_ids if h_by_id[qid]["question_type"] == "inference"]
n_inf = len(inf_ids)
check("S4-count", "# inference questions in first 100", n_inf, 15)

h_inf_acc = 100 * sum(1 for qid in inf_ids if is_correct(h_by_id[qid])) / n_inf if n_inf else 0
l_inf_acc = 100 * sum(1 for qid in inf_ids if is_correct(l_by_id[qid])) / n_inf if n_inf else 0

check_approx("S4-hybrid-acc", "Hybrid inference accuracy", h_inf_acc, 80.0, tol=0.15)
check_approx("S4-logicrag-acc", "LogicRAG inference accuracy", l_inf_acc, 100.0, tol=0.15)

# LogicRAG-only wins (inference) — claimed 4
logicrag_only_inf = [
    qid for qid in inf_ids
    if is_correct(l_by_id[qid]) and not is_correct(h_by_id[qid])
]
check("S4-logicrag-only-wins", "# LogicRAG-only wins (inference)", len(logicrag_only_inf), 4)

print("\n  [INFO] LogicRAG-only inference wins:")
for qid in logicrag_only_inf:
    q = h_by_id[qid]["question"]
    ha = h_by_id[qid].get("predicted_answer", "")
    la = l_by_id[qid].get("predicted_answer", "")
    print(f"         [{qid[:8]}] Q: {q[:70]}")
    print(f"                   H: '{ha[:40]}'  L: '{la[:40]}'")


# ===========================================================================
# SECTION 5 — Comparison tie
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 5 — Comparison tie")
print("=" * 70)

cmp_ids = [qid for qid in shared_ids if h_by_id[qid]["question_type"] == "comparison"]
n_cmp = len(cmp_ids)
check("S5-count", "# comparison questions in first 100", n_cmp, 20)

h_cmp_acc = 100 * sum(1 for qid in cmp_ids if is_correct(h_by_id[qid])) / n_cmp if n_cmp else 0
l_cmp_acc = 100 * sum(1 for qid in cmp_ids if is_correct(l_by_id[qid])) / n_cmp if n_cmp else 0

check_approx("S5-hybrid-acc", "Hybrid comparison accuracy", h_cmp_acc, 95.0, tol=0.15)
check_approx("S5-logicrag-acc", "LogicRAG comparison accuracy", l_cmp_acc, 95.0, tol=0.15)


# ===========================================================================
# SECTION 6 — Neither-correct
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 6 — Neither-correct")
print("=" * 70)

neither_ids = [
    qid for qid in shared_ids
    if not is_correct(h_by_id[qid]) and not is_correct(l_by_id[qid])
]
n_neither = len(neither_ids)
check("S6-count", "# neither-correct questions", n_neither, 7)

neither_types = defaultdict(int)
for qid in neither_ids:
    neither_types[h_by_id[qid]["question_type"]] += 1
check("S6-comp", "neither-correct compositional", neither_types.get("compositional", 0), 5)
check("S6-cmp", "neither-correct comparison", neither_types.get("comparison", 0), 1)
check("S6-bridge", "neither-correct bridge_comparison", neither_types.get("bridge_comparison", 0), 1)

print(f"\n  [INFO] neither-correct type distribution: {dict(neither_types)}")

# 6 of 7 have is_answerable: false
unanswerable_neither = sum(
    1 for qid in neither_ids if not h_by_id[qid].get("is_answerable", True)
)
check("S6-unanswerable", "6 of 7 neither-correct have is_answerable=false", unanswerable_neither, 6)

print("\n  [INFO] neither-correct details:")
for qid in neither_ids:
    q = h_by_id[qid]["question"][:60]
    qt = h_by_id[qid]["question_type"]
    ia = h_by_id[qid].get("is_answerable")
    ha = h_by_id[qid].get("predicted_answer", "")[:30]
    la = l_by_id[qid].get("predicted_answer", "")[:30]
    print(f"         [{qid[:8]}] {qt} | answerable={ia} | H:'{ha}' L:'{la}' | Q:{q}")


# ===========================================================================
# SECTION 7 — LogicRAG retrieval degradation (rounds vs accuracy)
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 7 — LogicRAG retrieval degradation by rounds")
print("=" * 70)

# Claimed: 0→39/36/92.3%, 1→16/15/93.8%, 2→34/23/67.6%, 3→11/6/54.5%
expected_rounds = {
    0: (39, 36, 92.3),
    1: (16, 15, 93.8),
    2: (34, 23, 67.6),
    3: (11, 6, 54.5),
}

rounds_data = defaultdict(list)
for r in logicrag:
    rounds_data[r.get("rounds", -1)].append(r)

print(f"\n  [INFO] Rounds distribution in logicrag: { {k: len(v) for k, v in sorted(rounds_data.items())} }")

for n_rounds, (exp_n, exp_correct, exp_acc) in sorted(expected_rounds.items()):
    recs = rounds_data.get(n_rounds, [])
    actual_n = len(recs)
    actual_correct = sum(1 for r in recs if is_correct(r))
    actual_acc = 100 * actual_correct / actual_n if actual_n else 0
    check("S7-r{n}-n".format(n=n_rounds), f"rounds={n_rounds}: count={exp_n}", actual_n, exp_n)
    check("S7-r{n}-correct".format(n=n_rounds), f"rounds={n_rounds}: correct={exp_correct}", actual_correct, exp_correct)
    check_approx("S7-r{n}-acc".format(n=n_rounds), f"rounds={n_rounds}: accuracy={exp_acc}%", actual_acc, exp_acc, tol=0.2)

# Also verify totals add up
total_in_table = sum(v[0] for v in expected_rounds.values())
check("S7-total", "rounds table totals to 100", total_in_table, 100)


# ===========================================================================
# SECTION 1 — Corpus sizes
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 1 — Corpus sizes")
print("=" * 70)

# 6,120 unique pages for 1,000 questions — check from trace data
# The logicrag file only has 100 records — we can only count unique pages seen in traces
unique_pages = set()
total_chunks = 0
for r in logicrag:
    trace = r.get("trace", {})
    # warmup chunks
    warmup = trace.get("warmup", {})
    for chunk in warmup.get("retrieved_chunks", []):
        if isinstance(chunk, str):
            title = chunk.split("Title:")[1].split(".")[0].strip() if "Title:" in chunk else chunk[:50]
        elif isinstance(chunk, dict):
            title = chunk.get("title", "")
        else:
            title = str(chunk)[:50]
        unique_pages.add(title)
        total_chunks += 1
    # round chunks
    for rnd in trace.get("rounds", []):
        for chunk in rnd.get("retrieved_chunks", []):
            if isinstance(chunk, str):
                title = chunk.split("Title:")[1].split(".")[0].strip() if "Title:" in chunk else chunk[:50]
            elif isinstance(chunk, dict):
                title = chunk.get("title", "")
            else:
                title = str(chunk)[:50]
            unique_pages.add(title)
            total_chunks += 1

print(f"  [INFO] Unique page titles seen in 100-q logicrag traces: {len(unique_pages)}")
print(f"  [INFO] Total chunk references in traces: {total_chunks}")
print("  [NOTE] Claim of 6,119 snippets / 6,120 unique pages refers to ALL 1000 questions —")
print("         cannot verify from 100-q subset. Skipping direct check.")

# Check if there's a corpus manifest or separate file
corpus_paths = [
    ROOT / "data/eval_1000q/logicrag_corpus.json",
    ROOT / "data/eval_1000q/logicrag_corpus.jsonl",
    ROOT / "data" / "logicrag_corpus.json",
]
for p in corpus_paths:
    if p.exists():
        print(f"  [INFO] Found corpus file: {p}")


# ===========================================================================
# SECTION 9 — Philippe/SPARQL example (if available)
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 9 — Example: Philippe / SPARQL")
print("=" * 70)

# Search for Philippe question in hybrid data
philippe_qs = [r for r in hybrid if "philippe" in r.get("question", "").lower() or "philippe" in r.get("predicted_answer", "").lower()]
print(f"  [INFO] Questions mentioning 'philippe': {len(philippe_qs)}")
for r in philippe_qs:
    print(f"         [{r['question_id'][:8]}] {r['question'][:80]}")
    print(f"           method={r.get('method')} route={r.get('route_source','')[:50]}")
    q = r.get("sparql_query", "")
    if q:
        print(f"           SPARQL (first 200): {q[:200]}")


# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

passed = [r for r in results if r[1] == PASS]
failed = [r for r in results if r[1] == FAIL]

print(f"\n  PASSED: {len(passed)}/{len(results)}")
print(f"  FAILED: {len(failed)}/{len(results)}\n")

if failed:
    print("  FAILED CHECKS:")
    for claim_id, status, desc, exp, act, note in failed:
        print(f"    [{claim_id}] {desc}")
        print(f"      Expected: {exp}  |  Actual: {act}")
        if note:
            print(f"      Note: {note}")

print()
