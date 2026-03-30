"""
Factcheck: validate claims in docs/hybrid-vs-logicrag-100q-comparison.md
against actual JSONL data (first 100 lines of each file).
"""

import json
import statistics
from collections import defaultdict

HYBRID_PATH = "/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa/data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl"
LOGICRAG_PATH = "/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa/data/eval_1000q/logicrag/logicrag_1000q_traced_judged.jsonl"


def load_jsonl(path, limit=100):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            rows.append(json.loads(line.strip()))
    return rows


def pct(num, den):
    return round(100 * num / den, 1) if den else None


errors = []

def check(label, stated, actual, note=""):
    ok = stated == actual
    marker = "OK" if ok else "XX"
    suffix = f"  [{note}]" if note else ""
    print(f"  {marker}  {label}: stated={stated}, actual={actual}{suffix}")
    if not ok:
        errors.append((label, stated, actual))
    return ok


# ── Load ──────────────────────────────────────────────────────────────────────
hybrid = load_jsonl(HYBRID_PATH)
logicrag = load_jsonl(LOGICRAG_PATH)

h_by_id = {r["question_id"]: r for r in hybrid}
l_by_id = {r["question_id"]: r for r in logicrag}

common_ids = sorted(set(h_by_id) & set(l_by_id))
n = len(common_ids)
print(f"\nLoaded: hybrid={len(hybrid)}, logicrag={len(logicrag)}, common={n}")
if n != 100:
    print(f"  WARNING: expected 100 common questions, got {n}")

h = [h_by_id[i] for i in common_ids]
l = [l_by_id[i] for i in common_ids]

# ── 1. Overall Results ────────────────────────────────────────────────────────
print("\n=== 1. Overall Results ===")

h_judge = sum(int(r["judge_correct"]) for r in h)
l_judge = sum(int(r["judge_correct"]) for r in l)
h_em = sum(r["em"] for r in h)
l_em = sum(r["em"] for r in l)
h_f1 = round(100 * sum(r["f1"] for r in h) / n, 1)
l_f1 = round(100 * sum(r["f1"] for r in l) / n, 1)

h_latencies = [r["latency_seconds"] for r in h if r.get("latency_seconds") is not None]
l_latencies = [r["latency_seconds"] for r in l if r.get("latency_seconds") is not None]

h_mean_lat = round(statistics.mean(h_latencies), 1) if h_latencies else None
l_mean_lat = round(statistics.mean(l_latencies), 1) if l_latencies else None
h_med_lat = round(statistics.median(h_latencies), 1) if h_latencies else None
l_med_lat = round(statistics.median(l_latencies), 1) if l_latencies else None

print(f"  [raw] Hybrid: judge={h_judge}, em={h_em}, f1={h_f1}%, mean_lat={h_mean_lat}s, med_lat={h_med_lat}s")
print(f"  [raw] LogicRAG: judge={l_judge}, em={l_em}, f1={l_f1}%, mean_lat={l_mean_lat}s, med_lat={l_med_lat}s")

check("Hybrid Judge count", 89, h_judge)
check("LogicRAG Judge count", 80, l_judge)
check("Hybrid EM count", 67, h_em)
check("LogicRAG EM count", 62, l_em)
check("Hybrid F1 %", 78.6, h_f1)
check("LogicRAG F1 %", 69.8, l_f1)
check("Hybrid mean latency (s)", 7.2, h_mean_lat)
check("LogicRAG mean latency (s)", 13.9, l_mean_lat)
check("Hybrid median latency (s)", 6.5, h_med_lat)
check("LogicRAG median latency (s)", 3.0, l_med_lat)

# ── 2. Win/Loss ───────────────────────────────────────────────────────────────
print("\n=== 2. Win/Loss Breakdown ===")

both_correct = sum(1 for hh, ll in zip(h, l) if hh["judge_correct"] and ll["judge_correct"])
hybrid_only = sum(1 for hh, ll in zip(h, l) if hh["judge_correct"] and not ll["judge_correct"])
logicrag_only = sum(1 for hh, ll in zip(h, l) if not hh["judge_correct"] and ll["judge_correct"])
neither = sum(1 for hh, ll in zip(h, l) if not hh["judge_correct"] and not ll["judge_correct"])
print(f"  [raw] both={both_correct}, hybrid_only={hybrid_only}, logicrag_only={logicrag_only}, neither={neither}, total={both_correct+hybrid_only+logicrag_only+neither}")

check("Both correct", 76, both_correct)
check("Hybrid-only correct", 13, hybrid_only)
check("LogicRAG-only correct", 4, logicrag_only)
check("Neither correct", 7, neither)

# ── 3. By Question Type ───────────────────────────────────────────────────────
print("\n=== 3. By Question Type ===")

by_type = defaultdict(list)
for hh, ll in zip(h, l):
    by_type[hh["question_type"]].append((hh, ll))

# Print actual type counts
for qtype, pairs in sorted(by_type.items()):
    print(f"  [raw] {qtype}: n={len(pairs)}")

types_stated = {
    "comparison":        {"n": 20, "h_judge": 95.0, "l_judge": 95.0, "h_em": 95.0, "l_em": 95.0},
    "bridge_comparison": {"n": 19, "h_judge": 94.7, "l_judge": 73.7, "h_em": 94.7, "l_em": 68.4},
    "inference":         {"n": 15, "h_judge": 80.0, "l_judge": 100.0, "h_em": 60.0, "l_em": 73.3},
    "compositional":     {"n": 46, "h_judge": 87.0, "l_judge": 69.6, "h_em": 45.7, "l_em": 41.3},
}

for qtype, stated in types_stated.items():
    pairs = by_type.get(qtype, [])
    nt = len(pairs)
    h_j = round(100 * sum(hh["judge_correct"] for hh, _ in pairs) / nt, 1) if nt else None
    l_j = round(100 * sum(ll["judge_correct"] for _, ll in pairs) / nt, 1) if nt else None
    h_e = round(100 * sum(hh["em"] for hh, _ in pairs) / nt, 1) if nt else None
    l_e = round(100 * sum(ll["em"] for _, ll in pairs) / nt, 1) if nt else None
    print(f"\n  [{qtype}] actual: n={nt}, h_judge={h_j}%, l_judge={l_j}%, h_em={h_e}%, l_em={l_e}%")
    check(f"{qtype} n", stated["n"], nt)
    check(f"{qtype} Hybrid Judge%", stated["h_judge"], h_j)
    check(f"{qtype} LogicRAG Judge%", stated["l_judge"], l_j)
    check(f"{qtype} Hybrid EM%", stated["h_em"], h_e)
    check(f"{qtype} LogicRAG EM%", stated["l_em"], l_e)

# ── 4. By KG Answerability ────────────────────────────────────────────────────
print("\n=== 4. By KG Answerability (is_answerable from hybrid JSONL) ===")

answerable_pairs = [(hh, ll) for hh, ll in zip(h, l) if hh.get("is_answerable")]
unanswerable_pairs = [(hh, ll) for hh, ll in zip(h, l) if not hh.get("is_answerable")]
na = len(answerable_pairs)
nu = len(unanswerable_pairs)

h_ans_j = sum(hh["judge_correct"] for hh, _ in answerable_pairs)
l_ans_j = sum(ll["judge_correct"] for _, ll in answerable_pairs)
h_unans_j = sum(hh["judge_correct"] for hh, _ in unanswerable_pairs)
l_unans_j = sum(ll["judge_correct"] for _, ll in unanswerable_pairs)

print(f"  [raw] answerable={na}, unanswerable={nu}")
print(f"  [raw] KG-ans: h_judge={h_ans_j}/{na}={pct(h_ans_j,na)}%, l_judge={l_ans_j}/{na}={pct(l_ans_j,na)}%")
print(f"  [raw] KG-unans: h_judge={h_unans_j}/{nu}={pct(h_unans_j,nu)}%, l_judge={l_unans_j}/{nu}={pct(l_unans_j,nu)}%")

check("KG-answerable n", 36, na)
check("KG-unanswerable n", 64, nu)
check("KG-answerable Hybrid correct count", 35, h_ans_j)
check("KG-answerable LogicRAG correct count", 30, l_ans_j)
check("KG-answerable Hybrid %", 97.2, pct(h_ans_j, na))
check("KG-answerable LogicRAG %", 83.3, pct(l_ans_j, na))
check("KG-unanswerable Hybrid correct count", 54, h_unans_j)
check("KG-unanswerable LogicRAG correct count", 50, l_unans_j)
check("KG-unanswerable Hybrid %", 84.4, pct(h_unans_j, nu))
check("KG-unanswerable LogicRAG %", 78.1, pct(l_unans_j, nu))

# ── 5. Failure Analysis ───────────────────────────────────────────────────────
print("\n=== 5. Failure Analysis: Hybrid-only wins ===")

hybrid_wins = [(hh, ll) for hh, ll in zip(h, l) if hh["judge_correct"] and not ll["judge_correct"]]
method_counts = defaultdict(int)
for hh, ll in hybrid_wins:
    method_counts[hh.get("method", "unknown")] += 1

print(f"  [raw] Hybrid-only wins total: {len(hybrid_wins)}")
for method, cnt in sorted(method_counts.items()):
    print(f"    method={method}: {cnt}")

sparql_wins = method_counts.get("sparql", 0)
logicrag_decomposed_wins = method_counts.get("logicrag_decomposed", 0)
check("sparql method hybrid-only wins", 7, sparql_wins)
check("logicrag_decomposed method hybrid-only wins", 6, logicrag_decomposed_wins)

# Count "Unknown" in LogicRAG answers for hybrid-only win cases
unknown_count = 0
for hh, ll in hybrid_wins:
    ans = (ll.get("predicted_answer") or "").strip().lower()
    raw = (ll.get("raw_answer") or "").strip().lower()
    if "unknown" in ans or "unknown" in raw:
        unknown_count += 1

check('LogicRAG "Unknown" in hybrid-only wins', 10, unknown_count)

print(f"\n  LogicRAG predicted_answer in hybrid-only wins:")
for hh, ll in hybrid_wins:
    ans = ll.get("predicted_answer", "")
    print(f"    [{hh.get('method')}]  logicrag_ans={repr(ans[:80])}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"ERRORS FOUND: {len(errors)}")
for label, stated, actual in errors:
    print(f"  XX  {label}: stated={stated}, actual={actual}")
print("Done.")
