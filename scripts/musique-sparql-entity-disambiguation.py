#!/usr/bin/env python3
"""Phase 4: SPARQL entity disambiguation audit for MuSiQue.

For each of the ~452 SPARQL sub-questions, extracts entity URIs from the SPARQL
query and sparql_gen_prompt, compares against GT supporting paragraph titles.
Classifies: correct_entity, wrong_entity, entity_not_found, query_error, uri_mismatch.

Output: data/cross_benchmark/musique/ablation/sparql_entity_disambiguation.json
"""
import importlib.util
import re
import urllib.parse
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

_spec = importlib.util.spec_from_file_location(
    "mu", Path(__file__).resolve().parent / "musique-analysis-utils.py"
)
_mu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mu)

load_hybrid = _mu.load_hybrid
load_musique_gt = _mu.load_musique_gt
normalize_answer = _mu.normalize_answer
save_json = _mu.save_json


def extract_uris_from_sparql(query: str) -> list[str]:
    """Extract DBpedia resource URIs from a SPARQL query string."""
    if not query:
        return []
    full = re.findall(r'<(http://dbpedia\.org/resource/[^>]+)>', query)
    # Strip trailing SPARQL punctuation but preserve valid URI chars like ')'
    prefix = [re.sub(r'[.;,}]+$', '', p) for p in re.findall(r'dbr:(\S+)', query)]
    combined = full + [f"http://dbpedia.org/resource/{u}" for u in prefix]
    seen = set()
    return [u for u in combined if u not in seen and not seen.add(u)]


def extract_uris_from_prompt(prompt: str) -> list[tuple[str, str]]:
    """Extract (entity_name, URI) pairs from sparql_gen_prompt."""
    if not prompt:
        return []
    pairs = []
    for m in re.finditer(r'Entity:\s*(.+?)\n.*?URI:\s*<?([^>\n]+)>?', prompt, re.DOTALL):
        pairs.append((m.group(1).strip(), m.group(2).strip()))
    return pairs


def title_to_uri(title: str) -> str:
    """Convert Wikipedia/MuSiQue paragraph title to expected DBpedia URI."""
    encoded = title.replace(" ", "_")
    return f"http://dbpedia.org/resource/{urllib.parse.quote(encoded, safe='_,()·:')}"


def uri_basename(uri: str) -> str:
    """Extract the resource name from a DBpedia URI, decoded and lowered."""
    name = uri.split("/resource/")[-1] if "/resource/" in uri else uri
    return urllib.parse.unquote(name).replace("_", " ").lower()


def classify_entity(prompt_uris: list[tuple[str, str]], gt_titles: list[str]) -> str:
    """Classify entity alignment between prompt URIs and GT titles."""
    if not prompt_uris:
        return "entity_not_found"
    gt_names = {t.lower() for t in gt_titles}
    gt_uris = {title_to_uri(t) for t in gt_titles}
    for _, uri in prompt_uris:
        if uri in gt_uris:
            return "correct_entity"
        if uri_basename(uri) in gt_names:
            return "correct_entity"
    # Check partial match (one contains the other)
    for _, uri in prompt_uris:
        base = uri_basename(uri)
        for gt_name in gt_names:
            if gt_name in base or base in gt_name:
                return "possible_match"
    return "wrong_entity"


def main():
    hybrid = load_hybrid()
    gt = load_musique_gt()

    sparql_subqs = []
    cat_counts = defaultdict(int)
    cat_by_type = defaultdict(lambda: defaultdict(int))
    disambiguation_errors = []

    for qid in sorted(hybrid):
        h = hybrid[qid]
        g = gt[qid]
        subqs = h.get("sub_question_log") or []
        gt_decomp = g["question_decomposition"]
        gt_paragraphs = g["paragraphs"]
        qtype = h.get("question_type", "unknown")

        for i, sq in enumerate(subqs):
            if sq.get("method") != "sparql":
                continue
            meta = sq.get("solver_meta") or {}
            query = meta.get("initial_query") or meta.get("retry_query") or ""
            prompt = meta.get("sparql_gen_prompt") or ""

            # Get GT supporting titles for this hop (positional alignment)
            gt_titles = []
            if i < len(gt_decomp):
                para_idx = gt_decomp[i].get("paragraph_support_idx")
                if para_idx is not None and para_idx < len(gt_paragraphs):
                    gt_titles.append(gt_paragraphs[para_idx]["title"])

            prompt_uris = extract_uris_from_prompt(prompt)
            query_uris = extract_uris_from_sparql(query)

            # Determine category
            has_error = bool(meta.get("initial_error")) and bool(meta.get("retry_error"))
            no_results = (meta.get("initial_result_count", 0) == 0
                          and meta.get("retry_result_count", 0) == 0)

            if has_error and not query:
                category = "query_error"
            elif not prompt_uris:
                category = "entity_not_found"
            else:
                category = classify_entity(prompt_uris, gt_titles)
                # Check if LLM used different URI than provided
                if category == "correct_entity" and query_uris:
                    prompt_uri_set = {u for _, u in prompt_uris}
                    if not any(u in prompt_uri_set for u in query_uris):
                        category = "uri_mismatch"
                # Refine: correct entity but wrong answer
                if category == "correct_entity" and not no_results:
                    sys_ans = str(sq.get("answer", "") or "")
                    gt_ans = gt_decomp[i]["answer"] if i < len(gt_decomp) else ""
                    if gt_ans and normalize_answer(sys_ans) != normalize_answer(gt_ans):
                        category = "correct_but_wrong_answer"

            cat_counts[category] += 1
            cat_by_type[qtype][category] += 1

            rec = {
                "question_id": qid,
                "hop": i,
                "sub_question": sq.get("sub_question", ""),
                "category": category,
                "prompt_uris": [u for _, u in prompt_uris],
                "query_uris": query_uris,
                "gt_titles": gt_titles,
                "has_results": not no_results,
                "used_retry": meta.get("used_retry", False),
            }
            sparql_subqs.append(rec)

            if category in ("wrong_entity", "possible_match", "uri_mismatch"):
                disambiguation_errors.append(rec)

    # Correlation: classify each question by worst-case entity status to avoid double-counting
    SEVERITY = {"wrong_entity": 3, "possible_match": 2, "uri_mismatch": 2,
                "entity_not_found": 1, "query_error": 1, "correct_but_wrong_answer": 0,
                "correct_entity": 0}
    qid_worst = {}
    for r in sparql_subqs:
        qid, cat = r["question_id"], r["category"]
        if qid not in qid_worst or SEVERITY.get(cat, 0) > SEVERITY.get(qid_worst[qid], 0):
            qid_worst[qid] = cat
    wrong_entity_qids = {q for q, c in qid_worst.items() if c in ("wrong_entity", "possible_match")}
    correct_entity_qids = {q for q, c in qid_worst.items()
                           if c in ("correct_entity", "correct_but_wrong_answer")}
    wrong_ent_fail = sum(1 for qid in wrong_entity_qids if not hybrid[qid].get("judge_correct"))
    correct_ent_fail = sum(1 for qid in correct_entity_qids if not hybrid[qid].get("judge_correct"))

    def _rate(n, d):
        return round(n / max(d, 1), 3)

    # Fisher's exact test: entity correctness vs final question outcome
    # [[wrong_entity_fail, wrong_entity_correct], [correct_entity_fail, correct_entity_correct]]
    we_n = len(wrong_entity_qids)
    ce_n = len(correct_entity_qids)
    _a = wrong_ent_fail          # wrong entity, final fail
    _b = we_n - wrong_ent_fail   # wrong entity, final correct
    _c = correct_ent_fail        # correct entity, final fail
    _d = ce_n - correct_ent_fail  # correct entity, final correct
    _odds_ratio, _p_value = fisher_exact([[_a, _b], [_c, _d]])

    output = {
        "summary": {"total_sparql_subqs": len(sparql_subqs),
                     "category_counts": dict(sorted(cat_counts.items(), key=lambda x: -x[1]))},
        "by_question_type": {t: dict(cat_by_type[t]) for t in sorted(cat_by_type)},
        "correlation": {
            "wrong_entity_questions": we_n,
            "wrong_entity_final_fail": wrong_ent_fail,
            "wrong_entity_fail_rate": _rate(wrong_ent_fail, we_n),
            "correct_entity_questions": ce_n,
            "correct_entity_final_fail": correct_ent_fail,
            "correct_entity_fail_rate": _rate(correct_ent_fail, ce_n),
            "fisher_p": round(_p_value, 6),
            "odds_ratio": round(_odds_ratio, 4),
        },
        "disambiguation_errors": disambiguation_errors[:50],
        "per_subq": sparql_subqs,
    }

    out = save_json(output, "sparql_entity_disambiguation.json")

    # ── Console summary ─────────────────────────────────────────────────────
    print(f"SPARQL ENTITY DISAMBIGUATION AUDIT (MuSiQue)\n{'='*50}")
    print(f"Total SPARQL sub-Qs: {len(sparql_subqs)}\n")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<28} {n:>4} ({100*n/max(len(sparql_subqs),1):.1f}%)")
    cor = output["correlation"]
    print(f"\nWrong entity → fail: {cor['wrong_entity_fail_rate']:.1%} | "
          f"Correct entity → fail: {cor['correct_entity_fail_rate']:.1%} | "
          f"OR={cor['odds_ratio']:.4f} p={cor['fisher_p']:.6f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
