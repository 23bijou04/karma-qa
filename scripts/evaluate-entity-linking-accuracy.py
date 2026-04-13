"""Evaluate entity linking precision/recall vs gold entities in coverage_analysis_1000.json.

Runs on all 1000 questions; samples 100 (seed=42) for detailed failure analysis.
Output: data/eval_1000q/ablation/entity_linking_accuracy.json

Gold URI correction: evidence subjects in 2WikiMultiHopQA use ambiguous names
(e.g. "Thomas Jefferson") while supporting_facts use disambiguated Wikipedia titles
(e.g. "Thomas Jefferson (film)"). We re-resolve gold URIs from supporting_facts
to fix ~126 disambiguation errors.

Matching: URI-strict with gold-side dedup. Label fallback only for gold entities
without URIs (~2.6% of gold).
"""

import json
import random
import re
from pathlib import Path

HYBRID_JSONL = Path("data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl")
COVERAGE_JSON = Path("data/eval_1000q/ground-truth/coverage_analysis_1000.json")
DEV_FULL_JSON = Path("data/dev_full.json")
OUT_PATH = Path("data/eval_1000q/ablation/entity_linking_accuracy.json")
SAMPLE_SEED = 42
SAMPLE_SIZE = 100


# ── normalisation helpers ──────────────────────────────────────────────────────

def norm_uri(uri: str) -> str:
    """Lowercase + strip trailing slash for URI comparison."""
    return uri.lower().rstrip("/") if uri else ""


def norm_label(text: str) -> str:
    """Lowercase + replace underscores with spaces for label comparison."""
    return text.lower().replace("_", " ").strip() if text else ""


def uri_slug(uri: str) -> str:
    """Extract last path segment of a DBpedia URI, then normalise as label."""
    if not uri:
        return ""
    slug = uri.rstrip("/").rsplit("/", 1)[-1]
    return norm_label(slug)


def strip_disambig(title: str) -> str:
    """Remove trailing parenthetical: 'Thomas Jefferson (film)' -> 'Thomas Jefferson'."""
    return re.sub(r'\s*\([^)]*\)\s*$', '', title).strip()


# ── gold URI correction from supporting_facts ─────────────────────────────────

def build_gold_uri_corrections(cov_data: dict, dev_data: dict) -> dict[str, dict[str, str]]:
    """Build a map: question_id -> {old_gold_name_lower: corrected_uri}.

    Uses supporting_facts disambiguated titles from dev_full.json to fix
    gold URIs that resolve to wrong entity due to ambiguous evidence names.
    """
    corrections: dict[str, dict[str, str]] = {}
    n_corrected = 0

    for q in cov_data["questions"]:
        qid = q["question_id"]
        orig = dev_data.get(qid)
        if not orig:
            continue

        sf_titles = [sf[0] for sf in orig.get("supporting_facts", [])]

        # Map base_name -> disambiguated SF titles (only those WITH disambiguation suffix)
        sf_by_base: dict[str, list[str]] = {}
        for sf_title in sf_titles:
            base = strip_disambig(sf_title)
            if base != sf_title:  # has a disambiguation suffix
                sf_by_base.setdefault(base.lower(), []).append(sf_title)

        q_corrections: dict[str, str] = {}
        for ent in q.get("entities", []):
            name = ent.get("name", "")
            gold_uri = ent.get("uri", "")
            if not name or not gold_uri:
                continue

            candidates = sf_by_base.get(name.lower(), [])
            if not candidates:
                continue

            sf_title = candidates[0]
            sf_slug = sf_title.replace(" ", "_")
            sf_uri = f"http://dbpedia.org/resource/{sf_slug}"

            if norm_uri(sf_uri) != norm_uri(gold_uri):
                q_corrections[name.lower()] = sf_uri
                n_corrected += 1

        if q_corrections:
            corrections[qid] = q_corrections

    print(f"Gold URI corrections from supporting_facts: {n_corrected} entities across {len(corrections)} questions")
    return corrections


def apply_gold_corrections(gold: list[dict], corrections: dict[str, str]) -> list[dict]:
    """Return gold list with corrected URIs where available."""
    if not corrections:
        return gold
    corrected = []
    for e in gold:
        name_lower = e.get("name", "").lower()
        if name_lower in corrections:
            e = dict(e)  # shallow copy to avoid mutating original
            e["uri"] = corrections[name_lower]
        corrected.append(e)
    return corrected


# ── per-question matching ──────────────────────────────────────────────────────

def match_entities(linked: list[dict], gold: list[dict]) -> dict:
    """
    Returns dict with precision, recall, tp, fp, fn counts.

    Matching: URI-strict with gold-side dedup.
      1. URI match (both sides normalised) — primary, used when gold has URI
      2. Label fallback: only for gold entities WITHOUT a URI
    Each gold entity is matched at most once (prevents TP inflation).
    """
    if not gold:
        return {"precision": None, "recall": None, "tp": 0, "fp": 0, "fn": 0,
                "full_coverage": True, "no_gold": True}

    # Index gold entities by URI and by label (label only for URI-less gold)
    gold_uri_to_idx: dict[str, int] = {}
    gold_label_to_idx: dict[str, int] = {}  # only gold without URI
    for i, e in enumerate(gold):
        u = norm_uri(e.get("uri", ""))
        if u:
            gold_uri_to_idx[u] = i
        else:
            # label fallback only for gold entities missing a URI
            n = norm_label(e.get("name", ""))
            if n:
                gold_label_to_idx[n] = i

    matched_gold_idxs: set[int] = set()
    tp = 0
    fp = 0

    for le in linked:
        lu = norm_uri(le.get("uri", ""))
        ll = norm_label(le.get("label", ""))
        slug = uri_slug(le.get("uri", ""))

        matched_idx = None
        # Priority 1: URI match (covers 97.4% of gold)
        if lu and lu in gold_uri_to_idx:
            matched_idx = gold_uri_to_idx[lu]
        # Priority 2: label/slug match ONLY for gold without URI
        elif ll and ll in gold_label_to_idx:
            matched_idx = gold_label_to_idx[ll]
        elif slug and slug in gold_label_to_idx:
            matched_idx = gold_label_to_idx[slug]

        if matched_idx is not None and matched_idx not in matched_gold_idxs:
            tp += 1
            matched_gold_idxs.add(matched_idx)
        else:
            fp += 1  # no match, or duplicate match to already-matched gold

    fn = len(gold) - len(matched_gold_idxs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None  # undefined when no links
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    full_coverage = fn == 0

    return {"precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "full_coverage": full_coverage,
            "no_gold": False}


# ── seed entity detection ─────────────────────────────────────────────────────

def is_seed_entity(entity_name: str, question: str) -> bool:
    """Check if a gold entity name appears in the question text (case-insensitive).

    Seed entities are directly mentioned in the question and thus reachable
    by the entity linker. Bridge entities (intermediate hops) are NOT mentioned
    and can only be discovered through KG traversal.
    """
    if not entity_name or not question:
        return False
    return entity_name.lower() in question.lower()


# ── failure categories ─────────────────────────────────────────────────────────

def classify_failure(linked: list[dict], gold: list[dict]) -> str | None:
    """High-level failure classification for recall < 1.0.

    Uses URI matching (consistent with match_entities) to identify missing gold.
    """
    if not linked:
        return "no_entities_linked"
    if not gold:
        return None
    linked_uris = {norm_uri(le.get("uri", "")) for le in linked}
    gold_missing = []
    for e in gold:
        gu = norm_uri(e.get("uri", ""))
        if gu and gu not in linked_uris:
            gold_missing.append(norm_label(e.get("name", "")))
        elif not gu:
            # URI-less gold: fall back to label check
            gl = norm_label(e.get("name", ""))
            linked_labels = {norm_label(le.get("label", "")) for le in linked}
            linked_slugs = {uri_slug(le.get("uri", "")) for le in linked}
            if gl and gl not in linked_labels and gl not in linked_slugs:
                gold_missing.append(gl)
    if gold_missing:
        return f"missing_gold_entities: {', '.join(sorted(gold_missing)[:3])}"
    return "uri_mismatch"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # load data
    records: dict[str, dict] = {}
    with HYBRID_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            records[r["question_id"]] = r

    with COVERAGE_JSON.open() as f:
        cov_data = json.load(f)

    # load original dataset for gold URI correction
    dev_data: dict[str, dict] = {}
    if DEV_FULL_JSON.exists():
        with DEV_FULL_JSON.open() as f:
            dev_data = {q["_id"]: q for q in json.load(f)}
        uri_corrections = build_gold_uri_corrections(cov_data, dev_data)
    else:
        print(f"WARNING: {DEV_FULL_JSON} not found — skipping gold URI correction")
        uri_corrections = {}

    gold_map: dict[str, list[dict]] = {}
    for q in cov_data["questions"]:
        qid = q["question_id"]
        gold = q.get("entities", [])
        # apply disambiguation corrections from supporting_facts
        gold = apply_gold_corrections(gold, uri_corrections.get(qid, {}))
        gold_map[qid] = gold

    # evaluate all questions
    all_results = []
    for qid, rec in records.items():
        linked = rec.get("debug_info", {}).get("entities_linked", [])
        gold = gold_map.get(qid, [])
        m = match_entities(linked, gold)
        all_results.append({
            "question_id": qid,
            "question": rec.get("question", ""),
            "question_type": rec.get("question_type", ""),
            "n_linked": len(linked),
            "n_gold": len(gold),
            **m,
        })

    # aggregate (exclude questions with no gold entities)
    valid = [r for r in all_results if not r.get("no_gold")]
    def safe_mean(vals):
        return sum(vals) / len(vals) if vals else None

    # filter None precision (questions where no entities were linked, tp+fp=0)
    agg_precision = safe_mean([r["precision"] for r in valid if r["precision"] is not None])
    agg_recall = safe_mean([r["recall"] for r in valid])
    full_cov_rate = sum(r["full_coverage"] for r in valid) / len(valid) if valid else None
    no_gold_count = sum(r.get("no_gold", False) for r in all_results)

    # per-type breakdown
    by_type: dict[str, dict] = {}
    for r in valid:
        qt = r["question_type"]
        by_type.setdefault(qt, {"precision": [], "recall": [], "full_coverage": 0, "n": 0})
        if r["precision"] is not None:
            by_type[qt]["precision"].append(r["precision"])
        by_type[qt]["recall"].append(r["recall"])
        by_type[qt]["full_coverage"] += int(r["full_coverage"])
        by_type[qt]["n"] += 1
    by_type_agg = {
        qt: {
            "precision": safe_mean(v["precision"]),
            "recall": safe_mean(v["recall"]),
            "full_coverage_rate": v["full_coverage"] / v["n"],
            "n": v["n"],
        }
        for qt, v in by_type.items()
    }

    # ── seed-only evaluation (filter gold to entities mentioned in question) ──
    seed_results = []
    for qid, rec in records.items():
        linked = rec.get("debug_info", {}).get("entities_linked", [])
        gold = gold_map.get(qid, [])
        seed_gold = [e for e in gold if is_seed_entity(e.get("name", ""), rec.get("question", ""))]
        m = match_entities(linked, seed_gold)
        seed_results.append({
            "question_id": qid,
            "question_type": rec.get("question_type", ""),
            "n_gold_total": len(gold),
            "n_gold_seed": len(seed_gold),
            **m,
        })

    seed_valid = [r for r in seed_results if not r.get("no_gold")]
    seed_precision = safe_mean([r["precision"] for r in seed_valid if r["precision"] is not None])
    seed_recall = safe_mean([r["recall"] for r in seed_valid])
    seed_fc = sum(r["full_coverage"] for r in seed_valid) / len(seed_valid) if seed_valid else None

    seed_by_type: dict[str, dict] = {}
    for r in seed_valid:
        qt = r["question_type"]
        seed_by_type.setdefault(qt, {"precision": [], "recall": [], "full_coverage": 0, "n": 0})
        if r["precision"] is not None:
            seed_by_type[qt]["precision"].append(r["precision"])
        seed_by_type[qt]["recall"].append(r["recall"])
        seed_by_type[qt]["full_coverage"] += int(r["full_coverage"])
        seed_by_type[qt]["n"] += 1
    seed_by_type_agg = {
        qt: {
            "precision": safe_mean(v["precision"]),
            "recall": safe_mean(v["recall"]),
            "full_coverage_rate": v["full_coverage"] / v["n"],
            "n": v["n"],
        }
        for qt, v in seed_by_type.items()
    }

    # sample 100 for failure analysis
    rng = random.Random(SAMPLE_SEED)
    sample = rng.sample(all_results, min(SAMPLE_SIZE, len(all_results)))
    failures = []
    for r in sample:
        if r.get("no_gold") or r.get("recall", 1.0) == 1.0:
            continue
        qid = r["question_id"]
        rec = records[qid]
        linked = rec.get("debug_info", {}).get("entities_linked", [])
        gold = gold_map.get(qid, [])
        failures.append({
            "question_id": qid,
            "question": r["question"],
            "question_type": r["question_type"],
            "precision": r["precision"],
            "recall": r["recall"],
            "linked_entities": [{"label": e.get("label"), "uri": e.get("uri"), "source": e.get("source")} for e in linked],
            "gold_entities": [{"name": e.get("name"), "uri": e.get("uri")} for e in gold],
            "failure_class": classify_failure(linked, gold),
        })

    output = {
        "meta": {
            "n_total": len(all_results),
            "n_with_gold": len(valid),
            "n_no_gold": no_gold_count,
            "sample_size": SAMPLE_SIZE,
            "sample_seed": SAMPLE_SEED,
            "matching": "URI-strict with gold-side dedup; label fallback only for URI-less gold",
            "gold_uri_corrections": sum(len(v) for v in uri_corrections.values()),
        },
        "aggregate": {
            "precision": round(agg_precision, 4) if agg_precision is not None else None,
            "recall": round(agg_recall, 4) if agg_recall is not None else None,
            "full_coverage_rate": round(full_cov_rate, 4) if full_cov_rate is not None else None,
        },
        "by_question_type": by_type_agg,
        "seed_only": {
            "note": "Gold filtered to entities whose name appears in question text (seed entities only, bridge entities excluded)",
            "n_with_seed_gold": len(seed_valid),
            "precision": round(seed_precision, 4) if seed_precision is not None else None,
            "recall": round(seed_recall, 4) if seed_recall is not None else None,
            "full_coverage_rate": round(seed_fc, 4) if seed_fc is not None else None,
        },
        "seed_only_by_question_type": seed_by_type_agg,
        "sample_failures": failures,
        "all_results": all_results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Questions evaluated : {len(all_results)}")
    print(f"With gold entities  : {len(valid)}  |  No gold: {no_gold_count}")
    print(f"Precision (mean)    : {agg_precision:.4f}" if agg_precision is not None else "Precision: N/A")
    print(f"Recall    (mean)    : {agg_recall:.4f}" if agg_recall is not None else "Recall: N/A")
    print(f"Full coverage rate  : {full_cov_rate:.4f}" if full_cov_rate is not None else "Full coverage: N/A")
    print("\nSeed-only (bridge entities excluded from gold):")
    print(f"  Precision (mean)  : {seed_precision:.4f}" if seed_precision is not None else "  Precision: N/A")
    print(f"  Recall    (mean)  : {seed_recall:.4f}" if seed_recall is not None else "  Recall: N/A")
    print(f"  Full coverage     : {seed_fc:.4f}" if seed_fc is not None else "  Full coverage: N/A")
    for qt, v in sorted(seed_by_type_agg.items()):
        print(f"    {qt:20s}  P={v['precision']:.3f}  R={v['recall']:.3f}  FC={v['full_coverage_rate']:.3f}  n={v['n']}")
    print("\nPer-type (all gold):")
    for qt, v in sorted(by_type_agg.items()):
        print(f"  {qt:20s}  P={v['precision']:.3f}  R={v['recall']:.3f}  FC={v['full_coverage_rate']:.3f}  n={v['n']}")
    print(f"\nSample failures (recall<1.0 in 100q sample): {len(failures)}")
    for fl in failures[:5]:
        print(f"  [{fl['question_type']}] {fl['question'][:70]}")
        print(f"    recall={fl['recall']:.2f}  class={fl['failure_class']}")
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
