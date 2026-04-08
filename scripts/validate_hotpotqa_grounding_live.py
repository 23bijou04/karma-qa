#!/usr/bin/env python3
"""
Live validation of HotpotQA DBpedia grounding results (sampling + spot-check).
Randomly samples 15 questions stratified across 5 categories and verifies
each against live DBpedia SPARQL and Wikipedia HTTP.

Usage: python scripts/validate_hotpotqa_grounding_live.py

Writes report to: plans/reports/debugger-260407-1438-hotpotqa-live-validation.md
"""
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

try:
    from SPARQLWrapper import SPARQLWrapper
    from SPARQLWrapper import JSON as SPARQL_JSON
except ImportError:
    print("SPARQLWrapper not installed. Run: pip install SPARQLWrapper")
    sys.exit(1)

DATA_FILE = BASE / "data/cross_benchmark/hotpotqa/dbpedia_grounding_results.json"
REPORT_PATH = BASE / "plans/reports/debugger-260407-1438-hotpotqa-live-validation.md"
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
SPARQL_DELAY = 0.3
SEED = 42
N_PER_STRATUM = 3

# ── SPARQL helpers ──────────────────────────────────────────────────────────

_sparql = SPARQLWrapper(SPARQL_ENDPOINT)
_sparql.setReturnFormat(SPARQL_JSON)
_sparql.setTimeout(20)
_sparql.addCustomHttpHeader("User-Agent", "DBpediaQA-LiveValidator/1.0 (academic)")

_last_sparql = 0.0


def _rate_limit():
    global _last_sparql
    elapsed = time.time() - _last_sparql
    if elapsed < SPARQL_DELAY:
        time.sleep(SPARQL_DELAY - elapsed)
    _last_sparql = time.time()


def sparql_ask(uri: str) -> bool | str:
    """ASK if URI has any triples. Returns True/False or error string."""
    _rate_limit()
    _sparql.setQuery(f"ASK {{ <{uri}> ?p ?o }}")
    try:
        r = _sparql.query().convert()
        return bool(r.get("boolean", False))
    except Exception as e:
        return f"error: {e}"


def sparql_connection(uri1: str, uri2: str) -> list[dict]:
    """Find direct properties between two URIs (both directions, noise-filtered)."""
    _rate_limit()
    query = f"""
SELECT DISTINCT ?prop ?dir WHERE {{
  {{
    <{uri1}> ?prop <{uri2}> .
    BIND("forward" AS ?dir)
  }} UNION {{
    <{uri2}> ?prop <{uri1}> .
    BIND("reverse" AS ?dir)
  }}
  FILTER(?prop != <http://dbpedia.org/ontology/wikiPageWikiLink>)
  FILTER(?prop != <http://www.w3.org/2002/07/owl#sameAs>)
  FILTER(!STRSTARTS(STR(?prop), "http://www.w3.org/"))
}}
LIMIT 20
"""
    _sparql.setQuery(query)
    try:
        r = _sparql.query().convert()
        rows = r.get("results", {}).get("bindings", [])
        return [{"prop": b["prop"]["value"], "dir": b["dir"]["value"]} for b in rows]
    except Exception as e:
        return [{"error": str(e)}]


def sparql_entity_literals(uri: str) -> list[dict]:
    """Fetch English literal values (all properties, short strings)."""
    _rate_limit()
    query = f"""
SELECT DISTINCT ?prop ?val WHERE {{
  <{uri}> ?prop ?val .
  FILTER (isLiteral(?val) && lang(?val) IN ("", "en"))
  FILTER (strlen(str(?val)) < 300)
}}
LIMIT 80
"""
    _sparql.setQuery(query)
    try:
        r = _sparql.query().convert()
        rows = r.get("results", {}).get("bindings", [])
        return [{"prop": b["prop"]["value"].split("/")[-1], "val": b["val"]["value"]}
                for b in rows]
    except Exception as e:
        return [{"error": str(e)}]


# ── Wikipedia helper ────────────────────────────────────────────────────────

def wiki_head(title: str) -> bool | str:
    """HEAD request to Wikipedia. Returns True/False or error."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/wiki/{encoded}"
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "DBpediaQA-Validator/1.0 (academic research)")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        return False if e.code == 404 else f"HTTP {e.code}"
    except Exception as e:
        return f"error: {e}"


# ── Sampling ────────────────────────────────────────────────────────────────

def sample_strata(pq: list[dict]) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    strata = {
        "connected":     [q for q in pq if q["all_resolved"] and q.get("pair_connected")],
        "disconnected":  [q for q in pq if q["all_resolved"] and not q.get("pair_connected")],
        "unresolved":    [q for q in pq if not q["all_resolved"]],
        "entity_answer": [q for q in pq if q.get("answer_grounded")
                          and q.get("answer_category") == "entity_answer"],
        "value_lookup":  [q for q in pq if q.get("answer_grounded")
                          and q.get("answer_category") == "value_lookup"],
    }
    return {name: rng.sample(pool, min(N_PER_STRATUM, len(pool)))
            for name, pool in strata.items()}


# ── Per-question validation ─────────────────────────────────────────────────

def validate_q(q: dict, stratum: str) -> dict[str, Any]:
    res: dict[str, Any] = {
        "stratum": stratum,
        "question_id": q["question_id"],
        "question": q["question"],
        "type": q["type"],
        "gold_answer": q["gold_answer"],
        "titles": q.get("titles", []),
        "uris": q.get("uris", []),
        "claimed": {
            "all_resolved": q["all_resolved"],
            "pair_connected": q.get("pair_connected"),
            "connection_type": q.get("connection_type"),
            "connecting_properties": q.get("connecting_properties", []),
            "answer_grounded": q.get("answer_grounded"),
            "answer_category": q.get("answer_category"),
            "grounding_source": q.get("grounding_source"),
            "matched_value": q.get("matched_value"),
        },
        "live": {},
    }

    # Entity ASK
    res["live"]["uri_ask"] = {uri: sparql_ask(uri) for uri in q.get("uris", []) if uri}

    # Wikipedia HEAD
    res["live"]["wiki_head"] = {t: wiki_head(t) for t in q.get("titles", [])}

    uris = [u for u in q.get("uris", []) if u]

    # Connection check
    if stratum == "connected" and len(uris) >= 2:
        res["live"]["connection_props"] = sparql_connection(uris[0], uris[1])
    elif stratum == "disconnected" and len(uris) >= 2:
        res["live"]["connection_props_verify_disconnect"] = sparql_connection(uris[0], uris[1])

    # Value grounding check
    if stratum == "value_lookup" and q.get("grounding_source"):
        src = q["grounding_source"]
        idx = 1 if src == "entity2" else 0
        if idx < len(uris):
            lits = sparql_entity_literals(uris[idx])
            if lits and "error" in lits[0]:
                res["live"]["grounding_literals"] = lits
            else:
                ans_low = q["gold_answer"].lower().strip()
                matches = [v for v in lits if ans_low in v["val"].lower()]
                res["live"]["grounding_literals"] = {
                    "fetched": len(lits),
                    "matches": matches[:6],
                    "sample_no_match": lits[:8] if not matches else [],
                }

    return res


# ── Report rendering ────────────────────────────────────────────────────────

STRATUM_LABELS = {
    "connected":     "Stratum 1: all_resolved=true, pair_connected=true",
    "disconnected":  "Stratum 2: all_resolved=true, pair_connected=false",
    "unresolved":    "Stratum 3: all_resolved=false (entity resolution failed)",
    "entity_answer": "Stratum 4: answer_grounded=true, answer_category=entity_answer",
    "value_lookup":  "Stratum 5: answer_grounded=true, answer_category=value_lookup",
}


def _yn(v: Any) -> str:
    if v is True:
        return "YES"
    if v is False:
        return "NO"
    return str(v)


def render(results: dict[str, list]) -> str:
    lines = [
        "# HotpotQA DBpedia Grounding — Live Validation Report",
        "",
        f"**Date:** 2026-04-07  |  **Seed:** {SEED}  |  **N per stratum:** {N_PER_STRATUM}",
        "",
        "Spot-checks claimed grounding against live DBpedia SPARQL and Wikipedia.",
        "",
    ]

    issue_count = 0

    for stratum, label in STRATUM_LABELS.items():
        lines.append(f"## {label}")
        lines.append("")
        for r in results.get(stratum, []):
            lines.append(f"### {r['question']}")
            lines.append(f"- **ID:** `{r['question_id']}` | **Type:** {r['type']}")
            lines.append(f"- **Gold answer:** `{r['gold_answer']}`")
            lines.append("")

            c = r["claimed"]
            lines.append("**Claimed state:**")
            lines.append(f"- all_resolved={c['all_resolved']}, pair_connected={c['pair_connected']}, "
                         f"connection_type={c['connection_type']}")
            lines.append(f"- answer_grounded={c['answer_grounded']}, "
                         f"answer_category={c['answer_category']}, "
                         f"grounding_source={c['grounding_source']}")
            if c["matched_value"]:
                lines.append(f"- matched_value: `{c['matched_value']}`")
            if c["connecting_properties"]:
                lines.append(f"- connecting_properties: `{c['connecting_properties']}`")
            lines.append("")

            # URI ASK
            lines.append("**Entity URI checks (DBpedia ASK):**")
            uri_issues = []
            for uri, exists in r["live"].get("uri_ask", {}).items():
                short = uri.split("/resource/")[-1] if "/resource/" in uri else uri
                lines.append(f"- `{short}` → {_yn(exists)}")
                if exists is False:
                    uri_issues.append(uri)
                    issue_count += 1
            if uri_issues:
                lines.append(f"- **ISSUE: {len(uri_issues)} URI(s) returned ASK=false**")
            lines.append("")

            # Wikipedia HEAD
            lines.append("**Wikipedia title checks (HTTP HEAD):**")
            wiki_issues = []
            for title, exists in r["live"].get("wiki_head", {}).items():
                lines.append(f"- `{title}` → {_yn(exists)}")
                if exists is False:
                    wiki_issues.append(title)
                    issue_count += 1
            if wiki_issues:
                lines.append(f"- **ISSUE: {len(wiki_issues)} title(s) returned 404**")
            lines.append("")

            # Connection props (connected)
            if "connection_props" in r["live"]:
                props = r["live"]["connection_props"]
                lines.append("**Live connection properties (DBpedia SELECT):**")
                if not props:
                    lines.append("- **ISSUE: claimed connected but 0 direct props found**")
                    issue_count += 1
                elif "error" in props[0]:
                    lines.append(f"- ERROR: {props[0]['error']}")
                else:
                    for p in props[:8]:
                        pname = p["prop"].split("/")[-1]
                        lines.append(f"- `{pname}` ({p['dir']})")
                    if len(props) > 8:
                        lines.append(f"- ... and {len(props) - 8} more")
                lines.append("")

            # Connection props (disconnected — verify)
            if "connection_props_verify_disconnect" in r["live"]:
                props = r["live"]["connection_props_verify_disconnect"]
                lines.append("**Live connection check (claimed disconnected — verify):**")
                if not props:
                    lines.append("- Disconnection confirmed: 0 direct props found")
                elif "error" in props[0]:
                    lines.append(f"- ERROR: {props[0]['error']}")
                else:
                    lines.append(f"- **UNEXPECTED: found {len(props)} connecting prop(s) "
                                 f"despite claimed pair_connected=false:**")
                    for p in props[:5]:
                        pname = p["prop"].split("/")[-1]
                        lines.append(f"  - `{pname}` ({p['dir']})")
                    issue_count += 1
                lines.append("")

            # Value grounding
            if "grounding_literals" in r["live"]:
                gl = r["live"]["grounding_literals"]
                lines.append("**Answer grounding literals (DBpedia SELECT):**")
                if isinstance(gl, list) and gl and "error" in gl[0]:
                    lines.append(f"- ERROR: {gl[0]['error']}")
                elif isinstance(gl, dict):
                    lines.append(f"- Literals fetched from entity: {gl['fetched']}")
                    if gl["matches"]:
                        lines.append(f"- Gold answer FOUND in {len(gl['matches'])} literal(s):")
                        for m in gl["matches"]:
                            lines.append(f"  - `{m['prop']}`: `{m['val'][:120]}`")
                    else:
                        lines.append("- **Gold answer NOT found in any fetched literal**")
                        if gl.get("sample_no_match"):
                            lines.append("- Sample values returned:")
                            for v in gl["sample_no_match"][:5]:
                                lines.append(f"  - `{v['prop']}`: `{v['val'][:100]}`")
                        issue_count += 1
                lines.append("")

            lines.append("---")
            lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total questions validated: {N_PER_STRATUM * 5} (3 per stratum × 5 strata)")
    lines.append(f"- Issues found: {issue_count}")
    lines.append("")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {DATA_FILE}...")
    with open(DATA_FILE) as f:
        data = json.load(f)
    pq = data["per_question"]
    print(f"Loaded {len(pq)} questions.\n")

    print("Sampling strata (seed=42)...")
    samples = sample_strata(pq)
    for name, qs in samples.items():
        print(f"  {name}: {len(qs)} questions")

    total_q = sum(len(v) for v in samples.values())
    done = 0
    all_results: dict[str, list] = {}

    for stratum, questions in samples.items():
        all_results[stratum] = []
        for q in questions:
            done += 1
            print(f"\n[{done}/{total_q}] {stratum}: {q['question'][:65]}...")
            r = validate_q(q, stratum)
            all_results[stratum].append(r)

            # Quick inline status
            asks = r["live"].get("uri_ask", {})
            wikis = r["live"].get("wiki_head", {})
            uri_ok = all(v is True for v in asks.values())
            wiki_ok = all(v is True for v in wikis.values())
            print(f"  URIs OK={uri_ok}  Wiki OK={wiki_ok}")

    report_text = render(all_results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {REPORT_PATH}")
    print("\n" + "=" * 70)
    print(report_text)


if __name__ == "__main__":
    main()
