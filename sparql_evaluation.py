#!/usr/bin/env python3
"""Evaluate direct LLM+SPARQL on WikiMultiHop answerable questions.

Gives an LLM the DBpedia schema for pre-resolved entities and asks it
to generate SPARQL, executes against DBpedia, then asks for a final answer.
Saves all intermediate results for investigation.

Usage:
    python3 sparql_evaluation.py --limit 5                      # quick test
    python3 sparql_evaluation.py                                # all 276 answerable
    python3 sparql_evaluation.py --model openai/gpt-oss-120b           # explicit model
"""

import argparse
import json
import logging
import os
import re
import socket
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from urllib.parse import unquote

from groq import Groq
from SPARQLWrapper import JSON, SPARQLWrapper

# ---------------------------------------------------------------------------
# sys.path guard (same pattern as graphrag_wiki.py)
# ---------------------------------------------------------------------------
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from dbpedia_qa.dbpedia.schema_discovery import (
    CACHED_CLASS_SCHEMAS,
    SchemaDiscovery,
    format_2hop_lines,
)
from dbpedia_qa.models import extract_final_answer
from evaluation.metrics import compute_containment, compute_em, compute_f1

logger = logging.getLogger(__name__)

_ipv4_patched = False


def force_ipv4():
    """Force IPv4 — DBpedia's IPv6 endpoint is unreachable from some networks.

    Only patches once; safe to call multiple times.
    """
    global _ipv4_patched
    if _ipv4_patched:
        return
    _orig = socket.getaddrinfo
    def _ipv4_only(*args, **kwargs):
        return [r for r in _orig(*args, **kwargs) if r[0] == socket.AF_INET]
    socket.getaddrinfo = _ipv4_only
    _ipv4_patched = True

# ---------------------------------------------------------------------------
# Data loading (same as grasp_evaluation.py)
# ---------------------------------------------------------------------------

DEFAULT_COVERAGE_FILE = "data/old_800q_results/coverage_analysis_800.json"

def load_answerable_questions(coverage_file: str) -> list[dict]:
    with open(coverage_file) as f:
        data = json.load(f)
    answerable = [q for q in data["questions"] if q["is_answerable"]]
    print(f"Loaded {len(answerable)} answerable questions from {coverage_file}")
    return answerable

def load_completed_ids(results_file: str) -> tuple[set[str], list[dict]]:
    records_by_id = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        records_by_id[rec["question_id"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        pass
    return set(records_by_id.keys()), list(records_by_id.values())

# ---------------------------------------------------------------------------
# Schema discovery (URI-direct, bypasses name resolution)
# ---------------------------------------------------------------------------

def get_schema_for_uris(schema_disc: SchemaDiscovery,
                        entities: list[dict],
                        question: str = "") -> str:
    """Build combined schema context for pre-resolved entity URIs.

    Args:
        question: Optional question text. When provided, properties matching
            question keywords are prioritized in the truncated list.
    """
    q_lower = question.lower() if question else ""

    def _prop_priority(prop_name: str) -> int:
        """0 = keyword match (show first), 1 = no match."""
        if q_lower and re.search(rf'\b{re.escape(prop_name.lower())}\b', q_lower):
            return 0
        return 1

    sections = []
    for ent in entities:
        uri = ent.get("uri", "")
        name = ent.get("name", uri.split("/")[-1].replace("_", " "))
        if not uri:
            continue

        entity_class = schema_disc.get_entity_class(uri)
        actual = schema_disc.get_actual_properties(uri)

        # Build schema text (mirrors SchemaDiscovery._build_schema_text)
        lines = [f"Entity: {name}", f"URI: <{uri}>"]
        if entity_class:
            lines.append(f"Class: dbo:{entity_class}")
        lines.append("")

        # Cached ontology schema
        if entity_class and entity_class in CACHED_CLASS_SCHEMAS:
            cached = CACHED_CLASS_SCHEMAS[entity_class]
            if cached.get("dbo"):
                lines.append(f"  dbo: {', '.join(cached['dbo'][:10])}")
            if cached.get("dbp"):
                lines.append(f"  dbp: {', '.join(cached['dbp'][:10])}")
            if cached.get("descriptions"):
                lines.append("  Property hints:")
                for prop, desc in list(cached["descriptions"].items())[:5]:
                    lines.append(f"    - {prop}: {desc}")
            lines.extend(format_2hop_lines(cached))
            lines.append("")

        # Actual properties with sample values (keyword-prioritized)
        values = actual.get("values", {})
        if actual["dbo"] or actual["dbp"]:
            lines.append("Actual properties found:")
            dbo_sorted = sorted(actual["dbo"], key=_prop_priority)
            dbp_sorted = sorted(actual["dbp"], key=_prop_priority)
            for prop in dbo_sorted[:15]:
                lines.append(f"  dbo:{prop} -> {values.get(f'dbo:{prop}', '')}")
            for prop in dbp_sorted[:15]:
                lines.append(f"  dbp:{prop} -> {values.get(f'dbp:{prop}', '')}")

        sections.append("\n".join(lines))

    prefix_text = """SPARQL Prefixes:
  PREFIX dbr: <http://dbpedia.org/resource/>
  PREFIX dbo: <http://dbpedia.org/ontology/>
  PREFIX dbp: <http://dbpedia.org/property/>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
  PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

IMPORTANT: Use the EXACT URIs shown for each entity."""

    return prefix_text + "\n\n" + "\n\n---\n\n".join(sections)

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(client: Groq, prompt: str, model: str,
             max_attempts: int = 5, initial_wait: float = 10.0) -> str:
    """Make a Groq LLM call with retry on 429 rate limit errors."""
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_attempts - 1:
                wait = initial_wait * (2 ** attempt)
                logger.info(f"Rate limited, waiting {wait:.0f}s "
                            f"(attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
            else:
                raise

def extract_sparql(response: str) -> str:
    """Extract SPARQL query from LLM response using brace-depth matching."""
    # Strip markdown fences
    if "```" in response:
        lines = response.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        response = "\n".join(lines)

    # Find where the query starts
    start = -1
    for keyword in ["PREFIX", "SELECT", "ASK"]:
        idx = response.find(keyword)
        if idx >= 0 and (start < 0 or idx < start):
            start = idx
    if start < 0:
        return response.strip()

    # Walk forward counting braces to find the end
    text = response[start:]
    depth = 0
    end = len(text)
    found_open = False
    for i, ch in enumerate(text):
        if ch == '{':
            depth += 1
            found_open = True
        elif ch == '}':
            depth -= 1
            if found_open and depth == 0:
                end = i + 1
                break

    query = text[:end].strip()
    # Append ORDER BY / LIMIT / OFFSET lines after the closing brace
    remainder = text[end:].strip()
    for line in remainder.split("\n"):
        line_s = line.strip().upper()
        if line_s.startswith(("ORDER", "LIMIT", "OFFSET", "GROUP")):
            query += "\n" + line.strip()
        else:
            break
    return query


# Characters that break dbr:/dbo:/dbp: prefixed names in SPARQL
_PREFIX_RE = re.compile(r"(dbr:|dbo:|dbp:)(\S+)")
_NEEDS_IRI = re.compile(r"['\",()!&#]")
_PREFIX_MAP = {
    "dbr:": "http://dbpedia.org/resource/",
    "dbo:": "http://dbpedia.org/ontology/",
    "dbp:": "http://dbpedia.org/property/",
}


def sanitize_sparql_uris(query: str) -> str:
    """Rewrite prefixed names with special chars to full IRI <...> form.

    E.g. dbr:God's_Gift_to_Women → <http://dbpedia.org/resource/God's_Gift_to_Women>
    Handles commas, parentheses, apostrophes, ampersands in entity names by
    greedily capturing the token then stripping trailing SPARQL delimiters.
    Skips matches inside string literals to avoid corrupting FILTER/BIND.
    """
    # Find quoted-string spans to skip
    string_spans = [(m.start(), m.end())
                    for m in re.finditer(r'"[^"]*"|\'[^\']*\'', query)]

    def _in_string(pos):
        return any(s <= pos < e for s, e in string_spans)

    def _rewrite(m):
        if _in_string(m.start()):
            return m.group(0)
        prefix, raw = m.group(1), m.group(2)
        local = raw
        trail = ""
        while local:
            ch = local[-1]
            # Count balanced parentheses — only strip unmatched trailing ')'
            if ch == ')':
                open_count = local.count('(')
                close_count = local.count(')')
                if open_count >= close_count:
                    break  # parens are balanced, stop stripping
            if ch in '.;},)':
                trail = ch + trail
                local = local[:-1]
            else:
                break
        if not local:
            return m.group(0)
        if _NEEDS_IRI.search(local):
            return f"<{_PREFIX_MAP[prefix]}{local}>{trail}"
        return m.group(0)
    return _PREFIX_RE.sub(_rewrite, query)


# ---------------------------------------------------------------------------
# SPARQL execution (lightweight, with retry)
# ---------------------------------------------------------------------------

SPARQL_ENDPOINT = "https://dbpedia.org/sparql"

def execute_sparql(query: str, endpoint: str = SPARQL_ENDPOINT,
                   timeout: int = 60,
                   max_retries: int = 2,
                   preserve_uris: bool = False) -> tuple[list[dict], str]:
    """Execute SPARQL query, return (results_list, error_string).

    Args:
        preserve_uris: If True, keep full URI values instead of extracting
            local names. Used by context entity discovery so downstream code
            can use URIs directly without re-encoding.

    Retries on transient network errors (timeouts, connection resets).
    Returns immediately on SPARQL syntax/query errors.
    """
    import urllib.error

    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)
    sparql.addCustomHttpHeader("User-Agent", "DBpediaQA/1.0 (research)")
    sparql.setQuery(query)

    last_err = ""
    for attempt in range(max_retries):
        try:
            raw = sparql.query().convert()
            bindings = raw.get("results", {}).get("bindings", [])
            rows = []
            for b in bindings:
                row = {}
                for var, val in b.items():
                    v = val.get("value", "")
                    if val.get("type") == "uri" and not preserve_uris:
                        v = unquote(v.split("/")[-1]).replace("_", " ")
                    row[var] = v
                rows.append(row)
            return rows, ""
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
        except Exception as e:
            return [], str(e)
    return [], last_err

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SPARQL_GEN_PROMPT = """You are a SPARQL expert for DBpedia. Generate a query to answer the question.

{schema}

QUESTION: {question}

RULES:
1. Use the EXACT entity URIs provided in the schema
2. For multi-hop queries, chain triple patterns using dbo: properties (they return URIs, not strings).
   Use the "2-Hop chains" section in the schema to identify valid chains.
   Example: if schema shows dbo:director -> Person: birthDate, write:
   ?film dbo:director ?person . ?person dbo:birthDate ?date .
3. Prefer dbo: properties for multi-hop joins; only use dbp: for single-hop leaf values
4. Always add LIMIT 100
5. For comparison questions (who/which was born first, etc.), return data for BOTH entities so results can be compared
6. If an entity URI contains special characters (apostrophes, commas, parentheses), use full IRI syntax <http://dbpedia.org/resource/...> instead of dbr: prefix
7. Generate ONLY the SPARQL query, no explanation"""

SPARQL_RETRY_PROMPT = """Your previous SPARQL query returned {problem}.

Previous query:
{sparql}

Fix the query using the schema below.

{schema}

QUESTION: {question}

Generate ONLY the corrected SPARQL query:"""

ANSWER_PROMPT = """Given the question and SPARQL query results from DBpedia, provide the answer.

Question: {question}
SPARQL Results:
{results}

End your response with FINAL ANSWER: <your answer>"""

# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def evaluate_question(q: dict, client: Groq, model: str,
                      schema_disc: SchemaDiscovery,
                      trace_dir: str) -> dict:
    """Run full pipeline for one question."""
    t0 = time.time()
    qid = q["question_id"]
    question = q["question"]
    gold = q["gold_answer"]
    entities = q.get("entities", [])
    entity_uris = [e["uri"] for e in entities if e.get("uri")]

    trace = {"question_id": qid, "entity_uris": entity_uris}
    record = {
        "question_id": qid,
        "question_type": q["question_type"],
        "question": question,
        "gold_answer": gold,
        "entity_uris": entity_uris,
    }

    try:
        # Step 1: Schema discovery
        schema_ctx = get_schema_for_uris(schema_disc, entities, question=question)
        trace["schema_context"] = schema_ctx
        record["schema_context"] = schema_ctx[:500]  # truncated in JSONL

        # Step 2: Generate SPARQL
        sparql_prompt = SPARQL_GEN_PROMPT.format(schema=schema_ctx, question=question)
        trace["sparql_prompt"] = sparql_prompt
        sparql_resp = call_llm(client, sparql_prompt, model)
        trace["sparql_response"] = sparql_resp
        sparql_query = sanitize_sparql_uris(extract_sparql(sparql_resp))
        trace["sparql_query"] = sparql_query
        record["sparql"] = sparql_query

        # Step 3: Execute SPARQL
        results, err = execute_sparql(sparql_query)
        trace["sparql_results"] = results
        record["sparql_results"] = results
        record["sparql_error"] = err

        # Step 4: Retry if empty or error
        record["sparql_retry"] = False
        if (not results and not err) or err:
            problem = f"an error: {err}" if err else "empty results (no rows)"
            retry_prompt = SPARQL_RETRY_PROMPT.format(
                problem=problem, sparql=sparql_query,
                schema=schema_ctx, question=question
            )
            trace["retry"] = {"prompt": retry_prompt}
            retry_resp = call_llm(client, retry_prompt, model)
            trace["retry"]["response"] = retry_resp
            retry_query = sanitize_sparql_uris(extract_sparql(retry_resp))
            trace["retry"]["query"] = retry_query
            record["sparql_retry"] = True
            record["sparql_retry_query"] = retry_query

            results2, err2 = execute_sparql(retry_query)
            trace["retry"]["results"] = results2
            record["sparql_retry_results"] = results2

            if results2:
                results = results2
                sparql_query = retry_query
                record["sparql"] = retry_query
                record["sparql_results"] = results2
                record["sparql_error"] = ""
            elif err2:
                record["sparql_error"] = err2

        # Step 5: Generate answer
        if results:
            fmt_results = json.dumps(results[:20], indent=2)
            answer_prompt = ANSWER_PROMPT.format(
                question=question, results=fmt_results
            )
            trace["answer_prompt"] = answer_prompt
            raw_answer = call_llm(client, answer_prompt, model)
            trace["answer_response"] = raw_answer
        else:
            raw_answer = ""
            trace["answer_prompt"] = ""
            trace["answer_response"] = ""

        record["raw_answer"] = raw_answer
        record["predicted_answer"] = extract_final_answer(raw_answer)
        record["error"] = ""

    except Exception as e:
        record.setdefault("raw_answer", "")
        record.setdefault("predicted_answer", "")
        record["error"] = str(e)
        logger.warning(f"Error on {qid}: {e}")

    # Compute metrics
    pred = record.get("predicted_answer", "")
    record["em"] = compute_em(pred, gold) if pred else 0.0
    record["f1"] = compute_f1(pred, gold) if pred else 0.0
    record["raw_em"] = record["em"]  # no separate cleaning step
    record["raw_f1"] = record["f1"]
    record["containment"] = compute_containment(
        record.get("raw_answer", ""), gold
    ) if record.get("raw_answer") else 0.0
    record["latency_seconds"] = round(time.time() - t0, 2)

    # Save trace
    trace_path = os.path.join(trace_dir, f"{qid}.json")
    record["trace_file"] = trace_path
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    return record

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    n = len(results)
    if n == 0:
        return

    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("question_type", "unknown")].append(r)

    def agg(recs):
        k = len(recs)
        return {
            "n": k,
            "em": sum(r["em"] for r in recs) / k,
            "f1": sum(r["f1"] for r in recs) / k,
            "containment": sum(r.get("containment", 0) for r in recs) / k,
            "answered": sum(1 for r in recs if r.get("predicted_answer")),
            "errors": sum(1 for r in recs if r.get("error")),
            "retries": sum(1 for r in recs if r.get("sparql_retry")),
        }

    overall = agg(results)
    print(f"\n{'=' * 75}")
    print("  Direct LLM+SPARQL Evaluation Results")
    print(f"{'=' * 75}")
    print(f"  N={n}  answered={overall['answered']}  "
          f"errors={overall['errors']}  retries={overall['retries']}")
    print()

    print(f"  {'Metric':<20} {'Overall':>10}", end="")
    for qt in sorted(by_type.keys()):
        print(f"  {qt[:12]:>12}", end="")
    print()
    print(f"  {'-'*20} {'-'*10}", end="")
    for _ in by_type:
        print(f"  {'-'*12}", end="")
    print()

    type_aggs = {qt: agg(recs) for qt, recs in sorted(by_type.items())}
    for name, key in [("EM", "em"), ("F1", "f1"), ("Containment", "containment")]:
        print(f"  {name:<20} {overall[key]:>9.1%}", end="")
        for qt in sorted(by_type.keys()):
            print(f"  {type_aggs[qt][key]:>11.1%}", end="")
        print()

    print(f"  {'N':<20} {n:>10}", end="")
    for qt in sorted(by_type.keys()):
        print(f"  {type_aggs[qt]['n']:>12}", end="")
    print()
    print(f"{'=' * 75}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    force_ipv4()
    parser = argparse.ArgumentParser(
        description="Direct LLM+SPARQL evaluation on WikiMultiHop"
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Max questions to evaluate (0 = all)")
    parser.add_argument("--model", default="openai/gpt-oss-120b",
                        help="Groq model (default: openai/gpt-oss-120b)")
    parser.add_argument("--output", default="data/sparql_eval.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--api-key", default=None,
                        help="Groq API key (default: GROQ_API_KEY env var)")
    parser.add_argument("--coverage-file", default=DEFAULT_COVERAGE_FILE,
                        help="Path to coverage analysis JSON")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-evaluate questions that had empty answers or errors")
    args = parser.parse_args()

    # Ensure API key
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY env var or --api-key required")
        sys.exit(1)

    client = Groq(api_key=api_key)
    schema_disc = SchemaDiscovery()

    # Trace directory
    trace_dir = "data/sparql_eval_traces"
    os.makedirs(trace_dir, exist_ok=True)

    # Load questions & resume
    questions = load_answerable_questions(args.coverage_file)
    completed_ids, completed_results = load_completed_ids(args.output)

    if args.retry_failed and completed_results:
        keep_records = [
            r for r in completed_results
            if r.get("predicted_answer") and not r.get("error")
        ]
        retry_count = len(completed_results) - len(keep_records)
        print(f"Retry mode: keeping {len(keep_records)}, retrying {retry_count}")

        # Atomic rewrite: temp file + os.replace() to prevent data loss on crash
        out_dir = os.path.dirname(args.output) or "."
        temp_fd, temp_path = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
        with os.fdopen(temp_fd, "w") as tmp:
            for rec in keep_records:
                tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(temp_path, args.output)

        completed_ids = {r["question_id"] for r in keep_records}
        completed_results = keep_records

    remaining = [q for q in questions if q["question_id"] not in completed_ids]

    if args.limit:
        remaining = remaining[:args.limit]

    print(f"Completed: {len(completed_ids)}, remaining: {len(remaining)}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print()

    all_results = list(completed_results)

    for i, q in enumerate(remaining, 1):
        qtype = q["question_type"]
        print(f"[{i}/{len(remaining)}] {qtype} | {q['question'][:60]}...", end=" ")

        record = evaluate_question(q, client, args.model, schema_disc, trace_dir)

        # Status indicator
        if record.get("error"):
            print(f"ERR ({record['latency_seconds']:.1f}s)")
        elif record["em"] > 0:
            print(f"EM=1 ({record['latency_seconds']:.1f}s)")
        elif record["containment"] > 0:
            print(f"cont ({record['latency_seconds']:.1f}s)")
        else:
            pred = record.get("predicted_answer", "")[:30]
            print(f"F1={record['f1']:.2f} pred='{pred}' ({record['latency_seconds']:.1f}s)")

        # Append to JSONL (resume-safe)
        with open(args.output, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        all_results.append(record)

        # Progress summary every 10
        if i % 10 == 0:
            n_done = len(all_results)
            em_so_far = sum(r["em"] for r in all_results) / n_done
            print(f"  --- Progress: {n_done} done, EM={em_so_far:.1%} ---")

        # Polite delay
        time.sleep(0.5)

    # Final summary
    print_summary(all_results)

    # Save summary JSON
    summary_path = args.output.replace(".jsonl", "_summary.json")
    n = len(all_results)
    if n > 0:
        by_type = defaultdict(list)
        for r in all_results:
            by_type[r.get("question_type", "unknown")].append(r)

        def agg(recs):
            k = len(recs)
            return {
                "n": k,
                "em": round(sum(r["em"] for r in recs) / k, 4),
                "f1": round(sum(r["f1"] for r in recs) / k, 4),
                "containment": round(sum(r.get("containment", 0) for r in recs) / k, 4),
                "answered": sum(1 for r in recs if r.get("predicted_answer")),
                "errors": sum(1 for r in recs if r.get("error")),
            }

        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "total": n,
            "overall": agg(all_results),
            "by_type": {qt: agg(recs) for qt, recs in sorted(by_type.items())},
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
