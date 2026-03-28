#!/usr/bin/env python3
"""Pre-flight checks before the long LogicRAG 1000q evaluation run.

Validates Python version, API key, dependencies, data files, cached embeddings,
output resume status, and a live Groq API call. Exits 0 if all critical checks
pass, 1 otherwise.

Usage:
    source baselines/LogicRAG/venv_logicrag/bin/activate
    python scripts/preflight-logicrag.py
"""

import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGICRAG_DIR = os.path.join(PROJECT_ROOT, "baselines", "LogicRAG")
sys.path.insert(0, LOGICRAG_DIR)

CORPUS_FILE = os.path.join(LOGICRAG_DIR, "dataset", "2wikimultihopqa_corpus.json")
COVERAGE_FILE = os.path.join(
    PROJECT_ROOT, "data", "eval_1000q", "coverage_analysis_1000_corrected.json"
)
HOP_FILE = os.path.join(
    PROJECT_ROOT, "data", "eval_1000q", "hop_verification_results.json"
)
EMBEDDINGS_DIR = os.path.join(LOGICRAG_DIR, "cache")
OUTPUT_FILE = os.path.join(
    PROJECT_ROOT, "data", "eval_1000q", "logicrag_1000q_traced.jsonl"
)

# ---------------------------------------------------------------------------
# Results accumulator
# ---------------------------------------------------------------------------
results: list[tuple[str, bool, str]] = []  # (label, passed, detail)


def check(label: str, passed: bool, detail: str = "") -> bool:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}{': ' + detail if detail else ''}")
    return passed


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python_version() -> bool:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    return check("Python >= 3.10", ok, f"found {major}.{minor}")


def check_groq_api_key() -> bool:
    key = os.environ.get("GROQ_API_KEY", "")
    ok = bool(key)
    return check("GROQ_API_KEY set", ok, "not set" if not ok else "set (hidden)")


def check_imports() -> bool:
    deps = ["sentence_transformers", "torch", "openai", "backoff", "ratelimit"]
    all_ok = True
    for dep in deps:
        try:
            __import__(dep)
            check(f"import {dep}", True)
        except ImportError as exc:
            check(f"import {dep}", False, str(exc))
            all_ok = False
    return all_ok


def check_corpus_file() -> bool:
    if not os.path.exists(CORPUS_FILE):
        return check("Corpus file exists", False, CORPUS_FILE)
    size_mb = os.path.getsize(CORPUS_FILE) / 1_048_576
    ok = size_mb >= 2.0  # expect ~3 MB
    return check("Corpus file size", ok, f"{size_mb:.1f} MB (expected ~3 MB)")


def check_dataset_file() -> bool:
    if not os.path.exists(COVERAGE_FILE):
        return check("Coverage dataset exists", False, COVERAGE_FILE)
    try:
        with open(COVERAGE_FILE) as f:
            data = json.load(f)
        questions = data.get("questions", data)  # top-level list or nested under "questions"
        n = len(questions) if isinstance(questions, list) else len(data)
        return check("Coverage dataset loaded", True, f"{n} questions")
    except Exception as exc:
        return check("Coverage dataset loaded", False, str(exc))


def check_hop_verification() -> bool:
    ok = os.path.exists(HOP_FILE)
    return check("Hop verification file exists", ok, "" if ok else HOP_FILE)


def check_cached_embeddings() -> bool:
    pts = [
        f for f in os.listdir(EMBEDDINGS_DIR) if f.startswith("embeddings_") and f.endswith(".pt")
    ] if os.path.isdir(EMBEDDINGS_DIR) else []
    found = len(pts) > 0
    detail = f"{len(pts)} file(s): {', '.join(pts)}" if found else "none found (cold start)"
    # not a critical check — always pass, just informational
    results.append(("Cached embeddings", None, detail))  # type: ignore[arg-type]
    print(f"  [INFO] Cached embeddings: {detail}")
    return True


def check_output_status() -> bool:
    if not os.path.exists(OUTPUT_FILE):
        results.append(("Output file status", None, "no existing output — fresh run"))  # type: ignore[arg-type]
        print("  [INFO] Output file status: no existing output — fresh run")
        return True
    completed = 0
    try:
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    completed += 1
    except Exception as exc:
        return check("Output file readable", False, str(exc))
    results.append(("Output file status", None, f"{completed} questions completed — will resume"))  # type: ignore[arg-type]
    print(f"  [INFO] Output file status: {completed} questions completed — will resume")
    return True


def check_groq_api_live() -> bool:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return check("Groq API live call", False, "no API key")
    try:
        import openai  # noqa: PLC0415
        client = openai.OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
            temperature=0,
        )
        latency = time.monotonic() - t0
        answer = resp.choices[0].message.content or ""
        return check("Groq API live call", True, f"{latency:.1f}s — '{answer.strip()}'")
    except Exception as exc:
        return check("Groq API live call", False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\n=== LogicRAG Pre-flight Checks ===\n")

    critical_pass = all([
        check_python_version(),
        check_groq_api_key(),
        check_imports(),
        check_corpus_file(),
        check_dataset_file(),
        check_hop_verification(),
    ])
    check_cached_embeddings()
    check_output_status()
    critical_pass = critical_pass and check_groq_api_live()

    print("\n--- Summary ---")
    for label, passed, detail in results:
        if passed is None:
            tag = "INFO"
        else:
            tag = "PASS" if passed else "FAIL"
        print(f"  [{tag:4}] {label}{': ' + detail if detail else ''}")

    verdict = "GO" if critical_pass else "NO-GO"
    print(f"\nVerdict: {verdict}\n")
    return 0 if critical_pass else 1


if __name__ == "__main__":
    sys.exit(main())
