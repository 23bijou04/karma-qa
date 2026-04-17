#!/usr/bin/env bash
# Phase 02 — Run 2 Execution (9 cells, ~35h sequential)
#
# USAGE:
#   cd /Users/e23e889b/Documents/2026/2026_02/dbpedia_qa
#   git checkout run2-instrumentation
#   tmux new -s run2
#   export GROQ_API_KEY=gsk_...
#   caffeinate -i bash scripts/run-phase02.sh
#
# Features:
#   - Auto-resume: scripts skip question_ids already in output JSONL
#   - Per-cell tee logging to data/.../ *_run2.log
#   - Per-cell validation: line count + error rate, abort if > 5% errors
#   - Fingerprint capture: every LLM call logs to analysis/groq_fingerprints_run2.jsonl
#   - Skip cell by setting SKIP_A1=1, etc.
#
# Branch assumption: run2-instrumentation (HEAD must be 3+ commits ahead of paper-baseline).

set -u  # strict on unset vars; NOT -e (we want to handle errors ourselves)
set -o pipefail

# ----------------------------------------------------------------------------
# Preflight guards
# ----------------------------------------------------------------------------
REPO_ROOT="/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa"
cd "$REPO_ROOT" || { echo "FAIL: cannot cd to $REPO_ROOT"; exit 1; }

if [ -z "${GROQ_API_KEY:-}" ]; then
    echo "FAIL: GROQ_API_KEY not exported"; exit 1
fi

EXPECTED_BRANCH="run2-instrumentation"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]; then
    echo "FAIL: on branch '$CURRENT_BRANCH', expected '$EXPECTED_BRANCH'"; exit 1
fi

PAPER_BASELINE_SHA=$(git rev-parse 'paper-baseline^{}')
MERGE_BASE=$(git merge-base HEAD paper-baseline)
if [ "$MERGE_BASE" != "$PAPER_BASELINE_SHA" ]; then
    echo "FAIL: HEAD does not descend from paper-baseline"; exit 1
fi

# Fingerprint capture location (read by llm_provider.py patch)
export LLM_FP_LOG="$REPO_ROOT/plans/260417-1307-experiment-run2-run3-setup/analysis/groq_fingerprints_run2.jsonl"
mkdir -p "$(dirname "$LLM_FP_LOG")"

echo "=========================================="
echo "Phase 02 kickoff at $(date)"
echo "Branch: $CURRENT_BRANCH"
echo "HEAD:   $(git rev-parse HEAD)"
echo "FP log: $LLM_FP_LOG"
echo "=========================================="

# ----------------------------------------------------------------------------
# Per-cell validation helper
# ----------------------------------------------------------------------------
# Verifies cell output has 1000 lines and <5% errors; aborts phase if not.
validate_cell() {
    local label="$1" jsonl="$2"
    if [ ! -f "$jsonl" ]; then
        echo "FAIL $label: output $jsonl missing"; exit 1
    fi
    local lines
    lines=$(wc -l < "$jsonl")
    local errors
    errors=$(python -c "
import json, sys
n = 0
with open('$jsonl') as f:
    for line in f:
        if not line.strip(): continue
        try:
            r = json.loads(line)
            if r.get('status') == 'error' or r.get('error'):
                n += 1
        except json.JSONDecodeError:
            n += 1
print(n)
")
    echo "  $label: lines=$lines errors=$errors"
    if [ "$lines" -lt 1000 ]; then
        echo "WARN $label: $lines < 1000 — rerun this script to resume"; return 1
    fi
    local err_pct=$(( errors * 100 / (lines > 0 ? lines : 1) ))
    if [ "$err_pct" -gt 5 ]; then
        echo "FAIL $label: error rate ${err_pct}% > 5%"; exit 1
    fi
    return 0
}

# ----------------------------------------------------------------------------
# Track A — Groq serial (single-key)
# ----------------------------------------------------------------------------

# A1. Hybrid 2Wiki (~1.91h)
if [ "${SKIP_A1:-0}" -ne 1 ]; then
    echo ""; echo "=== A1. Hybrid 2Wiki @ $(date) ==="
    source venv/bin/activate
    python scripts/hybrid_batch_eval.py --api-key "$GROQ_API_KEY" \
        --output-file data/eval_1000q/hybrid/hybrid_eval_1000_run2.jsonl \
        2>&1 | tee data/eval_1000q/hybrid/hybrid_eval_1000_run2.log
    validate_cell "A1" data/eval_1000q/hybrid/hybrid_eval_1000_run2.jsonl || exit 1
fi

# A2. Hybrid HotPotQA (~8.55h — bottleneck)
if [ "${SKIP_A2:-0}" -ne 1 ]; then
    echo ""; echo "=== A2. Hybrid HotPotQA @ $(date) ==="
    source venv/bin/activate
    python scripts/run-hybrid-cross-benchmark.py --api-key "$GROQ_API_KEY" \
        --benchmark hotpotqa \
        --output data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_run2.jsonl \
        2>&1 | tee data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_run2.log
    validate_cell "A2" data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_run2.jsonl || exit 1
fi

# A3. Hybrid MuSiQue (~10.94h — bottleneck)
if [ "${SKIP_A3:-0}" -ne 1 ]; then
    echo ""; echo "=== A3. Hybrid MuSiQue @ $(date) ==="
    source venv/bin/activate
    python scripts/run-hybrid-cross-benchmark.py --api-key "$GROQ_API_KEY" \
        --benchmark musique \
        --output data/cross_benchmark/musique/hybrid_musique_1000q_run2.jsonl \
        2>&1 | tee data/cross_benchmark/musique/hybrid_musique_1000q_run2.log
    validate_cell "A3" data/cross_benchmark/musique/hybrid_musique_1000q_run2.jsonl || exit 1
fi

# A4. LogicRAG 2Wiki (~4.25h — switches venv)
if [ "${SKIP_A4:-0}" -ne 1 ]; then
    echo ""; echo "=== A4. LogicRAG 2Wiki @ $(date) ==="
    deactivate 2>/dev/null || true
    source baselines/LogicRAG/venv_logicrag/bin/activate
    python scripts/run-logicrag-1000q.py --api-key "$GROQ_API_KEY" \
        --output data/eval_1000q/logicrag/logicrag_1000q_traced_run2.jsonl \
        2>&1 | tee data/eval_1000q/logicrag/logicrag_1000q_traced_run2.log
    validate_cell "A4" data/eval_1000q/logicrag/logicrag_1000q_traced_run2.jsonl || exit 1
fi

# A5. LogicRAG HotPotQA (~3.40h)
if [ "${SKIP_A5:-0}" -ne 1 ]; then
    echo ""; echo "=== A5. LogicRAG HotPotQA @ $(date) ==="
    source baselines/LogicRAG/venv_logicrag/bin/activate
    python scripts/run-logicrag-cross-benchmark.py --api-key "$GROQ_API_KEY" \
        --benchmark hotpotqa \
        --output data/cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_run2.jsonl \
        2>&1 | tee data/cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_run2.log
    validate_cell "A5" data/cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_run2.jsonl || exit 1
fi

# A6. LogicRAG MuSiQue (~4.75h)
if [ "${SKIP_A6:-0}" -ne 1 ]; then
    echo ""; echo "=== A6. LogicRAG MuSiQue @ $(date) ==="
    source baselines/LogicRAG/venv_logicrag/bin/activate
    python scripts/run-logicrag-cross-benchmark.py --api-key "$GROQ_API_KEY" \
        --benchmark musique \
        --output data/cross_benchmark/musique/logicrag_musique_1000q_run2.jsonl \
        2>&1 | tee data/cross_benchmark/musique/logicrag_musique_1000q_run2.log
    validate_cell "A6" data/cross_benchmark/musique/logicrag_musique_1000q_run2.jsonl || exit 1
fi

# A7. LinearRAG HotPotQA (~0.4h — switch venv + cwd)
if [ "${SKIP_A7:-0}" -ne 1 ]; then
    echo ""; echo "=== A7. LinearRAG HotPotQA @ $(date) ==="
    deactivate 2>/dev/null || true
    cd "$REPO_ROOT/baselines/LinearRAG"
    source venv_linearrag/bin/activate
    OPENAI_API_KEY="$GROQ_API_KEY" OPENAI_BASE_URL='https://api.groq.com/openai/v1' \
        python run-cross-benchmark-eval.py --dataset hotpotqa \
        --output ../../data/cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_run2.jsonl \
        2>&1 | tee ../../data/cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_run2.log
    cd "$REPO_ROOT"
    validate_cell "A7" data/cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_run2.jsonl || exit 1
fi

# A8. LinearRAG MuSiQue (~0.4h)
if [ "${SKIP_A8:-0}" -ne 1 ]; then
    echo ""; echo "=== A8. LinearRAG MuSiQue @ $(date) ==="
    cd "$REPO_ROOT/baselines/LinearRAG"
    source venv_linearrag/bin/activate
    OPENAI_API_KEY="$GROQ_API_KEY" OPENAI_BASE_URL='https://api.groq.com/openai/v1' \
        python run-cross-benchmark-eval.py --dataset musique \
        --output ../../data/cross_benchmark/musique/linearrag_musique_1000q_run2.jsonl \
        2>&1 | tee ../../data/cross_benchmark/musique/linearrag_musique_1000q_run2.log
    cd "$REPO_ROOT"
    validate_cell "A8" data/cross_benchmark/musique/linearrag_musique_1000q_run2.jsonl || exit 1
fi

# B1 + B2. LinearRAG 2Wiki (~0.5h — writes timestamped dir, needs conversion)
if [ "${SKIP_B1:-0}" -ne 1 ]; then
    echo ""; echo "=== B1. LinearRAG 2Wiki @ $(date) ==="
    cd "$REPO_ROOT/baselines/LinearRAG"
    source venv_linearrag/bin/activate
    OPENAI_API_KEY="$GROQ_API_KEY" OPENAI_BASE_URL='https://api.groq.com/openai/v1' \
        python run_full_eval.py 2>&1 | tee run2_2wiki.log
    RESULTS_DIR=$(ls -td results/2wikimultihop/eval_* | head -1)
    echo "B1: raw results in $RESULTS_DIR"
    cd "$REPO_ROOT"
    echo ""; echo "=== B2. Convert LinearRAG 2Wiki to JSONL @ $(date) ==="
    python scripts/convert-linearrag-predictions.py \
        "baselines/LinearRAG/$RESULTS_DIR/predictions.json" \
        --jsonl-out data/eval_1000q/linearrag/linearrag_1000q_run2.jsonl \
        --json-out data/eval_1000q/linearrag_predictions_1000_run2.json \
        2>&1 | tee data/eval_1000q/linearrag/linearrag_1000q_run2_convert.log
    validate_cell "B1-2" data/eval_1000q/linearrag/linearrag_1000q_run2.jsonl || exit 1
fi

echo ""
echo "=========================================="
echo "Phase 02 complete at $(date)"
echo "Next: bash plans/260417-1307-experiment-run2-run3-setup/run-phase03.sh"
echo "=========================================="
