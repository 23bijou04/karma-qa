#!/usr/bin/env bash
# Judge all 6 Run 2 output files with gemma3:12b (Ollama) for variance analysis.
# Sequential: single Ollama GPU bottleneck.
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="data/judge_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_$(date +%Y%m%d_%H%M).log"

FILES=(
  "data/eval_1000q/linearrag/linearrag_1000q_run2.jsonl"
  "data/eval_1000q/logicrag/logicrag_1000q_traced_run2.jsonl"
  "data/cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_run2.jsonl"
  "data/cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_run2.jsonl"
  "data/cross_benchmark/musique/linearrag_musique_1000q_run2.jsonl"
  "data/cross_benchmark/musique/logicrag_musique_1000q_run2.jsonl"
)

echo "=== START $(date) ===" | tee -a "$MASTER_LOG"
for f in "${FILES[@]}"; do
  echo "" | tee -a "$MASTER_LOG"
  echo "=== $(date) JUDGING $f ===" | tee -a "$MASTER_LOG"
  python scripts/judge_hybrid_eval.py "$f" 2>&1 | tee -a "$MASTER_LOG"
done
echo "" | tee -a "$MASTER_LOG"
echo "=== ALL DONE $(date) ===" | tee -a "$MASTER_LOG"
