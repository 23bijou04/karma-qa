# DBpedia Hybrid QA

Hybrid QA over DBpedia + Wikipedia for 2WikiMultiHopQA, HotPotQA, MuSiQue.

## Requirements

- Python 3.10+
- macOS or Linux (uses `signal.SIGALRM`)
- `GROQ_API_KEY` (free tier: <https://console.groq.com>) **or** `GITHUB_TOKEN`
- Optional: Ollama + `gemma3:12b` (for the LLM judge step)

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]" ".[rag]"
python -m spacy download en_core_web_sm

# Pre-cache embedding models (~100 MB each)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Clone baselines used by evaluation scripts
git clone https://github.com/chensyCN/LogicRAG.git baselines/LogicRAG
```

Key deps (declared in `pyproject.toml`): `groq`, `openai`, `sentence-transformers`, `faiss-cpu`, `spacy`, `tenacity`, `ddgs`, `websockets`, `huggingface_hub`.

## Run the main experiment

```bash
export GROQ_API_KEY="..."

# 2WikiMultiHopQA (1000q, ~5 h on Groq free tier, auto-resumes on 429s)
python scripts/hybrid_batch_eval.py --api-key $GROQ_API_KEY

# Cross-benchmark
python scripts/run-hybrid-cross-benchmark.py --benchmark hotpotqa --api-key $GROQ_API_KEY
python scripts/run-hybrid-cross-benchmark.py --benchmark musique  --api-key $GROQ_API_KEY

# Judge raw outputs (gemma3:12b via Ollama; or use --provider groq)
python scripts/judge_hybrid_eval.py data/eval_1000q/hybrid/hybrid_eval_1000.jsonl
```

LLM settings: `temperature=0`, exponential-backoff retry on 429s. Groq routes across heterogeneous accelerators — expect ±0.3pp drift across reruns.

## Result files

Headline: **Hybrid 89.4 %** (894/1000) Judge EM on 2WikiMultiHopQA, **92.9 %** (342/368) on the hop-verified subset.

| Result | JSONL |
|---|---|
| Hybrid 1000q (judged) | `data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl` |
| LinearRAG 1000q (judged) | `data/eval_1000q/linearrag/linearrag_1000q_judged.jsonl` |
| LogicRAG 1000q (judged) | `data/eval_1000q/logicrag/logicrag_1000q_traced_judged.jsonl` |
| SPARQL component 368q | `data/eval_1000q/sparql/sparql_*_judged.jsonl` |
| GRASP 368q | `data/eval_1000q/grasp/grasp_*_judged.jsonl` |
| Hybrid HotPotQA | `data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged.jsonl` |
| Hybrid MuSiQue | `data/cross_benchmark/musique/hybrid_musique_1000q_judged.jsonl` |
| 1000q manifest | `data/eval_1000q/ground-truth/coverage_analysis_1000.json` |
| 368q hop verification | `data/eval_1000q/ground-truth/hop_verification_results.json` |

Verify a row: `jq '[.judge_correct] | add' <file>`.
