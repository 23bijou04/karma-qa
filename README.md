# DBpedia KARMA

KARMA over DBpedia + Wikipedia for 2WikiMultiHopQA, HotPotQA, MuSiQue.

## Key Results

Evaluated on 1000-question subsets (from [LinearRAG](https://github.com/Zly0523/linear-rag) evaluation set). All systems judged with the same LLM judge (gemma3:12b) for fair comparison.

### Cross-Benchmark (1000q each)

| System    | 2WikiMultiHopQA | HotPotQA | MuSiQue |
|-----------|:-:|:-:|:-:|
| **KARMA** | **89.4%** | 86.1% | **65.6%** |
| LinearRAG | 84.0% | **86.9%** | 51.6% |
| LogicRAG  | 74.5% | 78.8% | 50.6% |

### By Question Type (2WikiMultiHopQA, 1000q)

| System | bridge_comparison | comparison | compositional | inference |
|--------|:-:|:-:|:-:|:-:|
| **KARMA** | 91.9% (216/235) | 95.9% (234/244) | **82.6%** (341/413) | **95.4%** (103/108) |
| LinearRAG | **93.6%** (220/235) | **96.7%** (236/244) | 71.9% (297/413) | 80.6% (87/108) |
| LogicRAG | 72.8% (171/235) | 92.6% (226/244) | 61.7% (255/413) | 86.1% (93/108) |

Bold = column maximum.

## Requirements

- Python 3.10+
- macOS or Linux (uses `signal.SIGALRM`)
- `GROQ_API_KEY` (free tier: <https://console.groq.com>) **or** `GITHUB_TOKEN`
- Ollama + `gemma3:12b` — required to reproduce headline Judge EM numbers

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]" ".[rag]"
python -m spacy download en_core_web_sm

# Pre-cache embedding models (~100 MB each)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# LLM judge (gemma3:12b via Ollama)
# macOS: brew install ollama && brew services start ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh && systemctl --user start ollama
ollama pull gemma3:12b   # ~8 GB, one-time
```

## Data

Benchmark question files are gitignored (large). Download into `data/`:

| Benchmark | Source | Expected path |
|---|---|---|
| 2WikiMultiHopQA | <https://github.com/Alab-NII/2wikimultihop> | `data/dev_full.json` |
| HotPotQA | <http://hotpotqa.github.io/> | `data/cross_benchmark/hotpotqa/dev.json` |
| MuSiQue | <https://github.com/StonyBrookNLP/musique> | `data/cross_benchmark/musique/dev.json` |

## Run

```bash
export GROQ_API_KEY="..."   # or GITHUB_TOKEN with --provider github

# Single question (interactive)
python hybrid_qa.py "Who directed Inception?" --verbose

# 2WikiMultiHopQA 1000q batch eval (~5 h on Groq free tier; auto-resumes on 429s)
python scripts/hybrid_batch_eval.py --api-key $GROQ_API_KEY

# Cross-benchmark
python scripts/run-hybrid-cross-benchmark.py --benchmark hotpotqa --api-key $GROQ_API_KEY
python scripts/run-hybrid-cross-benchmark.py --benchmark musique  --api-key $GROQ_API_KEY

# Score with the LLM judge (gemma3:12b via Ollama)
python scripts/judge_hybrid_eval.py data/eval_1000q/hybrid/hybrid_eval_1000.jsonl
```

## License

MIT — see [`LICENSE`](LICENSE).
