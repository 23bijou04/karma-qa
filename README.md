# KARMA

A hybrid multi-hop QA system combining DBpedia SPARQL retrieval, Wikipedia text retrieval, and LLM-driven sub-question decomposition. Evaluated on 2WikiMultiHopQA, HotPotQA, and MuSiQue.

## Key Results

Evaluated on 1000-question subsets (from the [LinearRAG](https://github.com/Zly0523/linear-rag) evaluation set). All systems judged with the same LLM judge (gemma3:12b) for fair comparison. **Bold = column maximum.**

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
| LogicRAG  | 72.8% (171/235) | 92.6% (226/244) | 61.7% (255/413) | 86.1% (93/108) |

## Requirements

- Python 3.10+ on macOS or Linux
- An LLM API key — set `GROQ_API_KEY` ([free tier](https://console.groq.com)) or `GITHUB_TOKEN`
- [Ollama](https://ollama.com) (for the `gemma3:12b` judge — see Install step 3)

## Install

```bash
# 1. Python deps
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,rag]"
python -m spacy download en_core_web_sm

# 2. Embedding models (~100 MB each, one-time download)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 3. LLM judge (Ollama + gemma3:12b, ~8 GB)
# macOS: brew install ollama && brew services start ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh && ollama serve &
ollama pull gemma3:12b

# 4. Benchmark datasets (bundled with the LogicRAG baseline repo)
git clone https://github.com/chensyCN/LogicRAG.git baselines/LogicRAG
```

## Data

Eval scripts read questions + corpora from `baselines/LogicRAG/dataset/` (populated by step 4 above):

| Benchmark | Questions | Corpus | Upstream source |
|---|---|---|---|
| 2WikiMultiHopQA | `baselines/LogicRAG/dataset/2wikimultihopqa.json` | `2wikimultihopqa_corpus.json` | [Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop) |
| HotPotQA | `baselines/LogicRAG/dataset/hotpotqa.json` | `hotpotqa_corpus.json` | [hotpotqa.github.io](https://hotpotqa.github.io/) |
| MuSiQue | `baselines/LogicRAG/dataset/musique.json` | `musique_corpus.json` | [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique) |

The 1000-question subset manifest (`data/eval_1000q/ground-truth/coverage_analysis_1000.json`) is already tracked in-repo — no download needed. It selects which question IDs to evaluate.

## Run

```bash
export GROQ_API_KEY="..."   # or set GITHUB_TOKEN and pass --provider github

# Quick check: answer a single question
python hybrid_qa.py "Who directed Inception?" --verbose

# Paper reproduction: 2WikiMultiHopQA 1000q batch eval
# (~2 h of LLM calls; longer with rate-limit waits — auto-resumes on 429s)
python scripts/hybrid_batch_eval.py --api-key $GROQ_API_KEY

# Paper reproduction: cross-benchmark
python scripts/run-hybrid-cross-benchmark.py --benchmark hotpotqa --api-key $GROQ_API_KEY
python scripts/run-hybrid-cross-benchmark.py --benchmark musique  --api-key $GROQ_API_KEY

# Score predictions with the LLM judge (gemma3:12b via Ollama)
python scripts/judge_hybrid_eval.py data/eval_1000q/hybrid/hybrid_eval_1000.jsonl
```

## License

MIT — see [LICENSE](LICENSE).
