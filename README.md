# DBpedia SchemaRAG

SchemaRAG over DBpedia + Wikipedia for 2WikiMultiHopQA, HotPotQA, MuSiQue.

## Key Results

Evaluated on 1000-question subsets (from [LinearRAG](https://github.com/Zly0523/linear-rag) evaluation set). All systems judged with the same LLM judge (gemma3:12b) for fair comparison. 95% CIs are Wilson intervals.

### Full 1000q (2WikiMultiHopQA, all questions)

| System | Judge EM | 95% CI |
|--------|----------|--------|
| **SchemaRAG (ours)** | **89.4%** (894/1000) | [87.3%, 91.2%] |
| LinearRAG | 84.0% (840/1000) | [81.6%, 86.1%] |
| LogicRAG | 74.5% (745/1000) | [71.7%, 77.1%] |

### Hop-Verified 368q (KG-Answerable Subset, 2WikiMultiHopQA)

368 questions where DBpedia has all required predicates at every reasoning hop.

| System | Judge EM | 95% CI |
|--------|----------|--------|
| **SchemaRAG (ours)** | **92.9%** (342/368) | [89.8%, 95.1%] |
| LinearRAG | 88.0% (324/368) | [84.3%, 91.0%] |
| LogicRAG | 73.9% (272/368) | [69.2%, 78.1%] |

### Cross-Benchmark (1000q each)

| System    | 2WikiMultiHopQA | HotPotQA | MuSiQue |
|-----------|:-:|:-:|:-:|
| **SchemaRAG** | **89.4%** | 86.1% | **65.6%** |
| LinearRAG | 84.0% | **86.9%** | 51.6% |
| LogicRAG  | 74.5% | 78.8% | 50.6% |

### By Question Type (2WikiMultiHopQA, 1000q)

| System | bridge_comparison | comparison | compositional | inference |
|--------|:-:|:-:|:-:|:-:|
| **SchemaRAG** | 91.9% (216/235) | 95.9% (234/244) | **82.6%** (341/413) | **95.4%** (103/108) |
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

# Clone baselines used by evaluation scripts
git clone https://github.com/chensyCN/LogicRAG.git baselines/LogicRAG
git clone https://github.com/Zly0523/linear-rag.git baselines/LinearRAG
# LinearRAG is self-contained with its own venv; see baselines/LinearRAG/REPRODUCE.md

# LLM judge (gemma3:12b via Ollama)
# macOS: brew install ollama && brew services start ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh && systemctl --user start ollama
ollama pull gemma3:12b   # ~8 GB, one-time
ollama list              # verify gemma3:12b is present
```

Key deps (declared in `pyproject.toml`): `groq`, `openai`, `sentence-transformers`, `faiss-cpu`, `spacy`, `tenacity`, `ddgs`, `websockets`, `huggingface_hub`.

## Data

Benchmark question files are gitignored (large). Download into `data/`:

| Benchmark | Source | Expected path |
|---|---|---|
| 2WikiMultiHopQA | <https://github.com/Alab-NII/2wikimultihop> | `data/dev_full.json` |
| HotPotQA | <http://hotpotqa.github.io/> | `data/cross_benchmark/hotpotqa/dev.json` |
| MuSiQue | <https://github.com/StonyBrookNLP/musique> | `data/cross_benchmark/musique/dev.json` |

The 1000q subset manifest (`data/eval_1000q/ground-truth/coverage_analysis_1000.json`) and 368q hop verification (`hop_verification_results.json`) are tracked in-repo.

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

### Baselines

```bash
# LogicRAG — 2WikiMultiHopQA + cross-benchmark
python scripts/run-logicrag-1000q.py --api-key $GROQ_API_KEY
python scripts/run-logicrag-cross-benchmark.py --benchmark hotpotqa --api-key $GROQ_API_KEY
python scripts/run-logicrag-cross-benchmark.py --benchmark musique  --api-key $GROQ_API_KEY

# LinearRAG — uses its own venv (see baselines/LinearRAG/REPRODUCE.md)
# Run on 2Wiki, then HotPotQA + MuSiQue per the same REPRODUCE.md instructions.
# After running natively, convert predictions to our schema:
python scripts/convert-linearrag-predictions.py
```

LLM settings: `temperature=0`, exponential-backoff retry on 429s. Groq routes across heterogeneous accelerators — expect ±0.3pp drift across reruns.

## Result files

Headline (canonical run = run 1, judge = gemma3:12b):
- **SchemaRAG 89.4%** (894/1000) on 2WikiMultiHopQA
- **86.1%** (861/1000) on HotPotQA
- **65.6%** (656/1000) on MuSiQue
- **92.9%** (342/368) on the hop-verified KG-answerable subset

| Result | JSONL |
|---|---|
| SchemaRAG 1000q (judged) | `data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl` |
| SchemaRAG 368q hop-verified (judged) | `data/eval_1000q/hybrid/hybrid_hop368_judged.jsonl` |
| LinearRAG 1000q (judged) | `data/eval_1000q/linearrag/linearrag_1000q_judged.jsonl` |
| LogicRAG 1000q (judged) | `data/eval_1000q/logicrag/logicrag_1000q_traced_judged.jsonl` |
| SchemaRAG HotPotQA | `data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged.jsonl` |
| SchemaRAG MuSiQue | `data/cross_benchmark/musique/hybrid_musique_1000q_judged.jsonl` |
| LinearRAG HotPotQA | `data/cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_judged.jsonl` |
| LinearRAG MuSiQue | `data/cross_benchmark/musique/linearrag_musique_1000q_judged.jsonl` |
| LogicRAG HotPotQA | `data/cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_judged.jsonl` |
| LogicRAG MuSiQue | `data/cross_benchmark/musique/logicrag_musique_1000q_judged.jsonl` |
| 1000q manifest | `data/eval_1000q/ground-truth/coverage_analysis_1000.json` |
| 368q hop verification | `data/eval_1000q/ground-truth/hop_verification_results.json` |

Files suffixed `_run2_judged.jsonl` and `_judged_claude_sonnet.jsonl` are robustness reruns (different seed / different judge); the unsuffixed `_judged.jsonl` files are canonical.

Verify a row: `jq '[.judge_correct] | add' <file>`.

## Repo layout

See paper §3 for the method. Entry point: `hybrid_qa.py`. Baselines in `baselines/`. Evaluation scripts in `scripts/`. KG client in `dbpedia/`. Retrieval in `rag/`. Metrics in `evaluation/metrics.py`.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

```bibtex
@inproceedings{schemarag2026,
  title  = {SchemaRAG: Schema-Routed Question Answering over DBpedia and Wikipedia for Multi-Hop QA},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```
