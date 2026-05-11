# Documentation Index

Paper-supporting documentation for the DBpedia KARMA system.

---

## Paper Artifacts

| File | Description |
|------|-------------|
| [paper-supplementary.tex](paper-supplementary.tex) | Supplementary material (LaTeX) for the paper |

---

## Data Files Reference

Evaluation results live in `data/eval_1000q/` and `data/cross_benchmark/`:

```
data/
├── eval_1000q/                          # 2WikiMultiHopQA 1000q
│   ├── ground-truth/                    # coverage + hop verification
│   ├── hybrid/hybrid_eval_1000*.jsonl
│   ├── linearrag/linearrag_1000q*.jsonl
│   ├── logicrag/logicrag_1000q_traced*.jsonl
│   ├── sparql/hop368/                   # SPARQL component on 368q
│   └── unified_comparison_1000.json
└── cross_benchmark/
    ├── hotpotqa/                        # 3 systems on HotPotQA 1000q
    └── musique/                         # 3 systems on MuSiQue 1000q
```
