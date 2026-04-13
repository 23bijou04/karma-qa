# Documentation Index

Quick reference for project documentation.

---

## System Documentation

| File | Description |
|------|-------------|
| [hybrid-system-technical-documentation.md](hybrid-system-technical-documentation.md) | Complete technical spec of the hybrid QA system (architecture, modules, data flow) |

## Evaluation Results

| File | Description |
|------|-------------|
| [evaluation_report_sparql_grasp.md](evaluation_report_sparql_grasp.md) | Schema-Guided SPARQL vs GRASP comparison on hop-verified 368q subset |
| [grasp_failure_deep_analysis.md](grasp_failure_deep_analysis.md) | Deep analysis of GRASP system failures |
| [grasp_trace_examples.md](grasp_trace_examples.md) | Annotated GRASP trace examples showing failure modes |

## KG Coverage & Answerability

| File | Description |
|------|-------------|
| [coverage-verification-1000q.md](coverage-verification-1000q.md) | DBpedia coverage verification on the 1000q eval subset (entity, property, hop coverage) |
| [wikimultihop-dbpedia-answerability.md](wikimultihop-dbpedia-answerability.md) | How many 2WikiMultiHopQA questions can be answered with DBpedia SPARQL? |

## Project Management

| File | Description |
|------|-------------|
| [project-changelog.md](project-changelog.md) | Record of features, fixes, improvements |

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
│   ├── grasp/hop368/                    # GRASP on 368q
│   └── unified_comparison_1000.json
└── cross_benchmark/
    ├── hotpotqa/                        # 3 systems on HotPotQA 1000q
    └── musique/                         # 3 systems on MuSiQue 1000q
```
