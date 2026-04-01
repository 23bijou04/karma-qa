# Documentation Index

Quick reference for all documentation in this project. Files grouped by topic.

---

## System Documentation

| File | Description |
|------|-------------|
| [hybrid-system-technical-documentation.md](hybrid-system-technical-documentation.md) | Complete technical spec of the hybrid QA system (architecture, modules, data flow) |
| [logicrag-vs-our-retrieval.md](logicrag-vs-our-retrieval.md) | Comparison of LogicRAG baseline retrieval vs our system's retrieval approach |

## Evaluation Results

| File | Description |
|------|-------------|
| [evaluation_report_sparql_grasp.md](evaluation_report_sparql_grasp.md) | Schema-Guided SPARQL vs GRASP comparison on hop-verified 368q subset (updated 2026-03-30) |
| [hybrid-vs-logicrag-100q-comparison.md](hybrid-vs-logicrag-100q-comparison.md) | Hybrid vs LogicRAG 100-question head-to-head with judge results |
| [hybrid-vs-logicrag-performance-analysis.md](hybrid-vs-logicrag-performance-analysis.md) | Code-and-data-grounded analysis of why hybrid outperforms LogicRAG |
| [answer-source-analysis.md](answer-source-analysis.md) | Analysis of where correct answers come from (KG vs text retrieval) |

## KG Coverage & Answerability

| File | Description |
|------|-------------|
| [coverage-verification-1000q.md](coverage-verification-1000q.md) | DBpedia coverage verification on the 1000q eval subset (entity, property, hop coverage) |
| [wikimultihop-dbpedia-answerability.md](wikimultihop-dbpedia-answerability.md) | How many 2WikiMultiHopQA questions can be answered with DBpedia SPARQL? |
| [kg-coverage-scenarios.md](kg-coverage-scenarios.md) | Classification of 1000q by KG chain break point (scenarios A-F) |
| [missing-hop1-predicate-investigation.md](missing-hop1-predicate-investigation.md) | Case study: missing hop-1 predicate investigation |

## Entity Linking

| File | Description |
|------|-------------|
| [ENTITY-LINKING-DOCS-INDEX.md](ENTITY-LINKING-DOCS-INDEX.md) | Navigation index for entity linking docs and stress tests |
| [entity-linking-quality-assurance.md](entity-linking-quality-assurance.md) | Multi-layered QA approach for entity linking robustness |
| [entity-linking-stress-test-catalog.md](entity-linking-stress-test-catalog.md) | 106 stress test cases across 10 categories |
| [entity-linking-stress-test-guide.md](entity-linking-stress-test-guide.md) | How to run and extend entity linking stress tests |

## Failure Analysis

| File | Description |
|------|-------------|
| [benchmark-errors-2wikimultihopqa.md](benchmark-errors-2wikimultihopqa.md) | Confirmed factual errors in benchmark questions/gold answers |
| [wikipedia-link-extraction-cases.md](wikipedia-link-extraction-cases.md) | Failure case taxonomy for Wikipedia link extraction in text fallback |
| [260317-text-retrieval-rag-edge-case-analysis.md](260317-text-retrieval-rag-edge-case-analysis.md) | RAG subsystem edge case analysis (chunker, vector store, reranker) |
| [grasp_failure_deep_analysis.md](grasp_failure_deep_analysis.md) | Deep analysis of GRASP system failures |
| [grasp_trace_examples.md](grasp_trace_examples.md) | Annotated GRASP trace examples showing failure modes |

## Paper Writing

| File | Description |
|------|-------------|
| [paper-section-baseline-comparison.tex](paper-section-baseline-comparison.tex) | LaTeX draft: baseline comparison section (updated to hop368 numbers, 2026-03-30) |

## Project Management

| File | Description |
|------|-------------|
| [daily-log.md](daily-log.md) | Daily progress tracking |
| [project-changelog.md](project-changelog.md) | Comprehensive record of features, fixes, improvements |

---

## `paper-notes/` — Consolidated Research Notes

Consolidated findings from investigation sessions, verified against code/data.

| File | Description |
|------|-------------|
| [coverage-and-evaluation.md](paper-notes/coverage-and-evaluation.md) | KG coverage stats, system eval results, key numbers for paper |
| [entity-linking-analysis.md](paper-notes/entity-linking-analysis.md) | Entity linking pipeline analysis |
| [failure-analysis-by-scenario.md](paper-notes/failure-analysis-by-scenario.md) | Failure analysis per KG coverage scenario |
| [logicrag-analysis.md](paper-notes/logicrag-analysis.md) | LogicRAG decomposition engine analysis |
| [research-notes.md](paper-notes/research-notes.md) | Paper-relevant research findings |
| [sparql-pipeline-analysis.md](paper-notes/sparql-pipeline-analysis.md) | SPARQL pipeline analysis (baseline vs hybrid comparison) |

---

## Data Files Reference

Evaluation results live in `data/eval_1000q/`:

```
data/eval_1000q/
├── sparql/hop368/          # SPARQL component (Hybrid's SPARQL path) on hop-verified 368q
│   ├── sparql_eval_1000_hop368.jsonl
│   ├── sparql_eval_1000_hop368_judged.jsonl
│   └── sparql_eval_1000_hop368_summary.json
├── grasp/hop368/           # GRASP baseline on hop-verified 368q
│   ├── grasp_eval_dbpedia_hop368.jsonl
│   └── grasp_eval_dbpedia.jsonl
├── hybrid_eval_1000.jsonl  # Hybrid system full 1000q
├── coverage_analysis_1000.json
└── hop_verification_results.json  # Ground truth (script: verify_hop_coverage.py)
```
