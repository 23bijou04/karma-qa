# Cross-benchmark routing breakdown

The main paper plots the Q-level and Sub-Q-level routing distributions. The
table below gives the underlying routing counts.

| Metric | 2WikiMHQA | HotpotQA | MuSiQue |
|---|---:|---:|---:|
| Direct SPARQL (%) | 29.5 | 6.1 | 2.4 |
| Decomposition fired (%) | 70.1 | 88.7 | 96.5 |
| Sub-questions generated | 1544 | 1807 | 2508 |
| Sub-SPARQL (%) | 40.5 | 7.6 | 18.0 |
| **Any-SPARQL touch (%)** | **70.0** | **17.4** | **42.6** |

Direct SPARQL means the whole question is answered by Text2SPARQL without
decomposition. Sub-Qs are the total sub-questions produced when decomposition
fires. Sub-SPARQL is the percentage of generated sub-questions resolved
symbolically. Any-SPARQL is the fraction of original questions in which at
least one sub-question hit the SPARQL path.

Routing distributions on 2WikiMHQA and MuSiQue are stable across runs
(any-SPARQL drift <= 2.6%). HotpotQA is more variable (15.4%), reflecting LLM
stochasticity in the coverage check when KG support is sparse.
