# Cross-System Evaluation Comparison

All systems judged with gemma3:12b via Ollama (consistent judge).
Metrics from `evaluation/metrics.py`: EM (exact match after normalization), F1 (token overlap), Containment (gold tokens in prediction), Judge EM (LLM semantic equivalence).

## Table 1: Full Metrics on Hop-368 (KG-Answerable Subset)

| System         | N   | EM%   | F1%   | Cont% | Judge EM% | 95% CI         |
| :------------- | ---:| ----: | ----: | ----: | --------: | :------------- |
| Hybrid         | 368 | 79.9  | 86.8  | 88.0  | 92.9      | [90.3%, 95.6%] |
| SPARQL (comp†) | 368 | 68.5  | 81.3  | 86.1  | 91.6      | [88.3%, 94.0%] |
| LinearRAG      | 368 | 73.9  | 81.9  | 83.4  | 88.0      | [84.3%, 91.0%] |
| GRASP          | 368 | 32.1  | 47.3  | 85.6  | 79.1      | [74.6%, 82.9%] |
| LogicRAG       | 368 | 63.9  | 70.1  | 70.1  | 73.9      | [69.2%, 78.1%] |

† SPARQL = Hybrid's schema-guided text2SPARQL component evaluated in isolation (hop-368 only). Included to show schema projection effectiveness vs GRASP's approach.

## Table 2: Full Metrics on Full 1000q

Only systems evaluated on all 1000 questions shown.

| System    | N    | EM%   | F1%   | Cont% | Judge EM% | 95% CI         |
| :-------- | ---: | ----: | ----: | ----: | --------: | :------------- |
| Hybrid    | 1000 | 72.6  | 80.1  | 82.8  | 89.4      | [87.5%, 91.3%] |
| LinearRAG | 1000 | 64.9  | 73.4  | 76.3  | 84.0      | [81.6%, 86.1%] |
| LogicRAG  | 1000 | 61.1  | 67.5  | 69.2  | 74.5      | [71.7%, 77.1%] |

## Table 3: EM / F1 / Judge EM by Question Type (Hop-368)

| System         | Metric   | bridge_comp (141) | comparison (80) | compositional (113) | inference (34) |
| :------------- | :------- | ----------------: | --------------: | ------------------: | -------------: |
| Hybrid         | EM       | 97.2              | 95.0            | 48.7                | 76.5           |
|                | F1       | 97.2              | 95.0            | 67.0                | 90.9           |
|                | Judge EM | 97.2              | 95.0            | 85.0                | 97.1           |
| SPARQL (comp†) | EM       | 94.3              | 87.5            | 22.1                | 70.6           |
|                | F1       | 96.2              | 90.4            | 53.1                | 91.9           |
|                | Judge EM | 97.2              | 92.5            | 81.4                | 100.0          |
| LinearRAG      | EM       | 96.5              | 95.0            | 36.3                | 55.9           |
|                | F1       | 97.1              | 95.5            | 54.4                | 77.8           |
|                | Judge EM | 97.2              | 96.2            | 70.8                | 88.2           |
| GRASP          | EM       | 55.3              | 50.0            | 0.0                 | 0.0            |
|                | F1       | 73.0              | 62.9            | 6.8                 | 38.4           |
|                | Judge EM | 91.5              | 85.0            | 55.8                | 91.2           |
| LogicRAG       | EM       | 73.0              | 90.0            | 34.5                | 61.8           |
|                | F1       | 73.6              | 91.5            | 46.3                | 84.7           |
|                | Judge EM | 73.0              | 91.2            | 56.6                | 94.1           |

## Table 4: EM / F1 / Judge EM by Question Type (Full 1000q)

| System    | Metric   | bridge_comp (235) | comparison (244) | compositional (413) | inference (108) |
| :-------- | :------- | ----------------: | ---------------: | ------------------: | --------------: |
| Hybrid    | EM       | 91.9              | 95.9             | 45.8                | 80.6            |
|           | F1       | 91.9              | 95.9             | 61.6                | 89.6            |
|           | Judge EM | 91.9              | 95.9             | 82.6                | 95.4            |
| LinearRAG | EM       | 93.2              | 96.3             | 33.7                | 51.9            |
|           | F1       | 93.6              | 96.5             | 48.5                | 72.4            |
|           | Judge EM | 93.6              | 96.7             | 71.9                | 80.6            |
| LogicRAG  | EM       | 72.3              | 91.4             | 36.6                | 62.0            |
|           | F1       | 72.7              | 92.5             | 46.4                | 80.0            |
|           | Judge EM | 72.8              | 92.6             | 61.7                | 86.1            |

## Table 5: McNemar Pairwise Tests — Judge EM (Hop-368)

Bonferroni-corrected alpha=0.0083 (6 comparisons). SPARQL rows use the schema-guided component (†) isolated from Hybrid.

| Comparison                           | N   | A only | B only | Net | Chi2  | Significant |
| :----------------------------------- | ---: | -----: | -----: | ---: | ----: | :---------- |
| Hybrid vs LinearRAG                  | 368 | 33     | 15     | +18 | 6.02  | No          |
| Hybrid vs LogicRAG                   | 368 | 85     | 15     | +70 | 47.61 | Yes         |
| Hybrid (full) vs SPARQL component†   | 368 | 16     | 11     | +5  | 0.59  | No          |
| Hybrid vs GRASP                      | 368 | 66     | 15     | +51 | 30.86 | Yes         |
| SPARQL component† vs GRASP           | 368 | 63     | 17     | +46 | 25.31 | Yes         |
| SPARQL component† vs LinearRAG       | 368 | 31     | 18     | +13 | 2.94  | No          |

## Table 6: McNemar Pairwise Tests — Judge EM (Full 1000q)

Bonferroni-corrected alpha=0.0167 (3 comparisons).

| Comparison            | N    | A only | B only | Net  | Chi2   | Significant |
| :-------------------- | ---: | -----: | -----: | ---: | -----: | :---------- |
| Hybrid vs LinearRAG   | 1000 | 106    | 52     | +54  | 17.78  | Yes         |
| Hybrid vs LogicRAG    | 1000 | 186    | 37     | +149 | 98.22  | Yes         |
| LogicRAG vs LinearRAG | 1000 | 69     | 164    | -95  | 37.92  | Yes         |

## Methodology

- **Judge**: gemma3:12b via Ollama (local, same model for all systems)
- **LLM**: llama-3.3-70b-versatile via Groq (all systems except GRASP)
- **GRASP**: uses openai/gpt-5-mini (external system, LLM not controllable)
- **Metrics**: EM, F1, containment from `evaluation/metrics.py`; Judge EM via LLM semantic equivalence
- **EM gap**: Judge EM − EM difference reflects format mismatches the judge resolves (dates, entity labels, abbreviations). GRASP has the largest gap (47pp on hop-368) due to verbose natural-language answers; Hybrid has a smaller gap (13pp) mostly from compositional questions.
- **Hop-368**: questions where all properties exist in DBpedia (KG-answerable subset)
- **McNemar**: chi-squared with continuity correction, Bonferroni alpha=0.0083 for 6 tests
- **Wilson CI**: 95% confidence intervals for proportions
- **†Schema-Guided SPARQL**: Hybrid's direct SPARQL routing path evaluated in isolation on the hop-368 set. Included to demonstrate that schema projection (injecting entity schema into LLM context) significantly improves text2SPARQL quality vs GRASP's approach. Not an independent competing system — SPARQL is a component of Hybrid. The Hybrid vs SPARQL component McNemar result (chi2=0.59, not significant) confirms that on KG-answerable questions, Hybrid's additional text paths provide minimal incremental gain beyond the SPARQL path alone.
