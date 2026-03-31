# Cross-System Judge EM Comparison

All systems judged with gemma3:12b via Ollama (consistent judge).
Metrics from `evaluation/metrics.py`. Judge EM = semantic equivalence.

## Table 1: Judge EM on Hop-368 (KG-Answerable Subset)

| System    | N   | Correct | Judge EM% | 95% CI         |
| :-------- | ---: | ---:--- | ---:----- | ---:---------- |
| Hybrid    | 368 | 324     | 88.0%     | [84.3%, 91.0%] |
| LogicRAG  | 368 | 272     | 73.9%     | [69.2%, 78.1%] |
| SPARQL    | 368 | 337     | 91.6%     | [88.3%, 94.0%] |
| GRASP     | 368 | 291     | 79.1%     | [74.6%, 82.9%] |
| LinearRAG | 368 | 324     | 88.0%     | [84.3%, 91.0%] |

## Table 2: Judge EM on Full 1000q

Only systems evaluated on all 1000 questions shown.

| System    | N    | Correct | Judge EM% | 95% CI         |
| :-------- | ---: | ---:--- | ---:----- | ---:---------- |
| Hybrid    | 1000 | 853     | 85.3%     | [83.0%, 87.4%] |
| LogicRAG  | 1000 | 745     | 74.5%     | [71.7%, 77.1%] |
| LinearRAG | 1000 | 840     | 84.0%     | [81.6%, 86.1%] |

## Table 3: Judge EM by Question Type (Hop-368)

| System    | bridge_comparison | comparison | compositional | inference   |
| :-------- | ---:------------- | ---:------ | ---:--------- | ---:------- |
| Hybrid    | 90.8% (141)       | 90.0% (80) | 82.3% (113)   | 91.2% (34)  |
| LogicRAG  | 73.0% (141)       | 91.2% (80) | 56.6% (113)   | 94.1% (34)  |
| SPARQL    | 97.2% (141)       | 92.5% (80) | 81.4% (113)   | 100.0% (34) |
| GRASP     | 91.5% (141)       | 85.0% (80) | 55.8% (113)   | 91.2% (34)  |
| LinearRAG | 97.2% (141)       | 96.2% (80) | 70.8% (113)   | 88.2% (34)  |

## Table 4: Judge EM by Question Type (Full 1000q)

| System    | bridge_comparison | comparison  | compositional | inference   |
| :-------- | ---:------------- | ---:------- | ---:--------- | ---:------- |
| Hybrid    | 86.0% (235)       | 91.4% (244) | 79.9% (413)   | 90.7% (108) |
| LogicRAG  | 72.8% (235)       | 92.6% (244) | 61.7% (413)   | 86.1% (108) |
| LinearRAG | 93.6% (235)       | 96.7% (244) | 71.9% (413)   | 80.6% (108) |

## Table 5: McNemar Pairwise Tests — Judge EM (Hop-368)

Bonferroni-corrected alpha=0.0083 (6 comparisons).

| Comparison          | N   | A only | B only | Net | Chi2  | Significant |
| :------------------ | ---: | ---:-- | ---:-- | ---: | ---:- | ---:------- |
| Hybrid vs LinearRAG | 368 | 32     | 32     | +0  | 0.02  | No          |
| Hybrid vs LogicRAG  | 368 | 79     | 27     | -52 | 24.54 | Yes         |
| Hybrid vs SPARQL    | 368 | 13     | 26     | +13 | 3.69  | No          |
| Hybrid vs GRASP     | 368 | 61     | 28     | -33 | 11.51 | Yes         |
| SPARQL vs GRASP     | 368 | 63     | 17     | -46 | 25.31 | Yes         |
| SPARQL vs LinearRAG | 368 | 31     | 18     | -13 | 2.94  | No          |

## Table 6: McNemar Pairwise Tests — Judge EM (Full 1000q)

Bonferroni-corrected alpha=0.0167 (3 comparisons).

| Comparison            | N    | A only | B only | Net  | Chi2  | Significant |
| :-------------------- | ---: | ---:-- | ---:-- | ---: | ---:- | ---:------- |
| Hybrid vs LinearRAG   | 1000 | 100    | 87     | +13  | 0.77  | No          |
| Hybrid vs LogicRAG    | 1000 | 173    | 65     | +108 | 48.11 | Yes         |
| LogicRAG vs LinearRAG | 1000 | 69     | 164    | -95  | 37.92 | Yes         |

## Methodology

- **Judge**: gemma3:12b via Ollama (local, same model for all systems)
- **LLM**: llama-3.3-70b-versatile via Groq (all systems except GRASP)
- **GRASP**: uses openai/gpt-5-mini (external system, LLM not controllable)
- **Metrics**: EM, F1, containment from `evaluation/metrics.py`
- **Hop-368**: questions where all properties exist in DBpedia (KG-answerable subset)
- **McNemar**: chi-squared with continuity correction, Bonferroni alpha=0.0083 for 6 tests
- **Wilson CI**: 95% confidence intervals for proportions
