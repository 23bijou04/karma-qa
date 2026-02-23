# Evaluation Report: Hybrid Pipeline — 100 2-Hop Questions

**Date:** 2026-02-20
**Dataset:** 2WikiMultiHopQA dev split — 100 2-hop questions (stratified sample)
**Pipeline:** Hybrid (rule-based routing + 2-hop chain inference + per-entity SPARQL decomposition)
**LLM:** Groq llama-3.3-70b-versatile (temp=0)
**Judge:** Ollama qwen2.5:7b (semantic equivalence)

---

## 1. Overall Results (n=100)

| Metric | Score |
|---|:---:|
| **Strict EM** | 47.0% |
| **Judge EM** | **60.0%** |
| **F1** | 57.2% |
| **Containment** | 61.0% |
| **Avg Latency** | 7.6s |
| **Errors** | 0/100 |

The LLM judge upgraded **13 answers** from incorrect (strict EM) to correct, a +13 pp gain.

---

## 2. Results by Route

| Route | n | Strict EM | Judge EM | F1 | Containment | Avg Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPARQL** | 48 | 68.8% | **77.1%** | 74.5% | 89.6% | 2.8s |
| **Text** | 52 | 26.9% | **44.2%** | 41.2% | 34.6% | 12.0s |

SPARQL route is ~1.8x more accurate (judge EM) and ~4.3x faster than the text fallback.

---

## 3. Results by Question Type

| Type | n | Strict EM | Judge EM | F1 | Containment |
|---|:---:|:---:|:---:|:---:|:---:|
| **comparison** | 20 | 70.0% | **70.0%** | 70.0% | 85.0% |
| **bridge_comparison** | 19 | 68.4% | **68.4%** | 68.4% | 84.2% |
| **compositional** | 46 | 32.6% | **56.5%** | 47.6% | 45.7% |
| **inference** | 15 | 33.3% | **46.7%** | 55.4% | 46.7% |

Comparison types achieve ~70% accuracy. The judge has the biggest impact on **compositional** questions (+23.9 pp), where partial answers are common.

---

## 4. Results by Type x Route

| Type | Route | n | Strict EM | Judge EM |
|---|---|:---:|:---:|:---:|
| bridge_comparison | sparql | 15 | 80.0% | 80.0% |
| bridge_comparison | text | 4 | 25.0% | 25.0% |
| comparison | sparql | 12 | 75.0% | 75.0% |
| comparison | text | 8 | 62.5% | 62.5% |
| compositional | sparql | 15 | 60.0% | **86.7%** |
| compositional | text | 31 | 19.4% | **41.9%** |
| inference | sparql | 6 | 50.0% | 50.0% |
| inference | text | 9 | 22.2% | **44.4%** |

Notable: **compositional + SPARQL** jumps from 60% to 86.7% with the judge — SPARQL produces correct but slightly differently formatted answers.

---

## 5. Upgraded Examples (EM=0, Judge=yes)

These 13 answers were marked wrong by strict EM but judged semantically correct:

| Question (truncated) | Gold | Predicted | Route |
|---|---|---|---|
| Why did John Middleton Murry's wife die? | tuberculosis | pulmonary tuberculosis | text |
| Where was the director of The Half-Way Girl born? | New York | New York City | sparql |
| Who is Thomas Stafford, 3rd Earl's father? | Hugh de Stafford, 2nd Earl of Stafford | Hugh Stafford, 2nd Earl of Stafford | text |
| What nationality is Beatrice I's husband? | Germany | German | text |
| Where was the director of En Aasai Rasave born? | Theni | Theni district | sparql |
| What nationality is Matilda Of Frisia's husband? | France | French | text |
| Where did Mate Mestrovic's father graduate? | Academy of Fine Arts Vienna | The Academy of Fine Arts in Vienna | text |
| Where was the founder of Track & Field News born? | San Diego | San Diego, California | text |
| Where did the director of Mary Jane's Pa die? | New York | New York City | sparql |
| Where was the father of David Ogilvy born? | Florence, Italy | Florence, Grand Duchy of Tuscany | text |
| Where was the composer of Love Is Here To Stay's death place? | Hollywood | Los Angeles, California | sparql |
| Where did Sholto Douglas's father study? | King's College | King's College, Cambridge | text |
| Who is Teriimaevarua III's paternal grandmother? | Pōmare IV | Pomare IV | text |

Most upgrades come from: **specificity** (city → city+state), **formatting** (diacritics, articles), and **synonyms** (country → nationality adjective).

---

## 6. Key Takeaways

1. **SPARQL route is the strong arm** — 77.1% judge EM at 2.8s/question, with near-perfect containment (89.6%).
2. **2-hop chain inference works** — 48% of 2-hop questions were routed to SPARQL (would have been ~0% without the feature), and those answers are highly accurate.
3. **Per-entity SPARQL decomposition excels at comparison** — ~70% EM for both comparison types.
4. **Text route is the bottleneck** — 44.2% judge EM at 12s/question. Compositional chains through Wikipedia text remain challenging.
5. **Strict EM underestimates by ~13 pp** — semantic equivalence (specificity, formatting, diacritics) accounts for a significant gap.
6. **Zero errors** — the pipeline is robust with proper fallback handling.
