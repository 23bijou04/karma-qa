# Evaluation Report: Schema-Guided SPARQL vs GRASP (Hop-Verified 368q)

**Date:** 2026-03-30 (updated to hop-verified 368q subset)
**Dataset:** 2WikiMultiHopQA dev split — 1000-question subset (from LinearRAG evaluation set, Zly0523/linear-rag)
**Evaluation subset:** 368 hop-verified DBpedia-answerable questions (all predicates exist per hop, ground truth from `scripts/verify_hop_coverage.py`)
**Judge model:** Ollama gemma3:12b (same model for both systems)

> **Note:** Previous version used 346q subset. This update uses the hop-verified 368q ground truth. Both systems are now evaluated on the full 368q set with the same judge model.

---

## 1. Systems Under Comparison

| System | Backend LLM | Approach | Latency |
|---|---|---|---|
| **Schema-Guided SPARQL** | Groq llama-3.3-70b-versatile | Pre-discovered schema → single-shot SPARQL + 1 retry | ~5s/question |
| **GRASP** | OpenAI gpt-5-mini | Agentic exploration → multi-step tool calls → SPARQL | ~72s/question |

---

## 2. Overall Results

### 2.1 Full Set (n=368 for both systems)

| Metric | Schema-Guided (n=368) | GRASP (n=368) | Delta |
|---|:---:|:---:|:---:|
| **EM** | **68.5%** | 32.1% | +36.4 pp |
| **F1** | **81.3%** | 47.3% | +34.0 pp |
| **Judge EM** | **91.6%** | 79.1% | +12.5 pp |
| **Containment** | **86.1%** | 85.6% | +0.5 pp |
| **Answered** | 366/368 (99.5%) | 355/368 (96.5%) | — |
| **Errors** | 0 | 0 | — |

---

## 3. Results by Question Type

### 3.1 Exact Match (EM)

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 141 | **94.3%** | 55.3% | +39.0 pp |
| comparison | 80 | **87.5%** | 50.0% | +37.5 pp |
| compositional | 113 | **22.1%** | 0.0% | +22.1 pp |
| inference | 34 | **70.6%** | 0.0% | +70.6 pp |

### 3.2 F1 Score

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 141 | **96.2%** | 73.0% | +23.2 pp |
| comparison | 80 | **90.4%** | 62.9% | +27.5 pp |
| compositional | 113 | **53.1%** | 6.8% | +46.3 pp |
| inference | 34 | **91.9%** | 38.4% | +53.5 pp |

### 3.3 Containment

| Type | n | Schema-Guided | GRASP |
|---|:---:|:---:|:---:|
| bridge_comparison | 141 | 95.0% | **95.7%** |
| comparison | 80 | **92.5%** | 87.5% |
| compositional | 113 | 71.7% | **73.5%** |
| inference | 34 | **82.4%** | 79.4% |

### 3.4 Judge EM

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 141 | **97.2%** | 91.5% | +5.7 pp |
| comparison | 80 | **92.5%** | 85.0% | +7.5 pp |
| compositional | 113 | **81.4%** | 55.8% | +25.6 pp |
| inference | 34 | **100.0%** | 91.2% | +8.8 pp |

### 3.5 Judge EM vs Auto EM Gap (per system)

| Type | Schema-Guided EM | Schema-Guided Judge | Gap | GRASP EM | GRASP Judge | Gap |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| bridge_comparison | 94.3% | 97.2% | 2.8 pp | 55.3% | 91.5% | 36.2 pp |
| comparison | 87.5% | 92.5% | 5.0 pp | 50.0% | 85.0% | 35.0 pp |
| compositional | 22.1% | 81.4% | 59.3 pp | 0.0% | 55.8% | 55.8 pp |
| inference | 70.6% | 100.0% | 29.4 pp | 0.0% | 91.2% | 91.2 pp |

**Key insight:** Both systems suffer from answer extraction gaps, but in different ways:
- **Schema-Guided** has small gaps on comparison/bridge (2.8-5.0 pp) but a massive 59 pp gap on compositional — the SPARQL returns correct data but string normalization fails.
- **GRASP** has large gaps across all types (35.0-91.2 pp) — its narrative answer format consistently defeats extraction heuristics. The 91.2 pp gap on inference is the largest: GRASP gets the right person 91.2% of the time but EM captures 0% due to URI suffixes and label variants.

---

## 4. Head-to-Head Analysis (EM, n=368)

| Outcome | Count | % |
|---|:---:|:---:|
| Both correct | 112 | 30.4% |
| Both wrong | 110 | 29.9% |
| SPARQL only correct | 140 | 38.0% |
| GRASP only correct | 6 | 1.6% |

### By Type

| Type | Both OK | Both Fail | SPARQL Only | GRASP Only |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 74 | 4 | 59 | 4 |
| comparison | 38 | 8 | 32 | 2 |
| compositional | 0 | 88 | 25 | 0 |
| inference | 0 | 10 | 24 | 0 |

**Key finding:** GRASP achieves EM=1 on only 6 questions where Schema-Guided fails. On compositional and inference types, GRASP never outperforms Schema-Guided (0 GRASP-only wins).

---

## 5. GRASP Containment Paradox

GRASP frequently embeds the correct answer within verbose explanations but fails exact match.

| Type | EM | Containment | Gap | Interpretation |
|---|:---:|:---:|:---:|---|
| bridge_comparison | 55.3% | 95.7% | **40.4 pp** | Answer present but wrapped in explanation |
| comparison | 50.0% | 87.5% | **37.5 pp** | Answer present but wrapped in explanation |
| compositional | 0.0% | 73.5% | **73.5 pp** | Gold answer in raw output, extraction fails |
| inference | 0.0% | 79.4% | **79.4 pp** | Gold answer in raw output, extraction fails |

---

## 6. GRASP Failure Analysis: Compositional (n=113, EM=0.0%)

### 6.1 The Problem

GRASP generates valid SPARQL and retrieves correct data, but produces narrative answers where extraction selects the wrong part.

**Example:**
```
Question: "Where was the director of film Bollywood Calling born?"
Gold:     "Hyderabad"

GRASP raw: "The director of Bollywood Calling is Nagesh Kukunoor
            (dbr:Nagesh_Kukunoor). DBpedia lists his birthplace
            as dbr:Hyderabad."
Extracted: "Nagesh Kukunoor"  ← intermediate entity, not final answer
EM: 0.0  |  Containment: 1.0
```

### 6.2 Root Cause Breakdown (from verified trace analysis)

| Failure Mode | Count | % of Failures |
|---|:---:|:---:|
| **URI_SUFFIX** — `(dbr:Entity)` not stripped | 61 | 44.9% |
| **SENTENCE_SPLIT** — gold in sentence 2+, extraction takes sentence 1 | 15 | 11.0% |
| **DELIMITER_SPLIT** — splitting on `;`, `:`, `—` discards gold | 23 | 16.9% |
| **EXTRACTION_UNKNOWN** — case-by-case | 29 | 21.3% |
| **FILLER_REMOVAL** — heuristic removes gold entity | 4 | 2.9% |
| **FORMAT_MISMATCH** — label variant | 4 | 2.9% |

### 6.3 Compositional-Specific (39/53 failures from 276q analysis)

| Sub-Category | Count | % |
|---|:---:|:---:|
| URI_SUFFIX | 14 | 35.9% |
| SENTENCE_SPLIT | 11 | 28.2% |
| DELIMITER_SPLIT | 8 | 20.5% |

**74% of compositional extraction failures** are from URI_SUFFIX or SENTENCE_SPLIT — both fixable.

---

## 7. GRASP Failure Analysis: Inference (n=34, EM=0.0%)

### The Problem

Inference questions ask about family relations (grandfather, mother-in-law, etc.). GRASP answers correctly but includes DBpedia URIs and parenthetical context.

**Examples:**
```
Q: "Who is Felicia Of Roucy's mother-in-law?"
Gold: "Ermesinda of Bigorre"
Pred: "Felicia of Roucy's mother-in-law is Ermesinda of Bigorre (dbr:Ermesinda_of_Bigorre)"
→ EM=0, Containment=1  (URI suffix breaks EM)

Q: "Who is Thomas Stafford's father?"
Gold: "Hugh de Stafford, 2nd Earl of Stafford"
Pred: "Hugh Stafford, 2nd Earl of Stafford (http://dbpedia.org/resource/Hugh_...)"
→ EM=0, Containment=0  (label variant: "de" vs no "de")
```

**Root cause:** Same as compositional — narrative format + URI suffixes + label mismatches.

---

## 8. Why GRASP Succeeds on Comparison Types

### bridge_comparison (EM=55.3%) and comparison (EM=50.0%)

These types ask "Which X is older/younger/born first?" — answers are entity names that appear at the **start** of GRASP's response.

**Example:**
```
Q: "Which film has the director born first, La Mazurka or Land Of Peace?"
Gold: "La Mazurka..."
Pred: "La mazurka del barone, della santa e del fico fiorone"
→ EM=1  (answer IS the first entity in the response)
```

GRASP's extraction succeeds because:
1. The answer entity appears in sentence 1 (not sentence 2)
2. No intermediate entity to confuse extraction
3. Simple entity name without URI suffixes in many cases

---

## 9. Summary Statistics

| Metric | Schema-Guided (n=368) | GRASP (n=368) | Delta |
|---|:---:|:---:|:---:|
| Overall EM | **68.5%** | 32.1% | +36.4 pp |
| Overall F1 | **81.3%** | 47.3% | +34.0 pp |
| Judge EM | **91.6%** | 79.1% | +12.5 pp |
| Containment | **86.1%** | 85.6% | +0.5 pp |
| Best type (EM) | bridge_comparison (94.3%) | bridge_comparison (55.3%) | — |
| Worst type (EM) | compositional (22.1%) | compositional (0.0%) | — |
| Avg latency | ~5s | ~72s | — |
| Runtime errors | 0 | 0 | — |
| Answer rate | 99.5% (366/368) | 96.5% (355/368) | — |

### McNemar Tests (EM, n=368)

| Test | χ² | p-value |
|---|:---:|:---:|
| EM | 121.2 | <0.001 |
| Judge EM | 25.3 | <0.001 (SG only=63, GRASP only=17) |

### Key Takeaways

1. **Schema-Guided dominates on both EM (+36.4 pp) and Judge EM (+12.5 pp)** on the full 368q set. Gap narrows with judge evaluation but remains substantial.
2. **GRASP's 0% EM on compositional/inference is an extraction problem, not a retrieval problem.** Judge EM reveals GRASP gets 55.8% of compositional and 91.2% of inference answers correct — auto EM captures none of this.
3. **Both systems suffer extraction losses, but differently:** Schema-Guided loses 59 pp on compositional (string normalization); GRASP loses 91.2 pp on inference (narrative format + URI suffixes).
4. **Both systems achieve ~85-86% containment overall**, indicating similar underlying retrieval quality — the gap is in answer formatting.
5. **Judge EM closes much of the gap:** EM delta is 36.4 pp, but Judge EM delta is only 12.5 pp. With the same judge model (gemma3:12b), the Judge EM gap is +12.5 pp, substantially narrower than the +17.2 pp gap from the prior cross-judge comparison.
6. **Both systems now evaluated with the same judge model (gemma3:12b)**, eliminating judge-model confounds from the comparison.

---

*Generated from `data/eval_1000q/sparql/hop368/sparql_eval_1000_hop368_judged.jsonl` and `data/eval_1000q/hop368_grasp/grasp_eval_dbpedia_hop368_judged.jsonl` (both judged with gemma3:12b)*
