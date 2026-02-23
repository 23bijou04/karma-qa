# Evaluation Report: Schema-Guided SPARQL vs GRASP (1000q Subset)

**Date:** 2026-02-23 (corrected SPARQL numbers from latest eval)
**Dataset:** 2WikiMultiHopQA dev split — 1000-question stratified sample
**Evaluation subset:** 346 DBpedia-answerable questions (all required properties verified)

---

## 1. Systems Under Comparison

| System | Backend LLM | Approach | Latency |
|---|---|---|---|
| **Schema-Guided SPARQL** | Groq llama-3.3-70b-versatile | Pre-discovered schema → single-shot SPARQL + 1 retry | ~5s/question |
| **GRASP** | OpenAI gpt-5-mini | Agentic exploration → multi-step tool calls → SPARQL | ~72s/question |

---

## 2. Overall Results (n=346)

| Metric | Schema-Guided SPARQL | GRASP | Delta |
|---|:---:|:---:|:---:|
| **EM** | **69.1%** | 32.9% | +36.2 pp |
| **F1** | **83.2%** | 48.7% | +34.5 pp |
| **Judge EM** | **91.9%** | 69.7% | +22.3 pp |
| **Containment** | **88.7%** | 88.2% | +0.6 pp |
| **Answered** | 346/346 (100%) | 333/346 (96.2%) | +3.8 pp |
| **Errors** | 0 | 0 | — |

---

## 3. Results by Question Type

### 3.1 Exact Match (EM)

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 141 | **94.3%** | 55.3% | +39.0 pp |
| comparison | 71 | **85.9%** | 50.7% | +35.2 pp |
| compositional | 100 | **21.0%** | 0.0% | +21.0 pp |
| inference | 34 | **70.6%** | 0.0% | +70.6 pp |

### 3.2 F1 Score

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 141 | **96.2%** | 73.0% | +23.2 pp |
| comparison | 71 | **92.3%** | 66.0% | +26.3 pp |
| compositional | 100 | **55.3%** | 5.8% | +49.5 pp |
| inference | 34 | **91.9%** | 38.4% | +53.5 pp |

### 3.3 Containment

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 141 | 95.0% | **95.7%** | -0.7 pp |
| comparison | 71 | 95.8% | **94.4%** | +1.4 pp |
| compositional | 100 | 77.0% | **76.0%** | +1.0 pp |
| inference | 34 | 82.4% | **79.4%** | +2.9 pp |

### 3.4 Judge EM (via Ollama qwen2.5:7b)

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 141 | **97.2%** | 85.1% | +12.1 pp |
| comparison | 71 | **95.8%** | 83.1% | +12.7 pp |
| compositional | 100 | **81.0%** | 34.0% | +47.0 pp |
| inference | 34 | **94.1%** | 82.4% | +11.8 pp |

### 3.5 Judge EM vs Auto EM Gap (per system)

| Type | Schema-Guided EM | Schema-Guided Judge | Gap | GRASP EM | GRASP Judge | Gap |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| bridge_comparison | 94.3% | 97.2% | 2.8 pp | 55.3% | 85.1% | 29.8 pp |
| comparison | 85.9% | 95.8% | 9.9 pp | 50.7% | 83.1% | 32.4 pp |
| compositional | 21.0% | 81.0% | 60.0 pp | 0.0% | 34.0% | 34.0 pp |
| inference | 70.6% | 94.1% | 23.5 pp | 0.0% | 82.4% | 82.4 pp |

**Key insight:** Both systems suffer from answer extraction gaps, but in different ways:
- **Schema-Guided** has small gaps on comparison/bridge (2.8-9.9 pp) but a massive 60 pp gap on compositional — the SPARQL returns correct data but string normalization fails.
- **GRASP** has large gaps across all types (29.8-82.4 pp) — its narrative answer format consistently defeats extraction heuristics. The 82.4 pp gap on inference is the largest: GRASP gets the right person 82.4% of the time but EM captures 0% due to URI suffixes and label variants.

---

## 4. Head-to-Head Analysis (EM)

| Outcome | Count | % |
|---|:---:|:---:|
| Both correct | 107 | 30.9% |
| Both wrong | 100 | 28.9% |
| SPARQL only correct | 132 | 38.2% |
| GRASP only correct | 7 | 2.0% |

### By Type

| Type | Both OK | Both Fail | SPARQL Only | GRASP Only |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 74 | 4 | 59 | 4 |
| comparison | 33 | 7 | 28 | 3 |
| compositional | 0 | 79 | 21 | 0 |
| inference | 0 | 10 | 24 | 0 |

**Key finding:** GRASP achieves EM=1 on only 7 questions where Schema-Guided fails. On compositional and inference types, GRASP never outperforms Schema-Guided (0 GRASP-only wins).

---

## 5. GRASP Containment Paradox

GRASP frequently embeds the correct answer within verbose explanations but fails exact match.

| Type | EM | Containment | Gap | Interpretation |
|---|:---:|:---:|:---:|---|
| bridge_comparison | 55.3% | 95.7% | **40.4 pp** | Answer present but wrapped in explanation |
| comparison | 50.7% | 94.4% | **43.7 pp** | Answer present but wrapped in explanation |
| compositional | 0.0% | 76.0% | **76.0 pp** | Gold answer in raw output, extraction fails |
| inference | 0.0% | 79.4% | **79.4 pp** | Gold answer in raw output, extraction fails |

---

## 6. GRASP Failure Analysis: Compositional (n=100, EM=0.0%)

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

### bridge_comparison (EM=55.3%) and comparison (EM=50.7%)

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

| Metric | Schema-Guided | GRASP |
|---|:---:|:---:|
| Total questions | 346 | 346 |
| Overall EM | **69.1%** | 32.9% |
| Overall F1 | **83.2%** | 48.7% |
| Judge EM | **91.9%** | 69.7% |
| Containment | **88.7%** | 88.2% |
| Best type (EM) | bridge_comparison (94.3%) | bridge_comparison (55.3%) |
| Worst type (EM) | compositional (21.0%) | compositional (0.0%) |
| Avg latency | ~5s | ~72s |
| Runtime errors | 0 | 0 |
| Answer rate | 100% | 96.2% |

### Key Takeaways

1. **Schema-Guided dominates on both EM (+36.2 pp) and Judge EM (+22.3 pp).** The gap narrows with judge evaluation but remains substantial.
2. **GRASP's 0% EM on compositional/inference is an extraction problem, not a retrieval problem.** Judge EM reveals GRASP actually gets 34.0% of compositional and 82.4% of inference answers correct — auto EM captures none of this.
3. **Both systems suffer extraction losses, but differently:** Schema-Guided loses 60 pp on compositional (string normalization); GRASP loses 82.4 pp on inference (narrative format + URI suffixes).
4. **Both systems achieve ~88% containment overall**, indicating similar underlying retrieval quality — the gap is in answer formatting.
5. **Judge EM closes much of the gap:** EM delta is 36.2 pp, but Judge EM delta is only 22.3 pp. With better extraction on both sides, the systems are closer than raw EM suggests.

---

*Generated from `data/eval_1000q/sparql_eval_1000.jsonl` and `data/eval_1000q/grasp_eval_dbpedia.jsonl`*
