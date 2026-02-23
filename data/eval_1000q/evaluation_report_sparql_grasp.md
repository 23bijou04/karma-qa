# Evaluation Report: Schema-Guided SPARQL vs GRASP (1000q Subset)

**Date:** 2026-02-21
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
| **EM** | **64.5%** | 32.9% | +31.5 pp |
| **F1** | **77.0%** | 48.7% | +28.3 pp |
| **Containment** | 88.7% | **88.2%** | +0.6 pp |
| **Judge EM** | **90.5%** | — | — |
| **Answered** | 346/346 (100%) | 333/346 (96.2%) | +3.8 pp |
| **Errors** | 0 | 0 | — |

---

## 3. Results by Question Type

### 3.1 Exact Match (EM)

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 141 | **86.5%** | 55.3% | +31.2 pp |
| comparison | 71 | **88.7%** | 50.7% | +38.0 pp |
| compositional | 100 | **17.0%** | 0.0% | +17.0 pp |
| inference | 34 | **61.8%** | 0.0% | +61.8 pp |

### 3.2 F1 Score

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 141 | **88.7%** | 73.0% | +15.7 pp |
| comparison | 71 | **90.2%** | 66.0% | +24.3 pp |
| compositional | 100 | **47.6%** | 5.8% | +41.8 pp |
| inference | 34 | **87.2%** | 38.4% | +48.8 pp |

### 3.3 Containment

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 141 | 95.0% | **95.7%** | -0.7 pp |
| comparison | 71 | 95.8% | **94.4%** | +1.4 pp |
| compositional | 100 | 77.0% | **76.0%** | +1.0 pp |
| inference | 34 | 82.4% | **79.4%** | +2.9 pp |

### 3.4 Judge EM (Schema-Guided only)

| Type | n | Judge EM | Auto EM | Gap |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 141 | 92.9% | 86.5% | 6.4 pp |
| comparison | 71 | 95.8% | 88.7% | 7.0 pp |
| compositional | 100 | 80.0% | 17.0% | 63.0 pp |
| inference | 34 | 100.0% | 61.8% | 38.2 pp |

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
| Overall EM | **64.5%** | 32.9% |
| Overall F1 | **77.0%** | 48.7% |
| Containment | **88.7%** | 88.2% |
| Best type (EM) | comparison (88.7%) | bridge_comparison (55.3%) |
| Worst type (EM) | compositional (17.0%) | compositional (0.0%) |
| Avg latency | ~5s | ~72s |
| Runtime errors | 0 | 0 |
| Answer rate | 100% | 96.2% |

### Key Takeaways

1. **Schema-Guided dominates overall** (+31.5 pp EM) due to direct SPARQL result binding extraction vs GRASP's narrative format.
2. **GRASP's 0% EM on compositional/inference is an extraction problem, not a retrieval problem.** Containment shows 76-79% of answers are present in raw output.
3. **Comparison types are GRASP's sweet spot** because answers appear at the start of responses, aligning with extraction heuristics.
4. **Both systems achieve ~88% containment overall**, indicating similar underlying retrieval quality — the gap is in answer formatting.
5. **Potential GRASP improvement:** Fixing URI_SUFFIX extraction alone (44.9% of failures) could lift compositional EM from 0% to ~30%.

---

*Generated from `data/eval_1000q/sparql_eval_1000.jsonl` and `data/eval_1000q/grasp_eval_dbpedia.jsonl`*
