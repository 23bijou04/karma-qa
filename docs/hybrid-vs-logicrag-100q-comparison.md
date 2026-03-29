# Hybrid vs LogicRAG Baseline: 100-Question Head-to-Head Comparison

**Date:** 2026-03-26
**Sample:** First 100 questions from `data/eval_1000q/` (same question set for both systems)
**Judge:** gemma3:12b via Ollama (local), with word-level containment fast path
**Data:** `hybrid_eval_1000_judged.jsonl` (Hybrid), `logicrag_1000q_traced_judged.jsonl` (LogicRAG)

> **Version provenance (2026-03-28):** This comparison used file versions from 2026-03-26 (both `.bak_v1`). Both JSONL files have since been re-judged (Hybrid: Mar 28, LogicRAG: Mar 27). Numbers below reflect the Mar 26 snapshot and may differ with current file versions. A re-comparison with current versions is pending.

> For architectural differences between the two retrieval systems, see [logicrag-vs-our-retrieval.md](logicrag-vs-our-retrieval.md).

---

## Overall Results

| Metric | Hybrid | LogicRAG | Delta |
|--------|--------|----------|-------|
| **Judge Correct** | **89%** (89/100) | 80% (80/100) | **+9pp** |
| **Exact Match** | **67%** (67/100) | 62% (62/100) | **+5pp** |
| **F1** | **78.6%** | 69.8% | **+8.8pp** |
| Mean Latency | 7.2s | 13.9s | −6.7s |
| Median Latency | 6.5s | 2.9s | +3.6s |

Hybrid wins on all accuracy metrics. LogicRAG has lower median latency (most questions exit in 0–1 rounds), but higher mean latency (multi-round questions are slow: up to 3 rounds of retrieval + LLM calls).

---

## Win/Loss Breakdown

| Category | Count | % |
|----------|-------|---|
| Both correct | 76 | 76% |
| Hybrid-only correct | 13 | 13% |
| LogicRAG-only correct | 4 | 4% |
| Neither correct | 7 | 7% |

**Key insight:** Hybrid wins more than 3× as many exclusive questions (13 vs 4). The 76 shared correct answers form a strong common core.

---

## By Question Type

| Type | n | Hybrid Judge | LogicRAG Judge | Delta | Hybrid EM | LogicRAG EM | Hybrid F1 | LogicRAG F1 |
|------|---|-------------|---------------|-------|-----------|-------------|-----------|-------------|
| comparison | 20 | 95.0% | 95.0% | 0pp | 95.0% | 95.0% | 95.0% | 95.0% |
| bridge_comparison | 19 | **94.7%** | 73.7% | **+21.1pp** | 94.7% | 68.4% | 94.7% | 71.4% |
| inference | 15 | 80.0% | **100.0%** | **−20.0pp** | 60.0% | 73.3% | 76.5% | 92.2% |
| compositional | 46 | **87.0%** | 69.6% | **+17.4pp** | 45.7% | 41.3% | 65.6% | 50.9% |

### Interpretation

- **comparison** — Tied at 95%. Both systems handle yes/no comparison questions well. These are structured queries with clear binary answers.
- **bridge_comparison** — Hybrid's strongest advantage (+21pp). These multi-hop questions compare properties across entity chains (e.g., "Which director died first, X or Y?"). The KG-guided SPARQL path resolves the chains directly, while LogicRAG struggles to retrieve both entities' properties from its fixed corpus.
- **inference** — LogicRAG's only win (100% vs 80%). Inference questions require extracting implicit relationships (e.g., "Who is X's maternal grandfather?"). LogicRAG's corpus-based retrieval finds direct textual evidence more reliably than our SPARQL path, which sometimes follows incorrect KG relations (see failure examples below).
- **compositional** — Hybrid's largest absolute gain (+17pp). These are bridge questions (e.g., "Where was the director of film X born?"). Hybrid solves them via direct SPARQL 2-hop chains or LogicRAG decomposition with SPARQL sub-queries. LogicRAG frequently returns "Unknown" — its corpus lacks the connecting information.

---

## By KG Answerability

| Subset | n | Hybrid Judge | LogicRAG Judge | Delta |
|--------|---|-------------|---------------|-------|
| KG-answerable | 36 | **97.2%** (35/36) | 83.3% (30/36) | **+13.9pp** |
| KG-unanswerable | 64 | **84.4%** (54/64) | 78.1% (50/64) | **+6.3pp** |

Hybrid dominates on KG-answerable questions — SPARQL resolves these directly. On KG-unanswerable questions, both systems must rely on text retrieval; Hybrid still leads by 6pp, likely from superior entity resolution and multi-hop decomposition routing.

**Note:** `is_answerable` uses the corrected coverage analysis, not hop-verified ground truth (which is absent from these 100 records). 4 questions are known to have discrepant labels between the two sources.

---

## Failure Analysis: Hybrid-Only Wins (13 cases)

Where Hybrid succeeds and LogicRAG fails — by Hybrid's routing method:

| Hybrid Method | Count | Typical Pattern |
|---------------|-------|-----------------|
| sparql | 7 | Direct 2-hop SPARQL chains (director→birthPlace, film→director→deathDate) |
| logicrag_decomposed | 6 | Sub-question decomposition with SPARQL sub-queries |

LogicRAG returned "Unknown" in 10/13 of these cases — its fixed corpus (6,119 benchmark snippets) simply lacked the connecting entity information. The remaining 3 cases were incorrect answers (e.g., "Brookline, Massachusetts" instead of "Brooklyn" — wrong city retrieved from a similarly-named director passage).

**Representative examples:**

| Question | Gold | Hybrid | LogicRAG |
|----------|------|--------|----------|
| Where was the director of film Thomas Jefferson born? | Brooklyn | Brooklyn (SPARQL) | Brookline, Massachusetts |
| Why did John Middleton Murry's wife die? | tuberculosis | Tuberculosis (decomposed) | Unspecified |
| Which film has the director who is older, The Gamecock or Monster A Go-Go? | Monster A Go-Go | Monster A Go-Go (SPARQL) | Unknown |

---

## Failure Analysis: LogicRAG-Only Wins (4 cases)

3 of the 4 are inference questions with genealogical relationships; the fourth (Vortex) is a compositional nationality question:

| Question | Type | Gold | Hybrid (wrong) | LogicRAG (correct) |
|----------|------|------|---------|----------|
| What nationality is the director of Vortex (1976 Film)? | compositional | Italy | Italian-French (too specific) | Italian |
| Who is Philippe, Duke Of Anjou's maternal grandfather? | inference | Stanisław Leszczyński | Ferdinand Maria, Elector of Bavaria (wrong ancestor) | Stanisław Leszczyński |
| Who is Archduchess Maria Antonia Of Austria's maternal grandfather? | inference | Carlos, Duke of Madrid | Louis Rudolph, Duke of Brunswick (wrong ancestor) | Carlos, Duke of Madrid |
| Who is Teriimaevarua III's paternal grandmother? | inference | Pōmare IV | Ariifaaite (wrong ancestor) | Queen Pōmare IV |

**Pattern:** The 3 inference cases are genealogical queries where Hybrid's SPARQL path followed the wrong KG relation (e.g., a different parent/ancestor than the benchmark expects). LogicRAG's corpus contains the correct textual evidence from the original benchmark context paragraphs, giving it an inherent advantage on questions whose answers come directly from the 2017 Wikipedia snapshot.

The Vortex case is a different failure mode — nationality precision: Hybrid answered "Italian-French" (more complete) while the benchmark gold is "Italy" and LogicRAG answered "Italian" (matching the expected form).

---

## Neither Correct (7 cases)

| Type | Count |
|------|-------|
| compositional | 5 |
| bridge_comparison | 1 |
| comparison | 1 |

These are genuinely hard questions where both systems fail — typically obscure entities with poor Wikipedia coverage and incomplete KG relations.

---

## LogicRAG Execution Profile

| Rounds | Count | Description |
|--------|-------|-------------|
| 0 | 39 | Answered from warmup retrieval (no follow-up needed) |
| 1 | 16 | One follow-up sub-query |
| 2 | 34 | Two follow-up rounds |
| 3 | 11 | Maximum rounds (exhausted budget) |

61% of questions require ≥2 rounds, explaining the high mean latency despite the low median. The round-0 fast path (39% of questions) covers comparison questions and simple factual lookups.

---

## Methodological Notes

1. **Same judge, same questions:** Both systems judged by the same gemma3:12b model with identical prompts on the same 100-question subset. The judge includes a word-level containment fast path (gold words ⊂ predicted words → auto-pass) to reduce false negatives from surface-form variation.

2. **Not a fair retrieval comparison:** Hybrid accesses live Wikipedia + DBpedia SPARQL. LogicRAG retrieves from 6,119 fixed benchmark snippets where the answer is guaranteed to exist. LogicRAG's "Unknown" responses indicate its corpus limitation, not a retrieval quality failure. See [logicrag-vs-our-retrieval.md](logicrag-vs-our-retrieval.md) for the full architectural comparison.

3. **Sample representativeness:** These 100 questions are the first 100 by insertion order from `dev_full.json`, not a stratified sample. Type distribution: 46 compositional, 20 comparison, 19 bridge_comparison, 15 inference. This over-represents compositional vs the full 1000q distribution.

4. **EM vs Judge gap:** Hybrid shows a 22pp Judge/EM gap (89% vs 67%) while LogicRAG shows 18pp (80% vs 62%). Both gaps are driven by surface-form mismatches (city abbreviations, date formats, nationality adjective vs country name). The Judge score better reflects true accuracy.

5. **Latency context:** Hybrid uses Groq API (cloud); LogicRAG uses the same Groq API for LLM calls but runs retrieval locally against a pre-built FAISS index. Hybrid additionally calls DBpedia Spotlight + SPARQL endpoints, adding network overhead.

---

## Summary

Hybrid outperforms LogicRAG on this 100-question sample by **+9pp judge accuracy** (89% vs 80%). The advantage comes primarily from:

1. **bridge_comparison (+21pp):** KG-guided SPARQL chains resolve multi-hop property comparisons that LogicRAG's fixed corpus cannot.
2. **compositional (+17pp):** Sub-question decomposition with SPARQL sub-queries retrieves entity properties more reliably than corpus search.

LogicRAG's sole advantage is on **inference questions (−20pp):** its benchmark corpus contains direct textual evidence for genealogical/implicit relationships where our SPARQL path sometimes follows incorrect KG relations.

**Implication for thesis:** The hybrid approach's value is strongest on structured multi-hop questions (bridge, bridge_comparison) where the KG provides reliable relational paths. For inference questions requiring implicit reasoning, corpus-based retrieval remains competitive — suggesting a future optimization: route inference questions directly to text retrieval rather than attempting SPARQL.
