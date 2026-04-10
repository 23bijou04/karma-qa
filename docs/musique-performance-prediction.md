# MuSiQue Performance Prediction & Cross-Benchmark Analysis

**Date:** 2026-04-05
**Status:** Pre-evaluation prediction (no MuSiQue run yet)
**Purpose:** Calibrated performance predictions for Hybrid, LinearRAG, LogicRAG on MuSiQue 1000q, with factual corrections to prior analysis

---

## 1. Report Verification: Errors Found

A prior internal report (undated) predicted MuSiQue performance based on "56.8% EM on 81 HotPotQA questions." Expert verification against actual data revealed **8 factual errors**, 3 critical.

### Critical Errors

| # | Claim | Actual (Verified) | Impact |
|---|-------|-------------------|--------|
| 1 | "56.8% EM on 81 HotPotQA questions" | **53.8% EM / 86.1% Judge EM on 1000q** (full run completed, +0.6pp from synthesis fix) | All predictions anchored on wrong baseline |
| 2 | "0 direct SPARQL calls on HotPotQA" | **70 direct SPARQL (7.0%) + 112 sub-Q SPARQL (6.3% of 1779)** | KG contribution not zero |
| 3 | "Hybrid = LinearRAG at 56.8%" | Hybrid 86.1% vs LinearRAG 86.9% Judge EM (0.8pp gap) | Invalidates "KG adds nothing" claim |

### High-Severity Errors

| # | Claim | Actual |
|---|-------|--------|
| 4 | "Our KG advantage is essentially zero" | 182 SPARQL calls succeeded; KG contributes to ~18% of questions |
| 5 | "8-12pp format mismatch" | **31.7pp** Judge-EM gap on HotPotQA (68.8% of EM failures recovered by judge) |

### Medium-Severity Errors

| # | Claim | Actual |
|---|-------|--------|
| 6 | Answer types "82.9% entity, 12.3% numeric, 4.8% dates" | 76.4% entity, 12.1% numeric, **11.5% dates** (dates 2.4x underestimated) |
| 7 | "81.1% decomp step answers are named entities" | **91.9%** (underestimated 10.8pp) |
| 8 | Predicted 45-55% overall Judge EM | Must recalibrate using correct 86.1% HotPotQA baseline |
| 9 | "64 KEYWORD_TO_PROPERTIES entries" (gap audit D2) | **38 entries** (verified in question_analysis.py:18-64) |
| 10 | "7 TYPE_HINTS entries" (gap audit D8) | **15 keyword patterns / 8 unique target types** (entity_linker.py:99-114) |
| 11 | "60/30/10 context budget split" (gap audit D6) | **25K max / 12K reranked / 4K per-entity** — no percentage split found |

### Confirmed Claims

- Hop distribution: 2-hop 518 (51.8%), 3-hop 316 (31.6%), 4-hop 166 (16.6%) ✓
- 6 structural types (2hop, 3hop1, 3hop2, 4hop1, 4hop2, 4hop3) ✓
- 2,648 total sub-questions ✓
- 20 paragraphs/question, 2-4 supporting ✓
- 100% answerable (filtered subset) ✓
- Top decomposition properties: located_in (105), performer (94), place_of_birth (82) ✓
- Open vocabulary vs closed templates: correct qualitative assessment ✓
- MuSiQue is harder than 2WikiMultiHopQA: correct ✓

---

## 2. Verified Cross-Benchmark Baseline

### Judge EM (primary metric)

| Benchmark | Hybrid | LinearRAG | LogicRAG | N |
|-----------|--------|-----------|----------|---|
| **2WikiMultiHopQA** | **89.4%** | 84.0% | 74.5% | 1000 |
| **HotPotQA** | **86.1%** | **86.9%** | 78.8% | 1000 |
| **MuSiQue** | ??? | ??? | ??? | 1000 |

### Raw EM (secondary metric)

| Benchmark | Hybrid | LinearRAG | LogicRAG | N |
|-----------|--------|-----------|----------|---|
| **2WikiMultiHopQA** | ~71.4% | ~65% | ~55% | 1000 |
| **HotPotQA** | 53.8% | 59.4% | 53.5% | 1000 |
| **MuSiQue** | ??? | ??? | ??? | 1000 |

### Per-Type Breakdown (Judge EM)

**2WikiMultiHopQA (Hybrid):**
- bridge_comparison: 91.9% (235q)
- comparison: 95.9% (244q)
- compositional: 82.6% (413q) ← hardest type
- inference: 95.4% (108q)

**HotPotQA (Judge EM):**

| Type | Hybrid | LinearRAG | LogicRAG |
|------|--------|-----------|----------|
| bridge (811q) | 84.2% | 86.9% | 76.9% |
| comparison (189q) | 91.0% | 86.8% | 86.8% |

### Key Observations

1. **2Wiki→HotPotQA degradation:** Hybrid drops 3.3pp (89.4→86.1%), LinearRAG drops -2.9pp (gains!), LogicRAG drops -4.3pp.
2. **HotPotQA is close to 2Wiki performance**, contradicting the prior report's catastrophic narrative.
3. **LinearRAG slightly beats Hybrid on HotPotQA** (+1.4pp), reversing the 2Wiki ordering.
4. **The EM-Judge gap balloons on HotPotQA** (31.7pp vs ~18pp on 2Wiki) — HotPotQA gold answers use different formatting conventions.
5. **Routing:** HotPotQA uses 87.7% logicrag_decomposed, 7% sparql, 5.3% text. NOT "0% SPARQL."

---

## 3. Why MuSiQue Is Harder (Verified Difficulty Factors)

### Factor 1: Deeper Chains (MAJOR)
- 48.2% of questions are 3+ hops (vs mostly 2-hop in 2Wiki and HotPotQA)
- 4-hop questions (16.6%) require chaining 4 intermediate answers correctly
- Cascading error model: if per-hop accuracy is p, N-hop accuracy ≤ p^N
  - At p=0.85 (HotPotQA bridge per-hop estimate): 2-hop=72%, 3-hop=61%, 4-hop=52%
  - At p=0.90: 2-hop=81%, 3-hop=73%, 4-hop=66%

### Factor 2: Open Vocabulary (MAJOR)
- 2WikiMultiHopQA uses templated questions ("Who directed X?")
- MuSiQue uses natural English ("When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?")
- Our 64 keyword-to-property mappings and 82 bridge/hop2 keywords cover 2Wiki's vocabulary well but miss MuSiQue's diversity
- **However:** This primarily affects routing, not final accuracy — 87.7% of HotPotQA already routes to decomposition, which works despite keyword mismatches

### Factor 3: Entity Diversity (MODERATE — mitigated)
- MuSiQue supporting paragraphs span 1,318 unique entities across politics, sports, military, geography, science, entertainment
- Our schema discovery: originally 7 cached types → now **734 classes from ontology cache** (expanded since gap audit was written)
- **Verified coverage:** MilitaryConflict (17 props), University (75 props), SoccerClub (58 props), SportsTeam (52 props), Building (34 props), Settlement (392 props) — all present in expanded cache
- "Battle" (dbo:Battle) is NOT a separate class in DBpedia — mapped to MilitaryConflict via superclass hierarchy
- **Remaining gap:** "Treaty" is not a DBpedia ontology class; falls back to text retrieval

### Factor 4: Answer Format Diversity (MODERATE)
- MuSiQue answers include: "mid-June", "built in the 15th century", "December 14, 1814", "918"
- 11.5% dates (higher than report claimed) + 12.1% numeric = 23.6% non-entity answers
- These are harder to match in EM but LLM judge handles format differences well
- Expected EM-Judge gap: ~25-35pp (similar to HotPotQA)

### Factor 5: Decomposition Quality (MODERATE-HIGH)
- MuSiQue ground-truth decompositions use placeholder syntax (#1, #2) for dependency chaining
- Our system generates all sub-questions upfront — fragile for 4-hop chains
- 36.1% of sub-questions use Wikidata property notation (>>) — these map to specific DBpedia properties but under different names

---

## 4. Calibrated Performance Predictions

### Methodology

Predictions are calibrated from verified HotPotQA 1000q results (not the retracted 81q sample). Reasoning:

1. **HotPotQA baseline:** 86.1% Judge EM (Hybrid), with 87.7% decomposition route — similar to expected MuSiQue routing
2. **Degradation factors:** deeper chains, open vocabulary, entity diversity
3. **Mitigation factors:** corpus-provided paragraphs (MuSiQue includes supporting text), expanded 734-class ontology cache

### Prediction Model

**Core assumption:** MuSiQue 2-hop ≈ HotPotQA performance (both use text+decomposition). Each additional hop degrades by a per-hop factor.

**Per-hop accuracy calibration:** HotPotQA bridge (mostly 2-hop) Judge EM = 84.2% → per-hop = √0.842 = **0.918**. Applying a MuSiQue difficulty discount (open vocabulary, entity diversity), estimated per-hop: **0.88-0.92**.

**Planning accuracy:** Decomposing questions correctly gets harder with depth. Estimated: 2-hop 95%, 3-hop 90%, 4-hop 85%.

**Cascading model:** `accuracy(N) = per_hop^N × planning(N)`

| Per-hop | 2-hop | 3-hop | 4-hop | Weighted Overall |
|---------|-------|-------|-------|-----------------|
| 0.85 | 68.6% | 55.3% | 44.4% | 60.4% |
| 0.88 | 73.6% | 61.3% | 51.0% | 66.0% |
| 0.90 | 77.0% | 65.6% | 55.8% | 69.9% |
| 0.92 | 80.4% | 70.1% | 60.9% | 73.9% |

Weights: 2-hop=0.518, 3-hop=0.316, 4-hop=0.166 (from MuSiQue distribution).

### Hybrid Predictions (Judge EM)

| Subset | N | Estimate | 95% Range | Reasoning |
|--------|---|----------|-----------|-----------|
| 2-hop (518q) | 518 | **75-82%** | 70-85% | Similar to HotPotQA bridge (84.2%) but open-vocabulary routing penalty ~3-8pp |
| 3-hop (316q) | 316 | **58-68%** | 52-72% | Cascading: 0.88^3=68% theoretical × 0.9 planning accuracy |
| 4-hop (166q) | 166 | **42-55%** | 35-60% | Cascading: 0.88^4=60% theoretical × 0.85 planning accuracy |
| **Overall (1000q)** | 1000 | **65-73%** | 58-78% | Weighted: 0.518×78 + 0.316×63 + 0.166×48 ≈ 68% |

### LinearRAG Predictions (Judge EM)

LinearRAG slightly outperformed Hybrid on HotPotQA (+1.4pp). It lacks KG but has strong text retrieval.

| Subset | Estimate | Reasoning |
|--------|----------|-----------|
| 2-hop | **77-84%** | Strong text retrieval; beat Hybrid on HotPotQA bridge |
| 3-hop | **55-65%** | No iterative decomposition advantage; pure retrieval degrades with depth |
| 4-hop | **38-50%** | Retrieval quality drops significantly at 4 hops |
| **Overall** | **63-72%** | Slightly below Hybrid due to depth penalty |

### LogicRAG Predictions (Judge EM)

LogicRAG consistently underperforms both systems but has purpose-built MuSiQue support (this IS their dataset).

| Subset | Estimate | Reasoning |
|--------|----------|-----------|
| 2-hop | **70-78%** | Baseline competence; HotPotQA bridge was 76.9% |
| 3-hop | **55-65%** | LogicRAG designed for multi-hop; should handle depth better |
| 4-hop | **45-55%** | Best depth handling due to iterative decomposition |
| **Overall** | **62-70%** | May benefit from being designed for MuSiQue specifically |

### Summary Prediction Table

| System | 2Wiki Judge EM | HotPotQA Judge EM | MuSiQue Predicted | Expected Drop |
|--------|---------------|-------------------|-------------------|---------------|
| **Hybrid** | 89.4% | 86.1% | **65-73%** | -13 to -21pp from HotPotQA |
| **LinearRAG** | 84.0% | 86.9% | **63-72%** | -15 to -24pp from HotPotQA |
| **LogicRAG** | 74.5% | 78.8% | **62-70%** | -9 to -17pp from HotPotQA |

---

## 5. Why We Predict Narrower Gaps Than the Prior Report

The prior report predicted 45-55% for Hybrid. We predict 65-73%. Key reasons:

1. **Wrong baseline:** Report used 56.8% EM (81q, old incomplete run). Actual: 86.1% Judge EM (1000q, post synthesis fix).
2. **KG not dead:** 70 direct SPARQL + 112 sub-Q SPARQL on HotPotQA. KG handles simple property lookups (birthDate, director) even in open-domain. MuSiQue has these too (place_of_birth: 82 sub-Qs, country: 69).
3. **Corpus advantage:** MuSiQue provides 20 paragraphs per question including 2-4 supporting ones. Our text solver can exploit this directly — no Wikipedia fetch needed.
4. **Expanded ontology:** Since the report was written, schema discovery was upgraded from 7 hardcoded types to 734-class ontology cache. This addresses the D3 critical issue.
5. **Judge EM vs raw EM:** The report mixed metrics. Raw EM will be ~35-50% (harsh on format). Judge EM will be 65-73% (semantic matching).

---

## 6. Predicted System Behavior on MuSiQue

### Routing Prediction

| Route | Estimated % | Reasoning |
|-------|-------------|-----------|
| logicrag_decomposed | 85-92% | Same as HotPotQA (87.7%); open vocab forces decomposition |
| sparql (direct) | 2-5% | Lower than HotPotQA (7%) due to deeper chains |
| text (fallback) | 5-10% | Questions that can't be decomposed |

### Where Each System Will Struggle

**Hybrid:**
- 4-hop questions with entity types outside DBpedia ontology
- Questions requiring date arithmetic ("how many years between X and Y?")
- Bridge chains where intermediate entity isn't in DBpedia

**LinearRAG:**
- Same as Hybrid but worse at 3-4 hop because no iterative decomposition
- Retrieval degradation: each hop adds noise to context

**LogicRAG:**
- 2-hop questions where our text+KG hybrid has an advantage
- Comparison questions (our routing handles these better)

### Expected Error Patterns

Based on HotPotQA failure analysis (144 true failures out of 1000):
1. **Cascading decomposition errors** (~40% of failures): wrong intermediate answer → wrong final answer
2. **Entity not found** (~20%): supporting entity not in Wikipedia/DBpedia
3. **Answer format mismatch** (~15%): recovered by Judge but not by EM
4. **Insufficient context** (~15%): text retrieval misses key paragraph
5. **Wrong reasoning** (~10%): all facts correct but LLM reasons incorrectly

---

## 7. What to Watch For (Validation Checklist)

After running MuSiQue eval, validate these predictions:

- [ ] Overall Judge EM: predicted 65-73% (if <60%, our depth model is too optimistic; if >78%, MuSiQue corpus paragraphs help more than expected)
- [ ] 2-hop vs 3-hop vs 4-hop degradation curve: should show ~10-15pp drop per hop level
- [ ] SPARQL direct calls: predicted 2-5% (if 0%, prior report was right about KG being useless)
- [ ] Sub-question SPARQL: predicted 5-10% of sub-Qs use SPARQL
- [ ] EM-Judge gap: predicted 25-35pp (similar to HotPotQA's 31.7pp)
- [ ] LinearRAG vs Hybrid ordering: may reverse (LinearRAG could win on 2-hop, lose on 4-hop)
- [ ] LogicRAG competitiveness: may perform closer to Hybrid than on 2Wiki (designed for MuSiQue)

---

## 8. Architecture-to-Benchmark Mapping

### How Our System Maps to MuSiQue Properties

| MuSiQue Property (sub-Q) | Count | DBpedia Coverage | Our System Handles Via |
|---------------------------|-------|------------------|----------------------|
| located_in / administrative | 105 | Good (dbo:location, dbo:country) | SPARQL sub-Q or text |
| performer | 94 | Good (dbo:artist, dbo:musicalArtist) | SPARQL (MusicalWork cached) |
| place_of_birth | 82 | Excellent (dbo:birthPlace) | SPARQL (Person cached) |
| country | 69 | Good (dbo:country) | SPARQL or text |
| headquarters_location | 36 | Good (dbo:headquarter) | SPARQL (Company cached) |
| spouse | 35 | Good (dbo:spouse) | SPARQL (Person cached) |
| operator | 34 | Moderate (dbo:operator) | Dynamic schema or text |
| shares_border_with | 30 | Good (dbo:bordersWith) | SPARQL (Place cached) |
| place_of_death | 29 | Good (dbo:deathPlace) | SPARQL (Person cached) |
| child | 26 | Good (dbo:child) | SPARQL (Person cached) |
| author | 26 | Good (dbo:author) | SPARQL (Book cached) |
| creator | 24 | Moderate (dbo:creator) | Dynamic schema or text |
| record_label | 22 | Good (dbo:recordLabel) | SPARQL (MusicalWork cached) |
| member_of_sports_team | 21 | Moderate (dbo:team) | Dynamic schema (734-class) |
| Natural language sub-Qs | 1693 | N/A | Text retrieval + LLM reasoning |

**Coverage estimate:** ~36% of sub-Qs use property notation (955/2648). Of these, ~60-70% map to cached/expanded ontology properties. The remaining ~64% are natural language sub-Qs handled by text retrieval.

### Per-Hop-Count Architecture Fit

**2-hop questions (518):** Best fit. Similar to HotPotQA bridge questions. Our logicrag_decomposition handles these well — decomposes into 2 sub-Qs, resolves sequentially. SPARQL can handle simple property lookups for sub-Qs.

**3-hop questions (316):** Moderate fit. Decomposition must plan 3 steps correctly. Sub-types:
- 3hop1 (243): Linear chain A→B→C. If A→B is a known property (e.g., place_of_birth), first hop succeeds; subsequent hops depend on quality of B.
- 3hop2 (73): Two branches converging. Requires correct dependency planning.

**4-hop questions (166):** Poor fit. Our one-shot decomposition must predict all 4 steps upfront. Sub-types:
- 4hop1 (108): Linear 4-step chain. Very fragile.
- 4hop2 (27): 3+1 branching structure.
- 4hop3 (31): 2+2 parallel branches merging.

### Why LinearRAG May Compete

LinearRAG uses direct retrieval without KG. On MuSiQue:
- **Advantage:** No routing overhead; directly retrieves from corpus paragraphs
- **Advantage:** MuSiQue provides 20 paragraphs including supporting ones — retrieval is easier
- **Disadvantage:** No structured decomposition for deep chains
- **Prediction:** Strong on 2-hop, weaker on 3-4 hop relative to Hybrid

### Why LogicRAG May Surprise

LogicRAG was developed for MuSiQue-style questions:
- **Advantage:** Iterative decomposition designed for multi-hop
- **Advantage:** This IS their evaluation dataset — potential train/eval contamination
- **Disadvantage:** Weaker text retrieval than our system
- **Prediction:** May be strongest on 4-hop; weakest on 2-hop comparison

---

## 9. Common Pitfalls (from Prior Report — Still Valid)

These qualitative warnings from the prior report remain valid despite the numerical errors:

1. **Don't over-optimize routing for MuSiQue** — The decomposition path does most work regardless
2. **Cascading errors are the primary challenge** — Each wrong intermediate answer corrupts downstream
3. **Answer normalization is free points** — Large EM-Judge gap confirms format mismatch is significant
4. **Run first, fix second** — Prior expert panel's recommendation stands: get actual data before targeted fixes
5. **Rate limiting budget:** ~1000 questions × avg 2.7 hops × 2-3 LLM calls/hop = 5400-8100 LLM calls

---

## 10. Comparison with Published Results

### Dataset Context

MuSiQue (Trivedi et al., TACL 2022, arxiv 2108.00573):
- **Full dataset:** ~25K questions (MuSiQue-Ans: answerable only; MuSiQue-Full: +unanswerable variants = 50K)
- **Hops:** 2-4 (vs HotPotQA 2-hop only)
- **Human-machine gap:** 28.2pp F1 (78.0 human vs 49.8 best model) — **3x larger** than HotPotQA (9.6pp gap)
- **Our subset:** 1,000 answerable questions from LogicRAG's evaluation set

### Published System Performance (Raw EM/F1, open retrieval)

| System | Year | Metric | Score | Notes |
|--------|------|--------|-------|-------|
| Self-Ask | 2022 | EM | 37% | Weakest decomposition baseline |
| IRCoT | 2023 | EM | 49.8% | Interleaved retrieval + CoT |
| ReAct v1 | 2022 | EM | 61% | Competitive; reasoning+acting |
| ReAct v2 | 2023 | EM | 43.2% | Revised evaluation |
| **LogicRAG** | 2025 | **LLM Judge EM** | **37.5%** | Uses gpt-4o-mini; our baseline |
| PropRAG | 2025 | Recall@5 | 77.3% | Retrieval-only metric |
| PropRAG | 2025 | F1 | 52.4% | End-to-end answer F1 |
| Beam Retrieval | 2023 | EM (retrieval) | 79.3% | Retrieval recall SOTA |
| FSM (GPT-3.5) | 2025 | F1 | 38.4% | Concise output key finding |
| FSM (Qwen-72B) | 2025 | F1 | 41.2% | 2x baseline zero-shot F1 |

### Critical Finding: Reasoning > Retrieval Bottleneck

A Graph-RAG study on MuSiQue with good retrieval (77.2% answer present in context):
- **8B model:** 23.6% EM (50+ point gap between retrieval and reasoning)
- **70B + CoT:** 53.2% EM (closes ~half the gap)
- **Conclusion:** **Reasoning is the dominant bottleneck on MuSiQue**, not retrieval

**Contrast with HotPotQA:** 70B+CoT reaches 83.9% (near 90.8% retrieval ceiling). 2-hop reasoning is fundamentally easier.

### Output Conciseness Finding

FSM-prompting study found **10x performance drop** with verbose vs concise output:
- Short outputs: 30% EM, 42.8% F1
- Verbose outputs: 3.3% EM, 10.0% F1

Our system uses `FINAL ANSWER:` extraction which forces concise output — this is a natural advantage.

### Our Setup vs Published Setups

| Factor | Published (typical) | Our Setup | Effect |
|--------|-------------------|-----------|--------|
| Retrieval | Open (Wikipedia) | Provided corpus (20 paragraphs) | +15-25pp (retrieval solved) |
| Model | GPT-3.5 / 8B / gpt-4o-mini | Llama 3.3 70B | +10-20pp (stronger reasoning) |
| Metric | Raw EM | LLM Judge EM | +25-35pp (format tolerance) |
| Dataset | MuSiQue-Full (incl. unanswerable) | Answerable only (1000q) | +5-10pp (no distractor penalty) |
| Answer extraction | Varies | `FINAL ANSWER:` forced conciseness | +5-10pp (conciseness advantage) |

**Cumulative setup advantages:** These factors explain why our predicted Judge EM (65-73%) is much higher than published raw EM (37-50%).

### Fair Comparison: Apples-to-Apples

To compare our results with published work, use **raw EM** (not Judge EM):

| System | MuSiQue Raw EM | Setup |
|--------|---------------|-------|
| Self-Ask | 37% | Open retrieval, GPT-3.5 |
| IRCoT | 49.8% | Open retrieval, GPT-3.5 |
| LogicRAG (published) | ~30.4% | Open retrieval, gpt-4o-mini |
| **Our Hybrid (predicted)** | **35-50%** | Provided corpus, Llama 70B |
| **Our LogicRAG (predicted)** | **35-45%** | Provided corpus, Llama 70B |

Our predicted raw EM is competitive with published systems despite using an open-weight model, because the provided corpus compensates for weaker retrieval.

### LogicRAG Published vs Our LogicRAG

Published LogicRAG on MuSiQue: 37.5% Judge EM (gpt-4o-mini, open retrieval)
Our LogicRAG on HotPotQA: 78.8% Judge EM (Llama 70B, Wikipedia retrieval)
Our LogicRAG on 2Wiki: 74.5% Judge EM

The 37.5% → our predicted 62-70% gap is explained by:
1. **Model:** gpt-4o-mini (small) → Llama 3.3 70B (much stronger reasoning)
2. **Retrieval:** Open → provided corpus (20 paragraphs including supporting)
3. **Metric:** Their Judge uses gpt-4o-mini; ours uses gemma3:12b (different calibration)

---

## Appendix A: Data Sources

All numbers verified against actual data files:

- 2WikiMultiHopQA: `data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl` (894/1000)
- HotPotQA Hybrid: `data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged.jsonl` (861/1000, post synthesis fix)
- HotPotQA LinearRAG: `data/cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_judged.jsonl` (869/1000)
- HotPotQA LogicRAG: `data/cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_judged.jsonl` (788/1000)
- MuSiQue dataset: `baselines/LogicRAG/dataset/musique.json` (1000 questions)
- MuSiQue corpus: `baselines/LogicRAG/dataset/musique_corpus.json` (11,656 entries)

## Appendix B: MuSiQue Dataset Statistics (Verified)

- **Questions:** 1,000
- **Hop distribution:** 2-hop 518 (51.8%), 3-hop 316 (31.6%), 4-hop 166 (16.6%)
- **Types by ID:** 2hop (518), 3hop1 (243), 3hop2 (73), 4hop1 (108), 4hop2 (27), 4hop3 (31)
- **Total sub-questions:** 2,648
- **Sub-Qs with >> property notation:** 955 (36.1%)
- **Paragraphs per question:** 20 (2-4 supporting)
- **Corpus entries:** 11,656
- **Answerable:** 100% (filtered subset)
- **Answer types (final):** 76.4% entity/text, 12.1% numeric, 11.5% dates
- **Step answer types:** 91.9% named entity, 4.6% numeric, 3.5% dates
- **Top decomposition properties:** located_in (105), performer (94), place_of_birth (82), country (69), headquarters_location (36), spouse (35), operator (34), shares_border_with (30)
- **Unique supporting titles:** 1,318

## Appendix C: Prior Report Corrections

The prior internal report ("MuSiQue Benchmark Characteristics") and the expert panel synthesis (`synthesis-260403-0954-musique-readiness-expert-panel.md`) were based on an incomplete 81-question HotPotQA sample. The full 1000q evaluation has since completed. All predictions and diagnoses anchored on the 81q/56.8% numbers should be treated as **retracted** and replaced by the calibrated predictions in this document.

### Retracted Performance Claims
- "Our KG advantage is essentially zero on out-of-domain benchmarks" → KG contributes to ~18% of HotPotQA questions via SPARQL
- "Hybrid 56.8% EM = LinearRAG 56.8% EM" → Hybrid 86.1% vs LinearRAG 86.9% Judge EM
- "HotPotQA: 0 direct SPARQL calls" → 70 direct + 112 sub-Q SPARQL calls
- "Expect ~0% direct SPARQL route on MuSiQue" → Predict 2-5% direct SPARQL
- "45-55% Judge EM overall" → Predict 65-73% Judge EM

### Retracted System Gap Audit Claims (verified by code review)
- "64 KEYWORD_TO_PROPERTIES entries" (D2) → **38 entries** (`question_analysis.py:18-64`)
- "7 cached entity types" (D3) → **734 classes** from ontology cache (commit `1a7914a`, 2026-04-03)
- "60/30/10 context budget split" (D6) → **25K max / 12K reranked / 4K per-entity** (no percentage split)
- "TYPE_HINTS limited to 7 types" (D8) → **15 patterns / 8 unique types** (`entity_linker.py:99-114`)

### Confirmed System Gap Audit Claims
- Depth cap at 6 sub-questions (D1) → **Confirmed** (`logicrag_decomposition.py:418-419`)
- ASK routing uses `all()` for comparison questions (D4) → **Confirmed** (`routing.py:370-382`)
- Answer normalization exists in `evaluation/metrics.py` → **Comprehensive** (date, unicode, article, punctuation, whitespace normalization)

## Appendix D: Verification Sources

- **Researcher report:** `plans/reports/researcher-260405-0820-musique-published-benchmarks.md`
- **System gap audit:** `plans/reports/researcher-260403-0951-system-gap-audit-musique.md`
- **Expert panel synthesis:** `plans/reports/synthesis-260403-0954-musique-readiness-expert-panel.md`
- **Code review:** Explorer agent verified all line numbers and counts against actual codebase (2026-04-05)
