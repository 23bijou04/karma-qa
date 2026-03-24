# Benchmark Data Errors in 2WikiMultiHopQA

**Date:** 2026-03-24
**Source dataset:** 2WikiMultiHopQA dev set (`data/dev_full.json`, 12,576 questions)
**Purpose:** Document confirmed factual errors in benchmark questions or gold answers, discovered during evaluation. These are cases where the benchmark's own supporting context contradicts the question's presupposition or the gold answer, representing annotation artifacts — not system failures.

---

## Error #1: Presupposed Marriage That Never Existed

**Question ID:** `fd848f860bdd11eba7f7acde48001122`
**Question:** "What nationality is Margaretha Van Mechelen's husband?"
**Question type:** compositional
**Scenario:** D (BOTH_HOPS_MISSING)
**Gold answer:** Dutch Republic
**Expected reasoning chain:**
1. Margaretha van Mechelen → spouse → Maurice of Nassau
2. Maurice of Nassau → nationality → Dutch Republic

### The Error

The question presupposes that Margaretha van Mechelen had a husband. She did not. The benchmark's **own supporting context** explicitly contradicts this:

> **Margaretha van Mechelen paragraph, sentence [1]:**
> "the **mistress** of Maurice of Nassau, Prince of Orange, with whom she had 3 sons"
>
> **Sentence [2]:**
> "Though Maurice **refused to marry her** (probably because she was a Catholic and only from the minor nobility), he did state his intent to do so from his **death bed**"
>
> **Louis of Nassau paragraph, sentence [1]:**
> "He was the **illegitimate** son of Margaretha van Mechelen and Maurice, Prince of Orange"

Three independent statements in the benchmark's own corpus confirm:
1. She was his **mistress**, not wife
2. Maurice **refused to marry her** during his lifetime
3. Their children were classified as **illegitimate** (i.e. born outside marriage)

### Root Cause — Wikidata P26 Temporal Analysis

2WikiMultiHopQA was constructed from a **Wikidata snapshot dated December 31, 2019** (Ho et al., COLING 2020, arxiv:2011.01060). We traced the Wikidata edit history to reconstruct exactly what happened:

| Date | Event | Source |
|---|---|---|
| **2008-07-11** | Wikipedia article "Margaretha van Mechelen" created — describes her as **"mistress"** from day one | Wikipedia revision history |
| **2018-04-05** | Wikidata P26 (spouse) claim **created** on both Q2377989 (Margaretha) and Q164062 (Maurice) — rank: **normal** | `wbsetclaim-create` / `wbcreateclaim-create` |
| **2019-06-16** | Reference added to P26 claim on Margaretha's entity | `wbsetreference-add` |
| **2019-12-31** | **Wikidata snapshot used by 2WikiMultiHopQA** — P26 claim exists with **normal rank** (not yet deprecated) | Ho et al. paper |
| **2021-11-16** | P26 claim **deprecated** on both entities — qualifier P2241 set to Q109324480 ("not married") | `wbsetclaim-update` |

**What this means:**

1. Someone incorrectly added P26 (spouse) in **April 2018** — even though the Wikipedia article had said "mistress" since 2008
2. When 2WikiMultiHopQA was built (**Dec 2019 snapshot**), this incorrect P26 claim existed with **normal rank** — it looked like a valid spouse relation
3. The question generation pipeline read `P26 = spouse` and generated the question "What nationality is Margaretha Van Mechelen's **husband**?"
4. The Wikidata community corrected this in **November 2021** by deprecating the claim and adding the "not married" qualifier — but the benchmark was already published

**The benchmark's own Wikipedia context** (included as supporting paragraphs) always said "mistress" and "refused to marry her." The question generation trusted the **Wikidata triple over its own text evidence**, and the Wikidata triple was wrong at the time.

This is not a case of Wikipedia changing after the benchmark was created — the text has been consistent since 2008. It is a case of an **incorrect Wikidata annotation (P26 with normal rank, April 2018)** that was later corrected (deprecated, November 2021), but not before it was consumed by the benchmark pipeline.

### Impact on Evaluation

Both our hybrid system and the LogicRAG baseline scored 0.0 EM / 0.0 F1 on this question:

| System | Answer | Gold | EM | F1 |
|---|---|---|---|---|
| Our hybrid (post-fix) | "None mentioned" | Dutch Republic | 0.0 | 0.0 |
| LogicRAG baseline | "Unknown" | Dutch Republic | 0.0 | 0.0 |
| Our hybrid (pre-fix) | "American" | Dutch Republic | 0.0 | 0.0 |

Both post-fix systems retrieved the correct Margaretha van Mechelen article (Context #1), identified Maurice of Nassau as the relevant person, and **correctly refused** to call him her "husband" because the text says otherwise. This is the factually correct behavior — the systems are being penalized for being right.

The pre-fix hybrid answered "American" (wrong entity linking led to an unrelated person), which is a genuine system error but on a question that is itself flawed.

### Classification

- **Error type:** Presupposition failure — question assumes a relationship (marriage) that the evidence explicitly denies
- **Severity:** The question is unanswerable as phrased given the actual historical facts
- **Affected metrics:** Both EM and F1 are 0.0 for any system that correctly reads the corpus
- **Benchmark artifact origin:** Incorrect Wikidata P26 (spouse) annotation (April 2018, normal rank) consumed by benchmark pipeline (Dec 2019 snapshot). Later corrected by Wikidata community (November 2021, deprecated with "not married" qualifier Q109324480). Wikipedia text was always correct.
- **Information drift?** **No.** The Wikipedia article has said "mistress" since its creation in 2008. The error was in Wikidata, not Wikipedia. The benchmark's own context paragraphs (sourced from Wikipedia) contradict its own question (sourced from Wikidata).

### Recommendation

This question should be excluded from evaluation, or the gold answer should be flagged as "disputed." Any system that answers "Dutch Republic" is doing so by ignoring the textual evidence and blindly following the (incorrect) question presupposition. Systems that refuse to answer are arguably more correct.

### Verification Sources

- Wikidata API: `wbgetclaims&entity=Q164062&property=P26` — confirms deprecated rank + P2241 qualifier = Q109324480 ("not married")
- Wikidata API: `wbgetclaims&entity=Q2377989&property=P26` — same deprecation on Margaretha's entity
- Wikidata revision history for Q164062: P26 created 2018-04-05, updated (deprecated) 2021-11-16
- Wikidata revision history for Q2377989: P26 created 2018-04-05, updated (deprecated) 2021-11-16
- Wikipedia revision history for "Margaretha van Mechelen": article created 2008-07-11, always said "mistress"
- 2WikiMultiHopQA paper (Ho et al., COLING 2020): Wikidata snapshot dated December 31, 2019

---

## Summary Table

| # | Question ID | Question (abbreviated) | Error Type | Gold Answer | Correct Behavior |
|---|---|---|---|---|---|
| 1 | `fd848f86...` | Margaretha Van Mechelen's husband nationality | Presupposition failure (no marriage existed) | Dutch Republic | Refuse / "Unknown" |
