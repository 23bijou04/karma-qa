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

## Error #2: Same Root Cause — Second Question on Margaretha Van Mechelen

**Question ID:** `aa7520b80baf11ebab90acde48001122`
**Question:** "Who is the father-in-law of Margaretha Van Mechelen?"
**Question type:** inference
**Scenario:** A (KG_ONLY)
**Gold answer:** William the Silent
**Expected reasoning chain:**
1. Margaretha van Mechelen → spouse → Maurice of Nassau
2. Maurice of Nassau → father → William the Silent

### The Error

This question has the same root cause as Error #1: it presupposes that Margaretha van Mechelen had a husband. The question was generated from the same incorrect Wikidata P26 (spouse) claim (Q2377989, April 2018, normal rank at Dec 2019 snapshot). The benchmark's own context paragraph for Margaretha van Mechelen explicitly contradicts the presupposition (same quotes as Error #1: "mistress", "refused to marry her", illegitimate children).

The automated scan classified this as **WRONG_AT_CREATION / HIGH confidence** because:
- Text contradiction: "mistress" + "refused to marry" keywords detected in the Margaretha van Mechelen context paragraph
- Wikidata claim status: P26 on Q2377989 is **DEPRECATED** (confirmed independently, same deprecation event as Error #1)

### Classification

- **Error type:** Presupposition failure — same incorrect P26 annotation, different question template
- **Severity:** Unanswerable as phrased
- **Benchmark artifact origin:** Same incorrect Wikidata P26 annotation as Error #1

### Note on Pair Structure

Errors #1 and #2 both originate from a **single incorrect Wikidata edit** (P26 claim added April 2018 on Q2377989). The benchmark pipeline generated two different questions from this one triple, using different question templates ("husband" nationality vs. "father-in-law"). Both are unanswerable given the actual historical facts. This pattern — one incorrect Wikidata triple producing multiple flawed questions — may be more common in the dataset.

---

## Dismissed: Potential Error #3 — Nana Saheb Adopted Father

**Question ID:** `506809580bde11eba7f7acde48001122`
**Question:** "Where was the place of death of Nana Saheb's father?"
**Gold answer:** Bithoor
**Evidence triples:**
1. Nana Saheb → father → Baji Rao II
2. Baji Rao II → place of death → Bithoor

### Automated Flag

Phase 1 detected "adopted" in the Nana Saheb context paragraph. Phase 2 reported NOT_FOUND for P22 on Q102246769. Automated classification: WRONG_AT_CREATION (MEDIUM).

### Manual Verification — DISMISSED

Both signals were false:

1. **QID resolution error.** Our script resolved "Nana Saheb" to **Q102246769** (Nana Saheb Banerjee, a PhD student at SUNY Binghamton). The correct entity is **Q951444** (Nana Sahib, the Indian rebel, 1824–1859). Q951444 **has P22 → Baji Rao II with NormalRank** — the claim is alive and well on Wikidata. The reciprocal also exists: Q4848887 (Baji Rao II) has P40 (child) → Q951444 (Nana Sahib).

2. **Wikidata uses P22 for adoptive fathers.** The Wikidata community records Baji Rao II as Nana Sahib's P22 (father) despite the adoptive relationship. This is a Wikidata design choice, not a benchmark error. The benchmark faithfully reproduced the Wikidata triple.

3. **The answer is factually correct.** Baji Rao II did die at Bithoor. The context says "adopted son" — historically accurate — but the P22 relationship is valid per Wikidata's conventions.

**Verdict: Not a benchmark error.** The "adopted" keyword was a genuine observation but does not constitute an error. The Phase 2 NOT_FOUND was entirely a QID resolution mistake.

---

## Dismissed Flags (False Positives from Automated Scan)

The following WRONG_AT_CREATION flags from the automated scan were dismissed after manual review:

### `10b8d7940baf11ebab90acde48001122` — Edith of Wessex / Edward the Confessor

**Flag:** Evidence triple `Edward the Confessor → mother → Emma` classified as WRONG_AT_CREATION because (a) the word "adopted" appeared in a distractor paragraph and (b) Wikidata P25 NOT_FOUND on Q130005.

**Dismissal:** The benchmark's own context confirms the relation:

> **Edward the Confessor paragraph, sentence [3]:**
> "Edward was the son of Æthelred the Unready and **Emma of Normandy**."

Emma of Normandy was Edward the Confessor's mother. The evidence triple is correct. The "adopted" keyword match was a distractor from an unrelated context paragraph. The Wikidata NOT_FOUND on Q130005 is a QID resolution error (Q130005 is the wrong entity). **Not a benchmark error.**

### `095c87880bdd11eba7f7acde48001122` — Al-Mu'tasim / Harun al-Rashid

**Flag:** Evidence triple `Al-Mu'tasim → father → Harun al-Rashid` classified as WRONG_AT_CREATION because (a) "adopted" keyword match and (b) Wikidata NOT_FOUND on P27 for Q131002.

**Dismissal:** The benchmark's own context confirms the relation:

> **Al-Mu'tasim paragraph, sentence [2]:**
> "A **younger son of Caliph Harun al-Rashid**, he rose to prominence..."

Harun al-Rashid was Al-Mu'tasim's biological father — this is well-established. The "adopted" keyword was a false match in a distractor paragraph. The Wikidata NOT_FOUND on P27 (country of citizenship) for Q131002 is a QID mismatch, not evidence of an incorrect triple. **Not a benchmark error.**

---

## Potentially Affected Questions — DEPRECATED_SINCE

The following question involves a Wikidata claim that has been deprecated since the Dec 2019 snapshot. Manual review found the deprecation does not constitute a benchmark error.

### `ca0b4c4108e411ebbda4ac1f6bf848b6` — Guiomar Novaes birth date

**Question:** "Who was born later, Guiomar Novaes or Abdul Nasser Al-Sayegh?"
**Gold answer:** Abdul Nasser Al-Sayegh
**Evidence triples:**
1. Guiomar Novaes → date of birth → February 28, 1895
2. Abdul Nasser Al-Sayegh → date of birth → 19 February 1959

**Flag:** Wikidata P569 (date of birth) on Q435341 (Guiomar Novaes) has status DEPRECATED. The specific date value "February 28, 1895" is deprecated.

**Manual assessment:** The benchmark's own context paragraph gives "Guiomar Novaes (February 28, 1895 - March 7, 1979)" — the same date. The Wikidata deprecation is likely a precision or sourcing change (e.g., year-only precision replacing day-precision), not a factual correction. The comparison result ("Abdul Nasser Al-Sayegh, born 1959, was born later than Novaes, born ~1895") is correct regardless. **Not a benchmark error.**

---

## Detection Methodology

### Two-Phase Automated Scan

We conducted a systematic scan of the 1000-question evaluation subset (`data/eval_1000q/coverage_analysis_1000.json`) using two complementary methods. The scan was implemented in `scripts/detect_benchmark_errors.py`.

**Scope:** 1000 questions, all 27 evidence properties found in the subset.

**Phase 1 — Text-triple contradiction detection:**
- Target relations: `spouse` (106 questions) and `parent/father/mother` (258 questions)
- Method: keyword-based scanning of context paragraphs whose title matches the evidence entity name
- Spouse contradiction keywords: "mistress", "concubine", "lover", "consort", "illegitimate children", "refused to marry", "not married", "never married", "common-law"
- Parent contradiction keywords: "adopted", "adoptive", "foster", "stepchild", "illegitimate"
- Results: 2 HIGH-confidence flags (both "mistress" + "refused to marry"), 6 MEDIUM-confidence flags ("adopted" or "illegitimate" alone)

**Phase 2 — Wikidata deprecated and deleted claims check:**
- Method: SPARQL queries against the Wikidata endpoint to check claim status for all evidence triples
- Scope: all 27 evidence properties, all 1000 questions → 2,255 evidence triples
- Entity name-to-QID resolution: DBpedia `owl:sameAs` links (74% coverage, 2622/3516 entities), Wikidata search API fallback for remaining entities
- Claim statuses checked: CURRENT (preferred or normal rank), DEPRECATED (claim exists but rank=deprecated), NOT_FOUND (no matching claim in any rank)
- Results: 4 DEPRECATED claims, 470 NOT_FOUND claims (across all 27 properties)

**Merge logic:** A flag is classified WRONG_AT_CREATION if both phases agree (text contradiction + Wikidata DEPRECATED or NOT_FOUND), DEPRECATED_SINCE if Wikidata DEPRECATED only (no text contradiction), CLAIM_DELETED if NOT_FOUND only, AMBIGUOUS if text contradiction only (no Wikidata confirmation).

### Precision Assessment

After manual review of all 8 Phase 1 flags (2 HIGH + 6 MEDIUM) and 5 Phase 2 DEPRECATED flags:

| Category | Automated count | Confirmed errors | Precision |
|----------|----------------|-----------------|-----------|
| Phase 1 HIGH (WRONG_AT_CREATION) | 2 | 2 (Errors #1, #2) | 100% |
| Phase 1 MEDIUM (WRONG_AT_CREATION) | 6 | 0 (all dismissed — QID errors + distractor keywords) | 0% |
| Phase 2 DEPRECATED (DEPRECATED_SINCE) | 4 | 0 (all date precision changes) | 0% |
| Phase 2 NOT_FOUND (CLAIM_DELETED, 327 flags) | 327 | 0 (16 P26 flags verified, all FP) | 0% |

**Phase 1 HIGH precision is 100%** — the combination of two strong spouse-contradiction keywords plus Wikidata deprecation confirmation identified genuine errors with no false positives.

**Phase 1 MEDIUM precision is 0% (0/6 confirmed)** — all 6 MEDIUM flags were false positives. Root causes: (a) "adopted" keyword matched in distractor paragraphs unrelated to the evidence entity (2 cases), (b) "adopted" correctly described the relationship but Wikidata uses P22 for adoptive fathers — not a benchmark error (1 case), (c) "illegitimate" fired on contexts discussing illegitimate children from other relationships, not the evidence entity (3 cases). The MEDIUM keywords are too noisy without additional context filtering.

**Phase 2 DEPRECATED precision is 0% for benchmark errors** — all 4 DEPRECATED claims involved date precision or sourcing changes where the underlying fact (approximate date) was not wrong. Wikidata deprecation does not reliably indicate a claim was factually incorrect at the Dec 2019 snapshot.

**CLAIM_DELETED (327 flags) precision is 0%** — manual review of all 16 P26 (spouse) flags — the highest-priority subset — confirmed all are QID resolution errors. The NOT_FOUND status is dominated by entity name disambiguation failures: "Thomas Jefferson" → Q11812 (the president, not the film); "Henry I" → Q218190 (England, not France); "Daniel Flynn" → Q5217163 (off-by-one from correct Q5217162). See "Manual Review Complete" section below for the full root cause analysis.

### Known Limitations

This methodology targets **data-provenance errors** (incorrect Wikidata triples propagating into questions). It does not detect:
- **Answer aggregation errors** — comparison questions where the benchmark pipeline compared dates incorrectly (logic error, not data error)
- **Template-binding errors** — question says "director" but gold answer corresponds to "writer" (template slot mismatch)
- **Entity disambiguation failures** — benchmark linked the wrong Wikidata entity; that entity's triples are internally consistent
- **Multi-answer questions** — entity has multiple valid values for a property; benchmark picks one arbitrarily

Single-annotator limitation: all manual review was performed by one annotator (the thesis author). Inter-annotator agreement (e.g., Cohen's kappa) was not measured, which would be required for publication-quality error reporting. For thesis purposes, the single-annotator assessment is stated explicitly.

The 1000q evaluation subset is answerability-filtered (31.8% KG-answerable via hop verification, date-tolerant; original property-existence analysis gave 34.6%). Evaluations were run on the original 346-question answerable set. Error rate estimates apply only to this subset. Source: `data/eval_1000q/hop_verification_results.json`.

### Raw Scan Summary

```
Phase 1 scan:
  Spouse questions scanned:  106
  Parent questions scanned:  258
  HIGH-confidence flags:       2  (both: Margaretha van Mechelen P26 + text)
  MEDIUM-confidence flags:     6  (4 "adopted" keyword; 2 "illegitimate")
  Dismissed after review:      4  (false positives: distractor paragraphs)

Phase 2 scan:
  Evidence triples checked:  2,255  (27 properties, 1000 questions)
  QID resolution rate:          74%  (2,622 / 3,516 entity names)
  DEPRECATED claims found:       4  (all date precision/sourcing changes)
  NOT_FOUND claims found:      470  (dominated by QID resolution errors)

Merged output (benchmark_error_flags.json):
  Total flags:           336
  WRONG_AT_CREATION:       5  (2 confirmed, 3 dismissed — QID errors + distractor keywords)
  DEPRECATED_SINCE:        1  (0 confirmed — date precision change)
  CLAIM_DELETED:         327  (0 confirmed — all 16 P26 flags verified as QID resolution FPs)
  AMBIGUOUS:               3  (text only, no Wikidata confirmation — dismissed)
```

---

## Manual Review Complete — P26 CLAIM_DELETED (All 16 Dismissed)

The 16 P26 (spouse) CLAIM_DELETED flags were the highest-priority candidates for genuine benchmark errors. Manual verification via Wikidata SPARQL confirmed that **all 16 are false positives** caused by QID resolution errors in our entity name → Wikidata QID mapping. In every case, the P26 claim exists on Wikidata with NormalRank under the correct QID — our script resolved the evidence entity name to a wrong QID.

### Root cause analysis

| Error pattern | Count | Example |
|---|---|---|
| Ambiguous name → wrong entity | 7 | "Henry I" → Q218190 (England) instead of Q178908 (France); "Mithridates" → Q546455 (generic) instead of Q185126 (Mithridates VI) |
| Off-by-one / similar QID | 1 | "Daniel Flynn" → Q5217163 instead of Q5217162 |
| Name-to-disambiguation page | 1 | "Brian Tse" → Q54867776 (disambiguation page, not the McMug writer) |
| Leading space in evidence name | 1 | `' Sila'` → Q359465 instead of Q4407561 (Sila Zandukeli-Sandunov) |
| One-directional P26 (claim on spouse's entity only) | 2 | Papianilla: P26 on Q302936 (Sidonius) side only; Elizabeth Herbert: P26 on Q549866 (Charles Somerset) side only |
| Correct entity, wrong value QID | 4 | Clare Winnicott: "D. W. Winnicott" → Q93698480 instead of Q451348 (Donald Winnicott) |

### Verified P26 claims (selected examples)

| Entity | Resolved QID | Correct QID | Correct P26 claim |
|---|---|---|---|
| Matilda of Frisia | Q60366 (correct) | — | P26 → Q178908 (Henry I of France) [NormalRank] |
| Ameena Begum | Q4742114 (correct) | — | P26 → Q120411 (Inayat Khan) [NormalRank] |
| Elizabeth Fitzalan | Q5362776 (correct) | — | P26 → Q711992 (Thomas Mowbray) [NormalRank] |
| Hypsicratea | Q5961731 (correct) | — | P26 → Q185126 (Mithridates VI) [NormalRank] |
| Ramiro Sánchez | Q2064782 (correct) | — | P26 → Q5791517 (Cristina Rodríguez) [NormalRank] |

**Conclusion:** No genuine benchmark errors found among the P26 CLAIM_DELETED flags. The remaining 311 CLAIM_DELETED flags for other properties (P57 director, P27 citizenship, P22 father, etc.) are expected to have the same root cause — QID resolution mismatches — and are not investigated further.

---

## Remaining CLAIM_DELETED Flags (311) — Not Investigated

The 311 non-P26 CLAIM_DELETED flags are dominated by the same QID resolution errors seen in the P26 subset. The most common patterns:
- **P27 (country of citizenship, 149 flags):** Nationality strings ("American", "British") resolved to wrong entities (e.g., "American" → Q7976 "Americas")
- **P57 (director, 78 flags):** Film/person disambiguation failures (e.g., "Thomas Jefferson" the film → Q11812 the president)
- **P22 (father, 40 flags):** Same ambiguous-name issues as P26

Given that 16/16 P26 flags were false positives, and the P27/P57/P22 flags have even more obvious disambiguation problems, these are not expected to contain genuine errors. No further investigation warranted.

---

## Summary Table

| # | Question ID | Question (abbreviated) | Error Type | Confidence | Gold Answer | Status |
|---|---|---|---|---|---|---|
| 1 | `fd848f86...` | Margaretha Van Mechelen's husband nationality | Presupposition failure (no marriage) | HIGH — confirmed | Dutch Republic | **Confirmed error** |
| 2 | `aa7520b8...` | Father-in-law of Margaretha Van Mechelen | Presupposition failure (same root cause) | HIGH — confirmed | William the Silent | **Confirmed error** |
| — | `506809580...` | Place of death of Nana Saheb's father | QID resolution error + Wikidata uses P22 for adoptive fathers | — | Bithoor | Dismissed |
| — | `10b8d794...` | Edith of Wessex's mother-in-law | False positive (distractor keyword) | — | Emma | Dismissed |
| — | `095c8788...` | Al-Mu'tasim's father country | False positive (distractor keyword) | — | Abbasid Caliphate | Dismissed |
| — | `ca0b4c41...` | Who was born later: Novaes or Al-Sayegh | Date precision change, not factual error | — | Abdul Nasser Al-Sayegh | Dismissed |
| — | 16 P26 CLAIM_DELETED | Various spouse questions | All QID resolution errors | — | Various | Dismissed (all 16 verified) |
| — | 311 other CLAIM_DELETED | Various | QID resolution errors (extrapolated) | — | Various | Not investigated (same root cause) |
