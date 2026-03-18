# Scenario Evaluation: 60-Question Case-by-Case Analysis

**Date:** 2026-03-17
**Data:** `data/eval_1000q/scenario_eval_hybrid.jsonl` (60 questions, 10 per scenario A–F)
**Mode:** Hybrid pipeline (`hybrid_qa.py`)

## Summary

| Scenario | Label | EM | F1 | Correct/Total |
|----------|-------|---:|---:|:-------------:|
| A | KG Complete | 90% | 94% | 9/10 |
| B | Hop-1 Missing | 70% | 70% | 7/10 |
| C | Hop-2 Missing | 40% | 48% | 4/10 |
| D | Both Hops Missing | 10% | 18% | 1/10 |
| E | Seed Not Found | 40% | 67% | 4/10 |
| F | No Entity + No Prop | 20% | 34% | 2/10 |
| **Overall** | | **45%** | **55%** | **27/60** |

## Failure Taxonomy

| Failure Type | Count | % of Failures | Affected Scenarios | Fixable? |
|---|---|---|---|---|
| Over-verbose answer (containment=1.0) | 7 | 21% | A(1), C(1), D(1), E(3), F(1) | **Yes** — post-processing |
| Info not found / Unknown | 8 | 24% | C(1), D(3), F(4) | Hard — data availability |
| Wrong entity / wrong person | 6 | 18% | B(1), C(2), D(1), F(2) | Partial — better entity linking |
| Wrong factual answer | 3 | 9% | C(1), D(2) | Hard — requires better sources |
| Semantic granularity mismatch | 3 | 9% | B(1), D(1), E(1) | Hard — Hollywood≠LA |
| Wrong comparison result | 2 | 6% | B(1), C(1) | Partial — date validation |
| Format mismatch (date/title) | 2 | 6% | E(2) | **Yes** — normalization |
| Wrong answer type | 2 | 6% | D(1), F(1) | Partial — prompt tuning |

**Key insight:** 9 of 33 failures (27%) are fixable with post-processing (over-verbose + format normalization). Fixing these would lift overall EM from 45% → 60%.

---

## Scenario A: KG Complete — 90% EM, 94% F1

Both hops fully resolved in DBpedia. SPARQL should handle end-to-end.

### Q1 ✅ EM=1.0 | logicrag_decomposed | 26.0s
- **Q:** Which film has the director born earlier, The Spooky Bunch or His Name Is Nobody?
- **Gold:** His Name Is Nobody → **Predicted:** His Name Is Nobody
- **Route:** `incomplete_coverage` — routed to LogicRAG despite KG being complete (routing inefficiency)
- **Note:** Correct but slow — SPARQL would have solved in ~2s

### Q2 ✅ EM=1.0 | sparql | 1.9s
- **Q:** Which film has the director who was born earlier, The Red Eagle or Vishukkani?
- **Gold:** Vishukkani → **Predicted:** Vishukkani
- **Route:** `rule:bridge_comparison+ask_verified` — fast SPARQL path

### Q3 ✅ EM=1.0 | sparql | 8.9s
- **Q:** Which film has the director died earlier, John Jaffer Janardhanan or Kamakalawa?
- **Gold:** Kamakalawa → **Predicted:** Kamakalawa
- **Route:** `rule:bridge_comparison+ask_verified`

### Q4 ❌ EM=0.0 F1=0.4 | sparql | 2.4s | **OVER-VERBOSE**
- **Q:** Where did Adolph Of Cleves, Lord Of Ravenstein's mother die?
- **Gold:** Kalkar → **Predicted:** Monterberg Castle, Kalkar, Cleves
- **Route:** `rule:direct_property`
- **Containment:** 1.0 — answer contains gold
- **Root cause:** SPARQL returned full deathPlace value (`Monterberg Castle, Kalkar, Cleves`) instead of just the city. DBpedia stores multi-part location strings.
- **Fix:** Post-process to extract city-level location, or normalize SPARQL results to match gold granularity.

### Q5 ✅ EM=1.0 | sparql | 1.6s
- **Q:** Who died first, Thomas Valentine Cooper or John Ellis Roosevelt?
- **Gold:** Thomas Valentine Cooper → **Predicted:** Thomas Valentine Cooper
- **Route:** `rule:direct_date_comparison`

### Q6 ✅ EM=1.0 | sparql | 2.7s
- **Q:** Which film director born earlier, Pod Banderą Miłości or World In My Corner?
- **Gold:** Pod Banderą Miłości → **Predicted:** Pod Banderą Miłości
- **Route:** `rule:bridge_comparison+ask_verified`

### Q7 ✅ EM=1.0 | sparql | 8.9s
- **Q:** Who died later, Alicia Little or Christopher Addison, 1St Viscount Addison?
- **Gold:** Christopher Addison, 1St Viscount Addison → **Predicted:** Christopher Addison, 1st Viscount Addison
- **Route:** `rule:direct_date_comparison`

### Q8 ✅ EM=1.0 | sparql | 2.7s
- **Q:** Which film director born earlier, The Korean Wedding Chest or True To The Navy?
- **Gold:** True To The Navy → **Predicted:** True to the Navy
- **Route:** `rule:bridge_comparison+ask_verified`

### Q9 ✅ EM=1.0 | sparql | 1.6s
- **Q:** Which film director died later, Chad Hanna or The Avenging Shadow?
- **Gold:** Chad Hanna → **Predicted:** Chad Hanna
- **Route:** `rule:bridge_comparison+ask_verified`

### Q10 ✅ EM=1.0 | sparql | 1.4s
- **Q:** Which film director died later, Hotel Paradis or Devil'S Doorway?
- **Gold:** Devil'S Doorway → **Predicted:** Devil's Doorway
- **Route:** `rule:bridge_comparison+ask_verified`

### Scenario A Summary
- **9/10 correct.** 8 solved via direct SPARQL (1.4–8.9s), 1 via LogicRAG (26s routing miss).
- **1 failure:** Over-verbose SPARQL result (Q4). The data is right; the answer extraction is too literal.
- **Action items:**
  - [ ] Investigate Q1 routing: why did `incomplete_coverage` fire for a KG-complete question?
  - [ ] Add location normalization post-processing for SPARQL answers

---

## Scenario B: Hop-1 Missing — 70% EM, 70% F1

Seed entity exists but lacks hop-1 property. Bridge entity has hop-2 data. LogicRAG TEXT→SPARQL chain expected.

### Q11 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 27.6s | **WRONG ENTITY**
- **Q:** What is the place of birth of the performer of song Changed It?
- **Gold:** Port of Spain → **Predicted:** New Orleans, Louisiana
- **Route:** `rule:bridge_hop2+ask_missing`
- **Root cause:** Identified wrong performer. "Changed It" likely by Nicki Minaj (born Port of Spain), but system found a different artist (possibly Lil Wayne, born New Orleans). Hop-1 disambiguation failure when property is missing.
- **Fix:** Improve performer resolution — verify performer identity via multiple Wikipedia sources before resolving hop-2.

### Q12 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 11.5s | **GRANULARITY MISMATCH**
- **Q:** Where did the composer of song You've Got What Gets Me die?
- **Gold:** Hollywood → **Predicted:** Los Angeles
- **Route:** `no_schema_fallback`
- **Root cause:** Hollywood is a neighborhood within Los Angeles. Semantically correct but fails EM. Gold answer uses neighborhood-level specificity that the pipeline doesn't produce.
- **Fix:** Hard to fix systematically — would need location hierarchy mapping.

### Q13 ✅ EM=1.0 | logicrag_decomposed | 19.5s
- **Q:** What is the place of birth of Bea Ballard's father?
- **Gold:** Shanghai → **Predicted:** Shanghai
- **Route:** `rule:bridge_hop2+ask_missing`

### Q14 ✅ EM=1.0 | logicrag_decomposed | 32.6s
- **Q:** Who was born earlier, Elmer W. Conti or Seth Joshua?
- **Gold:** Seth Joshua → **Predicted:** Seth Joshua
- **Route:** `llm_prop:no_match` — LLM fallback routing (no direct property match)

### Q15 ✅ EM=1.0 | logicrag_decomposed | 23.6s
- **Q:** Which film was released more recently, Mystify: Michael Hutchence or The Man On The Roof?
- **Gold:** Mystify: Michael Hutchence → **Predicted:** Mystify: Michael Hutchence
- **Route:** `incomplete_coverage`

### Q16 ✅ EM=1.0 | logicrag_decomposed | 11.1s
- **Q:** Where was the place of death of Aleksandr Aleksandrovich Bibikov's father?
- **Gold:** Bugulma → **Predicted:** Bugulma
- **Route:** `rule:bridge_hop2+ask_missing`

### Q17 ✅ EM=1.0 | logicrag_decomposed | 22.4s
- **Q:** Who is the paternal grandfather of Engelbert III Of The Mark?
- **Gold:** Engelbert II of the Mark → **Predicted:** Engelbert II of the Mark
- **Route:** `llm_prop:rule:bridge_hop2+ask_missing`

### Q18 ❌ EM=0.0 F1=0.0 | sparql | 6.8s | **WRONG COMPARISON**
- **Q:** Who was born first out of C. Maurice Patterson and Maj-Briht Bergström-Walan?
- **Gold:** C. Maurice Patterson → **Predicted:** Maj-Briht Bergström-Walan
- **Route:** `rule:direct_date_comparison`
- **Root cause:** SPARQL routed as direct date comparison, but one entity has incomplete/missing birth date in KG. SPARQL comparison produced wrong result on incomplete data.
- **Fix:** Validate that both birth dates exist in SPARQL before trusting comparison. Fall back to text if one date is missing.

### Q19 ✅ EM=1.0 | logicrag_decomposed | 12.8s
- **Q:** Where was the director of film Katka's Reinette Apples born?
- **Gold:** Rēzekne → **Predicted:** Rēzekne
- **Route:** `no_schema_fallback`

### Q20 ✅ EM=1.0 | logicrag_decomposed | 18.6s
- **Q:** Which film has the producer died later, A Midnight Romance or The Bedford Incident?
- **Gold:** The Bedford Incident → **Predicted:** The Bedford Incident
- **Route:** `incomplete_coverage`

### Scenario B Summary
- **7/10 correct.** Mostly solved by LogicRAG TEXT→SPARQL chain.
- **3 failures:** Wrong entity (Q11), granularity mismatch (Q12), wrong SPARQL comparison on incomplete data (Q18).
- **Action items:**
  - [ ] Add date-existence validation before SPARQL comparisons (Q18 class of bugs)
  - [ ] Investigate performer/composer disambiguation for songs (Q11)

---

## Scenario C: Hop-2 Missing — 40% EM, 48% F1

Hop-1 resolves via SPARQL; bridge entity lacks hop-2 property. LogicRAG SPARQL→TEXT chain expected.

### Q21 ✅ EM=1.0 | logicrag_decomposed | 45.1s
- **Q:** Who is the spouse of the director of film McDull: Me & My Mum?
- **Gold:** Alice Mak → **Predicted:** Alice Mak
- **Route:** `rule:direct_property` → sparql_fallback=true (SPARQL failed, fell through to LogicRAG)

### Q22 ✅ EM=1.0 | logicrag_decomposed | 28.3s
- **Q:** Where was the place of death of Licinia Eudoxia's father?
- **Gold:** Constantinople → **Predicted:** Constantinople
- **Route:** `rule:bridge_hop2+ask_verified` → sparql_fallback=true

### Q23 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 20.5s | **WRONG ENTITY**
- **Q:** Where was the director of film The Poll Diaries born?
- **Gold:** Göttingen → **Predicted:** The Bronx, New York
- **Route:** `rule:bridge_hop2+ask_verified` → sparql_fallback=true
- **Root cause:** SPARQL resolved hop-1 (director) but got the wrong director, or text retrieval for hop-2 confused the person. Predicted birthplace (Bronx) is completely unrelated to gold (Göttingen, Germany).
- **Fix:** Cross-validate director identity between SPARQL result and Wikipedia article before proceeding to hop-2.

### Q24 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 32.8s | **INFO NOT FOUND**
- **Q:** Where was the place of death of the director of film A Rare Bird?
- **Gold:** Le Plessis-Bouchard → **Predicted:** Unknown
- **Route:** `rule:bridge_hop2+ask_verified` → sparql_fallback=true
- **Root cause:** Director resolved via SPARQL, but Wikipedia text for the bridge entity didn't contain death location. Obscure French director — Wikipedia article may be sparse.
- **Fix:** Try multiple Wikipedia sources (French Wikipedia via interlanguage links?), or Wikidata fallback.

### Q25 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 22.8s | **WRONG FACTUAL ANSWER**
- **Q:** Do both Possession (1922 Film) and La Boum films have the directors from the same country?
- **Gold:** yes → **Predicted:** No
- **Route:** `incomplete_coverage`
- **Root cause:** Failed to identify both directors as French. Likely misidentified one director's nationality.
- **Fix:** For yes/no comparison questions, validate both entities' nationalities independently before comparing.

### Q26 ✅ EM=1.0 | logicrag_decomposed | 86.5s
- **Q:** Which film has the director born first, Sins Of Madeleine or Captain Apache?
- **Gold:** Sins Of Madeleine → **Predicted:** Sins Of Madeleine
- **Route:** `incomplete_coverage`
- **Note:** Correct but extremely slow (86s).

### Q27 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 9.1s | **WRONG ENTITY**
- **Q:** When was Princess Feodora Of Denmark's father born?
- **Gold:** 8 October 1876 → **Predicted:** July 15, 1750
- **Route:** `rule:bridge_hop2+ask_verified` → sparql_fallback=true
- **Root cause:** Retrieved wrong father — date is off by 126 years. SPARQL likely resolved `father` to a different ancestor. Multiple "Princess Feodora" entities exist; entity disambiguation failure.
- **Fix:** Validate entity identity (check if person/dates are plausible for the question context).

### Q28 ❌ EM=0.0 F1=0.8 | sparql | 2.6s | **OVER-VERBOSE**
- **Q:** Where was the director of film The Half-Way Girl born?
- **Gold:** New York → **Predicted:** New York City
- **Route:** `rule:bridge_hop2+ask_verified`
- **Containment:** 1.0
- **Root cause:** DBpedia stores "New York City" while gold expects "New York". Same place, different string.
- **Fix:** Post-processing: strip "City" suffix for location answers.

### Q29 ✅ EM=1.0 | sparql | 5.2s
- **Q:** Which film has the director who was born later, The Sleeping City or The Lawful Cheater?
- **Gold:** The Sleeping City → **Predicted:** The Sleeping City
- **Route:** `rule:bridge_comparison+ask_verified`

### Q30 ❌ EM=0.0 F1=0.0 | sparql | 34.7s | **WRONG COMPARISON**
- **Q:** Which film has the director born earlier, Mummy & Me or Gideon And Samson?
- **Gold:** Gideon And Samson: Great Leaders Of The Bible → **Predicted:** Mummy & Me
- **Route:** `rule:bridge_comparison+ask_verified`
- **Root cause:** SPARQL returned wrong comparison result. One director's birth date likely missing or incorrect in KG, leading to wrong comparison.
- **Fix:** Same as Q18 — validate both dates exist before trusting SPARQL comparison.

### Scenario C Summary
- **4/10 correct.** Mixed SPARQL + LogicRAG paths.
- **6 failures:** Wrong entity (Q23, Q27), info not found (Q24), wrong factual (Q25), over-verbose (Q28), wrong comparison (Q30).
- **Action items:**
  - [ ] Cross-validate entity identity between SPARQL and Wikipedia (Q23, Q27)
  - [ ] Add date-existence guard for bridge_comparison SPARQL (Q30)
  - [ ] Location normalization post-processing (Q28)

---

## Scenario D: Both Hops Missing — 10% EM, 18% F1

No structured KG data for either hop. Fully text-dependent. Hardest factual scenario.

### Q31 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 58.6s | **INFO NOT FOUND**
- **Q:** Who is Robert Vere Buxton's paternal grandfather?
- **Gold:** Sir Edward Buxton, 2nd Baronet → **Predicted:** Not specified
- **Route:** `no_schema_fallback`
- **Root cause:** Wikipedia article didn't contain explicit paternal grandfather information. Genealogy data is often sparse in Wikipedia prose.

### Q32 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 109.2s | **WRONG FACTUAL ANSWER**
- **Q:** Which award the director of film The Dream Is Alive earned?
- **Gold:** Member of the Order of Canada → **Predicted:** Special Achievement Genie
- **Route:** `rule:direct_property` → sparql_fallback=true
- **Root cause:** Director has multiple awards. Text retrieval found a different award (Genie Award) instead of the one matching gold (Order of Canada). Ambiguous question — director has many awards.

### Q33 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 66.6s | **INFO NOT FOUND**
- **Q:** Where was the place of death of Archibald Hamilton (1790–1815)'s father?
- **Gold:** Beaufort → **Predicted:** Unknown
- **Route:** `rule:bridge_hop2+ask_missing`
- **Root cause:** Obscure historical figure. Wikipedia likely has minimal information about the father's death location.

### Q34 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 26.6s | **WRONG ENTITY**
- **Q:** What nationality is Margaretha Van Mechelen's husband?
- **Gold:** Dutch Republic → **Predicted:** American
- **Route:** `llm_prop:no_match`
- **Root cause:** Resolved to completely wrong person. Margaretha Van Mechelen was a 16th-century noblewoman (husband: William of Orange), but system found a different person and returned "American".

### Q35 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 30.1s | **WRONG FACTUAL ANSWER**
- **Q:** Which country the director of film Littlerock (Film) is from?
- **Gold:** Filipino → **Predicted:** United States
- **Route:** `rule:bridge_hop2+ask_missing`
- **Root cause:** Director Mike Ott is Filipino-American. System returned "United States" (country of residence) instead of "Filipino" (ethnic origin). Gold answer uses ethnic identity, not citizenship.

### Q36 ❌ EM=0.0 F1=0.0 | sparql | 2.3s | **WRONG ANSWER TYPE**
- **Q:** Which award the performer of song You Changed Me won?
- **Gold:** Best Actor → **Predicted:** List of awards and nominations received by Jamie Foxx
- **Route:** `rule:direct_property`
- **Root cause:** SPARQL returned the URI/label of the awards list page instead of an actual award name. The SPARQL query pattern fetched `dbo:award` which pointed to the list article, not a specific award.
- **Fix:** Filter out "List of..." results from SPARQL answers. Or detect this pattern and fall through to text.

### Q37 ✅ EM=1.0 | logicrag_decomposed | 33.7s
- **Q:** What nationality is the director of film Paradox (2016 Film)?
- **Gold:** New Zealand → **Predicted:** New Zealand
- **Route:** `rule:bridge_hop2+ask_verified` → sparql_fallback=true
- **Note:** Only correct answer in scenario D — Wikipedia text had clear nationality info.

### Q38 ❌ EM=0.0 F1=0.8 | logicrag_decomposed | 35.0s | **OVER-VERBOSE**
- **Q:** Where was the place of death of Thaddeus P. Mott's father?
- **Gold:** New York → **Predicted:** Brooklyn, New York
- **Route:** `rule:bridge_hop2+ask_missing`
- **Containment:** 1.0
- **Root cause:** Returned "Brooklyn, New York" instead of just "New York". More specific than gold.
- **Fix:** Location normalization post-processing.

### Q39 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 54.0s | **INFO NOT FOUND**
- **Q:** Which film whose director was born first, Tree Without Fruit or Small Town Boy?
- **Gold:** Small Town Boy → **Predicted:** Unknown
- **Route:** `rule:bridge_comparison+ask_verified` → sparql_fallback=true
- **Root cause:** Couldn't find birth dates for either director in Wikipedia text.

### Q40 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 44.0s | **SEMANTIC MISMATCH**
- **Q:** Which country Lucius Antonius (Grandson Of Mark Antony)'s father is from?
- **Gold:** ancient Rome → **Predicted:** Italy
- **Route:** `rule:bridge_hop2+ask_missing`
- **Root cause:** "Italy" is the modern country; "ancient Rome" is the historical polity. Semantically close but factually different time period. LLM defaulted to modern geography.

### Scenario D Summary
- **1/10 correct.** Entirely text-dependent, highly error-prone.
- **9 failures:** Info not found (Q31, Q33, Q39), wrong entity (Q34), wrong factual (Q32, Q35), semantic mismatch (Q40), wrong answer type (Q36), over-verbose (Q38).
- **Action items:**
  - [ ] Filter "List of..." SPARQL results (Q36)
  - [ ] For ambiguous multi-value answers (awards), prefer most notable or prompt for specificity
  - [ ] Location normalization (Q38)
  - [ ] Historical vs modern geography awareness in prompts (Q40)

---

## Scenario E: Seed Entity Not Found — 40% EM, 67% F1

Seed entity not in DBpedia. Must rely on Wikipedia search + text retrieval.

**Critical observation:** F1 (67%) >> EM (40%). Most "failures" are actually correct answers in wrong format.

### Q41 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 31.1s | **GRANULARITY MISMATCH**
- **Q:** Where did the composer of song Aren't You Kind Of Glad We Did? die?
- **Gold:** Hollywood → **Predicted:** Los Angeles, California
- **Route:** `llm_prop:no_match`
- **Root cause:** Same issue as Q12 — Hollywood⊂Los Angeles. Semantically correct but fails EM.

### Q42 ✅ EM=1.0 | sparql | 9.4s
- **Q:** Who is Infanta Adelgundes, Duchess Of Guimarães's father?
- **Gold:** Miguel I of Portugal → **Predicted:** Miguel I of Portugal
- **Route:** `rule:direct_property`
- **Note:** Despite seed being classified as "not found", SPARQL still resolved. Entity linking worked via alternative path.

### Q43 ✅ EM=1.0 | text | 26.5s
- **Q:** Who did Joan Ramon II, Count Of Cardona marry?
- **Gold:** Joana de Prades → **Predicted:** Joana de Prades
- **Route:** `llm_prop:rule:bridge_hop2+ask_missing`
- **Note:** Pure text retrieval success on a medieval historical figure.

### Q44 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 23.5s | **FORMAT MISMATCH**
- **Q:** What is the date of death of the founder of University Of Los Andes (Colombia)?
- **Gold:** July 16, 2013 → **Predicted:** 2013-07-16
- **Route:** `rule:bridge_hop2+ask_missing`
- **Root cause:** Same date, different format. ISO 8601 vs English text. This is a pure evaluation metric artifact.
- **Fix:** Date normalization in answer post-processing or metric computation.

### Q45 ✅ EM=1.0 | logicrag_decomposed | 26.2s
- **Q:** Where was the wife of Alfred L. Copley born?
- **Gold:** Seyðisfjörður → **Predicted:** Seyðisfjörður
- **Route:** `rule:direct_property` → sparql_fallback=true

### Q46 ❌ EM=0.0 F1=0.4 | logicrag_decomposed | 9.3s | **OVER-VERBOSE**
- **Q:** Where did Prince Ferdinand Of Bavaria's mother die?
- **Gold:** Munich → **Predicted:** Nymphenburg Palace, Munich, Germany
- **Route:** `llm_prop:no_match`
- **Containment:** 1.0
- **Root cause:** Answer contains "Munich" but adds palace name and country.
- **Fix:** Location normalization post-processing.

### Q47 ✅ EM=1.0 | sparql | 4.1s
- **Q:** Who is the paternal grandfather of Hamad Bin Isa Al Khalifa?
- **Gold:** Salman bin Hamad Al Khalifa → **Predicted:** Salman bin Hamad Al Khalifa
- **Route:** `rule:direct_property`

### Q48 ❌ EM=0.0 F1=0.86 | sparql | 3.9s | **FORMAT MISMATCH**
- **Q:** Who is the maternal grandmother of Princess Catherine Of Württemberg?
- **Gold:** Henriette of Nassau-Weilburg → **Predicted:** Princess Henriette of Nassau-Weilburg
- **Route:** `rule:direct_property`
- **Containment:** 1.0
- **Root cause:** Added "Princess" title prefix. DBpedia stores full title, gold uses shorter form.
- **Fix:** Strip common title prefixes (Princess, Prince, Sir, etc.) or use fuzzy matching.

### Q49 ❌ EM=0.0 F1=0.8 | logicrag_decomposed | 33.5s | **OVER-VERBOSE**
- **Q:** Where was the father of Tony Bettenhausen Jr. born?
- **Gold:** Tinley Park → **Predicted:** Tinley Park, Illinois
- **Route:** `rule:bridge_hop2+ask_missing`
- **Containment:** 1.0
- **Root cause:** Added state name. Gold expects city only.
- **Fix:** Location normalization post-processing.

### Q50 ❌ EM=0.0 F1=0.67 | logicrag_decomposed | 35.0s | **OVER-VERBOSE**
- **Q:** Where was the founder of Bolivarian Military University Of Venezuela born?
- **Gold:** Sabaneta → **Predicted:** Sabaneta, Venezuela
- **Route:** `rule:bridge_hop2+ask_missing`
- **Containment:** 1.0
- **Root cause:** Added country name. Gold expects city only.
- **Fix:** Location normalization post-processing.

### Scenario E Summary
- **4/10 EM but effectively ~8/10 correct** when counting containment=1.0 answers.
- **6 "failures" breakdown:**
  - 4 over-verbose (Q46, Q48, Q49, Q50) — all containment=1.0, all fixable
  - 1 date format mismatch (Q44) — fixable
  - 1 granularity mismatch (Q41) — hard to fix
- **This scenario's EM is artificially depressed by answer format issues, not factual errors.**
- **Action items:**
  - [ ] **HIGH PRIORITY:** Add location normalization (strip state/country suffixes) — fixes 4 questions
  - [ ] Add date normalization (ISO↔text) — fixes 1 question
  - [ ] Strip title prefixes from person names — fixes 1 question

---

## Scenario F: No Entity + No Property — 20% EM, 34% F1

Worst case: entity linking fails AND needed properties missing from KG. Double text fallback.

### Q51 ✅ EM=1.0 | logicrag_decomposed | 27.4s
- **Q:** Who is Robert Petre, 8th Baron Petre's paternal grandmother?
- **Gold:** Mary Clifton → **Predicted:** Mary Clifton
- **Route:** `llm_prop:no_match`

### Q52 ❌ EM=0.0 F1=0.5 | logicrag_decomposed | 265.0s | **WRONG ENTITY**
- **Q:** Who is the uncle of Filippa Fleming?
- **Gold:** Ivar Fleming → **Predicted:** Erik Fleming
- **Route:** `llm_prop:no_match`
- **Root cause:** Found a different Fleming family member. Wikipedia text mentions multiple uncles or the system confused family members.
- **Note:** Extremely slow (265s).

### Q53 ❌ EM=0.0 F1=0.5 | sparql | 2.8s | **OVER-VERBOSE**
- **Q:** Where did Sonia Greene's husband die?
- **Gold:** Providence → **Predicted:** Providence, Rhode Island
- **Route:** `llm_prop:rule:bridge_hop2+ask_verified`
- **Containment:** 1.0
- **Fix:** Location normalization.

### Q54 ❌ EM=0.0 F1=0.0 | text | 85.0s | **INFO NOT FOUND**
- **Q:** Who is Fernando Bermúdez De Cea's uncle?
- **Gold:** Oveco Núñez → **Predicted:** Could not find answer
- **Root cause:** Obscure medieval Spanish figure. Wikipedia article too sparse.

### Q55 ❌ EM=0.0 F1=0.0 | text | 492.9s | **INFO NOT FOUND**
- **Q:** Who is the husband of Princess Elene Of Georgia?
- **Gold:** Prince Archil of Imereti → **Predicted:** Could not find answer
- **Root cause:** Very obscure Georgian royalty. Text fallback exhausted all sources.
- **Note:** 493s — extreme timeout waste on unanswerable question.

### Q56 ❌ EM=0.0 F1=0.4 | logicrag_decomposed | 171.7s | **WRONG ENTITY**
- **Q:** Who is the father-in-law of Lady Rose McLaren?
- **Gold:** Henry McLaren → **Predicted:** John Francis McLaren
- **Route:** `rule:direct_property` → sparql_fallback=true
- **Root cause:** Found wrong generation of McLaren family. Wikipedia text may mention multiple McLarens.

### Q57 ❌ EM=0.0 F1=0.0 | text | 560.8s | **INFO NOT FOUND**
- **Q:** Where did Sholto Douglas, 15th Earl Of Morton's father study?
- **Gold:** King's College → **Predicted:** Could not find answer
- **Root cause:** Obscure Scottish nobility. Education info not in Wikipedia.
- **Note:** 561s — very long timeout.

### Q58 ❌ EM=0.0 F1=0.0 | logicrag_decomposed | 268.9s | **WRONG ANSWER TYPE**
- **Q:** What nationality is the performer of song When The Stars Go Blue?
- **Gold:** America → **Predicted:** Ryan Adams
- **Route:** `rule:bridge_hop2+ask_verified` → sparql_fallback=true
- **Root cause:** Returned person name instead of nationality. The system resolved the performer but failed to answer the *actual question* (nationality). Answer extraction or final synthesis prompt failure.
- **Fix:** Verify that the answer matches the expected answer type (person, place, date, etc.).

### Q59 ❌ EM=0.0 F1=0.0 | text | 654.2s | **INFO NOT FOUND**
- **Q:** Who is the paternal grandfather of Prince Charles Louis, Duke Of Chartres?
- **Gold:** Henri, Count of Paris → **Predicted:** Could not find answer
- **Route:** `rule:direct_property` → sparql_fallback=true
- **Note:** 654s — slowest question in the entire eval. Text fallback spends enormous time searching before giving up.

### Q60 ✅ EM=1.0 | logicrag_decomposed | 9.1s
- **Q:** Where was the place of death of Abdul-Aziz Bin Muhammad's father?
- **Gold:** Diriyah → **Predicted:** Diriyah
- **Route:** `rule:bridge_hop2+ask_missing`

### Scenario F Summary
- **2/10 correct.** 4 "Could not find answer" (all extremely slow), 2 wrong entity, 1 over-verbose, 1 wrong answer type.
- **Performance issue:** "Could not find answer" questions average **448s** each. The text fallback exhaustively searches Wikipedia before giving up.
- **Action items:**
  - [ ] **HIGH PRIORITY:** Add timeout/early-exit for text fallback (cap at ~60s instead of 500s+)
  - [ ] Answer type validation: ensure final answer matches expected type (Q58)
  - [ ] Location normalization (Q53)
  - [ ] Consider "unanswerable" detection — if entity linking fails + no Wikipedia results after N attempts, exit early

---

## Actionable Improvements (Priority Ordered)

### P0 — Quick Wins (fix 9+ questions, no pipeline changes)

| Fix | Questions Fixed | Implementation |
|-----|:-:|---|
| **Location normalization** — strip state/country/building prefixes from location answers | Q4, Q28, Q38, Q46, Q49, Q50, Q53 (7) | Post-processing in `extract_final_answer()` or new `normalize_answer()` |
| **Date normalization** — ISO↔text format | Q44 (1) | Post-processing |
| **Title prefix stripping** — remove Princess/Prince/Sir from person names | Q48 (1) | Post-processing |
| **Expected EM lift:** 45% → **60%** (+15 points) | | |

### P1 — Medium Effort (fix 3–5 questions)

| Fix | Questions Fixed | Implementation |
|-----|:-:|---|
| **Date-existence validation** for SPARQL comparisons | Q18, Q30 (2) | Guard in `sparql_solver.py` comparison path |
| **Filter "List of..." SPARQL results** | Q36 (1) | Post-filter in SPARQL result processing |
| **Answer type validation** — verify answer matches expected type | Q58 (1) | Check in final synthesis |
| **Text fallback timeout** — cap at 60s | Q55, Q57, Q59 (latency, not accuracy) | Timeout in `text_solver.py` |

### P2 — Hard (structural improvements)

| Fix | Questions Affected | Implementation |
|-----|:-:|---|
| **Better entity disambiguation** for songs/performers | Q11, Q23, Q27, Q34 | Multi-source validation in entity resolution |
| **Cross-validate SPARQL↔Wikipedia** entity identity | Q23, Q27 | Compare SPARQL result against Wikipedia article |
| **Historical geography awareness** | Q40 | Prompt engineering for temporal context |
| **Unanswerable detection** | Q31, Q33, Q54, Q55, Q57, Q59 | Confidence scoring + early exit |

---

## Method Distribution

| Scenario | sparql | logicrag_decomposed | text | Total |
|----------|-------:|--------------------:|-----:|------:|
| A | 9 | 1 | 0 | 10 |
| B | 1 | 9 | 0 | 10 |
| C | 3 | 7 | 0 | 10 |
| D | 1 | 9 | 0 | 10 |
| E | 3 | 6 | 1 | 10 |
| F | 1 | 5 | 4 | 10 |
| **Total** | **18** | **37** | **5** | **60** |

## Timing Analysis

| Scenario | Avg Time (s) | Min | Max | Note |
|----------|-------------:|----:|----:|------|
| A | 5.8 | 1.4 | 26.0 | Fast — mostly SPARQL |
| B | 18.7 | 6.8 | 32.6 | Moderate — LogicRAG decomposition |
| C | 28.8 | 2.6 | 86.5 | Variable — depends on fallback depth |
| D | 46.0 | 2.3 | 109.2 | Slow — double text retrieval |
| E | 20.3 | 3.9 | 35.0 | Moderate |
| F | 254.7 | 2.8 | 654.2 | **Extremely slow** — text exhaustion |

**Scenario F timeout is the biggest operational issue.** 4 questions > 100s, 3 questions > 400s.
