# KG Coverage Scenarios (A–F)

Classification of 1000 evaluation questions by **where the DBpedia knowledge graph chain
breaks**. Each question follows a multi-hop pattern: seed entity → hop-1 property → bridge
entity → hop-2 property → answer. The scenario indicates which links in this chain are present
or missing.

**Companion document:** `wikipedia-link-extraction-cases.md` catalogs *how the pipeline fails*
(failure modes). This document catalogs *what the KG coverage landscape looks like* (data
scenarios). A question's scenario (A–F) determines which failure modes are possible, but
the same scenario can have different pipeline outcomes depending on entity linking, routing,
and context assembly.

**Data source:** `data/eval_1000q/scenario_classification_1000.json` (1000 questions from
2WikiMultiHopQA, classified by `scripts/classify_1000q_scenarios.py` from hop-verified
ground truth `data/eval_1000q/ground-truth/hop_verification_results.json`).

> **Recomputed 2026-03-31** from hop-verified ground truth. Previous counts (A=346, B=304, C=158, D=128, E=39, F=25) were from the original property-existence analysis (narrow PROPERTY_MAP). These hop-verified counts are authoritative.

---

## Overview

| Scenario | Label | Count | % | 95% CI | Description |
|----------|-------|------:|---:|--------|-------------|
| **A** | KG_COMPLETE | 368 | 36.8 | [33.9%, 39.8%] | Both hops have structured KG properties |
| **B** | HOP1_MISSING | 293 | 29.3 | [26.6%, 32.2%] | Seed exists, hop-1 property missing |
| **C** | HOP2_MISSING | 161 | 16.1 | [14.0%, 18.5%] | Hop-1 works, bridge entity missing hop-2 property |
| **D** | BOTH_HOPS_MISSING | 114 | 11.4 | [9.6%, 13.5%] | Both seed and bridge missing their needed properties |
| **E** | ENTITY_NOT_FOUND | 42 | 4.2 | [3.1%, 5.6%] | Entity not in DBpedia at all |
| **F** | NOT_FOUND + PROP_GAP | 22 | 2.2 | [1.5%, 3.3%] | Entity not found AND bridge missing property |

### Cross-tab: Scenario × Question Type

|           | bridge_comparison | comparison | compositional | inference | **TOTAL** |
|-----------|------------------:|-----------:|--------------:|----------:|----------:|
| **A**     | 141               | 80         | 113           | 34        | **368**   |
| **B**     | 32                | 164        | 79            | 18        | **293**   |
| **C**     | 54                | 0          | 95            | 12        | **161**   |
| **D**     | 8                 | 0          | 90            | 16        | **114**   |
| **E**     | 0                 | 0          | 25            | 17        | **42**    |
| **F**     | 0                 | 0          | 11            | 11        | **22**    |
| **TOTAL** | 235               | 244        | 413           | 108       | **1000**  |

**Notable patterns:**
- **Comparison** questions are either A (80) or B (164), never C–F. These are 1-hop questions
  ("Are X and Y in the same country?") where the property is either present or missing.
- **Bridge_comparison** concentrates in A (141) and C (54) — requires hop-1 to work for both
  entities, so B is rare (32) and D/E/F are near-zero.
- **Compositional** is the most evenly spread (113/79/95/90/25/11) because 2-hop questions
  can break at any point.
- **Inference** appears in all scenarios, including E (17) and F (11) — family relationship
  questions for obscure historical figures.

---

## Scenario A: KG-Complete (368 questions, 36.8%)

**Definition:** Both the seed entity's hop-1 property and the bridge entity's hop-2 property
exist as structured triples in DBpedia. The full SPARQL chain can resolve end-to-end.

### Representative Example

**Question:** "What is the place of birth of the director of film Peter's Friends?"
**Gold answer:** Belfast

| Component | DBpedia State |
|-----------|---------------|
| Seed: `dbr:Peter's_Friends` | `dbo:director → dbr:Kenneth_Branagh` |
| Bridge: `dbr:Kenneth_Branagh` | `dbo:birthPlace → dbr:Belfast` (also → dbr:Northern_Ireland) |
| Chain: Film → director → Person → birthPlace | Complete |

**Pipeline:** SPARQL resolves end-to-end. 0 LLM calls needed for the KG chain itself.

### KG State

Both hops have structured `dbo:` or `dbp:` properties. The seed entity has the relationship
property (director, author, father, etc.) pointing to the bridge entity, and the bridge entity
has the target property (birthPlace, deathDate, nationality, etc.) with the answer value.

### Information Sources Available

| Source | Available? | Notes |
|--------|-----------|-------|
| SPARQL 2-hop chain | Yes | Primary path — direct answer |
| Seed Wikipedia infobox | Yes (redundant) | Confirms hop-1 |
| Bridge Wikipedia infobox | Yes (redundant) | Confirms hop-2 |
| Wikipedia prose | Yes (redundant) | Backup only |

### Pipeline Can Solve?

**Yes** — SPARQL direct path. However, Cases 6 and 7 from `wikipedia-link-extraction-cases.md`
show that pipeline bugs (entity disambiguation, keyword filter) can cause failures even on
A-scenario questions. The KG data is complete; the failure is in the pipeline execution.

### Breakdown by Question Type

- **bridge_comparison (141):** Both entities have both hops → SPARQL can fetch and compare
- **comparison (80):** Single-hop property present on both entities
- **compositional (113):** Standard 2-hop chain
- **inference (34):** Family relationships (father → mother = "maternal grandfather")

---

## Scenario B: Hop-1 Missing (293 questions, 29.3%)

**Definition:** The seed entity exists in DBpedia but is **missing the hop-1 property** needed
by the question. The bridge entity (if it exists) has the hop-2 property. The SPARQL chain
breaks at the first hop.

### Representative Example

**Question:** "Where was the director of film Katka's Reinette Apples born?"
**Gold answer:** Rēzekne

| Component | DBpedia State |
|-----------|---------------|
| Seed: `dbr:Katka's_Reinette_Apples` | Has runtime, language, studio — **no `dbo:director`** |
| Bridge: `dbr:Fridrikh_Ermler` | `dbo:birthPlace → dbr:Rēzekne` |
| Chain break: Film → ~~director~~ | Hop-1 missing |

**Pipeline:** LogicRAG TEXT→SPARQL chain. Sub-Q 0 finds director via Wikipedia text
("Fridrikh Ermler"), bridge entity linking validates via SPARQL, then hop-2 resolves
via SPARQL (`dbo:birthPlace → Rēzekne`). ~7–9 LLM calls (route + failed SPARQL attempt
+ decomposition + 2 sub-Q resolutions + synthesis).

### Comparison Variant

**Question:** "Are both Țerova and Slava (River) located in the same country?"
**Gold answer:** yes

This is a 1-hop comparison where **both** seed entities are missing the `country` property.
No bridge entity — the answer requires extracting `country` from Wikipedia for both entities.

### KG State

The seed entity exists with `rdf:type` and some structured properties, but the specific
property asked about (director, father, country, publication date, etc.) is absent. DBpedia
either failed to extract it from the Wikipedia infobox or the infobox didn't have it.

### Information Sources Available

| Source | Available? | Notes |
|--------|-----------|-------|
| SPARQL hop-1 | No | Missing property |
| SPARQL hop-2 | Yes (if bridge exists) | Bridge entity has structured data |
| Seed Wikipedia infobox | Usually yes | Infobox often has the missing property |
| Seed Wikipedia prose | Yes | Fallback if no infobox |
| Bridge Wikipedia infobox | Yes (redundant) | Hop-2 already in KG |

### Pipeline Can Solve?

**Yes** — LogicRAG TEXT→SPARQL chain. The bridge entity fix (2026-03-04) converts what was
a TEXT→TEXT chain (hallucination-prone) to TEXT→SPARQL (grounded) when the bridge entity has
structured hop-2 data. However, this costs ~7–9 LLM calls. Infobox extraction from the seed
entity's Wikipedia page would yield the answer with 0 LLM calls for the hop-1 step.

### Top Missing Hop-1 Properties

| Property | Count | Notes |
|----------|------:|-------|
| publication date | 153 | Books, albums, films |
| director | 87 | Films |
| father | 77 | People |
| country of citizenship | 38 | People |
| country | 37 | Places, rivers |
| date of birth | 31 | People |
| spouse | 27 | People |
| composer | 23 | Musical works |
| date of death | 16 | People |
| country of origin | 15 | Various entities |

### Breakdown by Question Type

- **comparison (164):** Dominates B-scenario — simple property lookups where the property is
  missing. "Are X and Y in the same country?" when neither has `dbo:country`.
- **compositional (79):** 2-hop chains where hop-1 breaks.
- **bridge_comparison (32):** Rare — requires hop-1 to work for comparison, so B means only
  one of the two entities has a missing hop-1.
- **inference (18):** Family relationships with missing parent/spouse property.

---

## Scenario C: Hop-2 Missing (161 questions, 16.1%)

**Definition:** The seed entity has the hop-1 property, so SPARQL resolves the first hop
successfully. But the **bridge entity is missing the hop-2 property** — the SPARQL chain
breaks at the second hop.

### Representative Example

**Question:** "Who is the spouse of the director of film My Three Merry Widows?"
**Gold answer:** Mapy Cortés

| Component | DBpedia State |
|-----------|---------------|
| Seed: `dbr:My_Three_Merry_Widows` | `dbo:director → dbr:Fernando_Cortés` |
| Bridge: `dbr:Fernando_Cortés` | Has descriptions, wikiPageWikiLinks — **no `dbo:spouse`** |
| Chain break: Film → director → Person → ~~spouse~~ | Hop-2 missing |

The bridge entity is a "near-empty shell" — the Wikipedia page uses `{{Infobox person}}`
(visible in wikitext) but DBpedia failed to extract any structured fields from it. Even the
`wikiPageUsesTemplate → Infobox_person` triple is absent — the entire infobox extraction
failed for this entity. The Wikipedia infobox has `| spouse = [[Mapy Cortés]]`.

**Pipeline:** LogicRAG SPARQL→TEXT chain. Sub-Q 0 ("Who directed?") resolves via SPARQL
(Fernando Cortés), then sub-Q 1 ("Who is spouse of Fernando Cortés?") resolves via TEXT
from Wikipedia. Correct answer, ~7–9 LLM calls.

### KG State

Hop-1 works: the seed entity has a structured property pointing to the bridge entity.
The bridge entity exists in DBpedia but lacks the specific hop-2 property. Common pattern:
DBpedia has the entity with `rdf:type` and `description` but failed to extract structured
properties from the Wikipedia infobox.

### Information Sources Available

| Source | Available? | Notes |
|--------|-----------|-------|
| SPARQL hop-1 | Yes | Works — identifies bridge entity |
| SPARQL hop-2 | No | Bridge entity missing property |
| Bridge Wikipedia infobox | Usually yes | Infobox often has the missing property |
| Bridge Wikipedia prose | Yes | Fallback if no infobox field |

### Pipeline Can Solve?

**Yes** — LogicRAG SPARQL→TEXT chain. SPARQL resolves hop-1, then TEXT retrieves the answer
from the bridge entity's Wikipedia page for hop-2. More reliable than B-scenario because
hop-1 is grounded in KG (no hallucination risk for bridge entity identity). Infobox
extraction from the bridge entity's Wikipedia page would be more efficient (0 LLM calls
for hop-2 vs. 2–3 for TEXT).

### Top Missing Hop-2 Properties

| Property | Count | Notes |
|----------|------:|-------|
| country of citizenship | 132 | By far the most common hop-2 gap |
| place of birth | 34 | People |
| date of birth | 26 | People |
| father | 21 | People |
| place of death | 20 | People |
| date of death | 19 | People |
| mother | 11 | People |
| educated at | 11 | People |
| employer | 9 | People |
| spouse | 8 | People |

### Breakdown by Question Type

- **compositional (95):** Standard 2-hop chains breaking at hop-2.
- **bridge_comparison (54):** Comparison where one or both bridge entities lack hop-2.
- **inference (12):** Family chains where the intermediate person lacks the needed property.
- **comparison (0):** Never — comparison questions are 1-hop, so there's no hop-2 to miss.

---

## Scenario D: Both Hops Missing (114 questions, 11.4%)

**Definition:** Both the seed entity (hop-1) and the bridge entity (hop-2) are missing their
respective properties. The SPARQL chain breaks at both hops. Full text retrieval required.

### Representative Example

**Question:** "Why did John Middleton Murry's wife die?"
**Gold answer:** tuberculosis

| Component | DBpedia State |
|-----------|---------------|
| Seed: `dbr:John_Middleton_Murry` | Exists — **no `dbo:spouse`** |
| Bridge: `dbr:Katherine_Mansfield` | Exists — **no `dbo:causeOfDeath`** |
| Chain break: Both hops | Person → ~~spouse~~ → Person → ~~causeOfDeath~~ |

Missing properties: `John Middleton Murry.spouse` + `Katherine Mansfield.cause of death`.
Both entities exist in DBpedia with type information and descriptions, but lack the
specific structured properties needed.

**Pipeline:** LogicRAG TEXT→TEXT chain. Both sub-questions require Wikipedia text retrieval.
This is the most error-prone chain — no KG grounding at any step.

### KG State

Both entities exist in DBpedia (have `rdf:type`, `rdfs:label`, descriptions) but are missing
the specific properties needed. Note: this is not always a total extraction failure — e.g.,
John Middleton Murry has `dbo:birthPlace`, `dbo:birthDate`, `dbo:child` but lacks `dbo:spouse`;
Katherine Mansfield has `dbo:almaMater`, `dbo:occupation` but lacks `dbo:causeOfDeath`.
DBpedia extracted *some* structured properties from each infobox, just not the ones needed.

### Information Sources Available

| Source | Available? | Notes |
|--------|-----------|-------|
| SPARQL hop-1 | No | Seed missing property |
| SPARQL hop-2 | No | Bridge missing property |
| Seed Wikipedia infobox | Usually yes | Hop-1 answer likely there |
| Bridge Wikipedia infobox | Usually yes | Hop-2 answer likely there |
| Wikipedia prose | Yes | Fallback for both hops |

### Pipeline Can Solve?

**Partially** — LogicRAG TEXT→TEXT chain. Both hops require Wikipedia text retrieval and LLM
reasoning. Most error-prone scenario because:
1. Bridge entity identity depends entirely on LLM text extraction (no SPARQL validation)
2. Hop-2 answer also depends on LLM text extraction
3. Compound errors can cascade (wrong bridge → wrong hop-2 answer)

Reliability is lower than A/B/C because there is no KG grounding at any step. Infobox
extraction for both entities would convert this to a 0-LLM-call chain.

### Breakdown by Question Type

- **compositional (90):** Dominates — 2-hop chains that break at both hops.
- **inference (16):** Family relationships where both entities lack the needed property.
- **bridge_comparison (8):** Rare — both entities in the comparison having double failures.
- **comparison (0):** Never — comparison is 1-hop.

---

## Scenario E: Entity Not Found (42 questions, 4.2%)

**Definition:** The seed entity does **not exist** in DBpedia at all — no redirect, no stub,
or only a redirect stub to an unrelated page. Typically the bridge entity exists and has the
hop-2 property. No KG entry point for the question.

**Edge cases:** In some questions the *bridge* entity is also not found (e.g., "Joan
Ramon II, Count of Cardona" — both seed and bridge absent) or the seed exists but the bridge
doesn't (e.g., "3096 Days" — seed film exists, bridge director `Sherry Horman` not in
DBpedia). These are classified E because the primary blocker is a missing entity, not a
missing property.

### Representative Example

**Question:** "Which country Thomas Lyon-Bowes, Lord Glamis's father is from?"
**Gold answer:** British

| Component | DBpedia State |
|-----------|---------------|
| Seed: `Thomas George Lyon-Bowes, Lord Glamis` | **Not found** in DBpedia |
| Bridge: `Thomas Lyon-Bowes, 11th Earl of Strathmore and Kinghorne` | `dbp:nationality = "Scottish"` (literal, not structured URI) |
| Chain break: Entry point | No entity → no chain |

The DBpedia entry for "Thomas Lyon-Bowes, Lord Glamis" is a redirect stub to
`dbr:Earl_of_Strathmore_and_Kinghorne` (the earldom/peerage title article, not any
individual person's biography). No structured person properties. The 11th Earl's own
page has rich data (`dbo:birthPlace`, `dbo:birthDate`, `dbo:father`, `dbo:mother`) but
nationality is only a literal string (`dbp:nationality = "Scottish"`), not a structured
`dbo:citizenship` link.

**Pipeline:** No KG entry point → TEXT fallback only. Entity linking fails (Spotlight
doesn't recognize the name, SPARQL label search fails). LogicRAG decomposes but sub-questions
also rely entirely on text.

### KG State

The seed entity is absent from DBpedia. Possible sub-patterns:
- **Redirect stub:** DBpedia entry exists but only has `dbo:wikiPageRedirects` to an unrelated
  page (peerage title, disambiguation page)
- **Truly absent:** No DBpedia resource at all — entity is too obscure for Wikipedia/DBpedia
- **Label mismatch:** Entity exists under a different name/spelling that entity linking can't resolve

### Information Sources Available

| Source | Available? | Notes |
|--------|-----------|-------|
| SPARQL (any hop) | No | No entity to start from |
| Seed Wikipedia page | Maybe | May exist under different title |
| Bridge Wikipedia infobox | Yes | Bridge has structured data |
| Wikipedia search | Yes | Can find the person via search API |

### Pipeline Can Solve?

**Partially** — TEXT fallback only. Without entity linking, the pipeline has no KG context.
LogicRAG sub-questions rely on Wikipedia search and LLM text extraction. Success depends on
whether Wikipedia search finds the right article and the LLM extracts the correct answer.

### Breakdown by Question Type

- **compositional (25):** 2-hop chains that can't start.
- **inference (17):** Family relationships for obscure historical figures.
- **bridge_comparison (0):** Never — bridge_comparison requires entity linking for both entities.
- **comparison (0):** Never — comparison requires entity linking for both entities.

---

## Scenario F: Entity Not Found + Property Gap (22 questions, 2.2%)

**Definition:** The seed entity is **not found or is an empty typed stub** in DBpedia (like E),
AND the bridge entity is **missing the hop-2 property** (like C/D). Double failure — the
hardest scenario.

### Representative Example

**Question:** "Where was the husband of Jeanne Raunay born?"
**Gold answer:** Évreux

| Component | DBpedia State |
|-----------|---------------|
| Seed: `Jeanne Richomme Raunay` | Exists as typed stub (`dbo:Person`, `owl:sameAs`) — **no structured properties** |
| Bridge: `André Beaunier` | Exists — **no `dbo:birthPlace`** |
| Chain break: Both entry point AND hop-2 | No seed entity + bridge missing property |

Neither KG entry point works: the seed entity exists as a typed stub (`dbo:Person` with
`owl:sameAs` to Wikidata) but has zero `dbo:`/`dbp:` properties — no spouse, no birthPlace,
nothing to start a chain from. The bridge entity André Beaunier is also a near-empty stub
(only `owl:sameAs` triples) and lacks the needed `place of birth` property.

**Pipeline:** Double failure — TEXT fallback for everything. Must find Jeanne Raunay via
Wikipedia search, extract "husband = André Beaunier" from text, then find Beaunier's birthplace
from text. All steps are LLM-dependent with no KG grounding.

### KG State

Combines the worst of E (no usable seed entity — absent or typed stub with no properties)
and C/D (bridge entity missing properties). Both Wikipedia text retrieval steps are needed,
and neither has KG validation.

### Information Sources Available

| Source | Available? | Notes |
|--------|-----------|-------|
| SPARQL (any hop) | No | Seed not found, bridge missing property |
| Seed Wikipedia page | Maybe | May exist under different title |
| Bridge Wikipedia infobox | Maybe | Property might be in infobox but not in DBpedia |
| Wikipedia prose | Yes | Only reliable source for both hops |

### Pipeline Can Solve?

**Unlikely** — requires successful text retrieval for both hops with zero KG grounding.
This is the hardest scenario. The pipeline must:
1. Find the right Wikipedia article for an entity DBpedia doesn't know
2. Extract the bridge entity identity from text
3. Find the bridge entity's Wikipedia page
4. Extract the answer from text (since KG also lacks it)

Each step introduces error potential. At 2.5% of questions, this is mercifully rare.

### Breakdown by Question Type

- **compositional (11):** 2-hop chains with double failure.
- **inference (11):** Family relationship chains for obscure historical figures.

---

## Information Source Matrix

For each scenario, which information sources can provide the answer:

| Scenario | SPARQL Chain | Seed Infobox | Bridge Infobox | Wikipedia Prose | LLM Text Reasoning |
|----------|:---:|:---:|:---:|:---:|:---:|
| **A** (KG-Complete) | **Primary** | Redundant | Redundant | Redundant | Redundant |
| **B** (Hop-1 Missing) | Hop-2 only | **Hop-1 answer** | Redundant | Fallback | Fallback |
| **C** (Hop-2 Missing) | Hop-1 only | Redundant | **Hop-2 answer** | Fallback | Fallback |
| **D** (Both Missing) | Neither | **Hop-1 answer** | **Hop-2 answer** | Fallback | Fallback |
| **E** (Not Found) | Neither | Maybe | Redundant | **Primary** | **Primary** |
| **F** (Not Found + Gap) | Neither | Maybe | Maybe | **Primary** | **Primary** |

**Key insight:** Scenarios B, C, and D (568 questions, 56.8%) have answers that likely exist
in Wikipedia infoboxes but are missing from DBpedia's structured extraction. Infobox parsing
would recover structured answers for the majority of these without LLM calls.

---

## Pipeline Evaluation Summary

| Scenario | Can Solve? | Method | LLM Calls | Reliability | Notes |
|----------|-----------|--------|----------:|-------------|-------|
| **A** | Yes | SPARQL end-to-end | ~3 | High | Route (1) + SPARQL gen (1) + answer extract (1); pipeline bugs (Cases 6, 7) can still cause failures |
| **B** | Yes | LogicRAG TEXT→SPARQL | ~7–9 | Medium-High | Route+SPARQL attempt (~3) + decompose (1) + sub-Qs (2×2) + synthesis (1) |
| **C** | Yes | LogicRAG SPARQL→TEXT | ~7–9 | Medium-High | Same as B; hop-1 grounded, hop-2 via text |
| **D** | Partially | LogicRAG TEXT→TEXT | ~7–9 | Medium | Same call count, but no KG grounding — error cascades possible |
| **E** | Partially | TEXT fallback | ~4–7 | Low-Medium | No KG entry point; depends on Wikipedia search quality |
| **F** | Unlikely | TEXT fallback (both hops) | ~7–9 | Low | Double failure — hardest case, 2.5% of questions |

### Efficiency Opportunity

Scenarios B, C, and D together represent 568 questions (56.8%) where Wikipedia infobox extraction
could replace 7–9 LLM calls with 0 for the missing hops. The missing properties are typically
present in Wikipedia infoboxes (director, spouse, birthPlace, etc.) — DBpedia just failed to
extract them. Parsing `| key = [[Entity]]` patterns from wikitext is straightforward and would:
- Eliminate LLM hallucination risk for these hops
- Reduce per-question cost from ~7–9 LLM calls to ~3 (route + SPARQL for remaining hop)
- Provide pre-disambiguated entity links (wikilinks are already DBpedia URIs)

---

## Missing Property Rankings

### Hop-1 (seed entity missing property) — affects scenarios B, D

| Rank | Property | Count | Entity Types |
|------|----------|------:|--------------|
| 1 | publication date | 153 | Books, albums, films |
| 2 | director | 87 | Films |
| 3 | father | 77 | People |
| 4 | country of citizenship | 38 | People |
| 5 | country | 37 | Places, rivers |
| 6 | date of birth | 31 | People |
| 7 | spouse | 27 | People |
| 8 | composer | 23 | Musical works |
| 9 | date of death | 16 | People |
| 10 | country of origin | 15 | Various |

*Total hop-1 missing property instances across 407 questions (B=293, D=114).*

### Hop-2 (bridge entity missing property) — affects scenarios C, D, F

| Rank | Property | Count | Entity Types |
|------|----------|------:|--------------|
| 1 | country of citizenship | 132 | People |
| 2 | place of birth | 34 | People |
| 3 | date of birth | 26 | People |
| 4 | father | 21 | People |
| 5 | place of death | 20 | People |
| 6 | date of death | 19 | People |
| 7 | mother | 11 | People |
| 7 | educated at | 11 | People |
| 9 | employer | 9 | People |
| 10 | spouse | 8 | People |

*Total hop-2 missing property instances across 297 questions (C=161, D=114, F=22).*

**Observation:** Hop-2 missing properties are almost exclusively person properties.
`country of citizenship` alone accounts for 132 hop-2 property instances (~44%).
This is because DBpedia often has person entities with basic type information but lacks
biographical detail. The corresponding Wikipedia infoboxes almost always have
`| nationality =` or `| citizenship =` fields.
