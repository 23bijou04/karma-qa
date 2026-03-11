# Missing Hop-1 Predicate: Investigation Report

**Date**: 2026-03-04
**Status**: Issues identified, fixes pending
**Case study**: "Where was the director of film Katka's Reinette Apples born?"
**Golden answer**: Rēzekne | **Our answer**: Reval (WRONG)

---

## 1. Problem Statement

For 2-hop bridge questions like "Where was the director of film X born?", DBpedia may lack
the structured hop-1 predicate (e.g., `dbo:director`) even when the film entity exists and
the bridge entity (director) has full structured data. This causes all SPARQL-based approaches
to fail. The pipeline falls back to LogicRAG decomposition, which correctly decomposes the
question but fails during bridge entity resolution due to three cascading bugs.

---

## 2. DBpedia Data Landscape

### 2.1 Director Property Availability (177,390 films)

| Scenario | Count | % | Available Signal |
|----------|------:|----:|------------------|
| **A** `dbo:director` URI exists | 131,316 | 74.0% | Direct SPARQL 2-hop works |
| **B** Only `dbp:director` (literal) | 32,306 | 18.2% | Director name as string |
| **C** Only `wikiPageWikiLink` → Person | 9,583 | 5.4% | Untyped link (could be actor) |
| **D** Only `dct:subject` / `dbo:description` | 578 | 0.3% | Category or description text |
| **E** No KG signal at all | ~3,607 | ~2.0% | Wikipedia article text only |
| **F** Bridge entity not in DBpedia | unknown | — | Both hops from Wikipedia text |

The Katka film falls into **Scenario C+D** (has wikiPageWikiLink AND dct:subject category).

### 2.2 Katka's Reinette Apples — What DBpedia Has

```
Film entity: <http://dbpedia.org/resource/Katka's_Reinette_Apples>
├── dbo:director              → MISSING (the hop-1 gap)
├── dbp:director              → MISSING
├── wikiPageWikiLink          → dbr:Fridrikh_Ermler           ✓ (but untyped)
├── dct:subject               → Category:Films_directed_by_Fridrikh_Ermler  ✓
├── dbo:description           → "1926 film by Fridrikh Ermler"              ✓
└── wikiPageWikiLink          → dbr:Eduard_Ioganson (co-director)           ✓

Director entity: <http://dbpedia.org/resource/Fridrikh_Ermler>
├── dbo:birthPlace            → dbr:Rēzekne                  ✓ (hop-2 EXISTS)
├── dbo:birthDate             → 1898-05-13                    ✓
├── dbo:occupation            → dbr:Film_director             ✓
└── rdf:type                  → dbo:Person                    ✓
```

The 2-hop path is **broken at hop-1** but **hop-2 is fully available**. If we can identify
Fridrikh Ermler as the director, SPARQL trivially returns Rēzekne.

---

## 3. Pipeline Execution Trace

### 3.1 Overall Flow (current 3-path architecture)

```
hybrid_answer()
  ├── Step 1: Entity Linking → found "Katka's Reinette Apples" [sparql]
  ├── Step 2: Schema Discovery → Film class (dbo:director in schema, not on entity)
  ├── Step 3: Route → SPARQL (rule:bridge_hop2)
  ├── Step 4: SPARQL → 0 results (dbo:director missing)
  │   └── Retry with dbp:director → 0 results
  ├── Step 5: LogicRAG Decomposition → 2 sub-Qs [DETAILED BELOW]
  └── (Step 6: Text fallback — NOT reached, LogicRAG returned an answer)
```

### 3.2 Step 4: SPARQL Direct — Why It Fails

Generated query:
```sparql
SELECT ?birthPlace WHERE {
  <http://dbpedia.org/resource/Katka's_Reinette_Apples> dbo:director ?person .
  ?person dbo:birthPlace ?birthPlace .
}
```
Result: **0 rows** — the film has no `dbo:director` triple.

Retry with `dbp:director`: same — no `dbp:director` either.

**Note**: The schema discovery returns `dbo:director` as a class-level property for `dbo:Film`,
because it comes from `CACHED_CLASS_SCHEMAS`. The schema is correct at the class level but
this specific instance lacks the property. This is a fundamental limitation of class-level
schema caching.

### 3.3 Step 5: LogicRAG Decomposition — Where Things Go Wrong

#### Phase 1: Decomposition (correct)

```
Sub-Q 0: "Who directed Katka's Reinette Apples?"           (no deps)
Sub-Q 1: "Where was [director] born?"                      (depends on Q0)
```

This decomposition is correct — it breaks the 2-hop question into two single-hop questions.

#### Phase 2a: Resolving Sub-Q 0 (partially correct)

```
_resolve_subquestion(sub_q=Q0, ...)
  ├── Entities: [Katka's Reinette Apples]
  ├── Route: rule:direct_property (keyword='directed' → dbo:director)
  ├── SPARQL: SELECT ?answer WHERE { <...Katka...> dbo:director ?answer } → 0 results
  └── TEXT fallback: Wikipedia for film → "Fridrikh Ermler and Eduard Ioganson"
```

**Finding**: Sub-Q 0 correctly falls to TEXT and gets the right answer from Wikipedia.
The answer includes both co-directors as a compound string.

#### Phase 2b: Resolving Sub-Q 1 — THE CRITICAL FAILURE (3 cascading bugs)

The resolved text after substitution: `"Where was Fridrikh Ermler and Eduard Ioganson born?"`

**Bug 1: Compound Answer → Single Invalid URI**
(`logicrag_decomposition.py`, lines 331-343)

```python
# Current code: treats entire answer as one entity
answer_val = "Fridrikh Ermler and Eduard Ioganson"
local_name = url_quote(answer_val.replace(' ', '_'), safe="_")
# Creates: "http://dbpedia.org/resource/Fridrikh_Ermler_and_Eduard_Ioganson"
# This URI does NOT exist in DBpedia
```

Expected behavior: Split on " and " to create TWO entities:
- `http://dbpedia.org/resource/Fridrikh_Ermler` (EXISTS ✓)
- `http://dbpedia.org/resource/Eduard_Ioganson` (EXISTS ✓)

**Bug 2: Unused Entity Linker — No Validation**
(`logicrag_decomposition.py`, line 297 vs lines 331-343)

The `linker` parameter (a `DBpediaEntityLinker` instance) is passed into `_resolve_subquestion()`
but **never used**. Instead, the code constructs URIs blindly via `url_quote()` without verifying
they exist in DBpedia.

If the linker were used (via `link_entities()` or `link_entity()`), it would:
1. Run Spotlight or SPARQL label search on "Fridrikh Ermler"
2. Resolve to `http://dbpedia.org/resource/Fridrikh_Ermler` (verified URI)
3. Discover it's a `dbo:Person` entity

Even without splitting, linking "Fridrikh Ermler" alone would succeed.

**Bug 3: Schema Not Refreshed for Bridge Entity**
(`logicrag_decomposition.py`, lines 345-355)

Because the URI is invalid (`dbr:Fridrikh_Ermler_and_Eduard_Ioganson`), schema discovery
returns nothing. The `combined_schema` still only contains the **Film class schema**, which
lacks `dbo:birthPlace`.

Consequence in the SPARQL gap analysis:
```
Schema properties: [dbo:Film properties only...]
Keyword 'born': need [dbo:birthDate, dbo:birthPlace], have [], MISSING all
Route: None (inconclusive → LLM decides)
```

With a valid URI (`dbr:Fridrikh_Ermler`), schema discovery would find:
```
Class: dbo:Person
  dbo: birthDate, birthPlace, deathDate, deathPlace, occupation, ...
```
And the gap analysis would match `born → dbo:birthPlace` → route to SPARQL → correct answer.

**Bug 3 Downstream: SPARQL on Invalid URI**

```sparql
SELECT ?answer WHERE {
  dbr:Fridrikh_Ermler_and_Eduard_Ioganson dbo:birthPlace ?answer .
}
-- 0 results (URI doesn't exist)
```

**Bug 3 Downstream: TEXT Fallback Hallucination**

After SPARQL fails, TEXT fallback runs with entities:
`["Katka's Reinette Apples", "Fridrikh Ermler and Eduard Ioganson"]`

- Wikipedia retrieval uses the compound name → may not find the right pages
- Wikipedia intro for Fridrikh Ermler does NOT mention birthplace at all:
  > "Fridrikh Markovich Ermler (13 May 1898 – 12 July 1967) was a Soviet film director..."
  > (no city mentioned in intro)
- The LLM **hallucinates "Reval"** (old name for Tallinn, Estonia)
- Correct answer is **Rēzekne, Latvia** (available via `dbo:birthPlace`)

---

## 4. The Correct Path (what SHOULD happen)

```
Sub-Q 0: "Who directed Katka's Reinette Apples?"
  → TEXT fallback → "Fridrikh Ermler and Eduard Ioganson"    ← same as now

Sub-Q 1: "Where was [director] born?"
  ├── Split "Fridrikh Ermler and Eduard Ioganson" on " and "
  ├── Entity-link each:
  │   ├── "Fridrikh Ermler" → Spotlight/SPARQL → dbr:Fridrikh_Ermler ✓
  │   └── "Eduard Ioganson" → Spotlight/SPARQL → dbr:Eduard_Ioganson ✓
  ├── Schema discovery for Fridrikh_Ermler → Person class
  │   → dbo:birthPlace available!
  ├── Route: rule match (born → birthPlace) → SPARQL
  ├── SPARQL: SELECT ?answer WHERE {
  │     dbr:Fridrikh_Ermler dbo:birthPlace ?answer .
  │   }
  │   → Rēzekne ✓✓✓
  └── Return SPARQL-grounded answer
```

---

## 5. Bug Summary Table

| # | Bug | Location | Impact | Fix |
|---|-----|----------|--------|-----|
| 1 | Compound answer not split | `logicrag_decomposition.py:331-343` | Creates invalid URI | Split on " and ", ", ", " or " |
| 2 | Entity linker unused | `logicrag_decomposition.py:297` (param) vs `331-343` (usage) | No URI validation | Use `linker.link_entities()` on answer text |
| 3 | Schema not refreshed | `logicrag_decomposition.py:345-355` | Wrong schema for sub-Q | Consequence of Bug 1+2; fixes itself when URIs are valid |

**Root cause**: Bugs 1 and 2 are independent but compound. Bug 3 is a downstream consequence.
Fixing either Bug 1 OR Bug 2 alone would likely resolve this case, but fixing both provides
robustness:
- Bug 1 fix handles the case where the LLM returns multiple entities in one answer
- Bug 2 fix handles the case where URI construction guesses wrong (typos, disambiguation pages)

---

## 6. Broader Scenario Analysis

### 6.1 Scenario B: `dbp:director` Exists as Literal (18.2% of films)

Example: "A Night for the Dogs" has `dbp:director → "Max Woodward"` (string) but no `dbo:director` (URI).

**Current behavior**: Same failure as Katka — SPARQL tries `dbo:director`, gets 0 results,
falls to LogicRAG where the same 3 bugs apply.

**Additional fix opportunity**: Before LogicRAG, try `dbp:director` (literal) and entity-link the string.
This would recover 32,306 films (18.2%) at low cost.

### 6.2 Scenario C: wikiPageWikiLink → Person (5.4% of films)

The Katka case. The director is linked but the link type is `wikiPageWikiLink` (not `dbo:director`).
The challenge: the film also links to actors, cinematographers, etc. — need to disambiguate role.

**Possible signals for disambiguation**:
- `dct:subject → Category:Films_directed_by_X` (direct signal)
- `dbo:description → "film by X"` (parse with regex)
- Cross-reference: person has `dbo:occupation → Film_director`
- `wikiPageWikiLink` label matching against question keywords

### 6.3 Scenario E: No KG Signal (2.0% of films)

No structured director data at all. Only Wikipedia article text can provide the answer.

**Current behavior**: LogicRAG TEXT fallback already handles this — Sub-Q 0 retrieves Wikipedia
for the film and extracts the director name. The failure is in Sub-Q 1 (bridge entity linking),
not Sub-Q 0. So fixing Bugs 1-2 would help even in this scenario, IF the bridge entity exists
in DBpedia (enabling SPARQL for hop-2).

### 6.4 Scenario F: Bridge Entity Not in DBpedia

If the director doesn't have a DBpedia page, hop-2 must also come from Wikipedia text.
Both hops are pure TEXT. The current LogicRAG architecture handles this correctly in principle
(TEXT → TEXT chain), but the compound answer bug still applies.

---

## 7. Fixes to Implement

### Fix 1: Split Compound Answers (Bug 1)
**File**: `logicrag_decomposition.py`, lines 331-343
**Change**: Before constructing URIs, split compound answers on common delimiters
(" and ", ", ", " or "). Create separate Entity objects for each part.

### Fix 2: Use Entity Linker for Validation (Bug 2)
**File**: `logicrag_decomposition.py`, lines 331-343
**Change**: Replace blind `url_quote()` URI construction with `linker.link_entities(answer_text)`
or `linker.link_entity(name)`. This verifies URIs exist and discovers entity types.

### Fix 3 (Optional): `dbp:` Literal Fallback for Hop-1
**File**: `logicrag_decomposition.py` or `hybrid_qa.py`
**Change**: When `dbo:director` returns 0 results, try `dbp:director` (literal),
entity-link the resulting string, then proceed with SPARQL for hop-2.
This recovers Scenario B (18.2% of films) cheaply.

---

## 8. Expected Impact

| Scenario | Current | After Fix 1+2 | After Fix 1+2+3 |
|----------|---------|---------------|-----------------|
| A (74.0%) | ✓ works | ✓ works | ✓ works |
| B (18.2%) | ✗ TEXT hallucination | ✓ LogicRAG + SPARQL hop-2 | ✓ Direct SPARQL hop-2 |
| C (5.4%)  | ✗ TEXT hallucination | ✓ LogicRAG + SPARQL hop-2 | ✓ LogicRAG + SPARQL hop-2 |
| D (0.3%)  | ✗ TEXT hallucination | ✓ LogicRAG + SPARQL hop-2 | ✓ LogicRAG + SPARQL hop-2 |
| E (2.0%)  | ✗ TEXT hallucination | ✓ if bridge in DBpedia | ✓ if bridge in DBpedia |
| F         | ✗ TEXT hallucination | ✗ TEXT (no fix possible) | ✗ TEXT (no fix possible) |

Fixes 1+2 convert LogicRAG from TEXT→TEXT (hallucination-prone) to TEXT→SPARQL (grounded)
for any case where the bridge entity exists in DBpedia with structured hop-2 data.
