# Wikipedia Link Extraction: Failure Case Taxonomy

When the hybrid pipeline can't complete the full SPARQL hop chain, it falls through
compositional decomposition → LogicRAG → TEXT route. This document catalogs the distinct
failure modes observed in the 2WikiMultiHopQA evaluation, with real examples and
DBpedia/Wikipedia evidence.

The core idea: Wikipedia pages contain hyperlinks (`[[Entity]]`) embedded in context
(infoboxes, prose). These links are pre-disambiguated and contextually annotated by
surrounding text. Extracting them — instead of using `explaintext=True` which strips
all links — could replace the unreliable LLM-generated context SPARQL.

**Note on route labels:** `_rule_based_route()` provides a **hint** to the LLM router
(e.g., `rule:bridge_hop2`, `rule:direct_property`), but the LLM always makes the final
SPARQL/TEXT decision. The route labels in this document reflect the actual pipeline output
(rule hint + LLM agreement). The full fallback chain is: SPARQL → compositional
decomposition → LogicRAG sub-question decomposition → TEXT.

---

## KG Scenario Classification (A–F)

Each question in the 1000-question eval set is classified by WHERE the KG chain breaks
(see `data/eval_1000q/scenario_classification_1000.json`). The cases in this document
describe HOW the pipeline fails — a complementary but distinct taxonomy.

| Scenario | Label | Count | % | Representative Case |
|----------|-------|------:|----:|---------------------|
| **A** | KG_COMPLETE | 346 | 34.6% | _(original analysis; hop-verified: 368 predicate-complete, 318 chain-traversable)_ |
| **B** | HOP1_MISSING | 304 | 30.4% | Case 1 (Katka), Case 2 (Bea Ballard) |
| **C** | HOP2_MISSING | 158 | 15.8% | Case 5 (My Three Merry Widows) |
| **D** | BOTH_HOPS_MISSING | 128 | 12.8% | Case 3 (Consort Liang) |
| **E** | ENTITY_NOT_FOUND | 39 | 3.9% | Case 4 (Thomas Lyon-Bowes) |
| **F** | ENTITY_NOT_FOUND + PROP_GAP | 25 | 2.5% | _(Jeanne Raunay — see below)_ |

**Mapping notes:**
- Cases 1, 2, 4, 5 map cleanly to scenarios B, B, E, C respectively.
- Case 3 (Consort Liang) is **D** — both seed AND bridge have missing properties, not
  just a semantic mismatch.
- Case 6 (Thomas Jefferson disambiguation) appears as **B** in the classification because
  the coverage analysis inherited the same disambiguation error from entity linking. With
  the correct entity `dbr:Thomas_Jefferson_(film)`, this question would be **A**.
- Case 7 (Tammy and the T-Rex keyword filter) is **A** — the KG has all needed data.
  The failure is purely a pipeline bug (keyword filter drops answer-bearing paragraph),
  not a KG coverage gap.

**Scenario F example** (no dedicated case — double failure is uncommon, 2.5%):
"Where was the husband of Jeanne Raunay born?" (gold: Évreux). Seed entity
`Jeanne Richomme Raunay` is not found in DBpedia. Bridge entity `André Beaunier`
exists but is missing `place of birth`. Both KG entry points fail: no entity to
start the chain, and even if found, the bridge entity can't complete hop-2.

---

## Case 1: Missing Hop-1 Predicate (director not in DBpedia)

**KG Scenario:** B (HOP1_MISSING) — 304 questions (30.4%)

**Pattern:** The seed entity exists in DBpedia with some structured properties (runtime,
language, studio) but is **missing the specific hop-1 predicate** needed by the question.
The pipeline can't start the SPARQL chain for the asked relationship.

### Example

**Question:** "Where was the director of film Katka's Reinette Apples born?"
**Gold answer:** Rēzekne
**Route:** `rule:bridge_hop2` → SPARQL (fails, no `dbo:director` triple) → compositional (fails) → LogicRAG
**Pipeline result (before fix):** Wrong answer ("Reval, Russian Empire" via LogicRAG — TEXT→TEXT
chain hallucinated because compound answer created invalid bridge URI)
**Pipeline result (after bridge entity fix):** Correct — "Rēzekne" via LogicRAG TEXT→SPARQL chain.
Sub-Q 0 finds directors via TEXT ("Fridrikh Ermler and Eduard Ioganson"), bridge entity linking
splits compound answer → validates "Fridrikh Ermler" via SPARQL → `dbr:Fridrikh_Ermler` →
schema discovers `dbo:birthPlace` → SPARQL returns Rēzekne.

### DBpedia state

`dbr:Katka's_Reinette_Apples` — has some structured properties (runtime, language,
productionCompany/studio) but **no `dbo:director` or `dbp:director`**. The hop-1
predicate needed by the question is absent.

```sparql
SELECT ?p ?o WHERE {
  <http://dbpedia.org/resource/Katka%27s_Reinette_Apples> ?p ?o .
  FILTER(STRSTARTS(STR(?p), "http://dbpedia.org/ontology/")
      || STRSTARTS(STR(?p), "http://dbpedia.org/property/"))
  FILTER(!CONTAINS(STR(?p), "wikiPage"))
}
-- Returns: dbo:runtime, dbo:language, dbo:productionCompany, dbp:language,
--          dbp:studio, dbp:runtime, dbp:country, dbp:name (NO director)
```

### Wikipedia page (wikitext)

```
{{Infobox film
| name = Katka's Reinette Apples
| director = {{ubl|[[Fridrikh Ermler]]|[[Eduard Ioganson]]}}
| writer = {{ubl|[[Mikhail Borisoglebsky]]|[[Boris Leonidov]]}}
| cinematography = {{ubl|[[Yevgeni Mikhajlov]]|[[Andrei Moskvin]]}}
| studio = [[Sovkino]]
| released = {{film date|df=yes|1926|12|25}}
}}
```

The director `[[Fridrikh Ermler]]` is right there as a wikilink in the infobox.

### What link extraction would do

1. Parse infobox → `director = [[Fridrikh Ermler]]`
2. Convert → `dbr:Fridrikh_Ermler`
3. Match against schema: question says "director", infobox key IS "director" → direct hit
4. Check Ermler's DBpedia: `dbo:birthPlace → Rēzekne` → answer found via SPARQL, no LLM needed

### Previous failure (before bridge entity fix)

The route engine fires `rule:bridge_hop2` (schema says Film class has `dbo:director`, which
maps to Person with `birthPlace`). But the actual DBpedia entry lacks `dbo:director`, so
SPARQL returns 0 rows. Compositional decomposition also fails (same missing triple). LogicRAG
decomposes correctly into sub-Qs, but two bugs caused cascading failure:

1. **Bug 1 (compound answer):** Sub-Q 0 returned "Fridrikh Ermler and Eduard Ioganson" from
   TEXT. The old code created one invalid URI `dbr:Fridrikh_Ermler_and_Eduard_Ioganson` instead
   of splitting the answer.
2. **Bug 2 (no entity validation):** The `kg_client` parameter was passed to
   `_resolve_subquestion()` but never used — bridge entities were constructed as blind URIs
   without SPARQL validation.
3. **Downstream:** Invalid URI → empty schema → no SPARQL routing → TEXT fallback for hop-2 →
   Wikipedia says "born in Reval, Russian Empire" while DBpedia has `dbo:birthPlace → Rēzekne`
   (modern name) → wrong answer.

### Current status (fixed)

The bridge entity linking fix in `logicrag_decomposition.py` resolves this:
1. `_resolve_answer_entities()` tries the full compound string first ("Simon and Garfunkel"
   safety), then splits on " and " and validates each fragment via
   `kg_client.search_entity_by_name()`
2. "Fridrikh Ermler" → SPARQL exact label match → `dbr:Fridrikh_Ermler` (source=sparql)
3. "Eduard Ioganson" → SPARQL fails → fallback URI `dbr:Eduard_Ioganson` (source=logicrag_guessed)
4. Schema discovery finds Person class with `dbo:birthPlace` on validated Ermler entity
5. Sub-Q 1 routes to SPARQL → `dbo:birthPlace → Rēzekne` → correct grounded answer

The fix converts the TEXT→TEXT chain (hallucination-prone) to TEXT→SPARQL (grounded) for any
bridge entity that exists in DBpedia with structured hop-2 data. Infobox link extraction would
still be more efficient (0 LLM calls vs. 5), but is no longer required for correctness.

### Sub-example: Sparse entity variant (Piccoli Fuochi)

**Question:** "Which film has the director who was born first, Piccoli Fuochi or The Nice Guys?"
**Gold answer:** Piccoli Fuochi
**Type:** bridge_comparison (2-hop per entity: film → director → birthDate)
**Pipeline result:** Wrong — "The Nice Guys" (verified 2026-03-04)

This is a harder variant of Case 1 that the bridge entity fix does NOT solve, because the
failure is upstream in entity linking. The film entity `dbr:Piccoli_fuochi` is an **almost-empty
shell** — has `rdf:type → dbo:Film` but no structured film properties (`dbo:director`, etc.)
and crucially **no English `rdfs:label`** (only French and Italian labels: "Piccoli fuochi").
This makes it invisible to all entity linking methods.

**KG signals available** (via `wikiPageWikiLink`):

| Target | Type | `dbo:occupation` | KG Signal |
|--------|------|-----------------|-----------|
| **Peter Del Monte** | `dbo:Person` | `Film_director`, `Screenwriter` | Director ✓ |
| Tonino Nardi | `dbo:Person` | `Cinematographer` | Not director |
| Valeria Golino | `dbo:Person` | `PersonFunction__1` only | Ambiguous |
| `Category:Films_directed_by_Peter_Del_Monte` | Category | N/A | **Strongest signal** |

Peter Del Monte has full structured data: `dbo:birthDate → 1943-07-29`,
`dbo:birthPlace → San_Francisco`, `dbo:deathDate → 2021-05-31`. The other film (The Nice Guys)
has `dbo:director → Shane_Black` (birthDate 1961-12-16), so SPARQL works for one side.

**Actual pipeline behavior (verified):**
1. Entity linking: Spotlight links "The Nice Guys" → `dbr:The_Nice_Guys`. **Misses "Piccoli
   Fuochi" entirely** — no English label, Spotlight doesn't recognize it, SPARQL label search
   fails (no English `rdfs:label` to match against)
2. Route: `rule:direct_date_comparison` fires (wrong — treats it as single-entity date
   comparison instead of bridge_comparison, because only one entity is linked)
3. SPARQL generates query using only The Nice Guys → returns 1 row → "The Nice Guys"
4. Pipeline "succeeds" with wrong answer — **never reaches LogicRAG decomposition**

**Why the bridge entity fix doesn't help here:** The failure is upstream entity linking
(Case 4 pattern), not bridge entity resolution. LogicRAG decomposition is never reached because:
(a) only one entity is linked, (b) the route fires as direct SPARQL comparison, and (c) SPARQL
"succeeds" with partial data (one film only). Even if LogicRAG were reached, the sub-question
"Who directed Piccoli Fuochi?" would need Wikipedia text retrieval for an unlinked entity.

**Three potential KG-only approaches to identify the director:**
1. **Category parsing:** `wikiPageWikiLink → Category:Films_directed_by_Peter_Del_Monte` —
   parse category name regex `Films_directed_by_(.+)` → `dbr:Peter_Del_Monte`. Strongest
   signal, zero ambiguity.
2. **Occupation-based filtering:** enumerate `wikiPageWikiLink → Person` targets, filter by
   `dbo:occupation = Film_director` → Peter Del Monte is the only match.
3. **Wikipedia infobox text:** fetch wikitext, parse `| director = Peter Del Monte`.

**Expected pipeline behavior (if entity linking were fixed):**
1. Sub-Q 0: "Who directed Piccoli Fuochi?" → SPARQL fails (no `dbo:director`) → TEXT → "Peter Del Monte"
2. Sub-Q 1: "Who directed The Nice Guys?" → SPARQL succeeds → "Shane Black"
3. Sub-Q 2: "When was Peter Del Monte born?" → bridge entity-link via SPARQL → 1943-07-29
4. Sub-Q 3: "When was Shane Black born?" → SPARQL → 1961-12-16
5. Synthesis: 1943 < 1961 → Piccoli Fuochi

**Implications:** This sub-case shows that the bridge entity fix has a prerequisite —
upstream entity linking must succeed for at least the seed entities. Films with no English
`rdfs:label` in DBpedia (like Piccoli Fuochi) are invisible to the current Spotlight +
SPARQL label cascade. Potential fixes: (a) fuzzy URI construction from question text
(`Piccoli_Fuochi` → `Piccoli_fuochi` via case-insensitive redirect), (b) Wikipedia search
API to find the article and extract the DBpedia URI, (c) cross-language label search.

---

## Case 2: No Infobox Seed (prose-only links)

**KG Scenario:** B (HOP1_MISSING) — same scenario as Case 1, different Wikipedia surface

**Pattern:** The seed entity has no structured DBpedia properties AND no Wikipedia
infobox. But the Wikipedia prose contains hyperlinks with contextual clues about the
entity's role/relationship.

### Example

**Question:** "What is the place of birth of Bea Ballard's father?"
**Gold answer:** Shanghai
**Route:** `llm` → SPARQL (fails) → compositional (fails) → LogicRAG (fails to find father)
**Entity linking issue:** Spotlight doesn't recognize "Bea Ballard" (too obscure). The full
cascade (Spotlight → spaCy mentions → SPARQL `bif:contains`) fails to find the entity.
SPARQL label fallback links "birth" → `dbp:birth` (a property URI, not a resource), giving
the pipeline a useless schema context.
**Pipeline result:** None (LogicRAG sub-Q "Who is Bea Ballard's father?" returns no answer from text)

### DBpedia state

`dbr:Bea_Ballard` — only multilingual `description` fields and `wikiPageWikiLink` edges
(including one to `dbr:J._G._Ballard`, but as an untyped wiki link, not a structured relation):
```
  description → "British television executive producer"
  description → "televisieproducent"
  wikiPageWikiLink → J._G._Ballard, 10_Star_Entertainment, ... (~60 links)
```
No `dbo:parent`, no `dbo:child`, no `dbo:relative` — no structured family properties.

### Wikipedia page (wikitext) — no infobox

```
'''Bea Ballard''' (also known professionally as Beatrice Ballard) is a British
[[television]] [[executive producer]]. She is chief executive of
[[10 Star Entertainment]]... She is the daughter of novelist [[J. G. Ballard]].
```

No `{{Infobox person}}` template. But the lead paragraph contains:
- Link: `[[J. G. Ballard]]`
- Surrounding text: "She is the daughter of novelist"
- Question keyword "father" matches semantic role "daughter of"

### What link extraction would do

1. Parse prose → find all `[[links]]` with surrounding 10-word window
2. `[[J. G. Ballard]]` surrounded by "the daughter of novelist" → role keyword match
3. Convert → `dbr:J._G._Ballard`
4. Question asks "place of birth" → check `dbo:birthPlace` on J.G. Ballard
5. DBpedia has `dbo:birthPlace → Shanghai_International_Settlement` (label: "Shanghai") → answer found

### Why it fails today

Entity linking fails entirely for "Bea Ballard" — Spotlight doesn't recognize the name
(too obscure), and SPARQL label fallback links "birth" → `dbp:birth` instead. Without a
linked entity, there is no schema, no SPARQL, and compositional decomposition targets the
wrong URI. LogicRAG decomposes into "Who is Bea Ballard's father?" but the TEXT sub-question
resolver can't find the connection either — Wikipedia search for "Bea Ballard" doesn't surface
the article, so the LLM never sees "daughter of novelist J.G. Ballard".

### Complexity note

This case requires prose link extraction, not just infobox parsing. The keyword
matching is slightly harder: "father" in the question must match "daughter of" in
the surrounding text. A simple bidirectional family-keyword set
(`{father, mother, daughter, son, child, parent}`) would handle this.

---

## Case 3: Semantic Role Mismatch (link exists, role words differ)

**KG Scenario:** D (BOTH_HOPS_MISSING) — Consort Liang is missing `spouse` (hop-1)
AND Emperor Zhang is missing `place of birth` (hop-2). The semantic mismatch is a
pipeline observation; the underlying KG gap is a double failure.

**Pattern:** The seed entity has no structured properties, and the Wikipedia page
has links to the target entity, but the surrounding text uses different vocabulary
than the question. Simple keyword matching fails.

### Example

**Question:** "What is the place of birth of Consort Liang's husband?"
**Gold answer:** Chinese _(likely a dataset annotation error — "Chinese" is extracted from
"Chinese Han dynasty" in the supporting sentence, not an actual place of birth; Emperor
Zhang's Wikipedia page has no `birth_place` infobox field; his likely birthplace is Luoyang)_
**Route:** `llm` → SPARQL (fails) → compositional (fails) → LogicRAG
**Entity linking issue:** Spotlight doesn't recognize "Consort Liang". SPARQL label fallback
links "birth" → `dbp:birth` (same pathology as Case 2).
**Pipeline result:** Wrong answer ("Yaoxu" — LogicRAG identifies wrong husband: Emperor Shun of Han
instead of Emperor Zhang of Han, then retrieves Shun's birthplace)

### DBpedia state

`dbr:Consort_Liang` — multilingual `description` fields plus `wikiPageWikiLink` edges,
but no structured biographical properties:
```
  description → "1st century Han dynasty imperial consort"
  description → "gemalin uit Oostelijke Han-dynastie (62-83)"
  wikiPageWikiLink → Emperor_Zhang_of_Han, Emperor_Guangwu_of_Han, ... (13 links)
```
No `dbo:spouse`, no `dbo:birthPlace`, no family properties at all.

### Wikipedia page (wikitext) — no infobox, pure prose

```
'''Consort Liang''' (梁貴人) (62(?) – 83), formally '''Empress Gonghuai''',
was an imperial consort to [[Emperor Zhang of Han]]...

...Consort Liang was a daughter of Liang Song (梁竦), a younger brother of
[[Liang Ji]]...
```

The link `[[Emperor Zhang of Han]]` is the answer to hop-1
(the "husband"), but the surrounding text says **"imperial consort to"**, not "husband"
or "spouse".

### What link extraction would do (with difficulty)

1. Parse prose → `[[Emperor Zhang of Han]]` with context "was an imperial consort to"
2. Keyword match: question says "husband", text says "imperial consort to" → **no direct match**
3. Would need a semantic equivalence map: `consort/imperial consort ↔ husband/spouse`
4. OR: use the entity's `description` ("imperial consort") + type check
   (`Emperor Zhang` is a `dbo:Person`) to infer the spousal relationship

### Why it's harder

Simple keyword overlap between question tokens and surrounding text doesn't work.
This requires either:
- A curated synonym map for relationship terms (consort/imperial consort ≈ spouse)
- Type-based inference: the page is about a "consort" and the linked entity
  is typed as royalty/emperor, so they're likely in a spousal relationship
- Or just falling back to LLM reasoning over the retrieved text

Note: Entity linking also fails here — Spotlight doesn't link "Consort Liang" at all
(links "birth" → `dbp:birth` instead). LogicRAG attempts to resolve via text, but
identifies the **wrong husband** (Emperor Shun of Han instead of Emperor Zhang of Han).
There are multiple "Consort Liang" entries across Han dynasty emperors; without entity
linking to disambiguate, the LLM picks the wrong one from web search results.

### Prevalence

Relatively rare in 2WikiMultiHopQA. Most questions use standard vocabulary
(director, father, mother, spouse, author) that matches Wikipedia conventions.
Historical/royal questions are the main exception.

---

## Case 4: Entity Linking Failure (redirect stub, no person page)

**KG Scenario:** E (ENTITY_NOT_FOUND) — 39 questions (3.9%)

**Pattern:** The entity linking step (Spotlight + SPARQL label search) fails to link
the entity entirely. The DBpedia entry is a redirect stub pointing to a peerage title
article, not a person's biography. No entity is linked, so the pipeline has no KG
entry point.

### Example

**Question:** "Which country Thomas Lyon-Bowes, Lord Glamis's father is from?"
**Gold answer:** British
**Route:** no entities linked → `no_schema_fallback` → compositional (fails) → LogicRAG
**Pipeline result:** Scotland (wrong — LogicRAG finds 11th Earl of Strathmore, says "Scotland")

### DBpedia state

`dbr:Thomas_Lyon-Bowes,_Lord_Glamis` — **redirect stub** to `dbr:Earl_of_Strathmore_and_Kinghorne`.
Has `owl:sameAs`, `dbo:wikiPageRedirects`, and `gold:hypernym` triples but **no person properties**
(no birthDate, parent, nationality).

### Wikipedia page

The page is about the **Earl of Strathmore peerage title**, not a specific person
named Thomas Lyon-Bowes:

```
{{Infobox nobility title
| name = Earl of Strathmore and Kinghorne
| creation_date = 1606 (Scottish title)
| first_holder = [[Patrick Lyon, 1st Earl of Kinghorne]]
| present_holder = [[Simon Bowes-Lyon, 19th Earl of Strathmore...]]
}}

'''Earl of Strathmore and Kinghorne''' is a title in the [[Peerage of Scotland]]...
```

This is an `{{Infobox nobility title}}`, not an `{{Infobox person}}`. It lists
holders of the title across centuries, not biographical data about one person.

### Why no approach fixes this

The root cause is upstream: entity linking can't find "Thomas Lyon-Bowes, Lord Glamis"
at all — Spotlight doesn't recognize it, and SPARQL label search doesn't match. The
DBpedia entry is just a redirect stub to the peerage article. Even if it were linked,
the Wikipedia page has many `[[links]]` to various earls across centuries, and without
knowing WHICH Thomas Lyon-Bowes the question refers to, link extraction would add noise.

### What would help instead

- Better entity linking: Spotlight disambiguation or candidate ranking
- Detecting the mismatch: if the Wikipedia infobox type is `nobility title`
  but the question expects a person, flag it and try alternative entities
- Using the 2WikiMultiHopQA supporting facts to identify the correct entity

---

## Case 5: Hop-2 Entity is an Empty Shell

**KG Scenario:** C (HOP2_MISSING) — 158 questions (15.8%)

**Pattern:** The seed entity has the hop-1 predicate in DBpedia, so SPARQL resolves
hop-1 successfully. But the intermediate (hop-2) entity is a near-empty shell —
the hop-2 predicate doesn't exist. The SPARQL chain breaks at the second hop and
falls back to TEXT.

This is the **most common failure mode** for multi-hop questions where the seed
entity works (hop-1 OK). Many bridge questions follow "property of property of X"
where DBpedia can resolve hop-1 (e.g., `dbo:director`) but the intermediate entity
(e.g., the director) lacks the hop-2 property (e.g., `dbo:spouse`).

### Example

**Question:** "Who is the spouse of the director of film My Three Merry Widows?"
**Gold answer:** Mapy Cortés
**Route:** `rule:direct_property` → SPARQL (fails at hop-2, no `dbo:spouse` on Fernando Cortés) → LogicRAG
**Pipeline result:** "Mapy Cortés" (correct — LogicRAG decomposes into "Who directed?" → SPARQL finds
Fernando Cortés, then "Who is spouse of Fernando Cortés?" → TEXT finds Mapy Cortés)

### DBpedia state — Hop 1 (film → director): WORKS

`dbr:My_Three_Merry_Widows` has structured properties:
```
  dbo:director          → Fernando_Cortés
  dbo:cinematography    → José_Ortiz_Ramos
  dbo:editing           → Carlos_Savage
  dbo:language          → Spanish_language
  dbp:runtime           → 100.0
```
SPARQL resolves: `dbo:director → dbr:Fernando_Cortés` ✓

### DBpedia state — Hop 2 (director → spouse): BREAKS

`dbr:Fernando_Cortés` has ~50 triples but **no structured person properties**:
```
  description → "Puerto Rican film actor, writer and director (1909–1979)"
  description → "director de cine y actor puertorriqueño"
  wikiPageWikiLink → (34 filmography links)
  wikiPageUsesTemplate → Infobox_person, Birth_date, Death_date_and_age, ...
```
No `dbo:spouse`, no `dbo:birthPlace`, no `dbo:birthDate`. The Wikipedia `{{Infobox person}}`
template exists but DBpedia failed to extract its structured fields.
The SPARQL 2-hop chain `Film → director → Person → spouse` breaks here.

### Wikipedia page for Fernando Cortés (wikitext)

```
{{Infobox person
| name         = Fernando Cortés
| birth_date   = {{Birth date|1909|10|4|mf=y}}
| birth_place  = [[San Juan, Puerto Rico]]
| death_date   = {{Death date and age|1979|4|28|1909|10|4|mf=y}}
| death_place  = [[Mexico City]], Mexico
| spouse       = [[Mapy Cortés]]
}}
```

The answer `[[Mapy Cortés]]` is a structured link in the `spouse` infobox field.

### What link extraction would do

1. SPARQL resolves hop-1: `dbo:director → dbr:Fernando_Cortés`
2. Hop-2 SPARQL fails (no `dbo:spouse` on Fernando Cortés)
3. Fetch Fernando Cortés's Wikipedia page with wikitext (not `explaintext=True`)
4. Parse infobox → `spouse = [[Mapy Cortés]]`
5. Question asks "spouse" → infobox key IS "spouse" → **direct answer, no LLM needed**

### Why it now works (via LogicRAG)

LogicRAG decomposes this into two sub-questions: "Who directed My Three Merry Widows?"
(answered via SPARQL: Fernando Cortés) and "Who is the spouse of Fernando Cortés?"
(answered via TEXT: Mapy Cortés). This produces the correct full answer. However, the
LogicRAG path requires 2 LLM calls for decomposition + 2 for sub-Q resolution + 1 for
synthesis = 5 LLM calls total. With infobox extraction, the answer would be found in 0
LLM calls (SPARQL hop-1 → parse infobox `spouse = [[Mapy Cortés]]`).

### Variations

This pattern applies to many 2WikiMultiHopQA bridge questions:
- "Where was the director of [Film] born?" — Film has `dbo:director`, director lacks `dbo:birthPlace`
- "Who is [Person]'s spouse?" — Person exists but has no `dbo:spouse`
- "Where did [Person]'s father study?" — Person has `dbo:parent`, parent lacks `dbo:almaMater`

---

## Case 6: Entity Disambiguation Failure (wrong entity linked confidently)

**KG Scenario:** Appears as B (HOP1_MISSING) in classification, but this is an artifact.
The coverage analysis linked to `dbr:Thomas_Jefferson` (president) instead of
`dbr:Thomas_Jefferson_(film)`. The correct film entity has `dbo:director → Ken_Burns`,
so this question is actually **A (KG_COMPLETE)** with correct entity linking.
This is a pipeline failure (disambiguation), not a KG coverage gap.

**Pattern:** Entity linking resolves a named entity to the **wrong DBpedia resource** with
high confidence. SPARQL executes successfully and returns a plausible-looking but incorrect
answer. Unlike Case 4 (no entity linked), here the pipeline has full confidence in the wrong
entity and returns a wrong answer without falling back.

### Example

**Question:** "Where was the director of film Thomas Jefferson (Film) born?"
**Gold answer:** Brooklyn (the director is Ken Burns)
**Route:** `rule:bridge_hop2` → SPARQL (succeeds on wrong entity)
**Entity linked:** `dbr:Thomas_Jefferson` (the US president, not the film)
**Pipeline result:** "Shadwell, Virginia" (president's birthplace — wrong)

The "(Film)" disambiguation suffix is lost during entity linking. Spotlight returns the
president with high confidence. The SPARQL query in eval logs confirms:
```sparql
SELECT ?birthPlace WHERE {
  dbr:Thomas_Jefferson dbo:birthPlace ?birthPlace .
}
```
Returns `Shadwell,_Virginia` — the president's birthplace, not Ken Burns's.

### DBpedia state

The correct entity `dbr:Thomas_Jefferson_(film)` exists and has full structured data:
```
  dbo:director → dbr:Ken_Burns
  dbp:director → dbr:Ken_Burns
```
And Ken Burns has `dbo:birthPlace → dbr:Brooklyn`. The 2-hop chain
`Thomas_Jefferson_(film) → director → Ken_Burns → birthPlace → Brooklyn` would produce
the gold answer — but entity linking never reaches it.

### Why link extraction doesn't help

This is an upstream entity linking problem. Even with infobox link extraction, the pipeline
would be parsing the president's Wikipedia page (which has a `birthPlace` in the infobox),
confidently returning the wrong answer. The fix requires disambiguation-aware entity linking
that considers the question context ("director of film").

---

## Case 7: Context Assembly Filtering (answer on page, dropped by keyword filter)

**KG Scenario:** A (KG_COMPLETE) — the KG has `dbo:director` and `dbo:birthPlace` for
both hops. This is NOT a KG coverage gap — it is a pipeline execution failure.
The keyword-density filter in `extract_relevant_sections()` drops the answer-bearing
paragraph. Included here because it represents a failure mode on A-scenario questions
that should be getting correct answers.

**Pattern:** Entity linking and Wikipedia retrieval both succeed. The answer IS present
on the retrieved Wikipedia page, but `extract_relevant_sections()` inside `assemble_context()`
**drops the answer-bearing paragraph** because it fails the keyword-density filter. The LLM
sees an incomplete context and returns "not enough information."

### Example

**Question:** "What is the place of birth of the director of film Tammy And The T-Rex?"
**Gold answer:** British
**Route:** `rule:bridge_hop2` → SPARQL (fails, LLM-generated query wrong) → TEXT
**Pipeline result:** "There is not enough information in the text to answer this question."

### Why it happens

The bridge entity Stewart Raffill is correctly discovered via `_find_context_entities()`,
and his Wikipedia page is fetched (2,843 chars — well under the 12K budget). But
`extract_relevant_sections()` applies a keyword-density filter before the LLM sees the text:

| Paragraph | Content | Keyword hits | Survives? |
|-----------|---------|-------------|-----------|
| 0 (lead) | "Stewart Raffill is a **British** writer and director." | 1 (`director`, 22% freq → not rare) | **DROPPED** |
| 2 (films) | Filmography list mentioning Tammy and T-Rex | 4 (`director`, `film`, `tammy`, `t-rex`) | KEPT |

The lead paragraph — the **only** one containing "British" — has just one keyword hit
(`director`), which appears in 22% of paragraphs (not rare enough for the 1-rare-keyword
threshold). It gets dropped. The LLM sees only the filmography list and returns "not enough
information."

**Note:** This is NOT a 12K budget issue (the page is only 2.8K). The keyword-density
filter in `extract_relevant_sections()` is the mechanism — it can discard answer-bearing
paragraphs even on short pages when the answer uses vocabulary not present in the question.

### DBpedia state

Both hops exist in DBpedia:
```
  dbr:Tammy_and_the_T-Rex → dbo:director → dbr:Stewart_Raffill
  dbr:Stewart_Raffill → dbo:birthPlace → dbr:Kettering
```
But the gold answer is "British" (nationality), not "Kettering" (city). The answer comes
from Wikipedia prose, not from any DBpedia property.

### What link extraction would do

For this specific case, infobox extraction wouldn't help — Stewart Raffill's nationality
is in prose, not in an infobox field. However, for the broader pattern (answer on page but
filtered out), infobox extraction bypasses the keyword filter entirely for structured fields.

### Prevalence

~3–10% of TEXT-route failures across eval runs (3.7% mechanistically confirmed via the
keyword-filter mechanism, up to 9.7% including all "not enough information" responses on
answerable questions).

---

## Summary Table

| Case | KG Scenario | Seed DBpedia | Hop-1 | Hop-2 Entity | Wikipedia Signal | Status |
|------|-------------|-------------|-------|-------------|-----------------|--------|
| **1. Missing predicate** | **B** | Has props, no director | Missing | N/A | Seed infobox `[[links]]` | **Fixed** (bridge entity linking) |
| **2. No infobox** | **B** | descriptions + wiki links | Missing | N/A | Seed prose `[[links]]` | Open — Medium (prose + keyword) |
| **3. Semantic mismatch** | **D** | descriptions + wiki links | Missing | Also missing | Prose links, role words differ | Open — Hard (synonym map / LLM) |
| **4. Entity linking** | **E** | Redirect stub | N/A | N/A | Wrong page entirely | Open — Upstream (entity linking) |
| **5. Hop-2 shell** | **C** | Has hop-1 prop | Works | No person props | Hop-2 entity infobox `[[links]]` | Open — Easy (infobox regex) |
| **6. Disambiguation** | **A**\* | Wrong entity linked | Wrong entity | N/A | Wrong page, correct answer format | Open — Upstream (entity linking) |
| **7. Keyword filter** | **A** | Correct entity | Works or N/A | Varies | Answer on page, dropped by filter | Open — Medium (infobox bypasses filter) |

\* Case 6 appears as B in the classification data due to inherited disambiguation error; the correct entity has full KG coverage.

### Expected impact on 2WikiMultiHopQA

- **Case 1** (missing hop-1 predicate): **Fixed by bridge entity linking** (2026-03-04).
  LogicRAG now splits compound answers, validates entities via `kg_client.search_entity_by_name()`,
  and routes hop-2 to SPARQL when the bridge entity has structured data. Converts TEXT→TEXT
  (hallucination-prone) to TEXT→SPARQL (grounded). Affects ~26% of film questions where the seed
  entity lacks the hop-1 predicate but the intermediate entity has structured hop-2 data.
  Infobox extraction would still reduce LLM calls (0 vs. 5) but is no longer needed for
  correctness on this pattern.

- **Case 5** (hop-2 shell): Still the most common remaining failure. Most films, books,
  and notable people have Wikipedia infoboxes. Simple regex parsing of `| key = [[Entity]]`
  patterns. Zero LLM calls needed. Note: LogicRAG already handles many of these via TEXT
  fallback (see example), but infobox extraction would be more reliable and efficient.

- **Case 7** (keyword filter): Partially addressed by infobox extraction (bypasses the
  keyword filter for structured fields), but some answers live only in prose (e.g.,
  nationality in the lead paragraph). For those, the keyword filter itself needs tuning.

- **Case 2** (prose link extraction): Covers biographical pages without infoboxes.
  Requires HTML/wikitext parsing with surrounding-text keyword matching.
  Moderate implementation effort.

- **Case 3** (semantic mismatch): Rare in the benchmark. Historical/royal questions
  with non-standard vocabulary. Could be handled with a small synonym map or
  left to LLM reasoning as today.

- **Cases 4 + 6** (entity linking failures): Require upstream entity linking improvements.
  Case 4 (entity not found) and Case 6 (wrong entity disambiguated) are both out of scope
  for link extraction — the wrong page is retrieved regardless.

### Implementation priority

1. **Infobox link extraction** (Case 5) — highest ROI, simplest to implement
2. **Prose link extraction with keyword matching** (Case 2) — medium ROI
3. **Keyword filter tuning** (Case 7) — low effort, fixes ~3–10% of TEXT failures
4. **Semantic synonym map** (Case 3) — low ROI for the benchmark
5. **Entity linking improvements** (Cases 4 + 6) — separate workstream

### Changelog

- **2026-03-04:** Case 1 fixed by bridge entity linking in `logicrag_decomposition.py`.
  Added `_split_compound_answer()`, `_resolve_answer_entities()`, `_make_fallback_entity()`
  helpers. Replaces blind URI construction with SPARQL-validated entity linking for bridge
  answers. Verified on Katka's Reinette Apples question: now returns "Rēzekne" (correct)
  instead of "Reval" (hallucinated).
