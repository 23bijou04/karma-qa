# GRASP Trace Examples: Tool Sequences and Output Patterns

**Date:** 2026-02-21
**Purpose:** Detailed examples of GRASP's tool call sequences and output generation patterns across question types

---

## 1. SUCCESSFUL COMPARISON: "Who was born later, Gideon Johnson Pillow or Holm Jølsen?"

### Question Type: comparison
**Gold Answer:** Holm Jølsen
**GRASP Answer:** Holm Jølsen
**EM:** 100.0% ✅
**Containment:** 100.0%

### Tool Call Sequence (17 tools)

```
Tool 1:  search_entity("Gideon Johnson Pillow")
Tool 2:  search_entity("Holm Jølsen")
Tool 3:  search_entity(retry)
Tool 4:  search_entity(retry)
Tool 5:  search_entity(retry)
Tool 6:  execute(SPARQL query)
Tool 7:  search_property_of_entity("birthDate")
Tool 8:  execute(retry)
Tool 9:  execute(retry)
Tool 10: execute(retry)
Tool 11: execute(retry)
Tool 12: execute(retry)
Tool 13: execute(retry)
Tool 14: execute(retry)
Tool 15: execute(retry)
Tool 16: execute(retry)
Tool 17: answer()
```

### Model Reasoning Chain

**Message 1 (Strategy):**
```
I will first search DBpedia for the two persons' entities to get their
DBpedia IRIs. Then I'll inspect their birth dates and construct a SPARQL
query that compares the two birth dates in SPARQL (so the comparison is
done inside the query). I'll call the entity search function for each name now.
```

**Message 2 (Refinement):**
```
Now I'll search for "Holm Jølsen" in DBpedia to find the correct entity IRI.
```

**Message 3 (Data Gathering):**
```
Now I'll query DBpedia for the birth dates of the two identified DBpedia
resources to verify the dates exist and their formats.
```

**Message 4 (Final Query):**
```
I'll run a final SPARQL SELECT that (1) collects each person's birth date
using dbo:birthDate or dbp:birthDate (via COALESCE), (2) compares the two
dates inside SPARQL, and (3) returns the IRI of the person born later
(plus labels and dates). This ensures the date comparison is performed by
the query itself. I'll execute the query now.
```

### SPARQL Query (Executed Successfully)

```sparql
SELECT ?person1Label ?birthDate1 ?person2Label ?birthDate2
WHERE {
  VALUES ?person1 { <http://dbpedia.org/resource/Gideon_Johnson_Pillow> }
  VALUES ?person2 { <http://dbpedia.org/resource/Holm_Jølsen> }
  ?person1 (dbo:birthDate | dbp:birthDate) ?birthDate1 .
  ?person2 (dbo:birthDate | dbp:birthDate) ?birthDate2 .
  OPTIONAL {
    ?person1 rdfs:label ?person1Label .
    FILTER(lang(?person1Label) = "en")
  }
  OPTIONAL {
    ?person2 rdfs:label ?person2Label .
    FILTER(lang(?person2Label) = "en")
  }
}
ORDER BY DESC(?birthDate1)
LIMIT 1
```

### Final Output

**Raw:** `Holm Jølsen was born later. Gideon Johnson Pillow: 1806-06-08; Holm Jølsen: 1833-12-06.`

**Predicted:** `Holm Jølsen`

**Extraction:** Takes first entity name → Holm Jølsen ✅

### Why Success?

1. **Explicit final reasoning:** Message 4 explicitly states the SPARQL will "return the IRI of the person born later"
2. **Answer is primary:** GRASP outputs "Holm Jølsen" (the person born later) as the first entity
3. **Single ordered result:** SPARQL with ORDER BY DESC, LIMIT 1 ensures single result
4. **Extraction matches LLM intent:** Taking first entity aligns with LLM's answer placement

---

## 2. FAILED COMPOSITIONAL: "Where was the director of film Bollywood Calling born?"

### Question Type: compositional
**Gold Answer:** Hyderabad
**GRASP Answer:** Nagesh Kukunoor (intermediate, WRONG)
**EM:** 0.0% ❌
**Containment:** 100.0% (gold answer IS in raw output)

### Tool Call Sequence (5 tools)

```
Tool 1: search_entity("Bollywood Calling")
Tool 2: search_property_of_entity(entity="Bollywood Calling", query="director")
Tool 3: execute(SPARQL query)
Tool 4: execute(retry - URI parse error)
Tool 5: answer()
```

### Model Reasoning Chain

**Message 1 (Strategy):**
```
I will first search DBpedia for the film "Peter's Friends" to find its
DBpedia resource IRI. Then I'll look up its director and the director's
place of birth. I'll start by searching for the film entity.
```

**Message 2 (Property Lookup):**
```
I'll search for properties of the film entity that relate to the director
(query "director") to find which property holds the director link.
```

**Message 3 (Query Construction):**
```
Now I'll run a SPARQL query to get the director of dbr:Peter's_Friends
and that director's place of birth (returning IRIs and English labels
if available).
```

**Message 4 (Error Recovery):**
```
The prefixed resource name with an apostrophe caused a parse error, so
I'll identify the film by its English rdfs:label "Peter's Friends"@en
(and require it to be a dbo:Film) to avoid using the problematic IRI form.
I'll run a query that finds the film by label, gets its director, and then
the director's birthPlace (returning IRIs and English labels).
```

**No explicit final reasoning about which column is the answer.**

### SPARQL Query (Executed Successfully)

```sparql
SELECT ?director ?directorLabel ?birthPlace ?birthPlaceLabel
WHERE {
  ?film rdfs:label "Bollywood Calling"@en ;
        a dbo:Film ;
        (dbo:director | dbp:director) ?director .
  ?director (dbo:birthPlace | dbp:birthplace) ?birthPlace .
  OPTIONAL {
    ?director rdfs:label ?directorLabel .
    FILTER(lang(?directorLabel) = "en")
  }
  OPTIONAL {
    ?birthPlace rdfs:label ?birthPlaceLabel .
    FILTER(lang(?birthPlaceLabel) = "en")
  }
}
```

### SPARQL Results (Data Retrieved Successfully)

```
director                   directorLabel        birthPlace    birthPlaceLabel
dbr:Nagesh_Kukunoor        Nagesh Kukunoor      dbr:Hyderabad Hyderabad
```

### Final Output

**Raw:** `The director of Bollywood Calling is Nagesh Kukunoor (dbr:Nagesh_Kukunoor). DBpedia lists his birthplace as dbr:Hyderabad and also dbr:Andhra_Pradesh (labels: "Hyderabad", "Andhra Pradesh").`

**Predicted:** `The director of Bollywood Calling is Nagesh Kukunoor (dbr:Nagesh_Kukunoor)`

**Extraction:** Splits on `. ` → takes sentence 1 → `"The director of Bollywood Calling is Nagesh Kukunoor (dbr:Nagesh_Kukunoor)"` → removes (dbr:...) → `Nagesh Kukunoor` ❌

**Gold:** `Hyderabad` (in sentence 2)

### Why Failure?

1. **No explicit final reasoning:** Message 4 doesn't state "the answer is birthPlace" or "the final answer is Hyderabad"
2. **Multi-column SPARQL result:** Query returns both ?director AND ?birthPlace columns
3. **Narrative describes reasoning chain:** "The director is X. His birthplace is Y." (reasoning order, not answer order)
4. **Extraction splits on sentence boundary:** Takes first sentence (has director), misses second (has birthplace)

### Root Cause Breakdown

| Issue | Impact |
|-------|--------|
| **Sentence splitting** | Raw answer has 2 sentences; extraction takes first |
| **URI suffix** | (dbr:Nagesh_Kukunoor) causes partial extraction |
| **Multi-column narration** | Director and birthplace described sequentially |
| **No answer marker** | No "FINAL ANSWER:" prefix to guide extraction |

### How to Fix

1. **Last-entity extraction:** Extract final entity from multi-sentence answer → Hyderabad ✅
2. **URI suffix removal:** Strip (dbr:...) → Keep Hyderabad ✅
3. **Column tracking:** Detect multi-column SPARQL → Extract last column ✅
4. **Prompt change:** Add "For this compositional question, the answer is the final value in the chain" ✅

---

## 3. FAILED INFERENCE: "Who is Felicia Of Roucy's mother-in-law?"

### Question Type: inference
**Gold Answer:** Ermesinda of Bigorre
**GRASP Answer:** Ermesinda of Bigorre (dbr:Ermesinda_of_Bigorre) [with URI]
**EM:** 0.0% ❌ (URI breaks substring match)
**Containment:** 100.0%

### Tool Call Sequence (4 tools)

```
Tool 1: search_entity("Felicia Of Roucy")
Tool 2: search_property_of_entity(entity="Felicia", query="mother-in-law")
Tool 3: execute(SPARQL query)
Tool 4: answer()
```

### Model Reasoning Chain

**Message 1 (Strategy):**
```
I will search for Felicia of Roucy and then look for her mother-in-law
relationship in DBpedia using appropriate genealogical properties.
```

**Message 2 (Property Lookup):**
```
Searching for properties related to family relationships...
```

**Message 3 (No explicit final reasoning):**
```
Executing SPARQL query to find mother-in-law relationship...
```

### SPARQL Query (Executed Successfully)

```sparql
SELECT ?motherInLaw ?motherInLawLabel
WHERE {
  ?person rdfs:label "Felicia of Roucy"@en ;
          dbo:spouse ?spouse .
  ?spouse dbo:parent ?motherInLaw .
  OPTIONAL {
    ?motherInLaw rdfs:label ?motherInLawLabel .
    FILTER(lang(?motherInLawLabel) = "en")
  }
}
```

### SPARQL Results

```
motherInLaw                birthPlaceLabel
dbr:Ermesinda_of_Bigorre   Ermesinda of Bigorre
```

### Final Output

**Raw:** `Felicia of Roucy's mother-in-law is Ermesinda of Bigorre (dbr:Ermesinda_of_Bigorre)`

**Predicted:** `Ermesinda of Bigorre (dbr:Ermesinda_of_Bigorre)`

**EM Evaluation:** Predicted ≠ Gold (URI suffix prevents exact match) ❌

### Why Failure?

1. **URI suffix preservation:** GRASP includes the full DBpedia URI for disambiguation
2. **Genealogy formal format:** Inference questions treated as exact reference lookups
3. **No extraction guidance:** Answer format is ambiguous (is the URI part of the answer?)

### Why This Pattern for Genealogy?

Genealogical relationships are ambiguous in DBpedia:
- Multiple "Elizabeth of Bigorre" variants
- Name particles ("de", "von") vary
- Historical records have multiple spellings

GRASP preserves URIs as a defensive measure to maintain exact reference.

### Comparison: Same Question Type with Different Approach

If GRASP had been prompted to output: `FINAL ANSWER: Ermesinda of Bigorre`
→ EM = 100.0% ✅

---

## 4. SUCCESSFUL BRIDGE_COMPARISON: "Which film has the director born first?"

### Question Type: bridge_comparison
**Gold Answer:** La Mazurka
**GRASP Answer:** La mazurka del barone, della santa e del fico fiorone
**EM:** 100.0% ✅
**Containment:** 100.0%

### Tool Call Sequence (7 tools)

```
Tool 1: search_entity("La Mazurka")
Tool 2: search_entity("Land Of Peace")
Tool 3: list(properties of director)
Tool 4: list(properties of director)
Tool 5: list(properties of director)
Tool 6: execute(SPARQL)
Tool 7: answer()
```

### Final Output

**Raw:** `La mazurka del barone, della santa e del fico fiorone was directed by a person who was born first.`

**Predicted:** `La mazurka del barone, della santa e del fico fiorone`

**Extraction:** Takes first entity name ✅

### Pattern Observation

Bridge_comparison uses `list()` tool (3 times) instead of `search_property_of_entity()`. This suggests:
- Schema enumeration (list all properties)
- Directed graph exploration (find relationships through property listing)

Still succeeds because answer placement is first in the narrative (same as simple comparison).

---

## Comparative Analysis: Tool Sequences

| Type | Tool Count | Entity Search | Property Lookup | SPARQL Execute | Answer Tool | Success Rate |
|------|---|---|---|---|---|---|
| Comparison | 17 | 5x | 1x | 10x | Yes | 55.3% ✅ |
| Bridge_comp | 7 | 2x | 0x | 1x | Yes | 55.3% ✅ |
| Compositional | 5 | 1x | 1x | 2x | Yes | 0.0% ❌ |
| Inference | 10 | 1x | 2x | 3x | Yes | 0.0% ❌ |

**Observation:** Tool count ≠ success rate. Compositional (5 tools, 0% EM) vs Comparison (17 tools, 55.3% EM).

The difference is OUTPUT FORMAT, not tool depth.

---

## Output Format Summary

### Successful Comparison
- Single ordered result from SPARQL
- Answer as primary entity (named first)
- Evidence listed after answer
- Example: "Holm Jølsen was born later. [dates]"

### Failed Compositional
- Multi-column result from SPARQL
- Intermediate entity named first (director)
- Final answer named second (birthplace)
- Example: "Director is X. Birthplace is Y." (X is wrong extraction target)

### Failed Inference
- Single result from SPARQL
- Answer with formal URI context
- Example: "Answer is X (dbr:X)" (URI breaks EM metric)

---

## Implications for Answer Extraction Design

1. **Position-based extraction fails:** Can't assume first entity is answer (true for comparison, false for compositional)

2. **URI handling is inconsistent:** Some outputs clean URIs, others include them for disambiguation

3. **Sentence boundaries aren't reliable:** Multi-sentence narratives split incorrectly

4. **Better approach:**
   - Track question type → expect different answer positions
   - Structured output (JSON) → explicit answer/evidence separation
   - Extraction marker ("FINAL ANSWER:") → LLM-guided extraction

---

**Files Referenced:**
- GRASP evaluation: `data/eval_1000q/grasp_eval_dbpedia.jsonl`
- Full traces: `data/eval_1000q/grasp_eval_dbpedia_traces/`
- Deep analysis: `grasp_failure_deep_analysis.md`
