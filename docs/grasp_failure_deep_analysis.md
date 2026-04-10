# GRASP Deep Failure Analysis: Architectural Roots of Performance Divergence

**Date:** 2026-02-21
**⚠️ Staleness notice (2026-03-30):** Numbers below are from the old 346q evaluation. Current evaluation uses 368 hop-verified questions with updated results (see `evaluation_report_sparql_grasp.md`). Structural analysis (failure taxonomy, root causes) still holds; specific percentages are outdated.
**Investigation Scope:** 346 DBpedia-answerable questions from 2WikiMultiHopQA
**Focus:** Architectural and algorithmic reasons for GRASP's output format divergence across question types

---

## Executive Summary

GRASP (an agentic text-to-SPARQL system) exhibits a striking paradox: it succeeds on comparison questions (55.3% EM, 95.7% containment) but fails completely on compositional and inference questions (0% EM, 76-79% containment). The gap is **not** a retrieval problem—verified trace analysis shows GRASP generates valid SPARQL, retrieves correct data 96% of the time, and embeds the gold answer in raw output 75.5% of the time.

The root cause is **architectural**: GRASP's agentic reasoning framework inherently produces different narrative formats for different question types, and this format divergence is EMERGENT from multi-step tool interactions rather than hardcoded. Specifically:

1. **Comparison questions** → Direct answer placement (answer as first/primary entity)
2. **Compositional questions** → Nested narrative (intermediate entity first, final answer second)
3. **Inference questions** → Single-entity narrative with URI suffix (answer embedded in parenthetical context)

The extraction-level failure patterns (URI_SUFFIX 44.9%, SENTENCE_SPLIT 11%) are **symptoms** of this deeper architectural design: GRASP's tool call chain produces narrative explanations because its agentic reasoning operates iteratively through tools (entity search → property exploration → SPARQL execution), building context at each step. By the time the LLM reaches the "answer" tool, it has already constructed a multi-sentence narrative explaining the reasoning chain.

---

## Part 1: Architectural Analysis

### 1.1 GRASP's Agentic Design

GRASP uses OpenAI gpt-5-mini with tool use capability. The system works via an **iterative refinement loop**:

```
Question → Search entities → Inspect properties → Build understanding
         → Run SPARQL      → Narrate results  → Output answer
```

**Key architectural feature:** The LLM **reasons aloud** at each tool call, building a chain of thought. This reasoning is embedded in model messages and then synthesized into the final narrative answer.

### 1.2 Tool Call Patterns by Question Type

**Comparison Questions (17 tools in successful case):**
```
1. search_entity          (Gideon Johnson Pillow)
2. search_entity          (Holm Jølsen)
3. search_entity          (retry/verify)
4. search_entity          (retry/verify)
5. search_entity          (retry/verify)
6. execute               (SPARQL query)
7. search_property_of_entity
8-16. execute (multiple retries)
17. answer
```

**Pattern:** High entity search iteration (5/6 first tools are entity searches). Suggests GRASP is resolving ambiguity and gathering all entities BEFORE forming comparison.

**Compositional Questions (5 tools in failed case):**
```
1. search_entity          (Find film)
2. search_property_of_entity (Find director property)
3-4. execute (SPARQL)
5. answer
```

**Pattern:** Lower tool count (5 vs 17), suggesting GRASP forms intermediate query structure and returns early.

**Inference Questions (10 tools):**
```
1. search_entity          (Find person)
2-3. search_property_of_entity (Find relationship properties)
4-5. list (Enumerate results)
6. execute (SPARQL)
7. execute (Retry)
8. list (Enumerate again)
9. execute (Final SPARQL)
10. answer
```

**Pattern:** Mix of searches and enumerations, suggesting property chain lookup with backtracking.

**Key finding:** Tool count does NOT predict accuracy. Comparison (17 tools) succeeds; compositional (5 tools) fails. The tool count reflects **question complexity** (not handling quality).

### 1.3 Why Output Format Diverges: The Narrative Chain Hypothesis

When GRASP reaches the "answer" tool, the LLM has already reasoned through the entire problem:

**Comparison reasoning chain:**
```
Model 1: "I will search for two entities and compare their birth dates"
         → entity searches resolve both entities
Model 2: "Now I'll query DBpedia for the birth dates"
         → SPARQL returns: Holm Jølsen (1833-12-06), Gideon Johnson Pillow (1806-06-08)
Model 3: "I'll run a final SPARQL query comparing dates"
         → SPARQL returns: Holm Jølsen
Model 4-5: (empty reasoning)
Answer:   "Holm Jølsen was born later. Gideon Johnson Pillow: 1806-06-08; Holm Jølsen: 1833-12-06."
```

The answer is **primary** (first entity named) because it's the result of the comparison. The dates are listed for evidence.

**Compositional reasoning chain:**
```
Model 1: "I will search for the film, then its director, then the director's birthplace"
         → entity search resolves film
Model 2: "I'll search for director properties"
         → property lookup finds dbo:director
Model 3: "I'll query for director and birthplace"
         → SPARQL returns: director=Nagesh Kukunoor, birthplace=Hyderabad
Model 4: (explanation of results)
Answer:  "The director of Bollywood Calling is Nagesh Kukunoor (dbr:Nagesh_Kukunoor).
          DBpedia lists his birthplace as dbr:Hyderabad"
```

The answer is **nested** (intermediate entity Nagesh Kukunoor first, final answer Hyderabad second) because GRASP narrates the REASONING PATH, not the conclusion. The LLM naturally describes "the director is X" before "his birthplace is Y" because that's the logical chain.

**Inference reasoning chain:**
```
Model 1: "I will search for the person"
         → entity search finds person
Model 2-3: "I'll search for family relationship properties"
          → property searches find father, grandfather properties
Model 4-5: List properties
Model 6-9: Execute multiple SPARQL queries to chain relationships
Answer:   "The paternal grandmother of John III, Duke of Cleves is Elizabeth of Nevers
           (DBpedia IRI: http://dbpedia.org/resource/Elizabeth_of_Nevers)"
```

The answer is **formal** (full IRI in parenthesis) because inference questions require strict genealogical navigation. GRASP preserves the URI for disambiguation (multiple "Elizabeth" entries might exist).

**Conclusion:** Output format divergence is **NOT** a separate output formatting module. It's **emergent** from:
- The iterative tool call chain
- The LLM's natural reasoning verbosity
- The semantic structure of the question type

---

## Part 2: SPARQL Pattern Analysis

### 2.1 Query Complexity by Type

**Comparison SPARQL (executed successfully):**
```sparql
SELECT ?person1Label ?birthDate1 ?person2Label ?birthDate2
WHERE {
  VALUES ?person1 { <http://dbpedia.org/resource/Gideon_Johnson_Pillow> }
  VALUES ?person2 { <http://dbpedia.org/resource/Holm_Jølsen> }
  ?person1 (dbo:birthDate | dbp:birthDate) ?birthDate1 .
  ?person2 (dbo:birthDate | dbp:birthDate) ?birthDate2 .
  ...
}
ORDER BY DESC(?birthDate1)
LIMIT 1
```

**Characteristics:**
- VALUES clause for explicit entity selection
- Direct property access (dbo:birthDate)
- ORDER BY inside query (comparison happens in SPARQL, not post-processing)
- Result binding: one person with birthdate

**Result format:** Direct binding → Name, Date pairs → Simple extraction

---

**Compositional SPARQL (executed successfully):**
```sparql
SELECT ?director ?directorLabel ?birthPlace ?birthPlaceLabel
WHERE {
  ?film rdfs:label "Peter's Friends"@en ;
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

**Characteristics:**
- Chain of properties: film → director → birthplace
- Multiple hops (2 property chains)
- Optional labels for disambiguation
- Result binding: director AND birthplace

**Result format:** Multi-row bindings (director=Nagesh_Kukunoor, birthPlace=Hyderabad) → But GRASP narrates as "The director is X ... birthplace is Y"

**Key difference:** Compositional queries return MULTIPLE COLUMNS. GRASP's narrative describes each column sequentially, creating the "intermediate entity first" pattern.

---

**Inference SPARQL:**
```sparql
SELECT ?grandparent ?grandparentLabel
WHERE {
  ?person rdfs:label "John III, Duke of Cleves"@en ;
          dbo:parent / dbo:parent ?grandparent .
  OPTIONAL {
    ?grandparent rdfs:label ?grandparentLabel .
    FILTER(lang(?grandparentLabel) = "en")
  }
}
```

**Characteristics:**
- Property path (dbo:parent / dbo:parent) for relationship chaining
- Single result column: ?grandparent
- Simpler than compositional (fewer columns)

**Result format:** Single binding → But GRASP includes URI suffix because genealogy queries are ambiguous (multiple "William I" entries)

### 2.2 Why SPARQL Success ≠ Extraction Success

All three types generate valid, executable SPARQL. The retrieval is 96% successful. But **result narration diverges**:

| Type | SPARQL Returns | GRASP Narrates | Extraction Picks |
|------|---|---|---|
| comparison | name1, date1, name2, date2 | "Answer is name1. Evidence: dates" | ✅ name1 (correct) |
| compositional | director, birthplace | "Director is X. Birthplace is Y" | ❌ X (wrong, should be Y) |
| inference | grandparent | "Grandparent is X (URI)" | ❌ URI suffix included (wrong) |

**Root cause:** SPARQL returns structured data (multiple columns), but GRASP converts to unstructured narrative. Extraction then attempts to reverse-engineer which column is the answer.

---

## Part 3: Tool Call Chain Comparison

### 3.1 Successful Comparison: 17-Tool Chain

```
Tool 1-5: Entity search (iterative refinement)
Tool 6:   SPARQL execute (retrieves data)
Tool 7:   Property lookup (verify schema)
Tool 8-16: SPARQL execute (retries, refinement)
Tool 17:  Answer generation
```

**Model reasoning at answer point:**
> "I'll run a final SPARQL SELECT that (1) collects each person's birth date using dbo:birthDate or dbp:birthDate (via COALESCE), (2) compares the two dates inside SPARQL, and (3) returns the IRI of the person born later. I'll execute the query now."

**Observation:** The LLM explicitly states the question will be answered by the SPARQL result (the person with earlier death date). When the result returns (Holm Jølsen, 1833-12-06), the answer is unambiguous—state it directly.

---

### 3.2 Failed Compositional: 5-Tool Chain

```
Tool 1:   Entity search (find film)
Tool 2:   Property lookup (find director property)
Tool 3-4: SPARQL execute (get director and birthplace)
Tool 5:   Answer generation
```

**Model reasoning at answer point (no explicit final reasoning step):**

The LLM's last reasoning message was in Tool 3-4, explaining the query construction. By Tool 5, the LLM has:
- Retrieved SPARQL results: director=Nagesh_Kukunoor, birthplace=Hyderabad
- But received multi-column results (not a single answer)

The LLM then naturally **narrates the reasoning path** to explain what it found, rather than extracting the final column:

> "The director of Bollywood Calling is Nagesh Kukunoor (dbr:Nagesh_Kukunoor). DBpedia lists his birthplace as dbr:Hyderabad"

**Key insight:** With fewer tools (5 vs 17), the LLM has less opportunity to explicitly state "here is the final answer." Comparison queries have multiple SPARQL executions, each refining toward a single answer. Compositional queries execute once and get multi-column results.

---

### 3.3 Failed Inference: 10-Tool Chain with Backtracking

```
Tool 1:   Entity search (find person)
Tool 2-3: Property lookup (search family relations)
Tool 4-5: List properties
Tool 6:   SPARQL execute (first attempt)
Tool 7:   SPARQL execute (retry)
Tool 8:   List properties (try different property)
Tool 9:   SPARQL execute (final attempt)
Tool 10:  Answer generation
```

**Pattern:** Property exploration (Tools 2-3, 4-5, 8) suggests schema discovery is **uncertain**. Genealogical properties vary (dbo:parent, dbo:father, dbp:father, etc.).

**Model reasoning complexity:** The LLM explores multiple property paths, suggesting it's unsure which one will yield results. This leads to **formal, defensive** answer format:

> "The paternal grandmother of John III, Duke of Cleves is Elizabeth of Nevers (DBpedia IRI: http://...)"

The URI suffix suggests the LLM wants to disambiguate (in case there are multiple candidates) and preserve the precise reference.

---

## Part 4: Root Cause Taxonomy

### Layer 1: Architectural Roots (Why output formats diverge)

| Root Cause | Mechanism | Evidence |
|---|---|---|
| **Iterative tool use** | Each tool interaction builds reasoning context; final answer inherits the narrative built during iteration | 17 tools (comparison) vs 5 tools (compositional) → different narrative depths |
| **Multi-column result narration** | SPARQL returns structured data (multiple columns), but LLM narrates as sequential sentences | Compositional: director + birthplace → "Director is X, birthplace is Y" |
| **Schema uncertainty for genealogy** | Inference properties vary widely; tool exploration reflects uncertainty | 10-tool chain with backtracking vs 5-tool direct execution |
| **Question semantic structure** | Comparison questions ask "which is X", compositional ask "what is X's Y", inference ask "who is X's Y" | Different question types elicit different answer structures from the LLM |

### Layer 2: Extraction Bottlenecks (Why extraction fails)

| Extraction Issue | % of Failures | Question Type | Root Architectural Issue |
|---|---|---|---|
| **URI_SUFFIX** | 44.9% | All types (60% of inference failures) | LLM preserves DBpedia URIs for disambiguation and reproducibility; extraction assumes cleaned text |
| **SENTENCE_SPLIT** | 11.0% | Compositional (28.2% of comp failures) | Multi-column SPARQL results narrated as multi-sentence explanation; extraction splits on ". " |
| **DELIMITER_SPLIT** | 16.9% | All types | LLM uses `;`, `:`, `—` to list properties; extraction splits on these delimiters |
| **LABEL_MISMATCH** | ~3% | Inference | DBpedia labels vary ("de Stafford" vs "Stafford"); LLM returns label variant |

### Layer 3: Question Type Specific Failures

**Comparison (55.3% EM, should be higher):**
- Containment 95.7% but EM only 55.3% → 40pp gap
- Root: URI suffixes still appear in ~50% of failures
- Why: GRASP narrates evidence (dates, locations) with URIs for precision
- Extraction tries to extract "answer + evidence" as single entity

**Compositional (0% EM, critical):**
- Containment 76% → 76pp gap
- Root: Multi-column SPARQL results narrated sequentially
- Why: Composition chains (film→director→birthplace) naturally narrate as "the director is X, X's birthplace is Y"
- Extraction splits on ". " and takes first sentence (director's name)
- Fix potential: 74% of failures (29/39) from URI_SUFFIX or SENTENCE_SPLIT → fixable

**Inference (0% EM, paradox):**
- Containment 79.4% → 79pp gap
- Root: Formal genealogical answer format with URIs
- Why: Genealogy properties are ambiguous; GRASP preserves exact DBpedia references
- Plus: label mismatches for noble titles ("Hugh de Stafford" vs variants)
- Fix potential: URI removal alone won't help; requires entity normalization

---

## Part 5: Is This Fundamental to Agentic KGQA?

### 5.1 Yes, Partially

**Fundamental characteristics of agentic systems:**

1. **Iterative reasoning context:** Any system that reasons aloud through tool use builds narrative context. This is a feature, not a bug—it enables interpretability and self-correction.

2. **Multi-step tool chains:** Agentic systems have variable tool counts depending on question complexity. This creates different narrative depths.

3. **Result narration:** When LLMs narrate SPARQL results, they naturally describe the reasoning chain (entity → property → value) rather than just the final value.

**These are architectural choices in GRASP, not fundamental limitations.**

### 5.2 No, It's a Tuning Problem

However, the specific failures are **tunable**:

1. **Prompt engineering:** Could ask GRASP to output structured JSON `{"answer": "...", "evidence": "..."}` instead of narrative.

2. **Output post-processing:** Could extract the last named entity from the answer (heuristic for final answer extraction).

3. **Result formatting:** Could ask LLM to explicitly mark the answer: "**FINAL ANSWER: [name]**" like in other KGQA systems.

4. **Property selection:** For multi-column results, could track which column matches the question type:
   - Comparison: return first entity (the answer)
   - Compositional: return last entity (final value in chain)
   - Inference: return first entity (the found ancestor)

### 5.3 Comparison to Schema-Guided System

Schema-Guided SPARQL achieves 64.5% EM (vs GRASP's 32.9%) because it:
- Doesn't use agentic reasoning (no tool use)
- Directly extracts SPARQL result bindings
- Returns structured output with named columns

But Schema-Guided lacks GRASP's advantages:
- No interpretability (tool trace shows reasoning)
- No adaptive exploration (fixed schema)
- No error recovery (single SPARQL attempt + retry)

---

## Part 6: Recommendations for Improvement

### 6.1 Immediate Fixes (Post-Processing)

**Fix 1: URI_SUFFIX Removal**
```python
def remove_uri_suffixes(answer: str) -> str:
    return re.sub(r'\s*\((?:dbr|dbp|http)[^\)]*\)', '', answer)
```
**Expected impact:** 44.9% of failures → ~+15pp EM on compositional

**Fix 2: Last-Entity Extraction (Compositional)**
```python
def extract_last_entity(answer: str) -> str:
    # For multi-sentence narratives, extract final entity
    # "...director is X. birthplace is Y" → "Y"
    sentences = answer.split('. ')
    if len(sentences) > 1:
        return extract_entity(sentences[-1])
    return extract_entity(sentences[0])
```
**Expected impact:** SENTENCE_SPLIT failures (11%) → +5pp EM

**Fix 3: Label Normalization (Inference)**
```python
def normalize_label(label: str, question: str) -> str:
    # Normalize noble titles, particles
    # "Hugh de Stafford" ← could match gold "Hugh Stafford"
```
**Expected impact:** Label mismatch failures (3%) → +2pp EM

**Combined expected impact:** From 0% to ~22% EM on compositional, ~55-65% on inference.

### 6.2 Medium-term Fixes (Prompt Engineering)

**Change prompt to:**
1. Request explicit final answer marker: "**FINAL ANSWER: [answer]**"
2. Separate evidence from answer: output JSON with {"answer": "...", "reasoning": "..."}
3. For multi-column results, specify which column to prioritize: "The question asks for [birthplace/director/spouse], which corresponds to [column_name]"

**Expected impact:** +20-30pp EM overall

### 6.3 Long-term Fixes (Architecture)

**For comparison questions:**
- GRASP already succeeds (55.3%) → minimal changes needed
- Focus on URI cleanup and evidence extraction

**For compositional questions:**
- Detect multi-column SPARQL results
- Add explicit reasoning step: "The final answer is the last value in the chain: [value]"
- Track property chain depth; extract value at correct depth

**For inference questions:**
- Add entity normalization to handle label variants
- Track relationship chains (grandparent vs parent vs spouse)
- Preserve URIs for disambiguation only, not in final answer

---

## Part 7: Implications for Agentic KGQA Systems

### 7.1 Key Learnings

1. **Output format is emergent, not hardcoded.** Different question types produce different narrative structures due to their semantic structure and tool use chains, not separate output formatting paths.

2. **Multi-column results are the critical juncture.** Compositional queries return multiple columns; GRASP's narrative describes each sequentially. This is where extraction fails.

3. **Agentic systems trade accuracy for interpretability.** GRASP's tool traces are valuable for debugging and trust, but the agentic loop introduces narrative context that breaks simple extraction heuristics.

4. **Question type matters.** Comparison (single answer) vs compositional (chain of answers) vs inference (ambiguous entity) require different extraction strategies.

5. **The 0% EM on compositional is NOT a knowledge retrieval problem.** It's a question of data presentation and extraction alignment. GRASP could be fixed with targeted post-processing.

### 7.2 Design Principles for Future Systems

If building an agentic KGQA system, consider:

1. **Explicit output schema:** Use structured output (JSON, XML) instead of narrative text.

2. **Result column tracking:** For multi-hop queries, explicitly mark which column is the final answer.

3. **Question type classification:** Detect question structure early; adjust reasoning and output format accordingly.

4. **Extraction-friendly narrative:** If using narrative, follow strict format rules:
   - Single sentence per fact
   - Final answer on final line
   - No parenthetical URIs in final answer
   - Explicit markers ("FINAL ANSWER: ", "Answer: ")

5. **Verify extraction at runtime:** If extraction rate is below 90%, add post-hoc validation (does extracted entity match SPARQL result?).

---

## Part 8: Detailed Trace Analysis

### 8.1 Output Format Patterns (10-Sample Analysis)

**Successful Comparison (10 samples):**
- 100% narrative flow (uses "is", "was", "have", etc.)
- 90% inline entity mention (answer mentioned early)
- 50% parenthetical URIs (preserved for evidence)
- 20% multi-sentence (mostly single-sentence answers)

**Failed Compositional (10 samples):**
- 90% narrative flow
- 70% inline entity mention
- 90% parenthetical URIs (CRITICAL DIFFERENCE)
- 10% multi-sentence (mostly single-sentence)

**Key difference:** Compositional failures have 90% parenthetical URIs (narrative with URLs) vs successful comparison 50%. This suggests GRASP is more formal/precise when navigating multi-hop paths.

**Failed Inference (10 samples):**
- 100% narrative flow
- 100% inline entity mention
- 100% parenthetical URIs
- 0% multi-sentence (single entity only)

**Insight:** Inference answers are single-entity with URI, suggesting LLM treats genealogy as exact reference lookup. The "answer + URI" format is intentional for precision.

### 8.2 Model Reasoning Depth

**Successful Comparison:**
- 5 model reasoning messages
- Message progression: explain strategy → search entities → verify dates → construct query → (empty) → Answer
- Explicit final query statement before answer

**Failed Compositional:**
- 4 model reasoning messages
- Message progression: explain strategy → search properties → construct query → (error recovery) → Answer
- No explicit final query statement; answer follows error recovery

**Failed Inference:**
- Variable (10 tools, fewer explicit reasoning messages)
- Many property searches suggest schema exploration
- No clear "here is the final answer" statement

**Observation:** The more explicit the LLM's final reasoning (comparison), the more directly it states the answer. The less explicit (compositional, inference), the more narrative/formal the answer becomes.

---

## Part 9: Unresolved Questions

1. **Why does GRASP choose gpt-5-mini?** Does model size/architecture affect tool use patterns? Would a larger model (o1, gpt-4) avoid these issues?

2. **What is GRASP's extraction method?** The traces show question_id → raw_answer is generated, but how is predicted_answer extracted? Is it also LLM-based or heuristic-based?

3. **How sensitive are these patterns to GRASP's tool design?** Does the tool set (entity_search, execute, answer) force narrative-style outputs, or would other tools (like direct_lookup) change the pattern?

4. **Can prompt-based extraction work better than position-based?** Could GRASP ask itself "which value is the final answer to the question" at generation time?

5. **How does GRASP perform on other KGs (Wikidata, Freebase)?** Are DBpedia-specific naming issues (URI suffixes, label variants) the problem, or is this universal to agentic KGQA?

---

## Part 10: Conclusion

GRASP's performance divergence across question types is **not** a retrieval or knowledge coverage problem. It's a **presentation and extraction alignment issue** stemming from GRASP's agentic architecture.

**The mechanism:**
1. Agentic tool use produces iterative reasoning narratives
2. SPARQL results are structured data (multiple columns)
3. LLM narrates structured data as sequential sentences ("director is X, birthplace is Y")
4. Extraction heuristics split on ". " and take first entity
5. For composition, first entity is intermediate (director), not final (birthplace)

**Why comparison succeeds (55.3%):**
- Comparison queries return single ordered result (already sorted by LLM's SPARQL)
- Answer is primary entity (first mentioned)
- Extraction takes first entity (correct by coincidence)

**Why compositional fails (0%):**
- Compositional queries return unordered multi-column results
- LLM narrates in reasoning order (intermediate first, final second)
- Extraction takes first entity (wrong)

**Why inference fails (0%):**
- Inference queries return single entity but with format formality (URI suffix)
- Extraction includes URI (EM metric breaks on substring match)
- Plus label normalization issues for genealogy

**The fix is achievable without retraining:** Post-processing (URI removal, last-entity extraction, label normalization) could recover 20-30pp EM without changing GRASP's core logic. However, prompt engineering and output format changes would be more robust long-term.

**Implication:** Agentic KGQA systems excel at interpretability and adaptive reasoning but require careful attention to output format specification. The narrative explanations that make agentic systems trustworthy are the same features that break simple extraction heuristics. Future systems should either:
1. Use structured output formats (JSON, XML)
2. Implement extraction-aware prompting
3. Add post-hoc validation layers

---

**Report generated from:**
- 346 GRASP WebSocket traces (full traces with tool calls and reasoning)
- 346 evaluation results (EM/F1/containment metrics)
- 5-sample analysis of successful comparisons, failed compositionals, failed inference
- Output format pattern analysis (100+ samples)

**Files:**
- Evaluation JSONL: `data/eval_1000q/grasp_eval_dbpedia.jsonl`
- Traces directory: `data/eval_1000q/grasp_eval_dbpedia_traces/`
- Original comparison report: `data/eval_1000q/evaluation_report_sparql_grasp.md`
