# KARMA Runtime System Prompts

This file collects the active LLM prompt templates used by the reported KARMA
runtime. These are current runtime templates, not historical prompt snapshots
from earlier evaluation traces. The evaluation-only judge protocol is kept
separately in `docs/judge-prompt.md`.

## Source map

All prompts below are sent through `llm_provider.call_llm()` as user messages
unless noted otherwise.

| ID | Source | Runtime use |
|---|---|---|
| P1 | `dbpedia/entity_linker.py`, `DBpediaEntityLinker._llm_rerank_candidates()` | Builds the candidate-disambiguation f-string after fallback entity search returns multiple DBpedia candidates. |
| P2 | `routing.py`, `PROPERTY_EXTRACTION_PROMPT` | Used by `_extract_properties_llm()` after deterministic property matching is inconclusive. |
| P3 | `logicrag_decomposition.py`, `DECOMPOSE_PROMPT` | Used by `_decompose_subquestions()` to produce KARMA sub-questions and dependencies. |
| P4 | `logicrag_prompts.py`, `DEPENDENCY_RAG_PROMPT` | Used by `logicrag_decomposition._check_can_answer()` for early stopping during decomposition. |
| P5 | `sparql_solver.py`, `UNIFIED_SPARQL_PROMPT` | Used by `sparql_solver.solve()` for direct, sub-question, and pre-check SPARQL generation. |
| P6 | `sparql_evaluation.py`, `SPARQL_RETRY_PROMPT` | Imported by `sparql_solver.solve()` and used once after SPARQL errors or empty results. |
| P7 | `sparql_evaluation.py`, `ANSWER_PROMPT` | Imported by `sparql_solver.solve()` to verbalize direct SPARQL result rows; this is result extraction/verbalization, not KARMA synthesis. |
| P8 | `text_solver.py`, `TEXT_REASON_PROMPT` | Used by `text_solver.solve()` for whole-question text fallback. |
| P9 | `text_solver.py`, `TEXT_SUBQ_PROMPT` | Used by `text_solver.solve()` for decomposed sub-question text fallback. |
| P10 | `logicrag_prompts.py`, `get_final_answer_prompt()` | KARMA final synthesis prompt: used by `logicrag_decomposition._generate_final_answer()` to synthesize one answer from resolved sub-question facts; comparison instructions are conditionally inserted. |

## P1. Entity candidate disambiguation

Used only when fallback entity search returns multiple DBpedia candidates for an
unlinked mention.

```text
Given this question: "{question}"
Which entity best matches the mention "{mention}"?

{candidate_list}

Reply with ONLY the number (1-{num_candidates}) of the best match, or "none" if no candidate fits.
```

## P2. Schema-constrained property extraction

Used only when deterministic keyword/schema matching is inconclusive. Candidate
properties are passed back to deterministic routing and ASK checks.

```text
Given this question and DBpedia schema, which properties are needed to answer it?

{schema}

QUESTION: {question}

Output a comma-separated list of property names (no dbo:/dbp: prefix).
Only output properties listed above. For 2-hop paths (e.g., "director's birthdate"),
list both: director, birthDate
If no properties can help, output: NONE

Properties:
```

## P3. Decomposition prompt

Used whenever the orchestrator attempts KARMA decomposition after routing
and any direct SPARQL attempt. This includes explicit `DECOMPOSE` routes,
failed direct-SPARQL routes, and `TEXT` routes before final whole-question text
fallback.

```text
Decompose this multi-hop question into independent sub-questions.

QUESTION: {question}
ENTITIES: {entity_names}
SCHEMA SUMMARY: {schema_summary}
{type_hint}
Rules:
- Each sub-question should ask for ONE fact
- If a sub-question needs the answer from another, mark the dependency
- For comparison questions: extract the same property for each entity (parallel, no dependencies)
- For bridge questions: first find the intermediate entity, then ask about its property
- If the question is simple (single fact lookup), return exactly 1 sub-question with no dependencies

Return JSON:
{
  "sub_questions": ["sub-Q1 text", "sub-Q2 text", ...],
  "dependencies": [[dependent_idx, dependency_idx], ...]
}

Example - Bridge: "When was the director of Inception born?"
{
  "sub_questions": ["Who directed Inception?", "When was [Q0] born?"],
  "dependencies": [[1, 0]]
}

Example - Comparison: "Who was born first, the director of X or Y?"
{
  "sub_questions": ["Who directed X?", "Who directed Y?",
    "When was [Q0] born?", "When was [Q1] born?"],
  "dependencies": [[2, 0], [3, 1]]
}

Example - Simple: "Who directed Inception?"
{
  "sub_questions": ["Who directed Inception?"],
  "dependencies": []
}

Example - Nested relative clause (3-hop): "When was the region north of where X is located created?"
{
  "sub_questions": ["Where is X located?", "What region is immediately north of [Q0]?", "When was [Q1] created?"],
  "dependencies": [[1, 0], [2, 1]]
}

Example - Passive voice: "Who is played by the director of Film X in Film Y?"
{
  "sub_questions": ["Who directed Film X?", "What role did [Q0] play in Film Y?"],
  "dependencies": [[1, 0]]
}

Example - Possessive chain: "What is the capital of the country where X's creator was born?"
{
  "sub_questions": ["Who created X?", "In which country was [Q0] born?", "What is the capital of [Q1]?"],
  "dependencies": [[1, 0], [2, 1]]
}

Additional rules for complex questions:
- Preserve active/passive voice direction: "played BY X" -> ask "What did [Q0] play?" NOT "Who played [Q0]?"
- For possessive chains (X's Y's Z), decompose from innermost entity outward
- Each sub-question MUST retain qualifying context (where/who/that clauses) from the original

JSON only, no explanation:
```

## P4. Early stopping prompt

Used during decomposition to decide whether accumulated facts already answer the
original question.

```text
We pre-parsed the question into a list of dependencies, and the dependencies are sorted in a topological order, below is the question, the information summary, and the decomposed dependencies:

Question: {question}

Available Information:
{info_summary}

Decomposed dependencies:
{dependencies}

Current dependency to be answered:
{current_dependency}

Please analyze the question and the information summary, and the decomposed dependencies, and answer the following questions:
Please analyze:
1. Can the question be answered completely with this information? (Yes/No)
2. Summarize our current understanding based on available information.

Please format your response as a JSON object with these keys:
- "can_answer": boolean
- "current_understanding": string
```

## P5. SPARQL generation prompt

Used for direct SPARQL, routed sub-question SPARQL, and the direct-property
pre-check path before final whole-question text fallback.

```text
You are a SPARQL expert for DBpedia. Generate a query to answer the question.

{schema}
{known_facts_block}
QUESTION: {question}

RULES:
1. Use the EXACT entity URIs provided in the schema
2. For multi-hop queries, chain triple patterns using dbo: properties (they return URIs, not strings).
   Use the "2-Hop chains" section in the schema to identify valid chains.
   Example: if schema shows dbo:director -> Person: birthDate, write:
   ?film dbo:director ?person . ?person dbo:birthDate ?date .
3. Prefer dbo: properties for multi-hop joins; only use dbp: for single-hop leaf values
4. Always add LIMIT 100
5. For comparison questions (who/which was born first, etc.), return data for BOTH entities so results can be compared
6. If an entity URI contains special characters (apostrophes, commas, parentheses), use full IRI syntax <http://dbpedia.org/resource/...> instead of dbr: prefix
7. If known facts provide entity URIs, use those URIs directly in your query
8. Generate ONLY the SPARQL query, no explanation
```

## P6. SPARQL repair prompt

Used once when a generated SPARQL query errors or returns empty results.

```text
Your previous SPARQL query returned {problem}.

Previous query:
{sparql}

Fix the query using the schema below.

{schema}

QUESTION: {question}

Generate ONLY the corrected SPARQL query:
```

## P7. SPARQL result verbalization prompt

Used to convert direct SPARQL result rows into a concise answer. This is an
answer extraction/verbalization step over one SPARQL result table; it is not
the KARMA final synthesis prompt for decomposed questions.

```text
Given the question and SPARQL query results from DBpedia, provide the answer.
Base your answer ONLY on the SPARQL results shown below, not on your own knowledge.
For comparison questions: if data is missing for one of the entities being compared, say "could not determine" instead of guessing.

Question: {question}
SPARQL Results:
{results}

End your response with FINAL ANSWER: <your answer>
```

## P8. Whole-question text fallback prompt

Used when a whole question is routed to text or symbolic execution cannot
answer.

```text
Answer this question using the sources below.

Question: {question}

{kg_facts}Wikipedia text:
{context}

Instructions:
- Knowledge graph facts (if present) are verified structured data -- trust them
- Use Wikipedia text to fill gaps not covered by KG facts
- Show brief step-by-step reasoning
- End with exactly: FINAL ANSWER: <answer>
```

## P9. Sub-question text fallback prompt

Used when an individual decomposed sub-question is routed to text.

```text
Answer this sub-question using the sources below.

Sub-question: {question}

{kg_facts}Wikipedia text:
{context}

Instructions:
- Knowledge graph facts (if present) are verified structured data -- trust them
- Use Wikipedia text to fill gaps not covered by KG facts
- Give a brief, factual answer
- End with exactly: FINAL ANSWER: <answer>
```

## P10. KARMA final answer synthesis prompt

This is the final synthesis prompt for KARMA's decomposed execution.
After the system resolves sub-questions, `logicrag_decomposition._generate_final_answer()`
passes the accumulated `Information Summary` here to synthesize one final answer
for the original question. This prompt combines resolved sub-question facts and
refuses incomplete chains marked `[UNKNOWN]`; it is distinct from P7, which only
verbalizes direct SPARQL result rows.

```text
You must give ONLY the direct answer in the most concise way possible.
DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

IMPORTANT -- the sub-questions below form a reasoning chain:
- Earlier sub-questions find intermediate entities; later ones use those entities.
- The LAST sub-question's answer is usually what the original question asks for.
- Do NOT return an intermediate entity when a later sub-question has resolved the final answer.
- If a sub-question returns a list, pick the item that satisfies the condition in the original question.
- For yes/no questions with multiple sub-questions, combine ALL sub-answers (both must be true for "yes").
- If any sub-question is marked [UNKNOWN], the chain is incomplete. Do NOT fabricate
  a final answer by skipping the gap -- respond with: FINAL ANSWER: unknown

[Inserted only when is_comparison=True]
When comparing countries or nationalities:
- If answers contain city or state names, identify which country they belong to
- Compare at the country level, not city/state level
- Historical states (East Germany, Ottoman Empire) map to their modern successor

Question: {question}

Information Summary:
{info_summary}

End your response with exactly: FINAL ANSWER: <your answer>
```
