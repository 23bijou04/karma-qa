# Judge Protocol

Source: `scripts/judge_hybrid_eval.py`.

## Configuration

Default settings:

| Setting | Value |
|---|---|
| Provider | `ollama` |
| Model | `gemma3:12b` |
| Temperature | `0` |
| Max output tokens | `100` |
| Output path | input path with `_judged.jsonl` suffix |


## Input and Output

For each JSONL record, the judge reads:

```text
question_id
question
gold_answer
predicted_answer
gold_aliases, if present
```

It writes the original record plus:

```text
judge_correct: true | false
```

## Protocol

1. Empty predictions are rejected immediately.
2. If `predicted_answer` exactly matches `gold_answer` or any `gold_aliases`
   after normalization, the record is accepted without an LLM call. Aliases are
   used only for this deterministic precheck; they are not included in the LLM
   prompt.
3. A conservative containment precheck accepts the prediction when all normalized
   gold words occur in the normalized prediction and the gold answer has at
   least two words.
4. Remaining cases are sent to the LLM judge with the system and user prompt
   below.
5. The response parser searches for the last `VERDICT:` marker and returns
   `true` only when the first word after it is `yes`. If the marker is absent,
   the parser falls back to the first word of the last non-empty line.
6. Empty, malformed, or unparseable judge responses count as `false`.
7. If the judge call raises an exception, the record is also marked `false`.

Normalization lowercases text, removes punctuation, removes the articles
`a`, `an`, and `the`, and collapses whitespace.

## System Prompt

```text
You are a strict evaluation judge for a QA benchmark. Determine if two answers refer to the exact same entity or fact. Precision matters more than recall — when in doubt, answer no.
```

## User Prompt

```text
Determine if the predicted answer is semantically equivalent to the gold answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Matching rules (answer YES):
- Same entity/fact with different surface forms (e.g. "July 16, 2013" vs "2013-07-16")
- Gold contained in a more specific predicted answer (e.g. "Munich" vs "Nymphenburg Palace, Munich")
- Added title/honorific/disambiguator (e.g. "Henry III" vs "Henry III, Holy Roman Emperor")
- Full formal name of same person (e.g. "Giovanni Mario" vs "Giovanni Mario de Candia")
- Administrative region containing the exact gold location (e.g. gold "Theni" vs pred "Theni district")
- Unambiguous well-known city with or without country (e.g. gold "Florence, Italy" vs pred "Florence")
- Nationality adjective matching country name (e.g. gold "Italy" vs pred "Italian")

Rejection rules (answer NO):
- Wrong, empty, or completely different entity
- Predicted is broader/vaguer than gold (e.g. gold "Taipei Medical University Hospital" vs pred "Taipei")
- Year only when gold has full date (e.g. gold "December 25, 1757" vs pred "1757")
- Related but distinct entities (e.g. gold "Hollywood" vs pred "Los Angeles")
- Same first name but different person (e.g. gold "Paris" vs pred "Paris Hilton")
- Predicted lists multiple answers and gold is NOT clearly among them
- Completely different location even if in same region (e.g. gold "Theni" vs pred "Mallingapuram")

First, explain in ONE sentence why the answers match or don't.
Then write your verdict on a new line: VERDICT: yes or VERDICT: no
```

