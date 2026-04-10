# Entity Linking Stress Test Catalog

> **Version**: 1.0 (corrected) — 106 cases across 10 categories
> **Date**: 2026-03-25
> **Source files**: `dbpedia/entity_linker.py`, `dbpedia/sparql_utils.py`, `dbpedia/client.py`, `routing.py`
> **Purpose**: Exhaustive edge-case catalog for stress-testing the entity linking pipeline.
> Each case has been verified against the actual implementation.

## Severity Legend

- **P0**: Critical — affects common benchmark patterns, likely causes wrong answers
- **P1**: High — affects real entities/questions, measurable accuracy impact
- **P2**: Medium — affects uncommon but valid cases
- **P3**: Low/Theoretical — unlikely in 2WikiMultiHopQA, included for completeness

---

## 1. Mention Extraction (`_extract_mentions`)

### 1.1 Noun Chunk Stopword Stripping

Code: `entity_linker.py:145-149` — strips `_STRIP_WORDS` from front of noun chunks.

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 1 | All words are stopwords → empty chunk | `"Which of the in or?"` | Empty mention after stripping, silently lost | P3 |
| 2 | Entity name starts with strip word | `"Who is The Who?"` | `"The"` stripped → `"Who"` survives dedup (NOT blocklisted), goes to `bif:contains` as bare `"Who"` → massive false positives | P1 |
| 3 | Entity IS a strip word | `"Who starred in It?"` | Noun chunk path strips `"It"` completely. NER path would preserve it IF spaCy tagged it as WORK_OF_ART, but `en_core_web_sm` won't. Title-after-keyword requires explicit `"film"` keyword. Entity is invisible. | P1 |
| 4 | `"The"` + blocklisted word | `"When did The Band form?"` | `"The"` stripped → `"Band"` → `"band"` in `_MENTION_BLOCKLIST` → filtered out entirely | P1 |
| 5 | Entity starts with `"A"` | `"Who directed A Clockwork Orange?"` | `"A"` stripped → `"Clockwork Orange"` (partial — missing article) | P1 |

### 1.2 Possessive Splitting

Code: `entity_linker.py:154` — regex `r"^(.+?)[''\u2019]s\b"`

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 6 | Standard `'s` | `"What is Shakespeare's birthplace?"` | Covered — splits correctly → possessor `"Shakespeare"` | P3 |
| 7 | Curly apostrophe `\u2019s` | `"What is Shakespeare\u2019s birthplace?"` | `\u2019` in regex character class `[''\u2019]` → works | P3 |
| 8 | Double possessive | `"What is James's wife's name?"` | First `'s` splits → possessor `"James"`. Second `'s` on `"wife"` only fires if spaCy produces `"wife's name"` as separate chunk. `"wife"` not blocklisted → becomes spurious mention | P2 |
| 9 | `'s` that isn't possessive | `"What is McDonald's revenue?"` | Splits `"McDonald's"` → possessor `"McDonald"`. But `"McDonald's"` is the entity, not `"McDonald"` — wrong URI resolution | P1 |
| 10 | Left single quote `\u2018s` | `"Beyoncé\u2018s album"` | `\u2018` (LEFT single quotation mark) is NOT in regex character class `[''\u2019]` → no possessive split → full chunk searched including `\u2018s` noise | P2 |
| 11 | `s'` (plural possessive) | `"The Beatles' first album"` | Regex matches `'s` only, not `s'` → no possessor split. Full chunk `"Beatles' first album"` searched as one phrase → `bif:contains` gets noise words `"first"`, `"album"` | P2 |
| 12 | Possessor is 1 char | `"What is I's meaning?"` | `len(possessor) > 1` check (line 157) → filtered | P3 |

### 1.3 NER Entity Processing

Code: `entity_linker.py:165-183` — splits on `" or "`, strips trailing `"or"`.

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 13 | Entity spans `or` | `"The Singing Fool or The Jazz Singer"` as one NER span | Split on ` or ` → two parts (desired behavior for comparisons) | P3 |
| 14 | `or` inside entity name | `"Simon or Garfunkel"` (hypothetical) | Incorrectly splits a single entity into two fragments | P2 |
| 15 | Entity name ends with `"or"` substring | `"Who wrote The Aviator?"` | `ent_words[-1].lower() == "or"` checks last word → `"Aviator"` ≠ `"or"` → not stripped ✓ | P3 |
| 16 | NER span is `"Thor or"` | spaCy grabs `"Thor or"` as span | Trailing `"or"` stripped by while loop → `"Thor"` ✓ | P3 |
| 17 | Multiple trailing `"or"` | `"X or or"` | While loop strips both → `"X"` ✓ | P3 |
| 18 | Entity is just `"or"` | Degenerate NER output | Stripped to empty → `if part:` filters it out ✓ | P3 |
| 19 | NER type = CARDINAL | `"In 1984, who..."` where `"1984"` → CARDINAL | Filtered out by label check (line 166) — numeric year lost. May affect year-based comparison questions | P2 |
| 20 | NER type = ORDINAL | `"first"` in comparison questions | Filtered out — correct behavior | P3 |
| 21 | spaCy produces zero NER entities | `"Who directed The Departed?"` | `"The Departed"` invisible to NER — depends on noun chunks or title-after-keyword extraction | P2 |

### 1.4 Title-After-Keyword Extraction

Code: `entity_linker.py:189-221` — follows capitalized words after `_TITLE_KEYWORDS`.

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 22 | Keyword followed by uppercase | `"song Aren'T You Kind Of Glad We Did?"` | Works — extracts full title via capitalization chain | P3 |
| 23 | Keyword followed by lowercase | `"film was released first"` | `first_char.isupper()` fails → skipped (correct) | P3 |
| 24 | Keyword at end of question | `"What is this film"` | `i + 1 < len(words)` fails → skipped | P3 |
| 25 | Connector at end of title | `"film Lord of"` → `"of"` has no next word | Lookahead fails → connector dropped → `"Lord"` only | P1 |
| 26 | **Connector chain (lowercase sequence)** | `"book Lord of the Rings"` | `"of"` is connector, peek at `"the"` → lowercase → break. Only extracts `"Lord"`. Affects all multi-word titles with lowercase connector chains: "Wizard of Oz", "War of the Worlds", "Bridge on the River Kwai", "Lord of the Flies" | **P0** |
| 27 | All-uppercase title | `"film THE MATRIX"` | `clean[0].isupper()` → True for both → extracts `"THE MATRIX"` ✓ | P3 |
| 28 | Keyword IS the entity name | `"Who directed The Film?"` (where `"Film"` is entity) | `"Film?"` at last position → `w.lower().rstrip(".,?!:;")` = `"film"` matches keyword, but `i + 1 < len(words)` fails → skipped. Keyword consumed, entity missed | P2 |
| 29 | Two keywords, two titles | `"Is the song X or the film Y?"` | Both `"song"` and `"film"` fire → two title mentions extracted | P3 |
| 30 | Title starts with curly quote | `"\u201CInception\u201D"` | `lstrip("'\"\u2018\u201C")` handles left curly quote | P3 |
| 31 | Title is a single word | `"film Inception?"` | `len(title_words) >= 1` → `"Inception"` extracted ✓ | P3 |
| 32 | Keyword with trailing punctuation | `"Which song, Inception or..."` | `w.lower().rstrip(".,?!:;")` strips comma → `"song"` matches ✓ | P3 |

### 1.5 Deduplication & Filtering

Code: `entity_linker.py:223-245` — case-insensitive dedup, blocklist, substring removal.

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 33 | Case-insensitive dedup | NER: `"New York"`, noun chunk: `"new york"` | Second is deduplicated via `key = m.lower().strip()` ✓ | P3 |
| 34 | Entity name = blocklisted word | `"Who is the director?"` | `"director"` in `_MENTION_BLOCKLIST` → filtered out ✓ | P3 |
| 35 | Entity name = `"Prince"` (musician) | `"When was Prince born?"` | NOT in blocklist → passes through ✓ | P3 |
| 36 | Mention is len 1 | `"What is I?"` | `len(key) > 1` → filtered | P3 |
| 37 | Substring removal | `"York"` vs `"New York"` | `"York"` removed because it's substring of `"New York"` ✓ | P3 |
| 38 | Possessor-derived survives substring removal | `"Dr. Dre"` is substring of `"Dr. Dre's real name"` | `possessor_derived` set protects it (line 239-240) ✓ | P3 |
| 39 | Mutual non-substrings | `"Angel"` and `"The Badman"` | Neither is substring of other → both kept (may be wrong entities but dedup can't know) | P3 |

### 1.6 Quoted Entities

Code: `entity_linker.py:142` — `re.findall(r'"([^"]+)"', question)`

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 40 | Standard double quotes | `'Who wrote "Hamlet"?'` | Captured by `re.findall(r'"([^"]+)"')` ✓ | P3 |
| 41 | Single quotes | `"Who wrote 'Hamlet'?"` | NOT captured — regex only matches `"..."` | P2 |
| 42 | Curly/smart double quotes | `"Who wrote \u201CHamlet\u201D?"` | NOT captured — `"` ≠ `\u201C`/`\u201D`. If benchmark data has curly quotes, all quoted entities missed | P1 |
| 43 | Nested quotes | `'He said "don\'t say \\"hello\\""'` | Regex grabs first `[^"]+` match — may be partial | P3 |
| 44 | Empty quotes | `'Who wrote ""?'` | `[^"]+` requires 1+ chars → no match ✓ | P3 |
| 45 | Unbalanced quotes | `'Who wrote "Hamlet?'` | No closing `"` → no match | P2 |

---

## 2. Type Hinting (`_extract_type_hint`)

Code: `entity_linker.py:249-266` — checks `RELATION_TYPE_HINTS` first, then `TYPE_HINTS`.

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 46 | Relation hint priority | `"Who is the director of Inception?"` | `\bdirector\s+of\b` → `dbo:Film` (relation hint wins over `TYPE_HINTS \bdirector\b`) ✓ | P3 |
| 47 | No type keyword at all | `"Where was Obama born?"` | Returns `None` — no type validation applied | P2 |
| 48 | Multiple type keywords | `"Which film and book share a name?"` | `\bfilm\b` matches first → `dbo:Film` (book hint lost) | P2 |
| 49 | `directed` triggers Film hint | `"Who directed traffic?"` (non-film use) | False positive — `\bdirected\b` → `dbo:Film` type hint for a non-film question | P2 |
| 50 | **Missing type in `_TYPE_SUFFIX_MAP`** | Type hint `dbo:Person`, `dbo:MusicalArtist`, `dbo:City` | `_TYPE_SUFFIX_MAP` (line 91-97) lacks these 3 types. `_validate_entity_types` returns early (line 303-305) — no type correction applied even when Spotlight links wrong entity | **P0** |
| 51 | `\bperformed\b` → MusicalWork | `"She performed surgery"` | False positive — non-music `"performed"` triggers `dbo:MusicalWork` hint | P2 |

---

## 3. Type Validation (`_validate_entity_types`)

Code: `entity_linker.py:268-368` — validates Spotlight entities against expected type.

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 52 | Spotlight links person, expected Film | `"Who directed Lincoln?"` → Spotlight links Abraham Lincoln | Type mismatch → probe `dbr:Lincoln_(film)` → correct replacement ✓ | P3 |
| 53 | URI probe fails, `bif:contains` finds it | Entity with no `_(suffix)` URI pattern | Falls through probe → `search_entity_by_name` with type constraint | P3 |
| 54 | Both probe and search fail | Very obscure entity | Keeps original wrong entity — silent failure | P2 |
| 55 | `obj_phrase` extraction: `"X of Y"` form | `"the director of The Matrix"` | `obj_match.group(1)` = `"The Matrix"` ✓ | P3 |
| 56 | `obj_phrase` extraction: verb form | `"directed The Matrix"` | Verb form regex fires → extracts `"The Matrix"` | P3 |
| 57 | `obj_phrase` contains year | `"directed the 2010 documentary X"` | Year stripped by `_clean` regex `\b\d{4}\b` | P3 |
| 58 | `obj_phrase` cleaning strips too much | `"wrote The Year 2000"` (title contains `"2000"`) | `_clean` strips determiners (`"The"`) AND year (`"2000"`) → searches for just `"Year"` → wrong entity | P1 |
| 59 | Entity label partial match | `label.lower() in obj_lower` but wrong entity | False positive substring match triggers unnecessary validation | P2 |
| 60 | No relation word in question | `"When was Inception released?"` | No `obj_match` → returns entities unchanged | P3 |

---

## 4. Unlinked Detection (`_find_unlinked`)

Code: `entity_linker.py:370-407` — Jaccard similarity + substring containment.

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 61 | Jaccard below threshold | mention=`"A B"`, surface=`"A C"` → Jaccard = 1/3 < 0.5 | Not matched → treated as unlinked | P3 |
| 62 | Jaccard at boundary | mention=`"A B C"`, surface=`"A B D"` → 2/4 = 0.5 | Matched (≥ 0.5) ✓ | P3 |
| 63 | Mention substring of surface form | mention=`"York"`, surface=`"New York"` | `mention_in_sf = True` → matched ✓ | P3 |
| 64 | Surface form substring of mention | mention=`"New York City"`, surface=`"New York"` | `sf_in_mention = True` blocks Jaccard match → treated as unlinked. This is **correct disambiguation behavior**: triggers search for the more specific `"New York City"` entity | P3 |
| 65 | Word order differs | mention=`"York New"`, surface=`"New York"` | Sets → same words → Jaccard = 1.0 → matched | P3 |
| 66 | Empty entities list | Any mentions | All mentions → unlinked (triggers `bif:contains` for each) | P3 |
| 67 | Mention with extra punctuation | mention=`"Lincoln."`, surface=`"Lincoln"` | `m_lower="lincoln."` vs `sf_lower="lincoln"` — not substring, Jaccard: `{"lincoln."}` ∩ `{"lincoln"}` = ∅ → 0 → unlinked (false positive from trailing period) | P2 |

---

## 5. `bif:contains` Search (`search_entity_by_name` → `build_bif_expression`)

Code: `sparql_utils.py:25-46`, `client.py:455-494`

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 68 | All stopwords stripped | `"The One"` → only `"One"` | Thin search term — many false positives in `bif:contains` | P2 |
| 69 | Entity name is pure stopwords | `"A"` (film) | `len(w) <= 1` → filtered → `build_bif_expression` returns `None` → no search | P1 |
| 70 | Apostrophe in search | `"O'Brien"` | Doubled to `O''Brien` in `bif:contains` by `escape_bif_term` ✓ | P3 |
| 71 | Hyphen preserved | `"Spider-Man"` | Single hyphenated term, not split → `'Spider-Man'` ✓ | P3 |
| 72 | Short entity name | `"It"` (film) | `len("It")` = 2 > 1 → passes filter, but very ambiguous `bif:contains` search | P1 |
| 73 | Disambiguation: exact label match | `"Paris"` → exact match finds `dbr:Paris` | Returns city, not the film `"Paris"` — no type disambiguation at exact match step | P2 |
| 74 | Disambiguation: shortest URI fallback | Multiple candidates, no exact/type match | `min(candidates, key=lambda e: len(e.uri))` heuristic — can pick wrong entity | P2 |
| 75 | Type-constrained search fails, retry without type | Entity type given but no type-matching results | Falls back to untyped search (client.py:472-473) — may return wrong-type entity | P2 |

---

## 6. SPARQL Escaping (Injection Surface)

Code: `sparql_utils.py:49-74`

**Confirmed working (positive tests)**:

| # | Case | Input | Behavior | Status |
|---|------|-------|----------|--------|
| 76 | Double quote in entity name | `'Say "Hello"'` | `escape_sparql_string` → `Say \"Hello\"` | ✅ Correct |
| 77 | Backslash in entity name | `"C:\\Windows"` | Escaped to `C:\\\\Windows` | ✅ Correct |
| 78 | Newline injection attempt | `"Inception\n} INSERT DATA {..."` | `\n` → `\\n` in string, `%0A` in URI | ✅ Correct |
| 79 | Angle brackets in URI | `"http://example.org/<evil>"` | `<` → `%3C`, `>` → `%3E` | ✅ Correct |

**Actual risks**:

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 80 | **Apostrophe in URI** | `"dbr:Aren't_You_..."` | NOT escaped by `escape_sparql_uri` → raw `'` in `<uri>` — potential SPARQL syntax error. DBpedia uses `%27` but our escaper doesn't convert `'` → `%27` | P1 |
| 81 | **Question mark in URI** | `"dbr:...Did?"` | NOT escaped → `?` in `<...>` looks like SPARQL variable — potential breakage. DBpedia uses `%3F` but our escaper doesn't handle it | P1 |
| 82 | Percent-encoded URI passthrough | `"dbr:Aren%27t_..."` | Already encoded → passes through cleanly ✓ | P3 |

---

## 7. Comparison Question Patterns

Code: `routing.py:208-234` — `_has_incomplete_entity_coverage()`

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 83 | Standard `"X or Y?"` | `"..., Inception or The Matrix?"` | Regex `r',\s*(.+?)\s+or\s+(.+?)\s*\?'` captures both ✓ | P3 |
| 84 | `"both X and Y"` | `"Are both X and Y directed by..."` | Third regex `r'\bboth\s+(.+?)\s+and\s+(.+?)[\s?]'` fires ✓ | P3 |
| 85 | Missing comma | `"Which is older X or Y?"` | Regex requires `,\s*` before entity names → no match → coverage check silently passes | P2 |
| 86 | Three-way comparison | `"X, Y, or Z?"` | Regex captures group(1) = `"Y"`, group(2) = `"Z"` only → X (before comma) is never checked | P2 |
| 87 | Entity name contains ` or ` (with spaces) | `"..., Truth or Dare or The Matrix?"` | Regex `\s+or\s+` splits at first word-boundary `or` → group(1) = `"Truth"`, group(2) = `"Dare or The Matrix"`. Entity `"Truth or Dare"` is split. Note: `"or"` inside words (e.g., `"Aviator"`) is NOT affected because `\s+` requires whitespace | P2 |
| 88 | Entity name contains ` and ` (with spaces) | `"both Rock and Roll Hall of Fame and Madison Square Garden"` | Non-greedy `(.+?)` splits at first ` and ` → group(1) = `"Rock"`, group(2) = `"Roll"` (stops at first `[\s?]`). Both fragments are useless for entity matching | P2 |

---

## 8. spaCy NER Fundamental Limitations

These are upstream `en_core_web_sm` limitations — cannot be fixed in our code, only mitigated by Spotlight and keyword extraction fallbacks.

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 89 | Film title = person name | `"Who directed Lincoln?"` | spaCy tags PERSON, Spotlight may link to Abraham Lincoln | P2 |
| 90 | Film title = common noun | `"Who directed Drive?"` | spaCy misses entirely — not recognized as entity | P2 |
| 91 | Film title = location | `"Who directed Paris, Texas?"` | spaCy tags GPE `"Paris"` + GPE `"Texas"` → title split into two locations | P2 |
| 92 | Entity invisible to NER | `"The Departed"` in comparison Q | spaCy produces zero NER entities for it | P2 |
| 93 | Entity split across NER spans | `"Angel And The Badman"` → PERSON + WORK_OF_ART | Irrecoverable from NER output — two fragments | P2 |
| 94 | GPE absorbs part of title | `"New York, New York"` (film) | spaCy sees GPE `"New York"` only — film title lost | P2 |
| 95 | Short common-word entity | `"Her"`, `"Us"`, `"Up"` | Too short (filtered by `len > 1` for some), or spaCy misses, or in `_STRIP_WORDS` | P1 |
| 96 | Entity with non-English words | `"Crouching Tiger, Hidden Dragon"` | spaCy may partially recognize — depends on model training data | P2 |

---

## 9. `_sparql_fallback` Edge Cases

Code: `entity_linker.py:514-550` — last-resort capitalized-phrase extraction.

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 97 | Capital after sentence end | `"X? Who directed Y?"` | `words[i-1][-1] in ".!?"` → capital `"Who"` excluded (correct) | P3 |
| 98 | All lowercase question | `"who directed inception?"` | No capitals → nothing found | P2 |
| 99 | Non-ASCII capital | `"Where is Zürich?"` | `word[0].isupper()` → True for `"Zürich"` ✓ | P3 |
| 100 | Single word entity | `"Inception"` | Capital phrase = `["Inception"]` → searched via `get_entity_by_label` | P3 |

---

## 10. Cache Behavior

Code: `entity_linker.py:445-511`

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 101 | Cache key is exact text | `"Who directed Inception?"` vs `"who directed inception?"` | Different cache keys → separate API lookups, no dedup | P3 |
| 102 | Empty result not cached | Spotlight + spaCy both fail → `entities = []` | `if use_cache and entities:` (line 498) → nothing cached → re-fetches every call for hard questions | P3 |
| 103 | Stale cache after Spotlight improvement | Old cached result with wrong entity | Returns wrong entity forever until cache cleared (in-memory only, clears on restart) | P3 |

---

## 11. Additional Cases (from expert review)

### 11.1 Unicode Normalization

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 104 | NFC vs NFD normalization | `"Beyoncé"` (NFC: é = U+00E9) vs `"Beyoncé"` (NFD: e + U+0301) | Python `str.lower()`, `set()`, and `==` treat NFC/NFD as different strings. If question text and Spotlight surface forms use different normalization → Jaccard fails, dedup misses, entity treated as unlinked even when present | P1 |

### 11.2 Curly Quote Regex Gap

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 105 | Curly double quotes in `_extract_mentions` | `"Who directed \u201CInception\u201D?"` | `re.findall(r'"([^"]+)"')` uses ASCII `"` only. Curly `\u201C`/`\u201D` (common in formatted/copy-pasted text) are invisible to regex → quoted entity missed entirely. Extends #42 to the extraction code path | P1 |

### 11.3 Connector/NER Interaction

| # | Case | Input | Risk | Severity |
|---|------|-------|------|----------|
| 106 | Title-after-keyword vs NER `or`-splitting conflict | `"song Truth or Dare?"` (hypothetical entity with `or` in name) | Title-after-keyword: `"or"` in `_TITLE_CONNECTORS` → peek at `"Dare"` → uppercase → includes connector → extracts `"Truth or Dare"` (correct). Simultaneously, NER path splits on `" or "` → `["Truth", "Dare"]` (incorrect). Dedup can't reconcile: 3 mentions instead of 1, causing spurious `bif:contains` searches | P2 |

---

## Summary by Severity

| Severity | Count | Key Issues |
|----------|-------|------------|
| **P0** | 2 | #26 (connector chain truncation), #50 (missing `_TYPE_SUFFIX_MAP` entries) |
| **P1** | 14 | #2, #3, #4, #5, #9, #42, #58, #69, #72, #80, #81, #95, #104, #105 |
| **P2** | 27 | #8, #10, #11, #14, #19, #21, #28, #41, #45, #47, #48, #49, #51, #54, #59, #67, #68, #73, #74, #75, #85, #86, #87, #88, #89-96 (counted individually), #98, #106 |
| **P3** | 63 | All remaining — confirmed working or theoretical-only |

## Top 6 Issues for Thesis Discussion

1. **#26 — Connector chain truncation** (P0): `"Lord of the Rings"` → only `"Lord"` extracted. Affects all titles with lowercase connector sequences. Fix: multi-word lookahead or whitelist known patterns.

2. **#50 — Missing `_TYPE_SUFFIX_MAP` entries** (P0): `dbo:Person`, `dbo:MusicalArtist`, `dbo:City` have type hints but no suffix mapping → type validation silently skipped. Fix: add entries `'dbo:Person': 'person'`, `'dbo:MusicalArtist': 'musician'`, `'dbo:City': 'city'`.

3. **#80-81 — URI escaping gaps** (P1): Apostrophe `'` and question mark `?` not percent-encoded in `escape_sparql_uri` → SPARQL syntax errors for entities like "Aren't You Kind of Glad We Did?". Fix: add `'` → `%27` and `?` → `%3F` to escaper.

4. **#1-5 — `_STRIP_WORDS` destroying entity names** (P1): "The Who", "It", "The Band", "A Clockwork Orange" all damaged by front-stripping. Fix: check against known entity patterns before stripping, or only strip when resulting text is ≥ 2 words.

5. **#42, #105 — Curly quote blindness** (P1): Both `_extract_mentions` and the quoted-entity regex only handle ASCII double quotes. Fix: extend regex to include `\u201C`/`\u201D` and `\u2018`/`\u2019`.

6. **#89-96 — spaCy NER blind spots** (P2): Fundamental `en_core_web_sm` limitations — can only be mitigated, not fixed. Current mitigations (Spotlight, title-after-keyword, `bif:contains` fallback) cover most cases but gaps remain for common-noun and location-named titles.
