# Entity Linking & Resolution Edge Case Verification

Date: 2026-03-17

Files examined:
- `/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa/dbpedia/entity_linker.py`
- `/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa/logicrag_decomposition.py`
- `/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa/models.py`
- `/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa/dbpedia/client.py`
- `/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa/entity_discovery.py`
- `/Users/e23e889b/Documents/2026/2026_02/dbpedia_qa/hybrid_qa.py`

---

## 7. link_entities() cache data loss

**Verdict: Partial**

**Analysis:**

Cache serialization at `entity_linker.py:386-397` saves only `uri`, `label`, `confidence`, `source`. On cache hit at line 336, `Entity(**e)` reconstructs with defaults: `types=[]`, `abstract=None`, `discovery_relation=None`.

**Impact assessment -- who reads these fields?**

- **`types`**: Only read by `resolve_coreference()` (lines 533-546), which is dead code (see finding #9). No other caller in the pipeline reads `.types` from entities returned by `link_entities()`. The entity profile system (`entity_profile.py`) and schema projection independently query types via SPARQL, not from Entity objects. **No real data loss in practice.**

- **`abstract`**: Only read by `_disambiguate()` (lines 502-503), which is called from `link_entity()` (singular), not `link_entities()` (plural). `link_entities()` itself never populates `abstract` via Spotlight. However, the `search_entity_by_name` fallback CAN populate `.abstract` via its `get_entity_by_label` sub-call (client.py:137). If an entity is found this way, the cache serialization would discard the abstract. In practice, no downstream code reads `.abstract` from `link_entities()` output, so the data loss is invisible.

- **`discovery_relation`**: This field is populated downstream by `entity_discovery.py` and `routing.py`, never by `link_entities()`. So cache serialization correctly omits it.

**Verdict:** The cache serialization matches what `link_entities()` typically populates. `types` and `discovery_relation` are never populated by `link_entities()`. The `abstract` field can technically be populated via the `search_entity_by_name` → `get_entity_by_label` fallback path (client.py:137), so the cache does silently discard it in that case. However, no downstream code reads `.abstract` from `link_entities()` output, so the data loss has no practical impact. The code is safe but brittle -- if someone later adds field-fetching inside `link_entities()`, the cache would silently discard it.

---

## 8. _sparql_fallback() first word always uppercase

**Verdict: Partial**

**Analysis:**

The `_sparql_fallback()` method at lines 401-437 collects capitalized words into phrases. The blocklist at lines 419-421 and 430-432 contains: `{"the", "a", "an", "who", "what", "when", "where", "which", "how"}`.

Missing from the blocklist: `"Did"`, `"Does"`, `"Do"`, `"Are"`, `"Is"`, `"Were"`, `"Was"`, `"Can"`, `"Could"`, `"Would"`, `"Should"`, `"Has"`, `"Have"`, `"Had"`, `"Both"`, `"Either"`.

However, the condition at line 413 is:
```python
if word[0].isupper() and (i == 0 or words[i - 1][-1] not in ".!?"):
```

At `i == 0`, the condition `i == 0` evaluates to True (via `or` short-circuit), so the first word is ALWAYS added to `current_phrase` regardless. Then:
- If word 2 is lowercase (e.g., "directed"), the phrase accumulates only the first word, gets checked against the blocklist.
- "Did" is NOT in the blocklist, so `get_entity_by_label("Did")` would be called.

The saving grace: `get_entity_by_label("Did")` performs an exact SPARQL label match and returns `None` because there is no DBpedia resource labeled exactly "Did". So it is a wasted SPARQL call, not an incorrect entity.

For real benchmark questions like `"Do both films..."`, "Do" would be collected, not blocklisted, but `get_entity_by_label("Do")` returns None. Then "Both" would NOT be collected because it follows "Do" (lowercase check fails -- wait, "both" starts lowercase in "Do both", so it breaks the phrase). Actually looking more carefully:

For `"Are director of film X and director of film Y both from the same country?"`:
- "Are" (i=0, capital) -> phrase starts
- "director" (lowercase) -> phrase "Are" emitted, not in blocklist, SPARQL lookup for "Are"
- "of" (lowercase) -> skip
- "film" (lowercase) -> skip
- "X" (capital) -> new phrase starts

So "Are" gets a wasted SPARQL call. Not a bug (returns None), but an inefficiency.

**Key mitigating factor:** `_sparql_fallback()` is only called at line 382 when ALL other methods (Spotlight, spaCy mentions, bif:contains) have failed. This is a last-resort path that rarely triggers.

**Verdict:** Missing blocklist entries cause wasted SPARQL calls on question-starter words. No incorrect entities are returned because `get_entity_by_label` requires exact label match and DBpedia has no resources for "Are", "Did", "Do", etc. Low practical impact since this is a last-resort fallback.

---

## 9. resolve_coreference() no gender distinction

**Verdict: Handled (dead code)**

**Analysis:**

Grepping the entire codebase for `resolve_coreference` yields exactly one result:

```
dbpedia/entity_linker.py:511:    def resolve_coreference(
```

The method definition itself. It is **never called** anywhere in the pipeline. Not by `hybrid_qa.py`, not by `graphrag_wiki.py`, not by `logicrag_decomposition.py`, not by any test file.

The method is dead code. The gender-agnostic "he"/"she" both resolving to the first Person entity is technically imprecise, but since nothing calls it, there is zero runtime impact.

Additionally, even if it were called, it would fail to work properly because:
1. Entities from `link_entities()` have `types=[]` (Spotlight doesn't populate types).
2. The `entity.types` check at lines 533/539 would always be falsy, falling through to the default `entities[-1]` return at line 550.

**Verdict:** Dead code. No runtime impact. If ever activated, it would need both call sites and proper type population to function.

---

## 10. _resolve_answer_entities() date detection regex

**Verdict: Handled**

**Analysis:**

The regex at line 288:
```python
re.match(r'^[\d,.\-/\s]+$', answer_val.strip())
```

This matches: `"1990"`, `"1990-01-01"`, `"2,500"`, `"01/15/1990"`.

It does NOT match: `"January 1, 1900"`, `"1 January 1900"`, `"March 15th, 2020"`.

For text-format dates like "January 1, 1900":
1. The regex check fails, so execution continues past line 286.
2. `kg_client.search_entity_by_name("January 1, 1900")` is called at line 294.
3. `search_entity_by_name` first tries exact label match via `get_entity_by_label`. There is no DBpedia resource labeled "January 1, 1900".
4. Then tries fulltext search via `search_entities_fulltext`. This might return noise results (e.g., a page about January), but the disambiguation at lines 476-482 prefers exact label matches and shortest URIs.
5. If no match: falls through to `_split_compound_answer` (line 301), which does NOT split (no " and " in dates).
6. `_make_fallback_entity("January 1, 1900")` creates URI: `http://dbpedia.org/resource/January_1%2C_1900`.

**What happens downstream with this guessed URI?** It gets added to `all_entities` in `_resolve_subquestion()` (line 432), used for schema projection and Wikipedia retrieval. A guessed URI for a date is harmless because:
- Schema projection for a non-existent entity returns empty profile.
- Wikipedia retrieval for a non-existent page returns nothing.
- The entity is simply ignored in practice.

**However,** the real question is: should text-format dates even be treated as entities? The intent of the date check is to skip date answers entirely (they are values, not entities). Missing text-format dates means a wasted SPARQL call and a useless guessed entity, but no incorrect behavior.

**Verdict:** The regex misses text-format dates, causing unnecessary SPARQL lookups and harmless fallback entity creation. The downstream pipeline gracefully handles non-existent URIs. Low practical impact because: (a) LLM sub-question answers typically use numeric date formats ("1990" not "January 1, 1990"), and (b) the fallback entity is simply ignored by downstream components.

---

## 11. _find_unlinked() Jaccard threshold

**Verdict: Partial**

**Analysis:**

At `entity_linker.py:288`:
```python
if jaccard >= 0.5 or mention_in_sf:
```

The threshold uses `>=` (greater-than-or-equal), so 0.5 IS matched.

For single-word mention "Einstein" vs surface form "Albert Einstein":
- `m_words = {"einstein"}`, `sf_words = {"albert", "einstein"}`
- intersection = {"einstein"}, union = {"albert", "einstein"}
- jaccard = 1/2 = 0.5
- `0.5 >= 0.5` is True -> considered "linked" -> NOT added to unlinked list

**Is this correct behavior?** It depends:

**Case A -- "Einstein" IS linked to Albert Einstein by Spotlight:** Jaccard correctly identifies the mention as already covered. Good.

**Case B -- "Einstein" is linked by Spotlight to a WRONG Einstein (e.g., "Einstein (film)"):** Jaccard still matches because the surface form overlap is sufficient. The mention is NOT added to the unlinked list, so no SPARQL fallback search runs. The wrong entity persists. This is a subtle false negative.

**Case C -- Mention "Einstein" vs surface form "Einstein's Dreams" (a book):**
- m_words = {"einstein"}, sf_words = {"einstein's", "dreams"}
- intersection = {} (because "einstein" != "einstein's")
- jaccard = 0
- BUT: `mention_in_sf` = "einstein" in "einstein's dreams" = True -> matched

So substring containment catches this case.

**The real risk:** The 0.5 threshold is borderline for single-word mentions. For "Einstein" matching "Albert Einstein", 0.5 is appropriate -- it IS the same entity. The issue would only arise if Spotlight linked "Einstein" to the wrong entity AND that wrong entity's surface form happened to share 50%+ words with the mention. This is a narrow edge case.

**Verdict:** The `>=` operator is intentional and correct for the primary use case. Single-word mentions like "Einstein" matching "Albert Einstein" at exactly 0.5 is reasonable behavior -- they ARE the same entity. The borderline threshold could theoretically mask a mislink, but this is mitigated by the type validation step at lines 354-359 which runs before `_find_unlinked`.

---

## 12. _validate_entity_types() silent exception handling

**Verdict: Partial**

**Analysis:**

At `entity_linker.py:234-257`:

```python
try:
    types = self.sparql_client.get_entity_types(entity.uri)
    type_short = expected_type.replace('dbo:', '')
    if any(type_short in t for t in types):
        validated.append(entity)  # correct type, keep
        continue
except Exception:
    pass  # can't verify, try fallback
```

The flow when BOTH the type check AND the fallback search fail:

1. SPARQL type check raises Exception -> caught, falls through
2. `search_entity_by_name(search_query, entity_type=expected_type)` at line 250-252
3. If fallback returns None: `if fallback and fallback.uri != entity.uri:` at line 253 is False
4. Falls to line 257: `validated.append(entity)  # keep original if no better candidate`

So yes, the original (possibly wrong) entity is kept.

**Is this actually a problem?** The design is defensive -- "keep what we have rather than drop it." Consider the alternatives:
- Drop the entity: worse, because now we have NO entity where we had one. Downstream text retrieval needs URIs for Wikipedia lookup.
- Keep it: the entity might be wrong type but could still be useful for context. The LLM can reason around it.

**When does this scenario trigger?**
1. Spotlight links an entity.
2. Type hint says it should be type X.
3. SPARQL type check fails (network error, timeout).
4. SPARQL fallback search also fails (same network issue).
5. Original entity kept.

This is a transient network failure scenario. The broad `except Exception` is intentional because SPARQL calls can fail for many reasons (timeout, 503, rate limit). The alternative (letting the exception propagate) would crash the entire entity linking pipeline.

**Verdict:** The silent exception handling is a deliberate defensive choice. Keeping the original entity on double failure is the least-bad option -- dropping it would be worse. The broad `except Exception` is appropriate here because SPARQL failures are expected (network issues). However, the `pass` with no logging at line 242 means failures are completely invisible. A `logger.debug` would improve observability without changing behavior.

---

## Summary Table

| # | Edge Case | Verdict | Impact |
|---|-----------|---------|--------|
| 7 | Cache data loss | Partial | Safe today (lost fields were never populated), but brittle for future changes |
| 8 | _sparql_fallback blocklist | Partial | Wasted SPARQL calls, no incorrect entities. Last-resort path, low frequency |
| 9 | resolve_coreference gender | Handled | Dead code, never called anywhere |
| 10 | Date detection regex | Handled | Text-format dates cause harmless fallback entities, gracefully ignored downstream |
| 11 | Jaccard threshold 0.5 | Partial | Borderline matches at exactly 0.5 are correct for the common case. Narrow mislink risk |
| 12 | Silent exception handling | Partial | Defensive design is correct. Missing debug logging at line 242 reduces observability |

## Recommended Actions (if desired)

1. **#8**: Add "are", "is", "was", "were", "did", "does", "do", "has", "have", "had" to the `_sparql_fallback` blocklist. Low effort, eliminates wasted SPARQL calls.
2. **#12**: Add `logger.debug("Type check failed for %s: %s", entity.uri, e)` at line 242 instead of bare `pass`.
3. **#10**: Extend the date regex to catch text-format dates: `r'^[\d,.\-/\s]+$|^(?:January|February|March|April|May|June|July|August|September|October|November|December)\b'`. Low priority.
4. **#7**: Add a code comment at the cache serialization block noting that only these four fields are cached, and any new Entity fields must be explicitly added.
5. **#9**: Consider removing `resolve_coreference()` entirely as dead code cleanup.
