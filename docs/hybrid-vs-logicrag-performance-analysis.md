# Why Hybrid Outperforms LogicRAG: A Code-and-Data-Grounded Analysis

**Date:** 2026-03-26
**Source data:** `hybrid_eval_1000_judged.jsonl`, `logicrag_1000q_traced_judged.jsonl` (first 100 questions)
**Companion:** [hybrid-vs-logicrag-100q-comparison.md](hybrid-vs-logicrag-100q-comparison.md), [logicrag-vs-our-retrieval.md](logicrag-vs-our-retrieval.md)

> **Version provenance (2026-03-30):** This analysis was originally written using `.bak_v1` snapshots from 2026-03-26. The quantitative claims (percentages, deltas) reflect those older file versions. For current-current numbers, see [hybrid-vs-logicrag-100q-comparison.md](hybrid-vs-logicrag-100q-comparison.md) which was updated 2026-03-30 with consistent current versions. Key change: overall delta is +15pp (not +9pp), and inference sign-flip is corrected (Hybrid now leads on all types). The architectural analysis and code-path explanations in this document remain valid.

> Every conclusion below is traced to a code path or data record. No interpretive claims without evidence.

---

## 1. The Core Asymmetry: Corpus Scope

The fundamental difference between the two systems is **what they can retrieve**.

### LogicRAG's corpus

LogicRAG indexes 6,119 benchmark-provided snippets into a single FAISS index at startup (`baselines/LogicRAG/src/models/base_rag.py:42-52`). Each snippet is a Wikipedia intro paragraph (median 254 chars, 2 sentences). Retrieval is cosine similarity over `all-MiniLM-L6-v2` embeddings, top-k=5 (`base_rag.py:37,155`).

**Key constraint:** the corpus contains exactly the paragraphs bundled with the 2WikiMultiHopQA benchmark. For each question, the gold answer exists in exactly 2 of those 6,119 snippets by construction. But **bridging entities** (the intermediate entity in a 2-hop chain) may or may not have their own snippet in the corpus.

### Hybrid's information sources

Hybrid accesses three live sources:
1. **DBpedia SPARQL** — structured triples for any entity with a DBpedia URI (`sparql_solver.py`)
2. **Wikipedia API** — full articles fetched live, chunked into 512-char pieces (`text_solver.py`, `rag/chunker.py:27`)
3. **Sub-question decomposition** — routes each sub-question independently to SPARQL or text (`logicrag_decomposition.py`)

> **Statistical note:** This analysis covers 100 questions. Per-type differences (n=15 to n=46) are exploratory patterns; they do not reach statistical significance individually and await confirmation from the full 1000q evaluation.

**Key advantage:** no fixed corpus boundary. If an entity exists in DBpedia, Hybrid can query its properties directly via SPARQL without needing a text passage.

---

## 2. Where Hybrid Wins: Compositional Questions (+17.4pp)

**Data:** Hybrid 87.0% vs LogicRAG 69.6% on 46 compositional questions.

### The mechanism

Compositional questions follow a bridge pattern: "Where was the director of film X born?" requires two hops: (1) film → director, (2) director → birthPlace.

**Hybrid's code path** (`routing.py:151-167`): The rule-based router detects `bridge_keyword` + `hop2_keyword` patterns. For "Where was the director of film X born?", `question_analysis.py:20` maps "director" to the `director` property, and "born" to `birthPlace`. If `dbo:director` exists in the entity's actual properties, the router returns `rule:bridge_hop2`, verified by SPARQL ASK (`routing.py:303`). The SPARQL solver then generates a single 2-hop query. Note: LogicRAG also gets multiple retrieval attempts (warmup + up to 3 rounds with targeted sub-queries), so the comparison is not single-pass top-k=5 vs SPARQL — rather, fixed-corpus multi-round retrieval vs live-source access.

```sparql
SELECT ?birthPlace WHERE {
  dbr:Film_X dbo:director ?director .
  ?director dbo:birthPlace ?birthPlace .
}
```

**LogicRAG's code path** (`base_rag.py:132-167`): LogicRAG retrieves top-5 passages by cosine similarity to the question. For a 2-hop question, it needs both the film's snippet (naming the director) AND the director's snippet (containing birthPlace). If the director's snippet is not in the 6,119-passage corpus, retrieval fails.

### Evidence from data

In the 9 Hybrid-only wins, LogicRAG returned "Unknown" in 6/9 cases. Breakdown by Hybrid method:
- **sparql** (4): Direct 2-hop SPARQL chains — all via `rule:bridge_hop2+ask_verified`
- **logicrag_decomposed** (5): Sub-question decomposition — `rule:bridge_hop2+ask_missing` (4 cases) where ASK verification failed but decomposition succeeded, plus 1 LLM-routed case

**Concrete example** — "Where was the director of film The Blue And White Lion born?":
- Hybrid: routed via `rule:bridge_hop2+ask_verified`, SPARQL returns "Berlin" directly
- LogicRAG: corpus contains "The Blue and White Lion" snippet (naming the director) but lacks the director's biographical snippet → "Unknown" after 2 rounds

### Why LogicRAG's corpus lacks bridging entities

The 2WikiMultiHopQA corpus provides 10 context passages per question: 2 gold + 8 distractors. The gold passages cover the question's supporting facts. But the **intermediate entity** (e.g., the director named in a film snippet) may not have its own passage in the corpus. The corpus has 6,120 unique pages for 1,000 questions, of which 3,781 are pure distractors (`logicrag-vs-our-retrieval.md`). A director mentioned in a film passage is only present if that director happens to be a supporting fact for *some other* question.

LogicRAG's round distribution confirms this: most Hybrid-only wins occurred when LogicRAG used 2-3 rounds, indicating repeated failed retrieval attempts. Its accuracy correlates inversely with round count: 92.3% at 0 rounds → 67.6% at 2 rounds → 54.5% at 3 rounds (though see Section 7 for caveats on interpreting this pattern).

---

## 3. Where Hybrid Wins: Bridge Comparison (+21.1pp)

**Data:** Hybrid 94.7% vs LogicRAG 73.7% on 19 bridge_comparison questions.

### The mechanism

Bridge comparison questions compare a property across two entity chains: "Which film has the director who died first, X or Y?" requires resolving director → deathDate for both films, then comparing.

**Hybrid's code path** (`routing.py:124-149`): The `bridge_comparison` rule fires when `analysis.is_comparison` and `analysis.bridge_keyword` are set, AND a 2-hop chain exists in `CACHED_CLASS_SCHEMAS`. For Film entities, `dbo_2hop` includes `director → Person → [birthDate, deathDate, birthPlace, nationality]` (`schema_discovery.py:73-78`). The SPARQL solver generates paired 2-hop queries comparing the target property.

**Route success rate:** `rule:bridge_comparison+ask_verified` achieved 100% accuracy (12/12) in the first 100 questions — the highest of any route source.

**LogicRAG's failure mode:** To answer a bridge_comparison, LogicRAG needs 4 passages: film-A snippet, director-A snippet, film-B snippet, director-B snippet. With top-k=5 retrieval from 6,119 passages, finding all 4 relevant passages is unlikely. Data shows 4 Hybrid-only wins on bridge_comparison, all with LogicRAG answering "Unknown" (3 cases) or wrong (1 case: "No" instead of "yes" for a nationality comparison).

---

## 4. Where LogicRAG Wins: Inference Questions (-20pp)

**Data:** LogicRAG 100% vs Hybrid 80.0% on 15 inference questions.

### The mechanism: semantic loss in genealogical routing

All 3 LogicRAG-only wins are genealogical questions (e.g., "Who is Philippe, Duke of Anjou's maternal grandfather?"). These expose a specific weakness: the question analysis layer conflates maternal/paternal modifiers into a generic "parent" token, and this ambiguity propagates through routing and SPARQL generation.

**Root cause — Question analysis conflates genealogical modifiers** (`question_analysis.py:31-35`):

```python
"grandfather": ["father", "parent"],    # line 34
"grandmother": ["mother", "parent"],    # line 34
"paternal": ["parent"],                 # line 35
"maternal": ["parent"],                 # line 35
```

Both "paternal" and "maternal" map to the same generic `["parent"]`. The semantic distinction between maternal and paternal lineages is lost before routing even begins. Additionally, `_HOP2_KEYWORDS` (`question_analysis.py:85-90`) does not include genealogical terms like "grandfather" or "grandmother," so these questions are not recognized as having a 2-hop structure.

**Downstream propagation — Routing and SPARQL generation**: The parent chain rule (`routing.py:169-175`) fires when `bridge_keyword` is in `("father", "mother", "parent", "paternal", "maternal")` and the entity has any property in `_PARENT_PROPS`, but does not enforce that "maternal" should map specifically to `dbo:mother`. The schema context (`CACHED_CLASS_SCHEMAS` at `schema_discovery.py:18-51`, rendered by `_build_schema_text()` at lines 617-666) lists `father`, `mother`, `parent` as available properties, but the SPARQL generation prompt (`sparql_solver.py:64-82`) does not anchor the question's semantic direction to a specific property. The LLM must independently decide which property to use.

**Where it breaks — the Philippe example:**

The actual SPARQL query generated (extracted from eval trace):
```sparql
SELECT ?grandfather WHERE {
  dbr:Philip_V_of_Spain dbp:mother ?mother .
  ?mother dbo:parent ?grandfather .
}
```

Hop 1 is correct (`dbp:mother`). But hop 2 uses generic `dbo:parent` instead of `dbo:father` (maternal *grandfather* = mother's *father*). `dbo:parent` returns any parent — whichever appears first in the KG. The SPARQL returned "Ferdinand Maria, Elector of Bavaria" (wrong ancestor) instead of "Stanislaw Leszczynski" (correct maternal grandfather).

All 3 LogicRAG-only wins exhibit this pattern:

| Question | Gold | Hybrid (wrong) | Root cause |
|---|---|---|---|
| Philippe, Duke of Anjou's maternal grandfather | Stanisław Leszczyński | Ferdinand Maria, Elector of Bavaria | `dbo:parent` returned wrong parent |
| Maria Antonia of Austria's maternal grandfather | Carlos, Duke of Madrid | Louis Rudolph, Duke of Brunswick | Same `dbo:parent` ambiguity |
| Teriimaevarua III's paternal grandmother | Pōmare IV | Ariifaaite | Same `dbo:parent` ambiguity |

**Why LogicRAG succeeds:** The benchmark corpus was constructed from Wikidata relations and 2017 Wikipedia text. The gold passages explicitly state genealogical relationships in prose (e.g., "She was the daughter of X and Y"). LogicRAG retrieves these passages directly — no need to navigate ambiguous KG relations.

---

## 5. The Comparison Tie (95% each)

**Data:** Both systems achieve 95% on 20 comparison questions.

### Why they converge

Comparison questions ("Do X and Y share the same nationality?") have binary yes/no answers. Both systems can handle these because:

- **Hybrid**: Direct date/country comparison via `rule:direct_date_comparison` or `rule:direct_country_comparison` (`routing.py:94-122`). Requires the target property (e.g., `dbo:birthDate`) to exist on ALL compared entities (intersection check at line 104).
- **LogicRAG**: Comparison questions often answered in round 0 (warmup retrieval). Data shows 36/39 round-0 questions were correct (92.3%). Both entity snippets are typically in the benchmark's 10-passage context.

Both systems have access to sufficient information for these structured queries.

---

## 6. Neither-Correct Cases: Shared Blind Spots

**Data:** 7 questions where both systems fail — 5 compositional, 1 comparison, 1 bridge_comparison.

Examination of these 7 cases reveals two patterns:

1. **Ambiguous gold answers** (2 cases): "British" as birthplace for a director actually born in "Kettering, Northamptonshire"; "Chinese" as birthplace for an entity with unclear geographic context. Hybrid found "Kettering, Northamptonshire" — correct but doesn't match gold.

2. **Obscure entities** (5 cases): "Consort Liang's husband," "Gwladys's husband" — entities with minimal Wikipedia coverage and incomplete KG relations. Both systems returned "Unknown." 6 of 7 neither-correct cases have `is_answerable: false`, confirming these are genuinely hard.

---

## 7. LogicRAG's Retrieval Round Distribution

LogicRAG's accuracy correlates inversely with retrieval rounds:

| Rounds | Count | Correct | Rate |
|---|---|---|---|
| 0 (warmup) | 39 | 36 | 92.3% |
| 1 | 16 | 15 | 93.8% |
| 2 | 34 | 23 | 67.6% |
| 3 | 11 | 6 | 54.5% |

**Code-level explanation** (`base_rag.py:132-167`): Each round queries the same 6,119-passage FAISS index with a new sub-query. If the needed information isn't in the corpus, additional rounds retrieve increasingly irrelevant passages. The John Middleton Murry example is illustrative: after warmup identified Katherine Mansfield as the wife, rounds 1-3 searched for "cause of death" but retrieved "A Matter of Life and Death," "Dance of Death," and "Traces of Death" — disambig pages matching "death" semantically but containing no relevant facts.

> **Selection bias caveat:** The correlation between round count and accuracy likely reflects selection bias rather than pure degradation. Questions that exit at round 0 tend to be inherently simpler (comparisons, single-hop lookups), while questions reaching 2-3 rounds are harder by definition — the warmup already determined the needed passage was missing. Notably, round 1 accuracy (93.8%) is *higher* than round 0 (92.3%), inconsistent with a monotonic degradation model. A stratified analysis controlling for question difficulty would be needed to isolate the causal effect of additional rounds.

**Hybrid's decomposition avoids the fixed-corpus ceiling** because each sub-question routes independently to SPARQL or live Wikipedia. Sub-question "What is the cause of death of Katherine Mansfield?" can query `dbo:deathCause` directly on DBpedia.

---

## 8. The Fairness Caveat

This comparison is **structurally unfair** in both directions:

**Favors Hybrid:**
- Live Wikipedia + DBpedia SPARQL = unbounded information access
- LogicRAG's 6,119-snippet corpus is a strict subset of available information
- "Unknown" answers reflect corpus limitation, not retrieval quality
- Higher computation budget: multiple LLM calls (routing, decomposition, sub-question solving, final synthesis) vs LogicRAG's single-model approach with retry rounds
- Hand-crafted routing rules encode domain knowledge (bridge patterns, comparison detection) that LogicRAG must learn implicitly from retrieval
- Temporal advantage: 2026 DBpedia/Wikipedia may contain information updated since the 2017 benchmark snapshot

**Favors LogicRAG:**
- Benchmark answers were derived from the same corpus LogicRAG indexes
- Gold answers match the phrasing in benchmark passages (surface form alignment)
- LogicRAG never encounters entity linking errors (titles are pre-resolved)

This analysis explains *why* the systems differ, not which approach is "better" in absolute terms.

---

## 9. Summary of Causal Mechanisms

| Performance gap | Causal mechanism | Code evidence | Data evidence |
|---|---|---|---|
| Compositional +17.4pp | SPARQL 2-hop chains bypass corpus limitation | `routing.py:151-167` bridge_hop2 rule, `schema_discovery.py:73-78` Film→director→Person chains | 6/9 Hybrid-only wins: LogicRAG "Unknown" |
| Bridge comparison +21.1pp | Same as above, with comparison logic | `routing.py:124-149` bridge_comparison rule, 100% accuracy on `ask_verified` | 4/4 bridge_comparison Hybrid wins: LogicRAG couldn't retrieve 4 needed passages |
| Inference -20pp | `dbo:parent` ambiguity in genealogical chains | `question_analysis.py:31-35` conflates maternal/paternal, `routing.py:169-175` parent_chain rule | 3/3 LogicRAG wins: Hybrid followed wrong ancestor |
| Comparison tie | Both have sufficient access to compared entities' properties | `routing.py:94-122` direct comparison rule | 95% = 95%, both handle binary answers |
| LogicRAG round correlation | Harder questions require more rounds (selection bias); fixed corpus limits ceiling | `base_rag.py:132-167` same FAISS index queried each round | 92.3% at 0 rounds → 54.5% at 3 rounds (but round 1 > round 0) |
