# Hyperparameters & Magic Numbers Audit

All values verified against the codebase on 2026-03-11. Each entry was fact-checked by independent code-review agents.

---

## Tier 1: Core Architecture Parameters

These define the system architecture and **must be explained in the thesis**.

| # | Value | File:Line | Parameter | Rationale |
|---|-------|-----------|-----------|-----------|
| 1 | `0.5` | `entity_linker.py:89`, `hybrid_qa.py:139`, `graphrag_wiki.py:718` | Spotlight `min_confidence` | Filters noise entities. Class default and all callers use `0.5`. Lower values let through false positives (e.g., "Albert" -> Uncle Albert at confidence 0.40). |
| 2 | `12,000` | `graphrag_wiki.py:44` (named constant `MAX_CONTEXT_CHARS`), `text_solver.py:48,195` (raw value, DRY violation) | Total LLM context budget (chars) | Fits within LLaMA-3.3-70B's context window while leaving room for system prompt + question + reasoning. ~3K tokens. Note: `text_solver.py` independently hardcodes `12_000` rather than importing the named constant. |
| 3 | `3,000` | `graphrag_wiki.py:45` | `MAX_CHARS_PER_PAGE` | Per-page extraction budget. With 4-5 pages typical, `4x3K = 12K` total. Prevents a single verbose article from dominating context. |
| 4 | `60% / 30% / 10%` | `graphrag_wiki.py:651-652` | Seed/bridge/slack budget split | Seed entities (directly mentioned in question) get 60% of context budget, bridge entities (KG-discovered) get 30%. The remaining 10% is implicit slack absorbed by the final `MAX_CONTEXT_CHARS` cap (documented in comment at line 645). |
| 5 | `512` / `64` | `rag/chunker.py:27-28` | `chunk_size` / `overlap` | 512 chars ~= 100-120 tokens — standard for bi-encoder models like MiniLM (max 256 tokens). 64-char overlap (~12%) preserves sentence continuity across chunk boundaries. |
| 6 | `top_k=20` | `text_solver.py:183` | Bi-encoder retrieval count | Retrieves 20 candidate chunks from FAISS. Over-retrieves to give the cross-encoder reranker enough candidates (which then narrows to 8). Standard two-stage retrieval pattern. |
| 7 | `top_k=8` | `text_solver.py:192` | Cross-encoder rerank output | Final 8 chunks passed to LLM. ~8x512 chars = 4K chars of high-quality passages, leaving room for KG facts and prompts within the 12K budget. |
| 8 | `seed_boost=1.2` | `rag/vector_store.py:74` | Seed entity score multiplier | 20% boost for chunks from seed entity pages in bi-encoder scoring. Mild preference — doesn't override genuinely more relevant bridge entity text, but breaks ties in favor of directly-mentioned entities. |
| 9 | `6` | `logicrag_decomposition.py:362-363` | Max sub-questions cap | Safety limit on LLM-generated decomposition. 2WikiMultiHopQA has max 2-hop questions; 6 allows for comparison questions with 2 entities x 3 sub-Qs each. Guard at line 362, truncation `sub_qs[:6]` at line 363. |
| 10 | `temperature=0` | `llm_provider.py:63` | LLM determinism | Zero temperature for reproducible answers. Required for evaluation (EM/F1 metrics assume deterministic output). |

---

## Tier 2: Retrieval & Filtering Thresholds

These affect accuracy and should be explained if questioned.

| # | Value | File:Line | Parameter | Rationale |
|---|-------|-----------|-----------|-----------|
| 11 | `50,000` | `graphrag_wiki.py:510,525` (also 544, 561) | `max_chars` per Wikipedia fetch | Raw page download limit. Long articles (e.g., "United States") can be 200K+ chars. 50K captures lead + major sections without OOM risk. |
| 12 | `< 0.1` | `graphrag_wiki.py:353` | Rare keyword frequency threshold | A keyword appearing in <10% of paragraphs is "rare" and gets 2x scoring weight. Common keywords (appearing everywhere) provide less signal for passage ranking. |
| 13 | `2x` | `graphrag_wiki.py:369` | Relation section boost | Paragraphs matching a relation hint (e.g., "early life" for `birthPlace`) get doubled score via `score *= 2`. Exploits the KG-discovered relation as a topical signal. |
| 14 | `2+ hits or 1 rare` | `graphrag_wiki.py:363` | Density filter threshold | A paragraph needs >=2 keyword hits OR >=1 rare keyword hit to pass the density filter. Prevents diluting context with low-signal paragraphs. |
| 15 | `limit=30` | `entity_discovery.py:37` | wikiPageWikiLink discovery limit | How many linked entities to fetch per seed. 30 balances coverage vs. noise — most Wikipedia pages have 50-200+ wikiPageWikiLinks. |
| 16 | `max_per_seed=5` | `entity_discovery.py:136` | Discovered entities per seed | After scoring, keep top 5 discovered entities per seed. Prevents flooding the Wikipedia retrieval with too many low-relevance pages. |
| 17 | `ask_limit=10` | `entity_discovery.py:70` | SPARQL ASK verification limit | Only run ASK queries on the top 10 candidates (expensive). A cost/accuracy tradeoff — ASK signals (+3 backlink, +2 hop2 property) are strong but SPARQL queries are slow. |
| 18 | `limit_per_entity=10` | `hybrid_qa.py:59` | Simple neighbor limit | In `_sparql_neighbors()`, fetches 10 structured (dbo:/dbp:) neighbors per entity. Lighter than full `explore_kg` (50), used in the text fallback path. |
| 19 | `limit_per_entity=50` | `graphrag_wiki.py:210` | `explore_kg` neighbor limit | Full KG exploration fetches up to 50 neighbors per entity. Higher than simple path because schema-guided filtering reduces the effective count significantly. |
| 20 | `1500` | `graphrag_wiki.py:655` | Bridge per-page minimum budget | Each bridge entity page gets at least 1500 chars via `max(1500, bridge_budget // n_bridge)`. Ensures every discovered entity contributes meaningfully. |

---

## Tier 3: Retry & Rate Limiting (Operational)

These are operational parameters, not research hyperparameters.

| # | Value | File:Line | Parameter | Rationale |
|---|-------|-----------|-----------|-----------|
| 21 | `max_attempts=5` | `llm_provider.py:62` | LLM retry count | 5 attempts with exponential backoff (10s, 20s, 40s, 80s, 160s). Groq rate limits are aggressive; 5 retries covers typical bursts. |
| 22 | `initial_wait=10.0` | `llm_provider.py:62` | LLM initial retry delay | 10 seconds base wait. Groq 429s typically clear within 10-30s. Exponential: `10 x 2^attempt`. |
| 23 | `rate=10.0` | `dbpedia/client.py:24` | SPARQL token bucket rate | 10 req/sec to DBpedia. DBpedia's public endpoint doesn't publish rate limits; 10/sec is conservative enough to avoid bans. |
| 24 | `timeout=30` | `dbpedia/client.py:61` | SPARQL query timeout | 30s covers complex GROUP BY queries. Matches DBpedia Virtuoso's default server-side timeout. |
| 25 | `timeout=60` | `sparql_evaluation.py:274` | Evaluation SPARQL timeout | 60s for evaluation (more generous). Some LLM-generated queries are sub-optimal and need more time. |
| 26 | `max_retries=2` | `sparql_evaluation.py:275` | SPARQL execution attempts | 2 total attempts (1 initial + 1 retry) via `range(max_retries)`. Quick recovery without excessive delay. |
| 27 | `threshold=3, cooldown=60.0` | `graphrag_wiki.py:80,105` | DDG circuit breaker | DuckDuckGo silently blocks after ~50 calls. 3 consecutive failures -> 60s cooldown. Prevents wasting time on a blocked service. |
| 28 | `stop_after_attempt(3)` | `dbpedia/client.py:73`, `dbpedia/entity_linker.py:291` | Tenacity retries | 3 attempts with `wait_exponential(multiplier=1, max=10)` for Spotlight API and SPARQL. Standard retry pattern for external services. |
| 29 | `0.5s` | `sparql_evaluation.py:634` | Batch eval polite delay | `time.sleep(0.5)` between questions. Prevents overwhelming DBpedia/LLM during batch evaluation. |
| 30 | `max_tokens=1000` | `llm_provider.py:63` | Default LLM max tokens | Enough for SPARQL generation + reasoning. 1000 covers typical QA responses. |
| 31 | `max_tokens=500` | `graphrag_wiki.py:786` | GraphRAG LLM max tokens | Shorter for the text-only pipeline (simpler reasoning, no SPARQL generation). |
| 32 | `min_support=5` | `dbpedia/entity_linker.py:95` | Spotlight support threshold | Minimum Wikipedia inlinks for entity candidates. Lowered from 20 to 5 to catch obscure but real entities (e.g., Margaretha_van_Mechelen, support=8). Confidence threshold (0.5) remains the primary quality gate. |

---

## Tier 4: Display & Rendering Limits

These control how much schema/results are shown to the LLM. Not tunable hyperparameters — they're prompt engineering choices.

| # | Value | File:Line | Parameter | Rationale |
|---|-------|-----------|-----------|-----------|
| 33 | `[:5]` / `[:2]` | `schema_projection.py:40,45` | Routing render truncation | ~150 tokens for routing decision. Top 5 properties (line 40), top 2 chains (line 45). LLM routing prompt should be compact. |
| 34 | `[:10]` | `schema_projection.py:74,76` | SPARQL gen cached schema display | Top 10 cached properties per prefix (dbo/dbp). ~300 tokens for SPARQL generation — gives LLM enough schema context. |
| 35 | `[:8]` + `max(2, 10-shown)` | `schema_projection.py:102,106` | Actual property display | Show 8 dbo properties, then fill remaining slots (up to 10 total) with dbp. Prevents prompt bloat. |
| 36 | `max_chains=5, max_props=4` | `dbpedia/schema_discovery.py:219` | 2-hop chain formatting | Cap chain display at 5 chains x 4 properties each. Keeps schema context focused. |
| 37 | `[:30]` | `logicrag_decomposition.py:331` | Schema summary truncation | First 30 lines of schema for decomposition prompt. Compact enough for sub-question generation. |
| 38 | `[:10]` / `[:20]` | `sparql_solver.py:137,147` | Result truncation for LLM | `results[:10]` at line 137 (sub-Q preserve_uris path, human-readable). `results[:20]` at line 147 (default JSON path). Prevents context overflow from large result sets. |
| 39 | `LIMIT 100` | `sparql_solver.py:42` | SPARQL result limit | Instructed in SPARQL generation prompt rule #4. Caps SPARQL engine output. 100 rows is more than enough for QA. |
| 40 | `LIMIT 50` | `dbpedia/schema_discovery.py:498` | Actual properties query limit | 50 properties per entity. Most entities have 20-40 structured properties; 50 catches edge cases. |

---

## Tier 5: Entity Discovery Scoring Weights

Weights in `entity_discovery.py` for ranking wikiPageWikiLink candidates.

| Signal | Score | File:Line | Rationale |
|--------|-------|-----------|-----------|
| Label-question word overlap | `+1` | `entity_discovery.py:89` | Weak signal: name similarity to question |
| Already known entity | `-2` | `entity_discovery.py:93` | Penalty: avoid re-discovering what we already have |
| Backlink (bidirectional wikiLink) | `+3` | `entity_discovery.py:110` | Strong signal: mutual linkage suggests strong relationship |
| Has hop2 property | `+2` | `entity_discovery.py:122` | Medium signal: entity has the property we'll need downstream |

---

## Tier 6: Cached Knowledge (Architectural Decisions)

These are static knowledge structures, not tunable parameters.

| Item | File:Lines | Count | Details |
|------|-----------|-------|---------|
| Cached class schemas | `dbpedia/schema_discovery.py:18-188` | 7 | Person, Film, Company, Place, MusicalArtist, Book, MusicalWork — covers ~85% of 2WikiMultiHopQA entity types |
| `PROPERTY_RANGES` | `dbpedia/schema_discovery.py:168-207` | 30 | Property -> target class mapping for 2-hop chain resolution |
| `_CLASS_TO_SCHEMA` | `graphrag_wiki.py:128-138` | 25 | Subclass -> parent schema mapping (e.g., Scientist->Person, Country->Place) |
| `_NOISE_CLASSES` | `graphrag_wiki.py:141-144` | 9 | Genre, Language, TopicalConcept, TimePeriod, Award, EthnicGroup, Currency, Colour, OlympicResult |
| `KEYWORD_TO_PROPERTIES` | `question_analysis.py:14-57` | 49 | Deterministic keyword->property map for routing |
| `RELATION_SECTION_BOOST` | `graphrag_wiki.py:59-85` | 25 | Relation->section keyword mapping for context extraction boosting |
| `_EVICTION_DAYS` | `rag/cache.py:13` | 7 days | SQLite chunk cache TTL — Wikipedia content doesn't change that fast |
| Embedding dimension | `rag/cache.py:83` | 384 | Hardcoded for `all-MiniLM-L6-v2` output dimension in `reshape(-1, 384)` |

---

## Model Choices

Justify in the methodology section of the thesis.

| Model | Used For | File | Why |
|-------|----------|------|-----|
| `llama-3.3-70b-versatile` | All LLM calls (default) | `hybrid_qa.py:121`, `graphrag_wiki.py:715` | Best Groq-hosted model: 70B params, strong reasoning, free tier |
| `all-MiniLM-L6-v2` | Bi-encoder embeddings | `rag/vector_store.py:15,38` | 384-dim, fast, good quality for passage retrieval. Standard RAG choice |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking | `rag/reranker.py:13,37` | MS-MARCO trained, best lightweight cross-encoder for passage reranking |
| `en_core_web_sm` | spaCy NER/noun chunks | `dbpedia/entity_linker.py:17` | Small English model — fast, sufficient for entity mention extraction |

---

## Parameters Recommended for Sensitivity Analysis

These would benefit most from empirical justification (ablation/sensitivity study) in the thesis:

1. **`min_confidence=0.5`** — Spotlight threshold (affects entity linking precision/recall tradeoff)
2. **`MAX_CONTEXT_CHARS=12,000`** — Total context budget (affects information completeness vs. noise)
3. **`60/30/10` budget split** — Seed vs. bridge allocation (affects multi-hop answer coverage)
4. **`chunk_size=512`** — RAG chunk granularity (affects retrieval precision)
5. **`top_k=20->8` two-stage retrieval** — Pipeline width (affects recall vs. reranking quality)
6. **`seed_boost=1.2`** — Seed entity preference strength (affects entity prioritization)

---

## Known DRY Violations

| Value | Canonical Location | Duplicated At | Note |
|-------|-------------------|---------------|------|
| `12,000` | `graphrag_wiki.py:44` (`MAX_CONTEXT_CHARS`) | `text_solver.py:48,195` (raw `12_000`) | `text_solver.py` should import the constant |
