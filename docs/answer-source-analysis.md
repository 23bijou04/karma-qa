# Answer Source Analysis: Gold Answer Reachability

**Dataset**: 2WikiMultiHopQA — 1,000-question evaluation subset
**Pipeline**: DBpedia KG properties + Wikipedia text retrieval
**Script**: `scripts/check_answer_sources.py`
**Output data**: `data/eval_1000q/answer_source_analysis_1000.json`
**Wikipedia pages**: `data/eval_1000q/wiki_pages_phase2.json` (raw fetched text)

## Status

**Both phases complete.** Phase 1 (offline) + Phase 2 (live Wikipedia API) results are
in the JSON output. Phase 2 was run on 2026-03-11.

## Methodology

### Two-Phase Verification

**Phase 1 — Offline check** (no API calls):
1. **DBpedia property coverage**: From pre-computed `coverage_analysis_1000.json`
   — checks if the evidence property exists as a structured DBpedia property.
2. **Benchmark context check**: The 2WikiMultiHopQA benchmark provides curated
   Wikipedia paragraph excerpts per question (10 paragraphs each). For each
   evidence hop `(entity, property, value)`, we first check the paragraph
   titled after the `entity` (own-entity paragraph), then fall back to other
   paragraphs. Handles disambiguated titles (e.g., `"Thomas Jefferson (film)"`
   matches entity `"Thomas Jefferson"`). This distinguishes:
   - **Own-entity paragraph**: value found in the entity's own Wikipedia excerpt
   - **Different-entity paragraph**: value found only in another entity's excerpt

**Phase 2 — Live Wikipedia API** (for hops without DBpedia property):
- Fetches actual Wikipedia pages via `get_wikipedia_text_by_title()` (up to 8,000 chars)
- Detects disambiguation pages (treated as fetch failures)
- Tests whether exact-title Wikipedia retrieval can find the value
- Raw pages saved to `data/eval_1000q/wiki_pages_phase2.json`

### Definitions

| Term | Meaning |
|------|---------|
| **Reachable hop** | Value exists in DBpedia property OR confirmed in entity's Wikipedia page via API |
| **Unreachable hop** | Value absent from both DBpedia AND entity's Wikipedia page (via API) |
| **Unanswerable question** | Has ≥1 unreachable hop in its evidence chain |

### Key distinction: hops vs unique entities

- **990 non-DBpedia hops** — total count of evidence hops not covered by DBpedia
- **941 unique entities** — deduplicated entity names across those 990 hops
- Phase 2 fetches one Wikipedia page per unique entity (941 API calls)

## Phase 2 Results (Final)

### Hop-Level Value Location (2,507 total hops)

| Source | Hops | % | Description |
|--------|------|---|-------------|
| DBpedia structured property | 1,517 | 60.5% | Value exists as a structured `dbo:`/`dbp:` property |
| Wikipedia API (confirmed) | 608 | 24.3% | Value found on entity's Wikipedia page via live API |
| Own paragraph only (API missed) | 279 | 11.1% | In benchmark's own-entity paragraph but NOT found via API |
| Different entity's paragraph | 103 | 4.1% | Value only on another entity's page — unreachable by design |
| Not found anywhere | 0 | 0.0% | All values found in at least one benchmark paragraph |
| **Total** | **2,507** | **100%** | |

### Wikipedia API Fetch Stats (941 unique entities)

| Status | Count | % |
|--------|-------|---|
| Successfully fetched | 623 | 66.2% |
| Disambiguation pages (treated as failure) | 94 | 10.0% |
| Not found (title mismatch) | 224 | 23.8% |
| **Total** | **941** | **100%** |

### Question-Level Reachability

| Metric | Count | % |
|--------|-------|---|
| All hops reachable | 665 | 66.5% |
| **≥1 hop unreachable** | **335** | **33.5%** |
| Total | 1,000 | 100% |

### By Question Type

| Type | Reachable | Total | % Reachable | % Unreachable |
|------|-----------|-------|-------------|---------------|
| comparison | 205 | 244 | 84.0% | 16.0% |
| bridge_comparison | 198 | 235 | 84.3% | 15.7% |
| compositional | 212 | 413 | 51.3% | 48.7% |
| inference | 50 | 108 | 46.3% | 53.7% |

Comparison types remain highest-reachability. Compositional and inference questions
suffer most — they depend on properties like `father`, `director`, `spouse` which
are often absent from both DBpedia and the first 8,000 chars of Wikipedia.

### Top Unreachable Properties

| Property | Unreachable Hops | % of Total Unreachable |
|----------|-----------------|------------------------|
| father | 81 | 21.2% |
| director | 69 | 18.1% |
| country of citizenship | 33 | 8.6% |
| mother | 26 | 6.8% |
| spouse | 25 | 6.5% |
| place of birth | 23 | 6.0% |
| date of birth | 20 | 5.2% |
| performer | 16 | 4.2% |
| composer | 16 | 4.2% |
| publication date | 14 | 3.7% |

Family relations (`father`, `mother`, `spouse`) still dominate. `director` jumped
from 5 (Phase 1) to 69 unreachable — many director values exist in benchmark
paragraphs but the film's Wikipedia page (8K chars) doesn't always mention the
director by the exact benchmark string.

## Phase 1 vs Phase 2 Comparison

| Metric | Phase 1 (proxy) | Phase 2 (exact-title API) |
|--------|-----------------|---------------------------|
| Reachable questions | 899 (89.9%) | 665 (66.5%) |
| Unreachable questions | 101 (10.1%) | 335 (33.5%) |

### Why Phase 2 shows more unreachable

Phase 2 is **stricter** than Phase 1 for three reasons:

1. **Title mismatches (224 entities)**: `get_wikipedia_text_by_title()` uses exact
   title matching. Wikidata entity names (used in the benchmark) frequently differ
   from Wikipedia article titles (e.g., full formal name vs common name).

2. **Disambiguation pages (94 entities)**: Ambiguous entity names resolve to
   disambiguation pages rather than the target article.

3. **8,000-char truncation**: Wikipedia API returns up to 8K chars of article text.
   The benchmark's curated paragraphs may extract relevant sections from deeper in
   the article that the API truncation misses.

### Why the actual pipeline performs better

Our actual pipeline (`hybrid_qa.py` → `text_solver.py` → `graphrag_wiki.py`) uses
**Wikipedia search as a fallback** when exact-title retrieval fails:
1. First tries exact title match
2. Falls back to `action=query&list=search` which handles name variants
3. Falls back to DuckDuckGo search (with circuit breaker)

An earlier manual spot-check of the 105 Phase 1 "unreachable" hops showed ~80
were actually reachable via Wikipedia search. This suggests the **true pipeline
reachability is between 66.5% (exact-title lower bound) and 89.9% (benchmark
paragraph upper bound)**, likely closer to ~85–90%.

### Three-layer reachability picture

| Layer | Reachable | Method |
|-------|-----------|--------|
| **Lower bound** (Phase 2) | 66.5% | Exact Wikipedia title only |
| **Estimated actual** | ~85–90% | With Wikipedia search fallback |
| **Upper bound** (Phase 1) | 89.9% | Curated benchmark paragraphs |

## Running the Analysis

```bash
# Phase 1 only (fast, no API calls):
python scripts/check_answer_sources.py --skip-wiki

# Full analysis (Phase 1 + Phase 2, ~8 min at 0.5s/entity):
python scripts/check_answer_sources.py

# Custom delay between API calls:
python scripts/check_answer_sources.py --delay 1.0
```

## Output Files

- `data/eval_1000q/answer_source_analysis_1000.json` — Full per-question, per-hop results
- `data/eval_1000q/wiki_pages_phase2.json` — Raw Wikipedia page text for all 941 entities

## TODO

1. Run a Phase 2.5: use `get_wikipedia_text()` (with search fallback) instead of
   `get_wikipedia_text_by_title()` to measure true pipeline reachability
2. Analyze the 279 "own paragraph only" hops — how many are recoverable via search?
