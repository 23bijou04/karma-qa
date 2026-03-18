# Answer Source Analysis: Gold Answer Reachability

**Dataset**: 2WikiMultiHopQA — 1,000-question evaluation subset
**Pipeline**: DBpedia KG properties + Wikipedia text retrieval
**Script**: `scripts/check_answer_sources.py`
**Output data**: `data/eval_1000q/answer_source_analysis_1000.json`
**Wikipedia pages**: `data/eval_1000q/wiki_pages_phase2.json` (raw fetched text)

## Status

**Both phases complete.** Phase 1 (offline) + Phase 2 (live Wikipedia API with search
fallback) results are in the JSON output. Run on 2026-03-11.

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
- Fetches actual Wikipedia pages via `get_wikipedia_text()` — exact title match
  with search fallback (up to 8,000 chars)
- Detects disambiguation pages (treated as fetch failures)
- Confirms whether our pipeline's retrieval method can find the value
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

## Results (Final)

### Hop-Level Value Location (2,507 total hops)

| Source | Hops | % | Description |
|--------|------|---|-------------|
| DBpedia structured property | 1,517 | 60.5% | Value exists as a structured `dbo:`/`dbp:` property |
| Wikipedia API (confirmed) | 763 | 30.4% | Value found on entity's Wikipedia page via live API |
| Own paragraph only (API missed) | 204 | 8.1% | In benchmark's own-entity paragraph but NOT found via API |
| Different entity's paragraph | 23 | 0.9% | Value only on another entity's page — unreachable by design |
| Not found anywhere | 0 | 0.0% | All values found in at least one benchmark paragraph |
| **Total** | **2,507** | **100%** | |

### Wikipedia API Fetch Stats (941 unique entities)

| Status | Count | % |
|--------|-------|---|
| Successfully fetched | 844 | 89.7% |
| Disambiguation pages | 97 | 10.3% |
| Not found | 0 | 0.0% |
| **Total** | **941** | **100%** |

With search fallback enabled, all 941 entities resolve to a Wikipedia page or
disambiguation page. The previous exact-title-only run had 224 false negatives.

### Question-Level Reachability

| Metric | Count | % |
|--------|-------|---|
| All hops reachable | **798** | **79.8%** |
| ≥1 hop unreachable | 202 | 20.2% |
| Total | 1,000 | 100% |

### By Question Type

| Type | Reachable | Total | % Reachable | % Unreachable |
|------|-----------|-------|-------------|---------------|
| comparison | 222 | 244 | 91.0% | 9.0% |
| bridge_comparison | 211 | 235 | 89.8% | 10.2% |
| inference | 83 | 108 | 76.9% | 23.1% |
| compositional | 282 | 413 | 68.3% | 31.7% |

Comparison types achieve >89% reachability. Compositional questions are hardest —
they depend on properties like `director`, `father` which may not appear in the
first 8,000 chars of the Wikipedia article.

### Top Unreachable Properties (227 unreachable hops across 202 questions)

| Property | Unreachable Hops | % of Unreachable |
|----------|-----------------|------------------|
| director | 50 | 22.0% |
| father | 32 | 14.1% |
| place of birth | 19 | 8.4% |
| country of citizenship | 19 | 8.4% |
| date of birth | 16 | 7.0% |
| composer | 15 | 6.6% |
| performer | 13 | 5.7% |
| mother | 12 | 5.3% |
| spouse | 10 | 4.4% |
| publication date | 8 | 3.5% |

`director` is now the top unreachable property (50 hops). These are cases where
the entity's Wikipedia page (first 8K chars) doesn't contain the director's name
in the exact form used by the benchmark. Family relations (`father`, `mother`,
`spouse`) account for 54/227 (23.8%) of unreachable hops.

## Phase Comparison

| Metric | Phase 1 (proxy) | Phase 2 (exact-title) | Phase 2 (with search) |
|--------|-----------------|------------------------|------------------------|
| Reachable questions | 899 (89.9%) | 665 (66.5%) | **798 (79.8%)** |
| Unreachable questions | 101 (10.1%) | 335 (33.5%) | **202 (20.2%)** |

### Interpretation

- **Phase 1 (89.9%)** is an upper bound — benchmark paragraphs are curated and may
  include text our pipeline can't retrieve.
- **Phase 2 exact-title (66.5%)** was an artificial lower bound — 224 entities
  failed due to title mismatches, not actual unreachability.
- **Phase 2 with search (79.8%)** is the best estimate of real pipeline reachability.
  Uses the same search fallback as the actual pipeline.

### Remaining gap: 79.8% vs 89.9%

The 204 hops in "own paragraph only" are values the benchmark's curated paragraphs
contain but our 8,000-char API retrieval misses. Causes:
1. **8K truncation**: value appears deeper in the article than the first 8,000 chars
2. **Disambiguation mismatch**: search finds a related but different article
3. **Fuzzy match boundary**: benchmark paragraph has a variant form that matches
   while the full article text doesn't trigger the fuzzy matcher

## Running the Analysis

```bash
# Phase 1 only (fast, no API calls):
python scripts/check_answer_sources.py --skip-wiki

# Full analysis (Phase 1 + Phase 2, uses cache if present):
python scripts/check_answer_sources.py

# Retry failed fetches with search fallback:
python scripts/retry_wiki_fetches.py

# Custom delay between API calls:
python scripts/check_answer_sources.py --delay 1.0
```

## Output Files

- `data/eval_1000q/answer_source_analysis_1000.json` — Full per-question, per-hop results
- `data/eval_1000q/wiki_pages_phase2.json` — Raw Wikipedia page text for all 941 entities
