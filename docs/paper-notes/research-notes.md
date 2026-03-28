# Research Notes

Paper-relevant research findings consolidated from investigation reports.

## 1. No Wikidata Fallback — Data Leakage Risk

**Date:** 2026-03-24 | **Source:** `wikidata-deprecation-detection-research.md`

### Constraint
2WikiMultiHopQA sourced ground-truth answers from Wikidata (Dec 2019 snapshot). Using Wikidata as a knowledge source = data leakage. Our system uses **DBpedia only**.

### Wikidata Deprecation Detection (for future reference)
- Three approaches: SPARQL query service (recommended), REST API (`wbgetclaims`), dump-based
- `wdt:` prefix auto-excludes deprecated claims; must use `p:` prefix + `wikibase:rank` filter
- ~25K entities in our dataset, feasible via SPARQL endpoint (60s timeout)

### Implication for Paper
- Explicitly state why Wikidata is excluded (Section: Experimental Setup)
- This constraint makes our task harder — DBpedia property coverage is lower than Wikidata

---

## 2. Wikipedia Revision Analysis — Ground Truth Validity

**Date:** 2026-03-24 | **Source:** `margaretha-van-mechelen-wikipedia-revision-analysis.md`

### Case Study: Margaretha van Mechelen
- Wikipedia has **always** described her as Maurice of Nassau's "mistress" (since article creation 2008)
- 2WikiMultiHopQA gold answer: "spouse" (from Wikidata P26)
- Wikipedia text never supported this — terminology stable across 17 years

### Implication for Paper
- Demonstrates KG-text mismatch: Wikidata models relationship as `P26` (spouse), Wikipedia says "mistress"
- Gold answers may disagree with text evidence — affects text-only baselines
- Our system (DBpedia) has `dbo:spouse` for this entity, matching Wikidata — KG path is correct even when text path fails
- Use as illustrative example of why hybrid (KG + text) matters
