# Error Analysis

To understand why KARMA fails, we classify the 589 judged-incorrect questions
across the three benchmarks: 106/1000 on 2Wiki, 139/1000 on HotpotQA, and
344/1000 on MuSiQue, for an aggregate fail rate of 19.6%. The classification
uses full pipeline traces, including routing decisions, sub-question logs, and
retrieved chunks. All counts are verified against the raw judged JSONL with
gemma3:12b as judge.

MuSiQue's failure rate is 34.4%, or 3.2x the 2Wiki rate, consistent with its
higher hop count and lower KG connectivity. This analysis does not attempt to
fully decouple hop depth from other inter-benchmark construction differences.

## Failure taxonomy

The 589 failures fall into ten clusters. Six clusters dominate, each accounting
for at least 5% of failures.

| # | Cluster | 2Wiki | HPQA | MuSiQue | Total | % |
|---|---|---:|---:|---:|---:|---:|
| I | KG forces text, text fails | 61 | 74 | 102 | 237 | 40.2 |
| II | Sub-question returns wrong fact | 2 | 6 | 82 | 90 | 15.3 |
| III | Routing bypass on multi-hop | 22 | 30 | 13 | 65 | 11.0 |
| IV | Under-decomposition | 3 | 0 | 85 | 88 | 14.9 |
| V | Over-decomposition | - | - | 38 | 38 | 6.5 |
| VI | Comparison-logic error | 18 | 16 | - | 34 | 5.8 |
| VII | Entity-link / null sub-Q | 0 | 8 | - | 8 | 1.4 |
| VIII | Hedging / refusal | 0 | 0 | 14 | 14 | 2.4 |
| IX | DBpedia--Wikipedia format gap | 0 | 5 | - | 5 | 0.8 |
| X | Chain-text-miss intermediate | 0 | 0 | 10 | 10 | 1.7 |
| | **Total** | **106** | **139** | **344** | **589** | **100** |

## Cluster I: KG forces text, text fails

This is the dominant failure mode at 40.2%. The KG signals absence of a 2-hop
path, so the question falls back to text retrieval, which then fails for
benchmark-specific reasons.

On 2Wiki, 61 cases retrieve the right Wikipedia page, but the predicted string
refers to the same concept under a different label, such as "Ottoman Empire" vs.
"Turkey", so EM/F1 string matching fails.

On HotpotQA, 74 cases promote a topically similar but wrong Wikipedia page, such
as the director's biography instead of the film page. All 74 cases exhibit
topically related but incorrect retrieval.

On MuSiQue, 102 cases fail because DBpedia connectivity collapses beyond two
hops, so text must traverse the entire chain unaided, which Wikipedia pages do
not support.

## Cluster IV: under-decomposition

Under-decomposition accounts for 14.9% of failures and is almost entirely a
MuSiQue problem. The decomposition LLM produces fewer sub-questions than the hop
count:

| Pattern | Cases |
|---|---:|
| 4-hop -> 3 sub-Qs | 46 |
| 3-hop -> 2 sub-Qs | 30 |
| 4-hop -> 2 sub-Qs | 7 |
| 2-hop -> 1 sub-Q | 2 |

About 80% of these 88 cases co-occur with all-text retrieval. When the KG cannot
help, decomposition errors compound. Prompt revisions targeting this pattern are
left to future work.

## Cluster II: sub-question returns wrong fact

This cluster accounts for 15.3% of failures and is dominant on MuSiQue. A
sub-question executes but retrieves the wrong answer. The sharpest pattern is in
MuSiQue's 82 cases; the remaining 8 cases, 2 on 2Wiki and 6 on HotpotQA, were
not separately re-classified.

On the 82 MuSiQue cases, 91.5% (75/82) have the gold entity absent from the
entire retrieved chain. Only 7/82 are genuine final-synthesis errors. This
suggests the bottleneck is retrieval, not synthesis: improving the synthesis
prompt will not help if the gold fact was never retrieved.

## Cluster III: routing bypass on multi-hop

Routing bypass accounts for 11.0% of failures and appears across all
benchmarks. Multi-hop questions are answered by SPARQL alone because routing
matches a surface predicate, such as "director" -> `dbo:director`, and returns
early, skipping decomposition.

The counts are 22/30/13 on 2Wiki, HotpotQA, and MuSiQue. A single guard that
forces decomposition consideration when the question is multi-hop would, in
principle, unblock decomposition for all 65 cases. The recoverable subset is
bounded by sub-question success rates and is not directly measured here.

## Cluster V: over-decomposition

Over-decomposition accounts for 6.5% of failures and appears only in MuSiQue.
It is the opposite failure of under-decomposition. In 14/38 cases, a 2-hop
question is split into 3 or more sub-questions, and a phantom sub-question
introduces retrieval noise that propagates into the final answer.

A further 5 to 7 cases are 4-hop3 questions split into 5 to 6 sub-questions.
The remaining cases are over-decomposed by other patterns that were not
separately enumerated. The fix points to bidirectional decomposition guidance:
produce roughly N sub-questions for an N-hop question, not just a stricter
floor.

## Cluster VI: comparison logic

Comparison-logic errors account for 5.8% of failures. These are apparent
reasoning errors that, on inspection, are date/string-format or
value-multiplicity mismatches. For example, a film with multiple listed release
dates, 1948, 1968, and 1991, compared against a single-date counterpart, 1988,
can cause the comparison prompt to flip the answer. Nationality-equivalence
cases, such as "Greek and Italian" vs. "Greek", show a similar pattern.

A canonical date and nationality mapper applied prior to comparison addresses
these cases.

## Hop-depth as a structural ceiling

MuSiQue shows the joint behavior of failure rate and DBpedia chain connectivity.
Connectivity drops sharply with hop count and the failure rate climbs
accordingly. At 4-hop with intermediate bridge type 4hop2, connectivity reaches
0% and the system fails on 92.6% of questions. The sample is small at n=27.

| Hop type | Fail % | DBpedia connectivity |
|---|---:|---:|
| 2hop | 28.6 | 32.8 |
| 3hop1 | 24.7 | 14.4 |
| 3hop2 | 45.2 | 1.4 |
| 4hop1 | 55.6 | 2.8 |
| 4hop2 | **92.6** | 0.0 |
| 4hop3 | 58.1 | 0.0 |

Past two hops, DBpedia connectivity collapses on these benchmarks. Prompt-level
changes are unlikely to recover this regime without KG expansion or stronger
external retrieval.

## What is not the bottleneck

Entity linking and final-synthesis reasoning are not the main bottlenecks, even
though both could plausibly be. Of the 8 HotpotQA Cluster VII failures, none
stem from entity linking; all are null sub-question outputs from downstream LLM
failure. Entity linking is also absent from the top causes in 2Wiki and MuSiQue.
The Spotlight + SPARQL fallback chain does not appear among the top failure
causes.

Final-synthesis reasoning is also a minor contributor: only 7/82 MuSiQue
Cluster II cases are true synthesis errors.

## Caveats

All counts use gemma3:12b as judge, which is more lenient than strict EM. A
stricter judge would shift Cluster II and VI counts upward. The 4hop2 regime
(n=27) is small, so the 92.6% figure should be treated as indicative rather
than conclusive.

One untracked failure mode, final-synthesis hallucination overriding a literal
`Unknown` sub-question answer with parametric memory, is currently absorbed into
Cluster II. At least 1 case is confirmed, and up to 16 MuSiQue cases involve
sub-questions returning literal `Unknown` where override is plausible but
unverified.

## Summary

Four findings drive future work, paired by what they implicate. The KG draws a
boundary: 40% of failures arise when the KG forces a text fallback that fails
for label, retrieval, or chain-disconnect reasons; and on MuSiQue, hop depth
dictates outcome through DBpedia connectivity, defining a ceiling that
prompt-level changes alone cannot lift without KG expansion.

Routing leaks inside that boundary: decomposition errors are bidirectional, with
both under- and over-decomposition observable; and routing bypass on multi-hop
questions is the cheapest universal fix, covering 65 cases with a single guard.
