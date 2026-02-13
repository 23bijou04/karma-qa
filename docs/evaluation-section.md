# 5. Evaluation

A key finding of our coverage analysis (Section 4) is that only 31.1% of 2WikiMultiHopQA questions are fully answerable via SPARQL against DBpedia --- the remaining ~69% lack at least one required property in the knowledge graph. This dichotomy motivates a two-mode architecture: a *structured SPARQL mode* for the answerable subset, and a *GraphRAG mode* that augments KG entities with Wikipedia text retrieval for the remainder. In this section, we evaluate the structured SPARQL mode by comparing two text-to-SPARQL approaches on a representative sample of the answerable questions, establishing the accuracy ceiling for the KG-only path of our pipeline.

## 5.1 Experimental Setup

### Dataset and Evaluation Subset

We evaluate on the 2WikiMultiHopQA benchmark (Ho et al., 2020), a multi-hop question answering dataset constructed from Wikipedia with four question types: *bridge_comparison*, *comparison*, *compositional*, and *inference*. The dataset comprises three splits: train (167,454 questions), dev (12,576), and test (12,576). All experiments in this work use the **dev split** exclusively; the test split answers are withheld by the benchmark authors for leaderboard evaluation, and the train split is not used as our approach requires no fine-tuning.

Our prior coverage analysis on the full dev set (12,576 questions) established that DBpedia achieves 92.5% entity coverage but only 60.1% per-pair property coverage, yielding a full answerability rate of 31.1% (3,906 questions). Property coverage is the primary bottleneck: while most entities exist in DBpedia, many lack the specific properties required by the benchmark (e.g., `country_of_citizenship` at 7.7%, `publication_date` at 11.0%). Answerability varies by question type, ranging from 53.7% for bridge_comparison (where `director` and `birthDate` properties are well-populated) down to 22.2% for compositional questions (which require longer chains of properties, increasing the probability that at least one is missing).

From this answerable subset, we drew a stratified random sample of 800 questions, then retained the 276 for which all ground-truth evidence triples were confirmed to exist in DBpedia via live SPARQL verification. These 276 questions are representative of the 31.1% that a SPARQL-only pipeline can address; the remaining ~69% are candidates for our GraphRAG fallback mode, which retrieves Wikipedia article text for linked entities and uses an LLM to reason over the combined KG + text evidence (see Section 6). Entity URIs are pre-resolved from the coverage analysis and provided to both systems, ensuring that differences in entity linking do not confound the comparison.

The distribution by question type is: bridge_comparison (n=111), inference (n=63), compositional (n=53), and comparison (n=49).

### Systems Under Comparison

**GRASP (Baseline).** GRASP is a text-to-SPARQL system developed at the University of Freiburg that employs an agentic architecture for question answering over knowledge graphs. Given a natural language question, GRASP autonomously explores the KG schema through multiple tool calls --- inspecting classes, properties, and example values --- before generating a SPARQL query. The system is accessed via a WebSocket API and uses OpenAI gpt-5-mini as its backend language model. On average, GRASP issues 8.8 tool calls per question (range: 4--32) with a mean latency of 61.5 seconds (range: 18.1--289.9 seconds).

**Schema-Guided Direct SPARQL (Our Approach).** Our system pre-discovers the DBpedia schema for each linked entity prior to SPARQL generation. For every entity URI, we retrieve its RDF class (e.g., `dbo:Person`, `dbo:Film`), cached ontology properties for that class, and actual properties with sample values obtained via live SPARQL queries. This schema context --- including property hints and value examples --- is provided to the LLM in a single prompt, which then generates a SPARQL query directly. If the initial query returns an empty result set or produces a syntax error, a self-correction mechanism re-prompts the LLM with the error feedback for one additional attempt. The backend LLM is Groq-hosted gpt-oss-120b, a 120-billion-parameter open-source model.

### Metrics

We report four complementary metrics:

- **Exact Match (EM)**: The predicted answer matches the gold answer after normalization (lowercasing, article removal, punctuation stripping, whitespace collapsing), following the standard 2WikiMultiHopQA protocol.
- **Token-level F1**: Harmonic mean of token-level precision and recall between the normalized prediction and gold answer.
- **Containment**: A binary indicator of whether the normalized gold answer appears as a substring of the normalized prediction, serving as an upper bound on potential accuracy with improved answer extraction.
- **Judge EM**: For predictions that do not achieve exact match, a local LLM judge (Ollama qwen2.5:7b, temperature 0) assesses semantic equivalence. This accounts for valid alternative formulations (e.g., "Newark, New Jersey" vs. "Newark") that strict string matching would reject. Predictions that achieve auto EM are automatically counted as correct under Judge EM.

We designate a question as *answered* if the system produces a non-empty prediction, regardless of correctness.

## 5.2 Main Results

Table 1 presents the overall performance of both systems across all 276 answerable questions.

**Table 1.** Overall results on the 276-question answerable subset.

| Metric | Schema-Guided SPARQL | GRASP | Delta |
|---|:---:|:---:|:---:|
| Judge EM | 82.6% | 75.7% | +6.9 pp |
| EM (auto) | 67.8% | 33.7% | +34.1 pp |
| F1 | 76.9% | 51.9% | +25.0 pp |
| Containment | 77.2% | 90.9% | -13.7 pp |
| Answered | 258/276 (93.5%) | 265/276 (96.0%) | -2.5 pp |
| Errors | 0 | 0 | --- |

The schema-guided approach achieves a Judge EM of 82.6%, outperforming GRASP by 6.9 percentage points. The gap is substantially larger under automatic EM (34.1 pp) and F1 (25.0 pp), reflecting differences in answer formatting rather than semantic accuracy alone. Notably, both systems produce zero runtime errors across all 276 questions, indicating robust execution pipelines. GRASP achieves a marginally higher answer rate (96.0% vs. 93.5%), as its agentic exploration can sometimes produce partial answers even when SPARQL execution fails.

A striking observation is the *containment paradox*: GRASP achieves 90.9% containment, substantially exceeding its 33.7% EM. This indicates that GRASP frequently embeds the correct answer within a verbose explanation (e.g., "Charlotte Spurkeland was born later. DBpedia shows her birth year as 1987...") rather than returning a concise answer string. The schema-guided system exhibits a smaller containment--EM gap (77.2% vs. 67.8%), reflecting its more direct answer extraction from SPARQL result bindings.

## 5.3 Analysis by Question Type

Table 2 reports Judge EM broken down by question type, revealing that the two systems exhibit complementary strengths.

**Table 2.** Judge EM by question type.

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 111 | 82.9% | 88.3% | -5.4 pp |
| comparison | 49 | 93.9% | 85.7% | +8.2 pp |
| compositional | 53 | 69.8% | 32.1% | +37.7 pp |
| inference | 63 | 84.1% | 82.5% | +1.6 pp |

**Compositional questions show the largest gap.** The schema-guided approach achieves 69.8% on compositional questions compared to GRASP's 32.1%, a difference of 37.7 percentage points. Compositional questions require chaining multiple relations (e.g., "What is the birthplace of the director of the film X?"), and the pre-discovered schema provides the LLM with a complete view of available properties and their value types, enabling it to construct multi-hop SPARQL queries in a single generation step. GRASP's iterative exploration, despite averaging 7.8 tool calls on compositional questions, appears to struggle with assembling the complete query path.

**GRASP excels at bridge_comparison.** On bridge_comparison questions, GRASP achieves 88.3% compared to 82.9% for the schema-guided system. These questions require comparing entities across a bridging relation (e.g., "Which film's director was born first, X or Y?"), and GRASP's agentic exploration may better handle the intermediate reasoning steps needed to navigate from the bridge entity to the comparison property.

**Comparison questions favor the schema-guided approach.** With 93.9% vs. 85.7%, the schema-guided system's advantage on comparison questions likely stems from its ability to retrieve parallel property values for two entities and directly compare them within a single SPARQL query, avoiding the sequential exploration overhead of GRASP.

**Inference questions are comparable.** Both systems perform similarly on inference questions (84.1% vs. 82.5%), which typically require deriving an answer from a single factual lookup combined with a reasoning step.

Table 3 presents automatic EM by question type, exposing a severe answer formatting issue in GRASP.

**Table 3.** Auto EM by question type.

| Type | n | Schema-Guided | GRASP | Delta |
|---|:---:|:---:|:---:|:---:|
| bridge_comparison | 111 | 82.9% | 65.8% | +17.1 pp |
| comparison | 49 | 89.8% | 40.8% | +49.0 pp |
| compositional | 53 | 20.8% | 0.0% | +20.8 pp |
| inference | 63 | 63.5% | 0.0% | +63.5 pp |

GRASP achieves 0.0% auto EM on both compositional and inference question types. This is not because it fails to find the correct answer --- its Judge EM on these types is 32.1% and 82.5% respectively --- but because its answers are invariably formatted as explanatory sentences rather than concise values. For inference questions, the Judge EM--to--auto EM gap is 82.5 percentage points, indicating that the vast majority of GRASP's correct answers are expressed in a format incompatible with exact string matching. The schema-guided system also shows a Judge EM--to--auto EM gap on compositional questions (69.8% vs. 20.8%), suggesting that this question type is inherently more prone to generating verbose answers even with a direct SPARQL approach.

## 5.4 Error Analysis

### 5.4.1 SPARQL Generation Failures (Schema-Guided System)

Table 4 characterizes the SPARQL generation behavior of the schema-guided system.

**Table 4.** SPARQL generation analysis for the schema-guided system.

| Metric | Value |
|---|---|
| Self-correction retry rate | 25.0% (69/276) |
| Empty SPARQL results (after retry) | 6.5% (18/276) |
| SPARQL syntax errors (after retry) | 6.2% (17/276) |

The self-correction mechanism is triggered for 25.0% of questions (69 out of 276), indicating that a quarter of initially generated queries produce empty results or syntax errors. After one retry with error feedback, the residual failure rate drops to 6.5% empty results and 6.2% syntax errors. The retry mechanism thus recovers a substantial fraction of initially failed queries.

Failure rates vary by question type. Bridge_comparison questions exhibit the highest empty-result rate (10.8%), likely because the bridging entity's required property is sometimes expressed using alternative predicates not captured in the schema hint. Comparison questions have the lowest empty-result rate (2.0%), consistent with their simpler query structure.

### 5.4.2 GRASP Tool Usage Patterns

Table 5 summarizes GRASP's tool call behavior.

**Table 5.** GRASP tool call and latency statistics.

| Metric | Value |
|---|---|
| Average tool calls per question | 8.8 |
| Tool call range | 4--32 |
| Average latency per question | 61.5s |
| Latency range | 18.1s--289.9s |
| Highest avg. tool calls (by type) | inference (9.5) |
| Lowest avg. tool calls (by type) | compositional (7.8) |

An unexpected finding is that compositional questions, where GRASP performs worst (32.1% Judge EM), require the fewest average tool calls (7.8). One might expect that more complex questions would trigger more exploration, but the result suggests that GRASP may be terminating its exploration prematurely on these questions rather than exhaustively searching for the required multi-hop path. In contrast, inference questions trigger the most tool calls (9.5 on average) and achieve a high Judge EM (82.5%), suggesting that GRASP's exploration strategy is well-suited to questions requiring a single factual lookup augmented by reasoning.

### 5.4.3 Answer Coverage

Both systems fail to produce answers for a small fraction of questions, though the failure distributions differ.

The schema-guided system leaves 18 questions unanswered (6.5%), concentrated in bridge_comparison (12/111, 10.8%) and inference (4/63, 6.3%), with comparison (1/49, 2.0%) and compositional (1/53, 1.9%) rarely failing to produce output. These unanswered questions correspond to cases where the SPARQL query returns no results even after retry, and the system does not attempt to generate an answer without structured evidence.

GRASP leaves 11 questions unanswered (4.0%), more evenly distributed across types: comparison (3/49, 6.1%), inference (4/63, 6.3%), bridge_comparison (3/111, 2.7%), and compositional (1/53, 1.9%). GRASP's higher answer rate reflects its ability to generate explanatory text even when SPARQL execution fails, leveraging its agentic reasoning to provide a best-effort response.

## 5.5 Discussion

### Structured Schema vs. Agentic Exploration

The central finding of this evaluation is that pre-discovering the KG schema and presenting it to the LLM in a single prompt yields higher accuracy than an agentic approach that explores the schema interactively. The schema-guided system achieves this despite using an open-source 120B-parameter model (gpt-oss-120b) rather than the proprietary gpt-5-mini employed by GRASP. We attribute this advantage to three factors. First, the schema-guided prompt provides a complete view of available properties, eliminating the risk of incomplete exploration. Second, single-shot generation avoids the compounding errors inherent in multi-step tool-use chains, where each intermediate decision can introduce noise. Third, the self-correction mechanism provides a targeted recovery path for the most common failure mode (empty results), without the overhead of a full re-exploration.

However, GRASP's advantage on bridge_comparison questions (88.3% vs. 82.9%) suggests that agentic exploration retains value for questions requiring navigating intermediate entities. When the answer path involves traversing a chain of entities that may be connected by unexpected predicates, GRASP's ability to inspect actual KG content at each hop provides robustness that a pre-assembled schema may lack.

### The Answer Formatting Gap

The large discrepancy between Judge EM and auto EM for GRASP (42.0 pp gap overall vs. 14.8 pp for the schema-guided system) reveals a systematic answer formatting issue rather than a knowledge retrieval deficit. GRASP's agentic architecture, which reasons through multiple tool calls and generates explanatory text, naturally produces verbose answers. This verbosity preserves the correct information (as evidenced by 90.9% containment) but fails to distill it into the concise format expected by exact match evaluation. Post-hoc answer extraction heuristics (e.g., regular expression-based cleaning, as implemented in our evaluation pipeline) partially address this gap, but the results indicate that such heuristics are insufficient for compositional and inference questions where GRASP's explanations are particularly elaborate.

This finding has broader implications for evaluating agentic QA systems: standard EM/F1 metrics may significantly underestimate the effectiveness of systems that produce natural-language explanations rather than atomic answer strings. The use of an LLM judge, as employed here, partially mitigates this issue, though it introduces its own potential for bias.

### Implications for the Full Pipeline

These results establish the accuracy of the SPARQL-only path in our two-mode architecture. For the 31.1% of questions where DBpedia contains all required properties, the schema-guided approach achieves 82.6% Judge EM --- a strong accuracy ceiling for the structured retrieval mode. However, the remaining ~69% of questions (8,670 out of 12,576) lack at least one required property in DBpedia and cannot be addressed by SPARQL alone, regardless of the generation strategy.

For these questions, our system falls back to a GraphRAG mode: entity URIs discovered during schema exploration are used to retrieve the corresponding Wikipedia articles, which are then provided as textual context alongside any partial KG evidence to an LLM for answer generation. This hybrid approach leverages the fact that while DBpedia's property coverage is incomplete, its entity coverage is high (92.5%), meaning the system can still identify *which* entities are relevant and retrieve their Wikipedia text even when structured properties are unavailable. The GraphRAG mode is evaluated separately in Section 6.

The complementary strengths observed between the schema-guided and GRASP systems also suggest a potential routing strategy within the SPARQL mode itself: for bridge_comparison questions, where GRASP's agentic exploration outperforms by 5.4 pp, a question-type classifier could select the more suitable approach. However, the schema-guided system's consistent advantage on the other three types (and its lower computational cost) makes it the preferred default.

### Limitations

Several limitations should be considered when interpreting these results. First, the two systems use different backend LLMs (gpt-5-mini vs. gpt-oss-120b), precluding a perfectly controlled comparison of the schema-guided vs. agentic architectures. Differences in model capability, instruction following, and answer formatting tendencies may contribute to the observed performance gap. Second, the evaluation is restricted to 276 questions where DBpedia is confirmed to contain all required information, representing only the 31.1% answerable subset of the benchmark. This introduces a selection bias toward questions with well-populated properties (e.g., `director`, `birthDate`). Performance on questions at the boundary of answerability --- where some but not all properties exist --- may differ substantially, and such questions would likely benefit from the GraphRAG fallback. Third, the LLM judge (qwen2.5:7b) may exhibit its own biases in semantic equivalence judgments, particularly for ambiguous or context-dependent answers. Finally, the schema-guided approach relies on pre-resolved entity URIs from the coverage analysis, whereas a fully end-to-end system would need to perform entity linking (e.g., via DBpedia Spotlight) as an additional step, introducing potential errors not reflected in these results.

### Efficiency Considerations

Beyond accuracy, the two systems differ substantially in computational cost. GRASP averages 8.8 API calls per question with a mean latency of 61.5 seconds, totaling approximately 4.7 hours for the full evaluation. The schema-guided approach, while requiring an initial schema discovery phase (typically 2--3 SPARQL queries per entity), generates the final SPARQL query in a single LLM call with one optional retry, resulting in significantly lower per-question latency and API cost. This efficiency advantage becomes particularly relevant for deployment scenarios requiring real-time or near-real-time responses.

## 5.6 Summary

The evaluation demonstrates that a schema-guided approach to SPARQL generation outperforms the agentic exploration strategy of GRASP by 6.9 percentage points in Judge EM (82.6% vs. 75.7%) on the answerable subset of 2WikiMultiHopQA. The advantage is most pronounced on compositional questions (+37.7 pp), where complete schema visibility enables the LLM to construct complex multi-hop queries in a single step. However, GRASP's interactive exploration retains an advantage on bridge_comparison questions (-5.4 pp), where dynamic navigation of intermediate entities proves beneficial. Both systems achieve high accuracy on comparison and inference questions, and neither system produces runtime errors, reflecting mature SPARQL generation pipelines.

Critically, these results apply only to the 31.1% of benchmark questions where DBpedia contains all required structured information. For the remaining ~69%, the SPARQL-only path is insufficient regardless of the generation strategy, as the underlying data simply does not exist in the knowledge graph. This motivates the GraphRAG mode of our pipeline, which leverages the same entity linking infrastructure to retrieve Wikipedia article text as a complementary evidence source. In the following section, we evaluate this text-augmented mode and assess the end-to-end performance of the complete hybrid system across the full answerability spectrum.
