# Table 2 Normalized Operator Trace Examples

These examples are normalized operator traces: they preserve the routed operator sequence used for explanation, but raw route metadata can differ across runs because entity linking and ASK verification are live service calls.

## Decomp: all SPARQL

Question: "Who was born first out of Marco De Luigi and Fabio Maistro?"

| Step | Operator call | Output / state |
|---:|---|---|
| 1 | `EntityLink(Q)` | `E_Q = {dbr:Marco_de_Luigi, dbr:Fabio_Maistro}` |
| 2 | `SchemaProject(E_Q,Q)` | For each entity, `A_Q`: class=`dbo:Person`; `P_Q`: `{dbo:birthDate, ...}` |
| 3 | `Covers(S_Q,Q)` | normalized predicate evidence=`{born -> dbo:birthDate for Marco/Fabio}`; stored run branch=`DECOMPOSE` because noisy whole-question entity coverage prevented the direct comparison route; note that clean coverage can route this question directly as `rule:direct_date_comparison` |
| 4 | `Decompose(Q)` | `D=({q_1,q_2}, empty)`, ordered as `q_1,q_2`; `q_1="When was Marco De Luigi born?"`; `q_2="When was Fabio Maistro born?"`; `M=empty` |
| 5 | `Ground(q_1,M)` | `q_hat_1="When was Marco De Luigi born?"` |
| 6 | `EntityLink(q_hat_1)` | `E_{q_1}={dbr:Marco_de_Luigi}` |
| 7 | `SchemaProject(E_{q_1},q_hat_1)` | `A_{q_1}`: class=`dbo:Person`; `P_{q_1}`: `{dbo:birthDate, ...}` |
| 8 | `Covers(S_{q_1},q_hat_1)` | `true`; `covered={born -> dbo:birthDate}`; `uncovered=empty`; next branch=`SPARQL` |
| 9 | `Text2SPARQL(q_hat_1,S_{q_1},G)` | Returns `1978-03-21`; `M[q_1]=March 21, 1978` |
| 10 | `Ground(q_2,M)` | `q_hat_2="When was Fabio Maistro born?"` |
| 11 | `EntityLink(q_hat_2)` | `E_{q_2}={dbr:Fabio_Maistro}` |
| 12 | `SchemaProject(E_{q_2},q_hat_2)` | `A_{q_2}`: class=`dbo:Person`; `P_{q_2}`: `{dbo:birthDate, ...}` |
| 13 | `Covers(S_{q_2},q_hat_2)` | `true`; `covered={born -> dbo:birthDate}`; `uncovered=empty`; next branch=`SPARQL` |
| 14 | `Text2SPARQL(q_hat_2,S_{q_2},G)` | Returns `1998-04-05`; `M[q_2]=April 5, 1998` |
| 15 | `Synthesize(Q,M)` | Compares `M[q_1]` and `M[q_2]`; answer=`Marco De Luigi` |

SPARQL for `q_1`:

```sparql
SELECT ?birthDate
WHERE {
  dbr:Marco_de_Luigi dbo:birthDate ?birthDate .
}
LIMIT 100
```

SPARQL for `q_2`:

```sparql
SELECT ?birthDate
WHERE {
  dbr:Fabio_Maistro dbo:birthDate ?birthDate .
}
LIMIT 100
```

## Decomp: SPARQL + text

Question: "Where was the director of film Age-Old Friends born?"

| Step | Operator call | Output / state |
|---:|---|---|
| 1 | `EntityLink(Q)` | `E_Q={dbr:Age-Old_Friends}` |
| 2 | `SchemaProject(E_Q,Q)` | `A_Q`: class=`dbo:Film`; `P_Q`: `{dbo:director, ...}`; candidate bridge path: `dbo:director -> dbo:birthPlace` |
| 3 | `Covers(S_Q,Q)` | `false`; `covered={director -> dbo:director}`; recognized bridge candidate=`dbo:director -> dbo:birthPlace`; instance ASK result=`ask_missing`; next branch=`DECOMPOSE` |
| 4 | `Decompose(Q)` | `D=({q_1,q_2}, q_1 -> q_2)`; `q_1="Who directed Age-Old Friends?"`; `q_2="Where was [q_1] born?"`; `M=empty` |
| 5 | `Ground(q_1,M)` | `q_hat_1="Who directed Age-Old Friends?"` |
| 6 | `EntityLink(q_hat_1)` | `E_{q_1}={dbr:Age-Old_Friends}` |
| 7 | `SchemaProject(E_{q_1},q_hat_1)` | `A_{q_1}`: class=`dbo:Film`; `P_{q_1}`: `{dbo:director, ...}` |
| 8 | `Covers(S_{q_1},q_hat_1)` | `true`; `covered={directed -> dbo:director}`; `uncovered=empty`; next branch=`SPARQL` |
| 9 | `Text2SPARQL(q_hat_1,S_{q_1},G)` | Returns `dbr:Allan_Kroeker`; `M[q_1]=Allan Kroeker` |
| 10 | `Ground(q_2,M)` | `q_hat_2="Where was Allan Kroeker born?"` |
| 11 | `EntityLink(q_hat_2)` | `E_{q_2}={dbr:Allan_Kroeker}` |
| 12 | `SchemaProject(E_{q_2},q_hat_2)` | `A_{q_2}`: class=`dbo:Person`; `P_{q_2}` does not contain `dbo:birthPlace` |
| 13 | `Covers(S_{q_2},q_hat_2)` | `false`; `covered=empty`; `uncovered={born -> no coverage-compatible dbo:birthPlace}`; next branch=`TEXT` |
| 14 | `TextRAG(q_hat_2,E_{q_2},T)` | Retrieves text saying Allan Kroeker was born in Winnipeg, Manitoba; `M[q_2]=Winnipeg, Manitoba` |
| 15 | `Synthesize(Q,M)` | Answer=`Winnipeg, Manitoba` |

SPARQL for `q_1`:

```sparql
SELECT ?director
WHERE {
  dbr:Age-Old_Friends dbo:director ?director .
}
LIMIT 100
```

## Decomp: text-resolved sub-questions

Question: "Why did John Middleton Murry's wife die?"

| Step | Operator call | Output / state |
|---:|---|---|
| 1 | `EntityLink(Q)` | `E_Q={dbr:John_Middleton_Murry}` |
| 2 | `SchemaProject(E_Q,Q)` | `A_Q`: class=`dbo:Person`; `P_Q` contains neither `dbo:spouse` nor `dbo:deathCause` |
| 3 | `Covers(S_Q,Q)` | `covered=empty`; `uncovered={wife -> no observed dbo:spouse, die -> no observed dbo:deathCause}`; next branch=`DECOMPOSE` |
| 4 | `Decompose(Q)` | `D=({q_1,q_2}, q_1 -> q_2)`; `q_1="Who was John Middleton Murry's wife?"`; `q_2="Why did [q_1] die?"`; `M=empty` |
| 5 | `Ground(q_1,M)` | `q_hat_1="Who was John Middleton Murry's wife?"` |
| 6 | `EntityLink(q_hat_1)` | `E_{q_1}={dbr:John_Middleton_Murry}` |
| 7 | `SchemaProject(E_{q_1},q_hat_1)` | `A_{q_1}`: class=`dbo:Person`; `P_{q_1}` does not contain `dbo:spouse` |
| 8 | `Covers(S_{q_1},q_hat_1)` | `false`; `covered=empty`; `uncovered={wife -> no coverage-compatible dbo:spouse}`; next branch=`TEXT` |
| 9 | `TextRAG(q_hat_1,E_{q_1},T)` | Retrieves text identifying Katherine Mansfield as Murry's wife; `M[q_1]=Katherine Mansfield` |
| 10 | `Ground(q_2,M)` | `q_hat_2="Why did Katherine Mansfield die?"` |
| 11 | `EntityLink(q_hat_2)` | `E_{q_2}={dbr:Katherine_Mansfield}` |
| 12 | `SchemaProject(E_{q_2},q_hat_2)` | `A_{q_2}`: class=`dbo:Person`; `P_{q_2}` does not contain `dbo:deathCause` |
| 13 | `Covers(S_{q_2},q_hat_2)` | `false`; `covered=empty`; `uncovered={die -> no coverage-compatible dbo:deathCause}`; next branch=`TEXT` |
| 14 | `TextRAG(q_hat_2,E_{q_2},T)` | Retrieves text saying Mansfield died of tuberculosis; `M[q_2]=tuberculosis` |
| 15 | `Synthesize(Q,M)` | Answer=`Tuberculosis` |
