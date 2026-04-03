"""Data models for DBpedia Hybrid QA system."""

import re
from dataclasses import dataclass, field
from typing import Optional


def extract_final_answer(text: str) -> str:
    """Extract answer after the last 'FINAL ANSWER:' marker (case-insensitive).

    Uses last-match semantics (backward compatible with the old split()[-1] pattern).
    Shared utility for all pipelines to avoid inconsistent extraction logic.
    Returns the text after the marker, or the original text if no marker found.
    """
    if not text:
        return text or ""
    # Find all occurrences; take text after the last one
    matches = list(re.finditer(r"FINAL ANSWER:", text, re.IGNORECASE))
    if matches:
        return text[matches[-1].end():].strip()
    return text


@dataclass
class Entity:
    """Represents a knowledge graph entity."""

    uri: str
    label: str
    types: list[str] = field(default_factory=list)
    abstract: Optional[str] = None
    confidence: float = 1.0
    source: str = "dbpedia"  # dbpedia, spotlight, sparql
    discovery_relation: Optional[str] = None  # predicate that discovered this entity

    def __hash__(self) -> int:
        return hash(self.uri)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.uri == other.uri


@dataclass
class HybridResult:
    """Result from the hybrid QA pipeline."""

    answer: str
    method: str  # "sparql", "text", or "logicrag_decomposed"
    route_source: str = ""  # "rule:*[+ask_verified|ask_missing]", "llm_prop:*", "incomplete_coverage", "no_schema_fallback"
    sparql_fallback: bool = False  # True if routed SPARQL but fell back to text
    sparql_query: Optional[str] = None  # executed SPARQL query (provenance)
    kg_context: str = ""  # formatted KG facts injected into text prompt
    sub_question_log: list[dict] = field(default_factory=list)  # per-sub-Q method/query/answer
    debug_info: dict = field(default_factory=dict)
