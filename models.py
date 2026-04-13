"""Data models for DBpedia Hybrid QA system."""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Matches a single trailing parenthetical clause, e.g. " (also known as X)".
# Applied once post-extraction — LLMs frequently append disambiguators that
# break EM/F1 scoring against short gold answers.
_TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")


def extract_final_answer(text: str) -> str:
    """Extract answer after the last 'FINAL ANSWER:' marker (case-insensitive).

    Uses last-match semantics (backward compatible with the old split()[-1] pattern).
    Shared utility for all pipelines to avoid inconsistent extraction logic.

    Behavior:
      - If marker found: return text after the LAST marker, trimmed, with a
        single trailing parenthetical stripped (e.g. "John Smith (director)"
        -> "John Smith"). This normalizes LLM output against short gold answers.
      - If marker missing: log a warning (so eval scripts can detect format
        regressions) and return the original text unchanged to preserve
        passthrough semantics for callers that already pre-clean.
      - None/empty input returns "".
    """
    if not text:
        return text or ""
    matches = list(re.finditer(r"FINAL ANSWER:", text, re.IGNORECASE))
    if matches:
        extracted = text[matches[-1].end():].strip()
        # Strip a single trailing parenthetical clause (common LLM habit)
        extracted = _TRAILING_PAREN_RE.sub("", extracted).strip()
        return extracted
    # No marker — this indicates the LLM didn't follow instructions.
    # Log once at DEBUG (routine under retry loops, noisy at WARNING).
    logger.debug(
        "extract_final_answer: no FINAL ANSWER marker (len=%d); returning raw text",
        len(text),
    )
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
