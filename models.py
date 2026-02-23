"""Data models for DBpedia Hybrid QA system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Entity:
    """Represents a knowledge graph entity."""

    uri: str
    label: str
    types: list[str] = field(default_factory=list)
    abstract: Optional[str] = None
    confidence: float = 1.0
    source: str = "dbpedia"  # dbpedia, spotlight, sparql

    @property
    def local_name(self) -> str:
        """Extract local name from URI."""
        if "/" in self.uri:
            return self.uri.split("/")[-1].replace("_", " ")
        return self.uri

    @property
    def dbpedia_uri(self) -> str:
        """Return full DBpedia URI."""
        if self.uri.startswith("http"):
            return self.uri
        return f"http://dbpedia.org/resource/{self.uri.replace(' ', '_')}"

    def __hash__(self) -> int:
        return hash(self.uri)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.uri == other.uri


@dataclass
class Triple:
    """Represents a knowledge graph triple (subject, predicate, object)."""

    subject: str
    predicate: str
    obj: str
    subject_label: Optional[str] = None
    predicate_label: Optional[str] = None
    obj_label: Optional[str] = None
    confidence: float = 1.0
    source: Optional[str] = None

    def to_natural_language(self) -> str:
        """Convert triple to natural language sentence."""
        subj = self.subject_label or self.subject.split("/")[-1].replace("_", " ")
        pred = self.predicate_label or self.predicate.split("/")[-1].replace("_", " ")
        obj_val = self.obj_label or self.obj.split("/")[-1].replace("_", " ")

        if "date" in pred.lower() or "birth" in pred.lower() or "death" in pred.lower():
            return f"{subj}'s {pred} is {obj_val}."

        return f"{subj} {pred} {obj_val}."

    def __hash__(self) -> int:
        return hash((self.subject, self.predicate, self.obj))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Triple):
            return False
        return (
            self.subject == other.subject
            and self.predicate == other.predicate
            and self.obj == other.obj
        )


@dataclass
class HybridResult:
    """Result from the hybrid QA pipeline."""

    answer: str
    method: str  # "sparql" or "text"
    route_source: str = ""  # "rule:*", "llm", "no_schema_fallback"
    sparql_fallback: bool = False  # True if routed SPARQL but fell back to text
