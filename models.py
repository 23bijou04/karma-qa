"""Data models for DBpedia GraphRAG system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QuestionType(Enum):
    """Types of multi-hop questions in 2WikiMultiHopQA."""

    COMPARISON = "comparison"
    INFERENCE = "inference"
    COMPOSITIONAL = "compositional"
    BRIDGE_COMPARISON = "bridge_comparison"
    UNKNOWN = "unknown"


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

        # Handle date formatting
        if "date" in pred.lower() or "birth" in pred.lower() or "death" in pred.lower():
            return f"{subj}'s {pred} is {obj_val}."

        return f"{subj} {pred} {obj_val}."

    def to_structured(self) -> str:
        """Convert triple to structured format."""
        return f"({self.subject}, {self.predicate}, {self.obj})"

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
class Subgraph:
    """Represents a subgraph extracted from the knowledge graph."""

    entities: list[Entity] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    seed_entities: list[str] = field(default_factory=list)
    hops: int = 0

    def add_entity(self, entity: Entity) -> None:
        """Add entity if not already present."""
        if entity not in self.entities:
            self.entities.append(entity)

    def add_triple(self, triple: Triple) -> None:
        """Add triple if not already present."""
        if triple not in self.triples:
            self.triples.append(triple)

    def get_entity_by_uri(self, uri: str) -> Optional[Entity]:
        """Find entity by URI."""
        for entity in self.entities:
            if entity.uri == uri or entity.dbpedia_uri == uri:
                return entity
        return None

    def get_triples_for_entity(self, entity_uri: str) -> list[Triple]:
        """Get all triples involving an entity."""
        return [
            t for t in self.triples if t.subject == entity_uri or t.obj == entity_uri
        ]

    def to_text(self, format: str = "natural") -> str:
        """Convert subgraph to text representation."""
        if format == "natural":
            return "\n".join(t.to_natural_language() for t in self.triples)
        elif format == "structured":
            return "\n".join(t.to_structured() for t in self.triples)
        else:
            return "\n".join(f"- {t.to_natural_language()}" for t in self.triples)


@dataclass
class SubQuestion:
    """Represents a decomposed sub-question."""

    text: str
    order: int
    depends_on: list[int] = field(default_factory=list)
    answer: Optional[str] = None
    entities: list[Entity] = field(default_factory=list)


@dataclass
class Question:
    """Represents a question from the benchmark."""

    id: str
    text: str
    question_type: QuestionType = QuestionType.UNKNOWN
    sub_questions: list[SubQuestion] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    context_paragraphs: list[dict] = field(default_factory=list)
    gold_answer: Optional[str] = None
    gold_supporting_facts: list[tuple[str, int]] = field(default_factory=list)
    gold_evidence: list[Triple] = field(default_factory=list)


@dataclass
class Evidence:
    """Represents evidence for an answer."""

    triples: list[Triple] = field(default_factory=list)
    supporting_sentences: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_path: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Convert evidence to text."""
        parts = []
        if self.triples:
            parts.append("Evidence triples:")
            for t in self.triples:
                parts.append(f"  - {t.to_natural_language()}")
        if self.supporting_sentences:
            parts.append("Supporting sentences:")
            for s in self.supporting_sentences:
                parts.append(f"  - {s}")
        return "\n".join(parts)


@dataclass
class Answer:
    """Represents an answer to a question."""

    text: str
    confidence: float = 0.0
    evidence: Optional[Evidence] = None
    reasoning: Optional[str] = None
    answer_type: str = "span"  # span, yes, no, date, entity


@dataclass
class EvaluationResult:
    """Represents evaluation metrics for a single prediction."""

    question_id: str
    question_type: QuestionType
    answer_em: float = 0.0
    answer_f1: float = 0.0
    supporting_facts_em: float = 0.0
    supporting_facts_f1: float = 0.0
    evidence_em: float = 0.0
    evidence_f1: float = 0.0
    joint_em: float = 0.0
    joint_f1: float = 0.0


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    total_questions: int = 0
    results_by_type: dict[QuestionType, list[EvaluationResult]] = field(
        default_factory=dict
    )

    # Aggregate metrics
    answer_em: float = 0.0
    answer_f1: float = 0.0
    supporting_facts_em: float = 0.0
    supporting_facts_f1: float = 0.0
    evidence_em: float = 0.0
    evidence_f1: float = 0.0
    joint_em: float = 0.0
    joint_f1: float = 0.0

    def add_result(self, result: EvaluationResult) -> None:
        """Add a single evaluation result."""
        if result.question_type not in self.results_by_type:
            self.results_by_type[result.question_type] = []
        self.results_by_type[result.question_type].append(result)
        self.total_questions += 1

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics across all results."""
        if self.total_questions == 0:
            return

        all_results = [r for results in self.results_by_type.values() for r in results]

        self.answer_em = sum(r.answer_em for r in all_results) / len(all_results)
        self.answer_f1 = sum(r.answer_f1 for r in all_results) / len(all_results)
        self.supporting_facts_em = sum(r.supporting_facts_em for r in all_results) / len(
            all_results
        )
        self.supporting_facts_f1 = sum(r.supporting_facts_f1 for r in all_results) / len(
            all_results
        )
        self.evidence_em = sum(r.evidence_em for r in all_results) / len(all_results)
        self.evidence_f1 = sum(r.evidence_f1 for r in all_results) / len(all_results)
        self.joint_em = sum(r.joint_em for r in all_results) / len(all_results)
        self.joint_f1 = sum(r.joint_f1 for r in all_results) / len(all_results)
