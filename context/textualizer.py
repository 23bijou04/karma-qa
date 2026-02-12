"""Triple textualization for LLM consumption."""

import re
from typing import Optional

from ..models import Triple


class TripleTextualizer:
    """Converts knowledge graph triples to natural language text.

    Supports multiple output formats for different use cases.
    """

    # Predicate to natural language templates
    PREDICATE_TEMPLATES = {
        "birthDate": "{subject}'s date of birth is {object}.",
        "deathDate": "{subject}'s date of death is {object}.",
        "birthPlace": "{subject} was born in {object}.",
        "deathPlace": "{subject} died in {object}.",
        "father": "{subject}'s father is {object}.",
        "mother": "{subject}'s mother is {object}.",
        "spouse": "{subject} is married to {object}.",
        "child": "{subject} has a child named {object}.",
        "nationality": "{subject} is a citizen of {object}.",
        "occupation": "{subject}'s occupation is {object}.",
        "director": "{subject} was directed by {object}.",
        "producer": "{subject} was produced by {object}.",
        "releaseDate": "{subject} was released on {object}.",
        "foundedBy": "{subject} was founded by {object}.",
        "foundingDate": "{subject} was founded in {object}.",
        "almaMater": "{subject} studied at {object}.",
        "employer": "{subject} works at {object}.",
        "doctoralAdvisor": "{subject}'s doctoral advisor is {object}.",
        "country": "{subject} is located in {object}.",
        "artist": "{subject} was performed by {object}.",
        "author": "{subject} was written by {object}.",
        "composer": "{subject} was composed by {object}.",
    }

    def __init__(
        self,
        include_confidence: bool = True,
        include_source: bool = False,
        default_format: str = "natural",
    ):
        """Initialize textualizer.

        Args:
            include_confidence: Whether to include confidence scores
            include_source: Whether to include source information
            default_format: Default output format
        """
        self.include_confidence = include_confidence
        self.include_source = include_source
        self.default_format = default_format

    def textualize(
        self,
        triple: Triple,
        format: Optional[str] = None,
    ) -> str:
        """Convert triple to text representation.

        Args:
            triple: The triple to textualize
            format: Output format ('natural', 'structured', 'triple', 'verbose')

        Returns:
            Text representation of the triple
        """
        format = format or self.default_format

        if format == "natural":
            return self._to_natural_language(triple)
        elif format == "structured":
            return self._to_structured(triple)
        elif format == "triple":
            return self._to_triple_notation(triple)
        elif format == "verbose":
            return self._to_verbose(triple)
        else:
            return self._to_natural_language(triple)

    def _to_natural_language(self, triple: Triple) -> str:
        """Convert to natural language sentence."""
        # Extract labels or local names
        subject = triple.subject_label or self._extract_label(triple.subject)
        predicate = triple.predicate_label or self._extract_label(triple.predicate)
        obj = triple.obj_label or self._extract_label(triple.obj)

        # Try to find a specific template
        pred_name = self._extract_local_name(triple.predicate)
        template = self.PREDICATE_TEMPLATES.get(pred_name)

        if template:
            sentence = template.format(subject=subject, object=obj)
        else:
            # Generic template
            predicate_text = predicate.replace("_", " ")
            sentence = f"{subject} {predicate_text} {obj}."

        # Add confidence if requested
        if self.include_confidence and triple.confidence < 1.0:
            sentence = f"{sentence} (Confidence: {triple.confidence:.0%})"

        # Add source if requested
        if self.include_source and triple.source:
            sentence = f"{sentence} [Source: {triple.source}]"

        return sentence

    def _to_structured(self, triple: Triple) -> str:
        """Convert to structured key-value format."""
        subject = triple.subject_label or self._extract_label(triple.subject)
        predicate = triple.predicate_label or self._extract_label(triple.predicate)
        obj = triple.obj_label or self._extract_label(triple.obj)

        parts = [
            f"Subject: {subject}",
            f"Predicate: {predicate}",
            f"Object: {obj}",
        ]

        if self.include_confidence:
            parts.append(f"Confidence: {triple.confidence:.2f}")

        if self.include_source and triple.source:
            parts.append(f"Source: {triple.source}")

        return " | ".join(parts)

    def _to_triple_notation(self, triple: Triple) -> str:
        """Convert to triple notation (s, p, o)."""
        subject = triple.subject_label or self._extract_label(triple.subject)
        predicate = triple.predicate_label or self._extract_label(triple.predicate)
        obj = triple.obj_label or self._extract_label(triple.obj)

        return f"({subject}, {predicate}, {obj})"

    def _to_verbose(self, triple: Triple) -> str:
        """Convert to verbose multi-line format."""
        subject = triple.subject_label or self._extract_label(triple.subject)
        predicate = triple.predicate_label or self._extract_label(triple.predicate)
        obj = triple.obj_label or self._extract_label(triple.obj)

        lines = [
            f"Fact:",
            f"  Entity: {subject}",
            f"  Property: {predicate}",
            f"  Value: {obj}",
        ]

        if self.include_confidence:
            lines.append(f"  Confidence: {triple.confidence:.2f}")

        if self.include_source and triple.source:
            lines.append(f"  Source: {triple.source}")

        return "\n".join(lines)

    def textualize_batch(
        self,
        triples: list[Triple],
        format: Optional[str] = None,
        separator: str = "\n",
    ) -> str:
        """Convert multiple triples to text.

        Args:
            triples: List of triples
            format: Output format
            separator: Separator between triples

        Returns:
            Combined text representation
        """
        texts = [self.textualize(t, format) for t in triples]
        return separator.join(texts)

    def textualize_grouped(
        self,
        triples: list[Triple],
        group_by: str = "subject",
        format: Optional[str] = None,
    ) -> str:
        """Convert triples grouped by subject or predicate.

        Args:
            triples: List of triples
            group_by: Grouping key ('subject' or 'predicate')
            format: Output format

        Returns:
            Grouped text representation
        """
        groups = {}

        for triple in triples:
            if group_by == "subject":
                key = triple.subject_label or self._extract_label(triple.subject)
            else:
                key = triple.predicate_label or self._extract_label(triple.predicate)

            if key not in groups:
                groups[key] = []
            groups[key].append(triple)

        lines = []
        for key, group_triples in groups.items():
            lines.append(f"\n{key}:")
            for triple in group_triples:
                text = self.textualize(triple, format)
                lines.append(f"  - {text}")

        return "\n".join(lines)

    def _extract_label(self, uri: str) -> str:
        """Extract human-readable label from URI."""
        local_name = self._extract_local_name(uri)
        return local_name.replace("_", " ")

    def _extract_local_name(self, uri: str) -> str:
        """Extract local name from URI."""
        if "/" in uri:
            return uri.split("/")[-1]
        elif "#" in uri:
            return uri.split("#")[-1]
        return uri

    def format_date(self, date_value: str) -> str:
        """Format date values for readability.

        Args:
            date_value: Raw date string

        Returns:
            Formatted date string
        """
        # Handle common date formats
        patterns = [
            (r"(\d{4})-(\d{2})-(\d{2})", r"\1-\2-\3"),  # ISO format
            (r"(\d{4})", r"\1"),  # Year only
        ]

        for pattern, replacement in patterns:
            if re.match(pattern, date_value):
                return re.sub(pattern, replacement, date_value)

        return date_value
