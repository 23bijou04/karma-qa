"""SchemaProjection: per entity x question filtered schema with consumer rendering.

Given an EntityProfile and QuestionAnalysis, produces a scored projection
that each consumer (routing, SPARQL gen, TEXT) can render at appropriate verbosity.
Pure computation — no SPARQL or LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dbpedia_qa.dbpedia.schema_discovery import CACHED_CLASS_SCHEMAS, format_2hop_lines

if TYPE_CHECKING:
    from entity_profile import EntityProfile
    from question_analysis import QuestionAnalysis


@dataclass
class SchemaProjection:
    """Per entity x question schema projection."""

    entity_profile: EntityProfile
    question_analysis: QuestionAnalysis
    primary_class: str | None
    relevant_properties: list[str]  # ["dbo:director", "dbo:releaseDate", ...]
    relevant_chains: dict[str, tuple[str, list[str]]]  # filtered 2-hop chains
    completeness: str  # "complete" | "partial" | "missing"
    missing_hops: list[str]  # ["hop1:director"] if chain broken
    completeness_detail: dict[str, bool] = field(default_factory=dict)

    def render_for_routing(self) -> str:
        """~150 token summary for routing decision."""
        profile = self.entity_profile
        lines = [
            f"Entity: {profile.label} | Class: {self.primary_class or 'unknown'}"
            f" | Completeness: {self.completeness}",
        ]
        props = self.relevant_properties[:5]
        if props:
            lines.append(f"Props: {', '.join(props)}")
        if self.relevant_chains:
            chain_strs = []
            for prop, (target, tprops) in list(self.relevant_chains.items())[:2]:
                chain_strs.append(f"dbo:{prop} -> {target}: {', '.join(tprops[:3])}")
            lines.append(f"Chains: {'; '.join(chain_strs)}")
        if self.missing_hops:
            lines.append(f"Missing: {', '.join(self.missing_hops)}")
        return "\n".join(lines)

    def render_for_sparql(self) -> str:
        """~300 token detail for SPARQL generation."""
        profile = self.entity_profile
        lines = [
            "SPARQL Prefixes:",
            "  PREFIX dbr: <http://dbpedia.org/resource/>",
            "  PREFIX dbo: <http://dbpedia.org/ontology/>",
            "  PREFIX dbp: <http://dbpedia.org/property/>",
            "  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "",
            f"Entity: {profile.label}",
            f"URI: <{profile.uri}>",
        ]
        if self.primary_class:
            lines.append(f"Class: dbo:{self.primary_class}")

        # Cached schema properties (filtered to primary class)
        if self.primary_class and self.primary_class in CACHED_CLASS_SCHEMAS:
            cached = CACHED_CLASS_SCHEMAS[self.primary_class]
            if cached.get("dbo"):
                lines.append(f"  dbo: {', '.join(cached['dbo'][:10])}")
            if cached.get("dbp"):
                lines.append(f"  dbp: {', '.join(cached['dbp'][:10])}")
            if cached.get("descriptions"):
                lines.append("  Property hints:")
                for prop, desc in list(cached["descriptions"].items())[:5]:
                    lines.append(f"    - {prop}: {desc}")
            # Only include relevant chains
            if self.relevant_chains:
                chain_lines = format_2hop_lines(
                    {"dbo_2hop": self.relevant_chains}, max_chains=5, max_props=4,
                )
                lines.extend(chain_lines)

        # Actual property values (question-relevant first, then others up to 10)
        actual = profile.actual_properties
        values = actual.get("values", {})
        if values:
            lines.append("")
            lines.append("Actual properties found:")
            shown = 0
            for prop in actual.get("dbo", [])[:8]:
                val = values.get(f"dbo:{prop}", "")
                lines.append(f"  dbo:{prop} -> {val}")
                shown += 1
            for prop in actual.get("dbp", [])[:max(2, 10 - shown)]:
                val = values.get(f"dbp:{prop}", "")
                lines.append(f"  dbp:{prop} -> {val}")

        return "\n".join(lines)

    def render_for_text(self) -> str:
        """~50 token hint for TEXT route entity discovery."""
        analysis = self.question_analysis
        lines = []
        if analysis.expected_target_type:
            bridge = f" via {analysis.bridge_property}" if analysis.bridge_property else ""
            lines.append(f"Expect: {analysis.expected_target_type}{bridge}")
            lines.append(
                f"Discovery: wikiPageWikiLink + type filter "
                f"dbo:{analysis.expected_target_type}"
            )
        else:
            lines.append("Expect: unknown type")
            lines.append("Discovery: wikiPageWikiLink (unfiltered)")
        return "\n".join(lines)


def _score_class(cls_name: str, analysis: QuestionAnalysis) -> int:
    """Score a cached class by relevance to question.

    3 = class contains target_property
    2 = class contains bridge_property
    1 = class contains hop2 property
    0 = no match
    """
    schema = CACHED_CLASS_SCHEMAS.get(cls_name, {})
    all_props = set(schema.get("dbo", [])) | set(schema.get("dbp", []))

    target_names = {p.split(":")[-1] for p in analysis.target_properties}
    hop2_names = {p.split(":")[-1] for p in (analysis.hop2_properties or set())}

    score = 0
    if target_names & all_props:
        score = max(score, 3)
    if analysis.bridge_property and analysis.bridge_property in all_props:
        score = max(score, 2)
    if hop2_names & all_props:
        score = max(score, 1)
    return score


def project(
    profile: EntityProfile,
    analysis: QuestionAnalysis,
) -> SchemaProjection:
    """Compute question-aware schema projection. Pure computation."""
    # 1. Class selection: score each cached class, pick highest
    class_scores = [(cls, _score_class(cls, analysis)) for cls in profile.cached_classes]
    if class_scores:
        # Tie-break: prefer more specific (longer name = more specific heuristic)
        class_scores.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        primary_class = class_scores[0][0]
    else:
        primary_class = None

    # 2. Relevant properties: analysis targets intersected with actual
    relevant = sorted(analysis.target_properties & profile.actual_property_set)
    # Also add hop2 properties if present
    if analysis.hop2_properties:
        relevant.extend(sorted(analysis.hop2_properties & profile.actual_property_set))
    relevant = list(dict.fromkeys(relevant))  # deduplicate preserving order

    # 3. Chain filtering: keep chains whose entry property matches bridge or target
    target_names = {p.split(":")[-1] for p in analysis.target_properties}
    relevant_chains: dict[str, tuple[str, list[str]]] = {}
    for prop, chain_data in profile.all_2hop_chains.items():
        if prop == analysis.bridge_property or prop in target_names:
            relevant_chains[prop] = chain_data

    # 4. Completeness: prefix-independent check
    needed_names = {p.split(":")[-1] for p in analysis.target_properties}
    have_names = {p.split(":")[-1] for p in profile.actual_property_set}
    detail = {name: name in have_names for name in needed_names}

    if analysis.hop_structure == "bridge" and analysis.bridge_property:
        hop1_ok = analysis.bridge_property in have_names
        if not hop1_ok:
            completeness = "missing"
            missing_hops = [f"hop1:{analysis.bridge_property}"]
        elif needed_names <= have_names:
            completeness = "complete"
            missing_hops = []
        else:
            completeness = "partial"
            missing_hops = [f"hop1:{n}" for n in needed_names - have_names]
    else:
        if not needed_names:
            completeness = "complete"
            missing_hops = []
        elif needed_names <= have_names:
            completeness = "complete"
            missing_hops = []
        elif needed_names & have_names:
            completeness = "partial"
            missing_hops = [n for n in sorted(needed_names) if n not in have_names]
        else:
            completeness = "missing"
            missing_hops = sorted(needed_names)

    return SchemaProjection(
        entity_profile=profile,
        question_analysis=analysis,
        primary_class=primary_class,
        relevant_properties=relevant,
        relevant_chains=relevant_chains,
        completeness=completeness,
        missing_hops=missing_hops,
        completeness_detail=detail,
    )
