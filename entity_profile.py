"""EntityProfile: per-entity schema profile with all classes and properties.

Replaces the fused get_entity_class() + get_actual_properties() + text rendering
with a structured profile that downstream layers can query.
"""

from dataclasses import dataclass, field

from dbpedia_qa.dbpedia.schema_discovery import CACHED_CLASS_SCHEMAS


@dataclass
class EntityProfile:
    """Cached per-entity schema profile."""

    uri: str
    label: str
    all_classes: list[str] = field(default_factory=list)
    cached_classes: list[str] = field(default_factory=list)
    actual_properties: dict = field(default_factory=dict)
    actual_property_set: frozenset[str] = field(default_factory=frozenset)
    all_2hop_chains: dict[str, tuple[str, list[str]]] = field(default_factory=dict)


def build_entity_profile(uri: str, label: str, schema_disc) -> EntityProfile:
    """Build EntityProfile from DBpedia via SchemaDiscovery.

    SPARQL cost: 1 type query + 1 property query (same as current).

    Args:
        uri: Full DBpedia URI (e.g. "http://dbpedia.org/resource/David_Bowie")
        label: Human-readable label
        schema_disc: SchemaDiscovery instance

    Returns:
        EntityProfile with all classes, actual properties, merged 2-hop chains.
    """
    # 1. Get all classes (1 SPARQL query)
    all_classes, cached_classes = schema_disc.get_all_cached_classes(uri)

    # 2. Get actual properties (1 SPARQL query)
    actual = schema_disc.get_actual_properties(uri)

    # 3. Build actual_property_set: frozenset of "prefix:name"
    prop_set = set()
    for name in actual.get("dbo", []):
        prop_set.add(f"dbo:{name}")
    for name in actual.get("dbp", []):
        prop_set.add(f"dbp:{name}")
    actual_property_set = frozenset(prop_set)

    # 4. Merge 2-hop chains from ALL matching cached class schemas
    all_chains: dict[str, tuple[str, list[str]]] = {}
    for cls in cached_classes:
        dbo_2hop = CACHED_CLASS_SCHEMAS[cls].get("dbo_2hop", {})
        for prop, (target, props) in dbo_2hop.items():
            if prop not in all_chains:
                all_chains[prop] = (target, list(props))
            else:
                existing_props = set(all_chains[prop][1])
                existing_props.update(props)
                all_chains[prop] = (target, sorted(existing_props))

    return EntityProfile(
        uri=uri,
        label=label,
        all_classes=all_classes,
        cached_classes=cached_classes,
        actual_properties=actual,
        actual_property_set=actual_property_set,
        all_2hop_chains=all_chains,
    )
