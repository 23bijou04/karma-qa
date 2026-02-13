"""DBpedia integration module."""

from .client import DBpediaSPARQLClient
from .entity_linker import DBpediaEntityLinker
from .cache import CacheManager
from .schema_discovery import SchemaDiscovery

__all__ = [
    "DBpediaSPARQLClient",
    "DBpediaEntityLinker",
    "CacheManager",
    "SchemaDiscovery",
]
