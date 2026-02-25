"""DBpedia integration module."""

from .cache import CacheManager
from .client import DBpediaSPARQLClient
from .entity_linker import DBpediaEntityLinker
from .schema_discovery import SchemaDiscovery

__all__ = [
    "DBpediaSPARQLClient",
    "DBpediaEntityLinker",
    "CacheManager",
    "SchemaDiscovery",
]
