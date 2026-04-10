"""DBpedia integration module."""

from .cache import CacheManager
from .client import DBpediaSPARQLClient
from .entity_linker import DBpediaEntityLinker
from .schema_discovery import SchemaDiscovery
from .wiki_resolver import WikipediaResolver

__all__ = [
    "DBpediaSPARQLClient",
    "DBpediaEntityLinker",
    "CacheManager",
    "SchemaDiscovery",
    "WikipediaResolver",
]
