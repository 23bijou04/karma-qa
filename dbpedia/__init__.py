"""DBpedia integration module."""

from .client import DBpediaSPARQLClient
from .entity_linker import DBpediaEntityLinker
from .property_mapper import PropertyMapper
from .cache import CacheManager

__all__ = ["DBpediaSPARQLClient", "DBpediaEntityLinker", "PropertyMapper", "CacheManager"]
