"""Wikipedia integration module."""

from .client import WikipediaClient
from .supporting_facts import SupportingFactsFinder

__all__ = ["WikipediaClient", "SupportingFactsFinder"]
