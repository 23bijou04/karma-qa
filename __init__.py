"""DBpedia GraphRAG System for 2WikiMultiHopQA Benchmark.

A complete GraphRAG system that uses DBpedia knowledge graph to answer
multi-hop questions from the 2WikiMultiHopQA benchmark.

Pipeline:
    Question → Classify Type → Decompose → Extract Entities →
    DBpedia Traversal → Get Wikipedia Text → LLM Reasoning → Answer

Example:
    >>> from dbpedia_qa import create_system
    >>> system = create_system()
    >>> answer, facts, evidence = system.answer_question("Who was born first, Einstein or Newton?")
    >>> print(answer.text)
"""

from .config import Config
from .models import Entity, Triple, Subgraph, Question, Answer, Evidence, QuestionType
from .system import DBpediaGraphRAGSystem, create_system
from .sparql_generator import SPARQLGenerator, generate_sparql_for_question

__version__ = "0.1.0"
__all__ = [
    # Config & Models
    "Config",
    "Entity",
    "Triple",
    "Subgraph",
    "Question",
    "Answer",
    "Evidence",
    "QuestionType",
    # Main system
    "DBpediaGraphRAGSystem",
    "create_system",
    # SPARQL generation
    "SPARQLGenerator",
    "generate_sparql_for_question",
]
