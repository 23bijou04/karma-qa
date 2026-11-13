# rag/config.py

MAX_PAGES = 20

# Unité à aligner avec ton implémentation de TextChunker.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_TOP_K = 25
CONTEXT_TOP_K = 20

LLM_MODEL = "gemma3:4b"
LLM_TEMPERATURE = 0
OLLAMA_BASE_URL = "http://localhost:11434"