# rag/bootstrap.py
from rag import config
from rag.application.rag_service import RAGService
from rag.application.retrieval_service import RetrievalService
from rag.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from rag.infrastructure.faiss_vector_store import FaissVectorStore
from rag.infrastructure.ollama_provider import OllamaProvider
from rag.infrastructure.wiki_chunker import WikiChunker
from rag.infrastructure.wikipedia_search2 import WikipediaSearch


def create_rag() -> RAGService:
    document_source = WikipediaSearch()

    chunker = WikiChunker(
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
    )

    vector_store = FaissVectorStore(
        model_name=config.EMBEDDING_MODEL,
    )

    reranker = CrossEncoderReranker(
        model_name=config.RERANKER_MODEL,
    )

    retriever = RetrievalService(
        vector_store=vector_store,
        reranker=reranker,
        retrieval_top_k=config.RETRIEVAL_TOP_K,
        context_top_k=config.CONTEXT_TOP_K,
    )

    llm_provider = OllamaProvider(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        base_url=config.OLLAMA_BASE_URL,
    )

    return RAGService(
        document_source=document_source,
        chunker=chunker,
        retriever=retriever,
        llm_provider=llm_provider,
        max_pages=config.MAX_PAGES,
    )