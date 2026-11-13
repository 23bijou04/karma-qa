# rag/application/rag_service.py
from rag.application.ports import (
    Chunker,
    DocumentSource,
    LLMProvider,
)
from rag.application.retrieval_service import RetrievalService


class RAGService:
    def __init__(
        self,
        document_source: DocumentSource,
        chunker: Chunker,
        retriever: RetrievalService,
        llm_provider: LLMProvider,
        max_pages: int = 10,
    ):
        if max_pages < 1:
            raise ValueError("max_pages doit être supérieur à 0")

        self._document_source = document_source
        self._chunker = chunker
        self._retriever = retriever
        self._llm_provider = llm_provider
        self._max_pages = max_pages

    def answer(self, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("La question ne peut pas être vide")

        # 1. Rechercher et récupérer les documents.
        urls = self._document_source.search_pages(
            question,
            limit=self._max_pages,
        )
        
        documents = [
            self._document_source.fetch_page(url)
            for url in urls
        ]

        # 2. Découper les documents.
        chunks = []

        for document in documents:
            page_chunks = self._chunker.chunk_page(
                title=document.page_title,
                text=document.text,
                url=document.url,
                revid=(
                    int(document.revision_id)
                    if document.revision_id is not None
                    else None
                ),
            )
            chunks.extend(page_chunks)

        if not chunks:
            return "Aucun contenu trouvé pour répondre à cette question."

        # 3. Déléguer l’indexation, la recherche et le reranking.
        passages = self._retriever.retrieve(question, chunks)

        if not passages:
            return "Aucun passage pertinent trouvé pour cette question."

        # 4. Construire le contexte et générer la réponse.
        context = "\n\n".join(
            f"[{index}] {chunk.page_title}\n"
            f"Source : {chunk.url}\n"
            f"{chunk.text}"
            for index, chunk in enumerate(passages, start=1)
        )

       
        return self._llm_provider.generate(question, context)