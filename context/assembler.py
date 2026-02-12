"""Context assembly for LLM prompts using Wikipedia text."""

import logging
from typing import Optional

from ..models import Triple, Subgraph, QuestionType, Entity
from ..wikipedia.client import WikipediaClient

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Assembles context from Wikipedia text based on KG traversal results.

    The KG traversal identifies relevant entities and relations, but the actual
    context is the original Wikipedia text that contains those facts.
    This matches the 2WikiMultiHopQA benchmark format.
    """

    def __init__(
        self,
        wikipedia_client: Optional[WikipediaClient] = None,
        max_tokens: int = 4000,
        max_paragraphs: int = 10,
        sentences_per_entity: int = 10,
    ):
        """Initialize context assembler.

        Args:
            wikipedia_client: Wikipedia API client
            max_tokens: Maximum context tokens
            max_paragraphs: Maximum paragraphs to include
            sentences_per_entity: Max sentences to fetch per entity
        """
        self.wiki_client = wikipedia_client or WikipediaClient()
        self.max_tokens = max_tokens
        self.max_paragraphs = max_paragraphs
        self.sentences_per_entity = sentences_per_entity
        self._tokenizer = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.encoding_for_model("gpt-4")
            except ImportError:
                return len(text) // 4
        return len(self._tokenizer.encode(text))

    def assemble(
        self,
        question: str,
        subgraph: Subgraph,
        question_type: QuestionType,
    ) -> tuple[str, list[dict]]:
        """Assemble context from Wikipedia text for entities found in KG.

        Args:
            question: The question text
            subgraph: Subgraph from KG traversal (tells us which entities)
            question_type: Type of question

        Returns:
            Tuple of (formatted context string, list of paragraph dicts)
        """
        # Get Wikipedia paragraphs for all entities in the subgraph
        paragraphs = self._get_wikipedia_paragraphs(subgraph.entities)

        # Format context
        context = self._format_paragraphs(paragraphs)

        return context, paragraphs

    def _get_wikipedia_paragraphs(
        self,
        entities: list[Entity],
    ) -> list[dict]:
        """Retrieve Wikipedia paragraphs for entities.

        Args:
            entities: List of entities from KG traversal

        Returns:
            List of paragraph dicts with 'title' and 'sentences'
        """
        paragraphs = []
        seen_titles = set()
        current_tokens = 0

        for entity in entities:
            if len(paragraphs) >= self.max_paragraphs:
                break

            # Convert DBpedia URI to Wikipedia title
            title = self._uri_to_title(entity.uri)

            if title in seen_titles:
                continue
            seen_titles.add(title)

            # Get Wikipedia sentences
            sentences = self.wiki_client.get_sentences(title)

            if sentences:
                # Limit sentences per entity
                limited_sentences = sentences[:self.sentences_per_entity]

                # Check token budget
                para_text = " ".join(limited_sentences)
                para_tokens = self._count_tokens(para_text)

                if current_tokens + para_tokens <= self.max_tokens:
                    paragraphs.append({
                        "title": title,
                        "sentences": limited_sentences,
                    })
                    current_tokens += para_tokens

        return paragraphs

    def _format_paragraphs(self, paragraphs: list[dict]) -> str:
        """Format paragraphs as context string.

        Matches 2WikiMultiHopQA format with numbered sentences.

        Args:
            paragraphs: List of paragraph dicts

        Returns:
            Formatted context string
        """
        parts = []

        for para in paragraphs:
            title = para["title"]
            sentences = para["sentences"]

            parts.append(f"Paragraph: {title}")
            for i, sent in enumerate(sentences):
                parts.append(f"[{i}] {sent}")
            parts.append("")  # Empty line between paragraphs

        return "\n".join(parts)

    def assemble_for_comparison(
        self,
        question: str,
        entity1: Entity,
        entity2: Entity,
    ) -> tuple[str, list[dict]]:
        """Assemble context for comparison questions.

        Gets Wikipedia text for both entities being compared.

        Args:
            question: The question text
            entity1: First entity
            entity2: Second entity

        Returns:
            Tuple of (context string, paragraphs list)
        """
        paragraphs = []

        for entity in [entity1, entity2]:
            title = self._uri_to_title(entity.uri)
            sentences = self.wiki_client.get_sentences(title)

            if sentences:
                paragraphs.append({
                    "title": title,
                    "sentences": sentences[:self.sentences_per_entity],
                })

        context = self._format_paragraphs(paragraphs)
        return context, paragraphs

    def assemble_for_bridge(
        self,
        question: str,
        main_entity: Entity,
        bridge_entity: Entity,
    ) -> tuple[str, list[dict]]:
        """Assemble context for bridge/compositional questions.

        Gets Wikipedia text for main entity and bridge entity.

        Args:
            question: The question text
            main_entity: The entity mentioned in question
            bridge_entity: The intermediate entity found via KG

        Returns:
            Tuple of (context string, paragraphs list)
        """
        paragraphs = []

        for entity in [main_entity, bridge_entity]:
            title = self._uri_to_title(entity.uri)
            sentences = self.wiki_client.get_sentences(title)

            if sentences:
                paragraphs.append({
                    "title": title,
                    "sentences": sentences[:self.sentences_per_entity],
                })

        context = self._format_paragraphs(paragraphs)
        return context, paragraphs

    def assemble_for_bridge_comparison(
        self,
        question: str,
        entity1: Entity,
        entity2: Entity,
        bridge1: Entity,
        bridge2: Entity,
    ) -> tuple[str, list[dict]]:
        """Assemble context for bridge-comparison questions.

        Gets Wikipedia text for all 4 entities (2 main + 2 bridge).

        Args:
            question: The question text
            entity1: First main entity
            entity2: Second main entity
            bridge1: Bridge entity for first
            bridge2: Bridge entity for second

        Returns:
            Tuple of (context string, paragraphs list)
        """
        paragraphs = []

        for entity in [entity1, bridge1, entity2, bridge2]:
            title = self._uri_to_title(entity.uri)
            sentences = self.wiki_client.get_sentences(title)

            if sentences:
                paragraphs.append({
                    "title": title,
                    "sentences": sentences[:self.sentences_per_entity],
                })

        context = self._format_paragraphs(paragraphs)
        return context, paragraphs

    def find_supporting_sentence_indices(
        self,
        paragraphs: list[dict],
        triples: list[Triple],
    ) -> list[tuple[str, int]]:
        """Find sentence indices that support the KG triples.

        This helps identify which sentences in the Wikipedia text
        contain the facts we found in the KG.

        Args:
            paragraphs: Wikipedia paragraphs
            triples: Triples from KG traversal

        Returns:
            List of (title, sentence_index) tuples
        """
        supporting = []

        for triple in triples:
            # Get object value to search for
            obj_value = triple.obj_label or self._uri_to_title(triple.obj)
            obj_lower = obj_value.lower()

            for para in paragraphs:
                for i, sent in enumerate(para["sentences"]):
                    sent_lower = sent.lower()

                    # Check if sentence contains the object value
                    if obj_lower in sent_lower:
                        supporting.append((para["title"], i))

        return list(set(supporting))

    def _uri_to_title(self, uri: str) -> str:
        """Convert DBpedia URI to Wikipedia title.

        Args:
            uri: DBpedia resource URI

        Returns:
            Wikipedia article title
        """
        if "dbpedia.org/resource/" in uri:
            title = uri.split("/resource/")[-1]
        elif "/" in uri:
            title = uri.split("/")[-1]
        else:
            title = uri

        return title.replace("_", " ")

    def format_for_prompt(
        self,
        question: str,
        context: str,
        question_type: QuestionType,
    ) -> str:
        """Format question and Wikipedia context for LLM prompt.

        Args:
            question: The question text
            context: Assembled Wikipedia context
            question_type: Type of question

        Returns:
            Formatted prompt string
        """
        type_instructions = {
            QuestionType.COMPARISON: (
                "Compare the relevant information from both paragraphs "
                "to determine which entity matches the question criteria."
            ),
            QuestionType.INFERENCE: (
                "Follow the chain of relationships across paragraphs. "
                "For example, to find a grandfather, first find the father, "
                "then find that person's father."
            ),
            QuestionType.COMPOSITIONAL: (
                "First identify the intermediate entity from the first paragraph, "
                "then find the answer in the paragraph about that entity."
            ),
            QuestionType.BRIDGE_COMPARISON: (
                "First find the bridge entities (e.g., directors) for each subject, "
                "then compare the relevant properties of those bridge entities."
            ),
        }

        instruction = type_instructions.get(
            question_type,
            "Use the information in the paragraphs to answer the question."
        )

        return f"""Question: {question}

{context}

Instructions: {instruction}

Based on the paragraphs above, answer the question.
Also identify which sentences support your answer (format: "Title: sentence_number").

Answer:"""
