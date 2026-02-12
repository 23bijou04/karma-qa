"""Supporting facts extraction from Wikipedia text."""

import re
import logging
from typing import Optional

from ..models import Triple, Entity
from .client import WikipediaClient

logger = logging.getLogger(__name__)


class SupportingFactsFinder:
    """Finds supporting sentences in Wikipedia text for KG facts.

    Given triples from the knowledge graph, finds sentences in
    Wikipedia articles that support or contain those facts.
    """

    def __init__(self, wikipedia_client: Optional[WikipediaClient] = None):
        """Initialize finder.

        Args:
            wikipedia_client: Wikipedia API client
        """
        self.wiki_client = wikipedia_client or WikipediaClient()

    def find_supporting_sentences(
        self,
        triple: Triple,
        max_sentences: int = 3,
    ) -> list[tuple[str, int]]:
        """Find sentences that support a triple.

        Args:
            triple: The KG triple to find support for
            max_sentences: Maximum sentences to return

        Returns:
            List of (sentence, sentence_index) tuples
        """
        supporting = []

        # Get Wikipedia content for subject entity
        subject_title = self._uri_to_title(triple.subject)
        sentences = self.wiki_client.get_sentences(subject_title)

        if not sentences:
            return supporting

        # Extract search terms from triple
        obj_label = triple.obj_label or self._uri_to_title(triple.obj)
        pred_label = triple.predicate_label or self._extract_predicate_name(
            triple.predicate
        )

        # Search for sentences mentioning the object
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, obj_label, pred_label)
            if score > 0:
                supporting.append((sentence, i, score))

        # Sort by score and return top N
        supporting.sort(key=lambda x: x[2], reverse=True)
        return [(sent, idx) for sent, idx, _ in supporting[:max_sentences]]

    def find_all_supporting_facts(
        self,
        triples: list[Triple],
        entities: list[Entity],
    ) -> dict[str, list[tuple[str, int]]]:
        """Find supporting sentences for multiple triples.

        Args:
            triples: List of KG triples
            entities: List of entities involved

        Returns:
            Dict mapping entity titles to list of (sentence, index) tuples
        """
        supporting_facts = {}

        # Group triples by subject
        triples_by_subject = {}
        for triple in triples:
            subject = triple.subject
            if subject not in triples_by_subject:
                triples_by_subject[subject] = []
            triples_by_subject[subject].append(triple)

        # Find supporting sentences for each subject
        for subject_uri, subject_triples in triples_by_subject.items():
            title = self._uri_to_title(subject_uri)
            sentences = self.wiki_client.get_sentences(title)

            if not sentences:
                continue

            entity_supporting = []

            for triple in subject_triples:
                obj_label = triple.obj_label or self._uri_to_title(triple.obj)
                pred_label = triple.predicate_label or self._extract_predicate_name(
                    triple.predicate
                )

                for i, sentence in enumerate(sentences):
                    score = self._score_sentence(sentence, obj_label, pred_label)
                    if score > 0:
                        entity_supporting.append((sentence, i, score))

            # Deduplicate and sort
            seen = set()
            unique_supporting = []
            for sent, idx, score in sorted(
                entity_supporting, key=lambda x: x[2], reverse=True
            ):
                if idx not in seen:
                    seen.add(idx)
                    unique_supporting.append((sent, idx))

            if unique_supporting:
                supporting_facts[title] = unique_supporting[:5]

        return supporting_facts

    def extract_evidence_sentences(
        self,
        question: str,
        answer: str,
        entities: list[Entity],
        max_per_entity: int = 2,
    ) -> list[str]:
        """Extract sentences that serve as evidence for an answer.

        Args:
            question: The original question
            answer: The predicted answer
            entities: Entities involved in the question
            max_per_entity: Max sentences per entity

        Returns:
            List of evidence sentences
        """
        evidence = []
        question_words = set(question.lower().split())

        for entity in entities:
            title = self._uri_to_title(entity.uri)
            sentences = self.wiki_client.get_sentences(title)

            if not sentences:
                continue

            # Score sentences by relevance to question and answer
            scored = []
            for i, sent in enumerate(sentences):
                score = self._score_evidence_sentence(
                    sent, question_words, answer.lower()
                )
                if score > 0:
                    scored.append((sent, score))

            # Add top sentences
            scored.sort(key=lambda x: x[1], reverse=True)
            for sent, _ in scored[:max_per_entity]:
                if sent not in evidence:
                    evidence.append(sent)

        return evidence

    def get_context_paragraphs(
        self,
        entities: list[Entity],
        max_paragraphs: int = 10,
    ) -> list[dict]:
        """Get context paragraphs for a set of entities.

        Format matches 2WikiMultiHopQA context structure.

        Args:
            entities: List of entities
            max_paragraphs: Maximum total paragraphs

        Returns:
            List of paragraph dicts with 'title' and 'sentences'
        """
        paragraphs = []
        paragraphs_per_entity = max(1, max_paragraphs // len(entities)) if entities else 0

        for entity in entities:
            title = self._uri_to_title(entity.uri)
            sentences = self.wiki_client.get_sentences(title)

            if sentences:
                # Split into "paragraphs" (groups of sentences)
                para_sentences = sentences[:paragraphs_per_entity * 5]

                paragraphs.append({
                    "title": title,
                    "sentences": para_sentences,
                })

                if len(paragraphs) >= max_paragraphs:
                    break

        return paragraphs

    def _score_sentence(
        self, sentence: str, object_label: str, predicate_label: str
    ) -> float:
        """Score how well a sentence supports a triple.

        Args:
            sentence: The sentence to score
            object_label: Expected object value
            predicate_label: Predicate name

        Returns:
            Relevance score
        """
        sent_lower = sentence.lower()
        score = 0.0

        # Check for object mention
        if object_label.lower() in sent_lower:
            score += 0.5

        # Check for predicate-related words
        predicate_words = set(predicate_label.lower().replace("_", " ").split())
        for word in predicate_words:
            if word in sent_lower:
                score += 0.2

        # Bonus for specific patterns
        patterns = {
            "birth": ["born", "birth", "birthday"],
            "death": ["died", "death", "passed away"],
            "father": ["father", "son of", "daughter of"],
            "mother": ["mother", "son of", "daughter of"],
            "spouse": ["married", "wife", "husband", "spouse"],
            "director": ["directed", "director"],
            "founder": ["founded", "founder"],
        }

        for pred_key, keywords in patterns.items():
            if pred_key in predicate_label.lower():
                for kw in keywords:
                    if kw in sent_lower:
                        score += 0.3
                        break

        return score

    def _score_evidence_sentence(
        self, sentence: str, question_words: set, answer: str
    ) -> float:
        """Score sentence as evidence for question-answer pair.

        Args:
            sentence: The sentence to score
            question_words: Set of question words
            answer: The answer text

        Returns:
            Evidence score
        """
        sent_lower = sentence.lower()
        sent_words = set(sent_lower.split())

        score = 0.0

        # Answer mention is most important
        if answer in sent_lower:
            score += 0.5

        # Question word overlap
        overlap = question_words & sent_words
        score += len(overlap) * 0.05

        # Length penalty (prefer shorter, more focused sentences)
        if len(sentence) < 200:
            score += 0.1

        return score

    def _uri_to_title(self, uri: str) -> str:
        """Convert URI to Wikipedia title."""
        if "dbpedia.org/resource/" in uri:
            title = uri.split("/resource/")[-1]
        elif "/" in uri:
            title = uri.split("/")[-1]
        else:
            title = uri

        return title.replace("_", " ")

    def _extract_predicate_name(self, predicate_uri: str) -> str:
        """Extract predicate name from URI."""
        if "/" in predicate_uri:
            name = predicate_uri.split("/")[-1]
        elif "#" in predicate_uri:
            name = predicate_uri.split("#")[-1]
        else:
            name = predicate_uri

        return name.replace("_", " ")
