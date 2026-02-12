"""Lightweight Wikipedia section retriever for QA.

No vector DB required - uses keyword matching to find relevant sections.
"""

import re
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)


class WikipediaSectionRetriever:
    """Retrieve relevant Wikipedia sections for QA without full RAG."""

    # Section keywords for common QA relation types
    RELATION_TO_SECTIONS = {
        "father": ["early life", "family", "personal life", "biography"],
        "mother": ["early life", "family", "personal life", "biography"],
        "spouse": ["personal life", "family", "marriage", "relationships"],
        "birthplace": ["early life", "biography", "birth"],
        "birthdate": ["early life", "biography", "birth"],
        "deathplace": ["death", "later life", "biography"],
        "deathdate": ["death", "later life", "biography"],
        "director": ["production", "development", "filming", "cast"],
        "founder": ["history", "founding", "early history", "origins"],
        "nationality": ["early life", "biography", "citizenship"],
        "education": ["education", "early life", "career"],
        "occupation": ["career", "professional life", "work"],
        "children": ["family", "personal life", "children"],
        "grandfather": ["early life", "family", "ancestry", "parents"],
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "DBpediaQA/1.0"})

    def get_full_article(self, title: str) -> Optional[str]:
        """Fetch full Wikipedia article text."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }

        try:
            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                pages = resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    return page.get("extract", "")
        except Exception as e:
            logger.warning(f"Failed to fetch article {title}: {e}")

        return None

    def parse_sections(self, content: str) -> dict[str, str]:
        """Parse Wikipedia content into sections."""
        sections = {}

        # Split by section headers (== Section ==)
        pattern = r"(={2,})\s*([^=]+?)\s*\1"
        parts = re.split(pattern, content)

        current_section = "introduction"
        current_content = []

        i = 0
        while i < len(parts):
            part = parts[i]

            # Check if this is a section header marker
            if re.match(r"^={2,}$", part):
                # Save previous section
                if current_content:
                    sections[current_section.lower()] = "\n".join(current_content).strip()

                # Get section name (next part)
                if i + 1 < len(parts):
                    current_section = parts[i + 1].strip()
                    current_content = []
                    i += 2
                    continue
            else:
                # Regular content
                if part.strip():
                    current_content.append(part.strip())

            i += 1

        # Save last section
        if current_content:
            sections[current_section.lower()] = "\n".join(current_content).strip()

        return sections

    def get_relevant_sections(
        self,
        title: str,
        relation_hint: Optional[str] = None,
        question: Optional[str] = None,
        max_chars: int = 5000,
    ) -> str:
        """Get relevant Wikipedia sections for a QA query.

        Args:
            title: Wikipedia article title
            relation_hint: Relation type being queried (e.g., "father", "birthplace")
            question: Original question (for keyword extraction)
            max_chars: Maximum characters to return

        Returns:
            Relevant section text
        """
        content = self.get_full_article(title)
        if not content:
            return ""

        sections = self.parse_sections(content)

        if not sections:
            # Fallback: return first N chars
            return content[:max_chars]

        # Always include introduction
        result_parts = []
        if "introduction" in sections:
            result_parts.append(("introduction", sections["introduction"]))

        # Find relevant sections based on relation hint
        target_sections = set()

        if relation_hint:
            hint_lower = relation_hint.lower()
            for rel, section_names in self.RELATION_TO_SECTIONS.items():
                if rel in hint_lower:
                    target_sections.update(section_names)

        # Also extract keywords from question
        if question:
            question_lower = question.lower()
            for rel, section_names in self.RELATION_TO_SECTIONS.items():
                if rel in question_lower:
                    target_sections.update(section_names)

        # If no specific sections identified, use common biographical ones
        if not target_sections:
            target_sections = {"early life", "biography", "career", "personal life"}

        # Find matching sections
        for section_name, section_content in sections.items():
            if section_name == "introduction":
                continue

            # Check if section name matches any target
            for target in target_sections:
                if target in section_name.lower():
                    result_parts.append((section_name, section_content))
                    break

        # Build result, respecting max_chars
        result = []
        total_chars = 0

        for section_name, section_content in result_parts:
            if total_chars + len(section_content) > max_chars:
                # Add truncated content
                remaining = max_chars - total_chars
                if remaining > 100:
                    result.append(f"[{section_name}]\n{section_content[:remaining]}...")
                break

            result.append(f"[{section_name}]\n{section_content}")
            total_chars += len(section_content)

        return "\n\n".join(result)

    def get_content_for_qa(
        self,
        title: str,
        question: str,
        max_chars: int = 5000,
    ) -> dict:
        """Get Wikipedia content optimized for QA.

        Returns structured data with sections and metadata.
        """
        content = self.get_full_article(title)
        if not content:
            return {"title": title, "content": "", "sections": [], "total_chars": 0}

        sections = self.parse_sections(content)

        # Get relevant content
        relevant_text = self.get_relevant_sections(
            title, question=question, max_chars=max_chars
        )

        return {
            "title": title,
            "content": relevant_text,
            "sections": list(sections.keys()),
            "total_chars": len(content),
            "retrieved_chars": len(relevant_text),
        }


# Simple test
if __name__ == "__main__":
    retriever = WikipediaSectionRetriever()

    # Test with Barack Obama
    print("=" * 70)
    print("Testing section retrieval for 'Barack Obama'")
    print("Question: Who is the father of Barack Obama?")
    print("=" * 70)

    result = retriever.get_content_for_qa(
        "Barack Obama",
        question="Who is the father of Barack Obama?",
        max_chars=3000,
    )

    print(f"\nSections available: {result['sections']}")
    print(f"Total article: {result['total_chars']} chars")
    print(f"Retrieved: {result['retrieved_chars']} chars")
    print(f"\n--- CONTENT ---\n{result['content']}")
