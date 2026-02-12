"""Response parsing for LLM outputs."""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parses structured responses from LLMs.

    Handles various output formats and extracts answer, reasoning, and evidence.
    """

    def parse(self, response: str) -> dict:
        """Parse a structured LLM response.

        Args:
            response: Raw LLM response text

        Returns:
            Dict with parsed components (answer, reasoning, evidence, confidence)
        """
        result = {
            "answer": "",
            "reasoning": "",
            "evidence": [],
            "supporting_facts": [],
            "confidence": "medium",
        }

        # Try structured format first
        structured = self._parse_structured(response)
        if structured["answer"]:
            return structured

        # Try to extract answer from unstructured response
        result["answer"] = self._extract_answer(response)
        result["reasoning"] = response

        return result

    def _parse_structured(self, response: str) -> dict:
        """Parse structured format with labeled sections.

        Expects format like:
        ANSWER: ...
        REASONING: ...
        EVIDENCE: ...
        """
        result = {
            "answer": "",
            "reasoning": "",
            "evidence": [],
            "supporting_facts": [],
            "confidence": "medium",
        }

        # Pattern for ANSWER field
        answer_patterns = [
            r"ANSWER:\s*(.+?)(?=\n[A-Z]+:|$)",
            r"Answer:\s*(.+?)(?=\n[A-Z]|$)",
            r"The answer is[:\s]+(.+?)(?:\.|$)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                result["answer"] = match.group(1).strip()
                break

        # Pattern for REASONING field
        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?=\n[A-Z]+:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()

        # Pattern for EVIDENCE field
        evidence_match = re.search(
            r"EVIDENCE(?:_TRIPLES)?:\s*(.+?)(?=\n[A-Z]+:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if evidence_match:
            evidence_text = evidence_match.group(1).strip()
            result["evidence"] = self._parse_evidence_triples(evidence_text)

        # Pattern for SUPPORTING_FACTS field
        sf_match = re.search(
            r"SUPPORTING_FACTS:\s*(.+?)(?=\n[A-Z]+:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if sf_match:
            sf_text = sf_match.group(1).strip()
            result["supporting_facts"] = self._parse_supporting_facts(sf_text)

        # Pattern for CONFIDENCE field
        conf_match = re.search(
            r"CONFIDENCE:\s*(.+?)(?=\n|$)",
            response,
            re.IGNORECASE,
        )
        if conf_match:
            result["confidence"] = conf_match.group(1).strip().lower()

        return result

    def _extract_answer(self, response: str) -> str:
        """Extract answer from unstructured response.

        Args:
            response: Raw response text

        Returns:
            Extracted answer string
        """
        # Try common answer patterns
        patterns = [
            r"(?:the answer is|therefore)[:\s]+([^.]+)",
            r"(?:is|was|are|were)\s+([A-Z][^.]+?)(?:\.|$)",
            r"^([A-Z][^.]{2,50})(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up answer
                answer = re.sub(r"^\s*[-•]\s*", "", answer)
                if len(answer) > 2:
                    return answer

        # Fall back to first sentence or line
        first_line = response.split("\n")[0].strip()
        if len(first_line) < 200:
            return first_line

        return response[:100].strip()

    def _parse_evidence_triples(self, evidence_text: str) -> list[tuple[str, str, str]]:
        """Parse evidence triples from text.

        Args:
            evidence_text: Text containing triple notation

        Returns:
            List of (subject, predicate, object) tuples
        """
        triples = []

        # Pattern for (subject, predicate, object) format
        triple_pattern = r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)"
        matches = re.findall(triple_pattern, evidence_text)

        for match in matches:
            triples.append(tuple(s.strip() for s in match))

        # Pattern for "Subject predicate Object" format
        if not triples:
            lines = evidence_text.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("-"):
                    line = line[1:].strip()
                # Simple heuristic: split by common predicates
                for pred in ["is", "was", "has", "born", "died", "directed"]:
                    if f" {pred} " in line.lower():
                        parts = re.split(rf"\s+{pred}\s+", line, maxsplit=1, flags=re.IGNORECASE)
                        if len(parts) == 2:
                            triples.append((parts[0].strip(), pred, parts[1].strip()))
                            break

        return triples

    def _parse_supporting_facts(self, sf_text: str) -> list[tuple[str, int]]:
        """Parse supporting facts from text.

        Format: "Title: sentence_index, sentence_index"

        Args:
            sf_text: Supporting facts text

        Returns:
            List of (title, sentence_index) tuples
        """
        facts = []

        # Pattern for "Title: 1, 2, 3" format
        pattern = r"([^:]+):\s*([\d,\s]+)"
        matches = re.findall(pattern, sf_text)

        for title, indices in matches:
            title = title.strip()
            for idx_str in indices.split(","):
                try:
                    idx = int(idx_str.strip())
                    facts.append((title, idx))
                except ValueError:
                    continue

        return facts

    def parse_chain_of_thought(self, response: str) -> dict:
        """Parse chain-of-thought response.

        Args:
            response: CoT response text

        Returns:
            Dict with answer and reasoning steps
        """
        result = {
            "answer": "",
            "steps": [],
            "reasoning": response,
        }

        # Look for final answer
        final_patterns = [
            r"(?:final answer|therefore|thus|so)[:\s]+(.+?)(?:\.|$)",
            r"(?:the answer is)[:\s]+(.+?)(?:\.|$)",
            r"(?:^|\n)(\d+\.\s*)?(?:answer)[:\s]+(.+?)(?:\.|$)",
        ]

        for pattern in final_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                result["answer"] = match.groups()[-1].strip()
                break

        # Extract numbered steps
        step_pattern = r"(\d+)\.\s*(.+?)(?=\n\d+\.|$)"
        steps = re.findall(step_pattern, response, re.DOTALL)
        result["steps"] = [(int(n), s.strip()) for n, s in steps]

        # If no answer found, try last step
        if not result["answer"] and result["steps"]:
            result["answer"] = result["steps"][-1][1]

        return result

    def parse_yes_no(self, response: str) -> Optional[bool]:
        """Parse yes/no response.

        Args:
            response: Response text

        Returns:
            True for yes, False for no, None if unclear
        """
        response_lower = response.lower().strip()

        yes_indicators = ["yes", "true", "correct", "affirmative"]
        no_indicators = ["no", "false", "incorrect", "negative"]

        for indicator in yes_indicators:
            if response_lower.startswith(indicator):
                return True

        for indicator in no_indicators:
            if response_lower.startswith(indicator):
                return False

        # Check in full response
        yes_count = sum(1 for ind in yes_indicators if ind in response_lower)
        no_count = sum(1 for ind in no_indicators if ind in response_lower)

        if yes_count > no_count:
            return True
        elif no_count > yes_count:
            return False

        return None

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison.

        Args:
            answer: Raw answer string

        Returns:
            Normalized answer
        """
        # Lowercase
        answer = answer.lower()

        # Remove articles
        answer = re.sub(r"^(the|a|an)\s+", "", answer)

        # Remove punctuation
        answer = re.sub(r"[.,!?;:]", "", answer)

        # Normalize whitespace
        answer = " ".join(answer.split())

        return answer.strip()
