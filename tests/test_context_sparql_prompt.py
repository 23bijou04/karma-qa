"""Tests for Cases 11 & 13: CONTEXT_SPARQL_PROMPT changes."""

from hybrid_qa import CONTEXT_SPARQL_PROMPT


class TestContextSparqlPrompt:
    def test_limit_6(self):
        """CONTEXT_SPARQL_PROMPT uses LIMIT 6 (not 10)."""
        assert "LIMIT 6" in CONTEXT_SPARQL_PROMPT
        assert "LIMIT 10" not in CONTEXT_SPARQL_PROMPT

    def test_family_relations_mentioned(self):
        """CONTEXT_SPARQL_PROMPT mentions family relation properties."""
        prompt_lower = CONTEXT_SPARQL_PROMPT.lower()
        assert "father" in prompt_lower or "parent" in prompt_lower
        assert "dbo:parent" in CONTEXT_SPARQL_PROMPT or "dbp:father" in CONTEXT_SPARQL_PROMPT

    def test_example_uses_limit_6(self):
        """Example query in prompt uses LIMIT 6."""
        # Find the example section
        example_idx = CONTEXT_SPARQL_PROMPT.find("EXAMPLE")
        assert example_idx >= 0
        example_section = CONTEXT_SPARQL_PROMPT[example_idx:]
        assert "LIMIT 6" in example_section
