"""Tests for Case 11: Priority-based context budget allocation."""

from graphrag_wiki import assemble_context, MAX_CONTEXT_CHARS, MAX_CHARS_PER_PAGE


class TestSeedPriority:
    def test_seed_pages_always_in_context(self):
        """Seed pages appear in context even when bridge pages have higher keyword score."""
        raw_texts = {
            "Seed Entity": "The answer is here. Relevant text about the question topic.",
            "Bridge 1": "keyword keyword keyword keyword keyword keyword keyword keyword",
            "Bridge 2": "keyword keyword keyword keyword keyword keyword",
            "Bridge 3": "keyword keyword keyword keyword keyword",
        }
        seed_titles = {"Seed Entity"}

        context, filtered = assemble_context(
            raw_texts, "What is the keyword topic?", seed_titles=seed_titles,
        )
        assert "Seed Entity" in context

    def test_seed_pages_appear_before_bridge(self):
        """Seed pages should be ranked before bridge pages in assembled context."""
        raw_texts = {
            "Seed Page": "Some relevant content about the topic.",
            "Bridge Page": "Also relevant keyword keyword keyword.",
        }
        seed_titles = {"Seed Page"}

        context, filtered = assemble_context(
            raw_texts, "What is the topic?", seed_titles=seed_titles,
        )
        seed_pos = context.find("Seed Page")
        bridge_pos = context.find("Bridge Page")
        assert seed_pos >= 0, "Seed page must appear in context"
        # Seed should appear first
        if bridge_pos >= 0:
            assert seed_pos < bridge_pos

    def test_budget_split_seed_gets_more(self):
        """Seed pages get larger per-page budget than bridge pages."""
        # Create pages that are large enough to show budget differences
        raw_texts = {
            "Seed": "x " * 5000,  # 10K chars
            "Bridge1": "y " * 5000,
            "Bridge2": "z " * 5000,
            "Bridge3": "w " * 5000,
        }
        seed_titles = {"Seed"}

        context, filtered = assemble_context(
            raw_texts, "test question", seed_titles=seed_titles,
        )
        # Seed should get up to 60% of MAX_CONTEXT_CHARS
        seed_text = filtered.get("Seed", "")
        assert len(seed_text) >= MAX_CHARS_PER_PAGE

    def test_no_seed_titles_falls_back_to_uniform(self):
        """Without seed_titles, budget is uniform (backward compatibility)."""
        raw_texts = {
            "Page A": "Content about topic A.",
            "Page B": "Content about topic B.",
        }

        context, filtered = assemble_context(
            raw_texts, "What is the topic?",
        )
        # Should work without errors
        assert len(context) > 0
