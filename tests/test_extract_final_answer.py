"""Tests for extract_final_answer() — shared answer extraction utility."""

from models import extract_final_answer


class TestExtractFinalAnswer:
    """Core extraction logic."""

    def test_basic_extraction(self):
        text = "Some reasoning here.\nFINAL ANSWER: Paris"
        assert extract_final_answer(text) == "Paris"

    def test_case_insensitive(self):
        assert extract_final_answer("final answer: Berlin") == "Berlin"
        assert extract_final_answer("Final Answer: Berlin") == "Berlin"
        assert extract_final_answer("FINAL answer: Berlin") == "Berlin"

    def test_strips_whitespace(self):
        assert extract_final_answer("FINAL ANSWER:   Tokyo  ") == "Tokyo"

    def test_multiline_reasoning_before_marker(self):
        text = (
            "Let me think about this.\n"
            "The director of Inception is Christopher Nolan.\n"
            "He was born in London.\n"
            "FINAL ANSWER: Christopher Nolan"
        )
        assert extract_final_answer(text) == "Christopher Nolan"


class TestLastMatchSemantics:
    """Last-match behavior when multiple markers appear."""

    def test_multiple_markers_takes_last(self):
        text = (
            "I need to find the FINAL ANSWER: let me check...\n"
            "After research, FINAL ANSWER: 1879"
        )
        assert extract_final_answer(text) == "1879"

    def test_three_markers_takes_last(self):
        text = (
            "FINAL ANSWER: wrong1\n"
            "Wait, let me reconsider.\n"
            "FINAL ANSWER: wrong2\n"
            "Actually, FINAL ANSWER: correct"
        )
        assert extract_final_answer(text) == "correct"


class TestEdgeCases:
    """Edge cases: None, empty, no marker."""

    def test_none_returns_empty_string(self):
        assert extract_final_answer(None) == ""

    def test_empty_string_returns_empty(self):
        assert extract_final_answer("") == ""

    def test_no_marker_returns_original(self):
        text = "Christopher Nolan"
        assert extract_final_answer(text) == "Christopher Nolan"

    def test_marker_with_empty_answer(self):
        assert extract_final_answer("FINAL ANSWER:") == ""
        assert extract_final_answer("FINAL ANSWER:  ") == ""

    def test_marker_only_no_surrounding_text(self):
        assert extract_final_answer("FINAL ANSWER: yes") == "yes"

    def test_whitespace_only_input(self):
        assert extract_final_answer("   ") == "   "
