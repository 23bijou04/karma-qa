"""Unit tests for text_solver.solve() — Wikipedia retrieval, context assembly, LLM reasoning."""

from unittest.mock import MagicMock, patch

from dbpedia_qa.models import Entity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(uri="http://dbpedia.org/resource/Inception", label="Inception"):
    return Entity(uri=uri, label=label)


def _mock_llm_client():
    return MagicMock()


def _make_solve_args(**overrides):
    defaults = dict(
        question="Who directed Inception?",
        entities=[_entity()],
        llm_client=_mock_llm_client(),
        model="test-model",
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Happy path — keyword path (RAG unavailable)
# ---------------------------------------------------------------------------

class TestSolveKeywordPath:
    @patch("text_solver._RAG_AVAILABLE", False)
    @patch("text_solver.call_llm")
    @patch("text_solver._solve_keyword")
    def test_solve_returns_answer_via_keyword_path(self, mock_keyword, mock_call_llm):
        """With RAG unavailable, keyword path is used and answer is returned."""
        mock_keyword.return_value = "Inception is a 2010 film directed by Christopher Nolan."
        mock_call_llm.return_value = "Based on the text.\nFINAL ANSWER: Christopher Nolan"

        from text_solver import solve
        answer = solve(**_make_solve_args())

        assert answer == "Christopher Nolan"
        mock_keyword.assert_called_once()
        mock_call_llm.assert_called_once()

    @patch("text_solver._RAG_AVAILABLE", False)
    @patch("text_solver.call_llm")
    @patch("text_solver._solve_keyword")
    def test_solve_returns_none_when_no_context(self, mock_keyword, mock_call_llm):
        """If keyword retrieval returns None, solve() returns None without calling LLM."""
        mock_keyword.return_value = None

        from text_solver import solve
        answer = solve(**_make_solve_args())

        assert answer is None
        mock_call_llm.assert_not_called()

    @patch("text_solver._RAG_AVAILABLE", False)
    @patch("text_solver.call_llm")
    @patch("text_solver._solve_keyword")
    def test_solve_uses_subquestion_prompt_when_flagged(self, mock_keyword, mock_call_llm):
        """is_subquestion=True causes TEXT_SUBQ_PROMPT to be used."""
        mock_keyword.return_value = "Some Wikipedia context."
        mock_call_llm.return_value = "FINAL ANSWER: Christopher Nolan"

        from text_solver import solve
        answer = solve(**_make_solve_args(is_subquestion=True))

        assert answer == "Christopher Nolan"
        # Verify subquestion prompt was used (contains "Sub-question:")
        prompt_used = mock_call_llm.call_args[0][1]
        assert "Sub-question:" in prompt_used
        assert "FINAL ANSWER:" in prompt_used

    @patch("text_solver._RAG_AVAILABLE", False)
    @patch("text_solver.call_llm")
    @patch("text_solver._solve_keyword")
    def test_solve_uses_main_prompt_by_default(self, mock_keyword, mock_call_llm):
        """Default (is_subquestion=False) uses TEXT_REASON_PROMPT."""
        mock_keyword.return_value = "Wikipedia context about Inception."
        mock_call_llm.return_value = "FINAL ANSWER: Christopher Nolan"

        from text_solver import solve
        solve(**_make_solve_args())

        prompt_used = mock_call_llm.call_args[0][1]
        assert "Question:" in prompt_used


# ---------------------------------------------------------------------------
# RAG path vs keyword fallback branching
# ---------------------------------------------------------------------------

class TestSolveRagVsKeywordBranching:
    @patch("text_solver._RAG_AVAILABLE", True)
    @patch("text_solver.call_llm")
    @patch("text_solver._solve_rag")
    def test_rag_path_used_when_available(self, mock_rag, mock_call_llm):
        """When RAG is available, _solve_rag is called (not keyword path)."""
        mock_rag.return_value = "RAG-assembled context about Inception."
        mock_call_llm.return_value = "FINAL ANSWER: Christopher Nolan"

        from text_solver import solve
        answer = solve(**_make_solve_args())

        assert answer == "Christopher Nolan"
        mock_rag.assert_called_once()

    @patch("text_solver._RAG_AVAILABLE", True)
    @patch("text_solver.call_llm")
    @patch("text_solver._solve_keyword")
    @patch("text_solver._solve_rag")
    def test_keyword_fallback_when_rag_raises(self, mock_rag, mock_keyword, mock_call_llm):
        """RAG exception triggers keyword fallback gracefully."""
        mock_rag.side_effect = RuntimeError("FAISS index error")
        mock_keyword.return_value = "Keyword context about Inception."
        mock_call_llm.return_value = "FINAL ANSWER: Christopher Nolan"

        from text_solver import solve
        answer = solve(**_make_solve_args())

        assert answer == "Christopher Nolan"
        mock_keyword.assert_called_once()

    @patch("text_solver._RAG_AVAILABLE", True)
    @patch("text_solver.call_llm")
    @patch("text_solver._solve_rag")
    def test_solve_records_rag_retrieval_path_in_meta(self, mock_rag, mock_call_llm):
        """_last_solve_meta records 'rag' as retrieval_path after successful RAG."""
        mock_rag.return_value = "Some context."
        mock_call_llm.return_value = "FINAL ANSWER: Answer"

        import text_solver
        text_solver.solve(**_make_solve_args())

        assert text_solver._last_solve_meta.get("retrieval_path") == "rag"

    @patch("text_solver._RAG_AVAILABLE", False)
    @patch("text_solver.call_llm")
    @patch("text_solver._solve_keyword")
    def test_solve_records_keyword_retrieval_path_in_meta(self, mock_keyword, mock_call_llm):
        """_last_solve_meta records 'keyword' as retrieval_path when RAG unavailable."""
        mock_keyword.return_value = "Some keyword context."
        mock_call_llm.return_value = "FINAL ANSWER: Answer"

        import text_solver
        text_solver.solve(**_make_solve_args())

        assert text_solver._last_solve_meta.get("retrieval_path") == "keyword"


# ---------------------------------------------------------------------------
# Context budget assembly (_assemble_from_chunks)
# ---------------------------------------------------------------------------

class TestAssembleFromChunks:
    def test_assembles_chunks_within_budget(self):
        """Chunks fitting within max_chars are all included."""
        from rag.chunker import WikiChunk
        from text_solver import _assemble_from_chunks

        chunks = [
            WikiChunk(text="Fact one.", page_title="Page A", section_header="Lead",
                      chunk_index=0, char_offset=0),
            WikiChunk(text="Fact two.", page_title="Page B", section_header="Lead",
                      chunk_index=0, char_offset=0),
        ]
        result = _assemble_from_chunks(chunks, max_chars=10_000)

        assert "Page A" in result
        assert "Page B" in result
        assert "Fact one." in result
        assert "Fact two." in result

    def test_respects_max_chars_budget(self):
        """Chunks exceeding max_chars are not included."""
        from rag.chunker import WikiChunk
        from text_solver import _assemble_from_chunks

        big_text = "x" * 5_000
        chunks = [
            WikiChunk(text=big_text, page_title="Page A", section_header="Lead",
                      chunk_index=0, char_offset=0),
            WikiChunk(text="Short chunk.", page_title="Page B", section_header="Lead",
                      chunk_index=0, char_offset=0),
        ]
        result = _assemble_from_chunks(chunks, max_chars=6_000)

        # Page A consumed most budget, Page B header + text might exceed limit
        assert len(result) <= 6_100  # small slack for header text

    def test_respects_per_entity_budget(self):
        """Multiple chunks from same page are capped at max_per_entity."""
        from rag.chunker import WikiChunk
        from text_solver import _assemble_from_chunks

        chunks = [
            WikiChunk(text="A" * 3_000, page_title="Same Page", section_header="S1",
                      chunk_index=0, char_offset=0),
            WikiChunk(text="B" * 3_000, page_title="Same Page", section_header="S2",
                      chunk_index=1, char_offset=3000),
            WikiChunk(text="C" * 100, page_title="Other Page", section_header="Lead",
                      chunk_index=0, char_offset=0),
        ]
        result = _assemble_from_chunks(chunks, max_chars=20_000, max_per_entity=4_000)

        # Second chunk from "Same Page" should be dropped (exceeds per-entity cap)
        assert "Other Page" in result

    def test_prepends_kg_facts_when_provided(self):
        """kg_facts string is prepended before chunk content."""
        from rag.chunker import WikiChunk
        from text_solver import _assemble_from_chunks

        chunks = [
            WikiChunk(text="Wikipedia text.", page_title="Film", section_header="Lead",
                      chunk_index=0, char_offset=0),
        ]
        result = _assemble_from_chunks(chunks, kg_facts="KG: director = Nolan")

        assert result.startswith("KG: director = Nolan")
        assert "Wikipedia text." in result

    def test_empty_chunks_returns_empty_string(self):
        """No chunks and no kg_facts returns empty string."""
        from text_solver import _assemble_from_chunks

        result = _assemble_from_chunks([])

        assert result == ""


# ---------------------------------------------------------------------------
# Known facts / KG facts integration
# ---------------------------------------------------------------------------

class TestSolveKnownFacts:
    @patch("text_solver._RAG_AVAILABLE", False)
    @patch("text_solver.call_llm")
    @patch("text_solver._solve_keyword")
    def test_known_facts_included_in_llm_prompt(self, mock_keyword, mock_call_llm):
        """known_facts content appears in the LLM prompt."""
        mock_keyword.return_value = "Wikipedia context."
        mock_call_llm.return_value = "FINAL ANSWER: Christopher Nolan"

        from text_solver import solve
        solve(**_make_solve_args(known_facts="director = Christopher Nolan (from KG)"))

        prompt = mock_call_llm.call_args[0][1]
        assert "director = Christopher Nolan (from KG)" in prompt
