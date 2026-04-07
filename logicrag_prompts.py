"""LogicRAG prompt constants for early stopping and final answer generation.

Adapted from the LogicRAG framework (AAAI 2026):
  https://github.com/chensyCN/LogicRAG

DEPENDENCY_RAG_PROMPT — checks if accumulated facts suffice to answer the question
FINAL_ANSWER_PROMPT  — generates concise direct answers from accumulated facts
"""

DEPENDENCY_RAG_PROMPT = """We pre-parsed the question into a list of dependencies, and the \
dependencies are sorted in a topological order, below is the question, the information \
summary, and the decomposed dependencies:

Question: {question}

Available Information:
{info_summary}

Decomposed dependencies:
{dependencies}

Current dependency to be answered:
{current_dependency}

Please analyze the question and the information summary, and the decomposed \
dependencies, and answer the following questions:
Please analyze:
1. Can the question be answered completely with this information? (Yes/No)
2. Summarize our current understanding based on available information.

Please format your response as a JSON object with these keys:
- "can_answer": boolean
- "current_understanding": string"""

_FINAL_ANSWER_BASE = """You must give ONLY the direct answer in the most concise way possible.
DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

IMPORTANT — the sub-questions below form a reasoning chain:
- Earlier sub-questions find intermediate entities; later ones use those entities.
- The LAST sub-question's answer is usually what the original question asks for.
- Do NOT return an intermediate entity when a later sub-question has resolved the final answer.
- If a sub-question returns a list, pick the item that satisfies the condition in the original question.
- For yes/no questions with multiple sub-questions, combine ALL sub-answers (both must be true for "yes").
{comparison_instructions}
Question: {question}

Information Summary:
{info_summary}

End your response with exactly: FINAL ANSWER: <your answer>"""

_COMPARISON_INSTRUCTIONS = """When comparing countries or nationalities:
- If answers contain city or state names, identify which country they belong to
- Compare at the country level, not city/state level
- Historical states (East Germany, Ottoman Empire) map to their modern successor
"""


def get_final_answer_prompt(is_comparison: bool = False) -> str:
    """Return final answer prompt, with country-comparison instructions only for comparisons.

    The comparison instructions cause over-generalization on non-comparison questions
    (e.g. "Boston, Massachusetts" -> "United States" when the question asks birthPlace).
    Gating them on question type prevents this corruption while preserving comparison logic.
    """
    instructions = _COMPARISON_INSTRUCTIONS if is_comparison else ""
    return _FINAL_ANSWER_BASE.replace("{comparison_instructions}", instructions)


# Backward-compatible constant (no comparison instructions by default)
FINAL_ANSWER_PROMPT = get_final_answer_prompt(is_comparison=False)
