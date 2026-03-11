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

FINAL_ANSWER_PROMPT = """You must give ONLY the direct answer in the most concise way possible.
DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

Question: {question}

Information Summary:
{info_summary}

End your response with exactly: FINAL ANSWER: <your answer>"""
