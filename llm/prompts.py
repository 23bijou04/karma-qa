"""Prompt templates for different question types."""

from dataclasses import dataclass
from typing import Optional

from ..models import QuestionType


@dataclass
class PromptTemplate:
    """Template for generating LLM prompts."""

    system_prompt: str
    user_template: str
    output_format: str

    def format(
        self,
        question: str,
        context: str,
        **kwargs,
    ) -> str:
        """Format the prompt with given inputs.

        Args:
            question: The question text
            context: Assembled context
            **kwargs: Additional template variables

        Returns:
            Formatted prompt string
        """
        return self.user_template.format(
            question=question,
            context=context,
            output_format=self.output_format,
            **kwargs,
        )


# System prompts
BASE_SYSTEM_PROMPT = """You are a precise question-answering system that uses knowledge graph facts to answer questions.
You must:
1. Only use the facts provided in the context
2. Provide a direct answer when possible
3. Explain your reasoning briefly
4. If the answer cannot be determined from the facts, say so clearly"""

COMPARISON_SYSTEM_PROMPT = """You are a comparison expert that analyzes properties of entities.
Your task is to compare specific properties (like dates, numbers, locations) and determine which entity satisfies the question criteria.
Always show the values being compared and explain your comparison logic."""

INFERENCE_SYSTEM_PROMPT = """You are a logical reasoning expert that follows inference chains.
Your task is to follow logical rules (like family relationships) step by step.
Always show each step of the reasoning chain."""

COMPOSITIONAL_SYSTEM_PROMPT = """You are a multi-step reasoning expert.
Your task is to follow chains of relationships to find the answer.
First find the intermediate entity, then find the target property."""

BRIDGE_COMPARISON_SYSTEM_PROMPT = """You are a bridge-comparison expert.
Your task is to:
1. First find the bridge entities for each subject
2. Then compare the properties of those bridge entities
3. Return which original entity satisfies the comparison criteria"""


# User templates
COMPARISON_TEMPLATE = """Question: {question}

Context - Knowledge Graph Facts:
{context}

Instructions:
1. Identify the property being compared
2. Extract the relevant values for each entity
3. Compare the values
4. Determine which entity satisfies the question criteria

{output_format}

Answer:"""

INFERENCE_TEMPLATE = """Question: {question}

Context - Reasoning Chain:
{context}

Instructions:
1. Follow the reasoning chain step by step
2. Each step builds on the previous one
3. The final entity in the chain is the answer

{output_format}

Answer:"""

COMPOSITIONAL_TEMPLATE = """Question: {question}

Context - Knowledge Graph Facts:
{context}

Instructions:
1. Identify the intermediate entity (bridge) from the first relation
2. Find the target property of that intermediate entity
3. The value of that property is the answer

{output_format}

Answer:"""

BRIDGE_COMPARISON_TEMPLATE = """Question: {question}

Context - Knowledge Graph Facts:
{context}

Instructions:
1. For each entity mentioned in the question, find its bridge entity (e.g., director, producer)
2. Get the comparison property (e.g., birth date) for each bridge entity
3. Compare those properties
4. Return the original entity whose bridge entity satisfies the comparison

{output_format}

Answer:"""

GENERIC_TEMPLATE = """Question: {question}

Context:
{context}

{output_format}

Answer:"""


# Output formats
SIMPLE_OUTPUT = """Provide a direct answer (a name, date, yes/no, or short phrase).
If you cannot determine the answer from the given facts, respond with "Cannot determine"."""

STRUCTURED_OUTPUT = """Provide your response in this format:
ANSWER: [your direct answer]
REASONING: [brief explanation of how you reached the answer]
EVIDENCE: [list the facts you used]"""

EVIDENCE_OUTPUT = """Provide your response in this format:
ANSWER: [your direct answer]
SUPPORTING_FACTS: [list sentence indices that support the answer, e.g., "Title: 1, 3"]
EVIDENCE_TRIPLES: [list the (subject, predicate, object) triples used]
CONFIDENCE: [high/medium/low]"""


def get_prompt_for_question_type(
    question_type: QuestionType,
    output_style: str = "structured",
) -> PromptTemplate:
    """Get the appropriate prompt template for a question type.

    Args:
        question_type: The classified question type
        output_style: Output format style ('simple', 'structured', 'evidence')

    Returns:
        PromptTemplate instance
    """
    output_formats = {
        "simple": SIMPLE_OUTPUT,
        "structured": STRUCTURED_OUTPUT,
        "evidence": EVIDENCE_OUTPUT,
    }
    output_format = output_formats.get(output_style, STRUCTURED_OUTPUT)

    templates = {
        QuestionType.COMPARISON: PromptTemplate(
            system_prompt=COMPARISON_SYSTEM_PROMPT,
            user_template=COMPARISON_TEMPLATE,
            output_format=output_format,
        ),
        QuestionType.INFERENCE: PromptTemplate(
            system_prompt=INFERENCE_SYSTEM_PROMPT,
            user_template=INFERENCE_TEMPLATE,
            output_format=output_format,
        ),
        QuestionType.COMPOSITIONAL: PromptTemplate(
            system_prompt=COMPOSITIONAL_SYSTEM_PROMPT,
            user_template=COMPOSITIONAL_TEMPLATE,
            output_format=output_format,
        ),
        QuestionType.BRIDGE_COMPARISON: PromptTemplate(
            system_prompt=BRIDGE_COMPARISON_SYSTEM_PROMPT,
            user_template=BRIDGE_COMPARISON_TEMPLATE,
            output_format=output_format,
        ),
    }

    return templates.get(
        question_type,
        PromptTemplate(
            system_prompt=BASE_SYSTEM_PROMPT,
            user_template=GENERIC_TEMPLATE,
            output_format=output_format,
        ),
    )


def create_few_shot_prompt(
    question_type: QuestionType,
    examples: list[dict],
    question: str,
    context: str,
) -> str:
    """Create a few-shot prompt with examples.

    Args:
        question_type: The question type
        examples: List of example dicts with 'question', 'context', 'answer'
        question: The actual question
        context: The actual context

    Returns:
        Few-shot prompt string
    """
    template = get_prompt_for_question_type(question_type)

    parts = [template.system_prompt, "\nHere are some examples:\n"]

    for i, example in enumerate(examples[:3]):  # Max 3 examples
        parts.append(f"\nExample {i + 1}:")
        parts.append(f"Question: {example['question']}")
        parts.append(f"Context: {example['context']}")
        parts.append(f"Answer: {example['answer']}\n")

    parts.append("\nNow answer this question:")
    parts.append(template.format(question=question, context=context))

    return "\n".join(parts)
