"""Answer generation using LLMs."""

import logging
from typing import Optional, Any
from abc import ABC, abstractmethod

from ..models import QuestionType, Answer, Evidence, Triple
from ..config import LLMConfig
from .prompts import get_prompt_for_question_type
from .parser import ResponseParser

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            Generated response text
        """
        pass


class OllamaClient(LLMClient):
    """Ollama LLM client."""

    def __init__(
        self,
        model: str = "llama3.1:70b",
        host: str = "http://localhost:11434",
    ):
        """Initialize Ollama client.

        Args:
            model: Model name
            host: Ollama server URL
        """
        self.model = model
        self.host = host
        self._client = None

    def _get_client(self):
        """Lazy load the Ollama client."""
        if self._client is None:
            try:
                import ollama

                self._client = ollama.Client(host=self.host)
            except ImportError:
                raise ImportError("ollama package not installed. Run: pip install ollama")
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response using Ollama."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


class OpenAIClient(LLMClient):
    """OpenAI-compatible LLM client."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """Initialize OpenAI client.

        Args:
            model: Model name
            api_key: API key
            api_base: Optional custom API base URL
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self._client = None

    def _get_client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                kwargs = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.api_base:
                    kwargs["base_url"] = self.api_base

                self._client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response using OpenAI API."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class AnswerGenerator:
    """Generates answers to questions using LLM with KG context."""

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """Initialize answer generator.

        Args:
            llm_config: LLM configuration
            llm_client: Optional pre-configured LLM client
        """
        self.config = llm_config or LLMConfig()
        self.parser = ResponseParser()

        if llm_client:
            self.client = llm_client
        else:
            self.client = self._create_client()

    def _create_client(self) -> LLMClient:
        """Create LLM client based on configuration."""
        if self.config.provider == "ollama":
            return OllamaClient(
                model=self.config.model,
                host=self.config.api_base or "http://localhost:11434",
            )
        elif self.config.provider == "openai":
            return OpenAIClient(
                model=self.config.model,
                api_key=self.config.api_key,
                api_base=self.config.api_base,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.provider}")

    def generate_answer(
        self,
        question: str,
        context: str,
        question_type: QuestionType,
        triples: Optional[list[Triple]] = None,
    ) -> Answer:
        """Generate an answer to a question.

        Args:
            question: The question text
            context: Assembled context from KG
            question_type: Type of question
            triples: Optional triples used for context

        Returns:
            Answer object with response and evidence
        """
        # Get appropriate prompt template
        template = get_prompt_for_question_type(question_type, output_style="structured")

        # Format prompt
        prompt = template.format(question=question, context=context)

        # Generate response
        try:
            response = self.client.generate(
                prompt=prompt,
                system_prompt=template.system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return Answer(
                text="Error: Could not generate answer",
                confidence=0.0,
            )

        # Parse response
        parsed = self.parser.parse(response)

        # Build evidence if triples provided
        evidence = None
        if triples:
            evidence = Evidence(
                triples=triples,
                reasoning_path=parsed.get("reasoning", "").split(". "),
            )

        return Answer(
            text=parsed.get("answer", response.strip()),
            confidence=self._estimate_confidence(parsed),
            evidence=evidence,
            reasoning=parsed.get("reasoning"),
        )

    def generate_with_chain_of_thought(
        self,
        question: str,
        context: str,
        question_type: QuestionType,
    ) -> Answer:
        """Generate answer with explicit chain-of-thought reasoning.

        Args:
            question: The question text
            context: Assembled context
            question_type: Type of question

        Returns:
            Answer with detailed reasoning
        """
        cot_prompt = f"""Question: {question}

Context:
{context}

Think through this step by step:
1. What is the question asking for?
2. What relevant facts do we have?
3. How do these facts connect to answer the question?
4. What is the final answer?

Let's think step by step:"""

        try:
            response = self.client.generate(
                prompt=cot_prompt,
                system_prompt="You are a careful reasoning assistant. Think through problems step by step.",
                temperature=0.0,
                max_tokens=self.config.max_tokens,
            )
        except Exception as e:
            logger.error(f"CoT generation failed: {e}")
            return Answer(text="Error", confidence=0.0)

        # Extract final answer from CoT response
        parsed = self.parser.parse_chain_of_thought(response)

        return Answer(
            text=parsed.get("answer", ""),
            confidence=0.7,  # CoT typically has medium-high confidence
            reasoning=response,
        )

    def _estimate_confidence(self, parsed_response: dict) -> float:
        """Estimate confidence from parsed response.

        Args:
            parsed_response: Parsed response dict

        Returns:
            Confidence score between 0 and 1
        """
        # Check for explicit confidence
        if "confidence" in parsed_response:
            conf_str = parsed_response["confidence"].lower()
            if "high" in conf_str:
                return 0.9
            elif "medium" in conf_str:
                return 0.7
            elif "low" in conf_str:
                return 0.4

        # Check for uncertainty markers
        answer = parsed_response.get("answer", "").lower()
        if "cannot determine" in answer or "not sure" in answer:
            return 0.2

        # Default confidence
        return 0.6
