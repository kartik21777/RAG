"""
Synthetic query generators for creating evaluator training pairs.
Includes MockDeterministicGenerator for reproducible offline testing and
LLMSyntheticGenerator for live LLM prompt execution.
"""

from typing import Callable, Optional, Tuple
from src.core.schemas import Document
from src.synthetic_data.base import BaseSyntheticGenerator


class MockDeterministicGenerator(BaseSyntheticGenerator):
    """
    Deterministic rule-based generator for offline testing and fixture generation.
    Requires zero external API calls, operates deterministically, and preserves provenance.
    """

    def generate_sufficient_query(self, document: Document) -> Tuple[str, str]:
        """
        Synthesizes a query whose factual answer is fully contained in the document.
        """
        title = document.metadata.get("title", f"Document {document.doc_id}")
        words = document.text.split()
        lead_phrase = " ".join(words[:6]) if len(words) >= 6 else document.text

        query = f"According to {title}, what is the main mechanism described in: '{lead_phrase}'?"
        rationale = (
            f"The query directly targets the specific factual content of {document.doc_id} ('{title}'). "
            "The document provides complete factual information to answer it (intent_fit=1, evidence_sufficiency=1)."
        )
        return query, rationale

    def generate_insufficient_query(self, document: Document) -> Tuple[str, str]:
        """
        Synthesizes a query that asks for multiple aspects where the document provides only partial facts.
        """
        title = document.metadata.get("title", f"Document {document.doc_id}")
        words = document.text.split()
        lead_phrase = " ".join(words[:5]) if len(words) >= 5 else document.text

        query = (
            f"How does {lead_phrase} operate according to {title}, and what are its quantitative "
            "deployment metrics across distributed edge hardware clusters?"
        )
        rationale = (
            f"The query is on-topic and addresses {document.doc_id} ('{title}') (intent_fit=1), "
            "but asks for additional edge deployment metrics not covered in the document passage "
            "(evidence_sufficiency=0)."
        )
        return query, rationale


class LLMSyntheticGenerator(BaseSyntheticGenerator):
    """
    Prompt-based generator utilizing an LLM completion function to synthesize training pairs.
    """

    SUFFICIENT_PROMPT_TEMPLATE = (
        "You are an expert data curator for a Corrective RAG system.\n"
        "Given the following passage, write a concise question whose factual answer is COMPLETELY "
        "and EXCLUSIVELY answered by the passage.\n\n"
        "Passage: {text}\n\n"
        "Output format:\n"
        "QUESTION: <question>\n"
        "RATIONALE: <why the passage is completely sufficient to answer this question>"
    )

    INSUFFICIENT_PROMPT_TEMPLATE = (
        "You are an expert data curator for a Corrective RAG system.\n"
        "Given the following passage, write a question that matches the topic and intent of the passage, "
        "but requires additional details (such as specific real-world case studies or missing parameters) "
        "that are NOT present in the passage. The passage must be on-topic but incomplete.\n\n"
        "Passage: {text}\n\n"
        "Output format:\n"
        "QUESTION: <question>\n"
        "RATIONALE: <why the passage is on-topic (intent_fit=1) but incomplete (evidence_sufficiency=0)>"
    )

    def __init__(self, llm_fn: Callable[[str], str], model_name: str = "custom_llm"):
        self.llm_fn = llm_fn
        self.model_name = model_name

    def generate_sufficient_query(self, document: Document) -> Tuple[str, str]:
        prompt = self.SUFFICIENT_PROMPT_TEMPLATE.format(text=document.text)
        response = self.llm_fn(prompt)
        return self._parse_response(response, document.doc_id, "sufficient")

    def generate_insufficient_query(self, document: Document) -> Tuple[str, str]:
        prompt = self.INSUFFICIENT_PROMPT_TEMPLATE.format(text=document.text)
        response = self.llm_fn(prompt)
        return self._parse_response(response, document.doc_id, "insufficient")

    def _parse_response(self, response: str, doc_id: str, mode: str) -> Tuple[str, str]:
        question = ""
        rationale = ""
        for line in response.splitlines():
            line_clean = line.strip()
            if line_clean.upper().startswith("QUESTION:"):
                question = line_clean.split(":", 1)[1].strip()
            elif line_clean.upper().startswith("RATIONALE:"):
                rationale = line_clean.split(":", 1)[1].strip()

        if not question:
            question = f"Synthesized {mode} question for document {doc_id}."
        if not rationale:
            rationale = f"Synthesized {mode} rationale for document {doc_id}."

        return question, rationale
