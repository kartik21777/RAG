"""
Abstract base interfaces for synthetic evaluator data generation and LLM judging.
Ensures pluggability between deterministic offline mocks and live LLM APIs.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from src.core.schemas import Document


class BaseSyntheticGenerator(ABC):
    """
    Abstract generator interface for synthesizing query-document pairs.
    """

    @abstractmethod
    def generate_sufficient_query(self, document: Document) -> Tuple[str, str]:
        """
        Synthesizes a query whose information need is completely satisfied by the document.

        Args:
            document: Ground-truth passage.

        Returns:
            Tuple of (query_text, rationale).
        """
        pass

    @abstractmethod
    def generate_insufficient_query(self, document: Document) -> Tuple[str, str]:
        """
        Synthesizes a query where the document is on-topic (intent_fit=1)
        but only partially answers the query (evidence_sufficiency=0).

        Args:
            document: Ground-truth passage.

        Returns:
            Tuple of (query_text, rationale).
        """
        pass


class BaseEvaluatorJudge(ABC):
    """
    Abstract interface for LLM-as-a-judge evaluation of (query, document) pairs.
    """

    @abstractmethod
    def evaluate(self, query_text: str, doc_text: str) -> Tuple[int, int, str]:
        """
        Evaluates a query-document pair along intent-fit and evidence-sufficiency axes.

        Args:
            query_text: User question.
            doc_text: Candidate passage.

        Returns:
            Tuple of (intent_fit [0 or 1], evidence_sufficiency [0 or 1], rationale).

        Raises:
            ValueError: If the evaluation emits the prohibited (0, 1) combination.
        """
        pass
