r"""
Runtime Evaluator Interfaces and Mocks for Corrective RAG (CRAG).

Defines the abstract inference-time evaluation contract (`BaseRuntimeEvaluator`)
and a deterministic, offline `MockRuntimeEvaluator` for CPU-only testing.

Note on Member 2 Integration:
Member 2 will implement `BaseRuntimeEvaluator` with their fine-tuned DeBERTa-v3 checkpoint:
- Model loads cross-encoder weights: `[CLS] query [SEP] document [SEP]`
- Dual classification heads predict `intent_fit` and `evidence_sufficiency`
- Logits are passed through sigmoid $\sigma(z) \in [0, 1]$ and mapped to decision scores.
- Raw logits and checkpoint provenance are preserved in `prediction.metadata`.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.schemas import Document
from src.orchestration.schemas import EvaluatorPrediction


class BaseRuntimeEvaluator(ABC):
    """
    Abstract interface for runtime (inference-time) evaluation of (query, document) pairs.

    Evaluates whether candidate evidence matches query intent and provides sufficient factual
    completeness to generate a grounded answer.
    """

    @abstractmethod
    def evaluate_pair(self, query: str, doc: Document) -> EvaluatorPrediction:
        """
        Evaluates a specific query-document candidate pair.

        Args:
            query: User question or search query string.
            doc: Canonical Document candidate to evaluate.

        Returns:
            EvaluatorPrediction containing continuous decision scores in [0.0, 1.0].

        Raises:
            ValueError: If query or document is malformed or empty.
            TypeError: If doc is not a Document instance or query is not a string.
        """
        pass

    def evaluate_collection(self, query: str, docs: List[Document]) -> EvaluatorPrediction:
        """
        Evaluates a collection of documents collectively for query intent and joint evidence sufficiency.
        Default implementation combines non-empty document texts into a composite document candidate
        and evaluates it via evaluate_pair.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        q_clean = query.strip()
        if not q_clean:
            raise ValueError("query cannot be empty or whitespace only.")
        if not isinstance(docs, list):
            raise TypeError(f"docs must be a list of Document instances, got {type(docs).__name__}")

        if not docs:
            return EvaluatorPrediction(
                query_text=q_clean,
                doc_id="__empty_collection__",
                intent_fit_score=0.0,
                evidence_sufficiency_score=0.0,
                rationale="Empty document collection evaluated.",
                metadata={"evaluation_type": "collection", "doc_count": 0},
            )

        combined_text = "\n\n".join(d.text.strip() for d in docs if d.text and d.text.strip())
        composite_id = "+".join(d.doc_id for d in docs)
        composite_doc = Document(
            doc_id=composite_id,
            text=combined_text,
            metadata={"is_composite": True, "source_doc_ids": [d.doc_id for d in docs]},
        )
        return self.evaluate_pair(q_clean, composite_doc)


class MockRuntimeEvaluator(BaseRuntimeEvaluator):
    """
    Deterministic, configurable mock evaluator for offline testing without a GPU or API keys.

    Allows tests to configure explicit score overrides keyed by (query_text, doc_id) or doc_id,
    with a deterministic fallback.

    CRITICAL NOTE:
    Mock scores are strictly simulated decision scores for exercising pipeline routing logic.
    They do NOT represent a trained model's predictive performance or calibrated probabilities.
    """

    def __init__(
        self,
        pair_scores: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
        doc_scores: Optional[Dict[str, Tuple[float, float]]] = None,
        default_scores: Tuple[float, float] = (0.0, 0.0),
        rationales: Optional[Dict[Union[Tuple[str, str], str], str]] = None,
        default_rationale: Optional[str] = "Deterministic mock evaluation",
    ):
        """
        Args:
            pair_scores: Mapping of (query_text, doc_id) -> (intent_fit_score, evidence_sufficiency_score).
            doc_scores: Fallback mapping of doc_id -> (intent_fit_score, evidence_sufficiency_score).
            default_scores: (intent_fit, sufficiency) score tuple used if no key matches.
            rationales: Optional mapping of (query, doc_id) or doc_id -> diagnostic rationale string.
            default_rationale: Default rationale when none is explicitly configured.
        """
        self.pair_scores: Dict[Tuple[str, str], Tuple[float, float]] = pair_scores or {}
        self.doc_scores: Dict[str, Tuple[float, float]] = doc_scores or {}
        self.default_scores: Tuple[float, float] = default_scores
        self.rationales: Dict[Union[Tuple[str, str], str], str] = rationales or {}
        self.default_rationale = default_rationale

        # Validate that initial scores are within range
        for key, scores in self.pair_scores.items():
            self._validate_score_tuple(key, scores)
        for doc_id, scores in self.doc_scores.items():
            self._validate_score_tuple(doc_id, scores)
        self._validate_score_tuple("default_scores", self.default_scores)

    @staticmethod
    def _validate_score_tuple(key: Any, scores: Tuple[float, float]) -> None:
        if len(scores) != 2:
            raise ValueError(f"Score tuple for {key} must have exactly 2 elements (intent, suff), got {scores}")
        for s in scores:
            if not isinstance(s, (int, float)) or s < 0.0 or s > 1.0:
                raise ValueError(f"Configured mock score for {key} must be in [0.0, 1.0], got {s}")

    def evaluate_pair(self, query: str, doc: Document) -> EvaluatorPrediction:
        """
        Evaluates a query-document pair using configured deterministic lookups.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("query cannot be empty or whitespace only.")
        if not isinstance(doc, Document):
            raise TypeError(f"doc must be Document instance, got {type(doc).__name__}")

        q_clean = query.strip()
        d_id = doc.doc_id

        # Priority 1: Exact (query_text, doc_id) pair match
        if (q_clean, d_id) in self.pair_scores:
            intent_score, suff_score = self.pair_scores[(q_clean, d_id)]
            mock_source = "pair_override"
        # Priority 2: doc_id fallback match
        elif d_id in self.doc_scores:
            intent_score, suff_score = self.doc_scores[d_id]
            mock_source = "doc_override"
        # Priority 3: Default scores
        else:
            intent_score, suff_score = self.default_scores
            mock_source = "default"

        # Resolve rationale
        rationale = (
            self.rationales.get((q_clean, d_id))
            or self.rationales.get(d_id)
            or self.default_rationale
        )

        return EvaluatorPrediction(
            query_text=q_clean,
            doc_id=d_id,
            intent_fit_score=intent_score,
            evidence_sufficiency_score=suff_score,
            rationale=rationale,
            metadata={
                "evaluator_type": "MockRuntimeEvaluator",
                "mock_source": mock_source,
                "is_deterministic_mock": True,
            },
        )
