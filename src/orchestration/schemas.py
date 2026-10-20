"""
Orchestration schemas for the CRAG pipeline.
Defines runtime evaluator predictions, candidate-level routing decisions,
and query-level aggregation structures.
"""

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from src.core.schemas import CRAGAction, Document, RetrievalResult


class EvaluatorPrediction(BaseModel):
    """
    Inference-time prediction emitted by a runtime evaluator for a specific
    (query, document) evidence pair.

    CRITICAL SEMANTIC DISTINCTION:
    - intent_fit_score and evidence_sufficiency_score are decision scores bounded in [0.0, 1.0].
    - They represent continuous model outputs (e.g., sigmoid activations or normalized scores),
      NOT certified Bayesian probabilities of factual correctness.
    - Raw uncalibrated model logits or diagnostic metadata should be preserved in `metadata`
      rather than mixed with these decision scores.
    - Threshold-derived discrete labels (0/1) and CRAG actions (TRUST, VERIFY_MORE, REJECT_SEARCH)
      belong to the CorrectiveDecisionRouter, NOT this prediction model.
    """

    query_text: str = Field(..., min_length=1, description="Original or rewritten user query string evaluated")
    doc_id: str = Field(..., min_length=1, description="Document identifier of candidate evidence")
    intent_fit_score: float = Field(
        ...,
        description="Continuous decision score in [0.0, 1.0] indicating topical relevance and intent alignment",
    )
    evidence_sufficiency_score: float = Field(
        ...,
        description="Continuous decision score in [0.0, 1.0] indicating whether document provides sufficient factual completeness",
    )
    rationale: Optional[str] = Field(None, description="Optional diagnostic rationale or explanatory text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary audit or diagnostic metadata (raw logits, latency, model checkpoint, etc.)",
    )

    @field_validator("query_text", "doc_id")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty or whitespace only.")
        return v.strip()

    @field_validator("intent_fit_score", "evidence_sufficiency_score")
    @classmethod
    def validate_finite_bounded_score(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Score must be a finite number, got {v}")
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Score must be within [0.0, 1.0], got {v}")
        return float(v)


class CandidateDecision(BaseModel):
    """
    Discrete routing decision for an individual candidate evidence document.

    SEMANTIC CLARIFICATION:
    - intent_fit_label: Independently thresholded intent label (1 if intent_fit_score >= tau_intent else 0).
    - evidence_sufficiency_label: Independently thresholded sufficiency label (1 if evidence_sufficiency_score >= tau_suff else 0).
      Preserves the evaluator's raw prediction without silently zeroing it when intent fails.
    - action: Corrective action enforced by policy. If intent_fit_label == 0, action is ALWAYS REJECT_SEARCH
      under the Intent Dominance Rule, and is_contradictory_override is flagged True.
    """

    prediction: EvaluatorPrediction
    action: CRAGAction = Field(..., description="Chosen CRAG corrective policy action for this document")
    intent_fit_label: int = Field(..., ge=0, le=1, description="Independently thresholded intent-fit label (0 or 1)")
    evidence_sufficiency_label: int = Field(
        ...,
        ge=0,
        le=1,
        description="Independently thresholded evidence-sufficiency label (0 or 1)",
    )
    rationale: Optional[str] = Field(None, description="Explanation of the routing decision")
    is_contradictory_override: bool = Field(
        default=False,
        description="True if raw sufficiency met threshold while intent was below threshold, triggering intent dominance override to REJECT_SEARCH in action",
    )


class QueryDecision(BaseModel):
    """
    Aggregated routing decision across all evaluated candidates for a given query.
    Enforces independent evaluation: one poor candidate does NOT invalidate independent trusted evidence.
    """

    query_text: str = Field(..., min_length=1, description="User query text")
    overall_action: CRAGAction = Field(..., description="Query-level corrective action (TRUST, VERIFY_MORE, REJECT_SEARCH)")
    candidate_decisions: List[CandidateDecision] = Field(default_factory=list, description="Per-candidate decisions")
    trusted_doc_ids: List[str] = Field(default_factory=list, description="Doc IDs admitted into trusted evidence pool")
    verify_doc_ids: List[str] = Field(default_factory=list, description="Doc IDs requiring evidence refinement")
    rejected_doc_ids: List[str] = Field(default_factory=list, description="Doc IDs rejected as unhelpful or off-topic")
    rationale: str = Field(..., description="Justification for the overall query routing decision")
    empty_retrieval: bool = Field(
        default=False,
        description="True if zero candidate documents were provided to the router (distinguishes empty retrieval from all candidates rejected)",
    )


class RefinementResult(BaseModel):
    """
    Output emitted by an evidence refiner when attempting to complete VERIFY_MORE evidence.
    Preserves complete multi-engine retrieval provenance (scores, ranks, metadata).
    """

    query_text: str = Field(..., min_length=1, description="Target query text")
    original_doc_ids: List[str] = Field(
        ...,
        description="Doc IDs of candidates that triggered refinement",
    )
    refined_results: List[RetrievalResult] = Field(
        ...,
        description="Merged and deduplicated set of canonical RetrievalResult contracts with full provenance",
    )
    refined_documents: List[Document] = Field(
        ...,
        description="Merged and deduplicated set of refined evidence documents",
    )
    refinement_strategy: str = Field(
        ...,
        description="Strategy employed: 'adjacent_chunk' | 'secondary_retrieval' | 'recovery_retrieval' | 'skipped_no_targets' | 'mock'",
    )
    is_expanded: bool = Field(
        ...,
        description="True if new unique evidence documents or content were added",
    )
    skipped: bool = Field(
        default=False,
        description="True if refinement was skipped because no verification targets required refinement",
    )
    added_doc_ids: List[str] = Field(
        default_factory=list,
        description="IDs of newly introduced documents",
    )
    added_results: List[RetrievalResult] = Field(
        default_factory=list,
        description="Newly introduced RetrievalResult instances with their retrieval provenance",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Audit and diagnostic metadata",
    )


class RewriteResult(BaseModel):
    """
    Output emitted by a query rewriter when attempting to reformulate a rejected query.
    Preserves original query, rewritten query, transformation strategy, attempt counter,
    and terminal budget/cycle termination flag.
    """

    original_query: str = Field(..., min_length=1, description="Original input query")
    rewritten_query: str = Field(..., min_length=1, description="Rewritten query to be executed")
    strategy: str = Field(
        ...,
        description="Rewriting strategy applied (e.g. 'mock', 'keyword_simplification', 'synonym_expansion', 'terminal_budget_exhausted', 'terminal_cycle_avoided')",
    )
    attempt_index: int = Field(..., ge=1, description="1-indexed rewriting attempt counter")
    is_terminal: bool = Field(
        default=False,
        description="True if no further rewrite attempts are possible (budget exhausted or cycle detected)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Audit and diagnostic metadata",
    )
