"""
Orchestration schemas for the CRAG pipeline.
Defines runtime evaluator predictions and related data structures.
"""

import math
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


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
