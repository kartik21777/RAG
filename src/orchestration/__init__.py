"""
CRAG Orchestration Layer.

Provides runtime evaluation, corrective decision routing, evidence refinement,
query rewriting, and grounded generation for Semantic- and Evidence-Aware CRAG.
"""

from src.orchestration.evaluator import BaseRuntimeEvaluator, MockRuntimeEvaluator
from src.orchestration.refiner import BaseEvidenceRefiner, MockEvidenceRefiner, SecondaryRetrievalRefiner
from src.orchestration.router import CorrectiveDecisionRouter
from src.orchestration.schemas import (
    CandidateDecision,
    EvaluatorPrediction,
    QueryDecision,
    RefinementResult,
)

__all__ = [
    "BaseRuntimeEvaluator",
    "MockRuntimeEvaluator",
    "EvaluatorPrediction",
    "CandidateDecision",
    "QueryDecision",
    "CorrectiveDecisionRouter",
    "RefinementResult",
    "BaseEvidenceRefiner",
    "MockEvidenceRefiner",
    "SecondaryRetrievalRefiner",
]
