"""
CRAG Orchestration Layer.

Provides runtime evaluation, corrective decision routing, evidence refinement,
query rewriting, and grounded generation for Semantic- and Evidence-Aware CRAG.
"""

from src.orchestration.evaluator import BaseRuntimeEvaluator, MockRuntimeEvaluator
from src.orchestration.router import CorrectiveDecisionRouter
from src.orchestration.schemas import CandidateDecision, EvaluatorPrediction, QueryDecision

__all__ = [
    "BaseRuntimeEvaluator",
    "MockRuntimeEvaluator",
    "EvaluatorPrediction",
    "CandidateDecision",
    "QueryDecision",
    "CorrectiveDecisionRouter",
]
