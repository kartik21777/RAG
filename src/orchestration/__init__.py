"""
CRAG Orchestration Layer.

Provides runtime evaluation, corrective decision routing, evidence refinement,
query rewriting, and CRAG pipeline orchestration for Semantic- and Evidence-Aware CRAG.
(Grounded generation belongs to Milestone 6.)
"""

from src.orchestration.evaluator import BaseRuntimeEvaluator, MockRuntimeEvaluator
from src.orchestration.orchestrator import CRAGOrchestrator, OrchestratorResult
from src.orchestration.refiner import BaseEvidenceRefiner, MockEvidenceRefiner, SecondaryRetrievalRefiner
from src.orchestration.rewriter import BaseQueryRewriter, MockQueryRewriter, RuleBasedQueryRewriter
from src.orchestration.router import CorrectiveDecisionRouter
from src.orchestration.schemas import (
    CandidateDecision,
    EvaluatorPrediction,
    QueryDecision,
    RefinementResult,
    RewriteResult,
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
    "RewriteResult",
    "BaseQueryRewriter",
    "MockQueryRewriter",
    "RuleBasedQueryRewriter",
    "CRAGOrchestrator",
    "OrchestratorResult",
]
