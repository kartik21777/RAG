"""
Unit tests for CorrectiveDecisionRouter in src/orchestration/router.py.
Verifies candidate-level routing, independent thresholded labels, intent-dominance overrides,
duplicate rejection, mixed-query validation, empty retrieval diagnostics, and query-level aggregation policies.
"""

import math
import pytest
from src.core.schemas import CRAGAction
from src.orchestration.router import CorrectiveDecisionRouter
from src.orchestration.schemas import EvaluatorPrediction


@pytest.fixture
def default_router():
    return CorrectiveDecisionRouter(
        intent_threshold=0.5,
        sufficiency_threshold=0.5,
        min_trusted_candidates=1,
    )


# -----------------------------------------------------------------------------
# 1. Candidate Routing & Intent Dominance Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_route_candidate_trust_path(default_router):
    pred = EvaluatorPrediction(
        query_text="What is RRF?",
        doc_id="doc_001",
        intent_fit_score=0.9,
        evidence_sufficiency_score=0.85,
    )
    decision = default_router.route_candidate(pred)
    assert decision.action == CRAGAction.TRUST
    assert decision.intent_fit_label == 1
    assert decision.evidence_sufficiency_label == 1
    assert decision.is_contradictory_override is False


@pytest.mark.unit
def test_route_candidate_verify_more_path(default_router):
    pred = EvaluatorPrediction(
        query_text="What is RRF?",
        doc_id="doc_002",
        intent_fit_score=0.88,
        evidence_sufficiency_score=0.35,
    )
    decision = default_router.route_candidate(pred)
    assert decision.action == CRAGAction.VERIFY_MORE
    assert decision.intent_fit_label == 1
    assert decision.evidence_sufficiency_label == 0
    assert decision.is_contradictory_override is False


@pytest.mark.unit
def test_route_candidate_reject_search_normal_path(default_router):
    pred = EvaluatorPrediction(
        query_text="What is RRF?",
        doc_id="doc_003",
        intent_fit_score=0.15,
        evidence_sufficiency_score=0.10,
    )
    decision = default_router.route_candidate(pred)
    assert decision.action == CRAGAction.REJECT_SEARCH
    assert decision.intent_fit_label == 0
    assert decision.evidence_sufficiency_label == 0
    assert decision.is_contradictory_override is False


@pytest.mark.unit
def test_route_candidate_contradictory_intent_dominance_override(default_router):
    # Intent < 0.5, but Sufficiency >= 0.5 (contradictory state)
    # The independent sufficiency label must be PRESERVED as 1, while action is forced to REJECT_SEARCH!
    pred = EvaluatorPrediction(
        query_text="What is RRF?",
        doc_id="doc_004",
        intent_fit_score=0.20,
        evidence_sufficiency_score=0.95,
    )
    decision = default_router.route_candidate(pred)
    assert decision.action == CRAGAction.REJECT_SEARCH
    assert decision.intent_fit_label == 0
    assert decision.evidence_sufficiency_label == 1  # Independently thresholded label preserved
    assert decision.is_contradictory_override is True
    assert "Intent dominance override enforced in action: REJECT_SEARCH" in decision.rationale


# -----------------------------------------------------------------------------
# 2. Threshold Boundaries, Edge Values (0.0, 1.0) & Parameter Validation Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_exact_threshold_boundaries():
    router = CorrectiveDecisionRouter(intent_threshold=0.6, sufficiency_threshold=0.7)

    # Exactly at thresholds -> TRUST
    p_exact = EvaluatorPrediction(
        query_text="query",
        doc_id="doc_exact",
        intent_fit_score=0.6,
        evidence_sufficiency_score=0.7,
    )
    assert router.route_candidate(p_exact).action == CRAGAction.TRUST

    # Just below intent threshold (0.5999) -> REJECT_SEARCH with contradictory override
    p_below_intent = EvaluatorPrediction(
        query_text="query",
        doc_id="doc_below",
        intent_fit_score=0.5999,
        evidence_sufficiency_score=0.7,
    )
    dec = router.route_candidate(p_below_intent)
    assert dec.action == CRAGAction.REJECT_SEARCH
    assert dec.intent_fit_label == 0
    assert dec.evidence_sufficiency_label == 1
    assert dec.is_contradictory_override is True

    # Just below sufficiency threshold (0.6999) -> VERIFY_MORE
    p_below_suff = EvaluatorPrediction(
        query_text="query",
        doc_id="doc_suff",
        intent_fit_score=0.6,
        evidence_sufficiency_score=0.6999,
    )
    assert router.route_candidate(p_below_suff).action == CRAGAction.VERIFY_MORE


@pytest.mark.unit
def test_edge_threshold_values_zero_and_one():
    # Boundary threshold: 0.0 (everything passes)
    router_zero = CorrectiveDecisionRouter(intent_threshold=0.0, sufficiency_threshold=0.0)
    p_zero = EvaluatorPrediction(query_text="query", doc_id="doc_z", intent_fit_score=0.0, evidence_sufficiency_score=0.0)
    assert router_zero.route_candidate(p_zero).action == CRAGAction.TRUST

    # Boundary threshold: 1.0 (only perfect 1.0 scores pass)
    router_one = CorrectiveDecisionRouter(intent_threshold=1.0, sufficiency_threshold=1.0)
    p_sub = EvaluatorPrediction(query_text="query", doc_id="doc_sub", intent_fit_score=0.9999, evidence_sufficiency_score=1.0)
    dec_sub = router_one.route_candidate(p_sub)
    assert dec_sub.action == CRAGAction.REJECT_SEARCH
    assert dec_sub.intent_fit_label == 0
    assert dec_sub.evidence_sufficiency_label == 1
    assert dec_sub.is_contradictory_override is True

    p_perfect = EvaluatorPrediction(query_text="query", doc_id="doc_perf", intent_fit_score=1.0, evidence_sufficiency_score=1.0)
    assert router_one.route_candidate(p_perfect).action == CRAGAction.TRUST


@pytest.mark.unit
def test_router_invalid_initialization_parameters():
    # Out of range < 0.0
    with pytest.raises(ValueError):
        CorrectiveDecisionRouter(intent_threshold=-0.01)

    # Out of range > 1.0
    with pytest.raises(ValueError):
        CorrectiveDecisionRouter(sufficiency_threshold=1.01)

    # NaN threshold rejected
    with pytest.raises(ValueError) as exc_nan:
        CorrectiveDecisionRouter(intent_threshold=float("nan"))
    assert "must be a finite number" in str(exc_nan.value)

    # Positive Infinity threshold rejected
    with pytest.raises(ValueError) as exc_inf:
        CorrectiveDecisionRouter(intent_threshold=float("inf"))
    assert "must be a finite number" in str(exc_inf.value)

    # Negative Infinity threshold rejected
    with pytest.raises(ValueError) as exc_ninf:
        CorrectiveDecisionRouter(sufficiency_threshold=float("-inf"))
    assert "must be a finite number" in str(exc_ninf.value)

    # min_trusted_candidates < 1 rejected
    with pytest.raises(ValueError):
        CorrectiveDecisionRouter(min_trusted_candidates=0)


@pytest.mark.unit
def test_route_candidate_invalid_input_type(default_router):
    with pytest.raises(TypeError):
        default_router.route_candidate("not a prediction")  # type: ignore


# -----------------------------------------------------------------------------
# 3. Query-Level Aggregation & Mixed Candidates (Section 8A)
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_query_aggregation_simultaneous_mixed_candidates(default_router):
    # Simultaneous mix of TRUST, VERIFY_MORE, and REJECT_SEARCH
    preds = [
        EvaluatorPrediction(query_text="query", doc_id="doc_trust", intent_fit_score=0.92, evidence_sufficiency_score=0.88),   # TRUST
        EvaluatorPrediction(query_text="query", doc_id="doc_verify", intent_fit_score=0.85, evidence_sufficiency_score=0.20),  # VERIFY_MORE
        EvaluatorPrediction(query_text="query", doc_id="doc_reject", intent_fit_score=0.10, evidence_sufficiency_score=0.10),  # REJECT_SEARCH
    ]
    decisions = default_router.route_candidates(preds)
    q_decision = default_router.aggregate_query_decisions("query", decisions)

    # All candidate decisions must be preserved
    assert len(q_decision.candidate_decisions) == 3
    assert q_decision.candidate_decisions[0].action == CRAGAction.TRUST
    assert q_decision.candidate_decisions[1].action == CRAGAction.VERIFY_MORE
    assert q_decision.candidate_decisions[2].action == CRAGAction.REJECT_SEARCH

    # Overall action is TRUST (one poor candidate does not invalidate independent trusted candidate)
    assert q_decision.overall_action == CRAGAction.TRUST
    assert q_decision.trusted_doc_ids == ["doc_trust"]
    assert q_decision.verify_doc_ids == ["doc_verify"]
    assert q_decision.rejected_doc_ids == ["doc_reject"]
    assert q_decision.empty_retrieval is False


@pytest.mark.unit
def test_query_aggregation_no_trust_but_verify_more(default_router):
    preds = [
        EvaluatorPrediction(query_text="query", doc_id="doc_1", intent_fit_score=0.8, evidence_sufficiency_score=0.3), # VERIFY_MORE
        EvaluatorPrediction(query_text="query", doc_id="doc_2", intent_fit_score=0.1, evidence_sufficiency_score=0.1), # REJECT_SEARCH
    ]
    decisions = default_router.route_candidates(preds)
    q_decision = default_router.aggregate_query_decisions("query", decisions)

    assert q_decision.overall_action == CRAGAction.VERIFY_MORE
    assert q_decision.trusted_doc_ids == []
    assert q_decision.verify_doc_ids == ["doc_1"]
    assert q_decision.rejected_doc_ids == ["doc_2"]
    assert q_decision.empty_retrieval is False


@pytest.mark.unit
def test_query_aggregation_all_rejected_distinguished(default_router):
    preds = [
        EvaluatorPrediction(query_text="query", doc_id="doc_1", intent_fit_score=0.1, evidence_sufficiency_score=0.1),
        EvaluatorPrediction(query_text="query", doc_id="doc_2", intent_fit_score=0.2, evidence_sufficiency_score=0.9), # Contradictory -> REJECT
    ]
    decisions = default_router.route_candidates(preds)
    q_decision = default_router.aggregate_query_decisions("query", decisions)

    assert q_decision.overall_action == CRAGAction.REJECT_SEARCH
    assert q_decision.trusted_doc_ids == []
    assert q_decision.verify_doc_ids == []
    assert len(q_decision.rejected_doc_ids) == 2
    assert q_decision.empty_retrieval is False
    assert q_decision.rationale.startswith("ALL_CANDIDATES_REJECTED:")


@pytest.mark.unit
def test_query_aggregation_empty_retrieval_distinguished(default_router):
    q_decision = default_router.aggregate_query_decisions("query", [])
    assert q_decision.overall_action == CRAGAction.REJECT_SEARCH
    assert q_decision.trusted_doc_ids == []
    assert q_decision.empty_retrieval is True
    assert q_decision.rationale.startswith("EMPTY_RETRIEVAL:")


# -----------------------------------------------------------------------------
# 4. Strict Duplicate Candidate Rejection Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_query_aggregation_conflicting_duplicates_rejected(default_router):
    # Two predictions for the same doc_id that disagree on action
    preds = [
        EvaluatorPrediction(query_text="query", doc_id="doc_dup", intent_fit_score=0.9, evidence_sufficiency_score=0.9), # TRUST
        EvaluatorPrediction(query_text="query", doc_id="doc_dup", intent_fit_score=0.1, evidence_sufficiency_score=0.1), # REJECT_SEARCH
    ]
    decisions = default_router.route_candidates(preds)
    with pytest.raises(ValueError) as exc:
        default_router.aggregate_query_decisions("query", decisions)
    assert "Conflicting duplicate predictions detected for document 'doc_dup'" in str(exc.value)
    assert "Duplicate candidate document IDs are strictly prohibited" in str(exc.value)


@pytest.mark.unit
def test_query_aggregation_identical_duplicates_rejected(default_router):
    # Two identical predictions for the same doc_id must also be strictly rejected
    preds = [
        EvaluatorPrediction(query_text="query", doc_id="doc_dup", intent_fit_score=0.9, evidence_sufficiency_score=0.9),
        EvaluatorPrediction(query_text="query", doc_id="doc_dup", intent_fit_score=0.9, evidence_sufficiency_score=0.9),
    ]
    decisions = default_router.route_candidates(preds)
    with pytest.raises(ValueError) as exc:
        default_router.aggregate_query_decisions("query", decisions)
    assert "Duplicate predictions detected for document 'doc_dup'" in str(exc.value)
    assert "All candidate document IDs must be unique within a single query" in str(exc.value)


# -----------------------------------------------------------------------------
# 5. Mixed-Query Validation Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_query_aggregation_mismatched_query_rejected(default_router):
    preds = [
        EvaluatorPrediction(query_text="target query", doc_id="doc_1", intent_fit_score=0.9, evidence_sufficiency_score=0.9),
        EvaluatorPrediction(query_text="completely different query", doc_id="doc_2", intent_fit_score=0.9, evidence_sufficiency_score=0.9),
    ]
    decisions = default_router.route_candidates(preds)
    with pytest.raises(ValueError) as exc:
        default_router.aggregate_query_decisions("target query", decisions)
    assert "Mismatched query text in candidate decision for doc 'doc_2'" in str(exc.value)
    assert "expected 'target query', got 'completely different query'" in str(exc.value)


@pytest.mark.unit
def test_query_aggregation_invalid_query_text(default_router):
    with pytest.raises(ValueError):
        default_router.aggregate_query_decisions("", [])

    with pytest.raises(ValueError):
        default_router.aggregate_query_decisions("   ", [])
