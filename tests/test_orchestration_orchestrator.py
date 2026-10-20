"""
Tests for CRAG Pipeline Orchestrator (Option C Milestone 5).

Uses deterministic, CPU-only mocks for the retriever, evaluator, refiner, and
rewriter so that the orchestrator logic can be exercised without a loaded index,
GPU, or network access.

Test categories:
1. CRAGOrchestrator initialization validation
2. TRUST path (evaluator returns high scores for all candidates)
3. VERIFY_MORE path (evaluator returns intent-fit=1, suff=0 for all candidates)
4. REJECT_SEARCH path without a rewriter (terminates immediately)
5. REJECT_SEARCH -> rewrite -> TRUST (single rewrite resolves the query)
6. Rewrite budget exhaustion (max_rewrite_attempts honored)
7. Rewriter cycle detection (rewriter returns is_terminal=True)
8. Empty retrieval (facade returns no results)
9. run_pipeline input validation (empty / whitespace queries rejected)
10. audit trail integrity (OrchestratorResult fields populated correctly)
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.schemas import CRAGAction, Document, RetrievalResult
from src.orchestration.evaluator import MockRuntimeEvaluator
from src.orchestration.orchestrator import CRAGOrchestrator, OrchestratorResult
from src.orchestration.refiner import MockEvidenceRefiner, BaseEvidenceRefiner
from src.orchestration.rewriter import MockQueryRewriter, BaseQueryRewriter
from src.orchestration.router import CorrectiveDecisionRouter
from src.orchestration.schemas import (
    RefinementResult,
    RewriteResult,
)
from src.retrieval.facade import CRAGRetrieverFacade


# ---------------------------------------------------------------------------
# Test helpers / stubs
# ---------------------------------------------------------------------------

def make_doc(doc_id: str, text: str = "sample text") -> Document:
    return Document(doc_id=doc_id, text=text, metadata={})


def make_retrieval_result(doc_id: str, score: float = 1.0, rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        document=make_doc(doc_id),
        fused_score=score,
        fused_rank=rank,
        sparse_rank=None,
        dense_rank=None,
        sparse_score=None,
        dense_score=None,
    )


class FakeRetriever(CRAGRetrieverFacade):
    """
    CRAGRetrieverFacade stub that returns a fixed list of results per query,
    without requiring an indexed corpus or ML models.
    """

    def __init__(self, results_by_query: Optional[Dict[str, List[RetrievalResult]]] = None,
                 default_results: Optional[List[RetrievalResult]] = None):
        # Bypass parent __init__ entirely
        self._results_by_query: Dict[str, List[RetrievalResult]] = results_by_query or {}
        self._default_results: List[RetrievalResult] = default_results or []

    @property
    def is_indexed(self) -> bool:
        return True

    def retrieve(self, query, top_k: int = 5) -> List[RetrievalResult]:
        query_str = query if isinstance(query, str) else query.query_text
        return list(self._results_by_query.get(query_str, self._default_results))[:top_k]


def make_orchestrator(
    *,
    results_by_query: Optional[Dict[str, List[RetrievalResult]]] = None,
    default_results: Optional[List[RetrievalResult]] = None,
    pair_scores: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    default_scores: Tuple[float, float] = (0.0, 0.0),
    refiner: Optional[BaseEvidenceRefiner] = None,
    rewriter: Optional[BaseQueryRewriter] = None,
    top_k: int = 3,
    max_rewrite_attempts: int = 2,
) -> CRAGOrchestrator:
    retriever = FakeRetriever(
        results_by_query=results_by_query,
        default_results=default_results,
    )
    evaluator = MockRuntimeEvaluator(
        pair_scores=pair_scores or {},
        default_scores=default_scores,
    )
    router = CorrectiveDecisionRouter()
    if refiner is None:
        refiner = MockEvidenceRefiner()
    return CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
        rewriter=rewriter,
        top_k=top_k,
        max_rewrite_attempts=max_rewrite_attempts,
    )


# ---------------------------------------------------------------------------
# 1. Initialization validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_orchestrator_type_validation():
    """Constructor raises TypeError for wrong argument types."""
    with pytest.raises(TypeError, match="retriever must be CRAGRetrieverFacade"):
        CRAGOrchestrator(
            retriever=object(),
            evaluator=MockRuntimeEvaluator(),
            router=CorrectiveDecisionRouter(),
            refiner=MockEvidenceRefiner(),
        )

    fake_retriever = FakeRetriever()
    with pytest.raises(TypeError, match="evaluator must be BaseRuntimeEvaluator"):
        CRAGOrchestrator(
            retriever=fake_retriever,
            evaluator=object(),
            router=CorrectiveDecisionRouter(),
            refiner=MockEvidenceRefiner(),
        )

    with pytest.raises(TypeError, match="router must be CorrectiveDecisionRouter"):
        CRAGOrchestrator(
            retriever=fake_retriever,
            evaluator=MockRuntimeEvaluator(),
            router=object(),
            refiner=MockEvidenceRefiner(),
        )

    with pytest.raises(TypeError, match="refiner must be BaseEvidenceRefiner"):
        CRAGOrchestrator(
            retriever=fake_retriever,
            evaluator=MockRuntimeEvaluator(),
            router=CorrectiveDecisionRouter(),
            refiner=object(),
        )

    with pytest.raises(TypeError, match="rewriter must be BaseQueryRewriter or None"):
        CRAGOrchestrator(
            retriever=fake_retriever,
            evaluator=MockRuntimeEvaluator(),
            router=CorrectiveDecisionRouter(),
            refiner=MockEvidenceRefiner(),
            rewriter=object(),
        )


@pytest.mark.unit
def test_orchestrator_parameter_bounds():
    """Constructor raises ValueError for out-of-range top_k or max_rewrite_attempts."""
    fake_retriever = FakeRetriever()
    with pytest.raises(ValueError, match="top_k"):
        CRAGOrchestrator(
            retriever=fake_retriever,
            evaluator=MockRuntimeEvaluator(),
            router=CorrectiveDecisionRouter(),
            refiner=MockEvidenceRefiner(),
            top_k=0,
        )

    with pytest.raises(ValueError, match="max_rewrite_attempts"):
        CRAGOrchestrator(
            retriever=fake_retriever,
            evaluator=MockRuntimeEvaluator(),
            router=CorrectiveDecisionRouter(),
            refiner=MockEvidenceRefiner(),
            max_rewrite_attempts=0,
        )


# ---------------------------------------------------------------------------
# 2. TRUST path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_trust_path():
    """When evaluator returns (1.0, 1.0) for all candidates, action is TRUST."""
    doc_id = "doc_trust_01"
    results = [make_retrieval_result(doc_id)]
    query = "What is BM25?"

    orch = make_orchestrator(
        default_results=results,
        pair_scores={(query, doc_id): (1.0, 1.0)},
    )
    result = orch.run_pipeline(query)

    assert result.final_action == CRAGAction.TRUST
    assert result.original_query == query
    assert result.final_query == query
    assert any(r.document.doc_id == doc_id for r in result.final_results)
    assert result.rewrite_history == []
    assert result.refinement_result is None
    assert result.metadata["path"] == "trust"
    assert result.metadata["loop_iterations"] == 1


# ---------------------------------------------------------------------------
# 3. VERIFY_MORE path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_verify_more_path():
    """When evaluator returns (1.0, 0.0), action is VERIFY_MORE and refiner is called."""
    doc_id = "doc_verify_01"
    results = [make_retrieval_result(doc_id)]
    query = "Explain RRF"
    # provide refined document
    refined_doc = make_retrieval_result("doc_refined_01", score=0.8)
    refiner = MockEvidenceRefiner(
        expansion_map={query: [refined_doc]},
        strategy_name="mock_secondary",
    )

    orch = make_orchestrator(
        default_results=results,
        pair_scores={(query, doc_id): (1.0, 0.0)},
        refiner=refiner,
    )
    result = orch.run_pipeline(query)

    assert result.final_action == CRAGAction.VERIFY_MORE
    assert result.refinement_result is not None
    assert result.metadata["path"] == "verify_more"
    assert result.metadata["loop_iterations"] == 1
    assert result.rewrite_history == []


# ---------------------------------------------------------------------------
# 4. REJECT_SEARCH without rewriter
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reject_search_without_rewriter():
    """REJECT_SEARCH terminates immediately when no rewriter is configured."""
    doc_id = "doc_reject_01"
    results = [make_retrieval_result(doc_id)]
    query = "completely off-topic query"

    orch = make_orchestrator(
        default_results=results,
        default_scores=(0.0, 0.0),  # all reject
        rewriter=None,
    )
    result = orch.run_pipeline(query)

    assert result.final_action == CRAGAction.REJECT_SEARCH
    assert result.metadata["path"] == "reject_search_terminal"
    assert result.metadata["reason"] == "no_rewriter_configured"
    assert result.rewrite_history == []
    assert result.rewrite_results == []


# ---------------------------------------------------------------------------
# 5. REJECT_SEARCH -> rewrite -> TRUST
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reject_then_rewrite_then_trust():
    """A single rewrite resolves REJECT_SEARCH into TRUST on the next iteration."""
    original_query = "quantum computing BM25"
    rewritten_query = "quantum BM25"
    doc_id = "doc_quantum_01"

    # First query: doc rejected; second (rewritten) query: doc trusted
    results_map = {
        original_query: [make_retrieval_result(doc_id)],
        rewritten_query: [make_retrieval_result(doc_id)],
    }
    pair_scores = {
        (original_query, doc_id): (0.0, 0.0),  # rejected
        (rewritten_query, doc_id): (1.0, 1.0),  # trusted
    }

    rewriter = MockQueryRewriter(
        rewrite_map={original_query: rewritten_query},
        max_attempts=3,
    )
    orch = make_orchestrator(
        results_by_query=results_map,
        pair_scores=pair_scores,
        rewriter=rewriter,
        max_rewrite_attempts=3,
    )
    result = orch.run_pipeline(original_query)

    assert result.final_action == CRAGAction.TRUST
    assert result.final_query == rewritten_query
    assert result.original_query == original_query
    assert result.rewrite_history == [rewritten_query]
    assert len(result.rewrite_results) == 1
    assert result.metadata["loop_iterations"] == 2


# ---------------------------------------------------------------------------
# 6. Rewrite budget exhaustion
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rewrite_budget_exhausted():
    """After max_rewrite_attempts, orchestrator returns REJECT_SEARCH."""
    original_query = "hard query"
    rewritten_queries = ["hard query v2", "hard query v3", "hard query v4"]
    doc_id = "doc_hard"

    results_map: Dict[str, List[RetrievalResult]] = {}
    pair_scores: Dict[Tuple[str, str], Tuple[float, float]] = {}

    all_queries = [original_query] + rewritten_queries
    for q in all_queries:
        results_map[q] = [make_retrieval_result(doc_id)]
        pair_scores[(q, doc_id)] = (0.0, 0.0)

    rewriter = MockQueryRewriter(
        sequential_rewrites=rewritten_queries,
        max_attempts=len(rewritten_queries),
    )
    orch = make_orchestrator(
        results_by_query=results_map,
        pair_scores=pair_scores,
        rewriter=rewriter,
        max_rewrite_attempts=2,
    )
    result = orch.run_pipeline(original_query)

    assert result.final_action == CRAGAction.REJECT_SEARCH
    assert result.metadata["path"] == "reject_search_terminal"
    assert result.metadata["reason"] == "rewrite_budget_exhausted"
    assert len(result.rewrite_history) == 2


# ---------------------------------------------------------------------------
# 7. Rewriter cycle detection / is_terminal
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rewriter_cycle_terminates_pipeline():
    """If the rewriter returns is_terminal=True, the orchestrator stops immediately."""
    query = "query that cycles"
    doc_id = "doc_cycle"

    # Rewriter configured to produce the same rewrite as current query (no-op -> cycle)
    rewriter = MockQueryRewriter(
        rewrite_map={query: query},  # no-op -> cycle -> is_terminal=True from MockQueryRewriter
        max_attempts=3,
    )
    orch = make_orchestrator(
        default_results=[make_retrieval_result(doc_id)],
        default_scores=(0.0, 0.0),
        rewriter=rewriter,
        max_rewrite_attempts=5,
    )
    result = orch.run_pipeline(query)

    assert result.final_action == CRAGAction.REJECT_SEARCH
    assert result.metadata["path"] == "reject_search_rewriter_terminal"
    assert "rewriter_strategy" in result.metadata
    assert result.rewrite_history == []  # cycle detected before appending


# ---------------------------------------------------------------------------
# 8. Empty retrieval
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_empty_retrieval_returns_reject_search():
    """Empty retrieval set produces REJECT_SEARCH with empty_retrieval=True."""
    query = "unanswerable query"
    orch = make_orchestrator(
        default_results=[],  # retriever returns nothing
        rewriter=None,
    )
    result = orch.run_pipeline(query)

    assert result.final_action == CRAGAction.REJECT_SEARCH
    assert result.query_decision.empty_retrieval is True
    assert result.final_results == []
    assert result.metadata["loop_iterations"] == 1


# ---------------------------------------------------------------------------
# 9. Input validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_run_pipeline_rejects_empty_query():
    """ValueError is raised for empty, whitespace-only, or non-string queries."""
    orch = make_orchestrator()

    with pytest.raises(ValueError):
        orch.run_pipeline("")

    with pytest.raises(ValueError):
        orch.run_pipeline("   ")

    with pytest.raises(ValueError):
        orch.run_pipeline(None)


# ---------------------------------------------------------------------------
# 10. Audit trail integrity
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_audit_trail_populated_on_trust():
    """OrchestratorResult fields are correctly populated on a TRUST outcome."""
    query = "What is FAISS?"
    doc_id = "doc_faiss_01"
    results = [make_retrieval_result(doc_id, score=0.95)]

    orch = make_orchestrator(
        default_results=results,
        pair_scores={(query, doc_id): (1.0, 1.0)},
        top_k=1,
    )
    result = orch.run_pipeline(query)

    assert isinstance(result, OrchestratorResult)
    assert result.original_query == query
    assert result.final_query == query
    assert result.final_action == CRAGAction.TRUST
    assert result.final_results
    assert result.query_decision.overall_action == CRAGAction.TRUST
    assert result.refinement_result is None
    assert result.rewrite_history == []
    assert result.rewrite_results == []
    assert "loop_iterations" in result.metadata
    assert "path" in result.metadata


@pytest.mark.unit
def test_audit_trail_populated_on_verify_more():
    """OrchestratorResult fields are correctly populated on a VERIFY_MORE outcome."""
    query = "Explain dense retrieval"
    doc_id = "doc_dense_01"

    orch = make_orchestrator(
        default_results=[make_retrieval_result(doc_id)],
        pair_scores={(query, doc_id): (1.0, 0.0)},
    )
    result = orch.run_pipeline(query)

    assert result.final_action == CRAGAction.VERIFY_MORE
    assert result.refinement_result is not None
    assert isinstance(result.refinement_result, RefinementResult)
    assert result.metadata["path"] == "verify_more"
    assert "refinement_strategy" in result.metadata
    assert "is_expanded" in result.metadata


@pytest.mark.unit
def test_rewrite_history_grows_correctly():
    """Each successive rewrite appends to rewrite_history in order."""
    original_query = "semantic search"
    r1 = "semantic retrieval"
    r2 = "neural search"
    doc_id = "doc_sem"

    results_map = {
        original_query: [make_retrieval_result(doc_id)],
        r1: [make_retrieval_result(doc_id)],
        r2: [make_retrieval_result(doc_id)],
    }
    # All queries rejected until budget exhausted
    pair_scores = {
        (original_query, doc_id): (0.0, 0.0),
        (r1, doc_id): (0.0, 0.0),
        (r2, doc_id): (0.0, 0.0),
    }
    rewriter = MockQueryRewriter(
        sequential_rewrites=[r1, r2],
        max_attempts=5,
    )
    orch = make_orchestrator(
        results_by_query=results_map,
        pair_scores=pair_scores,
        rewriter=rewriter,
        max_rewrite_attempts=2,
    )
    result = orch.run_pipeline(original_query)

    assert result.rewrite_history == [r1, r2]
    assert len(result.rewrite_results) == 2
    assert result.final_action == CRAGAction.REJECT_SEARCH


# ---------------------------------------------------------------------------
# 11. Final Allowed Rewrite Handling Regression Tests
# ---------------------------------------------------------------------------

class CountingRetriever(CRAGRetrieverFacade):
    """Facade spy that tracks retrieve() calls."""

    def __init__(
        self,
        results_by_query: Optional[Dict[str, List[RetrievalResult]]] = None,
        default_results: Optional[List[RetrievalResult]] = None,
    ):
        self._results_by_query = results_by_query or {}
        self._default_results = default_results or []
        self.retrieve_calls: List[str] = []

    @property
    def is_indexed(self) -> bool:
        return True

    def retrieve(self, query, top_k: int = 5) -> List[RetrievalResult]:
        query_str = query if isinstance(query, str) else query.query_text
        self.retrieve_calls.append(query_str)
        return list(self._results_by_query.get(query_str, self._default_results))[:top_k]


class CountingRewriter(BaseQueryRewriter):
    """Rewriter spy that tracks rewrite() calls."""

    def __init__(self, wrapped: BaseQueryRewriter):
        self.wrapped = wrapped
        self.rewrite_calls: List[Dict[str, Any]] = []

    def rewrite(
        self,
        query: str,
        history: Optional[List[str]] = None,
        rejected_candidates: Optional[List[RetrievalResult]] = None,
        original_query: Optional[str] = None,
    ) -> RewriteResult:
        self.rewrite_calls.append({
            "query": query,
            "history": list(history) if history else [],
            "original_query": original_query,
        })
        return self.wrapped.rewrite(
            query=query,
            history=history,
            rejected_candidates=rejected_candidates,
            original_query=original_query,
        )


@pytest.mark.unit
def test_final_allowed_rewrite_limit_one_produces_trust():
    """With max_rewrite_attempts=1 and rewriter.max_attempts=1, the rewrite is retrieved and produces TRUST."""
    orig_q = "initial query"
    rewrite_q = "clarified query"
    doc_id = "doc_1"

    retriever = CountingRetriever(
        results_by_query={
            orig_q: [make_retrieval_result(doc_id)],
            rewrite_q: [make_retrieval_result(doc_id)],
        }
    )
    evaluator = MockRuntimeEvaluator(
        pair_scores={
            (orig_q, doc_id): (0.0, 0.0),
            (rewrite_q, doc_id): (1.0, 1.0),
        }
    )
    base_rewriter = MockQueryRewriter(
        rewrite_map={orig_q: rewrite_q},
        max_attempts=1,
    )
    rewriter = CountingRewriter(base_rewriter)

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=CorrectiveDecisionRouter(),
        refiner=MockEvidenceRefiner(),
        rewriter=rewriter,
        max_rewrite_attempts=1,
    )
    result = orch.run_pipeline(orig_q)

    assert result.final_action == CRAGAction.TRUST
    assert result.final_query == rewrite_q
    assert result.original_query == orig_q
    assert result.rewrite_history == [rewrite_q]
    assert len(result.rewrite_results) == 1
    assert result.rewrite_results[0].is_terminal is True
    assert retriever.retrieve_calls == [orig_q, rewrite_q]
    assert len(rewriter.rewrite_calls) == 1


@pytest.mark.unit
def test_final_allowed_rewrite_limit_one_rejected_terminates_without_extra_rewrite():
    """With limits=1, if final rewrite is also rejected, pipeline terminates without an extra rewrite call."""
    orig_q = "stubborn query"
    rewrite_q = "reformulated stubborn query"
    doc_id = "doc_stubborn"

    retriever = CountingRetriever(
        results_by_query={
            orig_q: [make_retrieval_result(doc_id)],
            rewrite_q: [make_retrieval_result(doc_id)],
        }
    )
    evaluator = MockRuntimeEvaluator(
        pair_scores={
            (orig_q, doc_id): (0.0, 0.0),
            (rewrite_q, doc_id): (0.0, 0.0),
        }
    )
    base_rewriter = MockQueryRewriter(
        rewrite_map={orig_q: rewrite_q},
        max_attempts=1,
    )
    rewriter = CountingRewriter(base_rewriter)

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=CorrectiveDecisionRouter(),
        refiner=MockEvidenceRefiner(),
        rewriter=rewriter,
        max_rewrite_attempts=1,
    )
    result = orch.run_pipeline(orig_q)

    assert result.final_action == CRAGAction.REJECT_SEARCH
    assert result.final_query == rewrite_q
    assert result.metadata["path"] == "reject_search_terminal"
    assert result.metadata["reason"] == "rewrite_budget_exhausted"
    assert result.rewrite_history == [rewrite_q]
    assert retriever.retrieve_calls == [orig_q, rewrite_q]
    assert len(rewriter.rewrite_calls) == 1


@pytest.mark.unit
def test_final_allowed_rewrite_limit_two_produces_trust():
    """With limits=2, second rewrite is terminal but retrieved, evaluated, and produces TRUST."""
    q0 = "initial question"
    q1 = "intermediate rewrite"
    q2 = "golden rewrite"
    doc_id = "doc_golden"

    retriever = CountingRetriever(
        results_by_query={
            q0: [make_retrieval_result(doc_id)],
            q1: [make_retrieval_result(doc_id)],
            q2: [make_retrieval_result(doc_id)],
        }
    )
    evaluator = MockRuntimeEvaluator(
        pair_scores={
            (q0, doc_id): (0.0, 0.0),
            (q1, doc_id): (0.0, 0.0),
            (q2, doc_id): (1.0, 1.0),
        }
    )
    base_rewriter = MockQueryRewriter(
        sequential_rewrites=[q1, q2],
        max_attempts=2,
    )
    rewriter = CountingRewriter(base_rewriter)

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=CorrectiveDecisionRouter(),
        refiner=MockEvidenceRefiner(),
        rewriter=rewriter,
        max_rewrite_attempts=2,
    )
    result = orch.run_pipeline(q0)

    assert result.final_action == CRAGAction.TRUST
    assert result.final_query == q2
    assert result.rewrite_history == [q1, q2]
    assert len(result.rewrite_results) == 2
    assert result.rewrite_results[0].is_terminal is False
    assert result.rewrite_results[1].is_terminal is True
    assert retriever.retrieve_calls == [q0, q1, q2]
    assert len(rewriter.rewrite_calls) == 2


@pytest.mark.unit
def test_final_allowed_rewrite_limit_two_rejected_terminates_without_extra_rewrite():
    """With limits=2, if second rewrite is rejected, terminates without making a 3rd rewrite call."""
    q0 = "hopeless q"
    q1 = "hopeless q v1"
    q2 = "hopeless q v2"
    doc_id = "doc_none"

    retriever = CountingRetriever(
        results_by_query={
            q0: [make_retrieval_result(doc_id)],
            q1: [make_retrieval_result(doc_id)],
            q2: [make_retrieval_result(doc_id)],
        }
    )
    evaluator = MockRuntimeEvaluator(
        pair_scores={
            (q0, doc_id): (0.0, 0.0),
            (q1, doc_id): (0.0, 0.0),
            (q2, doc_id): (0.0, 0.0),
        }
    )
    base_rewriter = MockQueryRewriter(
        sequential_rewrites=[q1, q2],
        max_attempts=2,
    )
    rewriter = CountingRewriter(base_rewriter)

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=CorrectiveDecisionRouter(),
        refiner=MockEvidenceRefiner(),
        rewriter=rewriter,
        max_rewrite_attempts=2,
    )
    result = orch.run_pipeline(q0)

    assert result.final_action == CRAGAction.REJECT_SEARCH
    assert result.final_query == q2
    assert result.metadata["path"] == "reject_search_terminal"
    assert result.metadata["reason"] == "rewrite_budget_exhausted"
    assert result.rewrite_history == [q1, q2]
    assert retriever.retrieve_calls == [q0, q1, q2]
    assert len(rewriter.rewrite_calls) == 2


@pytest.mark.unit
def test_cycle_terminal_avoids_extra_retrieval_and_preserves_query():
    """Cycle-terminal rewrite aborts immediately without attempting retrieval on invalid candidate."""
    orig_q = "cyclic input"
    doc_id = "doc_cycle"

    retriever = CountingRetriever(
        results_by_query={orig_q: [make_retrieval_result(doc_id)]}
    )
    evaluator = MockRuntimeEvaluator(default_scores=(0.0, 0.0))
    base_rewriter = MockQueryRewriter(
        rewrite_map={orig_q: orig_q},
        max_attempts=3,
    )
    rewriter = CountingRewriter(base_rewriter)

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=CorrectiveDecisionRouter(),
        refiner=MockEvidenceRefiner(),
        rewriter=rewriter,
        max_rewrite_attempts=3,
    )
    result = orch.run_pipeline(orig_q)

    assert result.final_action == CRAGAction.REJECT_SEARCH
    assert result.metadata["path"] == "reject_search_rewriter_terminal"
    assert result.metadata["rewriter_strategy"] == "terminal_cycle_avoided"
    assert result.final_query == orig_q
    assert result.rewrite_history == []
    assert retriever.retrieve_calls == [orig_q]
    assert len(rewriter.rewrite_calls) == 1


@pytest.mark.unit
def test_final_allowed_rewrite_limit_three_produces_trust():
    """With limits=3, 3rd rewrite is terminal, retrieved, evaluated, and produces TRUST (retrievals=4, rewrites=3)."""
    q0 = "initial q"
    q1 = "q v1"
    q2 = "q v2"
    q3 = "q v3 gold"
    doc_id = "doc_q3"

    retriever = CountingRetriever(
        results_by_query={
            q0: [make_retrieval_result(doc_id)],
            q1: [make_retrieval_result(doc_id)],
            q2: [make_retrieval_result(doc_id)],
            q3: [make_retrieval_result(doc_id)],
        }
    )
    evaluator = MockRuntimeEvaluator(
        pair_scores={
            (q0, doc_id): (0.0, 0.0),
            (q1, doc_id): (0.0, 0.0),
            (q2, doc_id): (0.0, 0.0),
            (q3, doc_id): (1.0, 1.0),
        }
    )
    base_rewriter = MockQueryRewriter(
        sequential_rewrites=[q1, q2, q3],
        max_attempts=3,
    )
    rewriter = CountingRewriter(base_rewriter)

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=CorrectiveDecisionRouter(),
        refiner=MockEvidenceRefiner(),
        rewriter=rewriter,
        max_rewrite_attempts=3,
    )
    result = orch.run_pipeline(q0)

    assert result.final_action == CRAGAction.TRUST
    assert result.final_query == q3
    assert result.original_query == q0
    assert result.rewrite_history == [q1, q2, q3]
    assert len(result.rewrite_results) == 3
    assert result.rewrite_results[0].is_terminal is False
    assert result.rewrite_results[1].is_terminal is False
    assert result.rewrite_results[2].is_terminal is True
    # Explicit assert: 4 retrievals (initial + 3 rewrites)
    assert retriever.retrieve_calls == [q0, q1, q2, q3]
    # Explicit assert: exactly 3 rewrites (never exceeds budget, no 4th call)
    assert len(rewriter.rewrite_calls) == 3


@pytest.mark.unit
def test_final_allowed_rewrite_limit_three_rejected_terminates_without_extra_rewrite():
    """With limits=3, 3rd rewrite remains rejected; terminates without a 4th rewrite (retrievals=4, rewrites=3)."""
    q0 = "hard q"
    q1 = "hard q v1"
    q2 = "hard q v2"
    q3 = "hard q v3"
    doc_id = "doc_hard_none"

    retriever = CountingRetriever(
        results_by_query={
            q0: [make_retrieval_result(doc_id)],
            q1: [make_retrieval_result(doc_id)],
            q2: [make_retrieval_result(doc_id)],
            q3: [make_retrieval_result(doc_id)],
        }
    )
    evaluator = MockRuntimeEvaluator(
        pair_scores={
            (q0, doc_id): (0.0, 0.0),
            (q1, doc_id): (0.0, 0.0),
            (q2, doc_id): (0.0, 0.0),
            (q3, doc_id): (0.0, 0.0),
        }
    )
    base_rewriter = MockQueryRewriter(
        sequential_rewrites=[q1, q2, q3],
        max_attempts=3,
    )
    rewriter = CountingRewriter(base_rewriter)

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=CorrectiveDecisionRouter(),
        refiner=MockEvidenceRefiner(),
        rewriter=rewriter,
        max_rewrite_attempts=3,
    )
    result = orch.run_pipeline(q0)

    assert result.final_action == CRAGAction.REJECT_SEARCH
    assert result.final_query == q3
    assert result.original_query == q0
    assert result.metadata["path"] == "reject_search_terminal"
    assert result.metadata["reason"] == "rewrite_budget_exhausted"
    assert result.rewrite_history == [q1, q2, q3]
    # Explicit assert: 4 retrievals (initial + 3 rewrites)
    assert retriever.retrieve_calls == [q0, q1, q2, q3]
    # Explicit assert: exactly 3 rewrites (never exceeds budget, no 4th call)
    assert len(rewriter.rewrite_calls) == 3


@pytest.mark.unit
def test_final_allowed_rewrite_produces_verify_more():
    """When a final permitted rewrite evaluates to VERIFY_MORE, evidence refiner is called and VERIFY_MORE returned."""
    orig_q = "ambiguous original query"
    rewrite_q = "focused rewritten query"
    doc_id = "doc_verify"

    retriever = CountingRetriever(
        results_by_query={
            orig_q: [make_retrieval_result(doc_id)],
            rewrite_q: [make_retrieval_result(doc_id)],
        }
    )
    # orig_q rejected (0,0); rewrite_q returns intent=1, suff=0 -> VERIFY_MORE
    evaluator = MockRuntimeEvaluator(
        pair_scores={
            (orig_q, doc_id): (0.0, 0.0),
            (rewrite_q, doc_id): (1.0, 0.0),
        }
    )
    base_rewriter = MockQueryRewriter(
        rewrite_map={orig_q: rewrite_q},
        max_attempts=1,
    )
    rewriter = CountingRewriter(base_rewriter)
    refiner = MockEvidenceRefiner()

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=CorrectiveDecisionRouter(),
        refiner=refiner,
        rewriter=rewriter,
        max_rewrite_attempts=1,
    )
    result = orch.run_pipeline(orig_q)

    # Asserts on established VERIFY_MORE contract
    assert result.final_action == CRAGAction.VERIFY_MORE
    assert result.final_query == rewrite_q
    assert result.original_query == orig_q
    assert result.refinement_result is not None
    assert isinstance(result.refinement_result, RefinementResult)
    assert result.metadata["path"] == "verify_more"
    assert result.metadata["refinement_strategy"] == result.refinement_result.refinement_strategy
    assert result.rewrite_history == [rewrite_q]
    # Exactly 2 retrievals: initial query, then rewritten query
    assert retriever.retrieve_calls == [orig_q, rewrite_q]
    # Exactly 1 rewrite call: no extra rewrite calls
    assert len(rewriter.rewrite_calls) == 1


@pytest.mark.unit
def test_orchestrator_is_usable_rewrite_audit_all_cases():
    """Audit all 8 usability and terminal cases against CRAGOrchestrator._is_usable_rewrite."""
    orig = "root query"
    curr = "current step query"
    hist = ["prior rewrite 1", "prior rewrite 2"]

    # 1. Whitespace-only rewritten query
    r1 = RewriteResult(
        original_query=orig,
        rewritten_query="   ",
        strategy="custom",
        attempt_index=1,
        is_terminal=False,
    )
    assert r1.is_usable is False
    assert CRAGOrchestrator._is_usable_rewrite(r1, curr, orig, hist) is False

    # 2. Candidate identical to current_query (case/whitespace insensitive)
    r2 = RewriteResult(
        original_query=orig,
        rewritten_query="  Current Step Query  ",
        strategy="custom",
        attempt_index=1,
        is_terminal=False,
    )
    assert r2.is_usable is True  # Schema-level non-empty
    assert CRAGOrchestrator._is_usable_rewrite(r2, curr, orig, hist) is False  # Orchestrator-level cycle guard

    # 3. Candidate identical to original_query (cyclic root reintroduction)
    r3 = RewriteResult(
        original_query=orig,
        rewritten_query="Root Query",
        strategy="custom",
        attempt_index=1,
        is_terminal=False,
    )
    assert r3.is_usable is True
    assert CRAGOrchestrator._is_usable_rewrite(r3, curr, orig, hist) is False

    # 4. Candidate already present in rewrite history
    r4 = RewriteResult(
        original_query=orig,
        rewritten_query="prior rewrite 1",
        strategy="custom",
        attempt_index=1,
        is_terminal=False,
    )
    assert r4.is_usable is True
    assert CRAGOrchestrator._is_usable_rewrite(r4, curr, orig, hist) is False

    # 5. Cycle-terminal result (terminal_cycle_avoided)
    r5 = RewriteResult(
        original_query=orig,
        rewritten_query=curr,
        strategy="terminal_cycle_avoided",
        attempt_index=2,
        is_terminal=True,
    )
    assert r5.is_usable is False
    assert CRAGOrchestrator._is_usable_rewrite(r5, curr, orig, hist) is False

    # 6. Budget-exhausted terminal result (terminal_budget_exhausted)
    r6 = RewriteResult(
        original_query=orig,
        rewritten_query=curr,
        strategy="terminal_budget_exhausted",
        attempt_index=3,
        is_terminal=True,
    )
    assert r6.is_usable is False
    assert CRAGOrchestrator._is_usable_rewrite(r6, curr, orig, hist) is False

    # 7. Valid, distinct candidate with is_terminal=True (final allowed rewrite)
    r7 = RewriteResult(
        original_query=orig,
        rewritten_query="brand new final query",
        strategy="mock_sequential_step_3",
        attempt_index=3,
        is_terminal=True,
    )
    assert r7.is_usable is True
    assert CRAGOrchestrator._is_usable_rewrite(r7, curr, orig, hist) is True

    # 8. Valid, distinct nonterminal candidate (is_terminal=False)
    r8 = RewriteResult(
        original_query=orig,
        rewritten_query="brand new intermediate query",
        strategy="mock_sequential_step_1",
        attempt_index=1,
        is_terminal=False,
    )
    assert r8.is_usable is True
    assert CRAGOrchestrator._is_usable_rewrite(r8, curr, orig, hist) is True


