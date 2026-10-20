"""
CRAG Pipeline Orchestrator -- Milestone 5.

Assembles and integrates the four standalone Option C components into a single,
end-to-end corrective feedback loop:

    [User Query]
        |
        v
    [CRAGRetrieverFacade]  -->  Top-K candidates
        |
        v
    [BaseRuntimeEvaluator]  -->  EvaluatorPredictions
        |
        v
    [CorrectiveDecisionRouter]  -->  QueryDecision
        |
        |-- TRUST         -->  return trusted evidence (OrchestratorResult)
        |
        |-- VERIFY_MORE   -->  [BaseEvidenceRefiner]  -->  RefinementResult
        |                          |
        |                    return refined evidence (OrchestratorResult)
        |
        `-- REJECT_SEARCH -->  [BaseQueryRewriter]  -->  RewriteResult
                                       |
                                  loop back to retrieval with rewritten query
                                  (bounded by max_rewrite_attempts)
                                       |
                                  if terminal: return REJECT_SEARCH result

Architectural Rules
-------------------
1. Minimal coupling: orchestrator calls components only through their established
   public contracts (evaluate_pair, route_candidate, aggregate_query_decisions,
   refine, rewrite). Internal state of components is never accessed directly.
2. Termination guarantee: rewrite loops are bounded by max_rewrite_attempts.
   When the budget is exhausted the orchestrator returns the last retrieved
   evidence (or an empty list) with final_action=REJECT_SEARCH.
3. Offline/CPU-safe: no GPU, no network, no external LLM calls in this module.
4. No grounded generation: that belongs to Milestone 6.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.schemas import CRAGAction, RetrievalResult
from src.orchestration.evaluator import BaseRuntimeEvaluator
from src.orchestration.refiner import BaseEvidenceRefiner
from src.orchestration.rewriter import BaseQueryRewriter
from src.orchestration.router import CorrectiveDecisionRouter
from src.orchestration.schemas import (
    CandidateDecision,
    QueryDecision,
    RefinementResult,
    RewriteResult,
)
from src.retrieval.facade import CRAGRetrieverFacade


# ---------------------------------------------------------------------------
# Output Schema
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResult:
    """
    Final output of a single CRAGOrchestrator.run_pipeline() call.

    Attributes:
        original_query:     The query string supplied by the caller.
        final_query:        The query that produced the final retrieval
                            (may differ from original_query if rewriting occurred).
        final_action:       Resolved corrective action for the response.
        final_results:      Ordered evidence documents recommended for generation.
        query_decision:     QueryDecision produced at the final routing step.
        rewrite_history:    Ordered list of rewritten queries (empty if no rewrites).
        refinement_result:  RefinementResult if VERIFY_MORE path was taken, else None.
        rewrite_results:    Ordered list of RewriteResult objects (one per rewrite attempt).
        metadata:           Audit and diagnostic information.
    """
    original_query: str
    final_query: str
    final_action: CRAGAction
    final_results: List[RetrievalResult]
    query_decision: QueryDecision
    rewrite_history: List[str] = field(default_factory=list)
    refinement_result: Optional[RefinementResult] = None
    rewrite_results: List[RewriteResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class CRAGOrchestrator:
    """
    End-to-end CRAG pipeline orchestrator.

    Orchestrates the corrective feedback loop by composing:
    - CRAGRetrieverFacade    (retrieval)
    - BaseRuntimeEvaluator   (evaluation)
    - CorrectiveDecisionRouter (routing)
    - BaseEvidenceRefiner    (VERIFY_MORE path)
    - BaseQueryRewriter      (REJECT_SEARCH path, bounded loop)

    The rewriter is optional; if omitted the orchestrator performs a single
    retrieval + evaluation + routing pass and returns the result directly without
    entering the rewrite loop.
    """

    def __init__(
        self,
        retriever: CRAGRetrieverFacade,
        evaluator: BaseRuntimeEvaluator,
        router: CorrectiveDecisionRouter,
        refiner: BaseEvidenceRefiner,
        rewriter: Optional[BaseQueryRewriter] = None,
        top_k: int = 5,
        max_rewrite_attempts: int = 3,
    ) -> None:
        """
        Args:
            retriever:            Indexed CRAGRetrieverFacade instance.
            evaluator:            Runtime evaluator implementing BaseRuntimeEvaluator.
            router:               CorrectiveDecisionRouter with configured thresholds.
            refiner:              Evidence refiner implementing BaseEvidenceRefiner.
            rewriter:             Optional query rewriter implementing BaseQueryRewriter.
                                  If None the REJECT_SEARCH branch terminates immediately.
            top_k:                Number of candidates to retrieve per query (>= 1).
            max_rewrite_attempts: Upper bound on rewrite loop iterations (>= 1).
        """
        if not isinstance(retriever, CRAGRetrieverFacade):
            raise TypeError(
                f"retriever must be CRAGRetrieverFacade, got {type(retriever).__name__}"
            )
        if not isinstance(evaluator, BaseRuntimeEvaluator):
            raise TypeError(
                f"evaluator must be BaseRuntimeEvaluator, got {type(evaluator).__name__}"
            )
        if not isinstance(router, CorrectiveDecisionRouter):
            raise TypeError(
                f"router must be CorrectiveDecisionRouter, got {type(router).__name__}"
            )
        if not isinstance(refiner, BaseEvidenceRefiner):
            raise TypeError(
                f"refiner must be BaseEvidenceRefiner, got {type(refiner).__name__}"
            )
        if rewriter is not None and not isinstance(rewriter, BaseQueryRewriter):
            raise TypeError(
                f"rewriter must be BaseQueryRewriter or None, got {type(rewriter).__name__}"
            )
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        if not isinstance(max_rewrite_attempts, int) or max_rewrite_attempts < 1:
            raise ValueError(
                f"max_rewrite_attempts must be a positive integer, got {max_rewrite_attempts!r}"
            )

        self.retriever = retriever
        self.evaluator = evaluator
        self.router = router
        self.refiner = refiner
        self.rewriter = rewriter
        self.top_k = top_k
        self.max_rewrite_attempts = max_rewrite_attempts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str) -> List[RetrievalResult]:
        """Delegates to the retrieval facade; returns an empty list on no results."""
        return self.retriever.retrieve(query, top_k=self.top_k)

    def _evaluate_and_route(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> QueryDecision:
        """Evaluates every retrieved candidate and aggregates into a QueryDecision."""
        if not results:
            return self.router.aggregate_query_decisions(query, [])

        candidate_decisions: List[CandidateDecision] = []
        for result in results:
            prediction = self.evaluator.evaluate_pair(query, result.document)
            decision = self.router.route_candidate(prediction)
            candidate_decisions.append(decision)

        return self.router.aggregate_query_decisions(query, candidate_decisions)

    @staticmethod
    def _results_for_trust(
        query_decision: QueryDecision,
        all_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Returns only the trusted evidence subset in retrieval order."""
        trusted_ids = set(query_decision.trusted_doc_ids)
        return [r for r in all_results if r.document.doc_id in trusted_ids]

    @staticmethod
    def _is_usable_rewrite(
        rewrite_result: RewriteResult,
        current_query: str,
        original_query: str,
        history: List[str],
    ) -> bool:
        """
        Determines whether a RewriteResult contains a usable new query candidate.

        A rewrite is non-usable if:
        1. It is explicitly marked terminal due to cycle avoidance or exhausted budget
           (strategy starting with 'terminal_' or is_usable is False).
        2. The candidate query is empty or whitespace-only.
        3. The candidate query normalizes to the current query (no-op rewrite).
        4. The candidate query normalizes to the original query (cyclic reintroduction).
        5. The candidate query has already been evaluated in rewrite history.

        A rewrite that has is_terminal=True BUT contains a valid, distinct candidate query
        (e.g. the final attempt within max_attempts) is usable for exactly one retrieval
        and evaluation pass.
        """
        if not rewrite_result.is_usable:
            return False
        candidate = rewrite_result.rewritten_query
        if not isinstance(candidate, str) or not candidate.strip():
            return False
        norm_candidate = BaseQueryRewriter.normalize_query(candidate)
        if norm_candidate == BaseQueryRewriter.normalize_query(current_query):
            return False
        if norm_candidate == BaseQueryRewriter.normalize_query(original_query):
            return False
        if BaseQueryRewriter.is_in_history(candidate, history):
            return False
        return True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_pipeline(self, query: str) -> OrchestratorResult:
        """
        Executes the full CRAG corrective pipeline for a single user query.

        Pipeline flow:
        1. Retrieve top-K candidates.
        2. Evaluate each candidate and aggregate into a QueryDecision.
        3a. TRUST        -> return evidence immediately.
        3b. VERIFY_MORE  -> call refiner; return refined evidence.
        3c. REJECT_SEARCH -> rewrite query (if rewriter configured) and loop.
                            After max_rewrite_attempts, return REJECT_SEARCH terminal.

        Args:
            query: Non-empty user query string.

        Returns:
            OrchestratorResult capturing final action, evidence, and full audit trail.

        Raises:
            ValueError: If query is empty or whitespace only.
            RuntimeError: If retriever is not indexed.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty, non-whitespace string.")

        original_query = query.strip()
        current_query = original_query
        rewrite_history: List[str] = []
        rewrite_results: List[RewriteResult] = []

        loop_count = 0
        while True:
            loop_count += 1

            # Step 1: Retrieve
            results = self._retrieve(current_query)

            # Step 2: Evaluate + Route
            query_decision = self._evaluate_and_route(current_query, results)

            # Step 3a: TRUST
            if query_decision.overall_action == CRAGAction.TRUST:
                final_results = self._results_for_trust(query_decision, results)
                return OrchestratorResult(
                    original_query=original_query,
                    final_query=current_query,
                    final_action=CRAGAction.TRUST,
                    final_results=final_results,
                    query_decision=query_decision,
                    rewrite_history=rewrite_history,
                    rewrite_results=rewrite_results,
                    metadata={
                        "loop_iterations": loop_count,
                        "path": "trust",
                        "trusted_doc_ids": query_decision.trusted_doc_ids,
                    },
                )

            # Step 3b: VERIFY_MORE
            if query_decision.overall_action == CRAGAction.VERIFY_MORE:
                verify_ids = query_decision.verify_doc_ids
                verify_results = [
                    r for r in results if r.document.doc_id in set(verify_ids)
                ]
                trusted_results = [
                    r for r in results
                    if r.document.doc_id in set(query_decision.trusted_doc_ids)
                ]

                refinement = self.refiner.refine(
                    query=current_query,
                    initial_results=verify_results,
                    verification_target_ids=verify_ids,
                    existing_results=trusted_results if trusted_results else None,
                )

                return OrchestratorResult(
                    original_query=original_query,
                    final_query=current_query,
                    final_action=CRAGAction.VERIFY_MORE,
                    final_results=refinement.refined_results,
                    query_decision=query_decision,
                    rewrite_history=rewrite_history,
                    refinement_result=refinement,
                    rewrite_results=rewrite_results,
                    metadata={
                        "loop_iterations": loop_count,
                        "path": "verify_more",
                        "refinement_strategy": refinement.refinement_strategy,
                        "is_expanded": refinement.is_expanded,
                        "verify_doc_ids": verify_ids,
                    },
                )

            # Step 3c: REJECT_SEARCH
            # Terminate immediately if no rewriter is configured, rewrite budget is exhausted,
            # or previous rewrite was terminal (final allowed rewrite was already evaluated).
            if (
                self.rewriter is None
                or len(rewrite_history) >= self.max_rewrite_attempts
                or (rewrite_results and rewrite_results[-1].is_terminal)
            ):
                return OrchestratorResult(
                    original_query=original_query,
                    final_query=current_query,
                    final_action=CRAGAction.REJECT_SEARCH,
                    final_results=results,
                    query_decision=query_decision,
                    rewrite_history=rewrite_history,
                    rewrite_results=rewrite_results,
                    metadata={
                        "loop_iterations": loop_count,
                        "path": "reject_search_terminal",
                        "reason": (
                            "no_rewriter_configured"
                            if self.rewriter is None
                            else "rewrite_budget_exhausted"
                        ),
                        "max_rewrite_attempts": self.max_rewrite_attempts,
                        "rewrite_history": rewrite_history,
                    },
                )

            # Attempt rewrite
            rewrite_result = self.rewriter.rewrite(
                query=current_query,
                history=rewrite_history,
                rejected_candidates=list(results),
                original_query=original_query,
            )
            rewrite_results.append(rewrite_result)

            if not self._is_usable_rewrite(
                rewrite_result, current_query, original_query, rewrite_history
            ):
                # Cycle detected or internal budget exhausted in the rewriter without usable candidate
                return OrchestratorResult(
                    original_query=original_query,
                    final_query=current_query,
                    final_action=CRAGAction.REJECT_SEARCH,
                    final_results=results,
                    query_decision=query_decision,
                    rewrite_history=rewrite_history,
                    rewrite_results=rewrite_results,
                    metadata={
                        "loop_iterations": loop_count,
                        "path": "reject_search_rewriter_terminal",
                        "rewriter_strategy": rewrite_result.strategy,
                        "rewrite_history": rewrite_history,
                    },
                )

            # Advance to the next rewrite iteration (usable candidate query)
            rewrite_history.append(rewrite_result.rewritten_query)
            current_query = rewrite_result.rewritten_query

