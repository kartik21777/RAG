"""
Targeted Verification & Refinement Test Suite (VERIFY_MORE).

Automated tests covering all 12 required scenarios for targeted evidence retrieval,
adjacent-chunk expansion, missing-information query generation, re-evaluation,
and safe termination.
"""

import pytest
from typing import Dict, List, Optional
from src.core.schemas import Document, RetrievalResult
from src.orchestration.schemas import CRAGAction, EvaluatorPrediction, GenerationStatus
from src.orchestration.evaluator import BaseRuntimeEvaluator, MockRuntimeEvaluator
from src.orchestration.router import CorrectiveDecisionRouter
from src.orchestration.refiner import (
    BaseEvidenceRefiner,
    SecondaryRetrievalRefiner,
    MockEvidenceRefiner,
    extract_missing_information_query,
    wrap_document_as_retrieval_result,
)
from src.orchestration.orchestrator import CRAGOrchestrator
from src.orchestration.generator import DeterministicGroundedGenerator
from src.retrieval.facade import CRAGRetrieverFacade


# ---------------------------------------------------------------------------
# Test Helpers & Stubs
# ---------------------------------------------------------------------------

class StubRetrieverFacade(CRAGRetrieverFacade):
    """Offline stub for CRAGRetrieverFacade to inspect query and top_k without real indices."""

    def __init__(self, canned_results: Optional[List[RetrievalResult]] = None, fail_on_query: Optional[str] = None):
        self.canned_results = canned_results or []
        self.retrieve_calls: List[tuple[str, int]] = []
        self.fail_on_query = fail_on_query

    @property
    def is_indexed(self) -> bool:
        return True

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievalResult]:
        self.retrieve_calls.append((query, top_k))
        if self.fail_on_query and query == self.fail_on_query:
            raise RuntimeError(f"Simulated retrieval failure for query: {query}")
        return [r.model_copy(deep=True) for r in self.canned_results[:top_k]]


def make_doc(doc_id: str, text: str, metadata: Optional[Dict] = None) -> Document:
    return Document(doc_id=doc_id, text=text, metadata=metadata or {})


def make_res(doc_id: str, text: str, score: float = 0.8, rank: int = 1, metadata: Optional[Dict] = None) -> RetrievalResult:
    return wrap_document_as_retrieval_result(make_doc(doc_id, text, metadata), rank=rank, score=score)


# ---------------------------------------------------------------------------
# 1. Adjacent chunks exist and are retrieved correctly
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_adjacent_chunks_exist_and_retrieved_correctly():
    query = "What was Company X's revenue in 2025?"
    doc_initial = make_doc(
        "chunk_center",
        "Company X continued growing operations.",
        metadata={"prev_chunk_id": "chunk_prev", "next_chunk_id": "chunk_next"},
    )
    doc_prev = make_doc("chunk_prev", "In early 2025, market share increased.", metadata={})
    doc_next = make_doc("chunk_next", "In 2025, revenue was 5 billion dollars.", metadata={})

    corpus_lookup = {"chunk_prev": doc_prev, "chunk_next": doc_next}
    facade = StubRetrieverFacade()
    evaluator = MockRuntimeEvaluator(
        doc_scores={
            "chunk_center": (0.9, 0.4),
            "chunk_prev": (0.8, 0.3),
            "chunk_next": (0.95, 0.95),  # TRUST
        }
    )
    router = CorrectiveDecisionRouter(intent_threshold=0.7, sufficiency_threshold=0.7)
    refiner = SecondaryRetrievalRefiner(
        retriever_facade=facade,
        corpus_lookup=corpus_lookup,
        evaluator=evaluator,
        router=router,
    )

    result = refiner.refine(query=query, initial_results=[doc_initial], verification_target_ids=["chunk_center"])

    assert result.is_expanded is True
    assert result.refinement_strategy == "adjacent_chunk"
    assert result.added_doc_ids == ["chunk_prev", "chunk_next"]
    assert result.is_sufficient is True
    assert result.re_evaluation_action == CRAGAction.TRUST
    # Ensure secondary hybrid retrieval was skipped because adjacent chunk satisfied the query
    assert len(facade.retrieve_calls) == 0


# ---------------------------------------------------------------------------
# 2. Adjacent links are missing, so targeted secondary retrieval is used
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_adjacent_links_missing_triggers_targeted_secondary_retrieval():
    query = "What was Company X's revenue in 2025?"
    doc_initial = make_doc("doc_initial", "Company X had major events in 2025.", metadata={})

    doc_target = make_res("doc_secondary", "Company X reported 2025 revenue of 5 billion.", score=0.92)
    facade = StubRetrieverFacade(canned_results=[doc_target])

    def mock_query_gen(q: str, docs: List[Document]) -> str:
        return "Company X 2025 exact revenue amount"

    refiner = SecondaryRetrievalRefiner(
        retriever_facade=facade,
        secondary_top_k=2,
        query_generator=mock_query_gen,
    )

    result = refiner.refine(query=query, initial_results=[doc_initial], verification_target_ids=["doc_initial"])

    assert result.is_expanded is True
    assert result.refinement_strategy == "secondary_retrieval"
    assert result.targeted_query == "Company X 2025 exact revenue amount"
    assert len(facade.retrieve_calls) == 1
    passed_q, passed_k = facade.retrieve_calls[0]
    assert passed_q == "Company X 2025 exact revenue amount"
    assert "doc_secondary" in result.added_doc_ids


# ---------------------------------------------------------------------------
# 3. Adjacent chunks exist but still do not supply sufficient evidence
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_adjacent_chunks_exist_but_insufficient_triggers_targeted_retrieval():
    query = "What was Company X's revenue in 2025?"
    doc_initial = make_doc("chunk_1", "Company X grew.", metadata={"adjacent_chunk_ids": ["chunk_2"]})
    doc_adjacent = make_doc("chunk_2", "Markets expanded in Europe.", metadata={})
    doc_targeted = make_res("chunk_3", "In 2025, Company X total revenue was 5 billion.", score=0.95)

    corpus_lookup = {"chunk_2": doc_adjacent}
    facade = StubRetrieverFacade(canned_results=[doc_targeted])

    evaluator = MockRuntimeEvaluator(
        doc_scores={
            "chunk_1": (0.85, 0.3),  # VERIFY_MORE
            "chunk_2": (0.80, 0.3),  # VERIFY_MORE (still insufficient!)
            "chunk_3": (0.95, 0.95), # TRUST from targeted retrieval
        }
    )
    router = CorrectiveDecisionRouter(intent_threshold=0.7, sufficiency_threshold=0.7)

    refiner = SecondaryRetrievalRefiner(
        retriever_facade=facade,
        corpus_lookup=corpus_lookup,
        evaluator=evaluator,
        router=router,
        query_generator=lambda q, d: "Company X 2025 revenue figure",
    )

    result = refiner.refine(query=query, initial_results=[doc_initial], verification_target_ids=["chunk_1"])

    # Both adjacent chunk and targeted secondary retrieval were executed
    assert result.metadata["adjacent_metadata_found"] is True
    assert result.metadata["secondary_retrieval_executed"] is True
    assert "chunk_2" in result.added_doc_ids
    assert "chunk_3" in result.added_doc_ids
    assert result.is_sufficient is True
    assert result.re_evaluation_action == CRAGAction.TRUST


# ---------------------------------------------------------------------------
# 4. Targeted retrieval finds complementary evidence and combined passes criteria
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_targeted_retrieval_finds_complementary_evidence_passing_evaluator():
    query = "What was Company X's revenue in 2025?"
    doc_partial = make_res("doc_p", "Company X reported positive performance.", score=0.7)
    doc_complement = make_res("doc_c", "Company X revenue reached 5 billion in 2025.", score=0.9)

    facade = StubRetrieverFacade(canned_results=[doc_complement])
    evaluator = MockRuntimeEvaluator(
        doc_scores={
            "doc_p": (0.85, 0.3),  # VERIFY_MORE
            "doc_c": (0.95, 0.95), # TRUST
        }
    )
    router = CorrectiveDecisionRouter(intent_threshold=0.7, sufficiency_threshold=0.7)
    refiner = SecondaryRetrievalRefiner(
        retriever_facade=facade,
        evaluator=evaluator,
        router=router,
        query_generator=lambda q, d: "Company X 2025 revenue figure",
    )
    orch = CRAGOrchestrator(
        retriever=facade,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
    )

    # Initial retrieval returns partial doc
    facade.canned_results = [doc_partial]
    # Secondary retrieval will return complementary doc
    refiner.retriever = StubRetrieverFacade(canned_results=[doc_complement])

    result = orch.run_pipeline(query)

    assert result.final_action == CRAGAction.TRUST
    assert result.metadata["path"] == "verify_more"
    assert result.metadata["is_sufficient"] is True

    # Generation produces grounded answer citing complementary doc
    generator = DeterministicGroundedGenerator()
    response = generator.generate_from_result(result)
    assert response.status == GenerationStatus.GROUNDED
    assert len(response.citations) > 0


# ---------------------------------------------------------------------------
# 5. Targeted retrieval returns irrelevant or insufficient evidence (not blindly trusted)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_targeted_retrieval_returns_irrelevant_evidence_not_automatically_trusted():
    query = "What was Company X's revenue in 2025?"
    doc_partial = make_res("doc_p", "Company X reported positive performance.", score=0.7)
    # Off-topic retrieved document from noisy search
    doc_offtopic = make_res("doc_noisy", "Astronomy observations show new galaxies in 2025.", score=0.3)

    facade = StubRetrieverFacade(canned_results=[doc_partial])
    secondary_facade = StubRetrieverFacade(canned_results=[doc_offtopic])

    evaluator = MockRuntimeEvaluator(
        doc_scores={
            "doc_p": (0.85, 0.3),     # VERIFY_MORE
            "doc_noisy": (0.10, 0.0), # REJECT_SEARCH (irrelevant)
        }
    )
    router = CorrectiveDecisionRouter(intent_threshold=0.7, sufficiency_threshold=0.7)
    refiner = SecondaryRetrievalRefiner(
        retriever_facade=secondary_facade,
        evaluator=evaluator,
        router=router,
        query_generator=lambda q, d: "Company X revenue",
    )

    result = refiner.refine(query=query, initial_results=[doc_partial], verification_target_ids=["doc_p"])

    # doc_noisy was rejected by evaluator, so it was NOT added to refined evidence
    assert "doc_noisy" not in result.added_doc_ids
    assert result.is_sufficient is False
    assert result.re_evaluation_action == CRAGAction.VERIFY_MORE


# ---------------------------------------------------------------------------
# 6. Multiple chunks jointly answer a query although individual chunks are insufficient
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_multiple_chunks_jointly_answer_query_individual_insufficient():
    query = "What was Company X's revenue in 2025?"
    doc_a = make_doc("chunk_a", "Company X released its annual financial statement for 2025.")
    doc_b = make_doc("chunk_b", "The annual statement showed total revenue of 5 billion dollars.")

    facade = StubRetrieverFacade(canned_results=[wrap_document_as_retrieval_result(doc_b)])

    # Both individually have sufficiency 0.4 (VERIFY_MORE)
    # But composite collection "chunk_a+chunk_b" has sufficiency 0.9 (TRUST)
    evaluator = MockRuntimeEvaluator(
        doc_scores={
            "chunk_a": (0.85, 0.4),
            "chunk_b": (0.85, 0.4),
            "chunk_a+chunk_b": (0.95, 0.90),
        }
    )
    router = CorrectiveDecisionRouter(intent_threshold=0.7, sufficiency_threshold=0.7)
    refiner = SecondaryRetrievalRefiner(
        retriever_facade=facade,
        evaluator=evaluator,
        router=router,
    )

    result = refiner.refine(query=query, initial_results=[doc_a], verification_target_ids=["chunk_a"])

    assert result.is_sufficient is True
    assert result.re_evaluation_action == CRAGAction.TRUST
    assert len(result.refined_results) == 2


# ---------------------------------------------------------------------------
# 7. Duplicate results are removed while provenance metadata is preserved
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_duplicate_results_removed_preserving_provenance_metadata():
    query = "Explain sparse retrieval"
    doc_1 = make_res("doc_1", "Sparse BM25 ranking", score=0.88, rank=1)
    doc_2 = make_res("doc_2", "Inverted index details", score=0.75, rank=2)

    # Secondary retrieval returns a duplicate of doc_1 with different score/rank, and a new doc_3
    sec_dup = make_res("doc_1", "Different text duplicate", score=0.99, rank=1)
    sec_new = make_res("doc_3", "Lexical matching features", score=0.65, rank=2)

    facade = StubRetrieverFacade(canned_results=[sec_dup, sec_new])
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade)

    result = refiner.refine(query=query, initial_results=[doc_1, doc_2], verification_target_ids=["doc_1"])

    assert result.added_doc_ids == ["doc_3"]
    assert len(result.refined_results) == 3
    # Initial occurrence of doc_1 is strictly preserved (score 0.88, rank 1)
    assert result.refined_results[0].fused_score == 0.88
    assert result.refined_results[0].fused_rank == 1
    assert result.refined_results[0].document.text == "Sparse BM25 ranking"


# ---------------------------------------------------------------------------
# 8. Missing-information query generation fails or returns unusable query -> bounded fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_missing_info_query_generator_failure_triggers_bounded_fallback():
    query = "What is dense retrieval?"
    docs = [make_doc("d1", "Dense representations.")]

    def failing_generator(q: str, d: List[Document]) -> str:
        raise RuntimeError("LLM service unavailable")

    targeted_query = extract_missing_information_query(query, docs, query_generator=failing_generator)
    # Safely falls back to the original clean query
    assert targeted_query == query

    def empty_generator(q: str, d: List[Document]) -> Optional[str]:
        return "   "

    fallback_query = extract_missing_information_query(query, docs, query_generator=empty_generator)
    assert fallback_query == query


# ---------------------------------------------------------------------------
# 9. No correction attempt finds sufficient evidence, and pipeline terminates safely
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_correction_finds_sufficient_evidence_terminates_safely():
    query = "What was the exact 2025 net profit of Company Y?"
    doc_initial = make_res("doc_y", "Company Y was active in 2025.", score=0.7)

    # Secondary search returns nothing new or empty
    facade = StubRetrieverFacade(canned_results=[doc_initial])
    evaluator = MockRuntimeEvaluator(default_scores=(0.85, 0.3))  # Always VERIFY_MORE
    router = CorrectiveDecisionRouter(intent_threshold=0.7, sufficiency_threshold=0.7)
    refiner = SecondaryRetrievalRefiner(
        retriever_facade=facade,
        evaluator=evaluator,
        router=router,
    )
    orch = CRAGOrchestrator(
        retriever=facade,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
    )

    result = orch.run_pipeline(query)

    assert result.final_action == CRAGAction.VERIFY_MORE
    assert result.metadata["is_sufficient"] is False
    assert result.metadata["loop_iterations"] == 1  # No infinite loop!

    # Generator cleanly abstains
    generator = DeterministicGroundedGenerator()
    response = generator.generate_from_result(result)
    assert response.status == GenerationStatus.INSUFFICIENT_EVIDENCE
    assert "cannot provide a factual answer" in response.answer


# ---------------------------------------------------------------------------
# 10. Existing REJECT_SEARCH query-rewriting behavior remains unchanged
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reject_search_query_rewriting_unchanged():
    from src.orchestration.rewriter import BaseQueryRewriter
    from src.orchestration.schemas import RewriteResult

    class StubRewriter(BaseQueryRewriter):
        def rewrite(self, query: str, history: Optional[List[str]] = None, rejected_candidates: Optional[List[RetrievalResult]] = None, original_query: Optional[str] = None, **kwargs) -> RewriteResult:
            attempt_num = (len(history) + 1) if history is not None else 1
            return RewriteResult(
                original_query=query,
                rewritten_query="rewritten query",
                strategy="clarification",
                attempt_index=attempt_num,
                is_terminal=True,
            )

    query = "completely unrelated question"
    doc_bad = make_res("bad_doc", "Off topic document", score=0.1)
    facade = StubRetrieverFacade(canned_results=[doc_bad])
    evaluator = MockRuntimeEvaluator(default_scores=(0.2, 0.1))  # REJECT_SEARCH
    router = CorrectiveDecisionRouter(intent_threshold=0.7, sufficiency_threshold=0.7)
    refiner = MockEvidenceRefiner()
    rewriter = StubRewriter()

    orch = CRAGOrchestrator(
        retriever=facade,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
        rewriter=rewriter,
        max_rewrite_attempts=1,
    )

    result = orch.run_pipeline(query)

    # REJECT_SEARCH path was taken and rewritten
    assert len(result.rewrite_history) == 1
    assert result.rewrite_history[0] == "rewritten query"
    assert result.refinement_result is None  # Refiner was NOT called for REJECT_SEARCH


# ---------------------------------------------------------------------------
# 11. Ordinary TRUST and non-corrective generation flows continue to work
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ordinary_trust_and_non_corrective_generation_work():
    query = "What is BM25?"
    doc_trust = make_res("doc_bm25", "BM25 is a probabilistic lexical ranking function.", score=0.95)
    facade = StubRetrieverFacade(canned_results=[doc_trust])
    evaluator = MockRuntimeEvaluator(default_scores=(0.95, 0.95))  # TRUST
    router = CorrectiveDecisionRouter(intent_threshold=0.7, sufficiency_threshold=0.7)
    refiner = MockEvidenceRefiner()

    orch = CRAGOrchestrator(
        retriever=facade,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
    )

    result = orch.run_pipeline(query)

    assert result.final_action == CRAGAction.TRUST
    assert result.refinement_result is None
    assert result.metadata["path"] == "trust"

    generator = DeterministicGroundedGenerator()
    response = generator.generate_from_result(result)
    assert response.status == GenerationStatus.GROUNDED
    assert "BM25 is a probabilistic lexical ranking function." in response.answer


# ---------------------------------------------------------------------------
# 12. Retriever or LLM failures handled without infinite retry loop
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_retriever_failure_handled_without_infinite_retry_loop():
    query = "What was the revenue?"
    doc_initial = make_res("doc_init", "Partial evidence.", score=0.6)

    # Facade configured to fail when targeted retrieval query is called
    facade = StubRetrieverFacade(
        canned_results=[doc_initial],
        fail_on_query="failing_target_query",
    )
    evaluator = MockRuntimeEvaluator(default_scores=(0.85, 0.3))  # VERIFY_MORE
    router = CorrectiveDecisionRouter(intent_threshold=0.7, sufficiency_threshold=0.7)
    refiner = SecondaryRetrievalRefiner(
        retriever_facade=facade,
        evaluator=evaluator,
        router=router,
        query_generator=lambda q, d: "failing_target_query",
    )

    orch = CRAGOrchestrator(
        retriever=facade,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
    )

    # Facade raises RuntimeError on secondary retrieval; verify it doesn't loop infinitely
    with pytest.raises(RuntimeError) as exc_info:
        orch.run_pipeline(query)
    assert "Simulated retrieval failure" in str(exc_info.value)
