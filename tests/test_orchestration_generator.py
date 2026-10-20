"""
Tests for Grounded Response Generator (Option C Milestone 6).

Verifies evidence-grounded answering, safe refusal on REJECT_SEARCH,
insufficient evidence handling, citation traceability, metadata tolerance,
contradictory evidence handling, and integration with OrchestratorResult.
"""

import pytest
from typing import Dict, List, Optional, Tuple

from src.core.schemas import CRAGAction, Document, RetrievalResult
from src.orchestration.evaluator import MockRuntimeEvaluator
from src.orchestration.generator import (
    BaseGroundedGenerator,
    DeterministicGroundedGenerator,
)
from src.orchestration.orchestrator import CRAGOrchestrator, OrchestratorResult
from src.orchestration.refiner import MockEvidenceRefiner
from src.orchestration.rewriter import MockQueryRewriter
from src.orchestration.router import CorrectiveDecisionRouter
from src.orchestration.schemas import (
    EvidenceCitation,
    GenerationStatus,
    GroundedResponse,
    QueryDecision,
    RefinementResult,
)
from src.retrieval.facade import CRAGRetrieverFacade


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def make_doc(doc_id: str, text: str, metadata: Optional[Dict] = None) -> Document:
    return Document(doc_id=doc_id, text=text, metadata=metadata or {})


def make_retrieval_result(
    doc_id: str,
    text: str = "sample factual evidence text",
    score: float = 1.0,
    rank: int = 1,
    metadata: Optional[Dict] = None,
) -> RetrievalResult:
    return RetrievalResult(
        document=make_doc(doc_id, text, metadata),
        fused_score=score,
        fused_rank=rank,
        sparse_rank=None,
        dense_rank=None,
        sparse_score=None,
        dense_score=None,
    )


class StubRetriever(CRAGRetrieverFacade):
    def __init__(self, results_by_query: Dict[str, List[RetrievalResult]]):
        self._results_by_query = results_by_query

    @property
    def is_indexed(self) -> bool:
        return True

    def retrieve(self, query, top_k: int = 5) -> List[RetrievalResult]:
        query_str = query if isinstance(query, str) else query.query_text
        return list(self._results_by_query.get(query_str, []))[:top_k]


# ---------------------------------------------------------------------------
# 1. Supported Question Produces Grounded Answer & Citations
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_grounded_answer_from_trusted_evidence():
    """A supported question produces a grounded answer with matching citations."""
    query = "What is Reciprocal Rank Fusion?"
    doc_text = (
        "Reciprocal Rank Fusion is a hybrid reranking technique. "
        "It combines rankings from sparse BM25 and dense embedding retrievers without score calibration."
    )
    result = make_retrieval_result("doc_rrf", doc_text, score=0.95, rank=1, metadata={"source": "rag_docs"})

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=[result],
        action=CRAGAction.TRUST,
    )

    assert response.status == GenerationStatus.GROUNDED
    assert response.action_used == CRAGAction.TRUST
    assert "Reciprocal Rank Fusion" in response.answer
    assert "[doc_rrf]" in response.answer
    assert len(response.citations) >= 1
    assert response.citations[0].doc_id == "doc_rrf"
    assert response.citations[0].source == "rag_docs"
    assert response.citations[0].rank == 1
    assert response.citations[0].score == 0.95


# ---------------------------------------------------------------------------
# 2. Citations Resolve Strictly to Actual Supplied Evidence
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_citations_strictly_resolve_to_supplied_evidence():
    """All emitted citations point to documents physically present in the evidence list with exact snippet matching."""
    query = "Explain BM25 algorithm"
    doc1 = make_retrieval_result("doc_bm25", "BM25 is a term-matching sparse retrieval algorithm based on TF-IDF.")
    doc2 = make_retrieval_result("doc_other", "Neural networks use backpropagation for gradient descent.")

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=[doc1, doc2],
        action=CRAGAction.TRUST,
    )

    assert response.status == GenerationStatus.GROUNDED
    supplied_doc_ids = {doc1.document.doc_id, doc2.document.doc_id}

    for citation in response.citations:
        assert citation.doc_id in supplied_doc_ids
        # Find source document
        matching_doc = next(d for d in [doc1, doc2] if d.document.doc_id == citation.doc_id)
        # Snippet must be a verbatim substring of the source document text
        assert citation.snippet in matching_doc.document.text


# ---------------------------------------------------------------------------
# 3. Unsupported Information Is Not Invented (Insufficient Evidence)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_insufficient_evidence_when_evidence_lacks_relevance():
    """When accepted evidence does not match the question intent, generator declines instead of hallucinating."""
    query = "How do quantum computers perform error correction?"
    # Off-topic evidence
    doc = make_retrieval_result("doc_cooking", "French cooking emphasizes butter, cream, and precise reductions.")

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=[doc],
        action=CRAGAction.TRUST,
    )

    assert response.status == GenerationStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == []
    assert "cannot provide" in response.answer or "insufficient" in response.answer.lower()
    assert response.refusal_reason is not None


# ---------------------------------------------------------------------------
# 4. Empty Evidence Set Handling
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_empty_evidence_produces_insufficient_evidence_status():
    """Empty evidence set returns explicit INSUFFICIENT_EVIDENCE status with zero citations."""
    query = "What is FAISS IndexFlatIP?"

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=[],
        action=CRAGAction.TRUST,
    )

    assert response.status == GenerationStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == []
    assert response.refusal_reason == "Evidence list is empty."


# ---------------------------------------------------------------------------
# 5. REJECT_SEARCH Safely Declines Factual Answering
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reject_search_safely_declines_generation():
    """When action is REJECT_SEARCH, factual generation from rejected evidence is strictly refused."""
    query = "What is the capital of Mars?"
    rejected_doc = make_retrieval_result("doc_mars", "Mars has no capital city or human government.")

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=[rejected_doc],
        action=CRAGAction.REJECT_SEARCH,
    )

    assert response.status == GenerationStatus.REJECTED_SEARCH
    assert response.action_used == CRAGAction.REJECT_SEARCH
    assert response.citations == []
    assert "judged off-topic or insufficient" in response.answer
    assert response.refusal_reason == "CRAG policy evaluated retrieval as REJECT_SEARCH."


# ---------------------------------------------------------------------------
# 6. VERIFY_MORE Uses Refinement Outcome
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_verify_more_uses_refinement_outcome():
    """VERIFY_MORE path incorporates refined evidence and documents refinement in metadata."""
    query = "What is dense retrieval?"
    refined_doc = make_retrieval_result(
        "doc_dense_refined",
        "Dense retrieval maps queries and passages into a shared continuous embedding space.",
    )
    refinement_res = RefinementResult(
        query_text=query,
        original_doc_ids=["doc_initial"],
        refined_results=[refined_doc],
        refined_documents=[refined_doc.document],
        refinement_strategy="secondary_retrieval",
        is_expanded=True,
    )

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=[refined_doc],
        action=CRAGAction.VERIFY_MORE,
        refinement_result=refinement_res,
    )

    assert response.status == GenerationStatus.GROUNDED
    assert response.action_used == CRAGAction.VERIFY_MORE
    assert response.metadata.get("refinement_applied") is True
    assert response.metadata.get("refinement_strategy") == "secondary_retrieval"
    assert len(response.citations) == 1
    assert response.citations[0].doc_id == "doc_dense_refined"


# ---------------------------------------------------------------------------
# 7. Contradictory Evidence Handled Explicitly
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_contradictory_evidence_handled_explicitly():
    """Explicit contradictory override flag produces refusal explaining intent-dominance override."""
    query = "Does transformer model use recurrence?"
    doc = make_retrieval_result("doc_attn", "Transformers rely entirely on self-attention mechanisms.")

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=[doc],
        action=CRAGAction.REJECT_SEARCH,
        metadata={"is_contradictory_override": True},
    )

    assert response.status == GenerationStatus.INSUFFICIENT_EVIDENCE
    assert "intent-dominance override" in response.answer.lower()
    assert response.citations == []
    assert "Intent-dominance override" in str(response.refusal_reason)


@pytest.mark.unit
def test_explicit_contradiction_metadata_handled():
    """Explicit contradictory=True metadata produces refusal explaining contradictory evidence."""
    query = "Does transformer model use recurrence?"
    doc = make_retrieval_result("doc_attn", "Transformers rely entirely on self-attention mechanisms.")

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=[doc],
        action=CRAGAction.REJECT_SEARCH,
        metadata={"contradictory": True},
    )

    assert response.status == GenerationStatus.INSUFFICIENT_EVIDENCE
    assert "contradictory" in response.answer.lower()
    assert response.citations == []
    assert "Contradictory" in str(response.refusal_reason)


# ---------------------------------------------------------------------------
# 8. Missing Optional Metadata Does Not Crash
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_missing_optional_metadata_tolerance():
    """Empty metadata in Document and RetrievalResult does not cause KeyError or crashes."""
    query = "Define cosine similarity"
    doc = make_retrieval_result("doc_cos", "Cosine similarity measures the angle between two vectors.", metadata={})

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=[doc],
        action=CRAGAction.TRUST,
        metadata=None,
    )

    assert response.status == GenerationStatus.GROUNDED
    assert len(response.citations) == 1
    assert response.citations[0].source is None


# ---------------------------------------------------------------------------
# 9. Fabricated Evidence Identifiers Cannot Appear
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_fabricated_evidence_identifiers():
    """All citations strictly originate from supplied doc_id values."""
    query = "What is sentence-transformers?"
    doc = make_retrieval_result("real_doc_42", "sentence-transformers provides state-of-the-art sentence embeddings.")

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=[doc],
        action=CRAGAction.TRUST,
    )

    assert len(response.citations) == 1
    assert response.citations[0].doc_id == "real_doc_42"
    # Never invents doc IDs like "doc_001", "wiki", or generic placeholders
    assert "unknown" not in response.citations[0].doc_id


# ---------------------------------------------------------------------------
# 10. Adapter Integration with OrchestratorResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_adapter_generate_from_orchestrator_result():
    """generate_from_result accurately consumes a real OrchestratorResult."""
    query = "What is BM25?"
    doc_id = "doc_bm25_01"
    res = make_retrieval_result(doc_id, "BM25 is a probabilistic relevance framework for information retrieval.")

    retriever = StubRetriever({query: [res]})
    evaluator = MockRuntimeEvaluator(pair_scores={(query, doc_id): (1.0, 1.0)})
    router = CorrectiveDecisionRouter()
    refiner = MockEvidenceRefiner()

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
    )
    orch_result = orch.run_pipeline(query)

    generator = DeterministicGroundedGenerator()
    grounded_res = generator.generate_from_result(orch_result)

    assert isinstance(grounded_res, GroundedResponse)
    assert grounded_res.status == GenerationStatus.GROUNDED
    assert grounded_res.query == query
    assert grounded_res.final_query == query
    assert grounded_res.action_used == CRAGAction.TRUST
    assert len(grounded_res.citations) == 1
    assert grounded_res.citations[0].doc_id == doc_id


# ---------------------------------------------------------------------------
# 11. Original Question Preserved Even When Retrieval Query Was Rewritten
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_original_question_preserved_after_rewriting():
    """When query rewriter reformulates the query, the original user question is retained in response."""
    orig_q = "what is rrf"
    rewritten_q = "reciprocal rank fusion algorithm"
    doc_id = "doc_rrf_def"
    res = make_retrieval_result(doc_id, "Reciprocal Rank Fusion combines ranked lists from multiple search algorithms.")

    retriever = StubRetriever({
        orig_q: [res],
        rewritten_q: [res],
    })
    # Initial rejected -> rewritten trusted
    evaluator = MockRuntimeEvaluator(
        pair_scores={
            (orig_q, doc_id): (0.0, 0.0),
            (rewritten_q, doc_id): (1.0, 1.0),
        }
    )
    router = CorrectiveDecisionRouter()
    refiner = MockEvidenceRefiner()
    rewriter = MockQueryRewriter(rewrite_map={orig_q: rewritten_q}, max_attempts=1)

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
        rewriter=rewriter,
        max_rewrite_attempts=1,
    )
    orch_result = orch.run_pipeline(orig_q)

    generator = DeterministicGroundedGenerator()
    grounded_res = generator.generate_from_result(orch_result)

    assert grounded_res.status == GenerationStatus.GROUNDED
    assert grounded_res.query == orig_q
    assert grounded_res.final_query == rewritten_q
    assert "Reciprocal Rank Fusion" in grounded_res.answer


# ---------------------------------------------------------------------------
# 12. Input Validation Bounds & Types
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_generator_input_validation():
    """Generator validates query strings and parameter bounds correctly."""
    generator = DeterministicGroundedGenerator()

    with pytest.raises(ValueError):
        generator.generate(query="", evidence=[], action=CRAGAction.TRUST)

    with pytest.raises(ValueError):
        generator.generate(query="   ", evidence=[], action=CRAGAction.TRUST)

    with pytest.raises(ValueError):
        generator.generate(query=None, evidence=[], action=CRAGAction.TRUST)

    with pytest.raises(TypeError):
        generator.generate_from_result("not an OrchestratorResult")

    with pytest.raises(ValueError):
        DeterministicGroundedGenerator(max_citations=0)

    with pytest.raises(ValueError):
        DeterministicGroundedGenerator(min_overlap_terms=0)


# ---------------------------------------------------------------------------
# 13. Audit QA Relevance: Directly Answerable Question
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_audit_qa_relevance_directly_answerable():
    """Evidence contains a sentence that directly answers the question."""
    query = "What is the primary function of BM25?"
    evidence = [
        make_retrieval_result(
            "doc_bm25",
            "The primary function of BM25 is ranking documents based on term frequency and inverted document frequency."
        )
    ]
    generator = DeterministicGroundedGenerator()
    response = generator.generate(query=query, evidence=evidence, action=CRAGAction.TRUST)

    assert response.status == GenerationStatus.GROUNDED
    assert len(response.citations) == 1
    assert response.citations[0].doc_id == "doc_bm25"
    assert "primary function of BM25" in response.answer


# ---------------------------------------------------------------------------
# 14. Audit QA Relevance: Keyword Overlap Without an Answer
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_audit_qa_relevance_keyword_overlap_without_answer():
    """Evidence mentions a keyword but does not answer the question; must refuse."""
    query = "How does BM25 handle sentence embeddings?"
    # Mentions BM25, but does NOT contain sentence embeddings or answer how BM25 handles embeddings
    evidence = [
        make_retrieval_result(
            "doc_bm25_sparse",
            "BM25 is a classical sparse bag-of-words retrieval model."
        )
    ]
    generator = DeterministicGroundedGenerator()
    response = generator.generate(query=query, evidence=evidence, action=CRAGAction.TRUST)

    assert response.status == GenerationStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == []
    assert "insufficient" in response.answer.lower() or "cannot provide" in response.answer.lower()


# ---------------------------------------------------------------------------
# 15. Audit QA Relevance: Question Asks 'Why' Without Causal Explanation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_audit_qa_relevance_why_without_causal_explanation():
    """Evidence mentions an event but contains no causal explanation; generator must not mark GROUNDED."""
    query = "Why did the transformer model fail during inference?"
    # Mentions transformer and inference, but provides no causal explanation (e.g. because, due to, reason)
    evidence = [
        make_retrieval_result(
            "doc_inference",
            "The transformer model was deployed in the production cluster for inference."
        )
    ]
    generator = DeterministicGroundedGenerator()
    response = generator.generate(query=query, evidence=evidence, action=CRAGAction.TRUST)

    assert response.status == GenerationStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == []


# ---------------------------------------------------------------------------
# 16. Audit QA Relevance: Question Asks for Specific Value Missing in Evidence
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_audit_qa_relevance_specific_value_missing():
    """Evidence contains related facts but not the requested numeric value/count."""
    query = "How many layers are in the BERT base architecture?"
    # Mentions BERT base architecture and layers conceptually, but lacks a numeric count/value
    evidence = [
        make_retrieval_result(
            "doc_bert",
            "BERT base architecture uses transformer encoder layers with bidirectional attention."
        )
    ]
    generator = DeterministicGroundedGenerator()
    response = generator.generate(query=query, evidence=evidence, action=CRAGAction.TRUST)

    assert response.status == GenerationStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == []


# ---------------------------------------------------------------------------
# 17. Audit Contradiction Handling End-to-End via OrchestratorResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_audit_contradiction_detection_end_to_end_pipeline():
    """
    Traces upstream contradiction signal (Intent Dominance override in router)
    through QueryDecision in OrchestratorResult to the GroundedResponseGenerator.
    """
    query = "Explain sparse retrieval"
    doc_id = "doc_contradictory"
    doc_text = "This document contains extensive sufficiency details but does not match user intent."
    res = make_retrieval_result(doc_id, doc_text)

    # Intent score fails (< 0.5), but sufficiency passes (>= 0.5)
    # CorrectiveDecisionRouter triggers Intent Dominance Rule: is_contradictory_override=True, action=REJECT_SEARCH
    retriever = StubRetriever({query: [res]})
    evaluator = MockRuntimeEvaluator(pair_scores={(query, doc_id): (0.1, 0.9)})
    router = CorrectiveDecisionRouter(intent_threshold=0.5, sufficiency_threshold=0.5)
    refiner = MockEvidenceRefiner()

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
        rewriter=None,  # Terminal on REJECT_SEARCH
    )
    orch_result = orch.run_pipeline(query)

    # Verify router flagged candidate decision
    assert orch_result.query_decision.candidate_decisions[0].is_contradictory_override is True

    # Generator adapter consumes OrchestratorResult
    generator = DeterministicGroundedGenerator()
    grounded_res = generator.generate_from_result(orch_result)

    assert grounded_res.status == GenerationStatus.INSUFFICIENT_EVIDENCE
    assert "intent-dominance override" in grounded_res.answer.lower()
    assert grounded_res.metadata.get("is_contradictory_override") is True
    assert grounded_res.citations == []


# ---------------------------------------------------------------------------
# 19. Audit Mixed-Candidate Scenario: TRUST With Rejected Flagged Candidate
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_audit_mixed_candidates_trust_with_rejected_flagged_candidate():
    """
    Mixed-candidate scenario:
    - Candidate 1 (doc_good): Intent passes, Sufficiency passes -> TRUST
    - Candidate 2 (doc_intent_fail): Intent fails, Sufficiency passes -> REJECT_SEARCH with is_contradictory_override=True
    - Overall Action: TRUST (because at least 1 candidate achieved TRUST)
    - final_results: [doc_good]
    - Generator must NOT let candidate 2's override flag invalidate candidate 1's trusted evidence.
    - Citations must contain doc_good only; doc_intent_fail must NOT be cited.
    """
    query = "What is Reciprocal Rank Fusion?"
    doc_good_id = "doc_rrf_good"
    doc_bad_id = "doc_offtopic_dense"

    res_good = make_retrieval_result(
        doc_good_id,
        "Reciprocal Rank Fusion is a hybrid ranking method combining rankings without score calibration.",
        score=0.95,
        rank=1,
    )
    res_bad = make_retrieval_result(
        doc_bad_id,
        "Cooking techniques require detailed temperature regulation and butter emulsification.",
        score=0.20,
        rank=2,
    )

    retriever = StubRetriever({query: [res_good, res_bad]})
    evaluator = MockRuntimeEvaluator(
        pair_scores={
            (query, doc_good_id): (1.0, 1.0),   # TRUST
            (query, doc_bad_id): (0.1, 0.9),    # REJECT_SEARCH with is_contradictory_override=True
        }
    )
    router = CorrectiveDecisionRouter(intent_threshold=0.5, sufficiency_threshold=0.5)
    refiner = MockEvidenceRefiner()

    orch = CRAGOrchestrator(
        retriever=retriever,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
    )
    orch_result = orch.run_pipeline(query)

    # Invariants from pipeline
    assert orch_result.final_action == CRAGAction.TRUST
    assert len(orch_result.final_results) == 1
    assert orch_result.final_results[0].document.doc_id == doc_good_id
    bad_cand = next(c for c in orch_result.query_decision.candidate_decisions if c.prediction.doc_id == doc_bad_id)
    assert bad_cand.is_contradictory_override is True

    # Generator execution
    generator = DeterministicGroundedGenerator()
    grounded_res = generator.generate_from_result(orch_result)

    # Assertions: Generator successfully answers from trusted evidence
    assert grounded_res.status == GenerationStatus.GROUNDED
    assert grounded_res.action_used == CRAGAction.TRUST
    assert "Reciprocal Rank Fusion" in grounded_res.answer
    assert f"[{doc_good_id}]" in grounded_res.answer
    # Citations strictly contain the trusted document and NOT the rejected candidate
    assert len(grounded_res.citations) == 1
    assert grounded_res.citations[0].doc_id == doc_good_id
    assert all(c.doc_id != doc_bad_id for c in grounded_res.citations)


# ---------------------------------------------------------------------------
# 20. Audit VERIFY_MORE Handling: Empty Refined Evidence
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_audit_verify_more_empty_refined_evidence():
    """VERIFY_MORE with empty refined evidence produces safe refusal."""
    query = "What is FAISS IndexHNSW?"
    refinement_res = RefinementResult(
        query_text=query,
        original_doc_ids=["doc_1"],
        refined_results=[],
        refined_documents=[],
        refinement_strategy="secondary_retrieval",
        is_expanded=False,
    )

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=[],
        action=CRAGAction.VERIFY_MORE,
        refinement_result=refinement_res,
    )

    assert response.status == GenerationStatus.INSUFFICIENT_EVIDENCE
    assert response.action_used == CRAGAction.VERIFY_MORE
    assert response.citations == []
    assert response.refusal_reason == "Evidence list is empty."


# ---------------------------------------------------------------------------
# 21. Audit Failure Handling: Malformed Evidence List Produces FAILED Status
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_audit_failure_handling_malformed_evidence():
    """Malformed or non-RetrievalResult items in evidence produce GenerationStatus.FAILED."""
    query = "What is RAG?"
    malformed_evidence = ["not_a_retrieval_result", None]  # type: ignore

    generator = DeterministicGroundedGenerator()
    response = generator.generate(
        query=query,
        evidence=malformed_evidence,  # type: ignore
        action=CRAGAction.TRUST,
    )

    assert response.status == GenerationStatus.FAILED
    assert response.citations == []
    assert "failed" in response.answer.lower()
    assert "malformed" in str(response.refusal_reason).lower()

