"""
Unit tests for evidence refiners in src/orchestration/refiner.py.
Verifies BaseEvidenceRefiner, MockEvidenceRefiner, SecondaryRetrievalRefiner,
trigger and skip behavior, bounded retrieval depth, provenance preservation,
adjacent-chunk metadata expansion, and strict deduplication without loading heavy models.
"""

from typing import List, Optional, Tuple, Union
import pytest

from src.core.schemas import Document, Query, RetrievalResult
from src.orchestration.refiner import (
    BaseEvidenceRefiner,
    MockEvidenceRefiner,
    SecondaryRetrievalRefiner,
    wrap_document_as_retrieval_result,
)
from src.orchestration.schemas import RefinementResult
from src.retrieval.facade import CRAGRetrieverFacade


class MockRetrieverFacade(CRAGRetrieverFacade):
    """
    Lightweight, deterministic test double for CRAGRetrieverFacade.
    Zero PyTorch loading, zero model weight downloads, zero FAISS indexing overhead.
    """

    def __init__(self, canned_results: Optional[List[RetrievalResult]] = None, is_ready: bool = True):
        self._canned_results = canned_results or []
        self._is_ready = is_ready
        self.retrieve_calls: List[Tuple[str, int]] = []

    @property
    def is_indexed(self) -> bool:
        return self._is_ready

    def retrieve(self, query: Union[str, Query], top_k: int = 5) -> List[RetrievalResult]:
        if not self.is_indexed:
            raise RuntimeError("Retriever facade is not indexed.")
        query_str = query.query_text if isinstance(query, Query) else query
        self.retrieve_calls.append((query_str, top_k))
        return [r.model_copy(deep=True) for r in self._canned_results[:top_k]]


def make_sample_result(doc_id: str, text: str, score: float = 0.032, rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        document=Document(doc_id=doc_id, text=text, metadata={"source": "test_corpus"}),
        fused_score=score,
        fused_rank=rank,
        sparse_rank=rank,
        dense_rank=rank,
        sparse_score=4.5,
        dense_score=0.75,
    )


# -----------------------------------------------------------------------------
# 1. BaseEvidenceRefiner & Deduplication Utility Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_deduplicate_results_utility():
    r1 = make_sample_result("doc_1", "Text 1", rank=1)
    r2 = make_sample_result("doc_2", "Text 2", rank=2)
    r1_dup = make_sample_result("doc_1", "Duplicate Text", rank=3)

    deduped = BaseEvidenceRefiner.deduplicate_results([r1, r2, r1_dup])
    assert len(deduped) == 2
    assert [r.document.doc_id for r in deduped] == ["doc_1", "doc_2"]
    assert deduped[0].document.text == "Text 1"
    assert deduped[0].fused_rank == 1


@pytest.mark.unit
def test_initial_pool_duplicate_rejection():
    r1 = make_sample_result("doc_dup", "Text A")
    r2 = make_sample_result("doc_dup", "Text B")

    with pytest.raises(ValueError) as exc:
        BaseEvidenceRefiner.validate_initial_pool_uniqueness([r1, r2])
    assert "Duplicate document ID 'doc_dup'" in str(exc.value)


# -----------------------------------------------------------------------------
# 2. Refinement Triggers & Skip Behavior Tests (Section 2)
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_refiner_skipped_when_no_verification_targets():
    # Non-empty initial results with zero verification targets -> SKIPPED without search
    canned = [make_sample_result("doc_001", "Text 1")]
    facade = MockRetrieverFacade(canned_results=canned)
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade)

    initial = [make_sample_result("doc_001", "Text 1")]
    result = refiner.refine(
        query="What is RRF?",
        initial_results=initial,
        verification_target_ids=[],  # Zero targets
    )

    assert result.skipped is True
    assert result.is_expanded is False
    assert result.refinement_strategy == "skipped_no_targets"
    assert len(result.added_results) == 0
    # Zero secondary retrieval calls made!
    assert len(facade.retrieve_calls) == 0


@pytest.mark.unit
def test_refiner_empty_initial_triggers_recovery():
    # Empty initial retrieval triggers recovery attempt
    canned = [make_sample_result("doc_rec", "Recovered document text")]
    facade = MockRetrieverFacade(canned_results=canned)
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade, secondary_top_k=2)

    result = refiner.refine(
        query="What is RRF?",
        initial_results=[],
    )

    assert result.skipped is False
    assert result.is_expanded is True
    assert result.refinement_strategy == "recovery_retrieval"
    assert result.added_doc_ids == ["doc_rec"]
    assert len(facade.retrieve_calls) == 1
    assert facade.retrieve_calls[0] == ("What is RRF?", 2)


@pytest.mark.unit
def test_refiner_attempted_but_no_new_evidence_distinguishable():
    # Retrieval was attempted, but the returned results already existed in initial pool
    existing_r1 = make_sample_result("doc_001", "Text 1")
    facade = MockRetrieverFacade(canned_results=[existing_r1])
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade)

    result = refiner.refine(
        query="What is RRF?",
        initial_results=[existing_r1],
        verification_target_ids=["doc_001"],
    )

    # Distinguishable: attempted (skipped is False), but discovered no new evidence (is_expanded is False)
    assert result.skipped is False
    assert result.is_expanded is False
    assert result.refinement_strategy == "secondary_retrieval"
    assert len(result.added_results) == 0
    assert len(result.refined_results) == 1
    assert len(facade.retrieve_calls) == 1


@pytest.mark.unit
def test_refiner_inconsistent_target_ids_rejected():
    initial = [make_sample_result("doc_001", "Text 1")]
    facade = MockRetrieverFacade()
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade)

    with pytest.raises(ValueError) as exc:
        refiner.refine(
            query="query",
            initial_results=initial,
            verification_target_ids=["doc_missing_target"],
        )
    assert "Inconsistent verification_target_ids: target 'doc_missing_target'" in str(exc.value)


@pytest.mark.unit
def test_refiner_at_most_one_retrieval_call():
    canned = [make_sample_result("doc_extra", "Extra text")]
    facade = MockRetrieverFacade(canned_results=canned)
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade)

    initial = [
        make_sample_result("doc_001", "Text 1"),
        make_sample_result("doc_002", "Text 2"),
    ]
    refiner.refine(
        query="query",
        initial_results=initial,
        verification_target_ids=["doc_001", "doc_002"],
    )
    # Must perform at most one secondary retrieval call per invocation
    assert len(facade.retrieve_calls) == 1


# -----------------------------------------------------------------------------
# 3. Retrieval Depth & Input Validation Tests (Section 3)
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_refiner_secondary_depth_validation():
    facade = MockRetrieverFacade()

    # Zero rejected
    with pytest.raises(ValueError):
        SecondaryRetrievalRefiner(retriever_facade=facade, secondary_top_k=0)

    # Negative rejected
    with pytest.raises(ValueError):
        SecondaryRetrievalRefiner(retriever_facade=facade, secondary_top_k=-2)

    # Boolean rejected (True is an int subclass in Python!)
    with pytest.raises(ValueError):
        SecondaryRetrievalRefiner(retriever_facade=facade, secondary_top_k=True)  # type: ignore

    # Non-integer float rejected
    with pytest.raises(ValueError):
        SecondaryRetrievalRefiner(retriever_facade=facade, secondary_top_k=3.5)  # type: ignore

    # String rejected
    with pytest.raises(ValueError):
        SecondaryRetrievalRefiner(retriever_facade=facade, secondary_top_k="3")  # type: ignore


@pytest.mark.unit
def test_refiner_bounded_fetch_depth_beyond_initial_pool():
    # Secondary retrieval requests fetch_depth = len(initial_pool) + secondary_top_k
    facade = MockRetrieverFacade()
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade, secondary_top_k=4)

    initial = [
        make_sample_result("doc_001", "Text 1"),
        make_sample_result("doc_002", "Text 2"),
    ]
    refiner.refine(
        query="test query",
        initial_results=initial,
        verification_target_ids=["doc_001"],
    )

    assert len(facade.retrieve_calls) == 1
    # Initial pool has 2 docs, secondary_top_k=4 -> fetch_depth must be 2 + 4 = 6
    assert facade.retrieve_calls[0][1] == 6


@pytest.mark.unit
def test_refiner_handles_corpus_returning_fewer_than_requested():
    # Requested fetch depth is 5, but corpus only has 1 result
    canned = [make_sample_result("doc_only", "Single corpus text")]
    facade = MockRetrieverFacade(canned_results=canned)
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade, secondary_top_k=4)

    result = refiner.refine("query", initial_results=[])
    assert result.is_expanded is True
    assert len(result.refined_results) == 1
    assert result.refined_results[0].document.doc_id == "doc_only"


# -----------------------------------------------------------------------------
# 4. Provenance Preservation & Immutability Tests (Section 4)
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_refinement_preserves_full_retrieval_result_provenance():
    r1 = make_sample_result("doc_001", "Text 1", score=0.032, rank=1)
    r_new = make_sample_result("doc_new", "Newly retrieved text", score=0.016, rank=2)

    facade = MockRetrieverFacade(canned_results=[r_new])
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade)

    initial = [r1]
    result = refiner.refine(
        query="What is RRF?",
        initial_results=initial,
        verification_target_ids=["doc_001"],
    )

    assert result.is_expanded is True
    assert len(result.refined_results) == 2

    # Initial result preserves original provenance
    assert result.refined_results[0].document.doc_id == "doc_001"
    assert result.refined_results[0].fused_score == 0.032
    assert result.refined_results[0].sparse_rank == 1
    assert result.refined_results[0].dense_rank == 1

    # Newly added result preserves secondary retrieval provenance
    assert result.refined_results[1].document.doc_id == "doc_new"
    assert result.refined_results[1].fused_score == 0.016
    assert result.refined_results[1].sparse_score == 4.5
    assert result.refined_results[1].dense_score == 0.75

    # Refined documents property provides easy access
    assert len(result.refined_documents) == 2
    assert result.refined_documents[0].doc_id == "doc_001"
    assert result.refined_documents[1].doc_id == "doc_new"

    # Caller's original input list and objects were not mutated
    assert len(initial) == 1
    assert initial[0].document.doc_id == "doc_001"


# -----------------------------------------------------------------------------
# 5. Adjacent-Chunk Fallback Tests (Section 5)
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_adjacent_chunk_used_when_metadata_exists():
    adj_doc = Document(doc_id="chunk_002", text="Adjacent continuation text.", metadata={"chunk_id": 2})
    corpus_lookup = {"chunk_002": adj_doc}

    incomplete_doc = Document(
        doc_id="chunk_001",
        text="First chunk.",
        metadata={"adjacent_chunk_ids": ["chunk_002"]},
    )
    incomplete_res = wrap_document_as_retrieval_result(incomplete_doc, rank=1)

    facade = MockRetrieverFacade()
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade, corpus_lookup=corpus_lookup)

    result = refiner.refine("query", initial_results=[incomplete_res], verification_target_ids=["chunk_001"])
    assert result.refinement_strategy == "adjacent_chunk"
    assert result.metadata["adjacent_metadata_found"] is True
    assert result.added_doc_ids == ["chunk_002"]
    # Zero secondary retrieval calls made because adjacent metadata succeeded!
    assert len(facade.retrieve_calls) == 0


@pytest.mark.unit
def test_adjacent_chunk_falls_back_when_metadata_lacks_chunk_links():
    # Document has no adjacent chunk links -> clean fallback to secondary retrieval
    incomplete_doc = Document(doc_id="doc_no_adj", text="Text without adjacent links.", metadata={"chunk_id": 1})
    incomplete_res = wrap_document_as_retrieval_result(incomplete_doc, rank=1)

    canned = [make_sample_result("doc_fallback", "Retrieved fallback evidence")]
    facade = MockRetrieverFacade(canned_results=canned)
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade)

    result = refiner.refine("query", initial_results=[incomplete_res], verification_target_ids=["doc_no_adj"])
    assert result.refinement_strategy == "secondary_retrieval"
    assert result.metadata["adjacent_metadata_found"] is False
    assert result.metadata["secondary_retrieval_executed"] is True
    assert result.added_doc_ids == ["doc_fallback"]
    assert len(facade.retrieve_calls) == 1
    assert facade.retrieve_calls[0][0] == "query"


# -----------------------------------------------------------------------------
# 6. MockEvidenceRefiner Contract Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_mock_evidence_refiner_full_contract():
    r1 = make_sample_result("doc_part", "Partial evidence")
    r_exp = make_sample_result("doc_exp", "Expanded evidence")

    refiner = MockEvidenceRefiner(expansion_map={"doc_part": [r_exp]})
    result = refiner.refine("query", initial_results=[r1], verification_target_ids=["doc_part"])

    assert result.is_expanded is True
    assert result.original_doc_ids == ["doc_part"]
    assert result.added_doc_ids == ["doc_exp"]
    assert len(result.refined_results) == 2
    assert [r.document.doc_id for r in result.refined_results] == ["doc_part", "doc_exp"]


# -----------------------------------------------------------------------------
# 7. Milestone 3 Final Scope Correction Regression Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_secondary_retrieval_passes_exact_unmodified_query():
    """
    Regression test for Milestone 3 final scope correction:
    Verifies that SecondaryRetrievalRefiner passes the original validated query
    to the retrieval facade completely UNCHANGED (no appending ' context details',
    no heuristic suffixes, no rewriting).
    Also verifies:
    - Expected increased top_k depth (len(initial) + secondary_top_k) is used
    - Secondary results deduplicated against initial pool
    - Original retrieval metadata, scores, and ranks are preserved
    - Recovery retrieval also preserves exact query
    """
    original_query = "What is dense passage retrieval?"

    initial_doc_1 = make_sample_result("doc_1", "Initial passage 1", score=0.88, rank=1)
    initial_doc_2 = make_sample_result("doc_2", "Initial passage 2", score=0.72, rank=2)
    initial_pool = [initial_doc_1, initial_doc_2]

    # Canned secondary results: one duplicate (doc_1) and one new document (doc_3)
    secondary_dup = make_sample_result("doc_1", "Duplicate of doc_1 from secondary", score=0.99, rank=1)
    secondary_new = make_sample_result("doc_3", "Newly retrieved passage 3", score=0.65, rank=2)
    facade = MockRetrieverFacade(canned_results=[secondary_dup, secondary_new])

    secondary_top_k = 3
    refiner = SecondaryRetrievalRefiner(retriever_facade=facade, secondary_top_k=secondary_top_k)

    result = refiner.refine(
        query=original_query,
        initial_results=initial_pool,
        verification_target_ids=["doc_1"],
    )

    # 1. Exact query passed to retrieval facade
    assert len(facade.retrieve_calls) == 1
    passed_query, passed_top_k = facade.retrieve_calls[0]
    assert passed_query == original_query
    assert "context details" not in passed_query
    assert passed_query == "What is dense passage retrieval?"

    # 2. Increased top_k depth used: len(initial_pool) + secondary_top_k = 2 + 3 = 5
    assert passed_top_k == 5

    # 3. Deduplication: doc_1 duplicate from secondary is filtered out, doc_3 is added
    assert result.is_expanded is True
    assert result.added_doc_ids == ["doc_3"]
    assert len(result.refined_results) == 3
    assert [r.document.doc_id for r in result.refined_results] == ["doc_1", "doc_2", "doc_3"]

    # 4. Original retrieval metadata and scores preserved
    assert result.refined_results[0].document.doc_id == "doc_1"
    assert result.refined_results[0].fused_score == 0.88
    assert result.refined_results[0].fused_rank == 1
    assert result.refined_results[0].document.text == "Initial passage 1"

    assert result.refined_results[1].document.doc_id == "doc_2"
    assert result.refined_results[1].fused_score == 0.72
    assert result.refined_results[1].fused_rank == 2

    # 5. Secondary result provenance preserved
    assert result.refined_results[2].document.doc_id == "doc_3"
    assert result.refined_results[2].fused_score == 0.65
    assert result.refined_results[2].document.text == "Newly retrieved passage 3"

    # 6. Verify another distinct query string also passes exact query text without modification
    facade_query_2 = MockRetrieverFacade(canned_results=[secondary_new])
    refiner_query_2 = SecondaryRetrievalRefiner(retriever_facade=facade_query_2, secondary_top_k=2)
    refiner_query_2.refine(
        query="How does CRAG handle ambiguity?",
        initial_results=[initial_doc_1],
        verification_target_ids=["doc_1"],
    )
    assert facade_query_2.retrieve_calls[0][0] == "How does CRAG handle ambiguity?"
    assert "context details" not in facade_query_2.retrieve_calls[0][0]

    # 7. Empty-retrieval recovery also uses exact query
    facade_recovery = MockRetrieverFacade(canned_results=[secondary_new])
    refiner_recovery = SecondaryRetrievalRefiner(retriever_facade=facade_recovery, secondary_top_k=4)
    recovery_result = refiner_recovery.refine(
        query="Explain CRAG routing",
        initial_results=[],
    )
    assert len(facade_recovery.retrieve_calls) == 1
    assert facade_recovery.retrieve_calls[0] == ("Explain CRAG routing", 4)
    assert recovery_result.refinement_strategy == "recovery_retrieval"
    assert recovery_result.is_expanded is True
