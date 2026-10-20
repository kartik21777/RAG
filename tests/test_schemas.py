"""
Unit tests for core schemas and validation rules in src/core/schemas.py.
Verifies Document, Query, RetrievalResult, and strict EvaluatorExample label rules.
"""

import pytest
from pydantic import ValidationError
from src.core.schemas import (
    Document,
    Query,
    RetrievalCandidate,
    RetrievalResult,
    EvaluatorExample,
    LabelEnum,
    CRAGAction,
)


@pytest.mark.unit
def test_valid_document_creation():
    doc = Document(doc_id="doc_001", text="This is sample text.", metadata={"title": "Test Doc"})
    assert doc.doc_id == "doc_001"
    assert doc.text == "This is sample text."
    assert doc.metadata["title"] == "Test Doc"


@pytest.mark.unit
def test_invalid_document_empty_fields():
    with pytest.raises(ValidationError):
        Document(doc_id="", text="valid text")

    with pytest.raises(ValidationError):
        Document(doc_id="doc_001", text="")


@pytest.mark.unit
def test_valid_query_creation():
    q = Query(query_id="q_001", query_text="What is RRF?", expected_doc_ids=["doc_001"])
    assert q.query_id == "q_001"
    assert q.query_text == "What is RRF?"
    assert q.expected_doc_ids == ["doc_001"]


@pytest.mark.unit
def test_query_whitespace_rejection():
    with pytest.raises(ValidationError):
        Query(query_id="q_002", query_text="   \t\n  ")


@pytest.mark.unit
def test_retrieval_candidate_and_result():
    doc = Document(doc_id="doc_001", text="Sample document text.")
    cand = RetrievalCandidate(document=doc, score=0.85, rank=1)
    assert cand.rank == 1
    assert cand.score == 0.85

    result = RetrievalResult(
        document=doc,
        fused_score=0.032,
        fused_rank=1,
        sparse_rank=2,
        dense_rank=1,
        sparse_score=4.5,
        dense_score=0.88,
    )
    assert result.fused_rank == 1
    assert result.sparse_rank == 2
    assert result.dense_rank == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "intent,sufficiency,expected_action",
    [
        (LabelEnum.POSITIVE, LabelEnum.POSITIVE, CRAGAction.TRUST),
        (LabelEnum.POSITIVE, LabelEnum.NEGATIVE, CRAGAction.VERIFY_MORE),
        (LabelEnum.NEGATIVE, LabelEnum.NEGATIVE, CRAGAction.REJECT_SEARCH),
    ],
)
def test_evaluator_example_valid_combinations(intent, sufficiency, expected_action):
    example = EvaluatorExample(
        example_id="ex_001",
        query_id="q_001",
        query_text="What is RRF?",
        doc_id="doc_001",
        doc_text="Reciprocal Rank Fusion formula...",
        intent_fit=intent,
        evidence_sufficiency=sufficiency,
        example_type="test_case",
        rationale="Clear justification for test.",
    )
    assert example.intent_fit == intent
    assert example.evidence_sufficiency == sufficiency
    assert example.crag_action == expected_action


@pytest.mark.unit
def test_evaluator_example_prohibited_combination():
    """
    Critical validation check:
    (intent_fit=0, evidence_sufficiency=1) MUST be rejected with a ValidationError.
    A document cannot have sufficient evidence to answer a question whose intent it misses.
    """
    with pytest.raises(ValidationError) as excinfo:
        EvaluatorExample(
            example_id="ex_invalid",
            query_id="q_001",
            query_text="What is RRF?",
            doc_id="doc_009",
            doc_text="Raft distributed consensus...",
            intent_fit=LabelEnum.NEGATIVE,
            evidence_sufficiency=LabelEnum.POSITIVE,
            example_type="prohibited_case",
            rationale="Should fail validation.",
        )
    assert "Prohibited label combination: (intent_fit=0, evidence_sufficiency=1)" in str(excinfo.value)
