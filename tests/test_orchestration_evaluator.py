"""
Unit tests for CRAG runtime evaluation contracts and MockRuntimeEvaluator.
Verifies EvaluatorPrediction schema validation, score bounding,
mock determinism, and configuration patterns without requiring a GPU or API keys.
"""

import math
import pytest
from pydantic import ValidationError

from src.core.schemas import Document
from src.orchestration.evaluator import BaseRuntimeEvaluator, MockRuntimeEvaluator
from src.orchestration.schemas import EvaluatorPrediction


@pytest.fixture
def sample_document():
    return Document(
        doc_id="doc_001",
        text="Reciprocal Rank Fusion (RRF) combines sparse and dense rankings.",
        metadata={"title": "RRF Overview"},
    )


# -----------------------------------------------------------------------------
# 1. EvaluatorPrediction Schema Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_evaluator_prediction_valid_construction():
    pred = EvaluatorPrediction(
        query_text="What is Reciprocal Rank Fusion?",
        doc_id="doc_001",
        intent_fit_score=0.95,
        evidence_sufficiency_score=0.90,
        rationale="Document explicitly defines RRF formula and methodology.",
        metadata={"raw_logits": [2.4, 1.8]},
    )
    assert pred.query_text == "What is Reciprocal Rank Fusion?"
    assert pred.doc_id == "doc_001"
    assert pred.intent_fit_score == 0.95
    assert pred.evidence_sufficiency_score == 0.90
    assert pred.rationale == "Document explicitly defines RRF formula and methodology."
    assert pred.metadata["raw_logits"] == [2.4, 1.8]


@pytest.mark.unit
def test_evaluator_prediction_score_bounds_rejection():
    # Score < 0.0 rejected
    with pytest.raises(ValidationError) as exc:
        EvaluatorPrediction(
            query_text="valid query",
            doc_id="doc_001",
            intent_fit_score=-0.01,
            evidence_sufficiency_score=0.5,
        )
    assert "Score must be within [0.0, 1.0]" in str(exc.value)

    # Score > 1.0 rejected
    with pytest.raises(ValidationError) as exc:
        EvaluatorPrediction(
            query_text="valid query",
            doc_id="doc_001",
            intent_fit_score=0.5,
            evidence_sufficiency_score=1.0001,
        )
    assert "Score must be within [0.0, 1.0]" in str(exc.value)


@pytest.mark.unit
def test_evaluator_prediction_nan_inf_rejection():
    # NaN rejected
    with pytest.raises(ValidationError) as exc:
        EvaluatorPrediction(
            query_text="valid query",
            doc_id="doc_001",
            intent_fit_score=float("nan"),
            evidence_sufficiency_score=0.5,
        )
    assert "Score must be a finite number" in str(exc.value)

    # Inf rejected
    with pytest.raises(ValidationError) as exc:
        EvaluatorPrediction(
            query_text="valid query",
            doc_id="doc_001",
            intent_fit_score=0.5,
            evidence_sufficiency_score=float("inf"),
        )
    assert "Score must be a finite number" in str(exc.value)


@pytest.mark.unit
def test_evaluator_prediction_required_fields_and_whitespace():
    # Empty query rejected
    with pytest.raises(ValidationError):
        EvaluatorPrediction(
            query_text="",
            doc_id="doc_001",
            intent_fit_score=0.8,
            evidence_sufficiency_score=0.8,
        )

    # Whitespace-only query rejected
    with pytest.raises(ValidationError):
        EvaluatorPrediction(
            query_text="   ",
            doc_id="doc_001",
            intent_fit_score=0.8,
            evidence_sufficiency_score=0.8,
        )

    # Empty doc_id rejected
    with pytest.raises(ValidationError):
        EvaluatorPrediction(
            query_text="valid query",
            doc_id="",
            intent_fit_score=0.8,
            evidence_sufficiency_score=0.8,
        )


# -----------------------------------------------------------------------------
# 2. MockRuntimeEvaluator Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_mock_evaluator_implements_interface():
    evaluator = MockRuntimeEvaluator()
    assert isinstance(evaluator, BaseRuntimeEvaluator)


@pytest.mark.unit
def test_mock_evaluator_deterministic_reproducibility(sample_document):
    evaluator = MockRuntimeEvaluator(
        pair_scores={("What is RRF?", "doc_001"): (0.95, 0.85)},
    )
    pred1 = evaluator.evaluate_pair("What is RRF?", sample_document)
    pred2 = evaluator.evaluate_pair("What is RRF?", sample_document)

    assert pred1.intent_fit_score == pred2.intent_fit_score == 0.95
    assert pred1.evidence_sufficiency_score == pred2.evidence_sufficiency_score == 0.85
    assert pred1.doc_id == sample_document.doc_id
    assert pred1.query_text == "What is RRF?"


@pytest.mark.unit
def test_mock_evaluator_all_three_score_patterns():
    doc_trust = Document(doc_id="doc_trust", text="Sufficient evidence text.")
    doc_verify = Document(doc_id="doc_verify", text="Partial evidence text.")
    doc_reject = Document(doc_id="doc_reject", text="Off-topic text.")

    query = "Explain hybrid RAG routing"

    evaluator = MockRuntimeEvaluator(
        pair_scores={
            (query, "doc_trust"): (0.95, 0.90),   # Pattern 1: High intent, high sufficiency
            (query, "doc_verify"): (0.92, 0.25),  # Pattern 2: High intent, low sufficiency
            (query, "doc_reject"): (0.10, 0.85),  # Pattern 3: Low intent, regardless of sufficiency score
        }
    )

    p_trust = evaluator.evaluate_pair(query, doc_trust)
    assert p_trust.intent_fit_score == 0.95
    assert p_trust.evidence_sufficiency_score == 0.90

    p_verify = evaluator.evaluate_pair(query, doc_verify)
    assert p_verify.intent_fit_score == 0.92
    assert p_verify.evidence_sufficiency_score == 0.25

    p_reject = evaluator.evaluate_pair(query, doc_reject)
    assert p_reject.intent_fit_score == 0.10
    assert p_reject.evidence_sufficiency_score == 0.85


@pytest.mark.unit
def test_mock_evaluator_multiple_documents_one_query():
    docs = [
        Document(doc_id=f"doc_{i}", text=f"Text for passage {i}")
        for i in range(1, 4)
    ]
    query = "Multi-document evaluation query"

    evaluator = MockRuntimeEvaluator(
        doc_scores={
            "doc_1": (0.9, 0.8),
            "doc_2": (0.8, 0.3),
            "doc_3": (0.1, 0.0),
        }
    )

    preds = [evaluator.evaluate_pair(query, d) for d in docs]
    assert [p.doc_id for p in preds] == ["doc_1", "doc_2", "doc_3"]
    assert [p.intent_fit_score for p in preds] == [0.9, 0.8, 0.1]
    assert [p.evidence_sufficiency_score for p in preds] == [0.8, 0.3, 0.0]


@pytest.mark.unit
def test_mock_evaluator_default_fallback_and_metadata(sample_document):
    evaluator = MockRuntimeEvaluator(
        default_scores=(0.4, 0.2),
        default_rationale="Generic fallback rationale",
    )
    pred = evaluator.evaluate_pair("Unmapped question", sample_document)
    assert pred.intent_fit_score == 0.4
    assert pred.evidence_sufficiency_score == 0.2
    assert pred.rationale == "Generic fallback rationale"
    assert pred.metadata["mock_source"] == "default"
    assert pred.metadata["is_deterministic_mock"] is True


@pytest.mark.unit
def test_mock_evaluator_invalid_input_rejection(sample_document):
    evaluator = MockRuntimeEvaluator()

    # Empty query string
    with pytest.raises(ValueError) as exc:
        evaluator.evaluate_pair("", sample_document)
    assert "cannot be empty" in str(exc.value)

    # Non-string query
    with pytest.raises(TypeError) as exc:
        evaluator.evaluate_pair(12345, sample_document)  # type: ignore
    assert "query must be str" in str(exc.value)

    # Non-Document document argument
    with pytest.raises(TypeError) as exc:
        evaluator.evaluate_pair("valid query", "not a document")  # type: ignore
    assert "doc must be Document" in str(exc.value)


@pytest.mark.unit
def test_mock_evaluator_invalid_configuration_bounds():
    # Invalid score in pair_scores
    with pytest.raises(ValueError) as exc:
        MockRuntimeEvaluator(pair_scores={("q", "d"): (1.5, 0.5)})
    assert "must be in [0.0, 1.0]" in str(exc.value)

    # Invalid score in doc_scores
    with pytest.raises(ValueError) as exc:
        MockRuntimeEvaluator(doc_scores={"d": (-0.1, 0.5)})
    assert "must be in [0.0, 1.0]" in str(exc.value)

    # Invalid default_scores length
    with pytest.raises(ValueError) as exc:
        MockRuntimeEvaluator(default_scores=(0.5,))  # type: ignore
    assert "must have exactly 2 elements" in str(exc.value)
