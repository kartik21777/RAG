"""
Unit and integration tests for dataset I/O and fixture integrity.
Verifies data/fixtures/ files and serialization round-tripping.
"""

from pathlib import Path
import pytest
from src.core.dataset import (
    load_corpus_from_jsonl,
    save_corpus_to_jsonl,
    load_queries_from_jsonl,
    save_queries_to_jsonl,
    load_evaluator_examples_from_jsonl,
    save_evaluator_examples_to_jsonl,
)
from src.core.schemas import Document, Query, EvaluatorExample, LabelEnum


FIXTURES_DIR = Path("data/fixtures")


@pytest.mark.unit
def test_load_fixture_corpus():
    corpus_path = FIXTURES_DIR / "corpus.jsonl"
    documents = load_corpus_from_jsonl(corpus_path)
    assert len(documents) == 10
    doc_ids = [d.doc_id for d in documents]
    assert doc_ids == [f"doc_{i:03d}" for i in range(1, 11)]

    # Check content of doc_001
    doc_1 = documents[0]
    assert "Reciprocal Rank Fusion" in doc_1.text
    assert doc_1.metadata.get("source") == "ir_textbook"


@pytest.mark.unit
def test_load_fixture_queries():
    queries_path = FIXTURES_DIR / "queries.jsonl"
    queries = load_queries_from_jsonl(queries_path)
    assert len(queries) == 6
    query_ids = [q.query_id for q in queries]
    assert query_ids == [f"q_{i:03d}" for i in range(1, 7)]

    # Check query 5 (unanswerable query)
    q5 = queries[4]
    assert "RSA encryption" in q5.query_text
    assert q5.expected_doc_ids == []
    assert q5.expected_action.value == "REJECT_SEARCH"


@pytest.mark.unit
def test_load_fixture_evaluator_examples():
    eval_path = FIXTURES_DIR / "evaluator_labels.jsonl"
    examples = load_evaluator_examples_from_jsonl(eval_path)
    assert len(examples) == 5

    # Confirm every fixture example adheres strictly to one of the 3 valid actions
    actions = {ex.crag_action.value for ex in examples}
    assert actions.issubset({"TRUST", "VERIFY_MORE", "REJECT_SEARCH"})

    # Confirm no example violates intent_fit=0, evidence_sufficiency=1
    for ex in examples:
        assert not (ex.intent_fit == LabelEnum.NEGATIVE and ex.evidence_sufficiency == LabelEnum.POSITIVE)


@pytest.mark.unit
def test_duplicate_document_id_rejection(tmp_path):
    bad_corpus = tmp_path / "bad_corpus.jsonl"
    bad_corpus.write_text(
        '{"doc_id": "doc_001", "text": "First doc"}\n'
        '{"doc_id": "doc_001", "text": "Duplicate doc"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_corpus_from_jsonl(bad_corpus)
    assert "Duplicate document ID detected" in str(exc.value)


@pytest.mark.unit
def test_invalid_json_rejection(tmp_path):
    corrupt_file = tmp_path / "corrupt.jsonl"
    corrupt_file.write_text(
        '{"doc_id": "doc_001", "text": "Valid doc"}\n'
        '{malformed_json: true}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_corpus_from_jsonl(corrupt_file)
    assert "Invalid JSON at line 2" in str(exc.value)


@pytest.mark.unit
def test_corpus_serialization_roundtrip(tmp_path):
    out_file = tmp_path / "test_corpus_roundtrip.jsonl"
    original = [
        Document(doc_id="d_1", text="Doc 1 content", metadata={"a": 1}),
        Document(doc_id="d_2", text="Doc 2 content", metadata={"b": 2}),
    ]
    save_corpus_to_jsonl(original, out_file)
    reloaded = load_corpus_from_jsonl(out_file)
    assert len(reloaded) == 2
    assert reloaded[0].doc_id == original[0].doc_id
    assert reloaded[0].text == original[0].text
    assert reloaded[0].metadata == original[0].metadata


@pytest.mark.unit
def test_evaluator_example_serialization_roundtrip(tmp_path):
    out_file = tmp_path / "test_eval_roundtrip.jsonl"
    original = [
        EvaluatorExample(
            example_id="ex_1",
            query_id="q_1",
            query_text="Sample query?",
            doc_id="d_1",
            doc_text="Sample doc text.",
            intent_fit=LabelEnum.POSITIVE,
            evidence_sufficiency=LabelEnum.POSITIVE,
            example_type="relevant_sufficient",
            rationale="Exact match with full detail.",
            provenance={"annotator": "test"},
        )
    ]
    save_evaluator_examples_to_jsonl(original, out_file)
    reloaded = load_evaluator_examples_from_jsonl(out_file)
    assert len(reloaded) == 1
    assert reloaded[0].example_id == original[0].example_id
    assert reloaded[0].crag_action == original[0].crag_action
