"""
Unit tests for CRAGRetrieverFacade and create_crag_retriever.
Verifies unified retrieval interface, RetrievalResult schema compliance,
persistence, and factory helper functions.
"""

from pathlib import Path
import pytest

from sentence_transformers import SentenceTransformer

from src.core.dataset import load_corpus_from_jsonl
from src.core.schemas import Document, Query, RetrievalResult
from src.retrieval.dense import DenseRetriever
from src.retrieval.facade import CRAGRetrieverFacade, create_crag_retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import BM25Retriever


FIXTURE_CORPUS_PATH = Path("data/fixtures/corpus.jsonl")


@pytest.fixture(scope="module")
def shared_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def fixture_corpus():
    return load_corpus_from_jsonl(FIXTURE_CORPUS_PATH)


@pytest.fixture
def sample_facade(fixture_corpus, shared_model):
    dense = DenseRetriever(model=shared_model)
    sparse = BM25Retriever()
    hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense)
    facade = CRAGRetrieverFacade(hybrid_retriever=hybrid)
    facade.index_corpus(fixture_corpus)
    return facade


@pytest.mark.unit
def test_facade_unindexed_raises_runtime_error():
    facade = CRAGRetrieverFacade()
    assert not facade.is_indexed
    with pytest.raises(RuntimeError) as exc_info:
        facade.retrieve("test query")
    assert "not indexed" in str(exc_info.value)


@pytest.mark.unit
def test_facade_invalid_query_args(sample_facade):
    with pytest.raises(ValueError) as exc:
        sample_facade.retrieve("")
    assert "empty" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        sample_facade.retrieve("valid query", top_k=0)
    assert "positive" in str(exc.value)

    with pytest.raises(TypeError) as exc:
        sample_facade.retrieve(12345)  # type: ignore
    assert "must be str or Query" in str(exc.value)


@pytest.mark.unit
def test_facade_retrieve_emits_valid_retrieval_results(sample_facade):
    assert sample_facade.is_indexed

    # String query targeting RRF formula
    query_str = "What is the mathematical formula for calculating Reciprocal Rank Fusion (RRF) scores?"
    results = sample_facade.retrieve(query_str, top_k=3)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert isinstance(r.document, Document)
        assert r.fused_rank >= 1
        assert r.fused_score > 0.0

    # doc_001 should be rank 1
    assert results[0].document.doc_id == "doc_001"
    assert results[0].fused_rank == 1
    assert results[0].sparse_rank == 1
    assert results[0].dense_rank == 1
    assert results[0].sparse_score is not None
    assert results[0].dense_score is not None

    # Canonical Query object query
    query_obj = Query(query_id="q_test", query_text="BM25 hyperparameters k1 and b")
    results_q = sample_facade.retrieve(query_obj, top_k=2)
    assert len(results_q) == 2
    assert results_q[0].document.doc_id == "doc_003"


@pytest.mark.unit
def test_facade_save_and_load_persistence(sample_facade, tmp_path, shared_model):
    index_dir = tmp_path / "facade_hybrid_index"
    sample_facade.save(index_dir)

    # Restore via load
    loaded_facade = CRAGRetrieverFacade.load(index_dir)
    assert loaded_facade.is_indexed

    query_str = "What is the mathematical formula for calculating Reciprocal Rank Fusion (RRF) scores?"
    results = loaded_facade.retrieve(query_str, top_k=2)
    assert len(results) == 2
    assert results[0].document.doc_id == "doc_001"

    # Restore via factory with index_dir
    factory_facade = create_crag_retriever(index_dir=index_dir)
    assert factory_facade.is_indexed
    results_f = factory_facade.retrieve(query_str, top_k=2)
    assert results_f[0].document.doc_id == "doc_001"

