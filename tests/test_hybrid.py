"""
Unit and integration tests for HybridRetriever in src/retrieval/hybrid.py.
Verifies Reciprocal Rank Fusion (RRF) rank aggregation math, consensus boost,
parameter boundaries, sub-retriever coordination, error handling, and persistence round-tripping.
"""

from pathlib import Path
import pytest
from src.core.dataset import load_corpus_from_jsonl
from src.core.schemas import Document, RetrievalCandidate
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import DEFAULT_RRF_K, METADATA_FILENAME, HybridRetriever
from src.retrieval.sparse import BM25Retriever


FIXTURE_CORPUS_PATH = Path("data/fixtures/corpus.jsonl")


@pytest.fixture(scope="module")
def shared_model():
    """Module-scoped transformer model to avoid redundant weight reloading."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def sample_corpus():
    return load_corpus_from_jsonl(FIXTURE_CORPUS_PATH)


@pytest.fixture
def indexed_hybrid_retriever(sample_corpus, shared_model):
    sparse = BM25Retriever()
    dense = DenseRetriever(model=shared_model)
    retriever = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense)
    retriever.index(sample_corpus)
    return retriever


@pytest.mark.unit
def test_hybrid_initial_state(shared_model):
    sparse = BM25Retriever()
    dense = DenseRetriever(model=shared_model)
    retriever = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense)
    assert not retriever.is_indexed
    assert retriever.document_count == 0
    assert retriever.rrf_k == DEFAULT_RRF_K
    assert retriever.sparse_weight == 1.0
    assert retriever.dense_weight == 1.0


@pytest.mark.unit
def test_parameter_validation():
    with pytest.raises(ValueError) as exc:
        HybridRetriever(rrf_k=0)
    assert "rrf_k smoothing constant must be a positive integer" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        HybridRetriever(sparse_weight=-0.5)
    assert "sparse_weight must be non-negative" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        HybridRetriever(dense_weight=-1.0)
    assert "dense_weight must be non-negative" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        HybridRetriever(sparse_weight=0, dense_weight=0)
    assert "At least one of sparse_weight or dense_weight must be strictly positive" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        HybridRetriever(sparse_top_k=0)
    assert "sparse_top_k must be a positive integer" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        HybridRetriever(dense_top_k=-2)
    assert "dense_top_k must be a positive integer" in str(exc.value)


@pytest.mark.unit
def test_index_empty_and_duplicate_guards(shared_model):
    retriever = HybridRetriever(
        sparse_retriever=BM25Retriever(),
        dense_retriever=DenseRetriever(model=shared_model),
    )
    with pytest.raises(ValueError):
        retriever.index([])

    dup_docs = [
        Document(doc_id="d_01", text="First doc"),
        Document(doc_id="d_01", text="Duplicate doc"),
    ]
    with pytest.raises(ValueError):
        retriever.index(dup_docs)


@pytest.mark.unit
def test_fuse_empty_candidate_lists():
    retriever = HybridRetriever()
    fused = retriever.fuse([], [], top_k=5)
    assert fused == []


@pytest.mark.unit
def test_fuse_mathematical_correctness_and_consensus_boost():
    """
    Validates Cormack et al. RRF equation:
    score(d) = sum(w_m / (k + rank_m(d)))
    """
    retriever = HybridRetriever(rrf_k=60, sparse_weight=1.0, dense_weight=1.0)

    doc_a = Document(doc_id="doc_A", text="Document A content")
    doc_b = Document(doc_id="doc_B", text="Document B content")
    doc_c = Document(doc_id="doc_C", text="Document C content")

    # Sparse ranking: doc_A (rank 1), doc_B (rank 2)
    sparse_candidates = [
        RetrievalCandidate(document=doc_a, score=4.5, rank=1),
        RetrievalCandidate(document=doc_b, score=2.5, rank=2),
    ]

    # Dense ranking: doc_B (rank 1), doc_C (rank 2)
    dense_candidates = [
        RetrievalCandidate(document=doc_b, score=0.85, rank=1),
        RetrievalCandidate(document=doc_c, score=0.65, rank=2),
    ]

    # Expected RRF calculations:
    # doc_B (consensus): 1/(60+2) [sparse] + 1/(60+1) [dense] = 1/62 + 1/61 ~= 0.032522
    # doc_A (sparse only): 1/(60+1) [sparse] = 1/61 ~= 0.016393
    # doc_C (dense only): 1/(60+2) [dense] = 1/62 ~= 0.016129

    expected_score_b = (1.0 / 62.0) + (1.0 / 61.0)
    expected_score_a = 1.0 / 61.0
    expected_score_c = 1.0 / 62.0

    fused = retriever.fuse(sparse_candidates, dense_candidates, top_k=3)
    assert len(fused) == 3

    # Rank 1 must be doc_B due to consensus agreement across both retrievers
    assert fused[0].document.doc_id == "doc_B"
    assert fused[0].rank == 1
    assert pytest.approx(fused[0].score, rel=1e-5) == expected_score_b

    # Rank 2 must be doc_A
    assert fused[1].document.doc_id == "doc_A"
    assert fused[1].rank == 2
    assert pytest.approx(fused[1].score, rel=1e-5) == expected_score_a

    # Rank 3 must be doc_C
    assert fused[2].document.doc_id == "doc_C"
    assert fused[2].rank == 3
    assert pytest.approx(fused[2].score, rel=1e-5) == expected_score_c


@pytest.mark.unit
def test_fuse_weighting_effect():
    # If dense_weight is set heavily (e.g. 10x), doc_C should beat doc_A
    retriever = HybridRetriever(rrf_k=60, sparse_weight=1.0, dense_weight=10.0)

    doc_a = Document(doc_id="doc_A", text="Document A content")
    doc_c = Document(doc_id="doc_C", text="Document C content")

    sparse_cands = [RetrievalCandidate(document=doc_a, score=5.0, rank=1)]
    dense_cands = [RetrievalCandidate(document=doc_c, score=0.9, rank=1)]

    # doc_A score = 1.0 / (60+1) = 0.01639
    # doc_C score = 10.0 / (60+1) = 0.16393
    fused = retriever.fuse(sparse_cands, dense_cands, top_k=2)
    assert fused[0].document.doc_id == "doc_C"
    assert fused[1].document.doc_id == "doc_A"


@pytest.mark.unit
def test_search_unindexed_error():
    retriever = HybridRetriever()
    with pytest.raises(RuntimeError) as exc:
        retriever.search("fusion pipeline")
    assert "are not indexed" in str(exc.value)


@pytest.mark.unit
def test_search_input_validation(indexed_hybrid_retriever):
    with pytest.raises(ValueError):
        indexed_hybrid_retriever.search("")
    with pytest.raises(ValueError):
        indexed_hybrid_retriever.search("   \t  ")
    with pytest.raises(ValueError):
        indexed_hybrid_retriever.search("valid query", top_k=0)
    with pytest.raises(ValueError):
        indexed_hybrid_retriever.search("valid query", top_k=-5)


@pytest.mark.unit
def test_search_end_to_end_fusion(indexed_hybrid_retriever):
    # Query targets RRF formula; doc_001 should be Rank 1
    results = indexed_hybrid_retriever.search(
        "What is the mathematical formula for calculating Reciprocal Rank Fusion (RRF) scores?",
        top_k=3,
    )
    assert len(results) == 3
    assert results[0].document.doc_id == "doc_001"
    assert results[0].rank == 1
    assert results[0].score > 0.0

    # Ensure strictly descending score ordering and sequential ranking
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
        assert results[i].rank == i + 1


@pytest.mark.unit
def test_save_unindexed_error(tmp_path):
    retriever = HybridRetriever()
    with pytest.raises(RuntimeError) as exc:
        retriever.save(tmp_path / "hybrid_test")
    assert "Cannot save an unindexed" in str(exc.value)


@pytest.mark.unit
def test_save_and_load_roundtrip(indexed_hybrid_retriever, tmp_path, shared_model):
    save_dir = tmp_path / "saved_hybrid"
    indexed_hybrid_retriever.save(save_dir)

    assert (save_dir / METADATA_FILENAME).is_file()
    assert (save_dir / "sparse" / "retriever_metadata.json").is_file()
    assert (save_dir / "dense" / "faiss.index").is_file()

    loaded_retriever = HybridRetriever.load(save_dir, model=shared_model)
    assert loaded_retriever.is_indexed
    assert loaded_retriever.document_count == indexed_hybrid_retriever.document_count
    assert loaded_retriever.rrf_k == indexed_hybrid_retriever.rrf_k

    query = "combining sparse keyword and dense semantic rankings"
    orig_results = indexed_hybrid_retriever.search(query, top_k=3)
    loaded_results = loaded_retriever.search(query, top_k=3)

    assert len(orig_results) == len(loaded_results)
    for orig, loaded in zip(orig_results, loaded_results):
        assert orig.document.doc_id == loaded.document.doc_id
        assert orig.rank == loaded.rank
        assert pytest.approx(orig.score, abs=1e-6) == loaded.score


@pytest.mark.unit
def test_load_nonexistent_directory():
    with pytest.raises(FileNotFoundError):
        HybridRetriever.load("nonexistent_hybrid_path_123")


@pytest.mark.unit
def test_load_incompatible_schema_version(indexed_hybrid_retriever, tmp_path, shared_model):
    save_dir = tmp_path / "incompatible_hybrid"
    indexed_hybrid_retriever.save(save_dir)

    meta_file = save_dir / METADATA_FILENAME
    content = meta_file.read_text(encoding="utf-8")
    content = content.replace('"1.0.0"', '"99.0.0"')
    meta_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        HybridRetriever.load(save_dir, model=shared_model)
    assert "Incompatible index schema version" in str(exc.value)
