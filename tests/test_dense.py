"""
Unit and integration tests for DenseRetriever in src/retrieval/dense.py.
Verifies embedding generation, FAISS index construction, semantic search,
score properties, boundary conditions, error handling, and persistence round-tripping.
"""

from pathlib import Path
import numpy as np
import pytest
from src.core.dataset import load_corpus_from_jsonl
from src.core.schemas import Document, RetrievalCandidate
from src.retrieval.dense import DenseRetriever, INDEX_FILENAME, METADATA_FILENAME


FIXTURE_CORPUS_PATH = Path("data/fixtures/corpus.jsonl")


@pytest.fixture(scope="module")
def shared_model():
    """Module-scoped SentenceTransformer instance to avoid re-loading weights for each test."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def sample_corpus():
    return load_corpus_from_jsonl(FIXTURE_CORPUS_PATH)


@pytest.fixture
def indexed_retriever(sample_corpus, shared_model):
    retriever = DenseRetriever(model=shared_model)
    retriever.index(sample_corpus)
    return retriever


@pytest.mark.unit
def test_retriever_initial_state(shared_model):
    retriever = DenseRetriever(model=shared_model)
    assert not retriever.is_indexed
    assert retriever.document_count == 0
    assert retriever.dimension == 384
    assert retriever.model_name_or_path == "all-MiniLM-L6-v2"


@pytest.mark.unit
def test_encode_dimensions_and_normalization(shared_model):
    retriever = DenseRetriever(model=shared_model, normalize_embeddings=True)

    empty_emb = retriever.encode([])
    assert empty_emb.shape == (0, 384)
    assert empty_emb.dtype == np.float32

    texts = ["First test sentence.", "Second test sentence about machine learning."]
    embeddings = retriever.encode(texts)
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32

    # L2 norm of normalized embeddings must equal 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    for norm in norms:
        assert pytest.approx(norm, rel=1e-5) == 1.0


@pytest.mark.unit
def test_index_construction(sample_corpus, shared_model):
    retriever = DenseRetriever(model=shared_model)
    retriever.index(sample_corpus)
    assert retriever.is_indexed
    assert retriever.document_count == 10


@pytest.mark.unit
def test_index_empty_corpus_rejection(shared_model):
    retriever = DenseRetriever(model=shared_model)
    with pytest.raises(ValueError) as exc:
        retriever.index([])
    assert "Cannot index an empty list" in str(exc.value)


@pytest.mark.unit
def test_index_duplicate_doc_id_rejection(shared_model):
    retriever = DenseRetriever(model=shared_model)
    docs = [
        Document(doc_id="d_dup", text="First doc content"),
        Document(doc_id="d_dup", text="Duplicate ID content"),
    ]
    with pytest.raises(ValueError) as exc:
        retriever.index(docs)
    assert "Duplicate document ID detected" in str(exc.value)


@pytest.mark.unit
def test_search_unindexed_error(shared_model):
    retriever = DenseRetriever(model=shared_model)
    with pytest.raises(RuntimeError) as exc:
        retriever.search("quantum computing")
    assert "has not been indexed" in str(exc.value)


@pytest.mark.unit
def test_search_empty_query_error(indexed_retriever):
    with pytest.raises(ValueError):
        indexed_retriever.search("")
    with pytest.raises(ValueError):
        indexed_retriever.search("    \t  ")


@pytest.mark.unit
def test_search_invalid_top_k(indexed_retriever):
    with pytest.raises(ValueError):
        indexed_retriever.search("neural retrieval", top_k=0)
    with pytest.raises(ValueError):
        indexed_retriever.search("neural retrieval", top_k=-3)


@pytest.mark.unit
def test_search_exact_target_retrieval(indexed_retriever):
    results = indexed_retriever.search(
        "What is the mathematical formula for calculating Reciprocal Rank Fusion (RRF) scores?",
        top_k=3,
    )
    assert len(results) > 0
    top_result = results[0]
    assert isinstance(top_result, RetrievalCandidate)
    assert top_result.document.doc_id == "doc_001"
    assert top_result.rank == 1
    assert -1.0 <= top_result.score <= 1.0
    assert "Reciprocal Rank Fusion" in top_result.document.text


@pytest.mark.unit
def test_search_semantic_matching(indexed_retriever):
    # Query expresses hybrid retrieval concepts through paraphrasing
    results = indexed_retriever.search("combining keyword search with semantic neural search in a unified pipeline", top_k=2)
    assert len(results) > 0
    assert results[0].document.doc_id == "doc_002"
    assert "hybrid retrieval pipelines" in results[0].document.text.lower()


@pytest.mark.unit
def test_search_descending_scores(indexed_retriever):
    results = indexed_retriever.search("neural language model representation", top_k=5)
    assert len(results) >= 2
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
        assert results[i].rank == i + 1


@pytest.mark.unit
def test_search_top_k_bounds(indexed_retriever):
    results_2 = indexed_retriever.search("information retrieval", top_k=2)
    assert len(results_2) == 2

    # Requesting more candidates than available in index returns at most document_count
    results_50 = indexed_retriever.search("information retrieval", top_k=50)
    assert len(results_50) == 10


@pytest.mark.unit
def test_save_unindexed_error(tmp_path, shared_model):
    retriever = DenseRetriever(model=shared_model)
    with pytest.raises(RuntimeError) as exc:
        retriever.save(tmp_path / "dense_test")
    assert "Cannot save an unindexed" in str(exc.value)


@pytest.mark.unit
def test_save_and_load_roundtrip(indexed_retriever, tmp_path, shared_model):
    save_dir = tmp_path / "saved_dense"
    indexed_retriever.save(save_dir)

    assert (save_dir / METADATA_FILENAME).is_file()
    assert (save_dir / INDEX_FILENAME).is_file()

    # Restore retriever and reuse shared model to avoid re-instantiation overhead
    loaded_retriever = DenseRetriever.load(save_dir, model=shared_model)
    assert loaded_retriever.is_indexed
    assert loaded_retriever.document_count == indexed_retriever.document_count
    assert loaded_retriever.dimension == indexed_retriever.dimension

    query = "Reciprocal Rank Fusion hybrid retrieval"
    orig_results = indexed_retriever.search(query, top_k=3)
    loaded_results = loaded_retriever.search(query, top_k=3)

    assert len(orig_results) == len(loaded_results)
    for orig, loaded in zip(orig_results, loaded_results):
        assert orig.document.doc_id == loaded.document.doc_id
        assert orig.rank == loaded.rank
        assert pytest.approx(orig.score, abs=1e-6) == loaded.score


@pytest.mark.unit
def test_load_nonexistent_directory():
    with pytest.raises(FileNotFoundError):
        DenseRetriever.load("nonexistent_directory_9999")


@pytest.mark.unit
def test_load_missing_metadata(tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError) as exc:
        DenseRetriever.load(empty_dir)
    assert "Missing required retriever metadata" in str(exc.value)


@pytest.mark.unit
def test_load_missing_index_file(tmp_path, indexed_retriever):
    save_dir = tmp_path / "missing_index"
    indexed_retriever.save(save_dir)
    (save_dir / INDEX_FILENAME).unlink()

    with pytest.raises(FileNotFoundError) as exc:
        DenseRetriever.load(save_dir)
    assert "Missing required FAISS index file" in str(exc.value)


@pytest.mark.unit
def test_load_incompatible_schema_version(indexed_retriever, tmp_path, shared_model):
    save_dir = tmp_path / "incompatible_dense"
    indexed_retriever.save(save_dir)

    meta_file = save_dir / METADATA_FILENAME
    content = meta_file.read_text(encoding="utf-8")
    content = content.replace('"1.0.0"', '"99.0.0"')
    meta_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        DenseRetriever.load(save_dir, model=shared_model)
    assert "Incompatible index schema version" in str(exc.value)
