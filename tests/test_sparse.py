"""
Unit and integration tests for BM25Retriever in src/retrieval/sparse.py.
Verifies index construction, query retrieval, ranking properties,
zero-overlap policies, input validation, and persistence round-tripping.
"""

from pathlib import Path
import pytest
from src.core.dataset import load_corpus_from_jsonl
from src.core.schemas import Document, RetrievalCandidate
from src.retrieval.sparse import BM25Retriever, METADATA_FILENAME


FIXTURE_CORPUS_PATH = Path("data/fixtures/corpus.jsonl")


@pytest.fixture
def sample_corpus():
    return load_corpus_from_jsonl(FIXTURE_CORPUS_PATH)


@pytest.fixture
def indexed_retriever(sample_corpus):
    retriever = BM25Retriever(k1=1.5, b=0.75, method="lucene")
    retriever.index(sample_corpus)
    return retriever


@pytest.mark.unit
def test_retriever_initial_state():
    retriever = BM25Retriever()
    assert not retriever.is_indexed
    assert retriever.document_count == 0


@pytest.mark.unit
def test_index_construction(sample_corpus):
    retriever = BM25Retriever()
    retriever.index(sample_corpus)
    assert retriever.is_indexed
    assert retriever.document_count == 10


@pytest.mark.unit
def test_index_empty_corpus_rejection():
    retriever = BM25Retriever()
    with pytest.raises(ValueError) as exc:
        retriever.index([])
    assert "Cannot index an empty list" in str(exc.value)


@pytest.mark.unit
def test_index_duplicate_doc_id_rejection():
    retriever = BM25Retriever()
    docs = [
        Document(doc_id="d_01", text="First doc"),
        Document(doc_id="d_01", text="Duplicate ID doc"),
    ]
    with pytest.raises(ValueError) as exc:
        retriever.index(docs)
    assert "Duplicate document ID detected" in str(exc.value)


@pytest.mark.unit
def test_search_unindexed_error():
    retriever = BM25Retriever()
    with pytest.raises(RuntimeError) as exc:
        retriever.search("test query")
    assert "has not been indexed" in str(exc.value)


@pytest.mark.unit
def test_search_empty_query_error(indexed_retriever):
    with pytest.raises(ValueError):
        indexed_retriever.search("")
    with pytest.raises(ValueError):
        indexed_retriever.search("   \t  ")


@pytest.mark.unit
def test_search_invalid_top_k(indexed_retriever):
    with pytest.raises(ValueError):
        indexed_retriever.search("formula", top_k=0)
    with pytest.raises(ValueError):
        indexed_retriever.search("formula", top_k=-5)


@pytest.mark.unit
def test_search_exact_retrieval(indexed_retriever):
    # Query targets RRF formula; doc_001 should be Rank 1
    results = indexed_retriever.search(
        "What is the mathematical formula for calculating Reciprocal Rank Fusion (RRF) scores?",
        top_k=3,
    )
    assert len(results) > 0
    top_result = results[0]
    assert isinstance(top_result, RetrievalCandidate)
    assert top_result.document.doc_id == "doc_001"
    assert top_result.rank == 1
    assert top_result.score > 0.0
    assert "Reciprocal Rank Fusion" in top_result.document.text
    assert top_result.document.metadata.get("source") == "ir_textbook"


@pytest.mark.unit
def test_search_bm25_parameter_retrieval(indexed_retriever):
    # Query targets BM25 b parameter; doc_003 should be Rank 1
    results = indexed_retriever.search("b parameter document length normalization BM25", top_k=2)
    assert len(results) > 0
    assert results[0].document.doc_id == "doc_003"


@pytest.mark.unit
def test_search_descending_scores(indexed_retriever):
    results = indexed_retriever.search("retrieval ranking search", top_k=5)
    assert len(results) >= 2
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
        assert results[i].rank == i + 1


@pytest.mark.unit
def test_search_top_k_bounds(indexed_retriever):
    results_2 = indexed_retriever.search("retrieval", top_k=2)
    assert len(results_2) <= 2

    # Requesting top_k > corpus size returns at most total documents
    results_100 = indexed_retriever.search("retrieval", top_k=100)
    assert len(results_100) <= 10


@pytest.mark.unit
def test_zero_vocabulary_overlap_returns_empty(indexed_retriever):
    # Terms completely absent from the corpus
    results = indexed_retriever.search("xylophone zymurgy quokka", top_k=5)
    assert results == []


@pytest.mark.unit
def test_save_unindexed_error(tmp_path):
    retriever = BM25Retriever()
    with pytest.raises(RuntimeError) as exc:
        retriever.save(tmp_path / "bm25_test")
    assert "Cannot save an unindexed" in str(exc.value)


@pytest.mark.unit
def test_save_and_load_roundtrip(indexed_retriever, tmp_path):
    save_dir = tmp_path / "saved_bm25"
    indexed_retriever.save(save_dir)

    # Verify expected files exist
    assert (save_dir / METADATA_FILENAME).is_file()
    assert (save_dir / "data.csc.index.npy").is_file()
    assert (save_dir / "vocab.index.json").is_file()

    # Load into a new retriever
    loaded_retriever = BM25Retriever.load(save_dir)
    assert loaded_retriever.is_indexed
    assert loaded_retriever.document_count == indexed_retriever.document_count

    # Compare search results on original vs reloaded
    query = "Reciprocal Rank Fusion hybrid retrieval"
    orig_results = indexed_retriever.search(query, top_k=3)
    loaded_results = loaded_retriever.search(query, top_k=3)

    assert len(orig_results) == len(loaded_results)
    for orig, loaded in zip(orig_results, loaded_results):
        assert orig.document.doc_id == loaded.document.doc_id
        assert orig.rank == loaded.rank
        assert pytest.approx(orig.score, rel=1e-5) == loaded.score


@pytest.mark.unit
def test_load_nonexistent_directory():
    with pytest.raises(FileNotFoundError):
        BM25Retriever.load("nonexistent_directory_12345")


@pytest.mark.unit
def test_load_missing_metadata(tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError) as exc:
        BM25Retriever.load(empty_dir)
    assert "Missing required retriever metadata" in str(exc.value)


@pytest.mark.unit
def test_load_incompatible_schema_version(indexed_retriever, tmp_path):
    save_dir = tmp_path / "incompatible_bm25"
    indexed_retriever.save(save_dir)

    meta_file = save_dir / METADATA_FILENAME
    meta_content = meta_file.read_text(encoding="utf-8")
    meta_content = meta_content.replace('"1.0.0"', '"99.0.0"')
    meta_file.write_text(meta_content, encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        BM25Retriever.load(save_dir)
    assert "Incompatible index schema version" in str(exc.value)
