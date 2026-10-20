"""
Verification tests for Stage 1 environment and core dependencies.
Ensures that all mandatory libraries import and run minimal operations.
"""

import pytest
import numpy as np


@pytest.mark.unit
def test_pydantic_installed():
    from pydantic import BaseModel, Field

    class SampleModel(BaseModel):
        id: str
        val: int = Field(gt=0)

    obj = SampleModel(id="test", val=42)
    assert obj.id == "test"
    assert obj.val == 42


@pytest.mark.unit
def test_bm25s_minimal_indexing():
    import bm25s

    corpus = [
        "Corrective RAG retrieves and evaluates evidence.",
        "BM25 is a term-frequency based sparse retriever.",
        "Dense vectors capture semantic intent through embeddings."
    ]
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    query_tokens = bm25s.tokenize(["sparse retriever"])
    docs, scores = retriever.retrieve(query_tokens, k=1)
    assert docs.shape == (1, 1)
    assert docs[0, 0] == 1  # Matches second document index


@pytest.mark.unit
def test_faiss_cpu_minimal_index():
    import faiss

    d = 64  # dimension
    nb = 10 # database size
    np.random.seed(42)
    xb = np.random.random((nb, d)).astype('float32')

    index = faiss.IndexFlatIP(d)
    index.add(xb)
    assert index.ntotal == nb

    xq = np.random.random((1, d)).astype('float32')
    D, I = index.search(xq, k=3)
    assert I.shape == (1, 3)
    assert D.shape == (1, 3)


@pytest.mark.unit
def test_torch_runtime():
    import torch

    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([4.0, 5.0, 6.0])
    dot = torch.dot(t1, t2)
    assert dot.item() == 32.0


@pytest.mark.unit
def test_sentence_transformers_import():
    import sentence_transformers
    assert hasattr(sentence_transformers, "__version__")
