"""
Retrieval modules for Semantic- and Evidence-Aware CRAG.
Includes sparse (BM25), dense (Sentence Transformers + FAISS), and hybrid retrieval components.
"""

from src.retrieval.base import BaseDenseRetriever, BaseHybridRetriever, BaseSparseRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import BM25Retriever

__all__ = [
    "BaseSparseRetriever",
    "BaseDenseRetriever",
    "BaseHybridRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
]
