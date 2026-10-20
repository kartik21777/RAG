"""
Unified Retrieval Facade for Semantic- and Evidence-Aware CRAG.
Exposes high-level retrieval methods yielding canonical RetrievalResult contracts
with complete multi-engine provenance for Member 3's orchestration pipeline.
"""

from pathlib import Path
from typing import List, Optional, Union

from src.core.schemas import Document, Query, RetrievalResult
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import BM25Retriever


class CRAGRetrieverFacade:
    """
    High-level facade orchestrating hybrid sparse-dense retrieval.
    Consolidates BM25, Sentence Transformers, FAISS, and Reciprocal Rank Fusion
    into a simple, production-ready interface emitting validated RetrievalResult contracts.
    """

    def __init__(self, hybrid_retriever: Optional[HybridRetriever] = None):
        """
        Initializes the facade with an underlying HybridRetriever instance.
        """
        self.hybrid = hybrid_retriever if hybrid_retriever is not None else HybridRetriever()

    @property
    def is_indexed(self) -> bool:
        """Returns True if both sparse and dense underlying indexes are ready for search."""
        return self.hybrid.is_indexed

    def index_corpus(self, documents: List[Document]) -> None:
        """
        Indexes a list of canonical Document instances across both sparse and dense backends.

        Args:
            documents: Non-empty list of Document instances.
        """
        self.hybrid.index(documents)

    def retrieve(self, query: Union[str, Query], top_k: int = 5) -> List[RetrievalResult]:
        """
        Executes hybrid retrieval and emits canonical RetrievalResult objects
        enriched with multi-engine provenance (fused score/rank, sparse score/rank, dense score/rank).

        Args:
            query: Query string or canonical Query object.
            top_k: Number of top-ranked results to return.

        Returns:
            List of RetrievalResult instances ordered by fused RRF ranking.
        """
        if isinstance(query, Query):
            query_str = query.query_text
        elif isinstance(query, str):
            query_str = query
        else:
            raise TypeError(f"query must be str or Query, got {type(query).__name__}")

        if not query_str or not query_str.strip():
            raise ValueError("Query cannot be empty or whitespace only.")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if not self.is_indexed:
            raise RuntimeError("Retriever facade is not indexed. Call index_corpus() or load() before retrieve().")

        # Query underlying backends
        sparse_cands = self.hybrid.sparse_retriever.search(query_str, top_k=self.hybrid.sparse_top_k)
        dense_cands = self.hybrid.dense_retriever.search(query_str, top_k=self.hybrid.dense_top_k)

        sparse_map = {c.document.doc_id: c for c in sparse_cands}
        dense_map = {c.document.doc_id: c for c in dense_cands}

        # Fuse candidates via RRF
        fused = self.hybrid.fuse(sparse_cands, dense_cands, top_k=top_k)

        results: List[RetrievalResult] = []
        for fused_cand in fused:
            doc_id = fused_cand.document.doc_id
            sparse_cand = sparse_map.get(doc_id)
            dense_cand = dense_map.get(doc_id)

            results.append(
                RetrievalResult(
                    document=fused_cand.document,
                    fused_score=fused_cand.score,
                    fused_rank=fused_cand.rank,
                    sparse_rank=sparse_cand.rank if sparse_cand else None,
                    dense_rank=dense_cand.rank if dense_cand else None,
                    sparse_score=sparse_cand.score if sparse_cand else None,
                    dense_score=dense_cand.score if dense_cand else None,
                )
            )

        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Persists underlying sparse and dense indexes and hybrid metadata to disk."""
        self.hybrid.save(directory)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "CRAGRetrieverFacade":
        """Loads and restores a complete, ready-to-query retriever facade from disk."""
        hybrid = HybridRetriever.load(directory)
        return cls(hybrid_retriever=hybrid)


def create_crag_retriever(
    corpus: Optional[List[Document]] = None,
    index_dir: Optional[Union[str, Path]] = None,
    dense_model_name: str = "all-MiniLM-L6-v2",
    rrf_k: int = 60,
) -> CRAGRetrieverFacade:
    """
    Factory function for instantiating and preparing a CRAGRetrieverFacade.

    Args:
        corpus: Optional list of documents to index immediately.
        index_dir: Optional directory to load existing pre-built indexes from.
        dense_model_name: SentenceTransformers model identifier.
        rrf_k: Reciprocal Rank Fusion smoothing constant.

    Returns:
        Configured and initialized CRAGRetrieverFacade.
    """
    if index_dir is not None and Path(index_dir).exists() and (Path(index_dir) / "hybrid_metadata.json").exists():
        return CRAGRetrieverFacade.load(index_dir)

    dense_retriever = DenseRetriever(model_name_or_path=dense_model_name)
    sparse_retriever = BM25Retriever()
    hybrid = HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        rrf_k=rrf_k,
    )
    facade = CRAGRetrieverFacade(hybrid_retriever=hybrid)

    if corpus is not None:
        facade.index_corpus(corpus)

    return facade
