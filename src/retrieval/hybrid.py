"""
Hybrid retrieval implementation using Reciprocal Rank Fusion (RRF).
Combines sparse lexical retrieval (BM25) and dense semantic retrieval (Sentence Transformers + FAISS)
to produce a robust, unified ranking without requiring cross-system score normalization.
"""

import json
from pathlib import Path
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer

from src.core.schemas import Document, RetrievalCandidate
from src.retrieval.base import BaseDenseRetriever, BaseHybridRetriever, BaseSparseRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import BM25Retriever


METADATA_FILENAME = "hybrid_metadata.json"
CURRENT_SCHEMA_VERSION = "1.0.0"
DEFAULT_RRF_K = 60


class HybridRetriever(BaseHybridRetriever):
    """
    Hybrid retriever fusing sparse and dense ranked candidates via Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{m in M} w_m / (k + rank_m(d))

    Where:
        - M is the set of retrievers (sparse and dense)
        - rank_m(d) is the 1-indexed rank position of document d in retriever m
        - k is the rank smoothing constant (default: 60)
        - w_m is the weight allocated to retriever m (default: 1.0 each)
    """

    def __init__(
        self,
        sparse_retriever: Optional[BaseSparseRetriever] = None,
        dense_retriever: Optional[BaseDenseRetriever] = None,
        rrf_k: int = DEFAULT_RRF_K,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        sparse_top_k: int = 20,
        dense_top_k: int = 20,
    ):
        if rrf_k <= 0:
            raise ValueError(f"rrf_k smoothing constant must be a positive integer, got {rrf_k}")
        if sparse_weight < 0:
            raise ValueError(f"sparse_weight must be non-negative, got {sparse_weight}")
        if dense_weight < 0:
            raise ValueError(f"dense_weight must be non-negative, got {dense_weight}")
        if sparse_weight == 0 and dense_weight == 0:
            raise ValueError("At least one of sparse_weight or dense_weight must be strictly positive.")
        if sparse_top_k <= 0:
            raise ValueError(f"sparse_top_k must be a positive integer, got {sparse_top_k}")
        if dense_top_k <= 0:
            raise ValueError(f"dense_top_k must be a positive integer, got {dense_top_k}")

        self.sparse_retriever = sparse_retriever if sparse_retriever is not None else BM25Retriever()
        self.dense_retriever = dense_retriever if dense_retriever is not None else DenseRetriever()
        self.rrf_k = rrf_k
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.sparse_top_k = sparse_top_k
        self.dense_top_k = dense_top_k

    @property
    def is_indexed(self) -> bool:
        """Returns True only when both underlying sparse and dense retrievers are indexed."""
        return getattr(self.sparse_retriever, "is_indexed", False) and getattr(
            self.dense_retriever, "is_indexed", False
        )

    @property
    def document_count(self) -> int:
        """Returns the indexed document count, validating parity across both backends."""
        sparse_count = getattr(self.sparse_retriever, "document_count", 0)
        dense_count = getattr(self.dense_retriever, "document_count", 0)
        return min(sparse_count, dense_count)

    def index(self, documents: List[Document]) -> None:
        """
        Indexes the corpus across both sparse and dense backends.

        Args:
            documents: Non-empty list of Document instances with unique doc_ids.

        Raises:
            ValueError: If documents list is empty or contains duplicate doc_ids.
        """
        if not documents:
            raise ValueError("Cannot index an empty list of documents.")

        seen_ids = set()
        for doc in documents:
            if doc.doc_id in seen_ids:
                raise ValueError(f"Duplicate document ID detected during indexing: '{doc.doc_id}'")
            seen_ids.add(doc.doc_id)

        self.sparse_retriever.index(documents)
        self.dense_retriever.index(documents)

    def fuse(
        self,
        sparse_candidates: List[RetrievalCandidate],
        dense_candidates: List[RetrievalCandidate],
        top_k: int = 5,
    ) -> List[RetrievalCandidate]:
        """
        Aggregates ordered candidate lists from sparse and dense retrieval using RRF.

        Args:
            sparse_candidates: Ordered candidates from sparse retrieval.
            dense_candidates: Ordered candidates from dense retrieval.
            top_k: Maximum number of fused candidates to return.

        Returns:
            Sorted list of RetrievalCandidate objects with RRF scores and updated ranks.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        rrf_scores = {}
        doc_store = {}
        sparse_ranks = {}
        dense_ranks = {}

        for cand in sparse_candidates:
            doc_id = cand.document.doc_id
            doc_store[doc_id] = cand.document
            sparse_ranks[doc_id] = cand.rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                self.sparse_weight / (self.rrf_k + cand.rank)
            )

        for cand in dense_candidates:
            doc_id = cand.document.doc_id
            doc_store[doc_id] = cand.document
            dense_ranks[doc_id] = cand.rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                self.dense_weight / (self.rrf_k + cand.rank)
            )

        if not rrf_scores:
            return []

        # Sort primarily by descending RRF score, with deterministic tie-breaking
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda did: (
                -rrf_scores[did],
                sparse_ranks.get(did, 999999),
                dense_ranks.get(did, 999999),
                did,
            ),
        )

        fused_candidates: List[RetrievalCandidate] = []
        for rank_idx, doc_id in enumerate(sorted_doc_ids[:top_k], start=1):
            doc = doc_store[doc_id]
            fused_candidates.append(
                RetrievalCandidate(
                    document=doc,
                    score=float(rrf_scores[doc_id]),
                    rank=rank_idx,
                )
            )

        return fused_candidates

    def search(self, query: str, top_k: int = 5) -> List[RetrievalCandidate]:
        """
        Executes hybrid retrieval: retrieves candidates from both engines and merges them via RRF.

        Args:
            query: Non-empty search query string.
            top_k: Maximum number of candidates to retrieve.

        Returns:
            List of fused RetrievalCandidate objects sorted in descending RRF score order.

        Raises:
            RuntimeError: If either backend is unindexed.
            ValueError: If query is empty/whitespace or top_k <= 0.
        """
        if not self.is_indexed:
            raise RuntimeError(
                "HybridRetriever backends are not indexed. Call index() or load() before search()."
            )

        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        sparse_candidates = self.sparse_retriever.search(query, top_k=self.sparse_top_k)
        dense_candidates = self.dense_retriever.search(query, top_k=self.dense_top_k)

        return self.fuse(sparse_candidates, dense_candidates, top_k=top_k)

    def save(self, save_dir: Union[str, Path]) -> None:
        """
        Persists both sparse and dense indexes along with hybrid fusion metadata.

        Args:
            save_dir: Destination directory.

        Raises:
            RuntimeError: If retriever is unindexed.
        """
        if not self.is_indexed:
            raise RuntimeError("Cannot save an unindexed HybridRetriever.")

        dir_path = Path(save_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        sparse_dir = dir_path / "sparse"
        dense_dir = dir_path / "dense"

        self.sparse_retriever.save(sparse_dir)
        self.dense_retriever.save(dense_dir)

        metadata = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "backend": "rrf_hybrid",
            "rrf_k": self.rrf_k,
            "sparse_weight": self.sparse_weight,
            "dense_weight": self.dense_weight,
            "sparse_top_k": self.sparse_top_k,
            "dense_top_k": self.dense_top_k,
        }
        metadata_file = dir_path / METADATA_FILENAME
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(
        cls,
        save_dir: Union[str, Path],
        model: Optional[SentenceTransformer] = None,
        device: Optional[str] = None,
    ) -> "HybridRetriever":
        """
        Restores a persisted HybridRetriever from disk.

        Args:
            save_dir: Directory containing saved hybrid artifacts.
            model: Optional pre-loaded SentenceTransformer to avoid re-instantiation.
            device: Optional device identifier ('cpu', 'cuda').

        Returns:
            Instantiated, ready-to-search HybridRetriever.

        Raises:
            FileNotFoundError: If save_dir or metadata file is missing.
            ValueError: If metadata is corrupted or schema version is incompatible.
        """
        dir_path = Path(save_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Hybrid index directory does not exist: {dir_path}")

        metadata_file = dir_path / METADATA_FILENAME
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Missing required hybrid metadata file: {metadata_file}")

        try:
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted metadata file in {dir_path}: {e}")

        version = metadata.get("schema_version")
        if version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible index schema version: expected '{CURRENT_SCHEMA_VERSION}', got '{version}'"
            )

        sparse_dir = dir_path / "sparse"
        dense_dir = dir_path / "dense"

        sparse_retriever = BM25Retriever.load(sparse_dir)
        dense_retriever = DenseRetriever.load(dense_dir, model=model, device=device)

        return cls(
            sparse_retriever=sparse_retriever,
            dense_retriever=dense_retriever,
            rrf_k=metadata.get("rrf_k", DEFAULT_RRF_K),
            sparse_weight=metadata.get("sparse_weight", 1.0),
            dense_weight=metadata.get("dense_weight", 1.0),
            sparse_top_k=metadata.get("sparse_top_k", 20),
            dense_top_k=metadata.get("dense_top_k", 20),
        )
