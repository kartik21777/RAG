"""
Abstract base interfaces for sparse and dense retrieval backends.
Ensures uniform contracts across all retrieval engines (BM25, Sentence Transformers / FAISS)
and isolates downstream consumers (HybridRetriever, Evaluator) from internal representations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Union
from src.core.schemas import Document, RetrievalCandidate


class BaseSparseRetriever(ABC):
    """
    Abstract contract for sparse lexical retrievers in the CRAG project.
    
    Subclasses must implement:
    - index: construct vocabulary and inverted index from Document records.
    - search: query the index, returning standardized RetrievalCandidate objects.
    - save: persist index structures, parameters, and document records to disk.
    - load: restore a complete, queryable retriever instance from disk.
    """

    @abstractmethod
    def index(self, documents: List[Document]) -> None:
        """
        Constructs the sparse inverted index from a collection of Document objects.

        Args:
            documents: Non-empty list of Document instances with unique, stable doc_ids.

        Raises:
            ValueError: If documents is empty or contains duplicate doc_ids.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[RetrievalCandidate]:
        """
        Executes sparse lexical retrieval for a given query string.

        Args:
            query: Non-empty, non-whitespace query string.
            top_k: Positive integer specifying the maximum number of candidates to retrieve.

        Returns:
            List of RetrievalCandidate objects sorted in descending order of score.
            If the query exhibits zero lexical overlap with the corpus, returns an empty list [].

        Raises:
            ValueError: If query is empty/whitespace or top_k is non-positive.
            RuntimeError: If called before an index has been built or loaded.
        """
        pass

    @abstractmethod
    def save(self, save_dir: Union[str, Path]) -> None:
        """
        Persists index artifacts, scoring hyperparameters, and document records to disk.

        Args:
            save_dir: Directory path where index artifacts will be saved.

        Raises:
            RuntimeError: If attempting to save before indexing has occurred.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, save_dir: Union[str, Path]) -> "BaseSparseRetriever":
        """
        Restores a persisted retriever instance from disk artifacts.

        Args:
            save_dir: Path to directory containing persisted index artifacts.

        Returns:
            A ready-to-search BaseSparseRetriever instance.

        Raises:
            FileNotFoundError: If the directory or required artifacts are missing.
            ValueError: If index files or metadata are corrupted or incompatible.
        """
        pass


class BaseDenseRetriever(ABC):
    """
    Abstract contract for dense semantic retrievers in the CRAG project.

    Subclasses must implement:
    - index: encode documents and construct the dense vector index.
    - search: encode query string and retrieve nearest neighbors as RetrievalCandidate objects.
    - save: persist vector index, embeddings metadata, and document records to disk.
    - load: restore a complete, queryable dense retriever instance from disk.
    - encode: transform texts into dense embedding vectors.
    """

    @abstractmethod
    def index(self, documents: List[Document]) -> None:
        """
        Encodes Document texts and builds the dense vector index.

        Args:
            documents: Non-empty list of Document instances with unique, stable doc_ids.

        Raises:
            ValueError: If documents is empty or contains duplicate doc_ids.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[RetrievalCandidate]:
        """
        Executes dense semantic retrieval for a given query string.

        Args:
            query: Non-empty, non-whitespace query string.
            top_k: Positive integer specifying the maximum number of candidates to retrieve.

        Returns:
            List of RetrievalCandidate objects sorted in descending order of similarity score.

        Raises:
            ValueError: If query is empty/whitespace or top_k is non-positive.
            RuntimeError: If called before an index has been built or loaded.
        """
        pass

    @abstractmethod
    def save(self, save_dir: Union[str, Path]) -> None:
        """
        Persists vector index artifacts, model metadata, and document records to disk.

        Args:
            save_dir: Directory path where index artifacts will be saved.

        Raises:
            RuntimeError: If attempting to save before indexing has occurred.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, save_dir: Union[str, Path]) -> "BaseDenseRetriever":
        """
        Restores a persisted dense retriever instance from disk artifacts.

        Args:
            save_dir: Path to directory containing persisted index artifacts.

        Returns:
            A ready-to-search BaseDenseRetriever instance.

        Raises:
            FileNotFoundError: If the directory or required artifacts are missing.
            ValueError: If index files or metadata are corrupted or incompatible.
        """
        pass

    @abstractmethod
    def encode(self, texts: List[str]) -> Any:
        """
        Encodes a list of text strings into dense embedding vectors.

        Args:
            texts: List of strings to encode.

        Returns:
            Numpy array of shape (len(texts), embedding_dim) containing dense embeddings.
        """
        pass


class BaseHybridRetriever(ABC):
    """
    Abstract contract for hybrid retrieval backends fusing sparse and dense candidates.

    Subclasses must implement:
    - index: index documents across underlying sparse and dense backends.
    - search: retrieve candidates from sparse and dense engines and fuse rankings.
    - fuse: rank aggregation logic (e.g. Reciprocal Rank Fusion) combining candidate lists.
    - save: persist underlying sparse/dense indexes and hybrid hyperparameters.
    - load: restore a complete, queryable hybrid retriever instance from disk.
    """

    @abstractmethod
    def index(self, documents: List[Document]) -> None:
        """
        Indexes documents across both underlying sparse and dense backends.

        Args:
            documents: Non-empty list of Document instances with unique, stable doc_ids.

        Raises:
            ValueError: If documents is empty or contains duplicate doc_ids.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[RetrievalCandidate]:
        """
        Executes hybrid retrieval by querying sparse and dense engines and fusing results.

        Args:
            query: Non-empty, non-whitespace query string.
            top_k: Positive integer specifying the maximum number of candidates to retrieve.

        Returns:
            List of RetrievalCandidate objects sorted in descending order of fused score.

        Raises:
            ValueError: If query is empty/whitespace or top_k is non-positive.
            RuntimeError: If called before indexes have been built or loaded.
        """
        pass

    @abstractmethod
    def fuse(
        self,
        sparse_candidates: List[RetrievalCandidate],
        dense_candidates: List[RetrievalCandidate],
        top_k: int = 5,
    ) -> List[RetrievalCandidate]:
        """
        Fuses ranked candidate lists from sparse and dense retrieval systems.

        Args:
            sparse_candidates: Ordered candidates from sparse retrieval.
            dense_candidates: Ordered candidates from dense retrieval.
            top_k: Maximum number of fused candidates to return.

        Returns:
            List of fused RetrievalCandidate objects with recalculated scores and ranks.
        """
        pass

    @abstractmethod
    def save(self, save_dir: Union[str, Path]) -> None:
        """
        Persists underlying indexes and hybrid configuration to disk.

        Args:
            save_dir: Directory path where hybrid artifacts will be saved.

        Raises:
            RuntimeError: If attempting to save before indexing has occurred.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, save_dir: Union[str, Path]) -> "BaseHybridRetriever":
        """
        Restores a persisted hybrid retriever instance from disk artifacts.

        Args:
            save_dir: Path to directory containing persisted hybrid index artifacts.

        Returns:
            A ready-to-search BaseHybridRetriever instance.

        Raises:
            FileNotFoundError: If the directory or required artifacts are missing.
            ValueError: If index files or metadata are corrupted or incompatible.
        """
        pass
