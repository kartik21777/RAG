"""
Abstract base interface for sparse retrieval backends.
Ensures uniform contract across all sparse engines (BM25s, Pyserini)
and isolates consumers like HybridRetriever from internal index representations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
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
