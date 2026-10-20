"""
BM25 sparse retrieval implementation using bm25s.
Adheres to BaseSparseRetriever interface and handles index persistence,
tokenization, scoring parameters, and zero-overlap query policies.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import bm25s

from src.core.schemas import Document, RetrievalCandidate
from src.retrieval.base import BaseSparseRetriever


METADATA_FILENAME = "retriever_metadata.json"
CURRENT_SCHEMA_VERSION = "1.0.0"


class BM25Retriever(BaseSparseRetriever):
    """
    High-performance BM25 sparse retriever powered by bm25s.
    
    Supports:
    - Lucene-compatible probabilistic BM25 scoring (default: method='lucene').
    - Configurable k1 (term saturation) and b (length normalization) parameters.
    - Full persistence of index structures, vocabulary, and document metadata.
    - Zero-vocabulary-overlap policy (returns empty list [] when scores are non-positive).
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        method: str = "lucene",
        stopwords: Union[str, List[str]] = "english",
        lower: bool = True,
        token_pattern: str = r"(?u)\b\w\w+\b",
        stemmer: Optional[Any] = None,
    ):
        if k1 < 0:
            raise ValueError(f"k1 parameter must be non-negative, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b parameter must be between 0.0 and 1.0, got {b}")

        self.k1 = k1
        self.b = b
        self.method = method
        self.stopwords = stopwords
        self.lower = lower
        self.token_pattern = token_pattern
        self.stemmer = stemmer

        self._model: Optional[bm25s.BM25] = None
        self._doc_ids: List[str] = []
        self._documents: Dict[str, Document] = {}
        self._is_indexed: bool = False

    @property
    def is_indexed(self) -> bool:
        """Returns True if the retriever has an active, searchable index."""
        return self._is_indexed

    @property
    def document_count(self) -> int:
        """Returns the number of documents in the index."""
        return len(self._doc_ids)

    def index(self, documents: List[Document]) -> None:
        """
        Builds the BM25 index from a list of Document objects.
        
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

        texts = [doc.text for doc in documents]
        corpus_tokens = bm25s.tokenize(
            texts,
            stopwords=self.stopwords,
            lower=self.lower,
            token_pattern=self.token_pattern,
            stemmer=self.stemmer,
            show_progress=False,
        )

        model = bm25s.BM25(k1=self.k1, b=self.b, method=self.method)
        model.index(corpus_tokens, show_progress=False)

        self._model = model
        self._doc_ids = [doc.doc_id for doc in documents]
        self._documents = {doc.doc_id: doc for doc in documents}
        self._is_indexed = True

    def search(self, query: str, top_k: int = 5) -> List[RetrievalCandidate]:
        """
        Retrieves top_k candidates for the query.
        
        Args:
            query: Non-empty search query string.
            top_k: Positive integer specifying max candidates.
            
        Returns:
            List of RetrievalCandidate objects sorted in descending score order.
            If the query has no vocabulary overlap with the corpus, returns [].
            
        Raises:
            ValueError: If query is empty/whitespace or top_k <= 0.
            RuntimeError: If called on an unindexed retriever.
        """
        if not self._is_indexed or self._model is None:
            raise RuntimeError("BM25Retriever has not been indexed. Call index() or load() before search().")

        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        query_tokens = bm25s.tokenize(
            [query],
            stopwords=self.stopwords,
            lower=self.lower,
            token_pattern=self.token_pattern,
            stemmer=self.stemmer,
            show_progress=False,
        )

        effective_k = min(top_k, len(self._doc_ids))
        docs, scores = self._model.retrieve(
            query_tokens,
            k=effective_k,
            sorted=True,
            show_progress=False,
        )

        doc_indices = docs[0]
        doc_scores = scores[0]

        candidates: List[RetrievalCandidate] = []
        rank = 1
        for idx, score in zip(doc_indices, doc_scores):
            score_float = float(score)
            # Filter zero-overlap / non-positive lexical matches
            if score_float <= 0.0:
                continue
            doc_id = self._doc_ids[int(idx)]
            candidate_doc = self._documents[doc_id]
            candidates.append(
                RetrievalCandidate(
                    document=candidate_doc,
                    score=score_float,
                    rank=rank,
                )
            )
            rank += 1

        return candidates

    def save(self, save_dir: Union[str, Path]) -> None:
        """
        Persists BM25 index artifacts and document mapping to the target directory.
        
        Args:
            save_dir: Destination directory path.
            
        Raises:
            RuntimeError: If index has not been built.
        """
        if not self._is_indexed or self._model is None:
            raise RuntimeError("Cannot save an unindexed BM25Retriever.")

        dir_path = Path(save_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save bm25s underlying array matrices and vocabulary
        self._model.save(dir_path, corpus=None, show_progress=False)

        # Save companion metadata with schema version and full document mappings
        metadata = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "backend": "bm25s",
            "method": self.method,
            "k1": self.k1,
            "b": self.b,
            "stopwords": self.stopwords,
            "lower": self.lower,
            "token_pattern": self.token_pattern,
            "doc_ids": self._doc_ids,
            "documents": {doc_id: doc.model_dump() for doc_id, doc in self._documents.items()},
        }
        metadata_path = dir_path / METADATA_FILENAME
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, save_dir: Union[str, Path]) -> "BM25Retriever":
        """
        Loads a saved BM25 index and restores full retriever state.
        
        Args:
            save_dir: Path to directory containing saved index artifacts.
            
        Returns:
            Instantiated, ready-to-search BM25Retriever.
            
        Raises:
            FileNotFoundError: If save_dir or metadata file is missing.
            ValueError: If metadata is invalid or schema version is incompatible.
        """
        dir_path = Path(save_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Index directory does not exist: {dir_path}")

        metadata_path = dir_path / METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing required retriever metadata file: {metadata_path}")

        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted metadata file in {dir_path}: {e}")

        version = metadata.get("schema_version")
        if version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible index schema version: expected '{CURRENT_SCHEMA_VERSION}', got '{version}'"
            )

        retriever = cls(
            k1=metadata["k1"],
            b=metadata["b"],
            method=metadata.get("method", "lucene"),
            stopwords=metadata.get("stopwords", "english"),
            lower=metadata.get("lower", True),
            token_pattern=metadata.get("token_pattern", r"(?u)\b\w\w+\b"),
        )

        try:
            retriever._model = bm25s.BM25.load(dir_path, load_corpus=False, show_progress=False)
        except Exception as e:
            raise ValueError(f"Failed to load underlying BM25 index artifacts from {dir_path}: {e}")

        retriever._doc_ids = metadata["doc_ids"]
        retriever._documents = {
            doc_id: Document.model_validate(doc_dict)
            for doc_id, doc_dict in metadata["documents"].items()
        }
        retriever._is_indexed = True

        return retriever
