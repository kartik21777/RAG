"""
Dense semantic retrieval implementation using Sentence Transformers and FAISS IndexFlatIP.
Adheres to BaseDenseRetriever interface with L2-normalized embeddings, exact inner product
(cosine similarity) search, full index persistence, and standardized candidate ranking.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.schemas import Document, RetrievalCandidate
from src.retrieval.base import BaseDenseRetriever


METADATA_FILENAME = "retriever_metadata.json"
INDEX_FILENAME = "faiss.index"
CURRENT_SCHEMA_VERSION = "1.0.0"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class DenseRetriever(BaseDenseRetriever):
    """
    High-performance dense semantic retriever backed by Sentence Transformers and FAISS.

    Features:
    - Pre-trained transformer embeddings (default: all-MiniLM-L6-v2, 384 dimensions).
    - Unit-hypersphere L2 normalization for exact cosine similarity via FAISS IndexFlatIP.
    - Deterministic bidirectional mapping between FAISS row IDs and Document instances.
    - Full persistence of FAISS vector index and companion document metadata.
    - Robust input validation and edge-case guards.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        model: Optional[SentenceTransformer] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        if model is not None:
            self._model = model
        else:
            self._model = SentenceTransformer(self.model_name_or_path, device=self.device)

        if hasattr(self._model, "get_embedding_dimension"):
            embedding_dim = self._model.get_embedding_dimension()
        else:
            embedding_dim = self._model.get_sentence_embedding_dimension()
        if embedding_dim is None or embedding_dim <= 0:
            raise ValueError(f"Could not determine valid embedding dimension from model '{model_name_or_path}'")
        self._dimension = int(embedding_dim)

        self._index: Optional[faiss.IndexFlatIP] = None
        self._doc_ids: List[str] = []
        self._documents: Dict[str, Document] = {}
        self._is_indexed: bool = False

    @property
    def is_indexed(self) -> bool:
        """Returns True if the retriever has an active, searchable FAISS index."""
        return self._is_indexed

    @property
    def document_count(self) -> int:
        """Returns the number of documents indexed in the vector store."""
        return len(self._doc_ids)

    @property
    def dimension(self) -> int:
        """Returns the vector embedding dimensionality (e.g. 384 for all-MiniLM-L6-v2)."""
        return self._dimension

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Transforms text strings into 2D float32 numpy embedding arrays.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for sentence-transformer inference.
            show_progress_bar: Whether to display a progress bar during encoding.

        Returns:
            np.ndarray of shape (len(texts), dimension) and dtype float32.
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        return embeddings_f32

    def index(
        self,
        documents: List[Document],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Constructs the FAISS vector index from a list of Document objects.

        Args:
            documents: Non-empty list of Document instances with unique doc_ids.
            batch_size: Inference batch size for embedding document texts.
            show_progress_bar: Whether to display inference progress.

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
        embeddings = self.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        index = faiss.IndexFlatIP(self._dimension)
        index.add(embeddings)

        self._index = index
        self._doc_ids = [doc.doc_id for doc in documents]
        self._documents = {doc.doc_id: doc for doc in documents}
        self._is_indexed = True

    def search(self, query: str, top_k: int = 5) -> List[RetrievalCandidate]:
        """
        Retrieves top_k nearest neighbor candidates for the query using cosine similarity.

        Args:
            query: Non-empty, non-whitespace query string.
            top_k: Positive integer specifying the maximum number of candidates.

        Returns:
            List of RetrievalCandidate objects sorted in descending order of similarity score.

        Raises:
            RuntimeError: If called before index() or load().
            ValueError: If query is empty/whitespace or top_k <= 0.
        """
        if not self._is_indexed or self._index is None:
            raise RuntimeError("DenseRetriever has not been indexed. Call index() or load() before search().")

        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        query_emb = self.encode([query])
        effective_k = min(top_k, len(self._doc_ids))

        scores, indices = self._index.search(query_emb, effective_k)

        candidates: List[RetrievalCandidate] = []
        for rank_idx, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            doc_id = self._doc_ids[int(idx)]
            candidate_doc = self._documents[doc_id]
            candidates.append(
                RetrievalCandidate(
                    document=candidate_doc,
                    score=float(score),
                    rank=rank_idx,
                )
            )

        return candidates

    def save(self, save_dir: Union[str, Path]) -> None:
        """
        Persists the FAISS vector index and companion metadata to disk.

        Args:
            save_dir: Destination directory path.

        Raises:
            RuntimeError: If index has not been built.
        """
        if not self._is_indexed or self._index is None:
            raise RuntimeError("Cannot save an unindexed DenseRetriever.")

        dir_path = Path(save_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / INDEX_FILENAME
        faiss.write_index(self._index, str(index_file))

        metadata = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "backend": "faiss_flat_ip",
            "model_name_or_path": self.model_name_or_path,
            "dimension": self._dimension,
            "normalize_embeddings": self.normalize_embeddings,
            "doc_ids": self._doc_ids,
            "documents": {doc_id: doc.model_dump() for doc_id, doc in self._documents.items()},
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
    ) -> "DenseRetriever":
        """
        Restores a persisted DenseRetriever from disk artifacts.

        Args:
            save_dir: Path to directory containing saved index artifacts.
            model: Optional pre-loaded SentenceTransformer instance to avoid re-instantiation.
            device: Optional device identifier ('cpu', 'cuda') if instantiating a new model.

        Returns:
            Instantiated, ready-to-search DenseRetriever.

        Raises:
            FileNotFoundError: If save_dir, metadata, or FAISS index file is missing.
            ValueError: If metadata is corrupted or schema version is incompatible.
        """
        dir_path = Path(save_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Index directory does not exist: {dir_path}")

        metadata_path = dir_path / METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing required retriever metadata file: {metadata_path}")

        index_path = dir_path / INDEX_FILENAME
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing required FAISS index file: {index_path}")

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

        model_name = metadata.get("model_name_or_path", DEFAULT_MODEL_NAME)
        normalize = metadata.get("normalize_embeddings", True)

        retriever = cls(
            model_name_or_path=model_name,
            model=model,
            device=device,
            normalize_embeddings=normalize,
        )

        try:
            retriever._index = faiss.read_index(str(index_path))
        except Exception as e:
            raise ValueError(f"Failed to load FAISS index from {index_path}: {e}")

        retriever._doc_ids = metadata["doc_ids"]
        retriever._documents = {
            doc_id: Document.model_validate(doc_dict)
            for doc_id, doc_dict in metadata["documents"].items()
        }
        retriever._is_indexed = True

        return retriever
