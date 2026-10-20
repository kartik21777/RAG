"""
Negative sampling strategies for training data generation.
Includes random negative sampling and retriever-driven hard negative mining.
"""

import random
from typing import Any, List, Optional, Set
from src.core.schemas import Document, Query


class NegativeSampler:
    """
    Samples negative document passages for query-document evaluator datasets.
    Supports both uncorrelated random negatives and retriever-mined hard negatives.
    """

    def __init__(self, corpus: List[Document], seed: Optional[int] = 42):
        if not corpus:
            raise ValueError("Corpus cannot be empty.")
        self.corpus = corpus
        self.doc_map = {doc.doc_id: doc for doc in corpus}
        self.rng = random.Random(seed)

    def sample_random_negatives(
        self,
        query: Query,
        num_samples: int = 1,
        exclude_doc_ids: Optional[Set[str]] = None,
    ) -> List[Document]:
        """
        Draws random documents from the corpus that are unrelated to the query.

        Args:
            query: Source Query instance.
            num_samples: Number of random negatives to sample.
            exclude_doc_ids: Optional set of document IDs to exclude from sampling.

        Returns:
            List of randomly sampled Document objects.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        excluded = set(query.expected_doc_ids)
        if exclude_doc_ids:
            excluded.update(exclude_doc_ids)

        eligible = [doc for doc in self.corpus if doc.doc_id not in excluded]
        if not eligible:
            raise ValueError(f"No eligible documents available to sample random negatives for query '{query.query_id}'.")

        sample_count = min(num_samples, len(eligible))
        return self.rng.sample(eligible, k=sample_count)

    def sample_hard_negatives(
        self,
        query: Query,
        retriever: Any,
        search_top_k: int = 10,
        num_samples: int = 1,
        exclude_doc_ids: Optional[Set[str]] = None,
    ) -> List[Document]:
        """
        Mines hard negative documents using a search retriever.
        Selects top-retrieved candidates that share topical/lexical overlap
        with the query but are not in the query's ground-truth relevant set.

        Args:
            query: Source Query instance.
            retriever: Search retriever instance (BM25, Dense, or Hybrid) supporting search(query, top_k).
            search_top_k: Number of candidates to retrieve prior to negative filtering.
            num_samples: Number of hard negative candidates to retain.
            exclude_doc_ids: Optional set of document IDs to exclude.

        Returns:
            List of high-scoring, non-relevant Document objects.
        """
        if search_top_k <= 0:
            raise ValueError(f"search_top_k must be positive, got {search_top_k}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        excluded = set(query.expected_doc_ids)
        if exclude_doc_ids:
            excluded.update(exclude_doc_ids)

        candidates = retriever.search(query.query_text, top_k=search_top_k)

        hard_negatives: List[Document] = []
        for cand in candidates:
            doc_id = cand.document.doc_id
            if doc_id not in excluded and doc_id not in [d.doc_id for d in hard_negatives]:
                hard_negatives.append(cand.document)
                if len(hard_negatives) >= num_samples:
                    break

        return hard_negatives
