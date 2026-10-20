"""
Core data schemas for the CRAG project.
Defines canonical Pydantic models for documents, queries, retrieval results,
and evaluator training examples with strict validation.
"""

from enum import IntEnum, Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class LabelEnum(IntEnum):
    """Binary label values for evaluator annotations."""
    NEGATIVE = 0
    POSITIVE = 1


class CRAGAction(str, Enum):
    """The 3-way corrective action determined by the CRAG policy."""
    TRUST = "TRUST"
    VERIFY_MORE = "VERIFY_MORE"
    REJECT_SEARCH = "REJECT_SEARCH"


class Document(BaseModel):
    """
    Canonical document representation across the corpus, indexes, and retrievers.
    Document IDs must be immutable and unique across the entire corpus.
    """
    doc_id: str = Field(..., min_length=1, description="Unique, stable document identifier (e.g. 'doc_001')")
    text: str = Field(..., min_length=1, description="Raw text content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary document metadata (source, title, etc.)")


class Query(BaseModel):
    """
    Represents a search query or benchmark evaluation question.

    Note on Action Scopes:
    - Pair-level actions (TRUST, VERIFY_MORE, REJECT_SEARCH) belong to EvaluatorExample
      and RetrievalResult, evaluating specific (query, document) evidence pairs.
    - expected_action on Query represents the benchmark corpus-level expectation:
      * TRUST: The corpus contains at least one relevant and sufficient document for this query.
      * VERIFY_MORE: The best document in the corpus is relevant but incomplete.
      * REJECT_SEARCH: The corpus contains no relevant evidence (unanswerable / out-of-domain).
    """
    query_id: str = Field(..., min_length=1, description="Unique query identifier (e.g. 'q_001')")
    query_text: str = Field(..., min_length=1, description="User question or search query string")
    expected_doc_ids: List[str] = Field(default_factory=list, description="Ground-truth relevant doc IDs in the corpus, if known")
    expected_action: Optional[CRAGAction] = Field(
        None,
        description="Corpus-level benchmark expectation for this query (e.g. REJECT_SEARCH if unanswerable, TRUST if sufficient evidence exists in corpus)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional query metadata")

    @model_validator(mode="after")
    def validate_non_whitespace(self) -> "Query":
        if not self.query_text.strip():
            raise ValueError("Query text cannot be whitespace only.")
        return self


class RetrievalCandidate(BaseModel):
    """Individual candidate returned by a retriever (sparse, dense, or fused)."""
    document: Document
    score: float = Field(..., description="Score assigned by the retriever or fusion function")
    rank: int = Field(..., ge=1, description="1-indexed rank of this document in retrieval results")


class RetrievalResult(BaseModel):
    """
    Final output contract consumed by Member 3's CRAG pipeline.
    Preserves document identity, raw retriever ranks, and fused scores.
    """
    document: Document
    fused_score: float = Field(..., description="Reciprocal Rank Fusion (RRF) fused score")
    fused_rank: int = Field(..., ge=1, description="1-indexed rank in the fused ranking")
    sparse_rank: Optional[int] = Field(None, ge=1, description="1-indexed rank in sparse search, if returned")
    dense_rank: Optional[int] = Field(None, ge=1, description="1-indexed rank in dense search, if returned")
    sparse_score: Optional[float] = Field(None, description="Raw sparse retriever score (e.g. BM25)")
    dense_score: Optional[float] = Field(None, description="Raw dense retriever score (e.g. Cosine/IP)")


class EvaluatorExample(BaseModel):
    """
    Training and evaluation example for Member 2's custom DeBERTa evaluator.
    Pairs a user query with a retrieved document and provides binary labels
    for Intent-Fit and Evidence-Sufficiency.
    
    ENFORCES: The 3 valid combinations:
      (1, 1) -> TRUST
      (1, 0) -> VERIFY_MORE
      (0, 0) -> REJECT_SEARCH
      (0, 1) -> PROHIBITED: Raises ValueError.
    """
    example_id: str = Field(..., min_length=1, description="Unique example identifier (hash or uuid)")
    query_id: str = Field(..., min_length=1, description="Source query ID")
    query_text: str = Field(..., min_length=1, description="User question")
    doc_id: str = Field(..., min_length=1, description="Candidate document ID")
    doc_text: str = Field(..., min_length=1, description="Candidate document text")
    
    intent_fit: LabelEnum = Field(
        ...,
        description="1 if document addresses the actual user query intent; 0 otherwise."
    )
    evidence_sufficiency: LabelEnum = Field(
        ...,
        description="1 if document contains sufficient facts to answer the user query; 0 otherwise."
    )
    
    example_type: str = Field(
        ...,
        description="Provenance category: 'relevant_sufficient' | 'relevant_insufficient' | 'hard_negative_topic' | 'random_negative'"
    )
    rationale: str = Field(..., min_length=1, description="Detailed explanation justifying the assigned labels")
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Audit metadata (source, generator model, timestamp, temperature, etc.)"
    )

    @model_validator(mode="after")
    def validate_label_consistency(self) -> "EvaluatorExample":
        # Rule: A document cannot have sufficient evidence for a query whose intent it does not fit.
        if self.intent_fit == LabelEnum.NEGATIVE and self.evidence_sufficiency == LabelEnum.POSITIVE:
            raise ValueError(
                "Prohibited label combination: (intent_fit=0, evidence_sufficiency=1). "
                "A document cannot provide sufficient evidence to answer a question "
                "whose intent it does not address."
            )
        return self

    @property
    def crag_action(self) -> CRAGAction:
        """Derives the deterministic CRAG action from the valid binary labels."""
        if self.intent_fit == LabelEnum.POSITIVE:
            if self.evidence_sufficiency == LabelEnum.POSITIVE:
                return CRAGAction.TRUST
            return CRAGAction.VERIFY_MORE
        return CRAGAction.REJECT_SEARCH
