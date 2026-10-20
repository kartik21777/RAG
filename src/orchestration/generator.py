"""
Grounded Response Generator for Corrective RAG (CRAG) — Milestone 6.

Consumes accepted evidence and terminal action outcomes from the CRAG orchestration
pipeline and synthesizes grounded answers with traceable source citations while strictly
preventing hallucination from rejected or absent evidence.

Core Architectural Principles:
1. Strict Grounding: All factual claims must derive directly from accepted evidence documents.
   Every citation points to an actual document present in the supplied evidence set.
2. Safe Refusal:
   - When action is REJECT_SEARCH, factual generation is declined with REJECTED_SEARCH status.
   - When evidence is empty or lacks query-answering facts, generation is declined with
     INSUFFICIENT_EVIDENCE status.
3. Separation of Responsibilities:
   The generator does not re-retrieve or re-evaluate. It consumes the authoritative
   OrchestratorResult (or raw evidence + action inputs) and focuses purely on grounded synthesis.
4. Deterministic & Offline:
   DeterministicGroundedGenerator operates locally without network access, external LLM APIs,
   or GPU dependencies, making it 100% offline-capable and CPU-compatible.
"""

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List, Optional, Set

from src.core.schemas import CRAGAction, RetrievalResult
from src.orchestration.orchestrator import OrchestratorResult
from src.orchestration.schemas import (
    EvidenceCitation,
    GenerationStatus,
    GroundedResponse,
    RefinementResult,
)


class BaseGroundedGenerator(ABC):
    """
    Abstract interface for grounded response generation in CRAG.
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        evidence: List[RetrievalResult],
        action: CRAGAction,
        final_query: Optional[str] = None,
        refinement_result: Optional[RefinementResult] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GroundedResponse:
        """
        Synthesizes a grounded answer or safe refusal based on the CRAG pipeline outcome.

        Args:
            query: The original user question.
            evidence: Accepted evidence documents from the CRAG pipeline.
            action: Terminal corrective action (TRUST, VERIFY_MORE, REJECT_SEARCH).
            final_query: Effective query string after rewriting, if applicable.
            refinement_result: RefinementResult if VERIFY_MORE path was executed.
            metadata: Optional diagnostic, routing, or provenance metadata.

        Returns:
            GroundedResponse containing answer text, status, and verified citations.
        """
        pass

    def generate_from_result(self, result: OrchestratorResult) -> GroundedResponse:
        """
        Convenience adapter that consumes an OrchestratorResult directly.

        Extracts metadata, evidence, actions, and inspects candidate routing decisions.
        Preserves the independent candidate evaluation rule: if the pipeline accepted
        evidence under TRUST or VERIFY_MORE, a separately rejected candidate (e.g. one
        with an intent-dominance override) does NOT invalidate the accepted evidence.

        Args:
            result: OrchestratorResult emitted by CRAGOrchestrator.run_pipeline().

        Returns:
            GroundedResponse preserving query, action, citations, and provenance.
        """
        if not isinstance(result, OrchestratorResult):
            raise TypeError(f"Expected OrchestratorResult, got {type(result).__name__}")

        metadata = dict(result.metadata or {})

        # When the overall action is REJECT_SEARCH, inspect if any candidate triggered
        # the intent dominance override (sufficiency met threshold but intent failed).
        if result.final_action == CRAGAction.REJECT_SEARCH:
            if result.query_decision and hasattr(result.query_decision, "candidate_decisions"):
                for cand in result.query_decision.candidate_decisions:
                    if getattr(cand, "is_contradictory_override", False):
                        metadata["is_contradictory_override"] = True
                        break

        return self.generate(
            query=result.original_query,
            evidence=result.final_results,
            action=result.final_action,
            final_query=result.final_query,
            refinement_result=result.refinement_result,
            metadata=metadata,
        )

    @staticmethod
    def validate_query(query: Any) -> str:
        """Validates that the query string is non-empty and non-whitespace."""
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query cannot be empty or whitespace only.")
        return cleaned


class DeterministicGroundedGenerator(BaseGroundedGenerator):
    """
    Deterministic, CPU-compatible grounded response generator.

    Synthesizes factual statements from accepted evidence documents with verbatim
    sentence-level citations, without requiring live LLM API calls or GPU models.
    """

    STOPWORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "can", "could", "will", "would", "shall", "should", "may", "might",
        "must", "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "it", "its", "how", "why", "where", "when", "explain", "describe",
    }

    CAUSAL_MARKERS: Set[str] = {
        "because", "due", "as", "since", "reason", "caused", "causing",
        "result", "results", "consequence", "therefore", "thus", "thereby",
        "why", "factor", "factors", "explains", "explanation", "leads", "lead",
    }

    NUMERIC_VALUE_MARKERS: Set[str] = {
        "how", "many", "much", "cost", "price", "number", "percentage", "rate",
        "count", "amount", "size", "distance", "height", "weight", "length",
        "total", "average", "ratio", "quantity", "value", "score",
    }

    def __init__(
        self,
        max_citations: int = 3,
        min_overlap_terms: int = 1,
    ) -> None:
        """
        Args:
            max_citations: Maximum number of evidence citations to emit per answer (>= 1).
            min_overlap_terms: Minimum keyword overlap required between query and sentence (>= 1).
        """
        if not isinstance(max_citations, int) or max_citations < 1:
            raise ValueError(f"max_citations must be an integer >= 1, got {max_citations!r}")
        if not isinstance(min_overlap_terms, int) or min_overlap_terms < 1:
            raise ValueError(f"min_overlap_terms must be an integer >= 1, got {min_overlap_terms!r}")

        self.max_citations = max_citations
        self.min_overlap_terms = min_overlap_terms

    # ------------------------------------------------------------------
    # Text Analysis Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _extract_keywords(cls, text: str) -> Set[str]:
        """Extracts lowercased alphanumeric keywords excluding common stopwords."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
        return {t for t in tokens if t not in cls.STOPWORDS}

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Splits document text into clean, non-empty candidate sentences."""
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    @classmethod
    def _has_numeric_token(cls, text: str) -> bool:
        """Checks if the sentence contains at least one digit or numeric representation."""
        return bool(re.search(r"\b\d+(\.\d+)?\b", text))

    @classmethod
    def _is_answerable_sentence(
        cls,
        sentence: str,
        query_text: str,
        overlap_keywords: Set[str],
        query_keywords: Set[str],
        min_overlap_terms: int,
    ) -> bool:
        """
        Evaluates whether a candidate sentence meets the answerability heuristic for a query.

        Heuristic rules:
        1. Overlap Threshold: Must meet the configured min_overlap_terms.
        2. Proportional Relevance: If the query has multiple key concepts (>=2 keywords),
           a single overlapping keyword is insufficient to establish answerability.
        3. Causal Inquiry Check ("Why..."): If the query asks for cause/explanation ("why"),
           the sentence must contain causal markers ("because", "due to", "caused by", etc.)
           or address the explanation rather than merely mentioning the topic entity.
        4. Specific Value Inquiry Check ("How many / much / cost / count..."): If the query
           asks for a specific quantity, count, or value, the sentence must contain numeric
           data or explicit value terms.
        """
        if len(overlap_keywords) < min_overlap_terms:
            return False

        # Proportional coverage: for queries with 2+ distinct topic keywords, require at least
        # 2 overlapping keywords (or all keywords if query has exactly 2) unless min_overlap_terms is higher.
        effective_min = max(min_overlap_terms, 2 if len(query_keywords) >= 2 else 1)
        if len(overlap_keywords) < effective_min:
            return False

        q_lower = query_text.lower()
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", q_lower))

        # Check 1: Causal "Why" question
        if "why" in q_tokens:
            s_lower = sentence.lower()
            s_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", s_lower))
            if not s_tokens.intersection(cls.CAUSAL_MARKERS):
                return False

        # Check 2: Specific quantity/value inquiry
        if any(term in q_tokens for term in ("many", "much", "cost", "price", "count", "amount", "percentage", "rate")):
            if not cls._has_numeric_token(sentence):
                return False

        return True

    # ------------------------------------------------------------------
    # Generation Logic
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        evidence: List[RetrievalResult],
        action: CRAGAction,
        final_query: Optional[str] = None,
        refinement_result: Optional[RefinementResult] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GroundedResponse:
        clean_query = self.validate_query(query)
        eff_query = final_query.strip() if (isinstance(final_query, str) and final_query.strip()) else clean_query
        meta = dict(metadata or {})

        # Safety Gate 1: Check for explicit contradiction or intent-dominance override
        if meta.get("is_contradictory_override"):
            return GroundedResponse(
                query=clean_query,
                final_query=eff_query,
                answer=(
                    "I cannot provide a factual answer because the retrieved candidate "
                    "triggered an intent-dominance override (evidence sufficiency met threshold "
                    "but failed question intent fit)."
                ),
                status=GenerationStatus.INSUFFICIENT_EVIDENCE,
                citations=[],
                action_used=action,
                refusal_reason="Intent-dominance override: candidate sufficiency met threshold but intent fit failed.",
                metadata=meta,
            )
        if meta.get("contradictory"):
            return GroundedResponse(
                query=clean_query,
                final_query=eff_query,
                answer="I cannot provide a factual answer because contradictory evidence was detected.",
                status=GenerationStatus.INSUFFICIENT_EVIDENCE,
                citations=[],
                action_used=action,
                refusal_reason="Contradictory evidence detected in candidate evaluation.",
                metadata=meta,
            )

        # Safety Gate 2: REJECT_SEARCH -> Strictly decline factual generation
        if action == CRAGAction.REJECT_SEARCH:
            return GroundedResponse(
                query=clean_query,
                final_query=eff_query,
                answer=(
                    "I cannot provide a factual answer because the available evidence was "
                    "judged off-topic or insufficient by the CRAG evaluation system."
                ),
                status=GenerationStatus.REJECTED_SEARCH,
                citations=[],
                action_used=action,
                refusal_reason="CRAG policy evaluated retrieval as REJECT_SEARCH.",
                metadata={
                    **meta,
                    "rejected_candidate_count": len(evidence or []),
                },
            )

        # Safety Gate 3: Empty evidence set or malformed evidence list
        if not evidence:
            return GroundedResponse(
                query=clean_query,
                final_query=eff_query,
                answer="I cannot provide an answer because no supporting evidence was provided.",
                status=GenerationStatus.INSUFFICIENT_EVIDENCE,
                citations=[],
                action_used=action,
                refusal_reason="Evidence list is empty.",
                metadata=meta,
            )

        # Validate evidence elements are valid RetrievalResults
        valid_evidence: List[RetrievalResult] = []
        for item in evidence:
            if not isinstance(item, RetrievalResult) or not hasattr(item, "document") or item.document is None:
                return GroundedResponse(
                    query=clean_query,
                    final_query=eff_query,
                    answer="Generation failed due to malformed or unusable evidence input.",
                    status=GenerationStatus.FAILED,
                    citations=[],
                    action_used=action,
                    refusal_reason="Evidence contains malformed or unparseable items.",
                    metadata=meta,
                )
            valid_evidence.append(item)

        # Step 4: Extract keywords for relevance matching
        query_keywords = self._extract_keywords(clean_query)
        # If final_query differs, also include its disambiguated keywords
        if eff_query != clean_query:
            query_keywords.update(self._extract_keywords(eff_query))

        # Collect candidate sentences from accepted evidence in ranking order
        matched_claims: List[str] = []
        citations: List[EvidenceCitation] = []
        cited_doc_ids: Set[str] = set()

        for result in valid_evidence:
            doc = result.document
            doc_id = doc.doc_id if (doc and doc.doc_id) else "unknown_doc"
            doc_text = doc.text if doc else ""
            doc_meta = doc.metadata if (doc and doc.metadata) else {}

            sentences = self._split_into_sentences(doc_text)
            for sentence in sentences:
                sentence_keywords = self._extract_keywords(sentence)
                overlap = query_keywords.intersection(sentence_keywords)

                if self._is_answerable_sentence(
                    sentence=sentence,
                    query_text=clean_query,
                    overlap_keywords=overlap,
                    query_keywords=query_keywords,
                    min_overlap_terms=self.min_overlap_terms,
                ):
                    # Clean snippet
                    clean_snippet = sentence.strip()
                    matched_claims.append(f"{clean_snippet} [{doc_id}]")
                    citations.append(
                        EvidenceCitation(
                            doc_id=doc_id,
                            snippet=clean_snippet,
                            source=doc_meta.get("source") or doc_meta.get("title"),
                            rank=result.fused_rank,
                            score=result.fused_score,
                        )
                    )
                    cited_doc_ids.add(doc_id)

                    if len(citations) >= self.max_citations:
                        break

            if len(citations) >= self.max_citations:
                break

        # Safety Gate 5: Evidence lacked factual relevance or answerability to the query
        if not citations:
            return GroundedResponse(
                query=clean_query,
                final_query=eff_query,
                answer=(
                    "I cannot provide a factual answer because the accepted evidence does not "
                    "contain sufficient factual details to answer the question."
                ),
                status=GenerationStatus.INSUFFICIENT_EVIDENCE,
                citations=[],
                action_used=action,
                refusal_reason="Accepted evidence lacks answer-bearing factual overlap with query.",
                metadata={
                    **meta,
                    "evidence_docs_inspected": len(valid_evidence),
                },
            )

        # Grounded answer synthesis
        answer_text = " ".join(matched_claims)
        generation_metadata: Dict[str, Any] = {
            **meta,
            "evidence_count": len(evidence),
            "citations_emitted": len(citations),
            "cited_doc_ids": list(cited_doc_ids),
            "refinement_applied": refinement_result is not None,
        }
        if refinement_result is not None:
            generation_metadata["refinement_strategy"] = refinement_result.refinement_strategy
            generation_metadata["is_expanded"] = refinement_result.is_expanded

        return GroundedResponse(
            query=clean_query,
            final_query=eff_query,
            answer=answer_text,
            status=GenerationStatus.GROUNDED,
            citations=citations,
            action_used=action,
            metadata=generation_metadata,
        )
