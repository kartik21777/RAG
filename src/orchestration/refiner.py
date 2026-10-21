"""
Evidence Refinement Layer for Semantic- and Evidence-Aware CRAG (Option C).

Defines the abstract interface and concrete refiners for expanding and completing
evidence marked as VERIFY_MORE (relevant but incomplete).

Key Architectural Rules:
1. Grounded & Safe: Never invent neighboring chunks or fabricate text.
2. Safe Fallback: If document metadata lacks adjacent chunk links (as in the 50-doc corpus),
   gracefully fall back to secondary targeted retrieval without pretending adjacent chunks exist.
3. Provenance & Deduplication: Merged evidence sets preserve complete RetrievalResult provenance
   (scores, ranks, metadata) and enforce strictly unique document IDs without mutating inputs.
4. Explicit Skip vs Attempted: Distinctly separates skipped refinement (no verification targets)
   from an attempted refinement that discovered no new candidates.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Union
from src.core.schemas import Document, RetrievalResult
from src.orchestration.schemas import CRAGAction, RefinementResult
from src.retrieval.facade import CRAGRetrieverFacade
from src.orchestration.evaluator import BaseRuntimeEvaluator
from src.orchestration.router import CorrectiveDecisionRouter

REFINER_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "it", "its", "how", "why", "where", "when", "tell", "give", "me",
}


def extract_missing_information_query(
    query: str,
    partial_docs: List[Document],
    query_generator: Optional[Callable[[str, List[Document]], Optional[str]]] = None,
) -> str:
    """
    Synthesizes a concise, targeted secondary retrieval query seeking missing information.

    Precedence:
    1. Injected query_generator callable (e.g. LLM integration or dedicated query generator).
       If it executes successfully and returns a non-empty string, that targeted query is used.
    2. Bounded fallback: if query_generator is omitted, raises an exception, or returns
       an empty/unusable string, falls back to the original query (the existing deeper retrieval
       mechanism) preserving query intent without guessing.
    """
    clean_query = query.strip()
    if query_generator is not None:
        try:
            generated = query_generator(clean_query, partial_docs)
            if isinstance(generated, str) and generated.strip():
                return generated.strip()
        except Exception:
            # Safe bounded fallback: do not crash on generator exception
            pass

    return clean_query


def wrap_document_as_retrieval_result(
    doc: Document,
    rank: int = 1,
    score: float = 0.0,
) -> RetrievalResult:
    """
    Wraps a bare Document instance into a canonical RetrievalResult with default ranking metadata.
    """
    return RetrievalResult(
        document=doc.model_copy(deep=True),
        fused_score=score,
        fused_rank=rank,
        sparse_rank=None,
        dense_rank=None,
        sparse_score=None,
        dense_score=None,
    )


class BaseEvidenceRefiner(ABC):
    """
    Abstract interface for refining incomplete candidate evidence.
    """

    @abstractmethod
    def refine(
        self,
        query: str,
        initial_results: Optional[List[Union[RetrievalResult, Document]]] = None,
        verification_target_ids: Optional[List[str]] = None,
        existing_results: Optional[List[Union[RetrievalResult, Document]]] = None,
    ) -> RefinementResult:
        """
        Refines evidence for a given query.

        Args:
            query: User question or search query string.
            initial_results: Initial candidates evaluated by the router (can be empty for recovery).
            verification_target_ids: Explicit list of doc IDs evaluated as VERIFY_MORE.
            existing_results: Optional existing trusted results to preserve and merge with.

        Returns:
            RefinementResult containing merged, deduplicated results with full provenance.
        """
        pass

    @staticmethod
    def normalize_results(
        items: Optional[List[Union[RetrievalResult, Document]]],
    ) -> List[RetrievalResult]:
        """
        Normalizes a list of RetrievalResult or Document items into non-mutated RetrievalResult objects.
        """
        if not items:
            return []
        normalized: List[RetrievalResult] = []
        for i, item in enumerate(items, start=1):
            if isinstance(item, RetrievalResult):
                normalized.append(item.model_copy(deep=True))
            elif isinstance(item, Document):
                normalized.append(wrap_document_as_retrieval_result(item, rank=i))
            else:
                raise TypeError(f"Item must be RetrievalResult or Document, got {type(item).__name__}")
        return normalized

    @staticmethod
    def validate_initial_pool_uniqueness(results: List[RetrievalResult]) -> None:
        """
        Validates that initial pool has strictly unique document IDs.
        """
        seen_ids: Set[str] = set()
        for r in results:
            d_id = r.document.doc_id
            if d_id in seen_ids:
                raise ValueError(
                    f"Duplicate document ID '{d_id}' detected in initial results pool. "
                    f"All candidate document IDs must be strictly unique."
                )
            seen_ids.add(d_id)

    @staticmethod
    def deduplicate_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Preserves the first occurrence of each unique doc_id in order of appearance.
        Distinct doc IDs are never merged merely because their text is similar.
        """
        seen_ids: Set[str] = set()
        deduped: List[RetrievalResult] = []
        for r in results:
            d_id = r.document.doc_id
            if d_id not in seen_ids:
                seen_ids.add(d_id)
                deduped.append(r)
        return deduped


class MockEvidenceRefiner(BaseEvidenceRefiner):
    """
    Deterministic, configurable mock refiner for offline testing without index or API calls.
    Supports targeted missing-information queries, re-evaluation, and adjacent/secondary expansion maps.
    """

    def __init__(
        self,
        expansion_map: Optional[Dict[str, List[Union[RetrievalResult, Document]]]] = None,
        default_expansion: Optional[List[Union[RetrievalResult, Document]]] = None,
        strategy_name: str = "mock",
        evaluator: Optional[BaseRuntimeEvaluator] = None,
        router: Optional[CorrectiveDecisionRouter] = None,
        query_generator: Optional[Callable[[str, List[Document]], Optional[str]]] = None,
        targeted_expansion_map: Optional[Dict[str, List[Union[RetrievalResult, Document]]]] = None,
        default_targeted_expansion: Optional[List[Union[RetrievalResult, Document]]] = None,
    ):
        self.expansion_map: Dict[str, List[Union[RetrievalResult, Document]]] = expansion_map or {}
        self.default_expansion: List[Union[RetrievalResult, Document]] = default_expansion or []
        self.strategy_name = strategy_name
        self.evaluator = evaluator
        self.router = router
        self.query_generator = query_generator
        self.targeted_expansion_map = targeted_expansion_map or {}
        self.default_targeted_expansion = default_targeted_expansion or []

    def refine(
        self,
        query: str,
        initial_results: Optional[List[Union[RetrievalResult, Document]]] = None,
        verification_target_ids: Optional[List[str]] = None,
        existing_results: Optional[List[Union[RetrievalResult, Document]]] = None,
    ) -> RefinementResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty or whitespace only.")
        clean_query = query.strip()

        norm_initial = self.normalize_results(initial_results)
        norm_existing = self.normalize_results(existing_results)
        self.validate_initial_pool_uniqueness(norm_initial)
        self.validate_initial_pool_uniqueness(norm_existing)

        # Merge initial and existing pools preserving priority order
        initial_pool = norm_existing + norm_initial
        initial_ids = [r.document.doc_id for r in initial_pool]

        # Case 1: Empty initial retrieval triggers recovery attempt
        if not initial_pool:
            norm_defaults = self.normalize_results(self.default_expansion)
            deduped = self.deduplicate_results(norm_defaults)
            added_ids = [r.document.doc_id for r in deduped]
            return RefinementResult(
                query_text=clean_query,
                original_doc_ids=[],
                refined_results=deduped,
                refined_documents=[r.document for r in deduped],
                refinement_strategy="recovery_retrieval",
                is_expanded=len(deduped) > 0,
                skipped=False,
                added_doc_ids=added_ids,
                added_results=deduped,
                metadata={"refiner_type": "MockEvidenceRefiner", "is_recovery": True},
            )

        # Validate verification_target_ids consistency
        if verification_target_ids is not None:
            initial_id_set = set(initial_ids)
            for tid in verification_target_ids:
                if tid not in initial_id_set:
                    raise ValueError(
                        f"Inconsistent verification_target_ids: target '{tid}' "
                        f"is not present in initial candidate results {initial_ids}."
                    )

            # Case 2: Non-empty initial results with zero verification targets -> SKIPPED
            if len(verification_target_ids) == 0:
                return RefinementResult(
                    query_text=clean_query,
                    original_doc_ids=[],
                    refined_results=initial_pool,
                    refined_documents=[r.document for r in initial_pool],
                    refinement_strategy="skipped_no_targets",
                    is_expanded=False,
                    skipped=True,
                    added_doc_ids=[],
                    added_results=[],
                    metadata={"refiner_type": "MockEvidenceRefiner", "reason": "No verification targets specified"},
                )

        # Determine target IDs to expand
        target_ids = verification_target_ids if verification_target_ids is not None else initial_ids
        seen_ids = set(initial_ids)

        added_results: List[RetrievalResult] = []
        for tid in target_ids:
            expansions = self.expansion_map.get(tid, self.default_expansion)
            norm_exp = self.normalize_results(expansions)
            for exp in norm_exp:
                d_id = exp.document.doc_id
                if d_id not in seen_ids:
                    seen_ids.add(d_id)
                    added_results.append(exp)

        adjacent_is_sufficient = False
        targeted_query: Optional[str] = None
        is_sufficient: Optional[bool] = None
        re_eval_action: Optional[CRAGAction] = None
        strategy_used = self.strategy_name

        if self.evaluator is not None and self.router is not None:
            current_pool = initial_pool + added_results
            for cand in current_pool:
                pred = self.evaluator.evaluate_pair(clean_query, cand.document)
                dec = self.router.route_candidate(pred)
                if dec.action == CRAGAction.TRUST:
                    adjacent_is_sufficient = True
                    break
            if not adjacent_is_sufficient and hasattr(self.evaluator, "evaluate_collection"):
                coll_pred = self.evaluator.evaluate_collection(clean_query, [c.document for c in current_pool])
                coll_dec = self.router.route_candidate(coll_pred)
                if coll_dec.action == CRAGAction.TRUST:
                    adjacent_is_sufficient = True

            # If still insufficient and targeted expansions are configured, execute targeted expansion
            if not adjacent_is_sufficient and (self.targeted_expansion_map or self.default_targeted_expansion or self.query_generator):
                current_docs = [r.document for r in (initial_pool + added_results)]
                targeted_query = extract_missing_information_query(
                    clean_query, current_docs, query_generator=self.query_generator
                )
                t_expansions = self.targeted_expansion_map.get(targeted_query, self.default_targeted_expansion)
                norm_t_exp = self.normalize_results(t_expansions)
                for exp in norm_t_exp:
                    d_id = exp.document.doc_id
                    if d_id not in seen_ids:
                        seen_ids.add(d_id)
                        pred = self.evaluator.evaluate_pair(clean_query, exp.document)
                        cand_dec = self.router.route_candidate(pred)
                        if cand_dec.action == CRAGAction.REJECT_SEARCH:
                            continue
                        added_results.append(exp)

            # Re-evaluate final pool
            merged = initial_pool + added_results
            has_trusted = False
            for cand in merged:
                pred = self.evaluator.evaluate_pair(clean_query, cand.document)
                dec = self.router.route_candidate(pred)
                if dec.action == CRAGAction.TRUST:
                    has_trusted = True
                    break
            if not has_trusted and hasattr(self.evaluator, "evaluate_collection"):
                coll_pred = self.evaluator.evaluate_collection(clean_query, [c.document for c in merged])
                coll_dec = self.router.route_candidate(coll_pred)
                if coll_dec.action == CRAGAction.TRUST:
                    has_trusted = True

            is_sufficient = has_trusted
            re_eval_action = CRAGAction.TRUST if has_trusted else CRAGAction.VERIFY_MORE
        else:
            merged = initial_pool + added_results

        added_ids = [r.document.doc_id for r in added_results]

        return RefinementResult(
            query_text=clean_query,
            original_doc_ids=target_ids,
            refined_results=merged,
            refined_documents=[r.document for r in merged],
            refinement_strategy=strategy_used,
            is_expanded=len(added_results) > 0,
            skipped=False,
            added_doc_ids=added_ids,
            added_results=added_results,
            targeted_query=targeted_query,
            is_sufficient=is_sufficient,
            re_evaluation_action=re_eval_action,
            metadata={
                "refiner_type": "MockEvidenceRefiner",
                "targeted_query": targeted_query,
                "is_sufficient": is_sufficient,
                "re_evaluation_action": re_eval_action.value if re_eval_action else None,
                "initial_count": len(initial_pool),
                "added_count": len(added_results),
                "final_count": len(merged),
            },
        )


class SecondaryRetrievalRefiner(BaseEvidenceRefiner):
    """
    Production-grade refiner that inspects chunk metadata for adjacent chunk links,
    and falls back cleanly to targeted secondary retrieval seeking missing information
    when sequential links are absent or insufficient.
    """

    def __init__(
        self,
        retriever_facade: CRAGRetrieverFacade,
        corpus_lookup: Optional[Dict[str, Document]] = None,
        secondary_top_k: int = 3,
        evaluator: Optional[BaseRuntimeEvaluator] = None,
        router: Optional[CorrectiveDecisionRouter] = None,
        query_generator: Optional[Callable[[str, List[Document]], Optional[str]]] = None,
    ):
        """
        Args:
            retriever_facade: Initialized and indexed CRAGRetrieverFacade instance.
            corpus_lookup: Optional mapping of doc_id -> Document for adjacent chunk retrieval.
            secondary_top_k: Number of candidates to retrieve during secondary search.
            evaluator: Optional BaseRuntimeEvaluator for validating sufficiency of expanded evidence.
            router: Optional CorrectiveDecisionRouter for evaluating candidate and collection actions.
            query_generator: Optional callable synthesizing targeted queries seeking missing information.
        """
        if not isinstance(retriever_facade, CRAGRetrieverFacade):
            raise TypeError(f"retriever_facade must be CRAGRetrieverFacade, got {type(retriever_facade).__name__}")
        if not retriever_facade.is_indexed:
            raise RuntimeError("CRAGRetrieverFacade must be indexed before passing to SecondaryRetrievalRefiner.")

        if (
            isinstance(secondary_top_k, bool)
            or not isinstance(secondary_top_k, int)
            or secondary_top_k <= 0
        ):
            raise ValueError(f"secondary_top_k must be a positive integer, got {secondary_top_k}")

        if evaluator is not None and not isinstance(evaluator, BaseRuntimeEvaluator):
            raise TypeError(f"evaluator must be BaseRuntimeEvaluator, got {type(evaluator).__name__}")
        if router is not None and not isinstance(router, CorrectiveDecisionRouter):
            raise TypeError(f"router must be CorrectiveDecisionRouter, got {type(router).__name__}")

        self.retriever = retriever_facade
        self.corpus_lookup: Dict[str, Document] = corpus_lookup or {}
        self.secondary_top_k = secondary_top_k
        self.evaluator = evaluator
        self.router = router
        self.query_generator = query_generator

    def refine(
        self,
        query: str,
        initial_results: Optional[List[Union[RetrievalResult, Document]]] = None,
        verification_target_ids: Optional[List[str]] = None,
        existing_results: Optional[List[Union[RetrievalResult, Document]]] = None,
    ) -> RefinementResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty or whitespace only.")
        clean_query = query.strip()

        norm_initial = self.normalize_results(initial_results)
        norm_existing = self.normalize_results(existing_results)
        self.validate_initial_pool_uniqueness(norm_initial)
        self.validate_initial_pool_uniqueness(norm_existing)

        # Merge initial and existing pools preserving priority order
        initial_pool = norm_existing + norm_initial
        initial_ids = [r.document.doc_id for r in initial_pool]

        # Case 1: Empty initial retrieval triggers recovery attempt
        if not initial_pool:
            recovery_results = self.retriever.retrieve(clean_query, top_k=self.secondary_top_k)
            deduped = self.deduplicate_results(recovery_results)
            added_ids = [r.document.doc_id for r in deduped]
            return RefinementResult(
                query_text=clean_query,
                original_doc_ids=[],
                refined_results=deduped,
                refined_documents=[r.document for r in deduped],
                refinement_strategy="recovery_retrieval",
                is_expanded=len(deduped) > 0,
                skipped=False,
                added_doc_ids=added_ids,
                added_results=deduped,
                metadata={
                    "refiner_type": "SecondaryRetrievalRefiner",
                    "is_recovery": True,
                    "retrieved_count": len(recovery_results),
                },
            )

        # Validate verification_target_ids consistency
        if verification_target_ids is not None:
            initial_id_set = set(initial_ids)
            for tid in verification_target_ids:
                if tid not in initial_id_set:
                    raise ValueError(
                        f"Inconsistent verification_target_ids: target '{tid}' "
                        f"is not present in initial candidate results {initial_ids}."
                    )

            # Case 2: Non-empty initial results with zero verification targets -> SKIPPED
            if len(verification_target_ids) == 0:
                return RefinementResult(
                    query_text=clean_query,
                    original_doc_ids=[],
                    refined_results=initial_pool,
                    refined_documents=[r.document for r in initial_pool],
                    refinement_strategy="skipped_no_targets",
                    is_expanded=False,
                    skipped=True,
                    added_doc_ids=[],
                    added_results=[],
                    metadata={
                        "refiner_type": "SecondaryRetrievalRefiner",
                        "reason": "No verification targets specified; refinement skipped without retrieval",
                    },
                )

        target_ids = verification_target_ids if verification_target_ids is not None else initial_ids
        target_results = [r for r in initial_pool if r.document.doc_id in target_ids]
        seen_ids = set(initial_ids)

        added_results: List[RetrievalResult] = []
        strategy_used = "none"
        adjacent_added = False
        targeted_query: Optional[str] = None
        is_sufficient: Optional[bool] = None
        re_eval_action: Optional[CRAGAction] = None

        # Step A: Attempt adjacent/parent chunk expansion if explicit metadata exists
        for r in target_results:
            meta = r.document.metadata or {}
            adjacent_ids: List[str] = []
            if "adjacent_chunk_ids" in meta and isinstance(meta["adjacent_chunk_ids"], list):
                adjacent_ids.extend([aid for aid in meta["adjacent_chunk_ids"] if isinstance(aid, str)])
            for field in ("prev_chunk_id", "next_chunk_id"):
                val = meta.get(field)
                if isinstance(val, str):
                    adjacent_ids.append(val)
                elif isinstance(val, list):
                    adjacent_ids.extend([v for v in val if isinstance(v, str)])

            valid_adj_ids = [aid for aid in adjacent_ids if aid in self.corpus_lookup]
            for aid in valid_adj_ids:
                if aid not in seen_ids:
                    seen_ids.add(aid)
                    wrapped = wrap_document_as_retrieval_result(
                        self.corpus_lookup[aid],
                        rank=len(initial_pool) + len(added_results) + 1,
                    )
                    added_results.append(wrapped)
                    adjacent_added = True

        # Check if adjacent chunks alone are sufficient
        adjacent_is_sufficient = False
        if adjacent_added:
            if self.evaluator is not None and self.router is not None:
                current_pool = initial_pool + added_results
                for cand in current_pool:
                    pred = self.evaluator.evaluate_pair(clean_query, cand.document)
                    dec = self.router.route_candidate(pred)
                    if dec.action == CRAGAction.TRUST:
                        adjacent_is_sufficient = True
                        break
                if not adjacent_is_sufficient and hasattr(self.evaluator, "evaluate_collection"):
                    coll_pred = self.evaluator.evaluate_collection(clean_query, [c.document for c in current_pool])
                    coll_dec = self.router.route_candidate(coll_pred)
                    if coll_dec.action == CRAGAction.TRUST:
                        adjacent_is_sufficient = True

                if adjacent_is_sufficient:
                    strategy_used = "adjacent_chunk"
                    is_sufficient = True
                    re_eval_action = CRAGAction.TRUST
            else:
                strategy_used = "adjacent_chunk"
                adjacent_is_sufficient = True

        # Step B & C: If adjacent chunks were unavailable OR resulting evidence remains insufficient
        needs_secondary = (not adjacent_added) or (not adjacent_is_sufficient)
        if needs_secondary:
            current_docs = [r.document for r in (initial_pool + added_results)]
            targeted_query = extract_missing_information_query(
                clean_query, current_docs, query_generator=self.query_generator
            )

            fetch_depth = len(initial_pool) + len(added_results) + self.secondary_top_k
            retrieved = self.retriever.retrieve(targeted_query, top_k=fetch_depth)

            rejected_from_targeted: List[str] = []
            for res in retrieved:
                d_id = res.document.doc_id
                if d_id not in seen_ids:
                    seen_ids.add(d_id)
                    # Step D: Re-evaluate newly retrieved candidate; do not trust irrelevant candidates
                    if self.evaluator is not None and self.router is not None:
                        pred = self.evaluator.evaluate_pair(clean_query, res.document)
                        cand_dec = self.router.route_candidate(pred)
                        if cand_dec.action == CRAGAction.REJECT_SEARCH:
                            rejected_from_targeted.append(d_id)
                            continue
                    added_results.append(res.model_copy(deep=True))

            strategy_used = "secondary_retrieval"

        # Final re-evaluation of merged pool
        merged = initial_pool + added_results
        if self.evaluator is not None and self.router is not None:
            has_trusted = False
            for cand in merged:
                pred = self.evaluator.evaluate_pair(clean_query, cand.document)
                dec = self.router.route_candidate(pred)
                if dec.action == CRAGAction.TRUST:
                    has_trusted = True
                    break
            if not has_trusted and hasattr(self.evaluator, "evaluate_collection"):
                coll_pred = self.evaluator.evaluate_collection(clean_query, [c.document for c in merged])
                coll_dec = self.router.route_candidate(coll_pred)
                if coll_dec.action == CRAGAction.TRUST:
                    has_trusted = True

            is_sufficient = has_trusted
            re_eval_action = CRAGAction.TRUST if has_trusted else CRAGAction.VERIFY_MORE

        added_ids = [r.document.doc_id for r in added_results]

        return RefinementResult(
            query_text=clean_query,
            original_doc_ids=target_ids,
            refined_results=merged,
            refined_documents=[r.document for r in merged],
            refinement_strategy=strategy_used,
            is_expanded=len(added_results) > 0,
            skipped=False,
            added_doc_ids=added_ids,
            added_results=added_results,
            targeted_query=targeted_query,
            is_sufficient=is_sufficient,
            re_evaluation_action=re_eval_action,
            metadata={
                "refiner_type": "SecondaryRetrievalRefiner",
                "adjacent_metadata_found": adjacent_added,
                "adjacent_is_sufficient": adjacent_is_sufficient if adjacent_added else False,
                "secondary_retrieval_executed": needs_secondary,
                "targeted_query": targeted_query,
                "is_sufficient": is_sufficient,
                "re_evaluation_action": re_eval_action.value if re_eval_action else None,
                "initial_count": len(initial_pool),
                "added_count": len(added_results),
                "final_count": len(merged),
            },
        )

