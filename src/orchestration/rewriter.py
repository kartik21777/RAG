"""
Query rewriting module for Semantic- and Evidence-Aware CRAG (Milestone 4).

When initial candidate documents are rejected (REJECT_SEARCH action) or empty,
the query rewriter reformulates the query to explore alternative terminology,
strip conversational filler, or expand technical abbreviations.

Includes cycle prevention to eliminate repetitive query loops and bounded attempt
budgeting to guarantee pipeline termination.
"""

from abc import ABC, abstractmethod
import re
import string
from typing import Any, Dict, List, Optional, Set

from src.core.schemas import RetrievalResult
from src.orchestration.schemas import RewriteResult


DEFAULT_FILLER_WORDS: Set[str] = {
    "a",
    "about",
    "an",
    "are",
    "can",
    "could",
    "define",
    "definition",
    "describe",
    "detail",
    "details",
    "do",
    "does",
    "explain",
    "for",
    "give",
    "how",
    "in",
    "is",
    "me",
    "of",
    "overview",
    "please",
    "provide",
    "summarize",
    "summary",
    "tell",
    "the",
    "to",
    "what",
    "which",
    "who",
    "why",
    "with",
    "you",
}

DEFAULT_SYNONYM_MAP: Dict[str, str] = {
    "rrf": "reciprocal rank fusion",
    "bm25": "bm25 sparse retrieval",
    "faiss": "faiss vector index",
    "crag": "corrective retrieval augmented generation",
    "dpr": "dense passage retrieval",
}


class BaseQueryRewriter(ABC):
    """
    Abstract interface for query rewriting in Corrective RAG.
    """

    @abstractmethod
    def rewrite(
        self,
        query: str,
        history: Optional[List[str]] = None,
        rejected_candidates: Optional[List[RetrievalResult]] = None,
        original_query: Optional[str] = None,
    ) -> RewriteResult:
        """
        Reformulates an unanswerable or rejected search query.

        Contract: Option 2 (Rewrite-Only History)
        - `query`: The current query to be rewritten (original user query or previous rewrite).
        - `history`: List of previously produced rewritten queries ONLY.
          Never contains the original query merely because it initiated the session.
          Empty or None indicates no previous rewrite attempts have occurred.
        - `attempt_index`: Directly computed as len(valid_history) + 1.
        - `original_query`: Optional root user query to protect against reintroduction
          during chained/iterative rewriting cycles. Defaults to `query` if omitted.

        Args:
            query: The current query string to rewrite.
            history: Optional list of previously rewritten queries in this session.
            rejected_candidates: Optional candidate results rejected by the router.
            original_query: Optional root user query to prevent cyclic reintroduction.

        Returns:
            RewriteResult containing original query, rewritten query, strategy,
            1-indexed rewrite attempt counter, and terminal cycle/budget flags.
        """
        pass

    @staticmethod
    def validate_query(query: Any) -> str:
        """
        Validates that the query is a non-empty, non-whitespace string.
        """
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query cannot be empty or whitespace only.")
        return cleaned

    @staticmethod
    def normalize_query(query: str) -> str:
        """
        Normalizes a query string for whitespace- and case-insensitive comparison.
        """
        cleaned = re.sub(r"\s+", " ", query.strip().lower())
        return cleaned

    @classmethod
    def is_in_history(cls, candidate: str, history: Optional[List[str]]) -> bool:
        """
        Checks whether candidate query matches any previously attempted query in history.
        """
        if not history:
            return False
        norm_candidate = cls.normalize_query(candidate)
        for prior in history:
            if isinstance(prior, str) and cls.normalize_query(prior) == norm_candidate:
                return True
        return False

    @classmethod
    def compute_attempt_index(
        cls,
        history_or_query: Any = None,
        history: Optional[List[str]] = None,
        query: Any = None,
    ) -> int:
        """
        Computes the 1-indexed rewrite attempt counter under Contract Option 2 (Rewrite-Only History).
        Since history contains only previously produced rewritten queries:
        attempt_index = len(valid_history) + 1.

        Supports compute_attempt_index(history), compute_attempt_index(query, history),
        and compute_attempt_index(history, query=...).
        """
        if history is not None:
            target = history
        elif isinstance(history_or_query, list):
            target = history_or_query
        else:
            target = None
        if not target:
            return 1
        valid_history = [h for h in target if isinstance(h, str)]
        return len(valid_history) + 1


class MockQueryRewriter(BaseQueryRewriter):
    """
    Deterministic, configurable test double for query rewriting.
    Supports predefined mapping, sequential rewrite progressions, and budget enforcement
    without requiring external LLM API calls or network connections.
    """

    def __init__(
        self,
        rewrite_map: Optional[Dict[str, str]] = None,
        sequential_rewrites: Optional[List[str]] = None,
        default_rewrite: Optional[str] = None,
        max_attempts: int = 2,
    ):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer >= 1.")

        self.rewrite_map = rewrite_map or {}
        self.sequential_rewrites = sequential_rewrites or []
        self.default_rewrite = default_rewrite
        self.max_attempts = max_attempts

    def rewrite(
        self,
        query: str,
        history: Optional[List[str]] = None,
        rejected_candidates: Optional[List[RetrievalResult]] = None,
        original_query: Optional[str] = None,
    ) -> RewriteResult:
        clean_query = self.validate_query(query)
        clean_original = self.validate_query(original_query) if original_query is not None else clean_query

        # Rewrite-Only History Contract: history contains only previous rewrites
        valid_history = [h for h in (history or []) if isinstance(h, str)]
        attempt_index = self.compute_attempt_index(valid_history)

        # Budget guard: if attempt_index exceeds allowed budget, terminate
        if attempt_index > self.max_attempts:
            return RewriteResult(
                original_query=clean_original,
                rewritten_query=clean_query,
                strategy="terminal_budget_exhausted",
                attempt_index=attempt_index,
                is_terminal=True,
                metadata={
                    "reason": f"Maximum rewriting budget of {self.max_attempts} attempts exceeded.",
                    "max_attempts": self.max_attempts,
                    "attempt_index": attempt_index,
                    "history": valid_history,
                },
            )

        # Determine candidate rewrite
        candidate_rewrite: Optional[str] = None
        strategy = "mock"

        if clean_query in self.rewrite_map:
            candidate_rewrite = self.rewrite_map[clean_query]
            strategy = "mock_mapped"
        elif self.sequential_rewrites:
            idx = attempt_index - 1
            if idx < len(self.sequential_rewrites):
                candidate_rewrite = self.sequential_rewrites[idx]
                strategy = f"mock_sequential_step_{attempt_index}"
            else:
                candidate_rewrite = self.sequential_rewrites[-1]
                strategy = "mock_sequential_exhausted"
        elif self.default_rewrite:
            candidate_rewrite = self.default_rewrite
            strategy = "mock_default"
        else:
            candidate_rewrite = f"{clean_query} reformulated"
            strategy = "mock_suffix"

        # Cycle avoidance: verify candidate was not in prior rewrites, is not no-op,
        # and does not reintroduce the root original user query.
        norm_candidate = self.normalize_query(candidate_rewrite)
        is_cycle = (
            self.is_in_history(candidate_rewrite, valid_history)
            or norm_candidate == self.normalize_query(clean_query)
            or norm_candidate == self.normalize_query(clean_original)
        )

        if is_cycle:
            return RewriteResult(
                original_query=clean_original,
                rewritten_query=clean_query,
                strategy="terminal_cycle_avoided",
                attempt_index=attempt_index,
                is_terminal=True,
                metadata={
                    "reason": f"Rewriting cycle detected: candidate '{candidate_rewrite}' already in history, matches current query, or reintroduces original query.",
                    "candidate_rewrite": candidate_rewrite,
                    "history": valid_history,
                    "original_query": clean_original,
                },
            )

        is_terminal = (attempt_index >= self.max_attempts)
        return RewriteResult(
            original_query=clean_original,
            rewritten_query=candidate_rewrite,
            strategy=strategy,
            attempt_index=attempt_index,
            is_terminal=is_terminal,
            metadata={
                "max_attempts": self.max_attempts,
                "history_length": len(valid_history),
                "rejected_count": len(rejected_candidates or []),
            },
        )


class RuleBasedQueryRewriter(BaseQueryRewriter):
    """
    Deterministic rule-based query rewriter for CRAG without live LLM dependencies.
    Applies multi-stage heuristic transformations:
      - Attempt 1: Keyword focus (strips conversational filler, punctuation, and question affixes).
      - Attempt 2: Domain synonym & technical abbreviation expansion.
      - Subsequent attempts / fallbacks: Term disambiguation appending domain context.
    Strictly prevents query cycles and enforces the configured max attempt budget.
    """

    def __init__(
        self,
        max_attempts: int = 2,
        filler_words: Optional[Set[str]] = None,
        synonym_map: Optional[Dict[str, str]] = None,
    ):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer >= 1.")

        self.max_attempts = max_attempts
        self.filler_words = filler_words if filler_words is not None else DEFAULT_FILLER_WORDS
        self.synonym_map = synonym_map if synonym_map is not None else DEFAULT_SYNONYM_MAP

    def _strip_filler_and_punct(self, query: str) -> str:
        """Strips punctuation and filler stop words, preserving core technical terms."""
        # Replace punctuation with spaces
        no_punct = "".join(c if c not in string.punctuation else " " for c in query.lower())
        tokens = [t for t in no_punct.split() if t]
        retained = [t for t in tokens if t not in self.filler_words]
        return " ".join(retained)

    def _expand_synonyms(self, query: str) -> str:
        """Replaces known domain abbreviations with their full terminology."""
        tokens = query.lower().split()
        expanded: List[str] = []
        replaced = False
        for t in tokens:
            cleaned_token = t.strip(string.punctuation)
            if cleaned_token in self.synonym_map:
                expanded.append(self.synonym_map[cleaned_token])
                replaced = True
            else:
                expanded.append(t)

        if replaced:
            return " ".join(expanded)
        return query

    def rewrite(
        self,
        query: str,
        history: Optional[List[str]] = None,
        rejected_candidates: Optional[List[RetrievalResult]] = None,
        original_query: Optional[str] = None,
    ) -> RewriteResult:
        clean_query = self.validate_query(query)
        clean_original = self.validate_query(original_query) if original_query is not None else clean_query

        valid_history = [h for h in (history or []) if isinstance(h, str)]
        attempt_index = self.compute_attempt_index(valid_history)

        # 1. Budget guard
        if attempt_index > self.max_attempts:
            return RewriteResult(
                original_query=clean_original,
                rewritten_query=clean_query,
                strategy="terminal_budget_exhausted",
                attempt_index=attempt_index,
                is_terminal=True,
                metadata={
                    "reason": f"Maximum rewriting budget of {self.max_attempts} attempts exceeded.",
                    "max_attempts": self.max_attempts,
                    "attempt_index": attempt_index,
                    "history": valid_history,
                },
            )

        candidate_rewrite: Optional[str] = None
        strategy = "rule_based"

        def is_valid_candidate(c: Optional[str]) -> bool:
            if not c:
                return False
            norm_c = self.normalize_query(c)
            return (
                not self.is_in_history(c, valid_history)
                and norm_c != self.normalize_query(clean_query)
                and norm_c != self.normalize_query(clean_original)
            )

        # 2. Stage-based rewriting progression
        if attempt_index == 1:
            # Stage 1: Keyword focus (strip question phrasing and fillers)
            simplified = self._strip_filler_and_punct(clean_query)
            if is_valid_candidate(simplified):
                candidate_rewrite = simplified
                strategy = "keyword_simplification"
            else:
                # Fall back to synonym expansion if keyword stripping didn't change query
                expanded = self._expand_synonyms(clean_query)
                if is_valid_candidate(expanded):
                    candidate_rewrite = expanded
                    strategy = "synonym_expansion"
                else:
                    candidate_rewrite = f"{clean_query} information retrieval"
                    strategy = "domain_disambiguation"

        elif attempt_index == 2:
            # Stage 2: Try synonym expansion
            expanded = self._expand_synonyms(clean_query)
            if is_valid_candidate(expanded):
                candidate_rewrite = expanded
                strategy = "synonym_expansion"
            else:
                # Fall back to stripped keywords with domain context
                simplified = self._strip_filler_and_punct(clean_query)
                base_text = simplified if simplified else clean_query
                contextual = f"{base_text} core principles"
                if is_valid_candidate(contextual):
                    candidate_rewrite = contextual
                    strategy = "domain_disambiguation"
                else:
                    candidate_rewrite = f"{base_text} implementation"
                    strategy = "domain_disambiguation"

        else:
            # Stage 3+: Disambiguate with attempt-indexed context
            simplified = self._strip_filler_and_punct(clean_query)
            base_text = simplified if simplified else clean_query
            candidate_rewrite = f"{base_text} technical context step {attempt_index}"
            strategy = "domain_disambiguation"

        # 3. Cycle guard check on final candidate
        norm_candidate = self.normalize_query(candidate_rewrite) if candidate_rewrite else ""
        is_cycle = (
            not candidate_rewrite
            or self.is_in_history(candidate_rewrite, valid_history)
            or norm_candidate == self.normalize_query(clean_query)
            or norm_candidate == self.normalize_query(clean_original)
        )

        if is_cycle:
            return RewriteResult(
                original_query=clean_original,
                rewritten_query=clean_query,
                strategy="terminal_cycle_avoided",
                attempt_index=attempt_index,
                is_terminal=True,
                metadata={
                    "reason": f"Rewriting cycle detected: '{candidate_rewrite}' already in history, matches current query, or reintroduces original.",
                    "candidate_rewrite": candidate_rewrite,
                    "history": valid_history,
                    "original_query": clean_original,
                },
            )

        is_terminal = (attempt_index >= self.max_attempts)
        return RewriteResult(
            original_query=clean_original,
            rewritten_query=candidate_rewrite,
            strategy=strategy,
            attempt_index=attempt_index,
            is_terminal=is_terminal,
            metadata={
                "max_attempts": self.max_attempts,
                "history_length": len(valid_history),
                "rejected_count": len(rejected_candidates or []),
            },
        )
