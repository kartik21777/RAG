"""
Unit tests for query rewriter components in src/orchestration/rewriter.py (Milestone 4).

Verifies BaseQueryRewriter, MockQueryRewriter, RuleBasedQueryRewriter,
RewriteResult schema validation, keyword simplification, abbreviation expansion,
cycle prevention, and bounded attempt budgeting.
"""

import pytest

from src.core.schemas import Document, RetrievalResult
from src.orchestration.rewriter import (
    BaseQueryRewriter,
    MockQueryRewriter,
    RuleBasedQueryRewriter,
)
from src.orchestration.schemas import RewriteResult


def make_dummy_result(doc_id: str = "doc_1") -> RetrievalResult:
    return RetrievalResult(
        document=Document(doc_id=doc_id, text="Dummy candidate text", metadata={}),
        fused_score=0.015,
        fused_rank=1,
    )


# -----------------------------------------------------------------------------
# 1. BaseQueryRewriter & Input Validation Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_base_rewriter_validation():
    # Valid string trimmed
    assert BaseQueryRewriter.validate_query("  dense retrieval  ") == "dense retrieval"

    # Empty string rejected
    with pytest.raises(ValueError) as exc:
        BaseQueryRewriter.validate_query("")
    assert "query cannot be empty" in str(exc.value)

    # Whitespace only rejected
    with pytest.raises(ValueError) as exc:
        BaseQueryRewriter.validate_query("   \t\n  ")
    assert "query cannot be empty" in str(exc.value)

    # Non-string types rejected
    with pytest.raises(ValueError) as exc:
        BaseQueryRewriter.validate_query(123)  # type: ignore
    assert "query must be a string" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        BaseQueryRewriter.validate_query(None)  # type: ignore
    assert "query must be a string" in str(exc.value)


@pytest.mark.unit
def test_base_rewriter_normalization_and_history_check():
    assert BaseQueryRewriter.normalize_query("  What   is  RRF? \n ") == "what is rrf?"

    history = ["what is rrf?", "reciprocal rank fusion"]
    assert BaseQueryRewriter.is_in_history("What is RRF?", history) is True
    assert BaseQueryRewriter.is_in_history("  RECIPROCAL   rank fusion  ", history) is True
    assert BaseQueryRewriter.is_in_history("bm25 ranking", history) is False
    assert BaseQueryRewriter.is_in_history("anything", None) is False


@pytest.mark.unit
def test_compute_attempt_index_helper():
    # Empty history produces attempt 1
    assert BaseQueryRewriter.compute_attempt_index(None) == 1
    assert BaseQueryRewriter.compute_attempt_index([]) == 1

    # One previous rewrite produces attempt 2
    assert BaseQueryRewriter.compute_attempt_index(["rewrite_1"]) == 2

    # Two previous rewrites produce attempt 3
    assert BaseQueryRewriter.compute_attempt_index(["rewrite_1", "rewrite_2"]) == 3

    # Duplicate history entries follow documented contract: len(valid_history) + 1
    assert BaseQueryRewriter.compute_attempt_index(["rewrite_1", "rewrite_1"]) == 3

    # Current query matching last rewritten query does NOT reset attempt count
    assert BaseQueryRewriter.compute_attempt_index(["rewrite_1"], query="rewrite_1") == 2
    assert BaseQueryRewriter.compute_attempt_index("rewrite_1", ["rewrite_1"]) == 2


# -----------------------------------------------------------------------------
# 2. MockQueryRewriter Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_mock_rewriter_initialization_bounds():
    # Zero or negative rejected
    with pytest.raises(ValueError):
        MockQueryRewriter(max_attempts=0)

    with pytest.raises(ValueError):
        MockQueryRewriter(max_attempts=-1)

    # Boolean rejected
    with pytest.raises(ValueError):
        MockQueryRewriter(max_attempts=True)  # type: ignore


@pytest.mark.unit
def test_mock_rewriter_mapped_mode():
    rewriter = MockQueryRewriter(
        rewrite_map={"what is rrf?": "reciprocal rank fusion algorithm"},
        max_attempts=2,
    )
    res = rewriter.rewrite("what is rrf?")

    assert isinstance(res, RewriteResult)
    assert res.original_query == "what is rrf?"
    assert res.rewritten_query == "reciprocal rank fusion algorithm"
    assert res.strategy == "mock_mapped"
    assert res.attempt_index == 1
    assert res.is_terminal is False


@pytest.mark.unit
def test_mock_rewriter_sequential_progression():
    seq = ["query step 1", "query step 2"]
    rewriter = MockQueryRewriter(sequential_rewrites=seq, max_attempts=2)

    # Attempt 1: empty history produces attempt 1
    res1 = rewriter.rewrite("initial query", history=[])
    assert res1.rewritten_query == "query step 1"
    assert res1.attempt_index == 1
    assert res1.is_terminal is False

    # Attempt 2: one previous rewrite produces attempt 2
    res2 = rewriter.rewrite("initial query", history=["query step 1"])
    assert res2.rewritten_query == "query step 2"
    assert res2.attempt_index == 2
    assert res2.is_terminal is True  # Reached max_attempts=2

    # Attempt 3: two previous rewrites produce attempt 3 -> budget exhausted
    res3 = rewriter.rewrite("initial query", history=["query step 1", "query step 2"])
    assert res3.attempt_index == 3
    assert res3.is_terminal is True
    assert res3.strategy == "terminal_budget_exhausted"


@pytest.mark.unit
def test_current_query_matching_last_rewrite_does_not_reset_attempt_count():
    # In chained rewriting, current query is the last rewrite
    rewriter = MockQueryRewriter(
        sequential_rewrites=["step 1", "step 2", "step 3"],
        max_attempts=3,
    )
    # Rewrite 1 produced "step 1". Now current query is "step 1" and history is ["step 1"]
    res = rewriter.rewrite(query="step 1", history=["step 1"])
    assert res.attempt_index == 2  # Must NOT reset to 1
    assert res.rewritten_query == "step 2"
    assert res.is_terminal is False


@pytest.mark.unit
def test_single_attempt_budget_max_attempts_one():
    # When max_attempts=1, permits exactly one rewrite operation
    rewriter = MockQueryRewriter(default_rewrite="reformulated query", max_attempts=1)

    # First rewrite with empty history -> permitted, attempt_index=1, terminal because max_attempts=1
    res1 = rewriter.rewrite("original query", history=[])
    assert res1.attempt_index == 1
    assert res1.rewritten_query == "reformulated query"
    assert res1.strategy == "mock_default"
    assert res1.is_terminal is True

    # Second rewrite with prior rewrite in history -> rejected, attempt_index=2 > max 1
    res2 = rewriter.rewrite("original query", history=["reformulated query"])
    assert res2.attempt_index == 2
    assert res2.is_terminal is True
    assert res2.strategy == "terminal_budget_exhausted"
    assert res2.rewritten_query == "original query"


@pytest.mark.unit
def test_mock_rewriter_budget_exhaustion():
    rewriter = MockQueryRewriter(
        sequential_rewrites=["step 1", "step 2"],
        max_attempts=2,
    )

    # Caller passes history with 2 prior attempts, exceeding max_attempts
    res = rewriter.rewrite("initial query", history=["q0", "q1"])

    assert res.is_terminal is True
    assert res.strategy == "terminal_budget_exhausted"
    assert res.rewritten_query == "initial query"
    assert res.attempt_index == 3
    assert "budget of 2 attempts exceeded" in res.metadata["reason"]


@pytest.mark.unit
def test_cycle_prevention_and_original_query_protection():
    # 1. No-op rewrite (candidate == current query) is prevented
    rewriter_noop = MockQueryRewriter(rewrite_map={"same query": "same query"})
    res_noop = rewriter_noop.rewrite("same query")
    assert res_noop.is_terminal is True
    assert res_noop.strategy == "terminal_cycle_avoided"

    # 2. Candidate matching previous rewrite in history is prevented
    rewriter_hist = MockQueryRewriter(rewrite_map={"query": "step 1"})
    res_hist = rewriter_hist.rewrite("query", history=["step 1"])
    assert res_hist.is_terminal is True
    assert res_hist.strategy == "terminal_cycle_avoided"

    # 3. Reintroduction of original root user query during chained rewrite is prevented
    rewriter_orig = MockQueryRewriter(
        rewrite_map={"step 2": "root user query"},
        max_attempts=3,
    )
    res_orig = rewriter_orig.rewrite(
        query="step 2",
        history=["step 1", "step 2"],
        original_query="root user query",
    )
    assert res_orig.is_terminal is True
    assert res_orig.strategy == "terminal_cycle_avoided"
    assert res_orig.original_query == "root user query"


# -----------------------------------------------------------------------------
# 3. RuleBasedQueryRewriter Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_rule_based_rewriter_initialization():
    with pytest.raises(ValueError):
        RuleBasedQueryRewriter(max_attempts=0)

    with pytest.raises(ValueError):
        RuleBasedQueryRewriter(max_attempts=False)  # type: ignore


@pytest.mark.unit
def test_rule_based_rewriter_keyword_simplification():
    rewriter = RuleBasedQueryRewriter(max_attempts=2)

    query = "What is the definition of Reciprocal Rank Fusion?"
    res = rewriter.rewrite(query)

    assert res.strategy == "keyword_simplification"
    assert res.attempt_index == 1
    assert res.original_query == query
    assert res.rewritten_query == "reciprocal rank fusion"
    assert res.is_terminal is False


@pytest.mark.unit
def test_rule_based_rewriter_synonym_expansion():
    rewriter = RuleBasedQueryRewriter(max_attempts=2)

    # In attempt 2, synonym expansion triggers (history contains prior rewrite)
    res = rewriter.rewrite(
        query="Explain RRF in search",
        history=["rrf search"],  # attempt 2
    )

    assert res.attempt_index == 2
    assert "reciprocal rank fusion" in res.rewritten_query
    assert res.is_terminal is True  # Reached max_attempts=2


@pytest.mark.unit
def test_rule_based_first_rewrite_and_original_query_protection():
    # Attempt 1 with empty history
    rewriter = RuleBasedQueryRewriter(max_attempts=1)
    query = "What is the definition of Reciprocal Rank Fusion?"
    res1 = rewriter.rewrite(query, history=[])

    assert res1.attempt_index == 1
    assert res1.strategy == "keyword_simplification"
    assert res1.rewritten_query == "reciprocal rank fusion"
    assert res1.is_terminal is True  # max_attempts=1 reached on successful rewrite

    # Attempt 2 with previous rewrite in history -> budget exhausted
    res2 = rewriter.rewrite(query, history=["reciprocal rank fusion"])
    assert res2.attempt_index == 2
    assert res2.strategy == "terminal_budget_exhausted"
    assert res2.is_terminal is True

    # Chained rewrite protects against reintroducing root original_query
    rewriter_chained = RuleBasedQueryRewriter(max_attempts=2)
    res3 = rewriter_chained.rewrite(
        query="reciprocal rank fusion",
        history=["reciprocal rank fusion"],
        original_query="reciprocal rank fusion",
    )
    assert res3.is_terminal is True
    assert res3.rewritten_query != "reciprocal rank fusion"
    assert res3.strategy == "domain_disambiguation"


@pytest.mark.unit
def test_rule_based_rewriter_cycle_avoidance():
    rewriter = RuleBasedQueryRewriter(max_attempts=3)

    # When stripped keywords already exist in history
    history = ["bm25"]
    res = rewriter.rewrite("What is BM25?", history=history)

    # Should not produce "bm25", but expand to synonym or disambiguation
    assert res.rewritten_query != "bm25"
    assert not BaseQueryRewriter.is_in_history(res.rewritten_query, history)


@pytest.mark.unit
def test_rule_based_rewriter_budget_exhaustion():
    rewriter = RuleBasedQueryRewriter(max_attempts=2)

    res = rewriter.rewrite(
        query="any technical query",
        history=["attempt 1", "attempt 2"],  # len=2 -> attempt 3 > max 2
    )

    assert res.is_terminal is True
    assert res.strategy == "terminal_budget_exhausted"
    assert res.attempt_index == 3
    assert res.rewritten_query == "any technical query"


@pytest.mark.unit
def test_rule_based_rewriter_preserves_rejected_candidate_context():
    dummy = make_dummy_result("doc_bad")
    rewriter = RuleBasedQueryRewriter(max_attempts=2)

    res = rewriter.rewrite(
        query="How does BM25 score?",
        rejected_candidates=[dummy],
    )

    assert res.metadata["rejected_count"] == 1
    assert res.strategy == "keyword_simplification"
    assert res.rewritten_query == "bm25 score"


# -----------------------------------------------------------------------------
# 4. RewriteResult Schema Tests
# -----------------------------------------------------------------------------

@pytest.mark.unit
def test_rewrite_result_schema_validation():
    # Valid schema creation
    res = RewriteResult(
        original_query="What is BM25?",
        rewritten_query="bm25",
        strategy="keyword_simplification",
        attempt_index=1,
    )
    assert res.is_terminal is False
    assert res.metadata == {}

    # Invalid attempt_index (< 1)
    with pytest.raises(Exception):
        RewriteResult(
            original_query="q",
            rewritten_query="q",
            strategy="test",
            attempt_index=0,
        )

    # Empty original_query
    with pytest.raises(Exception):
        RewriteResult(
            original_query="",
            rewritten_query="q",
            strategy="test",
            attempt_index=1,
        )
