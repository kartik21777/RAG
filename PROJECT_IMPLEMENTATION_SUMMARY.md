# CRAG (Corrective Retrieval-Augmented Generation) — Complete System Implementation Summary

**Project Repository**: `c:/RAG`  
**Current Test Status**: 214 tests passing (`pytest` in ~60s)  
**Branch / Working State**: All core retrieval, synthetic dataset generation, and CRAG orchestration modules implemented and verified.

---

## 1. High-Level Architecture Overview

The system implements a modular Corrective Retrieval-Augmented Generation (CRAG) architecture designed to minimize hallucinations, prevent ungrounded responses, and dynamically correct retrieval deficits.

```
                         [User Query]
                              │
                              ▼
                   [CRAGRetrieverFacade]
               (Hybrid: BM25-Okapi + FAISS Dense)
                              │
                      Initial Candidates
                              │
                              ▼
                  [Runtime Evaluator (DeBERTa)]
              (Evaluates Intent-Fit & Sufficiency)
                              │
                              ▼
                 [CorrectiveDecisionRouter]
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
         [TRUST]        [VERIFY_MORE]    [REJECT_SEARCH]
            │                 │                 │
            │          [EvidenceRefiner] [QueryRewriter]
            │                 │                 │
            │          - Adjacent Chunks        │
            │          - Targeted Search        │
            │          - Re-evaluation          │
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                              ▼
                 [DeterministicGroundedGenerator]
             - Grounded factual generation (TRUST)
             - Safe abstention (INSUFFICIENT_EVIDENCE)
             - Web fallback / refusal (REJECT_SEARCH)
```

---

## 2. Core Subsystems & Milestone Breakdown

### Phase 1: Retrieval Foundation & Corpus (Member 1 Workstream)

| Component | File | Key Features & Implementation Details |
|---|---|---|
| **Schemas & Data Models** | [schemas.py](file:///c:/RAG/src/core/schemas.py) | Pydantic v2 schemas: `Document`, `Query`, `RetrievalResult`, `EvaluatorExample`, `CRAGAction` (`TRUST`, `VERIFY_MORE`, `REJECT_SEARCH`). Strict validation preventing prohibited $(0, 1)$ labels. |
| **Curated Technical Corpus** | `data/processed/corpus.jsonl` | 50 diverse technical passages across IR, vector indexing, Reciprocal Rank Fusion, CRAG routing, distributed consensus, and quantum search. |
| **Sparse BM25 Retriever** | [sparse.py](file:///c:/RAG/src/retrieval/sparse.py) | Pure-Python `bm25s` backend ($k_1=1.5, b=0.75$), disk persistence (`save`/`load`), zero JVM dependency, duplicate ID guards. |
| **Dense Semantic Retriever** | [dense.py](file:///c:/RAG/src/retrieval/dense.py) | SentenceTransformers (`all-MiniLM-L6-v2`), unit $L_2$ normalized embeddings, FAISS `IndexFlatIP` for exact cosine similarity, persistence roundtrip. |
| **Hybrid Retriever (RRF)** | [hybrid.py](file:///c:/RAG/src/retrieval/hybrid.py) | Reciprocal Rank Fusion ($k=60$), score normalization, candidate deduplication, consensus rank boosting, weighted sparse/dense fusion. |
| **Unified Retrieval Facade** | [facade.py](file:///c:/RAG/src/retrieval/facade.py) | `CRAGRetrieverFacade` and `create_crag_retriever()` factory method. Encapsulates sparse + dense + RRF retrieval with unified `RetrievalResult` output. |
| **Reproducible Scripts** | `scripts/` | `prepare_corpus.py` (re-creates corpus), `build_indexes.py` (builds BM25 + FAISS index artifacts), `generate_evaluator_datasets.py` (CLI dataset generation). |

---

### Phase 2: Synthetic Data & Evaluator Pipeline

| Component | File | Key Features & Implementation Details |
|---|---|---|
| **LLM Judge & Prompting** | [judge.py](file:///c:/RAG/src/synthetic_data/judge.py) | `LLMEvaluatorJudge` (structured JSON extraction with bounded retries, injection-safe delimiters) and `MockEvaluatorJudge` for deterministic offline testing. |
| **Negative Sampling** | [pipeline.py](file:///c:/RAG/src/synthetic_data/pipeline.py) | Random negative and hard negative sampling with mandatory relevance verification (guards against mislabeling relevant documents as negatives). |
| **Quality Gate Validator** | [validator.py](file:///c:/RAG/src/synthetic_data/validator.py) | Asserts label distributions, schema compliance, prohibited label prevention, and semantic diversity. |
| **Generated Datasets** | `data/synthetic/` | `evaluator_train.jsonl` (160 records) and `evaluator_val.jsonl` (40 records) for multi-task DeBERTa-v3 training. |

---

### Phase 3: CRAG Orchestration Stack (Member 3 Workstream)

| Component | File | Key Features & Implementation Details |
|---|---|---|
| **Orchestration Schemas** | [schemas.py](file:///c:/RAG/src/orchestration/schemas.py) | Defines `EvaluatorPrediction`, `CandidateDecision`, `QueryDecision`, `RefinementResult`, `RewriteResult`, `OrchestratorResult`, `GroundedResponse`, and `GenerationStatus`. |
| **Runtime Evaluator** | [evaluator.py](file:///c:/RAG/src/orchestration/evaluator.py) | `BaseRuntimeEvaluator` (abstract) and `MockRuntimeEvaluator`. Features `evaluate_collection` to combine multi-document context into composite candidates for joint sufficiency evaluation. |
| **Corrective Decision Router** | [router.py](file:///c:/RAG/src/orchestration/router.py) | `CorrectiveDecisionRouter` implements CRAG intent dominance: candidate-level routing (`TRUST` if intent $\ge \tau_i$ and suff $\ge \tau_s$; `VERIFY_MORE` if intent $\ge \tau_i$ and suff $< \tau_s$; else `REJECT_SEARCH`), plus query-level aggregation with strict duplicate detection. |
| **Query Rewriter** | [rewriter.py](file:///c:/RAG/src/orchestration/rewriter.py) | `BaseQueryRewriter`, `MockQueryRewriter`, and `RuleBasedQueryRewriter`. Strict rewrite-only history contract, transformation rules (stopword pruning, entity focus, question expansion), cycle guards, and step tracking. |
| **CRAG Pipeline Orchestrator** | [orchestrator.py](file:///c:/RAG/src/orchestration/orchestrator.py) | `CRAGOrchestrator` coordinates the full loop: initial retrieval -> evaluation & routing -> branch handling (`TRUST` direct to generator; `VERIFY_MORE` through refiner; `REJECT_SEARCH` through rewriter + bounded retry loop up to `max_rewrite_attempts`). Captures full audit trails in `OrchestratorResult`. |
| **Grounded Response Generator** | [generator.py](file:///c:/RAG/src/orchestration/generator.py) | `DeterministicGroundedGenerator` with strict safety gates: refuses ungrounded generation, declines factual generation on uncorrected `REJECT_SEARCH`, and cleanly abstains (`INSUFFICIENT_EVIDENCE`) when verification does not find sufficient evidence. Anchors citations to verified document IDs. |

---

### Phase 4: Targeted `VERIFY_MORE` Refinement Enhancement

The `VERIFY_MORE` flow was significantly improved to move beyond naive retrieval depth increases to targeted evidence synthesis and re-validation:

1. **Adjacent/Parent Chunk Expansion**:
   - Inspects document metadata for `adjacent_chunk_ids`, `prev_chunk_id`, and `next_chunk_id`.
   - Fetches adjacent chunks directly from `corpus_lookup` without redundant index queries.
   - Re-evaluates whether adjacent context alone satisfies `TRUST`.
2. **Missing Information Query Formulation**:
   - [extract_missing_information_query](file:///c:/RAG/src/orchestration/refiner.py#L42): Inspects current query and retrieved text to target missing entities and unfulfilled facts. Supports an optional pluggable LLM `query_generator` with safe fallback to the original query.
3. **Secondary Targeted Retrieval**:
   - Executes secondary retrieval with dynamic depth: `fetch_depth = len(initial_pool) + len(added_results) + secondary_top_k`.
   - Filters out irrelevant (`REJECT_SEARCH`) candidates so low-quality secondary results do not pollute the pool.
4. **Multi-Document Joint Re-Evaluation**:
   - Employs `evaluate_collection` ([evaluator.py L48](file:///c:/RAG/src/orchestration/evaluator.py#L48)) to evaluate whether the merged pool collectively satisfies information requirements, enabling promotion to `TRUST`.
5. **Safe Abstention & Error Propagation**:
   - If secondary retrieval still cannot confirm sufficiency, sets `is_sufficient=False`, which triggers generator status `GenerationStatus.INSUFFICIENT_EVIDENCE`.
   - Upstream retriever exceptions (`RuntimeError`) propagate without infinite retry loops or silent swallowing.

---

## 3. Test Suite & Verification Matrix

The test suite contains **214 passing tests** across 15 test modules:

```text
======================= 214 passed in 60.08s =======================
```

| Test File | Tests | Focus Area |
|---|:---:|---|
| [test_environment.py](file:///c:/RAG/tests/test_environment.py) | 6 | Dependencies, PyTorch, FAISS CPU, BM25s, SentenceTransformers |
| [test_schemas.py](file:///c:/RAG/tests/test_schemas.py) | 8 | Core schemas, validation, forbidden $(0, 1)$ constraint |
| [test_dataset_io.py](file:///c:/RAG/tests/test_dataset_io.py) | 6 | JSONL loading/saving, schema deserialization roundtrips |
| [test_sparse.py](file:///c:/RAG/tests/test_sparse.py) | 16 | BM25 indexing, top-k scoring, empty queries, persistence |
| [test_dense.py](file:///c:/RAG/tests/test_dense.py) | 16 | Embeddings, FAISS IndexFlatIP, cosine normalization, persistence |
| [test_hybrid.py](file:///c:/RAG/tests/test_hybrid.py) | 18 | Reciprocal Rank Fusion, weight factors, consensus boosting |
| [test_retrieval_facade.py](file:///c:/RAG/tests/test_retrieval_facade.py) | 4 | Facade retrieval, factory function, save/load lifecycle |
| [test_synthetic_data.py](file:///c:/RAG/tests/test_synthetic_data.py) | 24 | Synthetic pipeline, negative sampling, LLM judging & validation |
| [test_orchestration_evaluator.py](file:///c:/RAG/tests/test_orchestration_evaluator.py) | 12 | Base & Mock evaluator contracts, input guards, collection eval |
| [test_orchestration_router.py](file:///c:/RAG/tests/test_orchestration_router.py) | 18 | CRAG decision rules, boundary thresholds, candidate & query aggregation |
| [test_orchestration_refiner.py](file:///c:/RAG/tests/test_orchestration_refiner.py) | 18 | Base/Secondary refiner, adjacent expansion, deduplication |
| [test_orchestration_rewriter.py](file:///c:/RAG/tests/test_orchestration_rewriter.py) | 16 | Rewriter contract, rule-based transforms, rewrite-only history |
| [test_orchestration_orchestrator.py](file:///c:/RAG/tests/test_orchestration_orchestrator.py) | 22 | CRAG loop state machine, rewrite attempt limits, cycle guards |
| [test_orchestration_generator.py](file:///c:/RAG/tests/test_orchestration_generator.py) | 19 | Grounded generation, citation anchors, safety gates, abstention |
| [test_targeted_verify_more.py](file:///c:/RAG/tests/test_targeted_verify_more.py) | 12 | 12 targeted verification scenarios (adjacent, query gen, re-eval, safe termination) |

---

## 4. Audit Findings & Teammate Integration Notes

An independent code audit identified the following key areas for teammate handoff:

1. **Application Bootstrap (Production Entry Point)**:
   - The codebase is currently structured as a modular library with extensive unit/integration tests and offline CLI scripts (`prepare_corpus.py`, `build_indexes.py`, `generate_evaluator_datasets.py`).
   - An application entry point (`main.py` or FastAPI server) should be added to instantiate `CRAGOrchestrator` with a live LLM `query_generator` and production generator.
2. **DeBERTa-v3 Evaluator (Member 2)**:
   - When plugging in the trained DeBERTa-v3 model, `LearnedRuntimeEvaluator` must implement token-aware truncation in `evaluate_collection` to respect DeBERTa's 512-token sequence limit when concatenating multiple chunks.
3. **External Web Search (Member 3 / Future Scope)**:
   - When `REJECT_SEARCH` is triggered and internal query rewrites exhaust their budget, the rewriter output is structured to be handed off to an external web search provider (e.g. Google Search API / Tavily) before final refusal.
