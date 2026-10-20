# Member 1 Handoff Package: Retrieval Foundation & Evaluator Training Data

**Workstream**: Member 1 (Hybrid Sparse + Dense Retrieval & Synthetic Evaluator Training Data)  
**Target Teammates**: 
- **Member 2**: Custom Evaluator Fine-Tuning & Calibration (`DeBERTa-v3`)
- **Member 3**: Corrective RAG Pipeline Orchestration & Generator Integration

---

## 1. Executive Summary & Deliverables

Member 1 has implemented, tested, and validated the core retrieval engine, curated technical corpus, and synthetic training data generation pipeline:
1. **Curated Technical Corpus (`data/processed/corpus.jsonl`)**:
   - 50 diverse technical passages covering Information Retrieval, BM25 saturation, Dense Vector Search, FAISS indexing, Reciprocal Rank Fusion, CRAG corrective routing, Multi-task DeBERTa evaluation, Distributed Consensus, and Quantum Search.
   - **Committed directly to Git** (~23 KB) so teammates can immediately build indexes and generate datasets on a fresh clone.
2. **Hybrid Retrieval Engine & Facade**:
   - `BM25Retriever` (pure Python, zero-Java `bm25s` backend with document length normalization $b=0.75$, term frequency saturation $k_1=1.5$).
   - `DenseRetriever` (`all-MiniLM-L6-v2` dense embeddings, unit L2-normalized, exact cosine search via FAISS `IndexFlatIP`).
   - `HybridRetriever` fusing sparse and dense candidates via Reciprocal Rank Fusion ($k=60$).
   - `CRAGRetrieverFacade`: Unified top-level retrieval interface returning canonical `RetrievalResult` objects with full rank/score provenance.
3. **Reproducible Scripts & Development Datasets**:
   - Standalone CLI scripts in `scripts/`:
     * `scripts/prepare_corpus.py`: Generates/recreates `data/processed/corpus.jsonl`.
     * `scripts/build_indexes.py`: Indexes the corpus and saves BM25 + FAISS artifacts to `indexes/hybrid/`.
     * `scripts/generate_evaluator_datasets.py`: Generates development training (160 examples) and validation (40 examples) datasets into `data/synthetic/`.
   - **Dataset Status & Caveat**: Datasets generated via construction strategy or mock judge are **development/bootstrap datasets** for schema verification, data-loader debugging, and multi-task loss-function integration. They are **not certified ground-truth labels**. For production evaluation, use live LLM judging with configured provider credentials or human-verified sets.
4. **Automated Test Suite**:
   - 98 automated unit and integration tests passing in ~45s (`pytest`), fully independent of local machine paths or pre-existing binary indexes.

---

## 2. Teammate Quickstart: Fresh Clone Reproduction

Any teammate (Member 2 or Member 3) can reproduce all artifacts from a fresh clone using these commands:

### Step 1: Environment Setup
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

### Step 2: (Optional) Re-create or Verify Curated Corpus
The curated corpus is tracked in Git at `data/processed/corpus.jsonl`. If you ever need to re-generate it:
```powershell
python scripts/prepare_corpus.py
```

### Step 3: Build Hybrid Retrieval Indexes
Binary index files are deliberately excluded from Git. Build the fresh FAISS + BM25 hybrid index locally:
```powershell
python scripts/build_indexes.py --corpus-path data/processed/corpus.jsonl --output-dir indexes/hybrid
```

### Step 4: Generate Development Datasets for Member 2
Generate the 160-example train and 40-example val datasets:
```powershell
# Mode A: Construction strategy based (default)
python scripts/generate_evaluator_datasets.py --corpus-path data/processed/corpus.jsonl --output-dir data/synthetic

# Mode B: Offline judged mode (using MockEvaluatorJudge)
python scripts/generate_evaluator_datasets.py --corpus-path data/processed/corpus.jsonl --output-dir data/synthetic --judge mock
```

### Step 5: Run Full Test Suite
Verify environment and contract compatibility:
```powershell
pytest
```

---

## 3. Guide for Member 2: Custom Evaluator Training (`DeBERTa-v3`)

### 3.1 Dataset Locations & Schema Contracts
Member 2 should load training and validation data using the existing dataset utility:

```python
from src.core.dataset import load_evaluator_examples_from_jsonl

train_data = load_evaluator_examples_from_jsonl("data/synthetic/evaluator_train.jsonl")
val_data = load_evaluator_examples_from_jsonl("data/synthetic/evaluator_val.jsonl")

# Each record is an EvaluatorExample:
# - example.query_text: str
# - example.doc_text: str
# - example.intent_fit: int (0 or 1)
# - example.evidence_sufficiency: int (0 or 1)
# - example.crag_action: CRAGAction (TRUST | VERIFY_MORE | REJECT_SEARCH)
```

### 3.2 Model Architecture Requirements
- **Backbone**: `microsoft/deberta-v3-small` or `microsoft/deberta-v3-base`.
- **Input Format**: Single concatenated sequence: `[CLS] query [SEP] document [SEP]`.
- **Dual Classification Heads**:
  - Head 1 ($y_1 \in [0, 1]$): `intent_fit` probability ($\sigma(z_1)$).
  - Head 2 ($y_2 \in [0, 1]$): `evidence_sufficiency` probability ($\sigma(z_2)$).
- **Multi-Task Loss with Constraint Penalty**:
  $$\mathcal{L} = \text{BCE}(y_1, \hat{y}_1) + \text{BCE}(y_2, \hat{y}_2) + \lambda \max(0, \hat{y}_2 - \hat{y}_1)$$
  The penalty term $\lambda \max(0, \hat{y}_2 - \hat{y}_1)$ penalizes the prohibited state where sufficiency exceeds intent-fit.

### 3.3 Evaluation Label Combinations
| Operational Action | $y_1$ (`intent_fit`) | $y_2$ (`evidence_sufficiency`) | Action Meaning |
| :--- | :---: | :---: | :--- |
| **`TRUST`** | `1` | `1` | Retrieved passage directly addresses query intent and contains complete facts. |
| **`VERIFY_MORE`** | `1` | `0` | Retrieved passage is on-topic but partial; trigger supplementary search. |
| **`REJECT_SEARCH`** | `0` | `0` | Off-topic or false positive; rewrite query and search external web. |
| **PROHIBITED** | `0` | `1` | **Invalid**: Passages cannot be insufficient for intent yet provide complete evidence. |

---

## 4. Guide for Member 3: CRAG Pipeline Orchestration

### 4.1 Unified Retrieval Interface
Member 3 can instantiate and query the production hybrid retrieval pipeline with zero boilerplate:

```python
from src.retrieval import create_crag_retriever, CRAGRetrieverFacade

# Option 1: Load pre-built hybrid indexes from disk
retriever = create_crag_retriever(index_dir="indexes/hybrid")

# Option 2: Query the retriever with string or Query object
results = retriever.retrieve("What is Reciprocal Rank Fusion?", top_k=5)

for res in results:
    doc = res.document          # Document(doc_id, text, metadata)
    rrf_score = res.fused_score # Reciprocal Rank Fusion score
    rrf_rank = res.fused_rank   # 1-indexed rank
    bm25_rank = res.sparse_rank # BM25 rank (or None)
    dense_rank = res.dense_rank # Dense cosine rank (or None)
```

### 4.2 Corrective Decision Routing Logic
Member 3's orchestrator routes retrieved candidates according to the evaluator's output:
```text
           [User Query] ──► [CRAGRetrieverFacade]
                                   │
                           Top-K Candidates
                                   │
                                   ▼
                      [Member 2 DeBERTa Evaluator]
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
   (1, 1) TRUST             (1, 0) VERIFY-MORE         (0, 0) REJECT-SEARCH
        │                          │                          │
  Direct Context           Secondary Retrieval         Query Rewriting &
  to Generator LLM         & Evidence Merging          External Search
```

---

## 5. Repository Structure & Artifact Paths

```text
C:\RAG\
├── data/
│   ├── processed/corpus.jsonl            <- 50-document curated technical corpus (tracked in Git)
│   ├── fixtures/corpus.jsonl             <- 10-document benchmark fixture corpus
│   └── synthetic/
│       ├── evaluator_train.jsonl         <- 160 development training examples for Member 2
│       └── evaluator_val.jsonl           <- 40 development validation examples for Member 2
├── indexes/
│   └── hybrid/                           <- Recreatable 50-document hybrid index (excluded from Git)
├── scripts/
│   ├── prepare_corpus.py                 <- Recreates data/processed/corpus.jsonl
│   ├── build_indexes.py                  <- Builds indexes/hybrid from corpus
│   └── generate_evaluator_datasets.py    <- Generates development train/val datasets
├── src/
│   ├── core/schemas.py                   <- Canonical Pydantic schemas (Document, Query, EvaluatorExample, RetrievalResult)
│   ├── core/dataset.py                   <- JSONL I/O dataset loaders
│   ├── retrieval/
│   │   ├── facade.py                     <- CRAGRetrieverFacade & create_crag_retriever()
│   │   ├── hybrid.py                     <- HybridRetriever (Reciprocal Rank Fusion)
│   │   ├── dense.py                      <- DenseRetriever (SentenceTransformers + FAISS IndexFlatIP)
│   │   └── sparse.py                     <- BM25Retriever (BM25-Okapi pure Python)
│   └── synthetic_data/
│       ├── judge.py                      <- LLMEvaluatorJudge & MockEvaluatorJudge
│       ├── pipeline.py                   <- SyntheticDataPipeline
│       └── validator.py                  <- Quality gate validator
└── tests/                                <- 98 passing unit & integration tests
```
