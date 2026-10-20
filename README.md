# Semantic- and Evidence-Aware Corrective RAG (CRAG)
## Member 1: Hybrid Retrieval & Synthetic Data Pipeline

This repository hosts the **Hybrid Retrieval Pipeline** and **Synthetic Evaluator Training Data Generation** components of the Semantic- and Evidence-Aware Corrective RAG (CRAG) system.

---

## 1. System Overview & Architecture

Standard Retrieval-Augmented Generation (RAG) blindly feeds retrieved context to a generator LLM. In contrast, **Corrective RAG (CRAG)** introduces a custom evaluator (trained by Member 2 and orchestrated by Member 3) that scores retrieved documents along two distinct operational dimensions:
1. **Intent-Fit**: Does the retrieved document directly address the question's underlying informational intent?
2. **Evidence-Sufficiency**: Does the retrieved document contain sufficient factual information to formulate a complete, grounded answer to the question?

Evaluator predictions govern the 3-way corrective policy:
- **`TRUST`** (`Intent-Fit=1, Sufficiency=1`): Accept evidence and proceed to generation.
- **`VERIFY-MORE`** (`Intent-Fit=1, Sufficiency=0`): Context is relevant but incomplete; trigger focused secondary retrieval.
- **`REJECT-SEARCH`** (`Intent-Fit=0, Sufficiency=0`): Evidence is unsuitable (topic overlap without intent match); rewrite the query and search externally.

### Member Responsibilities Matrix
- **Member 1 (This Repository)**: Hybrid sparse-dense retrieval foundation (BM25 + Sentence Transformers + FAISS + RRF) and reproducible synthetic training data generation.
- **Member 2**: Custom DeBERTa-based evaluator fine-tuning, calibration, and threshold optimization.
- **Member 3**: Pipeline orchestration, corrective decision policy routing, and final generator LLM integration.

---

## 2. Sparse Retrieval Backend: Pyserini vs BM25s Decision

During Stage 0 & Stage 1 feasibility testing, the sparse retriever requirements were tested and verified:

### Pyserini (Anserini / Lucene Wrapper)
- **External Prerequisite**: Requires Java JDK 11 or 21 installed on the host machine and `JAVA_HOME` configured.
- **Windows Verification**: Pyserini builds from source on Windows via `pyjnius`. However, attempting to initialize Lucene without Java yields:
  ```text
  Exception: Unable to find JAVA_HOME
  ```
- **Portability Impact**: Imposing a mandatory Java runtime creates high friction across diverse teammate environments and CI runners.

### BM25s (High-Performance Python Backend)
- **Zero External Dependencies**: Pure Python / NumPy / SciPy implementation of the BM25 algorithm (BM25-Okapi, BM25L, BM25+).
- **Verified on Windows**: Installs and executes in under 2 seconds with native index serialization.

### Architectural Solution: Pluggable Adapter Pattern
To satisfy both requirements without rewriting the pipeline:
1. We define a common abstract base class: `BaseSparseRetriever`.
2. **Default Backend**: `bm25s` is used as the default cross-platform, zero-dependency engine.
3. **Pyserini Backend**: A `PyseriniRetriever` adapter is provided for environments with Java JDK 11+ and `JAVA_HOME` configured.
4. Selection is switchable via an environment variable (`SPARSE_BACKEND=bm25s` or `SPARSE_BACKEND=pyserini`).

---

## 3. Evaluator Labeling Policy & Operational Definitions

To prevent schema mismatch and ambiguous synthetic data, the two evaluation labels are strictly defined:

### Operational Definitions
- **`intent_fit` (0 or 1)**:
  - **1 (Fit)**: The passage directly addresses the specific question being asked.
  - **0 (Mismatch)**: The passage is off-topic, answers a different question, or merely shares keywords/topic without matching the user's information need.
- **`evidence_sufficiency` (0 or 1)**:
  - **1 (Sufficient)**: The passage contains all essential facts, figures, or logical steps needed to answer the user's query without hallucination or external inference.
  - **0 (Insufficient)**: The passage is missing one or more essential facts, provides only partial steps, or gives vague generalizations.

### Valid Training Combinations (Exactly 3)
In the training and evaluation dataset for Member 2, there are **exactly three valid label combinations**:

| `intent_fit` | `evidence_sufficiency` | Valid in Training? | CRAG Policy Action | Semantics & Ground Truth Definition |
| :---: | :---: | :---: | :---: | :--- |
| **1** | **1** | **YES** | **`TRUST`** | Document matches the question intent and provides complete, self-contained evidence to answer it. |
| **1** | **0** | **YES** | **`VERIFY-MORE`** | Document matches the question intent, but provides only partial or incomplete evidence. Requires secondary retrieval. |
| **0** | **0** | **YES** | **`REJECT-SEARCH`** | Document fails to match question intent (off-topic or topical hard negative). Zero usable evidence. |
| **0** | **1** | **PROHIBITED** | *Invalid State* | **Strictly prohibited in training data.** A document cannot possess sufficient evidence to answer a question whose intent it does not address. The dataset schema validator rejects any example with this combination. |

### Runtime Policy Handling for Contradictory Predictions
While training data prohibits `(0, 1)`, a learned multi-task model (such as Member 2's DeBERTa) evaluates heads independently and may theoretically emit a false-positive sufficiency score when intent fit is low.

**Runtime Contract for Member 3 (Decision Policy):**
- **Intent dominance rule**: `intent_fit` is a hard prerequisite for evidence trust.
- If the evaluator predicts `intent_fit = 0` (or below its decision threshold), the runtime decision policy **must treat the prediction as `REJECT-SEARCH`**, unconditionally overriding any contradictory `evidence_sufficiency = 1` signal:
  ```python
  if intent_fit == 0:
      action = Action.REJECT_SEARCH  # Overrides sufficiency prediction
  elif evidence_sufficiency == 1:
      action = Action.TRUST
  else:
      action = Action.VERIFY_MORE
  ```


---

## 4. Environment Setup and Installation

### Prerequisites
- **Python**: **Python 3.13 (64-bit)** is the verified local development baseline on this workstation. For teammates on other operating systems (macOS, Ubuntu) or machines with Python 3.12 installed, the repository and dependency bounds explicitly support **Python `>=3.10, <=3.13`**.
- **Operating System**: Windows 10/11 (64-bit), macOS, or Ubuntu Linux.
- **Java JDK (Optional)**: Only required if using the optional Pyserini backend (`OpenJDK 11` or `21`). The default `bm25s` backend requires zero Java dependencies.

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/kartik21777/RAG.git
cd RAG
```

#### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux (Bash/Zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install main runtime dependencies
pip install -r requirements.txt

# (Optional) Install development and testing dependencies
pip install -r requirements-dev.txt
```

#### 4. Configure Environment Variables
Copy `.env.example` to create your local `.env`:
```bash
# Windows
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```
Edit `.env` to configure paths and parameters. Placeholder values are safe for local development and offline testing.

---

## 5. BM25 Sparse Retrieval Usage & Configuration

The sparse retrieval module (`src/retrieval/sparse.py`) implements `BaseSparseRetriever` using `bm25s`:

### Quickstart Example
```python
from src.core.dataset import load_corpus_from_jsonl
from src.retrieval.sparse import BM25Retriever

# 1. Load corpus
corpus = load_corpus_from_jsonl("data/fixtures/corpus.jsonl")

# 2. Instantiate and index
retriever = BM25Retriever(k1=1.5, b=0.75, method="lucene")
retriever.index(corpus)

# 3. Search
results = retriever.search("Reciprocal Rank Fusion formula", top_k=3)
for candidate in results:
    print(f"Rank {candidate.rank}: {candidate.document.doc_id} (Score: {candidate.score:.4f})")

# 4. Save and load index
retriever.save("indexes/bm25")
loaded_retriever = BM25Retriever.load("indexes/bm25")
```

### BM25 Scoring Formulation & Saturation Analysis
`bm25s` evaluates document relevance using the Lucene BM25 scoring formulation:

$$\text{Score}(Q, D) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $\text{TF}(t, D)$ is the raw term frequency in document $D$.
- $|D|$ and $\text{avgdl}$ represent the document length and average corpus document length.
- $\text{IDF}(t) = \ln\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)$ measures term rarity across corpus size $N$.
- **Term-Frequency Saturation ($k_1$)**:
  Notice the mathematical saturation behavior: while the unscaled ratio $\frac{\text{TF}}{\text{TF} + k_1 \cdot B}$ approaches $1$ as $\text{TF} \to \infty$, the complete BM25 term-frequency component with numerator $\text{TF} \cdot (k_1 + 1)$ approaches **$(k_1 + 1)$** for any fixed document:
  $$\lim_{\text{TF}(t, D) \to \infty} \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot B} = k_1 + 1$$
  For our default $k_1 = 1.5$, an infinitely repeated query term can contribute at most **$2.5 \times \text{IDF}(t)$**. This mathematically bounds the influence of repeated words and prevents keyword stuffing.
- **Document-Length Normalization ($b$)**:
  $b \in [0.0, 1.0]$ (default: $0.75$) scales the document length penalty $B = (1 - b) + b \cdot \frac{|D|}{\text{avgdl}}$. When $b=1$, the score is fully penalized in proportion to document length; when $b=0$, length normalization is completely disabled.

### Persistence Artifacts (`indexes/bm25/`)
- `data.csc.index.npy`, `indices.csc.index.npy`, `indptr.csc.index.npy`: Compressed sparse matrix of token scores.
- `vocab.index.json`: Vocabulary term-to-ID mapping.
- `params.index.json`: BM25 hyperparameters.
- `retriever_metadata.json`: Schema version, ordered document IDs, and full `Document` models for instant reconstruction.

---

## 6. Dense Semantic Retrieval (Sentence Transformers & FAISS IndexFlatIP)

The dense retrieval module (`src/retrieval/dense.py`) implements `BaseDenseRetriever` using `sentence-transformers` and `faiss-cpu`:

### Mathematical Formulation: Cosine Similarity via Inner Product

1. **Transformer Encodings**: Texts (queries and passages) are encoded into $D$-dimensional real-valued dense representations using `all-MiniLM-L6-v2` ($D = 384$).
2. **Unit-Hypersphere L2 Normalization**:
   Each vector $v \in \mathbb{R}^D$ is projected onto the unit sphere:
   $$\hat{v} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_{i=1}^D v_i^2}}, \quad \text{such that } \|\hat{v}\|_2 = 1.0$$
3. **Exact Inner Product Equivalence**:
   By geometric definition, the inner product between two unit-normalized vectors equals their cosine similarity:
   $$\langle \hat{u}, \hat{v} \rangle = \sum_{i=1}^D \hat{u}_i \hat{v}_i = \|\hat{u}\|_2 \|\hat{v}\|_2 \cos(\theta) = \cos(\theta) \in [-1.0, 1.0]$$
4. **FAISS IndexFlatIP**:
   Using `faiss.IndexFlatIP` performs exhaustive brute-force exact Maximum Inner Product Search (MIPS) across all indexed vectors with zero quantization distortion. This guarantees exact vector nearest neighbors relative to the embedding representations. However, end-to-end retrieval relevance and information-theoretic metrics like Recall@K depend fundamentally on the corpus coverage, embedding model quality, and task-specific relevance judgments.

### Quickstart Example
```python
from src.core.dataset import load_corpus_from_jsonl
from src.retrieval.dense import DenseRetriever

# 1. Load corpus
corpus = load_corpus_from_jsonl("data/fixtures/corpus.jsonl")

# 2. Instantiate and index (default model: all-MiniLM-L6-v2)
retriever = DenseRetriever()
retriever.index(corpus)

# 3. Search (returns top_k candidates with cosine similarity scores in [-1.0, 1.0])
results = retriever.search("neural vector search and cosine similarity", top_k=3)
for candidate in results:
    print(f"Rank {candidate.rank}: {candidate.document.doc_id} (Cosine Sim: {candidate.score:.4f})")

# 4. Save and load index
retriever.save("indexes/faiss")
loaded_retriever = DenseRetriever.load("indexes/faiss")
```

### Persistence Artifacts (`indexes/faiss/`)
- `faiss.index`: Serialized native binary FAISS index (written via `faiss.write_index`).
- `retriever_metadata.json`: Model architecture, embedding dimension, normalization flags, ordered document IDs, and full `Document` models for instant reconstruction.

---

## 7. Comparative Retrieval Analysis: Sparse (BM25) vs Dense (FAISS)

Empirical evaluation of the 6 fixture queries across the controlled 10-document corpus highlights the complementary nature of sparse and dense retrieval:

| Query ID | Informational Intent / Focus | Expected Target | BM25 Top 1 (Score) | Dense Top 1 (Cosine Sim) | Analysis & Observations |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`q_001`** | Exact formula for Reciprocal Rank Fusion | `doc_001` | **`doc_001`** (3.3278) | **`doc_001`** (0.7029) | Both retrievers succeed. BM25 leverages exact token overlap (`formula`, `RRF`), while Dense captures semantic relevance. |
| **`q_002`** | Combining sparse & dense rankings via RRF | `doc_001`, `doc_002` | **`doc_002`** (4.3566) | **`doc_002`** (0.7632) | Both retrievers place the two expected target documents in Rank 1 (`doc_002`) and Rank 2 (`doc_001`). |
| **`q_003`** | BM25 hyperparameter $b$ length normalization | `doc_003` | **`doc_003`** (3.6160) | **`doc_003`** (0.6205) | Strong agreement. BM25 directly scores $b$ parameter terms; dense captures parameter role semantics. |
| **`q_004`** | CRAG action when evidence is incomplete | `doc_006` | **`doc_006`** (4.3962) | **`doc_006`** (0.6098) | Both retrievers isolate `doc_006` (CRAG policy specification) as Rank 1. |
| **`q_005`** | Shor's quantum algorithm breaking RSA *(Unanswerable)* | `[]` | `doc_009` (0.6334) | `doc_010` (0.2239) | **Critical CRAG Insight**: Neither retriever returns zero candidates natively; Dense forces nearest neighbors even for out-of-domain queries. The lower similarity observed here (~0.22 vs ~0.60–0.76 for relevant queries) is a preliminary empirical observation on this small fixture set, **not a universal rejection threshold**. Rather than relying on rigid score heuristics, CRAG relies on Member 2's trained DeBERTa Evaluator to detect out-of-domain context and reliably trigger `REJECT_SEARCH`. |
| **`q_006`** | CRAG query rewriting mechanism *(Hard Negative)* | `doc_006` | **`doc_006`** (4.4521) | **`doc_006`** (0.6517) | Both retrievers rank `doc_006` at 1 and `doc_007` (web query rewriting) at 2. The document mentions rewriting but lacks mechanism details, motivating Member 2's sufficiency evaluation. |

---

## 8. Running Tests

The test suite covers environment sanity, data contracts, dataset serialization, sparse BM25 retrieval, and dense FAISS retrieval:

```bash
# Run the complete test suite (56 tests)
pytest

# Run fast unit tests
pytest -m unit

# Run sparse retrieval tests only
pytest tests/test_sparse.py

# Run dense retrieval tests only
pytest tests/test_dense.py

# Run with test coverage
pytest --cov=src tests/
```

---

## 9. Project Layout

```text
c:\RAG\
├── .gitignore                    <- Ignored venv, env secrets, caches, and indexes
├── .env.example                  <- Safe configuration template
├── README.md                     <- Setup, architecture, and mathematical formulation guide
├── pyproject.toml                <- Project metadata & pytest configuration
├── requirements.txt              <- Production dependencies
├── requirements-dev.txt          <- Development & testing dependencies
├── data/
│   ├── raw/                      <- Source raw texts
│   ├── processed/                <- Ingested corpus files
│   ├── fixtures/                 <- Small controlled evaluation fixtures
│   │   ├── corpus.jsonl          <- 10 curated test documents
│   │   ├── queries.jsonl         <- 6 test queries with expected doc IDs and CRAG actions
│   │   └── evaluator_labels.jsonl<- Ground-truth evaluator pairs with rationales
│   └── synthetic/                <- Generated (Query, Doc, Label) datasets
├── indexes/
│   ├── bm25/                     <- Saved BM25 index artifacts
│   └── faiss/                    <- Saved FAISS vector index & ID mappings
├── src/
│   ├── core/                     <- Shared data contracts (Document, Labels, Dataset I/O)
│   ├── retrieval/                <- BaseSparseRetriever, BaseDenseRetriever, BM25Retriever, DenseRetriever
│   └── synthetic_data/           <- Negative generator, LLM judge, and validator
└── tests/
    ├── test_environment.py       <- Environment and dependency import tests
    ├── test_schemas.py           <- Schema validation & label rule tests
    ├── test_dataset_io.py        <- Dataset I/O & fixture integrity tests
    ├── test_sparse.py            <- BM25 sparse retrieval tests
    └── test_dense.py             <- Dense semantic retrieval & FAISS tests
```

---

## 10. Collaboration Guidelines (Members 1, 2, and 3)

1. **Branching Strategy**:
   - `main`: Protected, production-ready releases only.
   - `feat/member1-retrieval`: Member 1 feature development branch.
   - `feat/member2-evaluator`: Member 2 training workstream.
   - `feat/member3-crag-pipeline`: Member 3 corrective pipeline workstream.
2. **Never Commit Secrets**:
   - `.env` is explicitly ignored in `.gitignore`. Only commit updates to `.env.example`.
3. **Stable Shared Contracts**:
   - Never alter schemas in `src/core/schemas.py` or retriever interfaces in `src/retrieval/base.py` without coordinating with Members 2 and 3.
