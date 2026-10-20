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

## 6. Running Tests

The test suite separates fast unit tests from model-loading integration tests:

```bash
# Run all fast offline unit tests
pytest -m unit

# Run environment verification test
pytest tests/test_environment.py

# Run with test coverage
pytest --cov=src tests/
```

---

## 7. Project Layout

```text
c:\RAG\
├── .gitignore                    <- Ignored venv, env secrets, caches, and indexes
├── .env.example                  <- Safe configuration template
├── README.md                     <- This setup and architecture guide
├── pyproject.toml                <- Project metadata & pytest configuration
├── requirements.txt              <- Production dependencies
├── requirements-dev.txt          <- Development & testing dependencies
├── data/
│   ├── raw/                      <- Source raw texts
│   ├── processed/                <- Ingested corpus files
│   ├── fixtures/                 <- Small controlled evaluation fixtures
│   └── synthetic/                <- Generated (Query, Doc, Label) datasets
├── indexes/
│   ├── bm25/                     <- Saved BM25 index artifacts
│   └── faiss/                    <- Saved FAISS vector index & ID mappings
├── src/
│   ├── core/                     <- Shared data contracts (Document, Labels, Dataset I/O)
│   ├── retrieval/                <- BaseSparseRetriever, BM25Retriever, and future dense/RRF
│   └── synthetic_data/           <- Negative generator, LLM judge, and validator
└── tests/
    ├── test_environment.py       <- Environment and dependency import tests
    ├── test_schemas.py           <- Schema validation & label rule tests
    ├── test_dataset_io.py        <- Dataset I/O & fixture integrity tests
    └── test_sparse.py            <- BM25 sparse retrieval tests
```

---

## 8. Collaboration Guidelines (Members 1, 2, and 3)

1. **Branching Strategy**:
   - `main`: Protected, production-ready releases only.
   - `feat/member1-retrieval`: Member 1 feature development branch.
   - `feat/member2-evaluator`: Member 2 training workstream.
   - `feat/member3-crag-pipeline`: Member 3 corrective pipeline workstream.
2. **Never Commit Secrets**:
   - `.env` is explicitly ignored in `.gitignore`. Only commit updates to `.env.example`.
3. **Stable Shared Contracts**:
   - Never alter schemas in `src/core/schemas.py` or retriever interfaces in `src/retrieval/base.py` without coordinating with Members 2 and 3.
