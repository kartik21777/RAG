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

## 7. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

The hybrid retrieval module (`src/retrieval/hybrid.py`) implements `BaseHybridRetriever` by fusing ranked candidates from `BM25Retriever` and `DenseRetriever` using **Reciprocal Rank Fusion (RRF)** (Cormack et al., SIGIR 2009).

### Why Rank Fusion Over Raw Score Normalization?

Combining sparse and dense retrievers by directly adding or weighting raw scores fails due to fundamental incompatibilities:
1. **Unbounded vs. Bounded Ranges**: BM25 scores are unbounded $[0, \infty)$ (dependent on document length and corpus-wide IDF), whereas unit-normalized dense cosine similarities are strictly bounded in $[-1.0, 1.0]$.
2. **Min-Max Distortion**: Linear min-max scaling $\frac{s - s_{\min}}{s_{\max} - s_{\min}}$ is acutely sensitive to outlier scores and produces query-dependent scale shifts.
3. **Incomparable Score Distributions**: Standard statistical normalizations (such as Z-score standardization $\frac{s - \mu}{\sigma}$) align first and second moments, but score normalization alone does not guarantee semantic comparability between BM25 lexical match scores and dense cosine similarities. Because lexical and dense models exhibit fundamentally different score dynamics across diverse queries, linear score combinations remain unstable without extensive per-corpus calibration.

RRF resolves this by relying purely on the **ordinal ranking position** of candidates within each retriever:

$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{w_m}{k + \text{rank}_m(d)}$$

Where:
- $M = \{\text{sparse}, \text{dense}\}$ is the set of retrievers.
- $\text{rank}_m(d) \in \{1, 2, \dots\}$ is the 1-indexed rank position of document $d$ in retriever $m$.
- $k$ is the smoothing constant (default: $k=60$).
  - For $k=60$, Rank 1 adds $\frac{1}{61} \approx 0.01639$, while Rank 2 adds $\frac{1}{62} \approx 0.01613$.
  - The constant $k$ dampens extreme penalties for lower ranks and prevents a high-ranking outlier from a single engine from completely dominating consensus documents surfaced by both engines.
- $w_m$ is an optional component weight (default: $w_{sparse} = 1.0, w_{dense} = 1.0$).

### Consensus Boosting
When a document is retrieved by **both** sparse and dense engines, its score accumulates additively:
$$\text{RRF}(d) = \frac{1}{k + \text{rank}_{sparse}(d)} + \frac{1}{k + \text{rank}_{dense}(d)}$$
For example, a document ranked at position 2 in both engines ($0.01613 + 0.01613 = 0.03226$) outranks a document appearing at Rank 1 in only one engine ($0.01639$). This naturally rewards multi-view agreement.

### Quickstart Example
```python
from src.core.dataset import load_corpus_from_jsonl
from src.retrieval.hybrid import HybridRetriever

# 1. Load corpus
corpus = load_corpus_from_jsonl("data/fixtures/corpus.jsonl")

# 2. Instantiate hybrid retriever (combines BM25 and Dense with RRF k=60)
hybrid_retriever = HybridRetriever(rrf_k=60)
hybrid_retriever.index(corpus)

# 3. Search
results = hybrid_retriever.search("Reciprocal Rank Fusion hybrid retrieval pipelines", top_k=3)
for candidate in results:
    print(f"Rank {candidate.rank}: {candidate.document.doc_id} (RRF Score: {candidate.score:.6f})")

# 4. Save and load complete hybrid index
hybrid_retriever.save("indexes/hybrid")
loaded_retriever = HybridRetriever.load("indexes/hybrid")
```

### Persistence Artifacts (`indexes/hybrid/`)
- `sparse/`: Nested BM25 index artifacts (`vocab.index.json`, sparse matrices, `retriever_metadata.json`).
- `dense/`: Nested FAISS index artifacts (`faiss.index`, `retriever_metadata.json`).
- `hybrid_metadata.json`: Top-level configuration storing `rrf_k`, `sparse_weight`, `dense_weight`, and candidate depths.

---

## 8. Comparative 3-Way Retrieval Analysis: Sparse vs Dense vs Hybrid

Empirical evaluation of the 6 fixture queries across the controlled 10-document corpus demonstrates how Hybrid RRF integrates the strengths of both engines:

| Query ID | Informational Intent / Focus | Expected Target | BM25 Top 1 (Score) | Dense Top 1 (Cosine) | Hybrid RRF Top 1 (Score) | Architectural Insight & Consensus Analysis |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`q_001`** | Exact formula for Reciprocal Rank Fusion | `doc_001` | **`doc_001`** (3.3278) | **`doc_001`** (0.7029) | **`doc_001`** (0.032787) | **Double Rank 1 Consensus**: Both retrievers place `doc_001` at Rank 1 ($1/61 + 1/61 = 0.032787$), cementing it at the top. |
| **`q_002`** | Combining sparse & dense rankings via RRF | `doc_001`, `doc_002` | **`doc_002`** (4.3566) | **`doc_002`** (0.7632) | **`doc_002`** (0.032787) | Both target documents dominate the top 2: `doc_002` at Rank 1 ($0.032787$) and `doc_001` at Rank 2 ($0.032258$). |
| **`q_003`** | BM25 hyperparameter $b$ length normalization | `doc_003` | **`doc_003`** (3.6160) | **`doc_003`** (0.6205) | **`doc_003`** (0.032787) | Both engines independently select `doc_003` as Rank 1; hybrid fusion yields the maximum possible single-candidate consensus score ($1/61 + 1/61 = 0.032787$), solidifying its Rank 1 position. |
| **`q_004`** | CRAG action when evidence is incomplete | `doc_006` | **`doc_006`** (4.3962) | **`doc_006`** (0.6098) | **`doc_006`** (0.032787) | Full consensus across keyword and semantic match on CRAG policy passage `doc_006`. |
| **`q_005`** | Shor's quantum algorithm breaking RSA *(Unanswerable)* | `[]` | `doc_009` (0.6334) | `doc_010` (0.2239) | `doc_009` (0.032018) | **Mathematical Rank Breakdown**: Sparse ranks `doc_009` at 1 (lexical overlap on 'algorithm', contribution $1/61 \approx 0.016393$). Because dense candidate depth ($K=20$) encompasses the small 10-document fixture corpus, `doc_009` also appears at Dense Rank 4 (cosine 0.1241, contribution $1/64 = 0.015625$). Their sum yields $\text{RRF}(doc\_009) = 1/61 + 1/64 = 0.032018$. This demonstrates that hybrid retrieval alone cannot reject unanswerable queries—it mechanically promotes candidate consensus regardless of semantic truth. This motivates the architectural requirement for Member 2's learned DeBERTa Evaluator to detect unanswerable inputs and trigger `REJECT_SEARCH`, though this fixture observation alone does not independently validate evaluator classification accuracy. |
| **`q_006`** | CRAG query rewriting mechanism *(Hard Negative)* | `doc_006` | **`doc_006`** (4.4521) | **`doc_006`** (0.6517) | **`doc_006`** (0.032787) | Both engines agree on `doc_006` at 1 and `doc_007` at 2. Demonstrates topic overlap where sufficiency evaluation is required. |

---

## 9. Synthetic Evaluator Training Data Generation (Stage 6)

To train Member 2's DeBERTa-v3 multi-task evaluator without relying on brittle manual annotations or leaking test sets, Member 1 provides an automated, provenance-tracked **Synthetic Training Data Pipeline**.

### 9.1 Evaluation Task Formulation & Strategy Distribution

The pipeline generates `(query, document, labels)` tuples across four explicit operational strategies:

| Strategy Name | Intent Fit ($y_1$) | Evidence Sufficiency ($y_2$) | CRAG Corrective Action | Mining / Synthesis Logic |
| :--- | :---: | :---: | :---: | :--- |
| **`relevant_sufficient`** | `1` | `1` | **`TRUST`** | Generated from salient factual assertions within the target passage. The passage contains complete, unambiguous answers. |
| **`relevant_insufficient`** | `1` | `0` | **`VERIFY-MORE`** | Focuses on partial concepts, edge cases, or preconditions mentioned in the passage where full context is deliberately missing. |
| **`hard_negative_topic`** | `0` | `0` | **`REJECT-SEARCH`** | Mined via BM25 retrieval for lexical overlap or shared topic keywords from candidate passages that do *not* answer the question. |
| **`random_negative`** | `0` | `0` | **`REJECT-SEARCH`** | Sampled uniformly at random from non-relevant corpus documents to establish a baseline discriminative floor. |

> [!IMPORTANT]
> **Strict Labeling Constraint**: As proven in Section 3, the combination `(intent_fit=0, evidence_sufficiency=1)` is **strictly prohibited**. A passage cannot be insufficient to address a query's intent yet simultaneously contain sufficient evidence to answer it. The pipeline's quality gates rigorously reject any candidate violating this constraint.

### 9.2 Pipeline Architecture & Components

```text
                  [Curated Document Corpus]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Query Generation]              [Negative Mining Engine]
   (Mock / LLM Generator)         (BM25 Lexical Hard Negatives)
            │                                 │
     (1,1) TRUST &                     (0,0) REJECT-SEARCH
     (1,0) VERIFY-MORE                 Topical & Random Negatives
            │                                 │
            └────────────────┬────────────────┘
                             ▼
               [Evaluator Label Assignment]
                 (Provenance & Metadata)
                             ▼
              [Strict Dataset Validation Gate]
             • Schema adherence check
             • Prohibition of (0,1) labels
             • 0% duplicate IDs
             • Class balance verification
                             ▼
            [evaluator_train.jsonl (Output)]
```

1. **Abstract Generator & Judge Contracts** (`src/synthetic_data/base.py`):
   - `BaseSyntheticGenerator`: Abstract interface for query and variation synthesis (`generate_sufficient_query`, `generate_insufficient_query`).
   - `BaseEvaluatorJudge`: Abstract interface for intent-fit and evidence-sufficiency assessment (`evaluate`).

2. **Concrete LLM-as-a-Judge Implementation** (`src/synthetic_data/judge.py`):
   - `LLMEvaluatorJudge`: Concrete judge implementing `BaseEvaluatorJudge` with prompt versioning (`JUDGE_PROMPT_VERSION = "v1.0"`), structured JSON parsing, prompt injection directives, and bounded retries.
   - `MockEvaluatorJudge`: Deterministic rule-based judge for reproducible offline testing of judged pipeline mode.
   - **Mandatory Hard-Negative Verification**: In judged mode, candidate negatives retrieved by BM25/Dense search are independently evaluated. If a candidate is determined to be relevant, it is not forced into `(0, 0)`; if the model evaluation fails or is ambiguous, it is routed to `pipeline.review_queue` and prevented from corrupting the training set.

3. **Negative Mining Engine** (`src/synthetic_data/negative_sampler.py`):
   - `sample_hard_negatives`: Uses BM25 or Dense retrievers to identify high-scoring false positives while strictly excluding positive document IDs (`exclude_doc_ids`).
   - `sample_random_negatives`: Uniformly samples negatives with deterministic seeding across non-target corpus documents.

4. **Generator Implementations** (`src/synthetic_data/generator.py`):
   - `MockDeterministicGenerator`: Zero-cost, 100% reproducible offline generator using SHA-256 hash seeds and heuristic linguistic templates. Used in CI and unit testing.
   - `LLMSyntheticGenerator`: Production-grade generator supporting structured OpenAI / Anthropic / Local LLM completions with JSON extraction and fallback guards.

5. **Validation Quality Gates** (`src/synthetic_data/validator.py`):
   - `validate_evaluator_dataset`: Audits generated datasets for Pydantic schema conformance, uniqueness of `example_id`, non-empty queries and documents, valid CRAG action mappings, and acceptable label class distributions.

6. **End-to-End Orchestrator** (`src/synthetic_data/pipeline.py`):
   - `SyntheticDataPipeline`: Coordinates query generation, negative candidate sampling with automatic fallback allocation for sparse corpora, sequential ID renumbering (`eval_syn_0001`, `eval_syn_0002`...), review queue management, and atomic JSONL persistence. Supports both deterministic construction mode (`judge=None`) and judged mode (`judge=LLMEvaluatorJudge(...)`).

### 9.3 Command-Line Interface (CLI)

Generate and validate synthetic datasets directly via the CLI:

```bash
# Mode 1: Deterministic construction mode (fast offline generation, zero LLM calls)
python -m src.synthetic_data.generate \
  --corpus-path data/fixtures/corpus.jsonl \
  --output-path data/synthetic/evaluator_train.jsonl \
  --judge none \
  --seed 42

# Mode 2: Offline judged mode (independent verification via MockEvaluatorJudge)
python -m src.synthetic_data.generate \
  --corpus-path data/fixtures/corpus.jsonl \
  --output-path data/synthetic/evaluator_train.jsonl \
  --judge mock \
  --seed 42
```

> [!NOTE]
> **Live LLM Judging Integration**: `LLMEvaluatorJudge` is intentionally provider-agnostic and accepts any completion callable (`llm_fn: Callable[[str], str]`). To run live judging against external providers (e.g. OpenAI or Anthropic), callers must supply a provider adapter wrapping their client and credentials (e.g. `LLMEvaluatorJudge(llm_fn=openai_adapter)`). The CLI does not bundle unpinned proprietary SDK dependencies or make network calls. Running `--judge llm` on the CLI directly signals that a provider adapter must be configured in code.

---

## 10. Running Tests

The automated test suite verifies all layers of the retrieval pipeline and synthetic data generation:

```bash
# Run the complete test suite (94 passing tests)
pytest

# Run fast unit tests
pytest -m unit

# Run specific component test suites
pytest tests/test_sparse.py
pytest tests/test_dense.py
pytest tests/test_hybrid.py
pytest tests/test_synthetic_data.py

# Run with test coverage
pytest --cov=src tests/
```

---

## 11. Project Layout

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
│   └── synthetic/                <- Generated (Query, Doc, Label) datasets (ignored in git)
├── indexes/
│   ├── bm25/                     <- Saved BM25 index artifacts
│   ├── faiss/                    <- Saved FAISS vector index & ID mappings
│   └── hybrid/                   <- Nested sparse and dense indexes with RRF metadata
├── src/
│   ├── core/                     <- Shared data contracts (Document, Labels, Dataset I/O)
│   ├── retrieval/                <- BaseSparseRetriever, BaseDenseRetriever, BaseHybridRetriever,
│   │                                BM25Retriever, DenseRetriever, HybridRetriever
│   └── synthetic_data/           <- Base contracts, Mock/LLM generators, LLMEvaluatorJudge,
│                                    MockEvaluatorJudge, NegativeSampler, Pipeline, Validator, CLI
└── tests/
    ├── test_environment.py       <- Environment and dependency import tests
    ├── test_schemas.py           <- Schema validation & label rule tests
    ├── test_dataset_io.py        <- Dataset I/O & fixture integrity tests
    ├── test_sparse.py            <- BM25 sparse retrieval tests
    ├── test_dense.py             <- Dense semantic retrieval & FAISS tests
    ├── test_hybrid.py            <- Reciprocal Rank Fusion hybrid retrieval tests
    └── test_synthetic_data.py    <- Synthetic generator, judge, sampler, validator & pipeline tests
```

---

## 12. Collaboration Guidelines (Members 1, 2, and 3)

1. **Branching Strategy**:
   - `main`: Protected, production-ready releases only.
   - `feat/member1-retrieval`: Member 1 feature development branch.
   - `feat/member2-evaluator`: Member 2 training workstream.
   - `feat/member3-crag-pipeline`: Member 3 corrective pipeline workstream.
2. **Never Commit Secrets**:
   - `.env` is explicitly ignored in `.gitignore`. Only commit updates to `.env.example`.
3. **Stable Shared Contracts**:
   - Never alter schemas in `src/core/schemas.py` or retriever interfaces in `src/retrieval/base.py` without coordinating with Members 2 and 3.
