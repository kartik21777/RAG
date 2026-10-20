"""
Prepares the curated 50-document technical corpus in data/processed/corpus.jsonl.
Provides diverse domain passages for hybrid retrieval indexing and synthetic data generation.
"""

import json
from pathlib import Path

DOCUMENTS = [
    {
        "doc_id": "doc_001",
        "title": "Reciprocal Rank Fusion in Information Retrieval",
        "source": "ir_textbook",
        "text": "Reciprocal Rank Fusion (RRF) is an algorithmic rank aggregation method that combines ranked lists from multiple retrieval systems without needing score normalization. For each document d, its RRF score is computed as score(d) = sum_{m in M} 1 / (k + rank_m(d)), where M is the set of retrievers, rank_m(d) is the 1-indexed rank of document d in retriever m, and k is a smoothing constant typically set to 60. RRF effectively balances top candidates from both sparse (BM25) and dense (vector) retrieval."
    },
    {
        "doc_id": "doc_002",
        "title": "Hybrid Retrieval Architecture Overview",
        "source": "engineering_blog",
        "text": "In hybrid retrieval pipelines, Reciprocal Rank Fusion is frequently deployed as the fusion layer. It merges sparse keyword candidates with dense semantic candidates to produce a single unified ranking. By taking the ranking positions from both retrievers, it helps surface relevant results even when raw retrieval scores operate on completely different scales."
    },
    {
        "doc_id": "doc_003",
        "title": "Understanding BM25 Parameters and Scoring",
        "source": "search_handbook",
        "text": "BM25 (Best Matching 25) is a probabilistic sparse retrieval algorithm based on term frequency and inverse document frequency. Its scoring function incorporates document length normalization controlled by hyperparameter b (default 0.75) and term frequency saturation controlled by hyperparameter k1 (default 1.2 to 2.0). Higher k1 values allow repeated terms to contribute more to the score, whereas b penalizes excessively long documents."
    },
    {
        "doc_id": "doc_004",
        "title": "Vector Embeddings and Inner Product Search",
        "source": "ml_guide",
        "text": "Dense vector retrieval maps textual queries and passages into high-dimensional latent space using transformer encoder models like all-MiniLM-L6-v2. When embeddings are L2-normalized, the cosine similarity between two vectors is mathematically identical to their inner product (dot product). This allows fast maximum inner product search (MIPS) using FAISS IndexFlatIP without expensive square root distance computations."
    },
    {
        "doc_id": "doc_005",
        "title": "FAISS Index Selection and ANN Tradeoffs",
        "source": "vector_db_docs",
        "text": "FAISS (Facebook AI Similarity Search) offers multiple vector index implementations. IndexFlatIP provides exact brute-force search with 100% recall but scales linearly O(N) in search latency. For million-scale datasets, Approximate Nearest Neighbor (ANN) indexes such as IndexIVFFlat (inverted file with Voronoi cells) or IndexHNSWFlat (Hierarchical Navigable Small World graphs) provide sub-linear search time at the cost of a slight recall tradeoff."
    },
    {
        "doc_id": "doc_006",
        "title": "Corrective RAG Decision Policy",
        "source": "crag_design_spec",
        "text": "Semantic- and Evidence-Aware Corrective RAG (CRAG) evaluates retrieved query-document candidates along two axes: Intent-Fit and Evidence-Sufficiency. Based on these two dimensions, the decision router chooses one of three actions: TRUST (evidence is relevant and sufficient), VERIFY-MORE (evidence is on-topic but incomplete, triggering supplementary retrieval), or REJECT-SEARCH (evidence is off-intent, triggering query rewriting and web fallback)."
    },
    {
        "doc_id": "doc_007",
        "title": "Web Search Query Rewriting Systems",
        "source": "web_search_survey",
        "text": "Query rewriting in commercial web search engines optimizes user input by expanding abbreviations, injecting domain synonyms, and correcting spelling errors. In contrast to database query planners, search rewrites focus on increasing recall across inverted document indexes by generating alternative lexical phrasing."
    },
    {
        "doc_id": "doc_008",
        "title": "DeBERTa Architecture and Disentangled Attention",
        "source": "nlp_paper",
        "text": "DeBERTa (Decoding-enhanced BERT with disentangled attention) improves upon BERT and RoBERTa by representing each token with two vectors: one encoding its semantic content and another encoding its relative position. Disentangled attention computes cross-attention scores across content-to-content, content-to-position, and position-to-content pairs, significantly improving performance on sequence classification and evaluator tasks."
    },
    {
        "doc_id": "doc_009",
        "title": "Raft Distributed Consensus Protocol",
        "source": "distributed_systems_notes",
        "text": "The Raft consensus algorithm maintains state machine consistency across distributed server clusters. Raft decomposes consensus into three independent sub-problems: leader election, log replication, and commit safety. If a leader fails, follower nodes initiate a new election term using randomized election timeouts to prevent split votes."
    },
    {
        "doc_id": "doc_010",
        "title": "Product Quantization for Memory-Constrained Search",
        "source": "compression_tutorial",
        "text": "Product Quantization (PQ) is a lossy compression technique used to reduce the memory footprint of vector databases. A high-dimensional vector of dimension D is divided into m smaller sub-vectors, and each sub-vector is quantized to its nearest centroid among a learned codebook. This compresses a 384-dimensional float32 vector (1536 bytes) into just a few dozen bytes, enabling billion-scale search on a single machine."
    },
    {
        "doc_id": "doc_011",
        "title": "ColBERT Late Interaction Retrieval Model",
        "source": "neural_ir_paper",
        "text": "ColBERT (Contextualized Late Interaction over BERT) introduces token-level vector representations for both queries and documents. Rather than compressing an entire passage into a single dense vector, ColBERT computes the MaxSim operator across all query and document token embeddings. This late interaction preserves fine-grained token alignments while maintaining fast retrieval via vector quantization."
    },
    {
        "doc_id": "doc_012",
        "title": "SPLADE Learned Sparse Representations",
        "source": "neural_ir_paper",
        "text": "SPLADE (Sparse Lexical and Expansion Model) predicts term weights in the vocabulary space using transformer representations and MLM heads. By projecting passages onto vocabulary dimensions with regularization, SPLADE performs neural query expansion and term weighting while remaining compatible with standard inverted index structures."
    },
    {
        "doc_id": "doc_013",
        "title": "Hierarchical Navigable Small World (HNSW) Graphs",
        "source": "ann_survey",
        "text": "HNSW constructs a multi-layer graph where lower layers contain dense connections between nearby vectors and upper layers contain long-range skip connections. Vector search performs greedy routing at top layers before descending to lower layers, achieving logarithmic search complexity O(log N) with high empirical recall."
    },
    {
        "doc_id": "doc_014",
        "title": "Inverted File Index with Voronoi Cells (IVFFlat)",
        "source": "vector_db_docs",
        "text": "IVFFlat partitions vector space into k Voronoi cells using k-means clustering. At query time, only the vectors belonging to the nprobe closest centroids are scanned. Adjusting nprobe balances search speed against nearest-neighbor recall."
    },
    {
        "doc_id": "doc_015",
        "title": "DiskANN Graph-Based Out-of-Core Search",
        "source": "systems_paper",
        "text": "DiskANN allows indexing and searching billions of vectors using solid-state drives (SSDs) rather than expensive DRAM. By storing compressed vectors in memory for candidate pruning and layout-optimized graph neighbors on disk, DiskANN achieves microsecond search latencies with minimal RAM usage."
    },
    {
        "doc_id": "doc_016",
        "title": "Corrective RAG Knowledge Refinement and Filtering",
        "source": "crag_paper",
        "text": "Knowledge refinement in CRAG strips irrelevant sentences and noisy tokens from retrieved passages before context generation. When an evaluator flags a document as partially sufficient, internal sentence-level scoring isolates key factual clauses to reduce context clutter."
    },
    {
        "doc_id": "doc_017",
        "title": "Web Fallback and External Search in CRAG",
        "source": "crag_paper",
        "text": "When internal retrieval triggers a REJECT-SEARCH decision due to query intent mismatch or missing knowledge base coverage, CRAG initiates external search queries. The original user question is transformed into search engine queries with site constraints and entity disambiguation to fetch external documents."
    },
    {
        "doc_id": "doc_018",
        "title": "DeBERTa Disentangled Position Bias and Relative Offsets",
        "source": "nlp_paper",
        "text": "Unlike standard self-attention which adds absolute position embeddings to content vectors, DeBERTa keeps content and position matrices separated. When evaluating (query, document) pairs, relative distance matrices ensure that token dependencies remain invariant to paragraph offset."
    },
    {
        "doc_id": "doc_019",
        "title": "Cross-Encoder Re-Ranking Architectures",
        "source": "ir_textbook",
        "text": "Cross-encoders concatenate query and document tokens into a single sequence [CLS] query [SEP] document [SEP] and pass it through all transformer self-attention layers. Cross-encoders achieve higher accuracy than bi-encoders because every query token directly attends to every document token, making them ideal for re-ranking top-K candidates."
    },
    {
        "doc_id": "doc_020",
        "title": "Calibration of Multi-Task Binary Classifiers",
        "source": "ml_handbook",
        "text": "Probabilistic calibration transforms raw model logits into empirical probabilities that reflect ground-truth confidence. Platt scaling and isotonic regression align sigmoid outputs with true class proportions, ensuring that decision thresholds for TRUST and VERIFY-MORE remain statistically sound across varying corpus domains."
    },
    {
        "doc_id": "doc_021",
        "title": "Paxos Consensus Protocol Invariants",
        "source": "distributed_systems_notes",
        "text": "The classic Paxos protocol achieves consensus across asynchronous networks with message loss. A proposer issues a Prepare request with ballot number n; once a majority of acceptors promise not to accept lower ballots, the proposer issues an Accept request with the chosen value."
    },
    {
        "doc_id": "doc_022",
        "title": "Vector Clocks and Causality Tracking",
        "source": "distributed_systems_notes",
        "text": "Vector clocks track partial ordering and causal relationships in distributed systems without synchronized physical clocks. Each process maintains an integer vector representing event counters across all participating nodes. An event a causally precedes b if every component of vector(a) is less than or equal to vector(b)."
    },
    {
        "doc_id": "doc_023",
        "title": "Chandy-Lamport Global State Snapshot Algorithm",
        "source": "distributed_systems_notes",
        "text": "The Chandy-Lamport algorithm records a consistent global state of a distributed system without halting execution. Marker messages sent along communication channels delineate local process states and in-flight channel messages without requiring global clock synchronization."
    },
    {
        "doc_id": "doc_024",
        "title": "CAP Theorem and Consistency Models",
        "source": "distributed_systems_notes",
        "text": "Eric Brewer's CAP theorem states that any distributed data store can simultaneously provide at most two out of three guarantees: Consistency (every read receives the most recent write), Availability (every non-failing node returns a response), and Partition Tolerance (system functions despite network drops)."
    },
    {
        "doc_id": "doc_025",
        "title": "Byzantine Fault Tolerance and PBFT Protocol",
        "source": "distributed_systems_notes",
        "text": "Practical Byzantine Fault Tolerance (PBFT) enables distributed consensus in adversarial environments where malicious nodes can broadcast conflicting messages. PBFT tolerates up to f faulty nodes in a network of 3f + 1 nodes using three phases: pre-prepare, prepare, and commit."
    },
    {
        "doc_id": "doc_026",
        "title": "Shor's Quantum Factoring Algorithm",
        "source": "quantum_computing_handbook",
        "text": "Shor's algorithm is a quantum polynomial-time algorithm for prime factorization and discrete logarithms. By applying the Quantum Fourier Transform (QFT) to compute the modular order r of a composite number N = p * q, Shor's algorithm solves factorization in O((log N)^3) quantum gates, breaking RSA and Diffie-Hellman encryption."
    },
    {
        "doc_id": "doc_027",
        "title": "Grover's Quantum Search Algorithm",
        "source": "quantum_computing_handbook",
        "text": "Grover's algorithm provides a quadratic speedup for searching an unsorted database of N items. By iteratively applying an oracle phase inversion and an amplitude amplification diffusion operator, Grover's algorithm finds the target item in O(sqrt(N)) evaluations compared to O(N) classically."
    },
    {
        "doc_id": "doc_028",
        "title": "RSA Public-Key Cryptosystem Fundamentals",
        "source": "cryptography_textbook",
        "text": "RSA encryption relies on the computational hardness of factoring large semiprime integers N = p * q. The public key consists of exponent e and modulus N, while the private key d satisfies e * d = 1 (mod (p - 1)(q - 1)). Without knowledge of the prime factors p and q, computing the private key from public parameters is computationally infeasible on classical computers."
    },
    {
        "doc_id": "doc_029",
        "title": "Shannon Entropy and Lossless Compression Limits",
        "source": "information_theory",
        "text": "Claude Shannon established that the minimum average code length required to represent a discrete random variable X without loss is bounded below by its entropy H(X) = - sum p(x) log2 p(x). Huffman coding and arithmetic coding asymptotically reach this fundamental information-theoretic bound."
    },
    {
        "doc_id": "doc_030",
        "title": "Differential Privacy in Machine Learning",
        "source": "security_handbook",
        "text": "Differential privacy provides mathematical guarantees that the output of an algorithm does not reveal whether any individual record was present in the training set. Mechanisms like the Laplace mechanism and Gaussian mechanism inject calibrated noise proportional to the sensitivity of the query function."
    },
    {
        "doc_id": "doc_031",
        "title": "Sentence-Transformers Architecture and Pooling Strategies",
        "source": "ml_guide",
        "text": "Sentence-transformers models derive fixed-size sentence embeddings from transformer token outputs. Common pooling strategies include Mean Pooling (averaging all token embeddings weighted by attention mask), CLS Pooling (using the [CLS] representation), and Max Pooling. For all-MiniLM-L6-v2, Mean Pooling with L2 normalization yields optimal cosine similarity rankings."
    },
    {
        "doc_id": "doc_032",
        "title": "Maximum Inner Product Search (MIPS) and Cosine Equivalence",
        "source": "ml_guide",
        "text": "When vector embeddings are normalized to unit Euclidean length (||u||_2 = 1 and ||v||_2 = 1), the cosine similarity cos(u, v) = (u . v) / (||u||_2 ||v||_2) simplifies directly to the dot product u . v. This equivalence enables hardware-accelerated SIMD matrix multiplication in FAISS IndexFlatIP."
    },
    {
        "doc_id": "doc_033",
        "title": "Reciprocal Rank Fusion k Parameter Sensitivity",
        "source": "ir_textbook",
        "text": "The smoothing constant k in RRF controls the penalization of lower-ranked documents. A small k (e.g. 10) heavily prioritizes rank-1 documents, whereas a large k (e.g. 100) smooths differences between adjacent ranks. Empirical benchmarks across TREC and MS MARCO show that k=60 robustly mitigates noise across sparse and dense candidates."
    },
    {
        "doc_id": "doc_034",
        "title": "BM25 Document Length Normalization Factor b",
        "source": "search_handbook",
        "text": "In BM25, the denominator term (1 - b + b * (doc_len / avg_doc_len)) adjusts term scores based on relative document length. Setting b=1.0 assumes long documents are repetitive and penalizes them proportionally, while b=0.0 completely disables length normalization."
    },
    {
        "doc_id": "doc_035",
        "title": "BM25 Term Frequency Saturation Factor k1",
        "source": "search_handbook",
        "text": "The k1 hyperparameter in BM25 regulates how fast term-frequency saturation occurs. As term occurrences in a document approach infinity, the term-frequency component approaches k1 + 1. Typical values range from 1.2 to 2.0."
    },
    {
        "doc_id": "doc_036",
        "title": "Hallucination Reduction via Grounded Attribution",
        "source": "crag_design_spec",
        "text": "Grounded attribution requires that every factual assertion in a generated response directly cites a verified document span. By checking sentence-level entailing relations between the generation and retrieved evidence, CRAG eliminates hallucinated claims."
    },
    {
        "doc_id": "doc_037",
        "title": "Self-Correction Mechanisms in Agentic RAG",
        "source": "crag_design_spec",
        "text": "Agentic self-correction cycles inspect intermediate generator drafts against the retrieved evidence set. When discrepancies or unsupported facts are detected, the system triggers prompt refinement and secondary retrieval before emitting the final response."
    },
    {
        "doc_id": "doc_038",
        "title": "Citation Graphs and Evidence Verification",
        "source": "crag_design_spec",
        "text": "Citation graphs construct directed edges from generated answer claims to source document IDs. Nodes with zero incoming evidence edges represent unsupported statements and are systematically excised during post-processing."
    },
    {
        "doc_id": "doc_039",
        "title": "Multi-Task Loss Formulation for Dual-Head Evaluators",
        "source": "nlp_paper",
        "text": "Dual-head evaluators predict intent-fit and evidence-sufficiency simultaneously. The multi-task objective combines two binary cross-entropy losses: L = BCE(y_intent, y_hat_intent) + BCE(y_suff, y_hat_suff). An auxiliary penalty term lambda * max(0, y_hat_suff - y_hat_intent) actively discourages the invalid (0, 1) prediction state."
    },
    {
        "doc_id": "doc_040",
        "title": "Hard Negative Mining with Lexical BM25 Retrievers",
        "source": "ir_textbook",
        "text": "Mining hard negatives requires identifying documents that exhibit high lexical overlap with a query without answering it. Running BM25 over the corpus and selecting non-ground-truth top candidates provides effective negatives that force evaluators to learn deep semantic intent."
    },
    {
        "doc_id": "doc_041",
        "title": "Random Negative Sampling Floor in Evaluator Training",
        "source": "ir_textbook",
        "text": "While hard negatives challenge the model with fine-grained distinctions, random negatives provide an anchor ensuring the model does not overfit to topical similarity. Mixing hard negatives and random negatives maintains balanced gradient updates."
    },
    {
        "doc_id": "doc_042",
        "title": "Prompt Injection Mitigation in Evaluator Pipelines",
        "source": "security_handbook",
        "text": "LLM evaluators reviewing third-party web content are vulnerable to indirect prompt injection. Wrapping passages in explicit untrusted delimiters and instructing the model to treat content strictly as inert evaluation text prevents malicious prompt override."
    },
    {
        "doc_id": "doc_043",
        "title": "Deterministic Synthetic Generation Hash Seeding",
        "source": "systems_paper",
        "text": "Deterministic synthetic data generation uses SHA-256 hash digests of document text and unique doc IDs as pseudorandom number generator seeds. This ensures 100% reproducible test fixtures across distributed development environments."
    },
    {
        "doc_id": "doc_044",
        "title": "Schema Validation and Quality Gates in ML Pipelines",
        "source": "systems_paper",
        "text": "Automated ML data quality gates validate schema typing, check null constraints, detect duplicate identifiers, and enforce class balance bounds before dataset persistence. Early failure at the data boundary prevents silent model degradation."
    },
    {
        "doc_id": "doc_045",
        "title": "Atomic File Persistence and Concurrent Writes",
        "source": "systems_paper",
        "text": "Writing datasets atomically requires writing to a temporary file in the target directory and invoking atomic filesystem rename (os.replace). This prevents partial file corruption if a process terminates abruptly during serialization."
    },
    {
        "doc_id": "doc_046",
        "title": "Two-Phase Commit (2PC) Distributed Transaction Protocol",
        "source": "distributed_systems_notes",
        "text": "The Two-Phase Commit protocol coordinates atomic distributed transactions across multiple database nodes. In the Prepare phase, the coordinator asks all participants if they can commit. If all reply affirmatively, the coordinator broadcasts a Commit command in the second phase."
    },
    {
        "doc_id": "doc_047",
        "title": "Log-Structured Merge (LSM) Trees in Storage Engines",
        "source": "systems_paper",
        "text": "LSM trees optimize write performance by appending updates to an in-memory MemTable backed by a write-ahead log. When the MemTable fills, it is flushed sequentially to disk as an immutable SSTable. Periodic compaction merges overlapping SSTables to reclaim deleted space."
    },
    {
        "doc_id": "doc_048",
        "title": "Bloom Filters for Fast Negative Set Lookups",
        "source": "systems_paper",
        "text": "A Bloom filter is a space-efficient probabilistic data structure that tests set membership with zero false negatives and a tunable false positive rate. Hashing keys into a bit array enables vector databases and storage engines to skip reading SSTables that do not contain a requested ID."
    },
    {
        "doc_id": "doc_049",
        "title": "Reranking Fusion Depth Tradeoffs in Retrieval Systems",
        "source": "ir_textbook",
        "text": "When combining sparse and dense candidate lists, the fusion depth K determines how many top candidates are pooled. Increasing K from 20 to 100 improves candidate recall for multi-hop queries, but increases latency and memory overhead linearly."
    },
    {
        "doc_id": "doc_050",
        "title": "Cross-Platform Java Headless Execution and PyJNIus Bounds",
        "source": "systems_paper",
        "text": "Running Lucene via Pyserini in Python requires configuring JAVA_HOME and loading dynamic shared libraries via PyJNIus. Missing JVM installations cause immediate runtime crashes, highlighting the portability benefits of pure-Python vector and BM25 implementations like bm25s."
    }
]


def prepare_processed_corpus(output_path: Path = Path("data/processed/corpus.jsonl")) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in DOCUMENTS:
            record = {
                "doc_id": item["doc_id"],
                "text": item["text"],
                "metadata": {
                    "title": item["title"],
                    "source": item["source"],
                    "chunk_id": 1,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Prepared {len(DOCUMENTS)} documents at: {output_path}")
    return output_path


if __name__ == "__main__":
    prepare_processed_corpus()
