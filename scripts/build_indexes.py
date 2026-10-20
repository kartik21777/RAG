"""
Builds and persists hybrid sparse-dense indexes from a source corpus JSONL file.
Usage:
    python scripts/build_indexes.py --corpus-path data/processed/corpus.jsonl --output-dir indexes/hybrid
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script invocation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dataset import load_corpus_from_jsonl
from src.retrieval.facade import create_crag_retriever


def parse_args():
    parser = argparse.ArgumentParser(description="Build hybrid retrieval indexes for CRAG.")
    parser.add_argument(
        "--corpus-path",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file to index.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="indexes/hybrid",
        help="Directory to save the resulting hybrid index artifacts.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model name for dense vector indexing.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion smoothing constant.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    corpus_path = Path(args.corpus_path)
    output_dir = Path(args.output_dir)

    print(f"Loading corpus from: {corpus_path}")
    corpus = load_corpus_from_jsonl(corpus_path)
    print(f"Loaded {len(corpus)} documents. Initializing and building hybrid index...")

    facade = create_crag_retriever(
        corpus=corpus,
        dense_model_name=args.model_name,
        rrf_k=args.rrf_k,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    facade.save(output_dir)
    print(f"Successfully built and persisted hybrid index to: {output_dir}")


if __name__ == "__main__":
    main()
