"""
Command-line interface for synthetic evaluator data generation.
Usage:
    python -m src.synthetic_data.generate --corpus-path data/fixtures/corpus.jsonl --output-path data/synthetic/evaluator_train.jsonl
"""

import argparse
import os
from pathlib import Path
from src.core.dataset import load_corpus_from_jsonl
from src.retrieval.sparse import BM25Retriever
from src.synthetic_data.generator import MockDeterministicGenerator
from src.synthetic_data.judge import LLMEvaluatorJudge, MockEvaluatorJudge
from src.synthetic_data.pipeline import SyntheticDataPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluator training data for CRAG.")
    parser.add_argument(
        "--corpus-path",
        type=str,
        default="data/fixtures/corpus.jsonl",
        help="Path to source corpus JSONL file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/synthetic/evaluator_train.jsonl",
        help="Path to write generated evaluator JSONL dataset.",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="none",
        choices=["none", "mock", "llm"],
        help=(
            "Judging mode: 'none' (deterministic construction), 'mock' (offline rule-based judge), "
            "or 'llm' (requires injecting a live LLM provider callable via Python API)."
        ),
    )
    parser.add_argument(
        "--num-hard",
        type=int,
        default=1,
        help="Number of retriever-mined hard negatives per query.",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=1,
        help="Number of random negatives per query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    corpus_path = Path(args.corpus_path)
    output_path = Path(args.output_path)

    print(f"Loading corpus from: {corpus_path}")
    corpus = load_corpus_from_jsonl(corpus_path)
    print(f"Loaded {len(corpus)} documents.")

    print("Building BM25 sparse index for hard negative mining...")
    retriever = BM25Retriever()
    retriever.index(corpus)

    judge = None
    if args.judge == "mock":
        print("Using MockEvaluatorJudge for independent offline judging...")
        judge = MockEvaluatorJudge()
    elif args.judge == "llm":
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                "Live LLM judge requested but LLM_API_KEY is not configured. "
                "Please set LLM_API_KEY in .env or environment variables, or use --judge mock / --judge none."
            )
        raise NotImplementedError(
            "Direct live API calls via CLI require a configured provider adapter. "
            "LLMEvaluatorJudge is provider-agnostic and expects an injected callable `llm_fn: Callable[[str], str]`. "
            "To execute live evaluations, instantiate LLMEvaluatorJudge(llm_fn=...) with your chosen provider adapter "
            "(e.g. OpenAI or Anthropic SDK client), or run the CLI with '--judge mock' (offline rule-based judge) "
            "or '--judge none' (deterministic construction)."
        )

    print(f"Initializing synthetic data generation pipeline (mode: {'judged' if judge else 'deterministic'})...")
    pipeline = SyntheticDataPipeline(
        corpus=corpus,
        generator=MockDeterministicGenerator(),
        retriever=retriever,
        judge=judge,
        seed=args.seed,
    )

    print("Generating and validating synthetic dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset, report = pipeline.generate_and_save(
        output_path=output_path,
        num_hard_negatives_per_query=args.num_hard,
        num_random_negatives_per_query=args.num_random,
    )

    print(report.summary())
    if pipeline.review_queue:
        print(f"Items in review queue (unverified / failed judgments): {len(pipeline.review_queue)}")
    print(f"\nSuccessfully wrote {len(dataset)} examples to: {output_path}")


if __name__ == "__main__":
    main()
