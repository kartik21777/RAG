"""
Dataset I/O and validation utilities for CRAG.
Supports JSONL serialization/deserialization for Documents, Queries,
and Evaluator Examples with deduplication and integrity checks.
"""

import json
from pathlib import Path
from typing import List, Union, Set
from src.core.schemas import Document, Query, EvaluatorExample


def load_corpus_from_jsonl(file_path: Union[str, Path]) -> List[Document]:
    """
    Loads documents from a JSONL file, validating each record against the Document schema
    and verifying that document IDs are unique and stable.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    documents: List[Document] = []
    seen_ids: Set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num} in {path}: {e}")

            doc = Document.model_validate(data)
            if doc.doc_id in seen_ids:
                raise ValueError(f"Duplicate document ID detected at line {line_num}: '{doc.doc_id}'")
            seen_ids.add(doc.doc_id)
            documents.append(doc)

    return documents


def save_corpus_to_jsonl(documents: List[Document], file_path: Union[str, Path]) -> None:
    """Saves a list of Document objects to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json() + "\n")


def load_queries_from_jsonl(file_path: Union[str, Path]) -> List[Query]:
    """Loads and validates a list of queries from a JSONL file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Queries file not found: {path}")

    queries: List[Query] = []
    seen_ids: Set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            query = Query.model_validate(data)
            if query.query_id in seen_ids:
                raise ValueError(f"Duplicate query ID at line {line_num}: '{query.query_id}'")
            seen_ids.add(query.query_id)
            queries.append(query)

    return queries


def save_queries_to_jsonl(queries: List[Query], file_path: Union[str, Path]) -> None:
    """Saves a list of Query objects to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(q.model_dump_json() + "\n")


def load_evaluator_examples_from_jsonl(file_path: Union[str, Path]) -> List[EvaluatorExample]:
    """Loads and strictly validates EvaluatorExample records from a JSONL file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluator examples file not found: {path}")

    examples: List[EvaluatorExample] = []
    seen_ids: Set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            example = EvaluatorExample.model_validate(data)
            if example.example_id in seen_ids:
                raise ValueError(f"Duplicate example ID at line {line_num}: '{example.example_id}'")
            seen_ids.add(example.example_id)
            examples.append(example)

    return examples


def save_evaluator_examples_to_jsonl(
    examples: List[EvaluatorExample], file_path: Union[str, Path]
) -> None:
    """Saves a list of EvaluatorExample objects to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
