"""
Quality gate and validation routines for synthetic evaluator datasets.
Verifies schema contracts, label validity, duplicate avoidance, and class balance.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from src.core.schemas import CRAGAction, EvaluatorExample, LabelEnum


@dataclass
class ValidationReport:
    """Detailed summary of evaluator dataset quality and distribution."""
    is_valid: bool
    total_count: int
    action_counts: Dict[str, int]
    example_type_counts: Dict[str, int]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== Evaluator Dataset Validation Report: {status} ===",
            f"Total Examples: {self.total_count}",
            f"Action Breakdown:",
        ]
        for action, cnt in self.action_counts.items():
            pct = (cnt / self.total_count * 100.0) if self.total_count > 0 else 0.0
            lines.append(f"  - {action}: {cnt} ({pct:.1f}%)")

        lines.append("Example Type Breakdown:")
        for etype, cnt in self.example_type_counts.items():
            pct = (cnt / self.total_count * 100.0) if self.total_count > 0 else 0.0
            lines.append(f"  - {etype}: {cnt} ({pct:.1f}%)")

        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  [ERROR] {err}")

        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  [WARN] {warn}")

        return "\n".join(lines)


def validate_evaluator_dataset(examples: List[EvaluatorExample]) -> ValidationReport:
    """
    Performs comprehensive validation across an evaluator training dataset.

    Args:
        examples: List of EvaluatorExample objects.

    Returns:
        ValidationReport containing pass/fail status, counts, errors, and warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not examples:
        return ValidationReport(
            is_valid=False,
            total_count=0,
            action_counts={},
            example_type_counts={},
            errors=["Dataset is empty."],
        )

    seen_ids = set()
    action_counts: Dict[str, int] = {
        CRAGAction.TRUST.value: 0,
        CRAGAction.VERIFY_MORE.value: 0,
        CRAGAction.REJECT_SEARCH.value: 0,
    }
    example_type_counts: Dict[str, int] = {}

    for idx, ex in enumerate(examples):
        # 1. Unique example_id check
        if ex.example_id in seen_ids:
            errors.append(f"Duplicate example_id detected at index {idx}: '{ex.example_id}'")
        seen_ids.add(ex.example_id)

        # 2. Strict label policy check: (0, 1) is unconditionally prohibited
        if ex.intent_fit == LabelEnum.NEGATIVE and ex.evidence_sufficiency == LabelEnum.POSITIVE:
            errors.append(
                f"Prohibited label combination (0, 1) found in example '{ex.example_id}'."
            )

        # 3. String content checks
        if not ex.query_text.strip():
            errors.append(f"Empty query_text in example '{ex.example_id}'.")
        if not ex.doc_text.strip():
            errors.append(f"Empty doc_text in example '{ex.example_id}'.")
        if not ex.rationale.strip():
            warnings.append(f"Empty or missing rationale in example '{ex.example_id}'.")

        # 4. Action and type tallies
        action_name = ex.crag_action.value
        action_counts[action_name] = action_counts.get(action_name, 0) + 1

        etype = ex.example_type
        example_type_counts[etype] = example_type_counts.get(etype, 0) + 1

    # 5. Class balance warnings
    total = len(examples)
    for action, cnt in action_counts.items():
        if cnt == 0:
            warnings.append(f"Class '{action}' has 0 examples in dataset.")
        elif (cnt / total) < 0.10:
            warnings.append(f"Class '{action}' represents less than 10% of dataset ({cnt}/{total}).")

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        total_count=total,
        action_counts=action_counts,
        example_type_counts=example_type_counts,
        errors=errors,
        warnings=warnings,
    )
