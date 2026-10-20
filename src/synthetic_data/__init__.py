"""
Synthetic training data generation pipeline for the CRAG Evaluator.
Coordinates positive synthesis, partial-evidence queries, negative mining,
and quality gate validation.
"""

from src.synthetic_data.base import BaseEvaluatorJudge, BaseSyntheticGenerator
from src.synthetic_data.generator import LLMSyntheticGenerator, MockDeterministicGenerator
from src.synthetic_data.judge import (
    JUDGE_PROMPT_VERSION,
    JudgeEvaluationError,
    JudgeParseError,
    JudgeValidationError,
    LLMEvaluatorJudge,
    MockEvaluatorJudge,
)
from src.synthetic_data.negative_sampler import NegativeSampler
from src.synthetic_data.pipeline import SyntheticDataPipeline
from src.synthetic_data.validator import ValidationReport, validate_evaluator_dataset

__all__ = [
    "BaseSyntheticGenerator",
    "BaseEvaluatorJudge",
    "MockDeterministicGenerator",
    "LLMSyntheticGenerator",
    "LLMEvaluatorJudge",
    "MockEvaluatorJudge",
    "JudgeEvaluationError",
    "JudgeParseError",
    "JudgeValidationError",
    "JUDGE_PROMPT_VERSION",
    "NegativeSampler",
    "SyntheticDataPipeline",
    "ValidationReport",
    "validate_evaluator_dataset",
]
