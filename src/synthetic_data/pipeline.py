"""
End-to-end synthetic dataset generation pipeline for training the CRAG Evaluator.
Coordinates query synthesis, negative sampling (hard & random), label assignment,
and dataset validation. Supports both deterministic offline mode and independent LLM judging.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.dataset import save_evaluator_examples_to_jsonl
from src.core.schemas import Document, EvaluatorExample, LabelEnum, Query
from src.synthetic_data.base import BaseEvaluatorJudge, BaseSyntheticGenerator
from src.synthetic_data.generator import MockDeterministicGenerator
from src.synthetic_data.judge import JudgeEvaluationError
from src.synthetic_data.negative_sampler import NegativeSampler
from src.synthetic_data.validator import ValidationReport, validate_evaluator_dataset


class SyntheticDataPipeline:
    """
    Coordinates synthetic training pair generation for Member 2's DeBERTa evaluator.
    Produces balanced, audited datasets containing:
    - (1, 1) TRUST: Relevant & Sufficient
    - (1, 0) VERIFY-MORE: Relevant & Insufficient
    - (0, 0) REJECT-SEARCH: Retriever-mined Hard Negatives & Random Negatives

    Supports two distinct operational modes:
    1. Deterministic Mode (judge=None): Assigns labels by construction strategy.
    2. Judged Mode (judge provided): Independently evaluates all candidate pairs
       (including mandatory negative verification) with strict label policy enforcement.
    """

    def __init__(
        self,
        corpus: List[Document],
        generator: Optional[BaseSyntheticGenerator] = None,
        retriever: Optional[Any] = None,
        judge: Optional[BaseEvaluatorJudge] = None,
        seed: int = 42,
    ):
        if not corpus:
            raise ValueError("Corpus cannot be empty.")
        self.corpus = corpus
        self.generator = generator if generator is not None else MockDeterministicGenerator()
        self.retriever = retriever
        self.judge = judge
        self.sampler = NegativeSampler(corpus, seed=seed)
        self.seed = seed
        self.review_queue: List[Dict[str, Any]] = []

    def generate_dataset(
        self,
        include_sufficient: bool = True,
        include_insufficient: bool = True,
        num_hard_negatives_per_query: int = 1,
        num_random_negatives_per_query: int = 1,
        search_top_k: int = 10,
    ) -> List[EvaluatorExample]:
        """
        Executes generation over the corpus and returns validated EvaluatorExamples.
        """
        examples: List[EvaluatorExample] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        for doc_idx, doc in enumerate(self.corpus, start=1):
            base_qid = f"syn_q_{doc_idx:03d}"

            # 1. Generate Relevant & Sufficient pair
            if include_sufficient:
                suff_query, suff_rationale = self.generator.generate_sufficient_query(doc)
                suff_qid = f"{base_qid}_suff"

                if self.judge is not None:
                    try:
                        intent_fit, suff, rationale = self.judge.evaluate(
                            query_text=suff_query,
                            doc_text=doc.text,
                        )
                        ex_type = (
                            "relevant_sufficient"
                            if (intent_fit, suff) == (1, 1)
                            else ("relevant_insufficient" if (intent_fit, suff) == (1, 0) else "hard_negative_topic")
                        )
                        examples.append(
                            EvaluatorExample(
                                example_id="temp",
                                query_id=suff_qid,
                                query_text=suff_query,
                                doc_id=doc.doc_id,
                                doc_text=doc.text,
                                intent_fit=LabelEnum(intent_fit),
                                evidence_sufficiency=LabelEnum(suff),
                                example_type=ex_type,
                                rationale=rationale,
                                provenance={
                                    "mode": "llm_judged",
                                    "generator": getattr(self.generator, "model_name", "deterministic_rule"),
                                    "source_doc_id": doc.doc_id,
                                    "judge_model": getattr(self.judge, "model_name", "custom_judge"),
                                    "prompt_version": getattr(self.judge, "prompt_version", "v1.0"),
                                    "timestamp": timestamp,
                                },
                            )
                        )
                    except JudgeEvaluationError as err:
                        self.review_queue.append({
                            "query_id": suff_qid,
                            "query_text": suff_query,
                            "doc_id": doc.doc_id,
                            "target_type": "sufficient",
                            "error": str(err),
                            "timestamp": timestamp,
                        })
                else:
                    # Deterministic construction mode
                    examples.append(
                        EvaluatorExample(
                            example_id="temp",
                            query_id=suff_qid,
                            query_text=suff_query,
                            doc_id=doc.doc_id,
                            doc_text=doc.text,
                            intent_fit=LabelEnum.POSITIVE,
                            evidence_sufficiency=LabelEnum.POSITIVE,
                            example_type="relevant_sufficient",
                            rationale=suff_rationale,
                            provenance={
                                "mode": "deterministic_construction",
                                "generator": getattr(self.generator, "model_name", "deterministic_rule"),
                                "source_doc_id": doc.doc_id,
                                "timestamp": timestamp,
                            },
                        )
                    )

                # Sample negatives for sufficient query
                query_obj = Query(
                    query_id=suff_qid,
                    query_text=suff_query,
                    expected_doc_ids=[doc.doc_id],
                )
                examples.extend(
                    self._sample_negatives_for_query(
                        query_obj=query_obj,
                        num_hard=num_hard_negatives_per_query,
                        num_random=num_random_negatives_per_query,
                        search_top_k=search_top_k,
                        timestamp=timestamp,
                    )
                )

            # 2. Generate Relevant & Insufficient pair
            if include_insufficient:
                insuff_query, insuff_rationale = self.generator.generate_insufficient_query(doc)
                insuff_qid = f"{base_qid}_insuff"

                if self.judge is not None:
                    try:
                        intent_fit, suff, rationale = self.judge.evaluate(
                            query_text=insuff_query,
                            doc_text=doc.text,
                        )
                        ex_type = (
                            "relevant_insufficient"
                            if (intent_fit, suff) == (1, 0)
                            else ("relevant_sufficient" if (intent_fit, suff) == (1, 1) else "hard_negative_topic")
                        )
                        examples.append(
                            EvaluatorExample(
                                example_id="temp",
                                query_id=insuff_qid,
                                query_text=insuff_query,
                                doc_id=doc.doc_id,
                                doc_text=doc.text,
                                intent_fit=LabelEnum(intent_fit),
                                evidence_sufficiency=LabelEnum(suff),
                                example_type=ex_type,
                                rationale=rationale,
                                provenance={
                                    "mode": "llm_judged",
                                    "generator": getattr(self.generator, "model_name", "deterministic_rule"),
                                    "source_doc_id": doc.doc_id,
                                    "judge_model": getattr(self.judge, "model_name", "custom_judge"),
                                    "prompt_version": getattr(self.judge, "prompt_version", "v1.0"),
                                    "timestamp": timestamp,
                                },
                            )
                        )
                    except JudgeEvaluationError as err:
                        self.review_queue.append({
                            "query_id": insuff_qid,
                            "query_text": insuff_query,
                            "doc_id": doc.doc_id,
                            "target_type": "insufficient",
                            "error": str(err),
                            "timestamp": timestamp,
                        })
                else:
                    # Deterministic construction mode
                    examples.append(
                        EvaluatorExample(
                            example_id="temp",
                            query_id=insuff_qid,
                            query_text=insuff_query,
                            doc_id=doc.doc_id,
                            doc_text=doc.text,
                            intent_fit=LabelEnum.POSITIVE,
                            evidence_sufficiency=LabelEnum.NEGATIVE,
                            example_type="relevant_insufficient",
                            rationale=insuff_rationale,
                            provenance={
                                "mode": "deterministic_construction",
                                "generator": getattr(self.generator, "model_name", "deterministic_rule"),
                                "source_doc_id": doc.doc_id,
                                "timestamp": timestamp,
                            },
                        )
                    )

        # Assign sequential, gapless example_ids
        for idx, ex in enumerate(examples, start=1):
            ex.example_id = f"eval_syn_{idx:04d}"

        return examples

    def _sample_negatives_for_query(
        self,
        query_obj: Query,
        num_hard: int,
        num_random: int,
        search_top_k: int,
        timestamp: str,
    ) -> List[EvaluatorExample]:
        neg_examples: List[EvaluatorExample] = []

        # 1. Hard negatives (mined via retriever)
        hard_neg_docs = []
        if num_hard > 0 and self.retriever is not None:
            hard_neg_docs = self.sampler.sample_hard_negatives(
                query=query_obj,
                retriever=self.retriever,
                search_top_k=search_top_k,
                num_samples=num_hard,
            )
            for h_doc in hard_neg_docs:
                if self.judge is not None:
                    # Mandatory hard-negative verification by independent judge
                    try:
                        intent_fit, suff, rationale = self.judge.evaluate(
                            query_text=query_obj.query_text,
                            doc_text=h_doc.text,
                        )
                        if (intent_fit, suff) == (0, 0):
                            ex_type = "hard_negative_topic"
                            strategy = "retriever_mined_hard_negative_verified"
                        elif (intent_fit, suff) == (1, 1):
                            ex_type = "relevant_sufficient"
                            strategy = "retriever_mined_candidate_judged_sufficient"
                        elif (intent_fit, suff) == (1, 0):
                            ex_type = "relevant_insufficient"
                            strategy = "retriever_mined_candidate_judged_insufficient"
                        else:
                            continue

                        neg_examples.append(
                            EvaluatorExample(
                                example_id="temp",
                                query_id=query_obj.query_id,
                                query_text=query_obj.query_text,
                                doc_id=h_doc.doc_id,
                                doc_text=h_doc.text,
                                intent_fit=LabelEnum(intent_fit),
                                evidence_sufficiency=LabelEnum(suff),
                                example_type=ex_type,
                                rationale=rationale,
                                provenance={
                                    "mode": "llm_judged",
                                    "strategy": strategy,
                                    "sampling_method": "retriever_mined_hard_negative",
                                    "retriever": type(self.retriever).__name__,
                                    "judge_model": getattr(self.judge, "model_name", "custom_judge"),
                                    "prompt_version": getattr(self.judge, "prompt_version", "v1.0"),
                                    "timestamp": timestamp,
                                },
                            )
                        )
                    except JudgeEvaluationError as err:
                        self.review_queue.append({
                            "query_id": query_obj.query_id,
                            "query_text": query_obj.query_text,
                            "doc_id": h_doc.doc_id,
                            "sampling_method": "retriever_mined_hard_negative",
                            "error": str(err),
                            "timestamp": timestamp,
                        })
                else:
                    # Deterministic construction mode
                    neg_examples.append(
                        EvaluatorExample(
                            example_id="temp",
                            query_id=query_obj.query_id,
                            query_text=query_obj.query_text,
                            doc_id=h_doc.doc_id,
                            doc_text=h_doc.text,
                            intent_fit=LabelEnum.NEGATIVE,
                            evidence_sufficiency=LabelEnum.NEGATIVE,
                            example_type="hard_negative_topic",
                            rationale=(
                                f"Document {h_doc.doc_id} was retrieved via high topical/lexical overlap "
                                f"for query '{query_obj.query_text}', but does not address the true intent "
                                "(intent_fit=0, evidence_sufficiency=0)."
                            ),
                            provenance={
                                "mode": "deterministic_construction",
                                "strategy": "retriever_mined_hard_negative",
                                "retriever": type(self.retriever).__name__,
                                "timestamp": timestamp,
                            },
                        )
                    )

        # 2. Random negatives (drawn from unrelated corpus docs)
        shortfall = max(0, num_hard - len(hard_neg_docs))
        effective_num_random = num_random + shortfall

        if effective_num_random > 0:
            rand_neg_docs = self.sampler.sample_random_negatives(
                query=query_obj,
                num_samples=effective_num_random,
                exclude_doc_ids={ex.doc_id for ex in neg_examples},
            )
            for r_doc in rand_neg_docs:
                if self.judge is not None:
                    # Mandatory random-negative verification by independent judge
                    try:
                        intent_fit, suff, rationale = self.judge.evaluate(
                            query_text=query_obj.query_text,
                            doc_text=r_doc.text,
                        )
                        if (intent_fit, suff) == (0, 0):
                            ex_type = "random_negative"
                            strategy = "random_corpus_sampling_verified"
                        elif (intent_fit, suff) == (1, 1):
                            ex_type = "relevant_sufficient"
                            strategy = "random_candidate_judged_sufficient"
                        elif (intent_fit, suff) == (1, 0):
                            ex_type = "relevant_insufficient"
                            strategy = "random_candidate_judged_insufficient"
                        else:
                            continue

                        neg_examples.append(
                            EvaluatorExample(
                                example_id="temp",
                                query_id=query_obj.query_id,
                                query_text=query_obj.query_text,
                                doc_id=r_doc.doc_id,
                                doc_text=r_doc.text,
                                intent_fit=LabelEnum(intent_fit),
                                evidence_sufficiency=LabelEnum(suff),
                                example_type=ex_type,
                                rationale=rationale,
                                provenance={
                                    "mode": "llm_judged",
                                    "strategy": strategy,
                                    "sampling_method": "random_corpus_sampling",
                                    "judge_model": getattr(self.judge, "model_name", "custom_judge"),
                                    "prompt_version": getattr(self.judge, "prompt_version", "v1.0"),
                                    "timestamp": timestamp,
                                },
                            )
                        )
                    except JudgeEvaluationError as err:
                        self.review_queue.append({
                            "query_id": query_obj.query_id,
                            "query_text": query_obj.query_text,
                            "doc_id": r_doc.doc_id,
                            "sampling_method": "random_corpus_sampling",
                            "error": str(err),
                            "timestamp": timestamp,
                        })
                else:
                    # Deterministic construction mode
                    neg_examples.append(
                        EvaluatorExample(
                            example_id="temp",
                            query_id=query_obj.query_id,
                            query_text=query_obj.query_text,
                            doc_id=r_doc.doc_id,
                            doc_text=r_doc.text,
                            intent_fit=LabelEnum.NEGATIVE,
                            evidence_sufficiency=LabelEnum.NEGATIVE,
                            example_type="random_negative",
                            rationale=(
                                f"Document {r_doc.doc_id} is an unrelated random passage from the corpus "
                                "and does not address the question (intent_fit=0, evidence_sufficiency=0)."
                            ),
                            provenance={
                                "mode": "deterministic_construction",
                                "strategy": "random_corpus_sampling",
                                "timestamp": timestamp,
                            },
                        )
                    )

        return neg_examples

    def generate_and_save(
        self,
        output_path: Union[str, Path],
        include_sufficient: bool = True,
        include_insufficient: bool = True,
        num_hard_negatives_per_query: int = 1,
        num_random_negatives_per_query: int = 1,
    ) -> Tuple[List[EvaluatorExample], ValidationReport]:
        """
        Executes generation, validates the dataset, and writes to target JSONL file.
        """
        dataset = self.generate_dataset(
            include_sufficient=include_sufficient,
            include_insufficient=include_insufficient,
            num_hard_negatives_per_query=num_hard_negatives_per_query,
            num_random_negatives_per_query=num_random_negatives_per_query,
        )

        report = validate_evaluator_dataset(dataset)
        if not report.is_valid:
            raise ValueError(f"Generated synthetic dataset failed validation: {report.errors}")

        save_evaluator_examples_to_jsonl(dataset, output_path)
        return dataset, report
