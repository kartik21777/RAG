"""
Corrective Decision Router for Semantic- and Evidence-Aware CRAG.

Evaluates continuous decision scores emitted by the runtime evaluator, enforces
intent dominance (preventing low-intent candidates from being trusted), and aggregates
candidate decisions into a robust query-level corrective action (TRUST, VERIFY_MORE, REJECT_SEARCH).
"""

import math
from typing import Dict, List
from src.core.schemas import CRAGAction
from src.orchestration.schemas import CandidateDecision, EvaluatorPrediction, QueryDecision


class CorrectiveDecisionRouter:
    """
    Routes candidate evidence pairs and aggregates them into query-level corrective policies.

    Thresholding and Intent Dominance Rules:
    - (intent >= tau_intent, suff >= tau_suff) -> TRUST (1, 1)
    - (intent >= tau_intent, suff < tau_suff)  -> VERIFY_MORE (1, 0)
    - (intent < tau_intent, suff < tau_suff)   -> REJECT_SEARCH (0, 0)
    - (intent < tau_intent, suff >= tau_suff)  -> REJECT_SEARCH (0, 1) [Contradictory Override]
      A document with insufficient intent alignment CANNOT provide trusted or verified evidence,
      unconditionally overriding false-positive sufficiency predictions in the action decision,
      while preserving the independently thresholded sufficiency label (1) and flagging is_contradictory_override.
    """

    def __init__(
        self,
        intent_threshold: float = 0.5,
        sufficiency_threshold: float = 0.5,
        min_trusted_candidates: int = 1,
    ):
        """
        Args:
            intent_threshold: Threshold tau_intent in [0.0, 1.0] for intent-fit binarization.
            sufficiency_threshold: Threshold tau_suff in [0.0, 1.0] for evidence-sufficiency binarization.
            min_trusted_candidates: Minimum number of TRUST candidates required for query-level TRUST.
        """
        self._validate_threshold("intent_threshold", intent_threshold)
        self._validate_threshold("sufficiency_threshold", sufficiency_threshold)
        if min_trusted_candidates < 1:
            raise ValueError(f"min_trusted_candidates must be >= 1, got {min_trusted_candidates}")

        self.intent_threshold = float(intent_threshold)
        self.sufficiency_threshold = float(sufficiency_threshold)
        self.min_trusted_candidates = min_trusted_candidates

    @staticmethod
    def _validate_threshold(name: str, value: float) -> None:
        if (
            not isinstance(value, (int, float))
            or math.isnan(value)
            or math.isinf(value)
            or value < 0.0
            or value > 1.0
        ):
            raise ValueError(f"{name} must be a finite number in [0.0, 1.0], got {value}")

    def route_candidate(self, prediction: EvaluatorPrediction) -> CandidateDecision:
        """
        Routes an individual candidate prediction according to decision thresholds
        and the intent dominance rule.

        Independent label semantics:
        - intent_fit_label is 1 if intent_fit_score >= intent_threshold, else 0.
        - evidence_sufficiency_label is 1 if evidence_sufficiency_score >= sufficiency_threshold, else 0.
        - action is governed by Intent Dominance: low intent fit unconditionally forces REJECT_SEARCH.
        """
        if not isinstance(prediction, EvaluatorPrediction):
            raise TypeError(f"prediction must be EvaluatorPrediction, got {type(prediction).__name__}")

        raw_intent_pass = prediction.intent_fit_score >= self.intent_threshold
        raw_suff_pass = prediction.evidence_sufficiency_score >= self.sufficiency_threshold

        intent_label = 1 if raw_intent_pass else 0
        suff_label = 1 if raw_suff_pass else 0

        if not raw_intent_pass:
            # Low intent fit: Intent Dominance mandates REJECT_SEARCH
            action = CRAGAction.REJECT_SEARCH
            if raw_suff_pass:
                # Contradictory condition: intent failed, but sufficiency met threshold.
                # Intent dominance overrides routing action to REJECT_SEARCH while preserving suff_label=1.
                is_contradictory = True
                rationale = (
                    f"Contradictory prediction: sufficiency score ({prediction.evidence_sufficiency_score:.3f}) "
                    f"met threshold ({self.sufficiency_threshold}), but intent score ({prediction.intent_fit_score:.3f}) "
                    f"failed threshold ({self.intent_threshold}). Intent dominance override enforced in action: REJECT_SEARCH."
                )
            else:
                # Normal rejection: both failed
                is_contradictory = False
                rationale = (
                    f"Candidate rejected: intent score ({prediction.intent_fit_score:.3f}) below threshold ({self.intent_threshold}) "
                    f"and sufficiency score ({prediction.evidence_sufficiency_score:.3f}) below threshold ({self.sufficiency_threshold})."
                )
        else:
            # Intent passed
            if raw_suff_pass:
                action = CRAGAction.TRUST
                is_contradictory = False
                rationale = (
                    f"Candidate trusted: intent score ({prediction.intent_fit_score:.3f} >= {self.intent_threshold}) "
                    f"and sufficiency score ({prediction.evidence_sufficiency_score:.3f} >= {self.sufficiency_threshold})."
                )
            else:
                action = CRAGAction.VERIFY_MORE
                is_contradictory = False
                rationale = (
                    f"Candidate requires verification: intent score ({prediction.intent_fit_score:.3f} >= {self.intent_threshold}) "
                    f"met threshold, but sufficiency score ({prediction.evidence_sufficiency_score:.3f} < {self.sufficiency_threshold}) is incomplete."
                )

        return CandidateDecision(
            prediction=prediction,
            action=action,
            intent_fit_label=intent_label,
            evidence_sufficiency_label=suff_label,
            rationale=rationale,
            is_contradictory_override=is_contradictory,
        )

    def route_candidates(self, predictions: List[EvaluatorPrediction]) -> List[CandidateDecision]:
        """
        Routes a list of candidate predictions independently.
        """
        return [self.route_candidate(pred) for pred in predictions]

    def aggregate_query_decisions(
        self,
        query: str,
        candidate_decisions: List[CandidateDecision],
    ) -> QueryDecision:
        """
        Aggregates candidate decisions into a query-level action.

        Independent Evaluation Rule:
        - If at least `min_trusted_candidates` achieve TRUST, overall action is TRUST.
          One rejected or partial candidate does NOT invalidate independent trusted evidence!
        - If no candidate achieves TRUST, but at least one candidate achieves VERIFY_MORE,
          overall action is VERIFY_MORE (refine partial candidates).
        - If all candidates are REJECT_SEARCH (or empty), overall action is REJECT_SEARCH (rewrite query).

        Validation & Duplicate Handling:
        - All candidate decisions must match the target query text.
        - Disagreeing/conflicting duplicate decisions for the same document ID raise ValueError.
        - Identical duplicate decisions for the same document ID also raise ValueError.
          All candidate document IDs must be strictly unique within a single query aggregation.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty or whitespace only.")

        clean_query = query.strip()

        if not candidate_decisions:
            return QueryDecision(
                query_text=clean_query,
                overall_action=CRAGAction.REJECT_SEARCH,
                candidate_decisions=[],
                trusted_doc_ids=[],
                verify_doc_ids=[],
                rejected_doc_ids=[],
                rationale="EMPTY_RETRIEVAL: Zero candidate documents retrieved. Unanswerable query or zero corpus match; trigger query rewriting.",
                empty_retrieval=True,
            )

        # 1. Validate query text consistency across candidate decisions
        for c in candidate_decisions:
            if c.prediction.query_text != clean_query:
                raise ValueError(
                    f"Mismatched query text in candidate decision for doc '{c.prediction.doc_id}': "
                    f"expected '{clean_query}', got '{c.prediction.query_text}'"
                )

        # 2. Audit duplicate candidate predictions - reject ALL duplicate document IDs
        doc_grouped: Dict[str, List[CandidateDecision]] = {}
        for c in candidate_decisions:
            doc_grouped.setdefault(c.prediction.doc_id, []).append(c)

        for d_id, c_list in doc_grouped.items():
            if len(c_list) > 1:
                actions = {c.action for c in c_list}
                if len(actions) > 1:
                    raise ValueError(
                        f"Conflicting duplicate predictions detected for document '{d_id}' in query '{clean_query}': "
                        f"disagreeing actions {[c.action.value for c in c_list]}. "
                        f"Duplicate candidate document IDs are strictly prohibited."
                    )
                raise ValueError(
                    f"Duplicate predictions detected for document '{d_id}' in query '{clean_query}'. "
                    f"All candidate document IDs must be unique within a single query."
                )

        trusted_cands = [c for c in candidate_decisions if c.action == CRAGAction.TRUST]
        verify_cands = [c for c in candidate_decisions if c.action == CRAGAction.VERIFY_MORE]
        rejected_cands = [c for c in candidate_decisions if c.action == CRAGAction.REJECT_SEARCH]

        trusted_ids = [c.prediction.doc_id for c in trusted_cands]
        verify_ids = [c.prediction.doc_id for c in verify_cands]
        rejected_ids = [c.prediction.doc_id for c in rejected_cands]

        if len(trusted_ids) >= self.min_trusted_candidates:
            overall_action = CRAGAction.TRUST
            rationale = (
                f"Accepted {len(trusted_ids)} trusted document(s) ({', '.join(trusted_ids)}) "
                f"out of {len(candidate_decisions)} total candidates. "
                f"Independent evidence sufficient for grounded generation."
            )
        elif len(verify_ids) > 0:
            overall_action = CRAGAction.VERIFY_MORE
            rationale = (
                f"No candidate met full sufficiency, but found {len(verify_ids)} relevant candidate(s) "
                f"({', '.join(verify_ids)}) requiring evidence refinement."
            )
        else:
            overall_action = CRAGAction.REJECT_SEARCH
            rationale = (
                f"ALL_CANDIDATES_REJECTED: All {len(rejected_ids)} unique candidate(s) were rejected. "
                f"Zero usable evidence in retrieved candidates; trigger query rewriting."
            )

        return QueryDecision(
            query_text=clean_query,
            overall_action=overall_action,
            candidate_decisions=candidate_decisions,
            trusted_doc_ids=trusted_ids,
            verify_doc_ids=verify_ids,
            rejected_doc_ids=rejected_ids,
            rationale=rationale,
            empty_retrieval=False,
        )
