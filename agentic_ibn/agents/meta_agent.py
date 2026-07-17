from __future__ import annotations

import json
from uuid import uuid4
from typing import Any

from ..schemas import (
    ActiveIntentRecord,
    ConflictReport,
    OptimizationResult,
    ResolutionDecision,
    ResolutionResult,
    ResolutionStrategy,
)
from .priority_resolution_agent import run_priority_resolution_agent
from .runtime import build_groq_agent, coerce_response_model
from .weighted_merge_agent import run_weighted_merge_agent

META_INSTRUCTIONS = [
    "You are the dynamic Meta Agent for conflict arbitration.",
    "Choose exactly one strategy: PRIORITY or WEIGHTED_MERGE.",
    "Use PRIORITY for any non-mergeable, boolean, HIGH, or CRITICAL conflict.",
    "Use WEIGHTED_MERGE only when every conflict is mergeable and no safety rule requires priority arbitration.",
    "Respect an explicit requested strategy unless it violates the non-mergeable safety rule.",
    "Return only the ResolutionDecision schema and a concise rationale.",
]


def create_meta_agent(model_id: str) -> Any:
    return build_groq_agent(
        model_id=model_id,
        name="Meta Agent",
        role="Selects the conflict-resolution policy and delegates to the matching specialist.",
        description="Dynamically used only when the conflict report is non-empty.",
        instructions=META_INSTRUCTIONS,
        output_schema=ResolutionDecision,
        structured_outputs=True,
        markdown=False,
        retries=2,
    )


def optimization_to_record(result: OptimizationResult) -> ActiveIntentRecord:
    return ActiveIntentRecord(
        result_id=result.result_id,
        created_at=result.created_at,
        intent=result.intent,
        baseline_configuration=result.baseline_configuration,
        proposed_configuration=result.final_configuration,
        predicted_kpis=result.predicted_kpis,
        hard_constraints=result.hard_constraints,
    )


class ConflictResolutionCoordinator:
    """Runs the Agno Meta Agent and the selected Agno resolution specialist."""

    def __init__(self, model_id: str | None, use_llm: bool = True):
        self.model_id = model_id
        self.use_llm = use_llm

    @staticmethod
    def _safe_strategy(report: ConflictReport, requested_strategy: str) -> str:
        requested = requested_strategy.upper()
        if requested not in {"AUTO", "PRIORITY", "WEIGHTED_MERGE"}:
            raise ValueError(f"Unsupported resolution strategy: {requested_strategy}")
        strategy = report.recommended_strategy if requested == "AUTO" else requested
        if strategy == "WEIGHTED_MERGE" and any(not detail.mergeable for detail in report.details):
            return "PRIORITY"
        return strategy

    def _agent_strategy(self, report: ConflictReport, requested_strategy: str) -> str:
        safe = self._safe_strategy(report, requested_strategy)
        if not self.use_llm or not self.model_id:
            return safe
        payload = {
            "requested_strategy": requested_strategy,
            "deterministic_recommended_strategy": report.recommended_strategy,
            "conflict_report": report.model_dump(mode="json"),
            "safety_rule": "WEIGHTED_MERGE is forbidden if any detail is non-mergeable.",
        }
        try:
            response = create_meta_agent(self.model_id).run(
                "Select the conflict-resolution strategy from this JSON context:\n"
                + json.dumps(payload, indent=2)
            )
            decision = coerce_response_model(response, ResolutionDecision)
            chosen = decision.strategy
            if chosen == "WEIGHTED_MERGE" and any(not detail.mergeable for detail in report.details):
                return "PRIORITY"
            if requested_strategy.upper() in {"PRIORITY", "WEIGHTED_MERGE"}:
                return safe
            return chosen
        except Exception:
            return safe

    def resolve(
        self,
        new_result: OptimizationResult,
        active_records: list[ActiveIntentRecord],
        report: ConflictReport,
        requested_strategy: str = "AUTO",
    ) -> ResolutionResult:
        if not report.conflict_detected:
            return ResolutionResult(
                meta_agent_id=None,
                strategy=ResolutionStrategy.NONE,
                final_configuration=new_result.final_configuration,
                participant_result_ids=[new_result.result_id],
                winner_result_id=new_result.result_id,
                suppressed_result_ids=[],
                rationale="No overlapping conflict was detected; no Meta Agent arbitration was required.",
            )

        participant_ids = {detail.active_result_id for detail in report.details}
        participants = [record for record in active_records if record.result_id in participant_ids]
        participants.append(optimization_to_record(new_result))
        strategy = self._agent_strategy(report, requested_strategy)
        meta_agent_id = f"meta_{uuid4().hex[:10]}"

        if strategy == "PRIORITY":
            return run_priority_resolution_agent(
                participants,
                meta_agent_id=meta_agent_id,
                model_id=self.model_id,
                use_llm=self.use_llm,
            )
        if strategy == "WEIGHTED_MERGE":
            return run_weighted_merge_agent(
                participants,
                meta_agent_id=meta_agent_id,
                model_id=self.model_id,
                use_llm=self.use_llm,
            )
        raise RuntimeError(f"Unexpected resolved strategy: {strategy}")
