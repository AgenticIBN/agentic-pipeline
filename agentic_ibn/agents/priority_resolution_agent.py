from __future__ import annotations

import json
from typing import Any

from ..schemas import ActiveIntentRecord, NetworkConfig, ResolutionResult, ResolutionStrategy
from .runtime import build_groq_agent, coerce_response_model

PRIORITY_INSTRUCTIONS = [
    "You are the Priority Resolution Agent implementing CRS-1.",
    "The deterministic CRS-1 result in the prompt is authoritative.",
    "Do not change the winner, participant IDs, suppressed IDs, final configuration, or strategy.",
    "You may clarify the rationale without changing the decision.",
    "Return only the ResolutionResult schema.",
]


def create_priority_resolution_agent(model_id: str) -> Any:
    return build_groq_agent(
        model_id=model_id,
        name="Priority Resolution Agent",
        role="Applies CRS-1 priority arbitration to non-mergeable conflicts.",
        instructions=PRIORITY_INSTRUCTIONS,
        output_schema=ResolutionResult,
        structured_outputs=True,
        markdown=False,
        retries=2,
    )


class PriorityResolver:
    def resolve(self, participants: list[ActiveIntentRecord], meta_agent_id: str) -> ResolutionResult:
        if not participants:
            raise ValueError("Priority resolution requires at least one participant")
        ordered = sorted(
            participants,
            key=lambda item: (-item.intent.priority.weight, item.created_at, item.result_id),
        )
        winner = ordered[0]
        losers = [item.result_id for item in ordered[1:]]
        return ResolutionResult(
            meta_agent_id=meta_agent_id,
            strategy=ResolutionStrategy.PRIORITY,
            final_configuration=NetworkConfig.model_validate(winner.proposed_configuration.model_dump()),
            participant_result_ids=[item.result_id for item in ordered],
            winner_result_id=winner.result_id,
            suppressed_result_ids=losers,
            rationale=(
                f"CRS-1 selected {winner.result_id} at priority {winner.intent.priority.value}. "
                "Equal priorities are resolved by earliest registration time, then lexicographic result ID."
            ),
        )


def _signature(result: ResolutionResult) -> tuple[Any, ...]:
    return (
        result.meta_agent_id,
        result.strategy.value,
        tuple(result.participant_result_ids),
        result.winner_result_id,
        tuple(result.suppressed_result_ids),
        json.dumps(result.final_configuration.model_dump(mode="json"), sort_keys=True),
    )


def run_priority_resolution_agent(
    participants: list[ActiveIntentRecord],
    *,
    meta_agent_id: str,
    model_id: str | None,
    use_llm: bool,
) -> ResolutionResult:
    deterministic = PriorityResolver().resolve(participants, meta_agent_id)
    if not use_llm or not model_id:
        return deterministic
    try:
        agent = create_priority_resolution_agent(model_id)
        response = agent.run(
            "Return the authoritative CRS-1 result, changing only rationale wording if useful.\n\n"
            + json.dumps(deterministic.model_dump(mode="json"), indent=2)
        )
        audited = coerce_response_model(response, ResolutionResult)
        return audited if _signature(audited) == _signature(deterministic) else deterministic
    except Exception:
        return deterministic
