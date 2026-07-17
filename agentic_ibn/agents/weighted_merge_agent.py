from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from ..schemas import ActiveIntentRecord, NetworkConfig, ResolutionResult, ResolutionStrategy, TxConfig
from .runtime import build_groq_agent, coerce_response_model

WEIGHTED_INSTRUCTIONS = [
    "You are the Weighted Merge Agent implementing CRS-2.",
    "The deterministic weighted-merge result in the prompt is authoritative.",
    "Do not change participant IDs, final configuration, strategy, or meta-agent ID.",
    "You may clarify the rationale without changing the decision.",
    "Return only the ResolutionResult schema.",
]


def create_weighted_merge_agent(model_id: str) -> Any:
    return build_groq_agent(
        model_id=model_id,
        name="Weighted Merge Agent",
        role="Applies CRS-2 priority-weighted merging to mergeable conflicts.",
        instructions=WEIGHTED_INSTRUCTIONS,
        output_schema=ResolutionResult,
        structured_outputs=True,
        markdown=False,
        retries=2,
    )


class WeightedMergeResolver:
    def resolve(self, participants: list[ActiveIntentRecord], meta_agent_id: str) -> ResolutionResult:
        if not participants:
            raise ValueError("Weighted merge requires at least one participant")

        hard_states: dict[int, list[tuple[bool, int, str]]] = defaultdict(list)
        for record in participants:
            for index, state in record.hard_constraints.tx_states.items():
                hard_states[index].append((state, record.intent.priority.weight, record.result_id))

        tx_configs: dict[str, TxConfig] = {}
        for index in range(4):
            weighted_on = total_weight = 0.0
            weighted_power = weighted_azimuth = weighted_elevation = 0.0
            for record in participants:
                weight = float(record.intent.priority.weight)
                tx = record.proposed_configuration.tx(index)
                total_weight += weight
                weighted_on += weight * float(tx.on)
                weighted_power += weight * float(tx.power_dbm)
                weighted_azimuth += weight * float(tx.azimuth_delta_deg)
                weighted_elevation += weight * float(tx.elevation_delta_deg)

            if hard_states.get(index):
                constraints = sorted(hard_states[index], key=lambda item: (-item[1], item[2]))
                on = constraints[0][0]
            else:
                on = weighted_on >= total_weight / 2.0

            tx_configs[f"tx{index}"] = TxConfig(
                on=on,
                power_dbm=weighted_power / total_weight if on else 0.0,
                azimuth_delta_deg=weighted_azimuth / total_weight if on else 0.0,
                elevation_delta_deg=weighted_elevation / total_weight if on else 0.0,
            )

        highest = max(participants, key=lambda item: (item.intent.priority.weight, -item.created_at.timestamp()))
        final_config = NetworkConfig(
            **tx_configs,
            k_users=highest.proposed_configuration.k_users,
            user_set_id=highest.proposed_configuration.user_set_id,
            coverage_threshold_dbm=highest.proposed_configuration.coverage_threshold_dbm,
        )
        return ResolutionResult(
            meta_agent_id=meta_agent_id,
            strategy=ResolutionStrategy.WEIGHTED_MERGE,
            final_configuration=final_config,
            participant_result_ids=[item.result_id for item in participants],
            winner_result_id=None,
            suppressed_result_ids=[],
            rationale=(
                "CRS-2 merged absolute numeric values with priority weights LOW=1, MEDIUM=2, "
                "HIGH=3, CRITICAL=4. Explicit hard TX states override the weighted boolean vote."
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


def run_weighted_merge_agent(
    participants: list[ActiveIntentRecord],
    *,
    meta_agent_id: str,
    model_id: str | None,
    use_llm: bool,
) -> ResolutionResult:
    deterministic = WeightedMergeResolver().resolve(participants, meta_agent_id)
    if not use_llm or not model_id:
        return deterministic
    try:
        agent = create_weighted_merge_agent(model_id)
        response = agent.run(
            "Return the authoritative CRS-2 result, changing only rationale wording if useful.\n\n"
            + json.dumps(deterministic.model_dump(mode="json"), indent=2)
        )
        audited = coerce_response_model(response, ResolutionResult)
        return audited if _signature(audited) == _signature(deterministic) else deterministic
    except Exception:
        return deterministic
