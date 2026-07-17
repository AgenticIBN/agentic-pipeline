from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..schemas import (
    ActiveIntentRecord,
    ConflictDetail,
    ConflictReport,
    ConflictType,
    KPIName,
    NetworkConfig,
    OptimizationResult,
    Severity,
)
from .runtime import build_groq_agent, coerce_response_model

CONFLICT_INSTRUCTIONS = [
    "You are the Conflict Detector Agent in a conflict-aware IBN system.",
    "Review the deterministic predicate scan supplied in the prompt.",
    "Do not add or remove conflicts and do not change severity, mergeability, parameters, participant IDs, or strategy.",
    "You may only improve the wording of each explanation while preserving its exact operational meaning.",
    "Return only the ConflictReport schema.",
]

_KPI_TRADEOFFS = {
    frozenset({KPIName.TOTAL_TX_POWER, KPIName.RX_POWER}),
    frozenset({KPIName.TOTAL_TX_POWER, KPIName.COVERAGE}),
    frozenset({KPIName.TOTAL_TX_POWER, KPIName.THROUGHPUT_5P}),
    frozenset({KPIName.TOTAL_TX_POWER, KPIName.THROUGHPUT_RR_5P}),
    frozenset({KPIName.RX_POWER, KPIName.SINR}),
    frozenset({KPIName.COVERAGE, KPIName.SINR}),
}


def create_conflict_detector_agent(model_id: str) -> Any:
    return build_groq_agent(
        model_id=model_id,
        name="Conflict Detector Agent",
        role="Audits deterministic spatial, temporal, parameter, and KPI conflict predicates.",
        instructions=CONFLICT_INSTRUCTIONS,
        output_schema=ConflictReport,
        structured_outputs=True,
        markdown=False,
        retries=2,
    )


def _normalize_area(area: str) -> str:
    return " ".join(area.lower().replace("_", " ").split())


def _areas_overlap(left: str, right: str) -> bool:
    a = _normalize_area(left)
    b = _normalize_area(right)
    if not a or not b:
        return True
    return a == b or a in b or b in a or "global" in {a, b} or "all" in {a, b}


def _aware(value: datetime | None, default: datetime) -> datetime:
    if value is None:
        return default
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _times_overlap(active: ActiveIntentRecord, new: OptimizationResult) -> bool:
    minimum = datetime.min.replace(tzinfo=timezone.utc)
    maximum = datetime.max.replace(tzinfo=timezone.utc)
    left_start = _aware(active.intent.time_constraint_start, minimum)
    left_end = _aware(active.intent.time_constraint_end, maximum)
    right_start = _aware(new.intent.time_constraint_start, minimum)
    right_end = _aware(new.intent.time_constraint_end, maximum)
    return max(left_start, right_start) <= min(left_end, right_end)


def _flat_config(config: NetworkConfig) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for index in range(4):
        tx = config.tx(index)
        flat[f"tx{index}_on"] = tx.on
        flat[f"tx{index}_P_dBm"] = tx.power_dbm
        flat[f"tx{index}_dAz"] = tx.azimuth_delta_deg
        flat[f"tx{index}_dEl"] = tx.elevation_delta_deg
    return flat


def _changes(baseline: NetworkConfig, final: NetworkConfig) -> dict[str, Any]:
    before = _flat_config(baseline)
    after = _flat_config(final)
    changes: dict[str, Any] = {}
    for parameter, final_value in after.items():
        original = before[parameter]
        if isinstance(final_value, bool):
            if final_value != original:
                changes[parameter] = final_value
        else:
            delta = float(final_value) - float(original)
            if abs(delta) > 1e-9:
                changes[parameter] = delta
    return changes


def _severity_for_boolean(active: ActiveIntentRecord, new: OptimizationResult, parameter: str) -> Severity:
    index = int(parameter[2])
    if index in active.hard_constraints.tx_states or index in new.hard_constraints.tx_states:
        return Severity.CRITICAL
    if active.intent.priority.value == "CRITICAL" or new.intent.priority.value == "CRITICAL":
        return Severity.CRITICAL
    return Severity.HIGH


class ConflictDetector:
    """Deterministic safety kernel used and audited by the Agno agent."""

    def detect(
        self,
        new_result: OptimizationResult,
        active_records: list[ActiveIntentRecord],
    ) -> ConflictReport:
        details: list[ConflictDetail] = []
        new_changes = _changes(new_result.baseline_configuration, new_result.final_configuration)

        for active in active_records:
            if active.status not in {"ACTIVE", "SUPPRESSED"}:
                continue
            if not _areas_overlap(active.intent.target_area, new_result.intent.target_area):
                continue
            if not _times_overlap(active, new_result):
                continue

            active_changes = _changes(active.baseline_configuration, active.proposed_configuration)
            shared_parameters = sorted(set(active_changes) & set(new_changes))
            for parameter in shared_parameters:
                active_change = active_changes[parameter]
                new_change = new_changes[parameter]
                if parameter.endswith("_on"):
                    if bool(active_change) != bool(new_change):
                        details.append(
                            ConflictDetail(
                                active_result_id=active.result_id,
                                new_result_id=new_result.result_id,
                                conflict_type=ConflictType.BOOLEAN_CONFLICT,
                                severity=_severity_for_boolean(active, new_result, parameter),
                                parameter=parameter,
                                active_value=active_change,
                                new_value=new_change,
                                mergeable=False,
                                explanation="The intents require incompatible ON/OFF states for the same transmitter.",
                            )
                        )
                    continue

                active_delta = float(active_change)
                new_delta = float(new_change)
                if active_delta * new_delta < 0:
                    details.append(
                        ConflictDetail(
                            active_result_id=active.result_id,
                            new_result_id=new_result.result_id,
                            conflict_type=ConflictType.PARAMETER_CONFLICT,
                            severity=Severity.HIGH,
                            parameter=parameter,
                            active_value=active_delta,
                            new_value=new_delta,
                            mergeable=False,
                            explanation="The intents move the same parameter in opposite directions.",
                        )
                    )
                elif abs(active_delta - new_delta) > 1e-6:
                    details.append(
                        ConflictDetail(
                            active_result_id=active.result_id,
                            new_result_id=new_result.result_id,
                            conflict_type=ConflictType.RESOURCE_CONTENTION,
                            severity=Severity.MEDIUM,
                            parameter=parameter,
                            active_value=active_delta,
                            new_value=new_delta,
                            mergeable=True,
                            explanation="The intents move the same parameter in the same direction but request different magnitudes.",
                        )
                    )

            for left in set(active.intent.target_kpis):
                for right in set(new_result.intent.target_kpis):
                    if frozenset({left, right}) in _KPI_TRADEOFFS:
                        details.append(
                            ConflictDetail(
                                active_result_id=active.result_id,
                                new_result_id=new_result.result_id,
                                conflict_type=ConflictType.KPI_DOMAIN_CONFLICT,
                                severity=Severity.MEDIUM,
                                parameter=None,
                                active_value=left.value,
                                new_value=right.value,
                                mergeable=True,
                                explanation=f"The KPI objectives {left.value} and {right.value} have a known operational trade-off.",
                            )
                        )

            active_txs = {parameter[:3] for parameter in active_changes}
            new_txs = {parameter[:3] for parameter in new_changes}
            for tx_name in sorted(active_txs & new_txs):
                has_direct = any(
                    detail.parameter and detail.parameter.startswith(tx_name)
                    for detail in details
                    if detail.active_result_id == active.result_id
                )
                if not has_direct:
                    details.append(
                        ConflictDetail(
                            active_result_id=active.result_id,
                            new_result_id=new_result.result_id,
                            conflict_type=ConflictType.BASE_STATION_INTERACTION,
                            severity=Severity.LOW,
                            parameter=tx_name,
                            active_value=None,
                            new_value=None,
                            mergeable=True,
                            explanation="Both intents modify different parameters on the same transmitter; the merged configuration must be re-evaluated by the surrogate.",
                        )
                    )

        if not details:
            recommendation = "NONE"
        elif any((not item.mergeable) or item.severity in {Severity.HIGH, Severity.CRITICAL} for item in details):
            recommendation = "PRIORITY"
        else:
            recommendation = "WEIGHTED_MERGE"
        return ConflictReport(
            conflict_detected=bool(details),
            details=details,
            recommended_strategy=recommendation,
        )


def _signature(report: ConflictReport) -> tuple[Any, ...]:
    rows = tuple(
        sorted(
            (
                item.active_result_id,
                item.new_result_id,
                item.conflict_type.value,
                item.severity.value,
                item.parameter,
                json.dumps(item.active_value, sort_keys=True, default=str),
                json.dumps(item.new_value, sort_keys=True, default=str),
                item.mergeable,
            )
            for item in report.details
        )
    )
    return report.conflict_detected, report.recommended_strategy, rows


def run_conflict_detector_agent(
    new_result: OptimizationResult,
    active_records: list[ActiveIntentRecord],
    *,
    model_id: str | None,
    use_llm: bool,
) -> ConflictReport:
    deterministic = ConflictDetector().detect(new_result, active_records)
    if not use_llm or not model_id:
        return deterministic

    payload = {
        "new_optimization": new_result.model_dump(mode="json"),
        "active_intents": [record.model_dump(mode="json") for record in active_records],
        "deterministic_predicate_scan": deterministic.model_dump(mode="json"),
    }
    try:
        agent = create_conflict_detector_agent(model_id)
        response = agent.run(
            "Audit the deterministic conflict scan. Preserve all decision fields exactly.\n\n"
            + json.dumps(payload, indent=2)
        )
        audited = coerce_response_model(response, ConflictReport)
        return audited if _signature(audited) == _signature(deterministic) else deterministic
    except Exception:
        return deterministic
