from __future__ import annotations

from datetime import datetime
from typing import Any

from .schemas import KPIName, KpiThreshold, Operator, ParsedIntent, Priority


_KPI_ALIASES = {
    "RX": KPIName.RX_POWER,
    "RX_POWER": KPIName.RX_POWER,
    "PRX": KPIName.RX_POWER,
    "SINR": KPIName.SINR,
    "COVERAGE": KPIName.COVERAGE,
    "RX_COVERAGE_RATIO": KPIName.COVERAGE,
    "THROUGHPUT": KPIName.THROUGHPUT_5P,
    "THROUGHPUT_5P": KPIName.THROUGHPUT_5P,
    "THROUGHPUT_RR_5P": KPIName.THROUGHPUT_RR_5P,
    "THRRR_P5": KPIName.THROUGHPUT_RR_5P,
    "TOTAL_TX_POWER": KPIName.TOTAL_TX_POWER,
    "ENERGY": KPIName.TOTAL_TX_POWER,
    "ENERGY_WATT": KPIName.TOTAL_TX_POWER,
    "LOAD_BALANCE": KPIName.LOAD_BALANCE,
    "LOAD_IMBALANCE": KPIName.LOAD_IMBALANCE,
    "SERVED_USERS": KPIName.SERVED_USERS,
}

_OPERATOR_ALIASES = {
    ">": Operator.GT,
    "GT": Operator.GT,
    ">=": Operator.GTE,
    "GTE": Operator.GTE,
    "<": Operator.LT,
    "LT": Operator.LT,
    "<=": Operator.LTE,
    "LTE": Operator.LTE,
    "BETWEEN": Operator.BETWEEN,
    "DELTA_UP": Operator.DELTA_UP,
    "DELTA_DOWN": Operator.DELTA_DOWN,
    "TARGET": Operator.TARGET,
}


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unsupported parser output type: {type(value)!r}")


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_kpi(value: Any) -> KPIName:
    key = str(value).strip().upper().replace(" ", "_")
    if key not in _KPI_ALIASES:
        raise ValueError(f"Unsupported KPI from intent parser: {value!r}")
    return _KPI_ALIASES[key]


def normalize_operator(value: Any) -> Operator:
    key = str(value).strip().upper()
    if key not in _OPERATOR_ALIASES:
        raise ValueError(f"Unsupported operator from intent parser: {value!r}")
    return _OPERATOR_ALIASES[key]


def adapt_intent_parse(raw_text: str, parser_output: Any, scenario: str) -> ParsedIntent:
    data = _as_dict(parser_output)
    target_kpis: list[KPIName] = []
    for item in data.get("target_kpis", []):
        try:
            target_kpis.append(normalize_kpi(item))
        except ValueError:
            continue

    thresholds: list[KpiThreshold] = []
    for raw_threshold in data.get("kpi_thresholds", []):
        threshold = _as_dict(raw_threshold)
        kpi_raw = threshold.get("kpi", threshold.get("kpi_name"))
        operator_raw = threshold.get("op", threshold.get("operator", "TARGET"))
        try:
            kpi = normalize_kpi(kpi_raw)
            operator = normalize_operator(operator_raw)
        except ValueError:
            continue
        thresholds.append(
            KpiThreshold(
                kpi=kpi,
                operator=operator,
                value=threshold.get("value"),
                value_low=threshold.get("value_low"),
                value_high=threshold.get("value_high"),
                delta=threshold.get("delta"),
                unit=threshold.get("unit"),
            )
        )
        if kpi not in target_kpis:
            target_kpis.append(kpi)

    if not thresholds:
        thresholds = [KpiThreshold(kpi=kpi, operator=Operator.TARGET) for kpi in target_kpis]

    priority_raw = str(data.get("priority", "MEDIUM")).upper()
    priority = Priority(priority_raw) if priority_raw in Priority._value2member_map_ else Priority.MEDIUM

    return ParsedIntent(
        raw_text=raw_text,
        target_area=str(data.get("target_area", "unknown")),
        scenario=scenario,
        target_kpis=target_kpis,
        kpi_thresholds=thresholds,
        time_constraint_start=_parse_datetime(data.get("time_constraint_start")),
        time_constraint_end=_parse_datetime(data.get("time_constraint_end")),
        priority=priority,
        confidence=float(data.get("confidence", 0.8) or 0.8),
    )
