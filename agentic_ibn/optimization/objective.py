from __future__ import annotations

from ..schemas import (
    KPIName,
    KPIPrediction,
    ObjectiveCheck,
    ObjectiveEvaluation,
    Operator,
    ParsedIntent,
)


_DIRECTION = {
    KPIName.RX_POWER: 1.0,
    KPIName.SINR: 1.0,
    KPIName.COVERAGE: 1.0,
    KPIName.THROUGHPUT_5P: 1.0,
    KPIName.THROUGHPUT_RR_5P: 1.0,
    KPIName.LOAD_BALANCE: 1.0,
    KPIName.LOAD_IMBALANCE: -1.0,
    KPIName.TOTAL_TX_POWER: -1.0,
}

_SCALE = {
    KPIName.RX_POWER: 10.0,
    KPIName.SINR: 10.0,
    KPIName.COVERAGE: 0.1,
    KPIName.THROUGHPUT_5P: 10.0,
    KPIName.THROUGHPUT_RR_5P: 5.0,
    KPIName.LOAD_BALANCE: 0.1,
    KPIName.LOAD_IMBALANCE: 0.1,
    KPIName.TOTAL_TX_POWER: 40.0,
}


class ObjectiveEvaluator:
    def evaluate(
        self,
        intent: ParsedIntent,
        prediction: KPIPrediction,
        baseline: KPIPrediction,
    ) -> ObjectiveEvaluation:
        checks: list[ObjectiveCheck] = []
        numeric_checks: list[ObjectiveCheck] = []
        score = 0.0

        for threshold in intent.kpi_thresholds:
            predicted = prediction.value_for(threshold.kpi)
            baseline_value = baseline.value_for(threshold.kpi)
            scale = _SCALE.get(threshold.kpi, 1.0)
            target: float | list[float] | None = threshold.value
            satisfied: bool | None = None
            violation = 0.0
            explanation = ""

            if predicted is None:
                violation = 10.0
                explanation = "The loaded surrogate artifact does not predict this KPI."
            elif threshold.operator == Operator.GTE:
                violation = max(0.0, float(threshold.value) - predicted) / scale
                satisfied = predicted >= float(threshold.value)
                explanation = f"Predicted {predicted:.4g}; required at least {float(threshold.value):.4g}."
            elif threshold.operator == Operator.GT:
                violation = max(0.0, float(threshold.value) - predicted + 1e-9) / scale
                satisfied = predicted > float(threshold.value)
                explanation = f"Predicted {predicted:.4g}; required above {float(threshold.value):.4g}."
            elif threshold.operator == Operator.LTE:
                violation = max(0.0, predicted - float(threshold.value)) / scale
                satisfied = predicted <= float(threshold.value)
                explanation = f"Predicted {predicted:.4g}; required at most {float(threshold.value):.4g}."
            elif threshold.operator == Operator.LT:
                violation = max(0.0, predicted - float(threshold.value) + 1e-9) / scale
                satisfied = predicted < float(threshold.value)
                explanation = f"Predicted {predicted:.4g}; required below {float(threshold.value):.4g}."
            elif threshold.operator == Operator.BETWEEN:
                low = float(threshold.value_low)
                high = float(threshold.value_high)
                target = [low, high]
                if predicted < low:
                    violation = (low - predicted) / scale
                elif predicted > high:
                    violation = (predicted - high) / scale
                satisfied = low <= predicted <= high
                explanation = f"Predicted {predicted:.4g}; required between {low:.4g} and {high:.4g}."
            elif threshold.operator in {Operator.DELTA_UP, Operator.DELTA_DOWN}:
                if baseline_value is None:
                    violation = 10.0
                    explanation = "A baseline KPI prediction is required for a delta objective."
                else:
                    delta = float(threshold.delta)
                    expected = baseline_value + delta if threshold.operator == Operator.DELTA_UP else baseline_value - delta
                    target = expected
                    if threshold.operator == Operator.DELTA_UP:
                        violation = max(0.0, expected - predicted) / scale
                        satisfied = predicted >= expected
                    else:
                        violation = max(0.0, predicted - expected) / scale
                        satisfied = predicted <= expected
                    explanation = f"Baseline {baseline_value:.4g}; derived target {expected:.4g}; predicted {predicted:.4g}."
            elif threshold.operator == Operator.TARGET:
                direction = _DIRECTION.get(threshold.kpi, 1.0)
                if threshold.value is not None:
                    target_value = float(threshold.value)
                    target = target_value
                    violation = abs(predicted - target_value) / scale
                    satisfied = violation <= 0.01
                    explanation = f"Predicted {predicted:.4g}; requested target {target_value:.4g}."
                elif baseline_value is None:
                    violation = 0.0
                    satisfied = None
                    explanation = "Open-ended objective with no baseline; the best candidate is selected after all iterations."
                else:
                    improvement = direction * (predicted - baseline_value) / scale
                    score += improvement
                    satisfied = None
                    explanation = (
                        f"Open-ended objective: baseline {baseline_value:.4g}, predicted {predicted:.4g}; "
                        f"normalized directional improvement {improvement:.4g}."
                    )

            check = ObjectiveCheck(
                kpi=threshold.kpi,
                operator=threshold.operator,
                target=target,
                predicted=predicted,
                satisfied=satisfied,
                normalized_violation=violation,
                explanation=explanation,
            )
            checks.append(check)
            if satisfied is not None:
                numeric_checks.append(check)
            score -= violation

        all_met = bool(numeric_checks) and all(check.satisfied is True for check in numeric_checks)
        return ObjectiveEvaluation(all_numeric_targets_met=all_met, score=score, checks=checks)
