from __future__ import annotations

from typing import Any

from ..schemas import (
    ArchiveEntry,
    KPIName,
    KPIPrediction,
    NetworkConfig,
    ObjectiveEvaluation,
    ParameterChange,
    ParsedIntent,
)

_TX_PARAMETERS = ("power_dbm", "azimuth_delta_deg", "elevation_delta_deg")
_PARAMETER_LABELS = {
    "on": "state",
    "power_dbm": "power",
    "azimuth_delta_deg": "azimuth",
    "elevation_delta_deg": "tilt",
}
# KPIs reported in deltas; LOAD_IMBALANCE mirrors LOAD_BALANCE and SERVED_USERS has no surrogate value.
_DELTA_KPIS = (
    KPIName.RX_POWER,
    KPIName.SINR,
    KPIName.COVERAGE,
    KPIName.THROUGHPUT_5P,
    KPIName.THROUGHPUT_RR_5P,
    KPIName.LOAD_BALANCE,
    KPIName.TOTAL_TX_POWER,
)


def configuration_key(configuration: NetworkConfig) -> tuple[Any, ...]:
    """Hashable identity of a configuration, used to skip candidates already evaluated."""
    key: list[Any] = [configuration.k_users, configuration.user_set_id, round(configuration.coverage_threshold_dbm, 3)]
    for index in range(4):
        tx = configuration.tx(index)
        if tx.on:
            key.append((True, round(tx.power_dbm, 3), round(tx.azimuth_delta_deg, 3), round(tx.elevation_delta_deg, 3)))
        else:
            key.append((False,))
    return tuple(key)


def configuration_changes(configuration: NetworkConfig, reference: NetworkConfig) -> list[ParameterChange]:
    """Per-transmitter parameter differences between a candidate and a reference configuration."""
    changes: list[ParameterChange] = []
    for index in range(4):
        new = configuration.tx(index)
        old = reference.tx(index)
        if new.on != old.on:
            changes.append(ParameterChange(tx=index, parameter="on", before=old.on, after=new.on))
            if not new.on:
                continue
        if not new.on:
            continue
        for parameter in _TX_PARAMETERS:
            before = float(getattr(old, parameter))
            after = float(getattr(new, parameter))
            if abs(after - before) > 1e-9:
                changes.append(ParameterChange(tx=index, parameter=parameter, before=before, after=after))
    return changes


def describe_change(change: ParameterChange) -> str:
    label = _PARAMETER_LABELS[change.parameter]
    if change.parameter == "on":
        return f"TX{change.tx} {'ON' if change.before else 'OFF'}->{'ON' if change.after else 'OFF'}"
    return f"TX{change.tx} {label} {float(change.before):g}->{float(change.after):g}"


def kpi_deltas(prediction: KPIPrediction, reference: KPIPrediction) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for kpi in _DELTA_KPIS:
        new = prediction.value_for(kpi)
        old = reference.value_for(kpi)
        if new is not None and old is not None:
            deltas[kpi.value] = float(new) - float(old)
    return deltas


def compact_configuration(configuration: NetworkConfig) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for index in range(4):
        tx = configuration.tx(index)
        compact[f"tx{index}"] = (
            {"on": True, "power_dbm": tx.power_dbm, "azimuth": tx.azimuth_delta_deg, "tilt": tx.elevation_delta_deg}
            if tx.on
            else {"on": False}
        )
    return compact


class OptimizationArchive:
    """Every configuration tried during one optimization run, with its scores and what changed.

    The archive is the optimizer's memory: the agent sees it sorted from worst to best (OPRO-style),
    and each entry records which parameters changed relative to the best entry at the time it was
    evaluated and how every KPI moved as a result.
    """

    def __init__(
        self,
        intent: ParsedIntent,
        baseline: NetworkConfig,
        baseline_prediction: KPIPrediction,
        baseline_evaluation: ObjectiveEvaluation,
    ):
        self.intent = intent
        self.entries: list[ArchiveEntry] = [
            ArchiveEntry(
                entry_id=0,
                round=-1,
                role="baseline",
                status="baseline",
                configuration=baseline,
                prediction=baseline_prediction,
                evaluation=baseline_evaluation,
            )
        ]
        self._keys = {configuration_key(baseline)}

    @property
    def baseline(self) -> ArchiveEntry:
        return self.entries[0]

    def contains(self, configuration: NetworkConfig) -> bool:
        return configuration_key(configuration) in self._keys

    def evaluated(self) -> list[ArchiveEntry]:
        """Candidates the optimizer proposed and the surrogate scored (baseline excluded)."""
        return [entry for entry in self.entries if entry.status == "evaluated"]

    def rejected(self) -> list[ArchiveEntry]:
        return [entry for entry in self.entries if entry.status == "rejected"]

    def best(self) -> ArchiveEntry | None:
        """Best proposed candidate; the baseline is the reference point, not a result."""
        candidates = self.evaluated()
        if not candidates:
            return None
        return max(candidates, key=lambda entry: entry.evaluation.score)  # type: ignore[union-attr]

    def reference(self) -> ArchiveEntry:
        """Entry new candidates are compared against: the best candidate so far, else the baseline."""
        best = self.best()
        if best is None or best.evaluation.score < self.baseline.evaluation.score:  # type: ignore[union-attr]
            return self.baseline
        return best

    def best_score(self) -> float | None:
        best = self.best()
        return None if best is None else best.evaluation.score  # type: ignore[union-attr]

    def add_evaluated(
        self,
        *,
        round_index: int,
        role: str,
        configuration: NetworkConfig,
        prediction: KPIPrediction,
        evaluation: ObjectiveEvaluation,
        rationale: str = "",
    ) -> ArchiveEntry:
        reference = self.reference()
        entry = ArchiveEntry(
            entry_id=len(self.entries),
            round=round_index,
            role=role,  # type: ignore[arg-type]
            status="evaluated",
            configuration=configuration,
            prediction=prediction,
            evaluation=evaluation,
            reference_entry_id=reference.entry_id,
            changes=configuration_changes(configuration, reference.configuration),
            kpi_deltas=kpi_deltas(prediction, reference.prediction),  # type: ignore[arg-type]
            score_delta=evaluation.score - reference.evaluation.score,  # type: ignore[union-attr]
            rationale=rationale,
        )
        self.entries.append(entry)
        self._keys.add(configuration_key(configuration))
        return entry

    def add_rejected(
        self,
        *,
        round_index: int,
        role: str,
        configuration: NetworkConfig,
        errors: list[str],
        rationale: str = "",
    ) -> ArchiveEntry:
        reference = self.reference()
        entry = ArchiveEntry(
            entry_id=len(self.entries),
            round=round_index,
            role=role,  # type: ignore[arg-type]
            status="rejected",
            configuration=configuration,
            reference_entry_id=reference.entry_id,
            changes=configuration_changes(configuration, reference.configuration),
            rationale=rationale,
            errors=errors,
        )
        self.entries.append(entry)
        self._keys.add(configuration_key(configuration))
        return entry

    def _row(self, entry: ArchiveEntry) -> dict[str, Any]:
        targets = list(dict.fromkeys([*self.intent.target_kpis, KPIName.TOTAL_TX_POWER]))
        row: dict[str, Any] = {
            "id": entry.entry_id,
            "round": entry.round,
            "role": entry.role,
            "configuration": compact_configuration(entry.configuration),
        }
        if entry.prediction is not None:
            row["predicted_kpis"] = {
                kpi.value: entry.prediction.value_for(kpi)
                for kpi in targets
                if entry.prediction.value_for(kpi) is not None
            }
        if entry.evaluation is not None:
            row["score"] = round(entry.evaluation.score, 4)
            row["all_targets_met"] = entry.evaluation.all_numeric_targets_met
            row["target_gaps"] = {
                check.kpi.value: round(check.normalized_violation, 4)
                for check in entry.evaluation.checks
                if check.normalized_violation > 0
            }
        if entry.reference_entry_id is not None:
            row["compared_with_id"] = entry.reference_entry_id
            row["changes"] = [describe_change(change) for change in entry.changes]
            if entry.kpi_deltas:
                row["kpi_change"] = {name: round(value, 4) for name, value in entry.kpi_deltas.items()}
            if entry.score_delta is not None:
                row["score_change"] = round(entry.score_delta, 4)
        if entry.errors:
            row["rejected_because"] = entry.errors
        return row

    def prompt_view(self, *, history: str = "full", top_k: int = 12, max_rejected: int = 4) -> dict[str, Any]:
        """What the optimizer agent sees each round.

        history="full": up to top_k scored candidates sorted ascending by score, so the best one is
        read last (the ordering OPRO found to work best), plus the most recent rejections.
        history="last": only the most recent scored candidate, which reproduces the original loop
        and is kept for ablation runs.
        """
        scored = self.evaluated()
        if history == "last":
            shown = scored[-1:]
            rejected: list[ArchiveEntry] = []
        else:
            shown = sorted(scored, key=lambda entry: entry.evaluation.score)[-top_k:]  # type: ignore[union-attr]
            rejected = self.rejected()[-max_rejected:]
        best = self.best()
        return {
            "baseline": self._row(self.baseline),
            "tried_candidates_worst_to_best": [self._row(entry) for entry in shown],
            "rejected_candidates": [self._row(entry) for entry in rejected],
            "best_candidate_id": None if best is None else best.entry_id,
            "evaluated_count": len(scored),
        }
