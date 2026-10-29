from __future__ import annotations

from copy import deepcopy

from ..schemas import (
    HardConstraints,
    NetworkConfig,
    TxConfig,
    ValidationCorrection,
    ValidationResult,
)
from ..surrogate.model import SurrogateModel


class CandidateValidator:
    def __init__(self, surrogate: SurrogateModel):
        self.surrogate = surrogate
        self.power_min, self.power_max = surrogate.bounds_for("power_dbm", 20.0, 46.0)
        self.az_min, self.az_max = surrogate.bounds_for("azimuth_delta_deg", -30.0, 30.0)
        self.el_min, self.el_max = surrogate.bounds_for("elevation_delta_deg", 5.0, 20.0)
        self.grid = {
            parameter: surrogate.grid_for(parameter)
            for parameter in ("power_dbm", "azimuth_delta_deg", "elevation_delta_deg")
        }

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return min(max(float(value), lower), upper)

    @staticmethod
    def _snap(value: float, grid: list[float]) -> float:
        return min(grid, key=lambda allowed: (abs(allowed - float(value)), allowed))

    def allowed_values(self) -> dict[str, list[float] | tuple[float, float]]:
        """Grid values per knob when the artifact has them, otherwise the (min, max) range."""
        ranges = {
            "power_dbm": (self.power_min, self.power_max),
            "azimuth_delta_deg": (self.az_min, self.az_max),
            "elevation_delta_deg": (self.el_min, self.el_max),
        }
        return {parameter: self.grid[parameter] or ranges[parameter] for parameter in ranges}

    def validate(
        self,
        proposal: NetworkConfig,
        hard_constraints: HardConstraints,
        context: NetworkConfig | None = None,
    ) -> ValidationResult:
        """Enforce hard TX states, keep values on the training grid (or range) and pin the environment.

        `context` is the baseline the optimizer started from: user count, user set and coverage
        threshold describe the environment, not knobs, so a candidate may not change them.
        """
        data = deepcopy(proposal.model_dump())
        corrections: list[ValidationCorrection] = []
        errors: list[str] = []

        if context is not None:
            for field_name in ("k_users", "user_set_id", "coverage_threshold_dbm"):
                proposed = data[field_name]
                applied = getattr(context, field_name)
                if proposed != applied:
                    corrections.append(
                        ValidationCorrection(
                            parameter=field_name,
                            proposed_value=proposed,
                            applied_value=applied,
                            reason="Environment context is fixed during optimization",
                        )
                    )
                    data[field_name] = applied

        for index in range(4):
            key = f"tx{index}"
            raw = TxConfig.model_validate(data[key])
            tx = raw

            required_state = hard_constraints.tx_states.get(index)
            if required_state is not None and tx.on != required_state:
                corrections.append(
                    ValidationCorrection(
                        parameter=f"tx{index}.on",
                        proposed_value=tx.on,
                        applied_value=required_state,
                        reason="Original intent hard constraint",
                    )
                )
                tx = tx.model_copy(update={"on": required_state})

            if not tx.on:
                normalized = {"power_dbm": 0.0, "azimuth_delta_deg": 0.0, "elevation_delta_deg": 0.0}
                for parameter, applied in normalized.items():
                    proposed = getattr(tx, parameter)
                    if proposed != applied:
                        corrections.append(
                            ValidationCorrection(
                                parameter=f"tx{index}.{parameter}",
                                proposed_value=proposed,
                                applied_value=applied,
                                reason="OFF transmitters use zeroed surrogate features",
                            )
                        )
                tx = tx.model_copy(update=normalized)
            else:
                clipped = {
                    "power_dbm": self._clip(tx.power_dbm, self.power_min, self.power_max),
                    "azimuth_delta_deg": self._clip(tx.azimuth_delta_deg, self.az_min, self.az_max),
                    "elevation_delta_deg": self._clip(tx.elevation_delta_deg, self.el_min, self.el_max),
                }
                clipped = {
                    parameter: self._snap(value, self.grid[parameter]) if self.grid[parameter] else value  # type: ignore[arg-type]
                    for parameter, value in clipped.items()
                }
                for parameter, applied in clipped.items():
                    proposed = getattr(tx, parameter)
                    if proposed != applied:
                        corrections.append(
                            ValidationCorrection(
                                parameter=f"tx{index}.{parameter}",
                                proposed_value=proposed,
                                applied_value=applied,
                                reason=(
                                    "Snapped to the nearest value in the surrogate's training grid"
                                    if self.grid[parameter]
                                    else "Clamped to the surrogate artifact's observed training range"
                                ),
                            )
                        )
                tx = tx.model_copy(update=clipped)

            data[key] = tx.model_dump()

        if not any(bool(data[f"tx{i}"]["on"]) for i in range(4)):
            errors.append("All transmitters are OFF. At least one transmitter must remain active.")

        configuration = NetworkConfig.model_validate(data)
        return ValidationResult(
            valid=not errors,
            configuration=configuration,
            corrections=corrections,
            errors=errors,
        )
