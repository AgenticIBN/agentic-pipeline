from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..schemas import KPIPrediction, NetworkConfig
from .features import configuration_to_feature_frame, engineer_features


TARGET_ALIASES = {
    "Prx_p5_dBm": "rx_power_p5_dbm",
    "Prx_mean_dBm": "rx_power_mean_dbm",
    "SINR_p5_dB": "sinr_p5_db",
    "SINR_mean_dB": "sinr_mean_db",
    "Thr_p5_Mbps": "throughput_p5_mbps",
    "ThrRR_p5_Mbps": "throughput_rr_p5_mbps",
    "rx_power_coverage_ratio": "coverage_ratio",
}


class SurrogateArtifactError(RuntimeError):
    pass


class SurrogateModel:
    """Loads a trusted joblib artifact containing one regressor per KPI target."""

    def __init__(self, artifact: dict[str, Any], source_path: str | Path | None = None):
        self.artifact = artifact
        self.source_path = Path(source_path).resolve() if source_path else None
        self.version = str(artifact.get("artifact_version", artifact.get("version", "unknown")))
        self.scenario = str(artifact.get("scenario", "unknown"))
        self.model_family = str(artifact.get("model_family", "unknown"))
        self.feature_names = list(artifact.get("feature_names") or artifact.get("feature_cols") or [])
        self.models: dict[str, Any] = dict(artifact.get("models") or {})
        self.parameter_bounds: dict[str, dict[str, float]] = dict(
            artifact.get("parameter_bounds") or artifact.get("param_ranges") or {}
        )
        self.parameter_grid: dict[str, list[float]] = {
            str(name): sorted(float(value) for value in values)
            for name, values in dict(artifact.get("parameter_grid") or {}).items()
            if values
        }
        self.metrics: dict[str, Any] = dict(artifact.get("metrics") or {})
        self.bandwidth_hz = float(artifact.get("bandwidth_hz", artifact.get("bw_hz", 400_000_000.0)))
        if not self.feature_names:
            raise SurrogateArtifactError("Surrogate artifact has no feature_names")
        if not self.models:
            raise SurrogateArtifactError("Surrogate artifact has no models")

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateModel":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Surrogate artifact not found: {source}")
        # joblib uses pickle semantics. Only load artifacts produced by this project or another trusted source.
        artifact = joblib.load(source)
        if not isinstance(artifact, dict):
            raise SurrogateArtifactError("Expected a dictionary-based surrogate artifact")
        return cls(artifact=artifact, source_path=source)

    def _predict_model(self, model: Any, features: np.ndarray) -> float:
        prediction = model.predict(features)
        value = float(np.asarray(prediction).reshape(-1)[0])
        if not math.isfinite(value):
            raise SurrogateArtifactError("Surrogate model produced a non-finite prediction")
        return value

    @staticmethod
    def _jain_fairness(values: list[float]) -> float:
        array = np.asarray(values, dtype=float)
        denominator = len(array) * float(np.square(array).sum())
        if denominator <= 0:
            return 0.0
        return float(array.sum() ** 2 / denominator)

    def _assemble_prediction(self, raw: dict[str, float], total_tx_power_watt: float) -> KPIPrediction:
        kwargs: dict[str, Any] = {"raw_targets": raw}
        for target, schema_field in TARGET_ALIASES.items():
            if target in raw:
                kwargs[schema_field] = raw[target]

        served = [raw.get(f"tx{i}_served_pct") for i in range(4)]
        if all(value is not None for value in served):
            load_balance = self._jain_fairness([float(value) for value in served])
            kwargs["load_balance"] = load_balance
            kwargs["load_imbalance"] = 1.0 - load_balance

        if kwargs.get("throughput_p5_mbps") is None and kwargs.get("throughput_rr_p5_mbps") is not None:
            kwargs["throughput_p5_mbps"] = kwargs["throughput_rr_p5_mbps"]

        if kwargs.get("throughput_p5_mbps") is None:
            sinr = kwargs.get("sinr_p5_db")
            if sinr is not None:
                sinr_linear = 10 ** (float(sinr) / 10.0)
                kwargs["throughput_p5_mbps"] = self.bandwidth_hz * math.log2(1.0 + sinr_linear) / 1e6

        kwargs["total_tx_power_watt"] = float(total_tx_power_watt)
        return KPIPrediction(**kwargs)

    def predict(self, configuration: NetworkConfig) -> KPIPrediction:
        frame = configuration_to_feature_frame(configuration, self.feature_names)
        matrix = frame.to_numpy(dtype=np.float32)
        raw = {target: self._predict_model(model, matrix) for target, model in self.models.items()}
        return self._assemble_prediction(raw, float(configuration.to_flat_dict()["total_tx_power_watt"]))

    def predict_flat_frame(self, flat: pd.DataFrame) -> list[KPIPrediction]:
        """Batch prediction for many configurations given as flat rows (NetworkConfig.to_flat_dict columns).

        Same numbers as calling predict() row by row, but every model runs once over the whole matrix.
        Used by the exhaustive-search benchmark.
        """
        engineered = engineer_features(flat)
        missing = [name for name in self.feature_names if name not in engineered.columns]
        if missing:
            raise ValueError(f"Artifact requests unsupported features: {missing}")
        matrix = engineered[self.feature_names].to_numpy(dtype=np.float32)
        outputs = {target: np.asarray(model.predict(matrix), dtype=float).reshape(-1) for target, model in self.models.items()}
        for target, values in outputs.items():
            if not np.all(np.isfinite(values)):
                raise SurrogateArtifactError(f"Surrogate model produced non-finite predictions for {target}")
        totals = engineered["total_tx_power_watt"].to_numpy(dtype=float)
        targets = list(outputs)
        return [
            self._assemble_prediction({target: float(outputs[target][row]) for target in targets}, totals[row])
            for row in range(len(matrix))
        ]

    def grid_for(self, parameter: str) -> list[float] | None:
        """Distinct training values of a knob (power, azimuth or tilt), or None for artifacts without a grid."""
        values = self.parameter_grid.get(parameter) or self.parameter_grid.get(
            {"power_dbm": "power", "azimuth_delta_deg": "dAz", "elevation_delta_deg": "dEl"}.get(parameter, parameter)
        )
        return list(values) if values else None

    def bounds_for(self, parameter: str, default_min: float, default_max: float) -> tuple[float, float]:
        bound = self.parameter_bounds.get(parameter) or self.parameter_bounds.get(
            {"power_dbm": "power", "azimuth_delta_deg": "dAz", "elevation_delta_deg": "dEl"}.get(parameter, parameter),
            {},
        )
        return float(bound.get("min", default_min)), float(bound.get("max", default_max))
