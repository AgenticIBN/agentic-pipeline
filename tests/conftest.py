"""Synthetic surrogate for offline tests: small LightGBM models fitted on a made-up radio response.

The response is not physics; it only needs the shape the optimizer relies on (power raises
coverage and RX power, more active transmitters raise interference, tilt has a sweet spot).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from agentic_ibn.schemas import NetworkConfig, ParsedIntent
from agentic_ibn.surrogate.features import DEFAULT_FEATURES, engineer_features
from agentic_ibn.surrogate.model import SurrogateModel

POWERS = [43.0, 46.0]
AZIMUTHS = [-30.0, 0.0, 30.0]
TILTS = [5.0, 10.0, 20.0]


def _synthetic_targets(flat: pd.DataFrame) -> pd.DataFrame:
    on = flat[[f"tx{i}_on" for i in range(4)]].to_numpy(dtype=bool)
    power = flat[[f"tx{i}_P_dBm" for i in range(4)]].to_numpy(dtype=float)
    azimuth = flat[[f"tx{i}_dAz" for i in range(4)]].to_numpy(dtype=float)
    tilt = flat[[f"tx{i}_dEl" for i in range(4)]].to_numpy(dtype=float)
    weights = np.array([1.0, 0.9, 1.1, 0.8])
    tilt_gain = 1.0 - np.abs(tilt - 10.0) / 15.0
    spread = np.abs(azimuth) / 30.0
    linear = np.where(on, 10 ** ((power - 43.0) / 10.0) * tilt_gain * weights, 0.0)
    signal = linear.sum(axis=1)
    active = on.sum(axis=1)
    interference = np.where(on, linear * (1.0 - 0.6 * spread), 0.0).sum(axis=1) - linear.max(axis=1)
    coverage = 1.0 / (1.0 + np.exp(-(signal - 2.2) * 1.6))
    sinr = 10 * np.log10(np.maximum(linear.max(axis=1), 1e-6) / (0.15 + np.maximum(interference, 0.0)))
    prx = 10 * np.log10(np.maximum(signal, 1e-6)) - 100.0
    shares = linear / np.maximum(signal[:, None], 1e-9) * 100.0
    out = pd.DataFrame(
        {
            "Prx_p5_dBm": prx - 8.0,
            "Prx_mean_dBm": prx,
            "SINR_p5_dB": sinr - 3.0,
            "SINR_mean_dB": sinr,
            "Thr_p5_Mbps": 40.0 * np.log2(1 + 10 ** ((sinr - 3.0) / 10.0)),
            "ThrRR_p5_Mbps": 40.0 * np.log2(1 + 10 ** ((sinr - 3.0) / 10.0)) / np.maximum(active, 1),
            "rx_power_coverage_ratio": coverage,
        }
    )
    for index in range(4):
        out[f"tx{index}_served_pct"] = shares[:, index]
    return out


def grid_flat_frame(k_users: int = 200) -> pd.DataFrame:
    options = [(False, 0.0, 0.0, 0.0)] + [(True, p, a, e) for p, a, e in itertools.product(POWERS, AZIMUTHS, TILTS)]
    rows = []
    for combo in itertools.product(options, repeat=4):
        if not any(tx[0] for tx in combo):
            continue
        row = {"K_users": k_users, "user_set_id": 0, "rx_power_thr_dBm": -60.0}
        for index, (on, p, a, e) in enumerate(combo):
            row.update({f"tx{index}_on": on, f"tx{index}_P_dBm": p, f"tx{index}_dAz": a, f"tx{index}_dEl": e})
        rows.append(row)
    return pd.DataFrame(rows)


def build_synthetic_surrogate() -> SurrogateModel:
    flat = grid_flat_frame().sample(n=25_000, random_state=0).reset_index(drop=True)
    features = engineer_features(flat)[DEFAULT_FEATURES].to_numpy(dtype=np.float32)
    targets = _synthetic_targets(flat)
    models = {
        name: LGBMRegressor(n_estimators=120, num_leaves=31, learning_rate=0.1, verbose=-1).fit(
            features, targets[name].to_numpy()
        )
        for name in targets.columns
    }
    artifact = {
        "artifact_version": "test",
        "scenario": "urban_area",
        "model_family": "lightgbm",
        "feature_names": list(DEFAULT_FEATURES),
        "target_names": list(targets.columns),
        "models": models,
        "parameter_bounds": {
            "power": {"min": 43.0, "max": 46.0},
            "dAz": {"min": -30.0, "max": 30.0},
            "dEl": {"min": 5.0, "max": 20.0},
        },
        "parameter_grid": {"power": POWERS, "dAz": AZIMUTHS, "dEl": TILTS},
    }
    return SurrogateModel(artifact)


@pytest.fixture(scope="session")
def surrogate() -> SurrogateModel:
    return build_synthetic_surrogate()


@pytest.fixture()
def baseline() -> NetworkConfig:
    return NetworkConfig()


@pytest.fixture()
def maintenance_intent() -> ParsedIntent:
    return ParsedIntent.model_validate(
        {
            "raw_text": "TX2 is under maintenance tonight. Keep coverage at least 99% and SINR p5 at least 3 dB.",
            "target_area": "downtown",
            "scenario": "urban_area",
            "target_kpis": ["COVERAGE", "SINR"],
            "kpi_thresholds": [
                {"kpi": "COVERAGE", "operator": "GTE", "value": 0.99},
                {"kpi": "SINR", "operator": "GTE", "value": 3.0},
            ],
            "priority": "HIGH",
        }
    )
