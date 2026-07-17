from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..schemas import NetworkConfig


BASE_FEATURES = [
    "K_users",
    "rx_power_thr_dBm",
    "total_tx_power_watt",
    "tx0_on", "tx1_on", "tx2_on", "tx3_on",
    "tx0_P_dBm", "tx1_P_dBm", "tx2_P_dBm", "tx3_P_dBm",
    "tx0_dAz", "tx1_dAz", "tx2_dAz", "tx3_dAz",
    "tx0_dEl", "tx1_dEl", "tx2_dEl", "tx3_dEl",
]

ENGINEERED_FEATURES = [
    "avg_power_on",
    "power_std_on",
    "n_active_tx",
    "avg_azimuth_on",
    "avg_elevation_on",
    "azimuth_range_on",
    "elevation_range_on",
    "azimuth_std_on",
    "elevation_std_on",
]
for _index in range(4):
    ENGINEERED_FEATURES.extend(
        [
            f"tx{_index}_P_x_Az",
            f"tx{_index}_P_x_El",
            f"tx{_index}_Az_x_El",
        ]
    )

DEFAULT_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES


def sanitize_configuration_frame(
        frame: pd.DataFrame,
        *,
        copy: bool = True,
    ) -> pd.DataFrame:
    
    df = frame.copy() if copy else frame
    required_defaults = {
        "K_users": 200,
        "rx_power_thr_dBm": -60.0,
        "user_set_id": 0,
    }
    for column, default in required_defaults.items():
        if column not in df.columns:
            df[column] = default

    for index in range(4):
        on_col = f"tx{index}_on"
        power_col = f"tx{index}_P_dBm"
        az_col = f"tx{index}_dAz"
        el_col = f"tx{index}_dEl"
        if on_col not in df.columns:
            df[on_col] = True
        df[on_col] = df[on_col].fillna(False).astype(bool)
        for column in (power_col, az_col, el_col):
            if column not in df.columns:
                df[column] = 0.0
            df[column] = pd.to_numeric(df[column], errors="coerce")
        sentinel = df[power_col] <= -1000
        df.loc[sentinel, on_col] = False
        off = ~df[on_col]
        df.loc[off, [power_col, az_col, el_col]] = 0.0

    total_watt = np.zeros(len(df), dtype=float)
    for index in range(4):
        on = df[f"tx{index}_on"].to_numpy(dtype=bool)
        power = df[f"tx{index}_P_dBm"].to_numpy(dtype=float)
        total_watt += np.where(on, 10 ** ((power - 30.0) / 10.0), 0.0)
    df["total_tx_power_watt"] = total_watt
    return df


def engineer_features(
        frame: pd.DataFrame,
        *,
        sanitize: bool = True,
        copy: bool = True,
    ) -> pd.DataFrame:

    if sanitize:
        df = sanitize_configuration_frame(frame, copy=copy)
    else:
        df = frame.copy() if copy else frame
        
    on = df[[f"tx{i}_on" for i in range(4)]].to_numpy(dtype=bool)
    powers = df[[f"tx{i}_P_dBm" for i in range(4)]].to_numpy(dtype=float)
    azimuths = df[[f"tx{i}_dAz" for i in range(4)]].to_numpy(dtype=float)
    elevations = df[[f"tx{i}_dEl" for i in range(4)]].to_numpy(dtype=float)

    active_count = on.sum(axis=1)
    safe_count = np.maximum(active_count, 1)

    def masked_mean(values: np.ndarray) -> np.ndarray:
        return (values * on).sum(axis=1) / safe_count

    def masked_std(values: np.ndarray) -> np.ndarray:
        mean = masked_mean(values)
        variance = (((values - mean[:, None]) ** 2) * on).sum(axis=1) / safe_count
        return np.sqrt(variance)

    def masked_range(values: np.ndarray) -> np.ndarray:
        minimum = np.where(on, values, np.inf).min(axis=1)
        maximum = np.where(on, values, -np.inf).max(axis=1)
        return np.where(active_count > 0, maximum - minimum, 0.0)

    df["avg_power_on"] = masked_mean(powers)
    df["power_std_on"] = masked_std(powers)
    df["n_active_tx"] = active_count
    df["avg_azimuth_on"] = masked_mean(azimuths)
    df["avg_elevation_on"] = masked_mean(elevations)
    df["azimuth_range_on"] = masked_range(azimuths)
    df["elevation_range_on"] = masked_range(elevations)
    df["azimuth_std_on"] = masked_std(azimuths)
    df["elevation_std_on"] = masked_std(elevations)

    for index in range(4):
        power = df[f"tx{index}_P_dBm"]
        azimuth = df[f"tx{index}_dAz"]
        elevation = df[f"tx{index}_dEl"]
        df[f"tx{index}_P_x_Az"] = power * azimuth
        df[f"tx{index}_P_x_El"] = power * elevation
        df[f"tx{index}_Az_x_El"] = azimuth * elevation

    return df


def configuration_to_feature_frame(
    configuration: NetworkConfig,
    feature_names: Iterable[str] = DEFAULT_FEATURES,
) -> pd.DataFrame:
    row = configuration.to_flat_dict()
    engineered = engineer_features(pd.DataFrame([row]))
    names = list(feature_names)
    missing = [name for name in names if name not in engineered.columns]
    if missing:
        raise ValueError(f"Artifact requests unsupported features: {missing}")
    return engineered[names].astype(float)
