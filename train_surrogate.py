# train_surrogate.py
from __future__ import annotations

import os
from typing import Dict, List

import duckdb
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

load_dotenv()

BW_HZ_DEFAULT = 10_000_000  # 10 MHz

# Conservative "physics-ish" bounds to remove sentinel / garbage targets.
# Override via .env if you want.
PRX_MIN = float(os.getenv("PRX_MIN", "-140.0"))
PRX_MAX = float(os.getenv("PRX_MAX", "-40.0"))
SINR_MIN = float(os.getenv("SINR_MIN", "-20.0"))
SINR_MAX = float(os.getenv("SINR_MAX", "40.0"))

BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE", "50000"))


# -----------------------------
# Helpers
# -----------------------------
def _rel_from_path(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".parquet":
        return f"read_parquet('{path}')"
    return f"read_csv_auto('{path}', header=True)"


def _safe_bool_to_int(s: pd.Series) -> pd.Series:
    # DuckDB sometimes returns bool, sometimes 0/1 depending on source
    if s.dtype == bool:
        return s.astype(np.int8)
    # allow NaNs -> 0
    return s.fillna(0).astype(np.int8)


def _sanitize_tx_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature-side cleanup:
    - OFF tx -> power/angles set to 0 (dataset may contain -9999 sentinel)
    - clip extreme power/angles (defensive)
    """
    df = df.copy()
    for i in range(4):
        on_col = f"tx{i}_on"
        p_col = f"tx{i}_P_dBm"
        az_col = f"tx{i}_dAz"
        el_col = f"tx{i}_dEl"

        df[on_col] = _safe_bool_to_int(df[on_col])

        off_mask = df[on_col] == 0

        # If off -> force values to 0
        df.loc[off_mask, p_col] = 0.0
        df.loc[off_mask, az_col] = 0.0
        df.loc[off_mask, el_col] = 0.0

        # If on -> remove sentinel-ish power values
        # (if dataset has -9999 even when on due to corruption)
        df.loc[df[p_col] < -1000, p_col] = 0.0

        # Clip to safe bounds (features only)
        df[p_col] = df[p_col].clip(lower=0.0, upper=100.0)
        df[az_col] = df[az_col].clip(lower=-180.0, upper=180.0)
        df[el_col] = df[el_col].clip(lower=-90.0, upper=90.0)

    return df


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Ensure numeric dtype; coerce errors -> NaN
    """
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _clean_targets_and_mask(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """
    Target-side cleanup:
    - turn extreme sentinels into NaN
    - filter rows with targets outside realistic bounds
    """
    df = df.copy()
    df = _coerce_numeric(df, target_cols)

    # Convert extreme garbage to NaN
    for c in target_cols:
        df.loc[(df[c] < -1e6) | (df[c] > 1e6), c] = np.nan

    m = np.ones(len(df), dtype=bool)

    if "Prx_p5_dBm" in df.columns:
        m &= df["Prx_p5_dBm"].between(PRX_MIN, PRX_MAX)

    if "SINR_p5_dB" in df.columns:
        m &= df["SINR_p5_dB"].between(SINR_MIN, SINR_MAX)

    if "rx_power_coverage_ratio" in df.columns:
        m &= df["rx_power_coverage_ratio"].between(0.0, 1.0)

    for i in range(4):
        c = f"tx{i}_served_pct"
        if c in df.columns:
            m &= df[c].between(0.0, 100.0)

    # Drop rows with NaN in any target
    m &= df[target_cols].notna().all(axis=1)

    return df.loc[m].reset_index(drop=True)


def _make_features(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _make_targets(df: pd.DataFrame, target_cols: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in target_cols:
        y = df[c].to_numpy(dtype=np.float32, copy=False)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        out[c] = y
    return out


def _compute_param_ranges(con: duckdb.DuckDBPyConnection, rel: str) -> Dict[str, Dict[str, float]]:
    """
    True "range" from entire dataset:
    power ranges only for ON tx and realistic power > 0
    """
    q = f"""
    WITH base AS (
      SELECT
        tx0_on, tx1_on, tx2_on, tx3_on,
        tx0_P_dBm, tx1_P_dBm, tx2_P_dBm, tx3_P_dBm,
        tx0_dAz, tx1_dAz, tx2_dAz, tx3_dAz,
        tx0_dEl, tx1_dEl, tx2_dEl, tx3_dEl
      FROM {rel}
    )
    SELECT
      min(CASE WHEN tx0_on AND tx0_P_dBm > 0 THEN tx0_P_dBm END) AS pmin0,
      max(CASE WHEN tx0_on AND tx0_P_dBm > 0 THEN tx0_P_dBm END) AS pmax0,
      min(CASE WHEN tx1_on AND tx1_P_dBm > 0 THEN tx1_P_dBm END) AS pmin1,
      max(CASE WHEN tx1_on AND tx1_P_dBm > 0 THEN tx1_P_dBm END) AS pmax1,
      min(CASE WHEN tx2_on AND tx2_P_dBm > 0 THEN tx2_P_dBm END) AS pmin2,
      max(CASE WHEN tx2_on AND tx2_P_dBm > 0 THEN tx2_P_dBm END) AS pmax2,
      min(CASE WHEN tx3_on AND tx3_P_dBm > 0 THEN tx3_P_dBm END) AS pmin3,
      max(CASE WHEN tx3_on AND tx3_P_dBm > 0 THEN tx3_P_dBm END) AS pmax3,

      min(CASE WHEN tx0_on THEN tx0_dAz END) AS azmin0,
      max(CASE WHEN tx0_on THEN tx0_dAz END) AS azmax0,
      min(CASE WHEN tx1_on THEN tx1_dAz END) AS azmin1,
      max(CASE WHEN tx1_on THEN tx1_dAz END) AS azmax1,
      min(CASE WHEN tx2_on THEN tx2_dAz END) AS azmin2,
      max(CASE WHEN tx2_on THEN tx2_dAz END) AS azmax2,
      min(CASE WHEN tx3_on THEN tx3_dAz END) AS azmin3,
      max(CASE WHEN tx3_on THEN tx3_dAz END) AS azmax3,

      min(CASE WHEN tx0_on THEN tx0_dEl END) AS elmin0,
      max(CASE WHEN tx0_on THEN tx0_dEl END) AS elmax0,
      min(CASE WHEN tx1_on THEN tx1_dEl END) AS elmin1,
      max(CASE WHEN tx1_on THEN tx1_dEl END) AS elmax1,
      min(CASE WHEN tx2_on THEN tx2_dEl END) AS elmin2,
      max(CASE WHEN tx2_on THEN tx2_dEl END) AS elmax2,
      min(CASE WHEN tx3_on THEN tx3_dEl END) AS elmin3,
      max(CASE WHEN tx3_on THEN tx3_dEl END) AS elmax3
    FROM base
    """
    row = con.execute(q).fetchone()

    pmins = [row[0], row[2], row[4], row[6]]
    pmaxs = [row[1], row[3], row[5], row[7]]
    azmins = [row[8], row[10], row[12], row[14]]
    azmaxs = [row[9], row[11], row[13], row[15]]
    elmins = [row[16], row[18], row[20], row[22]]
    elmaxs = [row[17], row[19], row[21], row[23]]

    def _nn(vals):
        return [v for v in vals if v is not None]

    return {
        "power": {"min": float(min(_nn(pmins))), "max": float(max(_nn(pmaxs)))},
        "dAz": {"min": float(min(_nn(azmins))), "max": float(max(_nn(azmaxs)))},
        "dEl": {"min": float(min(_nn(elmins))), "max": float(max(_nn(elmaxs)))},
    }


def main() -> None:
    data_path = os.getenv("DATA_PATH", "").strip()
    model_path = os.getenv("MODEL_PATH", "").strip()
    batch_size = int(os.getenv("BATCH_SIZE", str(BATCH_SIZE_DEFAULT)))
    bw_hz = int(os.getenv("BW_HZ", str(BW_HZ_DEFAULT)))

    if not data_path:
        raise ValueError("DATA_PATH env var is required.")
    if not model_path:
        raise ValueError("MODEL_PATH env var is required (where to save surrogate.joblib).")

    con = duckdb.connect(database=":memory:")
    rel = _rel_from_path(data_path)

    # Features (X)
    feature_cols = [
        "user_set_id",
        "K_users",
        "rx_power_thr_dBm",
        "total_tx_power_watt",
        "tx0_on", "tx1_on", "tx2_on", "tx3_on",
        "tx0_P_dBm", "tx1_P_dBm", "tx2_P_dBm", "tx3_P_dBm",
        "tx0_dAz", "tx1_dAz", "tx2_dAz", "tx3_dAz",
        "tx0_dEl", "tx1_dEl", "tx2_dEl", "tx3_dEl",
    ]

    # Targets (y)
    target_cols = [
        "Prx_p5_dBm",
        "SINR_p5_dB",
        "rx_power_coverage_ratio",
        "tx0_served_pct", "tx1_served_pct", "tx2_served_pct", "tx3_served_pct",
    ]

    # Validate columns exist (fast)
    sample = con.execute(f"SELECT {', '.join(feature_cols + target_cols)} FROM {rel} LIMIT 1").df()
    missing_f = [c for c in feature_cols if c not in sample.columns]
    missing_t = [c for c in target_cols if c not in sample.columns]
    if missing_f or missing_t:
        raise ValueError(f"Dataset missing columns. missing_features={missing_f}, missing_targets={missing_t}")

    # Parameter ranges from entire dataset
    param_ranges = _compute_param_ranges(con, rel)

    # Models
    models: Dict[str, SGDRegressor] = {}
    for t in target_cols:
        models[t] = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=1e-4,
            learning_rate="constant",
            eta0=1e-4,
            max_iter=1,
            tol=None,
            random_state=42,
            average=True,
        )

    scaler = StandardScaler(with_mean=True, with_std=True)
    y_scalers = {t: StandardScaler(with_mean=True, with_std=True) for t in target_cols}

    # -----------------------------
    # PASS 1: Fit scalers on full data (VALID rows only)
    # -----------------------------
    print("PASS 1: fitting scalers over full dataset (valid rows only)...")
    cur = con.execute(f"SELECT {', '.join(feature_cols + target_cols)} FROM {rel}")
    n_seen_raw = 0
    n_seen_valid = 0

    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break

        df = pd.DataFrame(rows, columns=feature_cols + target_cols)
        n_seen_raw += len(df)

        df = _sanitize_tx_params(df)
        df = _clean_targets_and_mask(df, target_cols)
        if df.empty:
            continue

        X = _make_features(df, feature_cols)
        scaler.partial_fit(X)

        ys = _make_targets(df, target_cols)
        for t in target_cols:
            y_scalers[t].partial_fit(ys[t].reshape(-1, 1))

        n_seen_valid += len(df)
        if n_seen_raw % (batch_size * 10) == 0:
            print(f"  seen_raw={n_seen_raw} valid_used={n_seen_valid}")

    print(f"Scalers fitted. Total rows raw: {n_seen_raw}, valid used: {n_seen_valid}")

    # -----------------------------
    # PASS 2: Train models on full data (VALID rows only)
    # -----------------------------
    print("PASS 2: training models over full dataset (valid rows only)...")
    cur2 = con.execute(f"SELECT {', '.join(feature_cols + target_cols)} FROM {rel}")
    n_trained_raw = 0
    n_trained_valid = 0

    while True:
        rows = cur2.fetchmany(batch_size)
        if not rows:
            break

        df = pd.DataFrame(rows, columns=feature_cols + target_cols)
        n_trained_raw += len(df)

        df = _sanitize_tx_params(df)
        df = _clean_targets_and_mask(df, target_cols)
        if df.empty:
            continue

        X = _make_features(df, feature_cols)
        Xs = scaler.transform(X)

        ys = _make_targets(df, target_cols)
        for t, model in models.items():
            y = ys[t].reshape(-1, 1)
            y_std = y_scalers[t].transform(y).ravel()
            model.partial_fit(Xs, y_std)

        n_trained_valid += len(df)
        if n_trained_raw % (batch_size * 10) == 0:
            print(f"  trained_raw={n_trained_raw} valid_used={n_trained_valid}")

    print(f"Training done. Total rows raw: {n_trained_raw}, valid used: {n_trained_valid}")

    artifact = {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "scaler": scaler,
        "y_scalers": y_scalers,
        "models": models,
        "param_ranges": param_ranges,
        "bw_hz": bw_hz,
        "notes": (
            "Surrogate trained with DuckDB streaming. "
            "OFF-tx features sanitized. "
            "Targets filtered to remove sentinel/garbage. "
            "Models predict standardized targets; inverse via y_scalers at inference. "
            "Throughput computed from SINR via Shannon (BW=10MHz) at inference."
        ),
        "bounds": {
            "PRX_MIN": PRX_MIN,
            "PRX_MAX": PRX_MAX,
            "SINR_MIN": SINR_MIN,
            "SINR_MAX": SINR_MAX,
        },
    }

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(artifact, model_path)
    print(f"Saved surrogate model artifact to: {model_path}")
    print(f"Param ranges: {param_ranges}")

    # quick sanity print
    prx_mean = float(y_scalers["Prx_p5_dBm"].mean_[0])
    sinr_mean = float(y_scalers["SINR_p5_dB"].mean_[0])
    print(f"Sanity means: Prx_p5_dBm mean={prx_mean:.3f}, SINR_p5_dB mean={sinr_mean:.3f}")


if __name__ == "__main__":
    main()