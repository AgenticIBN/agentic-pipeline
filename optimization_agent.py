# optimization_agent.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import duckdb
import joblib
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.groq import Groq

load_dotenv()

# -----------------------------
# 1) INPUT schema
# -----------------------------
KpiName = Literal["RX_POWER", "SINR", "THROUGHPUT_5P", "SERVED_USERS", "RX_COVERAGE_RATIO"]
Op = Literal["GT", "GTE", "LT", "LTE", "BETWEEN", "DELTA_UP", "DELTA_DOWN", "TARGET"]


class KpiThreshold(BaseModel):
    kpi: KpiName
    op: Op
    value: Optional[float] = None
    value_low: Optional[float] = None
    value_high: Optional[float] = None
    delta: Optional[float] = None
    unit: Optional[str] = None


class IntentParse(BaseModel):
    target_area: str
    target_kpis: List[KpiName]
    kpi_thresholds: List[KpiThreshold] = Field(default_factory=list)
    time_constraint_start: Optional[str] = None
    time_constraint_end: Optional[str] = None
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    configuration_change: List[Dict[str, Any]] = Field(default_factory=list)
    affected_sectors: List[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=0.8, ge=0, le=1)  # Optional with default

    current_config_id: Optional[int] = None
    current_config: Optional[Dict[str, Any]] = None  # Direct config dict (alternative to config_id)


# -----------------------------
# 2) OUTPUT schema
# -----------------------------
class ParamChange(BaseModel):
    param: str
    before: Any
    change: Any  # The change/delta to apply (was 'after')
    unit: Optional[str] = None


class KpiSnapshot(BaseModel):
    RX_POWER: Optional[float] = None
    SINR: Optional[float] = None
    THROUGHPUT_5P: Optional[float] = None
    LOAD_IMBALANCE: Optional[float] = None
    RX_COVERAGE_RATIO: Optional[float] = None


class OptimizationPlan(BaseModel):
    selected_config_id: int
    current_config_id: Optional[int] = None
    changes: List[ParamChange]
    expected_kpis: KpiSnapshot
    current_kpis: Optional[KpiSnapshot] = None
    constraints_satisfied: bool


# -----------------------------
# 3) DB + relation
# -----------------------------
_CON: Optional[duckdb.DuckDBPyConnection] = None


def _get_con() -> duckdb.DuckDBPyConnection:
    global _CON
    if _CON is None:
        _CON = duckdb.connect(database=":memory:")
    return _CON


def _rel_from_path(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".parquet":
        return f"read_parquet('{path}')"
    return f"read_csv_auto('{path}', header=True)"


# -----------------------------
# 4) Surrogate model loader + predictor
# -----------------------------
@dataclass
class Surrogate:
    feature_cols: List[str]
    target_cols: List[str]
    scaler: Any
    models: Dict[str, Any]
    param_ranges: Dict[str, Dict[str, float]]
    bw_hz: int
    y_scalers: Dict[str, Any]

    def predict(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """
        Predicts:
          - Prx_p5_dBm, SINR_p5_dB, rx_power_coverage_ratio, tx*_served_pct
        Then throughput is computed from SINR via Shannon with BW.
        """
        # Build a single-row feature vector
        row = {}
        row["user_set_id"] = int(context.get("user_set_id", 0))
        row["K_users"] = int(context.get("K_users", 0))
        row["rx_power_thr_dBm"] = float(context.get("rx_power_thr_dBm", -95.0))
        row["total_tx_power_watt"] = float(context.get("total_tx_power_watt", 0.0))

        for i in range(4):
            row[f"tx{i}_on"] = 1 if bool(config.get(f"tx{i}_on", True)) else 0
            # if off -> set 0
            if row[f"tx{i}_on"] == 0:
                row[f"tx{i}_P_dBm"] = 0.0
                row[f"tx{i}_dAz"] = 0.0
                row[f"tx{i}_dEl"] = 0.0
            else:
                row[f"tx{i}_P_dBm"] = float(config.get(f"tx{i}_P_dBm", 0.0))
                row[f"tx{i}_dAz"] = float(config.get(f"tx{i}_dAz", 0.0))
                row[f"tx{i}_dEl"] = float(config.get(f"tx{i}_dEl", 0.0))

        # vectorize
        X = np.array([[float(row[c]) for c in self.feature_cols]], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self.scaler.transform(X)

        preds: Dict[str, float] = {}
        for t in self.target_cols:
            # model standardize edilmiş target tahmin ediyor (y_std)
            y_std = float(self.models[t].predict(Xs)[0])

            # tekrar gerçek birime döndür
            y = float(self.y_scalers[t].inverse_transform([[y_std]])[0, 0])
            preds[t] = y

        # Clip bounded ones
        preds["rx_power_coverage_ratio"] = float(np.clip(preds["rx_power_coverage_ratio"], 0.0, 1.0))
        for i in range(4):
            k = f"tx{i}_served_pct"
            preds[k] = float(np.clip(preds[k], 0.0, 100.0))

        return preds

    def throughput_from_sinr(self, sinr_db: float) -> float:
        sinr_lin = 10.0 ** (sinr_db / 10.0)
        thr_mbps = (self.bw_hz * np.log2(1.0 + sinr_lin)) / 1e6
        return float(thr_mbps)


_SUR: Optional[Surrogate] = None


def _load_surrogate() -> Surrogate:
    global _SUR
    if _SUR is not None:
        return _SUR

    model_path = os.getenv("MODEL_PATH", "").strip()
    if not model_path:
        raise ValueError("MODEL_PATH env var is not set. Train and set MODEL_PATH first.")

    art = joblib.load(model_path)
    _SUR = Surrogate(
        feature_cols=art["feature_cols"],
        target_cols=art["target_cols"],
        scaler=art["scaler"],
        y_scalers=art["y_scalers"],
        models=art["models"],
        param_ranges=art["param_ranges"],
        bw_hz=int(art.get("bw_hz", 10_000_000)),
    )
    return _SUR


# -----------------------------
# 5) KPI + constraints helpers
# -----------------------------
def _derived_load_imbalance(served_pcts: List[float], k_users: float) -> float:
    p = np.array(served_pcts, dtype=float)
    # imbalance = std dev of served percentages (simple & robust)
    return float(np.std(p))


def _check_threshold(kpi_value: float, thr: KpiThreshold, baseline_value: Optional[float]) -> bool:
    if thr.op == "GTE":
        return kpi_value >= float(thr.value)
    if thr.op == "GT":
        return kpi_value > float(thr.value)
    if thr.op == "LTE":
        return kpi_value <= float(thr.value)
    if thr.op == "LT":
        return kpi_value < float(thr.value)
    if thr.op == "BETWEEN":
        return float(thr.value_low) <= kpi_value <= float(thr.value_high)
    if thr.op == "TARGET":
        return abs(kpi_value - float(thr.value)) < 1e-9
    if thr.op in ("DELTA_UP", "DELTA_DOWN"):
        # Needs baseline
        if baseline_value is None or thr.delta is None:
            return True
        if thr.op == "DELTA_UP":
            return (kpi_value - baseline_value) >= float(thr.delta)
        else:
            return (baseline_value - kpi_value) >= float(thr.delta)
    return True


def _relaxed_sql_filter_for_thresholds(thresholds: List[KpiThreshold]) -> str:
    """
    Apply ONLY hard constraints that exist as dataset columns:
      RX_POWER -> Prx_p5_dBm
      SINR -> SINR_p5_dB
      THROUGHPUT_5P -> Thr_p5_Mbps
      RX_COVERAGE_RATIO -> rx_power_coverage_ratio (but careful: depends on rx_power_thr_dBm in dataset)
    For coverage ratio, we still filter by ratio alone (best-effort); exact threshold alignment is handled by surrogate in online phase.
    """
    clauses = []
    for thr in thresholds:
        if thr.kpi == "RX_POWER":
            col = "Prx_p5_dBm"
        elif thr.kpi == "SINR":
            col = "SINR_p5_dB"
        elif thr.kpi == "THROUGHPUT_5P":
            col = "Thr_p5_Mbps"
        elif thr.kpi == "RX_COVERAGE_RATIO":
            col = "rx_power_coverage_ratio"
        else:
            continue

        if thr.op in ("GTE", "GT", "LTE", "LT") and thr.value is not None:
            op_map = {"GTE": ">=", "GT": ">", "LTE": "<=", "LT": "<"}
            clauses.append(f"{col} {op_map[thr.op]} {float(thr.value)}")
        elif thr.op == "BETWEEN" and thr.value_low is not None and thr.value_high is not None:
            clauses.append(f"{col} BETWEEN {float(thr.value_low)} AND {float(thr.value_high)}")

    return " AND ".join(clauses)


def _priority_weight(priority: str) -> float:
    return {"LOW": 0.5, "MEDIUM": 1.0, "HIGH": 1.5, "CRITICAL": 2.0}.get(priority, 1.0)


# -----------------------------
# 6) Seed selection + search
# -----------------------------
def _fetch_seed_rows(
    con: duckdb.DuckDBPyConnection,
    rel: str,
    intent: IntentParse,
    limit: int,
) -> List[Dict[str, Any]]:
    where = []

    if intent.user_set_id is not None:
        where.append(f"user_set_id = {int(intent.user_set_id)}")
    if intent.k_users is not None:
        where.append(f"K_users = {int(intent.k_users)}")

    thr_sql = _relaxed_sql_filter_for_thresholds(intent.kpi_thresholds)
    if thr_sql:
        where.append(thr_sql)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    # Order by most relevant KPI label from dataset as a starting point
    order_by = []
    if "RX_POWER" in intent.target_kpis:
        order_by.append("Prx_p5_dBm DESC")
    if "SINR" in intent.target_kpis:
        order_by.append("SINR_p5_dB DESC")
    if "THROUGHPUT_5P" in intent.target_kpis:
        order_by.append("Thr_p5_Mbps DESC")
    # For served users goal, prefer lower imbalance: approximate via served pct spread is not directly in SQL -> skip

    order_sql = ("ORDER BY " + ", ".join(order_by)) if order_by else ""

    q = f"""
    SELECT
      config_id, user_set_id, K_users, total_tx_power_watt,
      tx0_on, tx1_on, tx2_on, tx3_on,
      tx0_P_dBm, tx1_P_dBm, tx2_P_dBm, tx3_P_dBm,
      tx0_dAz, tx1_dAz, tx2_dAz, tx3_dAz,
      tx0_dEl, tx1_dEl, tx2_dEl, tx3_dEl,
      Prx_p5_dBm, SINR_p5_dB, Thr_p5_Mbps,
      rx_power_thr_dBm, rx_power_coverage_ratio,
      tx0_served_pct, tx1_served_pct, tx2_served_pct, tx3_served_pct
    FROM {rel}
    {where_sql}
    {order_sql}
    LIMIT {int(limit)}
    """
    df = con.execute(q).df()
    if df.empty:
        return []
    return [r for r in df.to_dict(orient="records")]


def _config_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {}
    for i in range(4):
        cfg[f"tx{i}_on"] = bool(row[f"tx{i}_on"])
        cfg[f"tx{i}_P_dBm"] = float(row[f"tx{i}_P_dBm"])
        cfg[f"tx{i}_dAz"] = float(row[f"tx{i}_dAz"])
        cfg[f"tx{i}_dEl"] = float(row[f"tx{i}_dEl"])
    return cfg


def _apply_guardrails(new_cfg: Dict[str, Any], base_cfg: Dict[str, Any], warnings: List[str]) -> None:
    for i in range(4):
        if bool(new_cfg[f"tx{i}_on"]) and bool(base_cfg[f"tx{i}_on"]):
            # power ±3 dB
            diff = abs(float(new_cfg[f"tx{i}_P_dBm"]) - float(base_cfg[f"tx{i}_P_dBm"]))
            if diff > 3.0:
                sign = np.sign(float(new_cfg[f"tx{i}_P_dBm"]) - float(base_cfg[f"tx{i}_P_dBm"]))
                new_cfg[f"tx{i}_P_dBm"] = float(base_cfg[f"tx{i}_P_dBm"]) + float(sign) * 3.0
                warnings.append(f"Power change for tx{i} capped at 3 dB (guardrail)")

        # tilt ±2°
        diff_el = abs(float(new_cfg[f"tx{i}_dEl"]) - float(base_cfg[f"tx{i}_dEl"]))
        if diff_el > 2.0:
            sign = np.sign(float(new_cfg[f"tx{i}_dEl"]) - float(base_cfg[f"tx{i}_dEl"]))
            new_cfg[f"tx{i}_dEl"] = float(base_cfg[f"tx{i}_dEl"]) + float(sign) * 2.0
            warnings.append(f"Tilt change for tx{i} capped at 2° (guardrail)")

        # azimuth ±10°
        diff_az = abs(float(new_cfg[f"tx{i}_dAz"]) - float(base_cfg[f"tx{i}_dAz"]))
        if diff_az > 10.0:
            sign = np.sign(float(new_cfg[f"tx{i}_dAz"]) - float(base_cfg[f"tx{i}_dAz"]))
            new_cfg[f"tx{i}_dAz"] = float(base_cfg[f"tx{i}_dAz"]) + float(sign) * 10.0
            warnings.append(f"Azimuth change for tx{i} capped at 10° (guardrail)")


def _score_candidate(
    intent: IntentParse,
    kpis: KpiSnapshot,
    thr_list: List[KpiThreshold],
) -> float:
    """
    Higher is better. Uses target_kpis + constraints margins.
    """
    w = _priority_weight(intent.priority)
    score = 0.0

    # Reward improvements over constraints / typical targets
    for thr in thr_list:
        if thr.kpi == "RX_POWER" and kpis.RX_POWER is not None and thr.value is not None:
            # margin above threshold
            score += w * (float(kpis.RX_POWER) - float(thr.value))
        elif thr.kpi == "SINR" and kpis.SINR is not None and thr.value is not None:
            score += w * (float(kpis.SINR) - float(thr.value))
        elif thr.kpi == "THROUGHPUT_5P" and kpis.THROUGHPUT_5P is not None and thr.value is not None:
            score += w * (float(kpis.THROUGHPUT_5P) - float(thr.value))
        elif thr.kpi == "RX_COVERAGE_RATIO" and kpis.RX_COVERAGE_RATIO is not None and thr.value is not None:
            score += w * (float(kpis.RX_COVERAGE_RATIO) - float(thr.value))
        elif thr.kpi == "SERVED_USERS" and kpis.LOAD_IMBALANCE is not None and thr.value is not None:
            # smaller imbalance is better -> invert
            score += w * (float(thr.value) - float(kpis.LOAD_IMBALANCE))

    # Also consider direct optimization even without explicit threshold
    if "RX_POWER" in intent.target_kpis and kpis.RX_POWER is not None:
        score += 0.2 * w * float(kpis.RX_POWER)
    if "SINR" in intent.target_kpis and kpis.SINR is not None:
        score += 0.2 * w * float(kpis.SINR)
    if "THROUGHPUT_5P" in intent.target_kpis and kpis.THROUGHPUT_5P is not None:
        score += 0.02 * w * float(kpis.THROUGHPUT_5P)
    if "SERVED_USERS" in intent.target_kpis and kpis.LOAD_IMBALANCE is not None:
        score += 0.5 * w * (-float(kpis.LOAD_IMBALANCE))

    return float(score)


def _constraints_ok(intent: IntentParse, cand: KpiSnapshot, base: Optional[KpiSnapshot]) -> Tuple[bool, List[str]]:
    warns = []
    ok = True
    for thr in intent.kpi_thresholds:
        if thr.kpi == "RX_POWER":
            v = cand.RX_POWER
            b = base.RX_POWER if base else None
        elif thr.kpi == "SINR":
            v = cand.SINR
            b = base.SINR if base else None
        elif thr.kpi == "THROUGHPUT_5P":
            v = cand.THROUGHPUT_5P
            b = base.THROUGHPUT_5P if base else None
        elif thr.kpi == "RX_COVERAGE_RATIO":
            v = cand.RX_COVERAGE_RATIO
            b = base.RX_COVERAGE_RATIO if base else None
        else:  # SERVED_USERS uses imbalance
            v = cand.LOAD_IMBALANCE
            b = base.LOAD_IMBALANCE if base else None

        if v is None:
            # Can't verify -> treat as soft warning
            warns.append(f"Cannot evaluate constraint for {thr.kpi} (missing KPI).")
            continue

        if not _check_threshold(float(v), thr, float(b) if b is not None else None):
            ok = False
    return ok, warns


# -----------------------------
# 7) Main tool: optimize_from_intent
# -----------------------------
def optimize_from_intent(intent_json: str) -> str:
    data_path = os.getenv("DATA_PATH", "").strip()
    if not data_path:
        raise ValueError("DATA_PATH env var is not set (path to dataset).")

    intent = IntentParse.model_validate_json(intent_json)

    con = _get_con()
    rel = _rel_from_path(data_path)
    sur = _load_surrogate()

    # ----------------------------------------
    # Current config (from dict or dataset)
    # ----------------------------------------
    current_row = None
    current_cfg = None
    current_kpis = None

    # Option 1: Current config provided directly as dict
    if intent.current_config is not None:
        current_cfg = dict(intent.current_config)
        # Compute current KPIs using surrogate
        context_temp = {
            "user_set_id": int(intent.user_set_id or 0),
            "K_users": int(intent.k_users or 0),
            "rx_power_thr_dBm": -95.0,  # default
            "total_tx_power_watt": 0.0,
        }
        preds = sur.predict(current_cfg, context_temp)
        served = [float(preds[f"tx{i}_served_pct"]) for i in range(4)]
        imb = _derived_load_imbalance(served, float(context_temp["K_users"]))
        thr_mbps = sur.throughput_from_sinr(float(preds["SINR_p5_dB"]))
        current_kpis = KpiSnapshot(
            RX_POWER=float(preds["Prx_p5_dBm"]),
            SINR=float(preds["SINR_p5_dB"]),
            THROUGHPUT_5P=thr_mbps,
            LOAD_IMBALANCE=float(imb),
            RX_COVERAGE_RATIO=float(preds["rx_power_coverage_ratio"]),
        )
    # Option 2: Fetch from dataset by config_id
    elif intent.current_config_id is not None:
        q = f"SELECT * FROM {rel} WHERE config_id = {int(intent.current_config_id)} LIMIT 1"
        df = con.execute(q).df()
        if not df.empty:
            current_row = df.iloc[0].to_dict()
            current_cfg = _config_from_row(current_row)
            # current KPIs from dataset actual labels:
            served = [float(current_row[f"tx{i}_served_pct"]) for i in range(4)]
            imb = _derived_load_imbalance(served, float(current_row["K_users"]))
            current_kpis = KpiSnapshot(
                RX_POWER=float(current_row["Prx_p5_dBm"]),
                SINR=float(current_row["SINR_p5_dB"]),
                THROUGHPUT_5P=float(current_row["Thr_p5_Mbps"]),
                LOAD_IMBALANCE=float(imb),
                RX_COVERAGE_RATIO=float(current_row.get("rx_power_coverage_ratio", np.nan)),
            )

    # ----------------------------------------
    # Determine threshold context for coverage
    # ----------------------------------------
    rx_thr = None
    for thr in intent.kpi_thresholds:
        if thr.kpi == "RX_POWER" and thr.value is not None:
            rx_thr = float(thr.value)
        if thr.kpi == "RX_COVERAGE_RATIO" and thr.unit is not None:
            # not used; ratio is unitless
            pass
    if rx_thr is None:
        rx_thr = -95.0  # default if not provided

    # ----------------------------------------
    # Seed selection from dataset
    # ----------------------------------------
    seed_limit = int(os.getenv("SEED_LIMIT", "50"))
    seed_rows = _fetch_seed_rows(con, rel, intent, seed_limit)

    if not seed_rows and current_cfg is None:
        # fallback: build a reasonable default from ranges (NOT zeros)
        pr = sur.param_ranges
        mid_p = (pr["power"]["min"] + pr["power"]["max"]) / 2.0
        mid_az = 0.0
        mid_el = 0.0
        base_cfg = {}
        for i in range(4):
            base_cfg[f"tx{i}_on"] = True
            base_cfg[f"tx{i}_P_dBm"] = mid_p
            base_cfg[f"tx{i}_dAz"] = mid_az
            base_cfg[f"tx{i}_dEl"] = mid_el
        seed_cfgs = [base_cfg]
    else:
        seed_cfgs = [(_config_from_row(r)) for r in seed_rows]

    # If current config exists, search starts from current config first
    if current_cfg is not None:
        seed_cfgs = [current_cfg] + seed_cfgs

    # ----------------------------------------
    # Online search using surrogate
    # ----------------------------------------
    iters = int(os.getenv("SEARCH_ITERS", "1500"))
    rng = np.random.default_rng(int(os.getenv("SEARCH_SEED", "42")))

    pr = sur.param_ranges
    pmin, pmax = pr["power"]["min"], pr["power"]["max"]
    azmin, azmax = pr["dAz"]["min"], pr["dAz"]["max"]
    elmin, elmax = pr["dEl"]["min"], pr["dEl"]["max"]

    # step sizes (tunable)
    p_step = float(os.getenv("P_STEP_DB", "1.0"))
    az_step = float(os.getenv("AZ_STEP_DEG", "2.0"))
    el_step = float(os.getenv("EL_STEP_DEG", "0.5"))

    best_cfg = None
    best_kpis = None
    best_score = -1e18

    context = {
        "user_set_id": int(intent.user_set_id or 0),
        "K_users": int(intent.k_users or 0),
        "rx_power_thr_dBm": float(rx_thr),
        "total_tx_power_watt": 0.0,  # not strictly needed; can be 0
    }

    def eval_cfg(cfg: Dict[str, Any]) -> Tuple[KpiSnapshot, float, bool, List[str]]:
        preds = sur.predict(cfg, context)
        rx = float(preds["Prx_p5_dBm"])
        sinr = float(preds["SINR_p5_dB"])
        cov = float(preds["rx_power_coverage_ratio"])
        thr_mbps = sur.throughput_from_sinr(sinr)

        served = [float(preds[f"tx{i}_served_pct"]) for i in range(4)]
        imb = _derived_load_imbalance(served, float(context["K_users"]))

        snap = KpiSnapshot(
            RX_POWER=rx,
            SINR=sinr,
            THROUGHPUT_5P=thr_mbps,
            LOAD_IMBALANCE=imb,
            RX_COVERAGE_RATIO=cov,
        )
        ok, warn2 = _constraints_ok(intent, snap, current_kpis)
        sc = _score_candidate(intent, snap, intent.kpi_thresholds)
        return snap, sc, ok, warn2

    # Search: for each seed, do random local perturbations
    for seed in seed_cfgs[: max(1, min(len(seed_cfgs), 10))]:
        # Evaluate seed itself
        snap, sc, ok, warn2 = eval_cfg(seed)
        if ok and sc > best_score:
            best_score, best_cfg, best_kpis = sc, dict(seed), snap
        for w in warn2:
            if w not in warnings:
                warnings.append(w)

        # Perturb around this seed
        for _ in range(iters):
            cfg = dict(seed)

            for i in range(4):
                if not bool(cfg[f"tx{i}_on"]):
                    continue

                # random perturbations (small steps)
                cfg[f"tx{i}_P_dBm"] = float(np.clip(
                    cfg[f"tx{i}_P_dBm"] + rng.normal(0.0, p_step),
                    pmin, pmax
                ))
                cfg[f"tx{i}_dAz"] = float(np.clip(
                    cfg[f"tx{i}_dAz"] + rng.normal(0.0, az_step),
                    azmin, azmax
                ))
                cfg[f"tx{i}_dEl"] = float(np.clip(
                    cfg[f"tx{i}_dEl"] + rng.normal(0.0, el_step),
                    elmin, elmax
                ))

            snap, sc, ok, warn2 = eval_cfg(cfg)
            if ok and sc > best_score:
                best_score, best_cfg, best_kpis = sc, cfg, snap

    if best_cfg is None or best_kpis is None:
        # If nothing satisfies constraints, pick the best-scoring regardless
        # fallback: evaluate first seed
        best_cfg = seed_cfgs[0]
        best_kpis, best_score, _, _ = eval_cfg(best_cfg)

    # Apply guardrails if current config exists
    if current_cfg is not None:
        _apply_guardrails(best_cfg, current_cfg, [])

    # Build changes list
    changes: List[ParamChange] = []
    if current_cfg is not None:
        # With current config: before=current, change=delta (change amount)
        for i in range(4):
            for param_type in ["on", "P_dBm", "dEl", "dAz"]:
                col = f"tx{i}_{param_type}"
                before = current_cfg[col]
                after_val = best_cfg[col]
                if before != after_val:
                    unit = None
                    # For numeric params, 'change' shows the DELTA (change amount)
                    if param_type in ["P_dBm", "dEl", "dAz"]:
                        delta = float(after_val) - float(before)
                        change = delta  # Show change amount (+ for increase, - for decrease)
                        if param_type == "P_dBm":
                            unit = "dBm"
                        else:
                            unit = "deg"
                    else:
                        # For boolean 'on', show new state
                        change = after_val
                    
                    changes.append(ParamChange(param=col, before=before, change=change, unit=unit))
    else:
        # Without current config: before=None, change=new_value
        for i in range(4):
            changes.append(ParamChange(param=f"tx{i}_on", before=None, change=best_cfg[f"tx{i}_on"]))
            changes.append(ParamChange(param=f"tx{i}_P_dBm", before=None, change=best_cfg[f"tx{i}_P_dBm"], unit="dBm"))
            changes.append(ParamChange(param=f"tx{i}_dEl", before=None, change=best_cfg[f"tx{i}_dEl"], unit="deg"))
            changes.append(ParamChange(param=f"tx{i}_dAz", before=None, change=best_cfg[f"tx{i}_dAz"], unit="deg"))

    # Final constraint status
    constraints_ok, _ = _constraints_ok(intent, best_kpis, current_kpis)

    # Generate unique config ID based on timestamp
    timestamp_id = int(datetime.now().strftime("%Y%m%d%H%M%S"))
    selected_id = timestamp_id

    plan = OptimizationPlan(
        selected_config_id=selected_id,
        current_config_id=intent.current_config_id,
        changes=changes,
        expected_kpis=best_kpis,
        current_kpis=current_kpis,
        constraints_satisfied=constraints_ok,
    )
    return plan.model_dump_json(indent=2)


# -----------------------------
# 8) Agent wiring
# -----------------------------
OPT_INSTRUCTIONS = [
    "You are a Configuration Optimization Agent for a cellular base station network.",
    "Input: IntentParse JSON. Output: OptimizationPlan JSON.",
    "You MUST call the optimize_from_intent tool.",
    "Optimization method: offline-trained surrogate model + online search (seeded from dataset) to generate a new configuration.",
    "Adjustable parameters: tx*_on, tx*_P_dBm, tx*_dAz, tx*_dEl.",
    "KPI mappings: RX_POWER -> Prx_p5_dBm, SINR -> SINR_p5_dB, THROUGHPUT_5P computed from SINR via Shannon at BW=10MHz, SERVED_USERS via load imbalance from served_pct predictions.",
    "Guardrails when baseline exists: power ±3dB, tilt ±2°, azimuth ±10°.",
    "Return only the OptimizationPlan JSON.",
]

optimization_agent = Agent(
    name="Optimization Agent",
    description="Generates a base-station configuration using a trained surrogate model and dataset-seeded search.",
    model=Groq(id=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")),
    tools=[optimize_from_intent],
    output_schema=OptimizationPlan,
    instructions=OPT_INSTRUCTIONS,
)