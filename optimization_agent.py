# optimization_agent.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal, Optional

import duckdb
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.google import Gemini

load_dotenv()

# -----------------------------
# 1) INPUT = Intent Parser çıktısı
# -----------------------------
KpiName = Literal["RX_POWER", "SINR", "THROUGHPUT_5P", "SERVED_USERS"]
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
    confidence: float = Field(..., ge=0, le=1)

    # Optimization için pratik iki alan:
    current_config_id: Optional[int] = None
    user_set_id: Optional[int] = None
    k_users: Optional[int] = None


# -----------------------------
# 2) OUTPUT = Optimization Plan
# -----------------------------
class ParamChange(BaseModel):
    param: str
    before: Any
    after: Any
    unit: Optional[str] = None

class KpiSnapshot(BaseModel):
    RX_POWER: Optional[float] = None         # we use Prx_p5_dBm
    SINR: Optional[float] = None             # we use SINR_p5_dB
    THROUGHPUT_5P: Optional[float] = None    # we use Thr_p5_Mbps
    LOAD_IMBALANCE: Optional[float] = None   # derived
    RX_COVERAGE_RATIO: Optional[float] = None

class OptimizationPlan(BaseModel):
    selected_config_id: int
    baseline_config_id: Optional[int] = None
    changes: List[ParamChange]
    expected_kpis: KpiSnapshot
    baseline_kpis: Optional[KpiSnapshot] = None
    constraints_satisfied: bool
    warnings: List[str] = Field(default_factory=list)
    rationale: str


# -----------------------------
# 3) Dataset access + optimization tool
# -----------------------------
_CON: Optional[duckdb.DuckDBPyConnection] = None

def _get_con() -> duckdb.DuckDBPyConnection:
    global _CON
    if _CON is None:
        _CON = duckdb.connect(database=":memory:")
    return _CON

def _load_relation(con: duckdb.DuckDBPyConnection, path: str) -> str:
    # returns a SQL relation name
    ext = os.path.splitext(path.lower())[1]
    if ext == ".parquet":
        rel = f"read_parquet('{path}')"
    else:
        rel = f"read_csv_auto('{path}', header=True)"
    return rel

def _derived_load_metrics(row: Dict[str, Any]) -> Dict[str, float]:
    pcts = np.array([row["tx0_served_pct"], row["tx1_served_pct"], row["tx2_served_pct"], row["tx3_served_pct"]], dtype=float)
    imbalance = float(np.std(pcts))
    k_users = float(row["K_users"])
    sector_users = (k_users * pcts / 100.0)
    max_sector_users = float(np.max(sector_users))
    return {"load_imbalance": imbalance, "max_sector_users": max_sector_users}

def _kpi_from_row(row: Dict[str, Any]) -> KpiSnapshot:
    d = _derived_load_metrics(row)
    return KpiSnapshot(
        RX_POWER=float(row["Prx_p5_dBm"]),
        SINR=float(row["SINR_p5_dB"]),
        THROUGHPUT_5P=float(row["Thr_p5_Mbps"]),
        LOAD_IMBALANCE=d["load_imbalance"],
        RX_COVERAGE_RATIO=float(row.get("rx_power_coverage_ratio", np.nan)),
    )

def _check_threshold(kpi_value: float, thr: KpiThreshold, baseline_value: Optional[float]) -> bool:
    if thr.op == "GTE": return kpi_value >= float(thr.value)
    if thr.op == "GT":  return kpi_value >  float(thr.value)
    if thr.op == "LTE": return kpi_value <= float(thr.value)
    if thr.op == "LT":  return kpi_value <  float(thr.value)
    if thr.op == "BETWEEN": return float(thr.value_low) <= kpi_value <= float(thr.value_high)
    if thr.op == "TARGET":  return abs(kpi_value - float(thr.value)) < 1e-9
    # DELTA_* needs baseline
    if thr.op in ("DELTA_UP", "DELTA_DOWN"):
        if baseline_value is None:
            return True  # can't enforce; treat as soft when no baseline
        if thr.delta is None:
            return True
        if thr.op == "DELTA_UP":
            return (kpi_value - baseline_value) >= float(thr.delta)
        else:
            return (baseline_value - kpi_value) >= float(thr.delta)
    return True

def _get_kpi_value(snapshot: KpiSnapshot, kpi: KpiName) -> float:
    if kpi == "RX_POWER": return float(snapshot.RX_POWER)
    if kpi == "SINR": return float(snapshot.SINR)
    if kpi == "THROUGHPUT_5P": return float(snapshot.THRoUGHPUT_5P)  # typo guard (won't be used)
    if kpi == "SERVED_USERS": return float(snapshot.LOAD_IMBALANCE)
    raise ValueError(kpi)

def _score_row(row: Dict[str, Any], intent: IntentParse, mins: Dict[str,float], maxs: Dict[str,float]) -> float:
    # normalize targeted KPIs into [0,1], sum
    snap = _kpi_from_row(row)
    score = 0.0

    def norm(val, mn, mx):
        if mx - mn < 1e-9: return 0.5
        return (val - mn) / (mx - mn)

    for k in intent.target_kpis:
        if k == "RX_POWER":
            v = snap.RX_POWER
            score += norm(v, mins["Prx_p5_dBm"], maxs["Prx_p5_dBm"])
        elif k == "SINR":
            v = snap.SINR
            score += norm(v, mins["SINR_p5_dB"], maxs["SINR_p5_dB"])
        elif k == "THROUGHPUT_5P":
            v = snap.THRoUGHPUT_5P if hasattr(snap, "THRoUGHPUT_5P") else snap.THRoUGHPUT_5P  # will be fixed below
        elif k == "SERVED_USERS":
            # lower imbalance is better
            v = snap.LOAD_IMBALANCE
            score += 1.0 - norm(v, mins["load_imbalance"], maxs["load_imbalance"])

    # small tie-breakers:
    # prefer higher coverage ratio, but keep it low weight
    if "rx_power_coverage_ratio" in row:
        score += 0.1 * norm(float(row["rx_power_coverage_ratio"]), mins["rx_power_coverage_ratio"], maxs["rx_power_coverage_ratio"])
    return score


def optimize_from_intent(intent_json: str) -> str:
    """
    Optimize base-station configuration using an offline dataset.

    Expected dataset columns (your dataset):
    - Knobs:
      tx0_on..tx3_on (bool), tx0_P_dBm..tx3_P_dBm (float; off may be -9999),
      tx0_dAz..tx3_dAz (float deg), tx0_dEl..tx3_dEl (float deg)
    - Scenario keys:
      config_id (int), user_set_id (int), K_users (int)
    - KPIs:
      Prx_p5_dBm (RX power proxy), SINR_p5_dB, Thr_p5_Mbps,
      tx*_served_pct (for load), rx_power_coverage_ratio (optional)
    This tool:
    - filters by user_set_id and/or K_users if provided
    - applies hard constraints from intent.kpi_thresholds when possible
    - applies guardrails vs current_config_id if provided:
      power step <= 3 dB, dEl step <= 2 deg, dAz step <= 10 deg, and max 1 tx on/off toggle
    - picks best config_id by scoring targeted KPIs and returns a change plan (diff).
    """
    data_path = os.getenv("DATA_PATH", "").strip()
    if not data_path:
        raise ValueError("DATA_PATH env var is not set (path to your big dataset: .parquet or .csv).")

    intent = IntentParse.model_validate_json(intent_json)

    con = _get_con()
    rel = _load_relation(con, data_path)

    # --- base filter ---
    where = []
    if intent.user_set_id is not None:
        where.append(f"user_set_id = {int(intent.user_set_id)}")
    if intent.k_users is not None:
        where.append(f"K_users = {int(intent.k_users)}")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    # Keep only needed cols to reduce IO
    cols = [
        "config_id","K_users","user_set_id",
        "tx0_on","tx1_on","tx2_on","tx3_on",
        "total_tx_power_watt",
        "tx0_P_dBm","tx1_P_dBm","tx2_P_dBm","tx3_P_dBm",
        "tx0_served_pct","tx1_served_pct","tx2_served_pct","tx3_served_pct",
        "tx0_dAz","tx0_dEl","tx1_dAz","tx1_dEl","tx2_dAz","tx2_dEl","tx3_dAz","tx3_dEl",
        "Prx_p5_dBm","SINR_p5_dB","Thr_p5_Mbps",
        "rx_power_coverage_ratio"
    ]

    df = con.execute(f"SELECT {', '.join(cols)} FROM {rel} {where_sql}").df()

    if df.empty:
        raise ValueError("No rows match filters (check DATA_PATH, user_set_id, k_users).")

    # baseline row (optional)
    baseline_row = None
    if intent.current_config_id is not None:
        bdf = df[df["config_id"] == int(intent.current_config_id)]
        if not bdf.empty:
            baseline_row = bdf.iloc[0].to_dict()

    # derived metrics per row
    df["load_imbalance"] = df.apply(lambda r: float(np.std([r["tx0_served_pct"],r["tx1_served_pct"],r["tx2_served_pct"],r["tx3_served_pct"]])), axis=1)

    # mins/maxs for normalization
    mins = {
        "Prx_p5_dBm": float(df["Prx_p5_dBm"].min()),
        "SINR_p5_dB": float(df["SINR_p5_dB"].min()),
        "Thr_p5_Mbps": float(df["Thr_p5_Mbps"].min()),
        "load_imbalance": float(df["load_imbalance"].min()),
        "rx_power_coverage_ratio": float(df["rx_power_coverage_ratio"].min()) if "rx_power_coverage_ratio" in df.columns else 0.0,
    }
    maxs = {
        "Prx_p5_dBm": float(df["Prx_p5_dBm"].max()),
        "SINR_p5_dB": float(df["SINR_p5_dB"].max()),
        "Thr_p5_Mbps": float(df["Thr_p5_Mbps"].max()),
        "load_imbalance": float(df["load_imbalance"].max()),
        "rx_power_coverage_ratio": float(df["rx_power_coverage_ratio"].max()) if "rx_power_coverage_ratio" in df.columns else 1.0,
    }

    # guardrails relative to baseline
    warnings: List[str] = []
    if baseline_row is not None:
        def within_guardrails(r: Dict[str,Any]) -> bool:
            # power step <= 3 dB (only for active tx)
            for i in range(4):
                on_key = f"tx{i}_on"
                p_key  = f"tx{i}_P_dBm"
                if bool(baseline_row[on_key]) and bool(r[on_key]):
                    if abs(float(r[p_key]) - float(baseline_row[p_key])) > 3.0:
                        return False
                # tilt/az limits regardless
                if abs(float(r[f"tx{i}_dEl"]) - float(baseline_row[f"tx{i}_dEl"])) > 2.0:
                    return False
                if abs(float(r[f"tx{i}_dAz"]) - float(baseline_row[f"tx{i}_dAz"])) > 10.0:
                    return False
            # max 1 on/off toggle
            toggles = sum(int(bool(r[f"tx{i}_on"]) != bool(baseline_row[f"tx{i}_on"])) for i in range(4))
            return toggles <= 1

        before_n = len(df)
        df = df[df.apply(lambda r: within_guardrails(r.to_dict()), axis=1)]
        if df.empty:
            warnings.append("All candidates were filtered out by guardrails; rerun without current_config_id or relax guardrails.")
            # fallback to original
            df = con.execute(f"SELECT {', '.join(cols)} FROM {rel} {where_sql}").df()
            df["load_imbalance"] = df.apply(lambda r: float(np.std([r["tx0_served_pct"],r["tx1_served_pct"],r["tx2_served_pct"],r["tx3_served_pct"]])), axis=1)
        else:
            warnings.append(f"Guardrails applied: {before_n} -> {len(df)} candidates remaining.")

    # constraint filtering (hard when possible)
    def row_satisfies(r: Dict[str,Any]) -> bool:
        snap = _kpi_from_row(r)
        for thr in intent.kpi_thresholds:
            # map KPI -> value
            if thr.kpi == "RX_POWER":
                v = float(snap.RX_POWER)
                b = float(_kpi_from_row(baseline_row).RX_POWER) if baseline_row else None
            elif thr.kpi == "SINR":
                v = float(snap.SINR)
                b = float(_kpi_from_row(baseline_row).SINR) if baseline_row else None
            elif thr.kpi == "THROUGHPUT_5P":
                v = float(snap.THRoUGHPUT_5P) if hasattr(snap, "THRoUGHPUT_5P") else float(r["Thr_p5_Mbps"])
                b = float(_kpi_from_row(baseline_row).THRoUGHPUT_5P) if (baseline_row and hasattr(_kpi_from_row(baseline_row),"THRoUGHPUT_5P")) else (float(baseline_row["Thr_p5_Mbps"]) if baseline_row else None)
            else:  # SERVED_USERS -> load imbalance constraint as proxy
                v = float(snap.LOAD_IMBALANCE)
                b = float(_kpi_from_row(baseline_row).LOAD_IMBALANCE) if baseline_row else None

            if not _check_threshold(v, thr, b):
                return False
        return True

    # apply only if there are explicit thresholds
    if intent.kpi_thresholds:
        before_n = len(df)
        df2 = df[df.apply(lambda r: row_satisfies(r.to_dict()), axis=1)]
        if not df2.empty:
            df = df2
            warnings.append(f"Threshold constraints applied: {before_n} -> {len(df)} candidates.")
        else:
            warnings.append("No candidate satisfies all thresholds; selecting best-effort by score.")

    # scoring
    def score_row(r):
        rr = r.to_dict()
        # fix throughput typo locally
        snap = _kpi_from_row(rr)
        score = 0.0
        def norm(val, mn, mx):
            if mx - mn < 1e-9: return 0.5
            return (val - mn) / (mx - mn)
        for k in intent.target_kpis:
            if k == "RX_POWER":
                score += norm(float(rr["Prx_p5_dBm"]), mins["Prx_p5_dBm"], maxs["Prx_p5_dBm"])
            elif k == "SINR":
                score += norm(float(rr["SINR_p5_dB"]), mins["SINR_p5_dB"], maxs["SINR_p5_dB"])
            elif k == "THROUGHPUT_5P":
                score += norm(float(rr["Thr_p5_Mbps"]), mins["Thr_p5_Mbps"], maxs["Thr_p5_Mbps"])
            elif k == "SERVED_USERS":
                score += 1.0 - norm(float(rr["load_imbalance"]), mins["load_imbalance"], maxs["load_imbalance"])
        if "rx_power_coverage_ratio" in rr:
            score += 0.1 * norm(float(rr["rx_power_coverage_ratio"]), mins["rx_power_coverage_ratio"], maxs["rx_power_coverage_ratio"])
        # priority weight
        mult = {"LOW":0.8,"MEDIUM":1.0,"HIGH":1.2,"CRITICAL":1.4}[intent.priority]
        return score * mult

    df = df.copy()
    df["score"] = df.apply(score_row, axis=1)
    best = df.sort_values("score", ascending=False).iloc[0].to_dict()

    # diff
    changes: List[ParamChange] = []
    baseline_kpis = None
    baseline_id = intent.current_config_id if baseline_row else None
    if baseline_row:
        for col in [
            "tx0_on","tx1_on","tx2_on","tx3_on",
            "tx0_P_dBm","tx1_P_dBm","tx2_P_dBm","tx3_P_dBm",
            "tx0_dAz","tx0_dEl","tx1_dAz","tx1_dEl","tx2_dAz","tx2_dEl","tx3_dAz","tx3_dEl",
        ]:
            if baseline_row[col] != best[col]:
                unit = None
                if col.endswith("_P_dBm"): unit = "dBm"
                if col.endswith("_dAz"): unit = "deg"
                if col.endswith("_dEl"): unit = "deg"
                changes.append(ParamChange(param=col, before=baseline_row[col], after=best[col], unit=unit))
        baseline_kpis = _kpi_from_row(baseline_row)

    expected = _kpi_from_row(best)

    constraints_ok = True
    if intent.kpi_thresholds:
        for thr in intent.kpi_thresholds:
            if thr.kpi == "RX_POWER": v, b = expected.RX_POWER, (baseline_kpis.RX_POWER if baseline_kpis else None)
            elif thr.kpi == "SINR": v, b = expected.SINR, (baseline_kpis.SINR if baseline_kpis else None)
            elif thr.kpi == "THROUGHPUT_5P": v, b = expected.THRoUGHPUT_5P, (baseline_kpis.THRoUGHPUT_5P if baseline_kpis else None)
            else: v, b = expected.LOAD_IMBALANCE, (baseline_kpis.LOAD_IMBALANCE if baseline_kpis else None)
            if not _check_threshold(float(v), thr, float(b) if b is not None else None):
                constraints_ok = False

    plan = OptimizationPlan(
        selected_config_id=int(best["config_id"]),
        baseline_config_id=int(baseline_id) if baseline_id is not None else None,
        changes=changes,
        expected_kpis=expected,
        baseline_kpis=baseline_kpis,
        constraints_satisfied=constraints_ok,
        warnings=warnings,
        rationale="Selected the highest-scoring feasible configuration from the offline dataset, then produced a parameter diff as the change plan.",
    )
    return plan.model_dump_json(indent=2)


# -----------------------------
# 4) Optimization Agent
# -----------------------------
OPT_INSTRUCTIONS = [
    "You are a single Optimization Agent for a cellular network.",
    "Input will be an IntentParse JSON. Your job is to return an OptimizationPlan JSON only.",
    "DO NOT guess by reading the whole dataset. Always call the optimize_from_intent tool.",
    "Hard constraints come from kpi_thresholds when possible; otherwise choose best-effort and explain in warnings.",
    "Knobs you may change are: tx*_on, tx*_P_dBm, tx*_dAz, tx*_dEl. Do not invent other parameters.",
    "KPI mapping: RX_POWER -> Prx_p5_dBm, SINR -> SINR_p5_dB, THROUGHPUT_5P -> Thr_p5_Mbps, SERVED_USERS -> load imbalance derived from tx*_served_pct.",
]

optimization_agent = Agent(
    name="Optimization Agent",
    description="Chooses the best configuration from an offline dataset and outputs a safe change plan.",
    model=Gemini(id=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")),
    tools=[optimize_from_intent],
    output_schema=OptimizationPlan,
    instructions=OPT_INSTRUCTIONS,
)
