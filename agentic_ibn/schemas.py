from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Priority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def weight(self) -> int:
        return {self.LOW: 1, self.MEDIUM: 2, self.HIGH: 3, self.CRITICAL: 4}[self]


class Operator(str, Enum):
    GT = "GT"
    GTE = "GTE"
    LT = "LT"
    LTE = "LTE"
    BETWEEN = "BETWEEN"
    DELTA_UP = "DELTA_UP"
    DELTA_DOWN = "DELTA_DOWN"
    TARGET = "TARGET"


class KPIName(str, Enum):
    RX_POWER = "RX_POWER"
    SINR = "SINR"
    COVERAGE = "COVERAGE"
    THROUGHPUT_5P = "THROUGHPUT_5P"
    THROUGHPUT_RR_5P = "THROUGHPUT_RR_5P"
    TOTAL_TX_POWER = "TOTAL_TX_POWER"
    LOAD_BALANCE = "LOAD_BALANCE"
    LOAD_IMBALANCE = "LOAD_IMBALANCE"
    SERVED_USERS = "SERVED_USERS"


class KpiThreshold(StrictModel):
    kpi: KPIName
    operator: Operator
    value: float | None = None
    value_low: float | None = None
    value_high: float | None = None
    delta: float | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def validate_operator_fields(self) -> "KpiThreshold":
        if self.operator == Operator.BETWEEN and (self.value_low is None or self.value_high is None):
            raise ValueError("BETWEEN requires value_low and value_high")
        if self.operator in {Operator.DELTA_UP, Operator.DELTA_DOWN} and self.delta is None:
            raise ValueError(f"{self.operator.value} requires delta")
        if self.operator in {Operator.GT, Operator.GTE, Operator.LT, Operator.LTE} and self.value is None:
            raise ValueError(f"{self.operator.value} requires value")
        return self




class IntentParse(StrictModel):
    """Structured semantic output produced directly by the Agno Intent Parser Agent."""

    target_area: str
    target_kpis: list[KPIName] = Field(default_factory=list)
    kpi_thresholds: list[KpiThreshold] = Field(default_factory=list)
    time_constraint_start: datetime | None = None
    time_constraint_end: datetime | None = None
    priority: Priority = Priority.MEDIUM
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ParsedIntent(StrictModel):
    raw_text: str
    target_area: str
    scenario: str
    target_kpis: list[KPIName] = Field(default_factory=list)
    kpi_thresholds: list[KpiThreshold] = Field(default_factory=list)
    time_constraint_start: datetime | None = None
    time_constraint_end: datetime | None = None
    priority: Priority = Priority.MEDIUM
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class TxConfig(StrictModel):
    on: bool = True
    power_dbm: float = 43.0
    azimuth_delta_deg: float = 0.0
    elevation_delta_deg: float = 10.0


class NetworkConfig(StrictModel):
    tx0: TxConfig = Field(default_factory=TxConfig)
    tx1: TxConfig = Field(default_factory=TxConfig)
    tx2: TxConfig = Field(default_factory=TxConfig)
    tx3: TxConfig = Field(default_factory=TxConfig)
    k_users: int = Field(default=200, ge=1)
    user_set_id: int = Field(default=0, ge=0)
    coverage_threshold_dbm: float = -60.0

    @field_validator("tx0", "tx1", "tx2", "tx3")
    @classmethod
    def finite_tx_values(cls, value: TxConfig) -> TxConfig:
        for number in (value.power_dbm, value.azimuth_delta_deg, value.elevation_delta_deg):
            if number != number or number in {float("inf"), float("-inf")}:
                raise ValueError("TX parameters must be finite")
        return value

    def tx(self, index: int) -> TxConfig:
        if index not in range(4):
            raise IndexError("TX index must be between 0 and 3")
        return getattr(self, f"tx{index}")

    def with_tx(self, index: int, tx_config: TxConfig) -> "NetworkConfig":
        data = self.model_dump()
        data[f"tx{index}"] = tx_config.model_dump()
        return NetworkConfig.model_validate(data)

    def to_flat_dict(self) -> dict[str, float | int | bool]:
        row: dict[str, float | int | bool] = {
            "K_users": self.k_users,
            "user_set_id": self.user_set_id,
            "rx_power_thr_dBm": self.coverage_threshold_dbm,
        }
        total_watt = 0.0
        for index in range(4):
            tx = self.tx(index)
            row[f"tx{index}_on"] = tx.on
            row[f"tx{index}_P_dBm"] = tx.power_dbm if tx.on else 0.0
            row[f"tx{index}_dAz"] = tx.azimuth_delta_deg if tx.on else 0.0
            row[f"tx{index}_dEl"] = tx.elevation_delta_deg if tx.on else 0.0
            if tx.on:
                total_watt += 10 ** ((tx.power_dbm - 30.0) / 10.0)
        row["total_tx_power_watt"] = total_watt
        return row

    @classmethod
    def from_flat_dict(cls, data: dict[str, Any]) -> "NetworkConfig":
        txs: dict[str, Any] = {}
        for index in range(4):
            on = bool(data.get(f"tx{index}_on", True))
            power = float(data.get(f"tx{index}_P_dBm", 43.0))
            if power < -1000:
                power = 0.0
                on = False
            txs[f"tx{index}"] = {
                "on": on,
                "power_dbm": power if on else 0.0,
                "azimuth_delta_deg": float(data.get(f"tx{index}_dAz", 0.0)) if on else 0.0,
                "elevation_delta_deg": float(data.get(f"tx{index}_dEl", 10.0)) if on else 0.0,
            }
        return cls(
            **txs,
            k_users=int(data.get("K_users", 200)),
            user_set_id=int(data.get("user_set_id", 0)),
            coverage_threshold_dbm=float(data.get("rx_power_thr_dBm", -60.0)),
        )


class HardConstraints(StrictModel):
    tx_states: dict[int, bool] = Field(default_factory=dict)
    all_available_requested: bool = False
    notes: list[str] = Field(default_factory=list)


class CandidateProposal(StrictModel):
    configuration: NetworkConfig
    rationale: str
    expected_tradeoffs: list[str]


class ValidationCorrection(StrictModel):
    parameter: str
    proposed_value: Any
    applied_value: Any
    reason: str


class ValidationResult(StrictModel):
    valid: bool
    configuration: NetworkConfig
    corrections: list[ValidationCorrection] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class KPIPrediction(StrictModel):
    rx_power_p5_dbm: float | None = None
    rx_power_mean_dbm: float | None = None
    sinr_p5_db: float | None = None
    sinr_mean_db: float | None = None
    throughput_p5_mbps: float | None = None
    throughput_rr_p5_mbps: float | None = None
    coverage_ratio: float | None = None
    load_balance: float | None = None
    load_imbalance: float | None = None
    total_tx_power_watt: float
    raw_targets: dict[str, float] = Field(default_factory=dict)

    def value_for(self, kpi: KPIName) -> float | None:
        mapping = {
            KPIName.RX_POWER: self.rx_power_p5_dbm if self.rx_power_p5_dbm is not None else self.rx_power_mean_dbm,
            KPIName.SINR: self.sinr_p5_db if self.sinr_p5_db is not None else self.sinr_mean_db,
            KPIName.COVERAGE: self.coverage_ratio,
            KPIName.THROUGHPUT_5P: self.throughput_p5_mbps,
            KPIName.THROUGHPUT_RR_5P: self.throughput_rr_p5_mbps,
            KPIName.TOTAL_TX_POWER: self.total_tx_power_watt,
            KPIName.LOAD_BALANCE: self.load_balance,
            KPIName.LOAD_IMBALANCE: self.load_imbalance,
        }
        return mapping.get(kpi)


class ObjectiveCheck(StrictModel):
    kpi: KPIName
    operator: Operator
    target: float | list[float] | None
    predicted: float | None
    satisfied: bool | None
    normalized_violation: float
    explanation: str


class ObjectiveEvaluation(StrictModel):
    all_numeric_targets_met: bool
    score: float
    checks: list[ObjectiveCheck]


class OptimizationIteration(StrictModel):
    iteration: int
    proposal: CandidateProposal
    validation: ValidationResult
    prediction: KPIPrediction
    evaluation: ObjectiveEvaluation


class OptimizationResult(StrictModel):
    result_id: str
    created_at: datetime
    intent: ParsedIntent
    baseline_configuration: NetworkConfig
    baseline_prediction: KPIPrediction
    final_configuration: NetworkConfig
    predicted_kpis: KPIPrediction
    target_satisfied: bool
    best_score: float
    iterations: list[OptimizationIteration]
    hard_constraints: HardConstraints
    model_scenario: str


class ConflictType(str, Enum):
    PARAMETER_CONFLICT = "PARAMETER_CONFLICT"
    RESOURCE_CONTENTION = "RESOURCE_CONTENTION"
    BOOLEAN_CONFLICT = "BOOLEAN_CONFLICT"
    BASE_STATION_INTERACTION = "BASE_STATION_INTERACTION"
    KPI_DOMAIN_CONFLICT = "KPI_DOMAIN_CONFLICT"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConflictDetail(StrictModel):
    active_result_id: str
    new_result_id: str
    conflict_type: ConflictType
    severity: Severity
    parameter: str | None = None
    active_value: Any = None
    new_value: Any = None
    mergeable: bool
    explanation: str


class ConflictReport(StrictModel):
    conflict_detected: bool
    details: list[ConflictDetail] = Field(default_factory=list)
    recommended_strategy: Literal["NONE", "PRIORITY", "WEIGHTED_MERGE"]


class ResolutionStrategy(str, Enum):
    NONE = "NONE"
    PRIORITY = "PRIORITY"
    WEIGHTED_MERGE = "WEIGHTED_MERGE"




class ResolutionDecision(StrictModel):
    strategy: Literal["PRIORITY", "WEIGHTED_MERGE"]
    rationale: str


class ActiveIntentRecord(StrictModel):
    result_id: str
    created_at: datetime
    intent: ParsedIntent
    baseline_configuration: NetworkConfig
    proposed_configuration: NetworkConfig
    predicted_kpis: KPIPrediction
    hard_constraints: HardConstraints
    status: Literal["ACTIVE", "SUPPRESSED", "EXPIRED", "REJECTED"] = "ACTIVE"
    resolution_group_id: str | None = None


class ResolutionResult(StrictModel):
    meta_agent_id: str | None
    strategy: ResolutionStrategy
    final_configuration: NetworkConfig
    participant_result_ids: list[str]
    winner_result_id: str | None = None
    suppressed_result_ids: list[str] = Field(default_factory=list)
    rationale: str


class StrategicNarrative(StrictModel):
    summary: str
    optimization_explanation: str
    conflict_explanation: str
    resolution_explanation: str
    safety_notes: list[str]


class WorkflowResult(StrictModel):
    run_id: str
    created_at: datetime
    input_text: str
    scenario: str
    optimization: OptimizationResult
    conflict_report: ConflictReport
    resolution: ResolutionResult
    final_prediction: KPIPrediction
    reasoning: StrategicNarrative
    result_directory: str


class AgentHistoryEntry(StrictModel):
    step: str
    started_at: datetime
    finished_at: datetime
    status: Literal["success", "error"]
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None
