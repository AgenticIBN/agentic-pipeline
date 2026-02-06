#!/usr/bin/env python3
"""
optimization_agent_v2.py

Complete working optimization agent for telecom base-station configuration.
Uses surrogate models + local search to generate optimal configurations.

Key Features:
- SurrogateModel class for KPI prediction
- OptimizationAgent class for configuration optimization
- Local search with bounded parameter deltas
- Constraint satisfaction checking
- Multi-objective scoring
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from dotenv import load_dotenv

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

load_dotenv()


# ============================================================================
# SURROGATE MODEL CLASS
# ============================================================================

@dataclass
class SurrogateModel:
    """Surrogate model for predicting KPIs from base-station configuration."""
    
    feature_cols: List[str]
    target_cols: List[str]
    models: Dict[str, Any]
    param_ranges: Dict[str, Dict[str, float]]
    bw_hz: int
    has_throughput_col: bool
    throughput_col: Optional[str]
    
    @classmethod
    def load(cls, model_path: str) -> SurrogateModel:
        """Load trained surrogate model from file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        artifact = joblib.load(model_path)
        
        return cls(
            feature_cols=artifact["feature_cols"],
            target_cols=artifact["target_cols"],
            models=artifact["models"],
            param_ranges=artifact["param_ranges"],
            bw_hz=artifact.get("bw_hz", 10_000_000),
            has_throughput_col=artifact.get("has_throughput_col", False),
            throughput_col=artifact.get("throughput_col", None),
        )
    
    def predict(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict KPIs from configuration.
        
        Args:
            config: TX configuration dict with keys like tx0_on, tx0_P_dBm, etc.
            context: Context dict with user_set_id, K_users, rx_power_thr_dBm, etc.
        
        Returns:
            Dict of predicted KPIs: Prx_p5_dBm, SINR_p5_dB, rx_power_coverage_ratio,
            tx0_served_pct..tx3_served_pct, and optionally ThrRR_p5
        """
        # Build feature vector
        row = {}
        row["user_set_id"] = int(context.get("user_set_id", 0))
        row["K_users"] = int(context.get("K_users", 800))
        row["rx_power_thr_dBm"] = float(context.get("rx_power_thr_dBm", -95.0))
        row["total_tx_power_watt"] = float(context.get("total_tx_power_watt", 0.0))
        
        # TX parameters
        for i in range(4):
            row[f"tx{i}_on"] = 1 if bool(config.get(f"tx{i}_on", True)) else 0
            
            # If OFF, set parameters to 0
            if row[f"tx{i}_on"] == 0:
                row[f"tx{i}_P_dBm"] = 0.0
                row[f"tx{i}_dAz"] = 0.0
                row[f"tx{i}_dEl"] = 0.0
            else:
                row[f"tx{i}_P_dBm"] = float(config.get(f"tx{i}_P_dBm", 0.0))
                row[f"tx{i}_dAz"] = float(config.get(f"tx{i}_dAz", 0.0))
                row[f"tx{i}_dEl"] = float(config.get(f"tx{i}_dEl", 0.0))
        
        # Build feature array
        X = np.array([[float(row.get(c, 0.0)) for c in self.feature_cols]], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Predict all targets
        preds: Dict[str, float] = {}
        for target in self.target_cols:
            if target not in self.models:
                continue
            
            model = self.models[target]
            
            # Predict based on model type
            if HAS_LIGHTGBM and isinstance(model, lgb.Booster):
                pred = model.predict(X)[0]
            else:
                pred = model.predict(X)[0]
            
            preds[target] = float(pred)
        
        # Clip bounded values
        if "rx_power_coverage_ratio" in preds:
            preds["rx_power_coverage_ratio"] = float(np.clip(preds["rx_power_coverage_ratio"], 0.0, 1.0))
        
        for i in range(4):
            key = f"tx{i}_served_pct"
            if key in preds:
                preds[key] = float(np.clip(preds[key], 0.0, 100.0))
        
        return preds
    
    def compute_throughput_from_sinr(self, sinr_db: float) -> float:
        """Compute throughput from SINR using Shannon capacity formula."""
        sinr_linear = 10.0 ** (sinr_db / 10.0)
        throughput_bps = self.bw_hz * np.log2(1.0 + sinr_linear)
        throughput_mbps = throughput_bps / 1e6
        return float(throughput_mbps)
    
    def compute_load_imbalance(self, served_pcts: List[float]) -> float:
        """Compute load imbalance from served percentages."""
        return float(np.std(served_pcts))


# ============================================================================
# OPTIMIZATION AGENT CLASS
# ============================================================================

class OptimizationAgent:
    """Agent for optimizing base-station configuration using surrogate models."""
    
    def __init__(
        self,
        surrogate: SurrogateModel,
        power_deltas: List[float] = None,
        azimuth_deltas: List[float] = None,
        elevation_deltas: List[float] = None,
        search_iterations: int = 1000,
        random_seed: int = 42,
    ):
        """
        Initialize optimization agent.
        
        Args:
            surrogate: Trained SurrogateModel instance
            power_deltas: List of power change options in dB
            azimuth_deltas: List of azimuth change options in degrees
            elevation_deltas: List of elevation change options in degrees
            search_iterations: Number of random search iterations per seed
            random_seed: Random seed for reproducibility
        """
        self.surrogate = surrogate
        self.power_deltas = power_deltas or [-3, -2, -1, 0, 1, 2, 3]
        self.azimuth_deltas = azimuth_deltas or [-10, -5, -2, 0, 2, 5, 10]
        self.elevation_deltas = elevation_deltas or [-3, -2, -1, 0, 1, 2, 3]
        self.search_iterations = search_iterations
        self.rng = np.random.default_rng(random_seed)
    
    def optimize(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize configuration based on intent.
        
        Args:
            intent: Intent dict with keys:
                - target_area (str)
                - target_kpis (List[str]): e.g., ["RX_POWER", "SINR", "THROUGHPUT_5P"]
                - kpi_thresholds (List[dict]): Constraints like {"kpi":"RX_POWER","op":"GTE","value":-95.0}
                - priority (str): "LOW", "MEDIUM", "HIGH", "CRITICAL"
                - confidence (float): 0.0 to 1.0
                - current_config (dict): Current TX config
                - k_users (int): Number of users (e.g., 800)
                - user_set_id (int): User set identifier
        
        Returns:
            Result dict with test_name, passed, input, output fields
        """
        # Extract intent fields
        target_kpis = intent.get("target_kpis", [])
        kpi_thresholds = intent.get("kpi_thresholds", [])
        priority = intent.get("priority", "MEDIUM")
        current_config = intent.get("current_config", {})
        k_users = intent.get("k_users", 800)
        user_set_id = intent.get("user_set_id", 0)
        rx_power_thr = -95.0  # Default threshold
        
        # Extract RX power threshold if specified
        for thr in kpi_thresholds:
            if thr.get("kpi") == "RX_POWER" and "value" in thr:
                rx_power_thr = float(thr["value"])
                break
        
        # Context for predictions
        context = {
            "user_set_id": user_set_id,
            "K_users": k_users,
            "rx_power_thr_dBm": rx_power_thr,
            "total_tx_power_watt": 0.0,
        }
        
        # Compute current KPIs if current config exists
        current_kpis = None
        if current_config:
            current_kpis = self._compute_kpis(current_config, context)
        
        # Generate and evaluate candidate configurations
        best_config = None
        best_kpis = None
        best_score = -1e18
        
        # Use current config as seed if available, otherwise generate from ranges
        if current_config:
            seed_configs = [current_config]
        else:
            seed_configs = [self._generate_default_config()]
        
        # Add a few random seeds for diversity
        for _ in range(3):
            seed_configs.append(self._generate_random_config())
        
        # Search from each seed
        for seed_config in seed_configs:
            # Evaluate seed
            kpis = self._compute_kpis(seed_config, context)
            score = self._score_candidate(kpis, target_kpis, kpi_thresholds, priority, current_kpis)
            constraints_ok = self._check_constraints(kpis, kpi_thresholds, current_kpis)
            
            if constraints_ok and score > best_score:
                best_score = score
                best_config = dict(seed_config)
                best_kpis = kpis
            
            # Local search around seed
            for _ in range(self.search_iterations // len(seed_configs)):
                candidate = self._perturb_config(seed_config, current_config)
                kpis = self._compute_kpis(candidate, context)
                score = self._score_candidate(kpis, target_kpis, kpi_thresholds, priority, current_kpis)
                constraints_ok = self._check_constraints(kpis, kpi_thresholds, current_kpis)
                
                if constraints_ok and score > best_score:
                    best_score = score
                    best_config = dict(candidate)
                    best_kpis = kpis
        
        # If no valid config found, use best seed
        if best_config is None:
            best_config = seed_configs[0]
            best_kpis = self._compute_kpis(best_config, context)
        
        # Apply guardrails if current config exists
        if current_config:
            best_config = self._apply_guardrails(best_config, current_config)
            best_kpis = self._compute_kpis(best_config, context)
        
        # Build changes list
        changes = self._build_changes(best_config, current_config)
        
        # Check final constraint satisfaction
        constraints_satisfied = self._check_constraints(best_kpis, kpi_thresholds, current_kpis)
        
        # Build output
        output = {
            "selected_config_id": int(datetime.now().strftime("%Y%m%d%H%M%S")),
            "current_config_id": None,
            "changes": changes,
            "expected_kpis": {
                "RX_POWER": best_kpis.get("Prx_p5_dBm"),
                "SINR": best_kpis.get("SINR_p5_dB"),
                "THROUGHPUT_5P": best_kpis.get("THROUGHPUT_5P"),
                "LOAD_IMBALANCE": best_kpis.get("LOAD_IMBALANCE"),
                "RX_COVERAGE_RATIO": best_kpis.get("rx_power_coverage_ratio"),
            },
            "current_kpis": None if current_kpis is None else {
                "RX_POWER": current_kpis.get("Prx_p5_dBm"),
                "SINR": current_kpis.get("SINR_p5_dB"),
                "THROUGHPUT_5P": current_kpis.get("THROUGHPUT_5P"),
                "LOAD_IMBALANCE": current_kpis.get("LOAD_IMBALANCE"),
                "RX_COVERAGE_RATIO": current_kpis.get("rx_power_coverage_ratio"),
            },
            "constraints_satisfied": constraints_satisfied,
        }
        
        result = {
            "test_name": intent.get("test_name", "optimization_test"),
            "passed": constraints_satisfied,
            "input": intent,
            "output": output,
        }
        
        return result
    
    def _compute_kpis(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """Compute all KPIs for a configuration."""
        preds = self.surrogate.predict(config, context)
        
        # Add computed KPIs
        kpis = dict(preds)
        
        # Compute throughput if not already predicted
        if "SINR_p5_dB" in preds:
            if self.surrogate.has_throughput_col and self.surrogate.throughput_col in preds:
                kpis["THROUGHPUT_5P"] = preds[self.surrogate.throughput_col]
            else:
                kpis["THROUGHPUT_5P"] = self.surrogate.compute_throughput_from_sinr(preds["SINR_p5_dB"])
        
        # Compute load imbalance
        served_pcts = [preds.get(f"tx{i}_served_pct", 0.0) for i in range(4)]
        kpis["LOAD_IMBALANCE"] = self.surrogate.compute_load_imbalance(served_pcts)
        
        return kpis
    
    def _score_candidate(
        self,
        kpis: Dict[str, float],
        target_kpis: List[str],
        kpi_thresholds: List[Dict],
        priority: str,
        current_kpis: Optional[Dict[str, float]],
    ) -> float:
        """Score a candidate configuration. Higher is better."""
        priority_weight = {"LOW": 0.5, "MEDIUM": 1.0, "HIGH": 1.5, "CRITICAL": 2.0}.get(priority, 1.0)
        
        score = 0.0
        
        # Reward meeting thresholds with margin
        for thr in kpi_thresholds:
            kpi_name = thr.get("kpi")
            op = thr.get("op")
            value = thr.get("value")
            
            # Map KPI name to actual key in kpis dict
            kpi_key = self._map_kpi_name(kpi_name)
            if kpi_key not in kpis or value is None:
                continue
            
            kpi_value = kpis[kpi_key]
            
            if op == "GTE":
                margin = kpi_value - value
                score += priority_weight * max(0, margin)
            elif op == "GT":
                margin = kpi_value - value
                score += priority_weight * max(0, margin)
            elif op == "LTE":
                margin = value - kpi_value
                score += priority_weight * max(0, margin)
            elif op == "LT":
                margin = value - kpi_value
                score += priority_weight * max(0, margin)
        
        # Reward target KPIs
        if "RX_POWER" in target_kpis and "Prx_p5_dBm" in kpis:
            score += 0.1 * priority_weight * kpis["Prx_p5_dBm"]
        
        if "SINR" in target_kpis and "SINR_p5_dB" in kpis:
            score += 0.1 * priority_weight * kpis["SINR_p5_dB"]
        
        if "THROUGHPUT_5P" in target_kpis and "THROUGHPUT_5P" in kpis:
            score += 0.01 * priority_weight * kpis["THROUGHPUT_5P"]
        
        if "SERVED_USERS" in target_kpis and "LOAD_IMBALANCE" in kpis:
            # Lower imbalance is better
            score += 0.5 * priority_weight * (-kpis["LOAD_IMBALANCE"])
        
        # Penalize degradation from current if available
        if current_kpis:
            if "SINR_p5_dB" in kpis and "SINR_p5_dB" in current_kpis:
                sinr_change = kpis["SINR_p5_dB"] - current_kpis["SINR_p5_dB"]
                if sinr_change < -2.0:  # Penalize >2dB SINR loss
                    score -= 5.0 * priority_weight * abs(sinr_change)
        
        return score
    
    def _check_constraints(
        self,
        kpis: Dict[str, float],
        kpi_thresholds: List[Dict],
        current_kpis: Optional[Dict[str, float]],
    ) -> bool:
        """Check if all constraints are satisfied."""
        for thr in kpi_thresholds:
            kpi_name = thr.get("kpi")
            op = thr.get("op")
            value = thr.get("value")
            delta = thr.get("delta")
            
            kpi_key = self._map_kpi_name(kpi_name)
            if kpi_key not in kpis:
                continue
            
            kpi_value = kpis[kpi_key]
            
            if op == "GTE" and value is not None:
                if kpi_value < value:
                    return False
            elif op == "GT" and value is not None:
                if kpi_value <= value:
                    return False
            elif op == "LTE" and value is not None:
                if kpi_value > value:
                    return False
            elif op == "LT" and value is not None:
                if kpi_value >= value:
                    return False
            elif op == "BETWEEN":
                value_low = thr.get("value_low")
                value_high = thr.get("value_high")
                if value_low is not None and value_high is not None:
                    if not (value_low <= kpi_value <= value_high):
                        return False
            elif op == "DELTA_UP" and current_kpis and delta is not None:
                if kpi_key in current_kpis:
                    if (kpi_value - current_kpis[kpi_key]) < delta:
                        return False
            elif op == "DELTA_DOWN" and current_kpis and delta is not None:
                if kpi_key in current_kpis:
                    if (current_kpis[kpi_key] - kpi_value) < delta:
                        return False
        
        return True
    
    def _map_kpi_name(self, kpi_name: str) -> str:
        """Map high-level KPI name to actual dictionary key."""
        mapping = {
            "RX_POWER": "Prx_p5_dBm",
            "SINR": "SINR_p5_dB",
            "THROUGHPUT_5P": "THROUGHPUT_5P",
            "RX_COVERAGE_RATIO": "rx_power_coverage_ratio",
            "SERVED_USERS": "LOAD_IMBALANCE",  # Special case
        }
        return mapping.get(kpi_name, kpi_name)
    
    def _generate_default_config(self) -> Dict[str, Any]:
        """Generate a reasonable default configuration from parameter ranges."""
        pr = self.surrogate.param_ranges
        
        # Use mid-range values
        mid_power = (pr["power"]["min"] + pr["power"]["max"]) / 2.0
        mid_az = (pr["dAz"]["min"] + pr["dAz"]["max"]) / 2.0
        mid_el = (pr["dEl"]["min"] + pr["dEl"]["max"]) / 2.0
        
        config = {}
        for i in range(4):
            config[f"tx{i}_on"] = True
            config[f"tx{i}_P_dBm"] = mid_power
            config[f"tx{i}_dAz"] = mid_az
            config[f"tx{i}_dEl"] = mid_el
        
        return config
    
    def _generate_random_config(self) -> Dict[str, Any]:
        """Generate a random configuration within parameter ranges."""
        pr = self.surrogate.param_ranges
        
        config = {}
        for i in range(4):
            config[f"tx{i}_on"] = self.rng.random() > 0.1  # 90% chance ON
            
            if config[f"tx{i}_on"]:
                config[f"tx{i}_P_dBm"] = float(self.rng.uniform(pr["power"]["min"], pr["power"]["max"]))
                config[f"tx{i}_dAz"] = float(self.rng.uniform(pr["dAz"]["min"], pr["dAz"]["max"]))
                config[f"tx{i}_dEl"] = float(self.rng.uniform(pr["dEl"]["min"], pr["dEl"]["max"]))
            else:
                config[f"tx{i}_P_dBm"] = 0.0
                config[f"tx{i}_dAz"] = 0.0
                config[f"tx{i}_dEl"] = 0.0
        
        return config
    
    def _perturb_config(
        self,
        config: Dict[str, Any],
        current_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Apply random perturbations to a configuration."""
        pr = self.surrogate.param_ranges
        new_config = dict(config)
        
        for i in range(4):
            if not bool(new_config.get(f"tx{i}_on", True)):
                continue
            
            # Random deltas
            power_delta = float(self.rng.choice(self.power_deltas))
            az_delta = float(self.rng.choice(self.azimuth_deltas))
            el_delta = float(self.rng.choice(self.elevation_deltas))
            
            # Apply and clip
            new_config[f"tx{i}_P_dBm"] = float(np.clip(
                config[f"tx{i}_P_dBm"] + power_delta,
                pr["power"]["min"],
                pr["power"]["max"],
            ))
            new_config[f"tx{i}_dAz"] = float(np.clip(
                config[f"tx{i}_dAz"] + az_delta,
                pr["dAz"]["min"],
                pr["dAz"]["max"],
            ))
            new_config[f"tx{i}_dEl"] = float(np.clip(
                config[f"tx{i}_dEl"] + el_delta,
                pr["dEl"]["min"],
                pr["dEl"]["max"],
            ))
        
        return new_config
    
    def _apply_guardrails(
        self,
        new_config: Dict[str, Any],
        current_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply guardrails to limit changes from current config."""
        guarded_config = dict(new_config)
        
        for i in range(4):
            if not bool(current_config.get(f"tx{i}_on", True)):
                continue
            if not bool(new_config.get(f"tx{i}_on", True)):
                continue
            
            # Power: ±3 dB max
            current_power = float(current_config[f"tx{i}_P_dBm"])
            new_power = float(new_config[f"tx{i}_P_dBm"])
            power_change = new_power - current_power
            if abs(power_change) > 3.0:
                guarded_config[f"tx{i}_P_dBm"] = current_power + np.sign(power_change) * 3.0
            
            # Tilt: ±2° max
            current_el = float(current_config[f"tx{i}_dEl"])
            new_el = float(new_config[f"tx{i}_dEl"])
            el_change = new_el - current_el
            if abs(el_change) > 2.0:
                guarded_config[f"tx{i}_dEl"] = current_el + np.sign(el_change) * 2.0
            
            # Azimuth: ±10° max
            current_az = float(current_config[f"tx{i}_dAz"])
            new_az = float(new_config[f"tx{i}_dAz"])
            az_change = new_az - current_az
            if abs(az_change) > 10.0:
                guarded_config[f"tx{i}_dAz"] = current_az + np.sign(az_change) * 10.0
        
        return guarded_config
    
    def _build_changes(
        self,
        new_config: Dict[str, Any],
        current_config: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build list of parameter changes."""
        changes = []
        
        if current_config is None:
            # No current config: all params are "new"
            for i in range(4):
                changes.append({
                    "param": f"tx{i}_on",
                    "before": None,
                    "change": new_config[f"tx{i}_on"],
                    "unit": None,
                })
                changes.append({
                    "param": f"tx{i}_P_dBm",
                    "before": None,
                    "change": new_config[f"tx{i}_P_dBm"],
                    "unit": "dBm",
                })
                changes.append({
                    "param": f"tx{i}_dAz",
                    "before": None,
                    "change": new_config[f"tx{i}_dAz"],
                    "unit": "deg",
                })
                changes.append({
                    "param": f"tx{i}_dEl",
                    "before": None,
                    "change": new_config[f"tx{i}_dEl"],
                    "unit": "deg",
                })
        else:
            # With current config: show before and delta
            for i in range(4):
                for param_type in ["on", "P_dBm", "dAz", "dEl"]:
                    param_name = f"tx{i}_{param_type}"
                    before = current_config[param_name]
                    after = new_config[param_name]
                    
                    if before != after:
                        if param_type == "on":
                            # Boolean: show new state
                            changes.append({
                                "param": param_name,
                                "before": before,
                                "change": after,
                                "unit": None,
                            })
                        else:
                            # Numeric: show delta
                            delta = float(after) - float(before)
                            unit = "dBm" if param_type == "P_dBm" else "deg"
                            changes.append({
                                "param": param_name,
                                "before": before,
                                "change": delta,
                                "unit": unit,
                            })
        
        return changes


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

def main():
    """Example usage of optimization agent."""
    
    # Load trained surrogate model
    model_path = os.getenv("MODEL_PATH", "./models/surrogate.joblib")
    print(f"Loading surrogate model from: {model_path}")
    
    surrogate = SurrogateModel.load(model_path)
    print(f"Loaded model with {len(surrogate.models)} KPI predictors")
    print(f"Parameter ranges: {surrogate.param_ranges}")
    
    # Initialize optimization agent
    agent = OptimizationAgent(
        surrogate=surrogate,
        search_iterations=1000,
        random_seed=42,
    )
    print("\nOptimization agent initialized")
    
    # Sample intent
    sample_intent = {
        "test_name": "example_optimization",
        "target_area": "sector_A",
        "target_kpis": ["RX_POWER", "THROUGHPUT_5P"],
        "kpi_thresholds": [
            {
                "kpi": "RX_POWER",
                "op": "GTE",
                "value": -95.0,
                "unit": "dBm",
            },
        ],
        "priority": "HIGH",
        "confidence": 0.95,
        "current_config": {
            "tx0_on": True,
            "tx0_P_dBm": 43.0,
            "tx0_dAz": 0.0,
            "tx0_dEl": 0.0,
            "tx1_on": True,
            "tx1_P_dBm": 43.0,
            "tx1_dAz": 120.0,
            "tx1_dEl": 0.0,
            "tx2_on": True,
            "tx2_P_dBm": 43.0,
            "tx2_dAz": -120.0,
            "tx2_dEl": 0.0,
            "tx3_on": False,
            "tx3_P_dBm": 0.0,
            "tx3_dAz": 0.0,
            "tx3_dEl": 0.0,
        },
        "k_users": 800,
        "user_set_id": 0,
    }
    
    print("\n" + "="*80)
    print("Running optimization with sample intent:")
    print(json.dumps(sample_intent, indent=2))
    print("="*80 + "\n")
    
    # Run optimization
    result = agent.optimize(sample_intent)
    
    # Print result
    print("\n" + "="*80)
    print("OPTIMIZATION RESULT:")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80)
    
    # Summary
    output = result["output"]
    print(f"\n✓ Optimization complete!")
    print(f"  Constraints satisfied: {output['constraints_satisfied']}")
    print(f"  Number of changes: {len(output['changes'])}")
    print(f"  Expected RX_POWER: {output['expected_kpis']['RX_POWER']:.2f} dBm")
    print(f"  Expected SINR: {output['expected_kpis']['SINR']:.2f} dB")
    print(f"  Expected THROUGHPUT_5P: {output['expected_kpis']['THROUGHPUT_5P']:.2f} Mbps")
    print(f"  Expected LOAD_IMBALANCE: {output['expected_kpis']['LOAD_IMBALANCE']:.2f}")
    

if __name__ == "__main__":
    main()
