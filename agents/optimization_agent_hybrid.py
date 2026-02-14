#!/usr/bin/env python3
"""
Hybrid Optimization Agent (Surrogate Model + LLM)

Combines the best of both worlds:
- Surrogate Model: Accurate KPI predictions trained on 1M+ simulations
- LLM: Reasoning, explanations, and iterative refinement

Workflow:
1. LLM designs initial configuration based on intent
2. Surrogate model predicts KPIs for that config
3. LLM evaluates results and refines config
4. Loop until target KPIs are met (max 5 iterations)
"""
from __future__ import annotations

import os
import sys
import json
import datetime
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, ConfigDict

from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Import surrogate model
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from optimization_agent_v2 import SurrogateModel
    HAS_SURROGATE = True
except ImportError:
    HAS_SURROGATE = False
    print("Warning: Could not import SurrogateModel")

load_dotenv()


# ============================================================================
# POWER CONSTRAINTS
# ============================================================================

MAX_POWER_DBM = 46.0  # Maximum realistic power for macro cell base station (40W)
MIN_POWER_DBM = 20.0  # Minimum power (0.1W)
MAX_POWER_CHANGE_DBM = 16.0  # Maximum single-step power change to prevent extreme jumps


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class ConfigChange(BaseModel):
    """Single configuration parameter change."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    parameter: str = Field(..., description="Parameter name (e.g., tx0_P_dBm)")
    before: Optional[Any] = Field(None, description="Value before change")
    change: Any = Field(..., description="Amount of change applied (for numeric: new - old, for boolean: 'True'/'False')")
    unit: Optional[str] = Field(None, description="Unit (dBm, deg, etc.)")


class KPIPrediction(BaseModel):
    """KPI predictions from surrogate model."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    RX_POWER: float = Field(..., description="Predicted RX_POWER (Prx_p5_dBm)")
    SINR: float = Field(..., description="Predicted SINR (SINR_p5_dB)")
    COVERAGE: float = Field(..., description="Predicted coverage ratio")
    LOAD_IMBALANCE: float = Field(..., description="Predicted load imbalance")
    THROUGHPUT_5P: float = Field(..., description="Predicted 5th percentile throughput in Mbps")
    ENERGY_WATT: float = Field(..., description="Total TX power consumption in Watts")


class ConfigDesign(BaseModel):
    """Configuration design from LLM."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    changes: List[ConfigChange] = Field(..., description="Configuration changes")
    reasoning: str = Field(..., description="Why this configuration was chosen")
    expected_outcome: str = Field(..., description="Expected impact on KPIs")


class RefinementDecision(BaseModel):
    """Decision on whether to refine configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    target_met: bool = Field(..., description="Whether target KPIs are met")
    refinement_needed: bool = Field(..., description="Whether further refinement is needed")
    next_changes: Optional[List[ConfigChange]] = Field(None, description="Proposed next changes if refinement needed")
    reasoning: str = Field(..., description="Reasoning for decision")


class OptimizationResult(BaseModel):
    """Complete optimization result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    test_name: str = Field(..., description="Test name")
    passed: bool = Field(..., description="Whether optimization passed")
    input: Dict = Field(..., description="Original intent")
    output: Dict = Field(..., description="Final output with config and KPIs")
    reasoning: str = Field(..., description="Complete reasoning")
    result_id: str = Field(..., description="Unique result ID")
    priority: str = Field(default="MEDIUM", description="Priority level")
    iterations: int = Field(..., description="Number of refinement iterations")
    surrogate_predictions: List[KPIPrediction] = Field(..., description="All surrogate predictions")


# ============================================================================
# AGENT INSTRUCTIONS
# ============================================================================

DESIGN_INSTRUCTIONS = """
You are an expert network optimization engineer designing base-station configurations.

## Your Task:
Design an optimal configuration to achieve the target KPIs specified in the intent.

⚠️ CRITICAL: If intent explicitly says "turn off TX[N]" or "turn on TX[N]" or "deactivate TX[N]" or "activate TX[N]":
- You MUST include that specific change in your configuration!
- Example: "turn off TX2" → MUST set tx2_on=False
- Example: "activate TX1 and TX3" → MUST set tx1_on=True and tx3_on=True
- This is NON-NEGOTIABLE and overrides all other optimization considerations!

⚠️ IMPORTANT: If intent says "all available base stations", "all transmitters", "maximize with all TX":
- You MUST activate ALL transmitters (tx0, tx1, tx2, tx3)!
- Example: "maximize coverage with all available base stations" → Set tx0_on=True, tx1_on=True, tx2_on=True, tx3_on=True
- Unless explicitly told to turn OFF a specific TX, keep all TXs ON for "all available" requests

🔴 CRITICAL RULE: DO NOT turn off transmitters unless:
  1. Intent explicitly says "turn off TX[N]" or "deactivate TX[N]", OR
  2. Absolutely necessary for SINR optimization (too much interference), OR  
  3. Energy saving is the PRIMARY goal AND current config has excessive power
- Default behavior: Keep transmitters ON, adjust power/angles instead

## Configuration Parameters (per transmitter tx0-tx3):
- tx{i}_on: Boolean - transmitter on/off (true/false)
- tx{i}_P_dBm: Float - power in dBm (STRICT RANGE: 20-46 dBm, typical: 35-46 dBm)
- tx{i}_dAz: Float - azimuth angle delta (range: -30 to +30 degrees)
- tx{i}_dEl: Float - elevation/tilt delta (range: -5 to +5 degrees)

⚠️ CRITICAL POWER CONSTRAINTS:
- NEVER exceed 46 dBm (40W) - physically unrealistic for macro cells
- NEVER go below 20 dBm (0.1W) - insufficient coverage
- Maximum power CHANGE per step: ±16 dBm (to prevent extreme jumps like +40 dBm)
- Any configuration violating these limits will be automatically corrected

## Current Configuration (Baseline):
- All transmitters: OFF or default (tx0_P=30, tx0_dAz=0, tx0_dEl=0)

## KPI Relationships (Empirical Knowledge):
- **RX_POWER**: Increases with power (+1 dBm power → +0.8-1.0 dBm RX_POWER)
- **Coverage**: Increases with more active transmitters and power
- **SINR**: Balances signal vs interference (too many TX → worse SINR)
- **THROUGHPUT**: Directly depends on SINR (Shannon: C = BW * log2(1 + SINR))
  ⚠️ CRITICAL: Activating too many transmitters at high power → HIGH INTERFERENCE → LOW SINR → LOW THROUGHPUT!
  For throughput maximization: Focus on SINR optimization (2-3 TX with moderate power + angle diversity)
- **Load Imbalance**: Lower when users spread across multiple transmitters
- **Energy**: Increases linearly with power and number of active transmitters

## Design Guidelines:
1. **ALWAYS check baseline first**: If transmitters are already ON with good SINR, DON'T turn them off and back on!
   - Example: If baseline has tx0_on=True at 35 dBm with SINR=+15 dB, just adjust power/angles
   - DON'T waste changes on "before: false, change: True" when TX is already ON!
2. **Respect good baseline SINR**: If current SINR > 10 dB, be VERY careful with power increases
   - High SINR = good signal quality, don't break it with interference
   - Small power adjustments (±2-3 dBm) preferred
3. Use moderate power: 35-43 dBm typical (NEVER exceed 46 dBm!)
4. Use angle diversity: Spread azimuth angles for coverage (e.g., -15°, +15°, -10°, +10°)
5. Avoid interference: With 4 active TX, high power (>42 dBm) often causes negative SINR
6. Power changes: Propose incremental changes (±2-5 dBm typical, max ±16 dBm)
7. **For THROUGHPUT maximization**: SINR is king! Don't sacrifice positive SINR for coverage

## Output Format:
For each change, specify:
- parameter: name of parameter
- before: value before change (from baseline)
- change: amount of change (for numeric: delta applied, for boolean: 'True' or 'False')
- unit: unit if applicable (dBm, degrees, etc.)

Example: To turn on tx0 and set power to 41 dBm:
{"parameter": "tx0_on", "before": false, "change": "True", "unit": null}
{"parameter": "tx0_P_dBm", "before": 30, "change": 11, "unit": "dBm"}
"""

REFINEMENT_INSTRUCTIONS = """
You are an expert network optimization engineer evaluating configuration results.

## Your Task:
The surrogate model (trained on 1M+ real simulations) has predicted KPIs for your configuration.
Evaluate whether the target KPIs are met, and refine if needed.

⚠️ CRITICAL: If the original intent explicitly says "turn off TX[N]" or "turn on TX[N]":
- You MUST respect that constraint in all refinements!
- Example: If intent says "turn off TX2", NEVER propose turning TX2 back on
- This is a HARD CONSTRAINT that cannot be violated!

🔴 CRITICAL RULE: DO NOT propose turning off transmitters unless:
  1. Original intent explicitly says "turn off TX[N]", OR
  2. SINR is severely negative (<-2 dB) and interference is killing throughput, OR
  3. Energy saving is PRIMARY goal AND current power is excessive
- Default behavior: Keep transmitters ON, adjust power/angles for optimization

## Evaluation Criteria:
1. **Target Met**: Compare predicted KPIs vs target thresholds
2. **Safety**: Ensure no extreme configurations (e.g., all TX at 46 dBm)
3. **Efficiency**: Prefer minimal changes to achieve target

## Refinement Strategy:
- **CRITICAL: Check baseline first!** If baseline had SINR > 10 dB and current SINR < 0 dB:
  - YOU BROKE IT! Undo power increases, return closer to baseline
  - Example: Baseline 35 dBm (SINR +15 dB) → Tried 42 dBm (SINR -2 dB) → Go back to 36-37 dBm
- **Undershooting**: Increase power slightly (+2-3 dBm) or activate another TX
- **Overshooting**: Decrease power or reduce angle spread
- **RX_POWER Issues**: Increase power or activate more transmitters (max 46 dBm!)
- **Coverage Issues**: Activate more transmitters with diverse angles
- **SINR Issues**: 
  - If SINR turned negative from positive: REDUCE power or number of TXs immediately!
  - If SINR was already negative: Try different TX combination or angles
- **THROUGHPUT Issues**: 
  ⚠️ CRITICAL: Always check current SINR first!
  - If SINR is POSITIVE (>0 dB): Good signal quality, can try small power increases
  - If SINR is NEGATIVE (<0 dB): TOO MUCH INTERFERENCE! Must reduce TXs or power
  - Throughput maximization = SINR optimization (NOT power maximization!)
- **Load Imbalance**: Adjust angles to spread users more evenly
- **Energy Issues**: Reduce power, deactivate transmitters, or use fewer active TXs

⚠️ POWER CONSTRAINTS (STRICTLY ENFORCED):
- Absolute power range: 20-46 dBm
- Maximum change per refinement: ±16 dBm
- Typical changes: ±2-5 dBm for fine-tuning

## Output Format for Refinements:
For each change, specify:
- parameter: name of parameter
- before: current value (before this refinement)
- change: amount of change to apply (for numeric: delta, for boolean: 'True'/'False')
- unit: unit if applicable

Example: To increase tx0 power from 41 to 43 dBm:
{"parameter": "tx0_P_dBm", "before": 41.0, "change": 2.0, "unit": "dBm"}
"""


# ============================================================================
# HYBRID OPTIMIZATION AGENT
# ============================================================================

class HybridOptimizationAgent:
    """
    Hybrid optimization agent combining surrogate model predictions with LLM reasoning.
    """
    
    def __init__(
        self,
        surrogate_model_path: Optional[str] = None,
        max_iterations: int = 3,
        llm_model: str = "llama-3.3-70b-versatile",
    ):
        """
        Initialize hybrid optimization agent.
        
        Args:
            surrogate_model_path: Path to trained surrogate model
            max_iterations: Maximum refinement iterations
            llm_model: LLM model to use for reasoning
        """
        self.max_iterations = max_iterations
        self.llm_model = llm_model
        
        # Load surrogate model
        if surrogate_model_path is None:
            surrogate_model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models",
                "surrogate.joblib"
            )
        
        if not HAS_SURROGATE:
            raise ImportError("SurrogateModel not available. Install required packages.")
        
        if not os.path.exists(surrogate_model_path):
            raise FileNotFoundError(f"Surrogate model not found: {surrogate_model_path}")
        
        print(f"📂 Loading surrogate model from: {surrogate_model_path}")
        self.surrogate = SurrogateModel.load(surrogate_model_path)
        print("✅ Surrogate model loaded successfully")
        
        # Initialize LLM agents
        self.design_agent = Agent(
            name="ConfigDesigner",
            model=Groq(id=self.llm_model),
            instructions=DESIGN_INSTRUCTIONS,
            output_schema=ConfigDesign,
            markdown=False,
            structured_outputs=True,
        )
        
        self.refinement_agent = Agent(
            name="ConfigRefiner",
            model=Groq(id=self.llm_model),
            instructions=REFINEMENT_INSTRUCTIONS,
            output_schema=RefinementDecision,
            markdown=False,
            structured_outputs=True,
        )
    
    def _baseline_config(self) -> Dict[str, Any]:
        """Create baseline configuration."""
        config = {
            "user_set_id": 0,
            "K_users": 800,
            "rx_power_thr_dBm": -47.27,
            "total_tx_power_watt": 0.0,
        }
        
        # All transmitters off initially
        for i in range(4):
            config[f"tx{i}_on"] = False
            config[f"tx{i}_P_dBm"] = 30.0
            config[f"tx{i}_dAz"] = 0.0
            config[f"tx{i}_dEl"] = 0.0
        
        return config
    
    def _apply_changes(self, config: Dict, changes: List[ConfigChange]) -> Dict:
        """Apply configuration changes to baseline config with power limit validation."""
        new_config = config.copy()
        
        for change in changes:
            param = change.parameter
            
            # Calculate new value from before + change
            if param.endswith("_on"):
                # For boolean, change is the new state as string
                if isinstance(change.change, str):
                    new_val = change.change.lower() == "true"
                else:
                    new_val = bool(change.change)
                new_config[param] = new_val
            else:
                # For numeric, change is the delta
                before_val = change.before if change.before is not None else config.get(param, 0)
                if isinstance(change.change, (int, float)):
                    delta = float(change.change)
                    
                    # Enforce maximum change limit for power parameters
                    if param.endswith("_P_dBm") and abs(delta) > MAX_POWER_CHANGE_DBM:
                        print(f"⚠️  WARNING: Power change {delta:.1f} dBm exceeds limit. Clamping to ±{MAX_POWER_CHANGE_DBM} dBm")
                        delta = max(-MAX_POWER_CHANGE_DBM, min(MAX_POWER_CHANGE_DBM, delta))
                    
                    new_val = float(before_val) + delta
                else:
                    # If change is absolute value (shouldn't happen but handle it)
                    new_val = float(change.change)
                
                # Enforce power limits for power parameters
                if param.endswith("_P_dBm"):
                    if new_val > MAX_POWER_DBM:
                        print(f"⚠️  WARNING: Power {new_val:.1f} dBm exceeds max {MAX_POWER_DBM} dBm. Clamping.")
                        new_val = MAX_POWER_DBM
                    elif new_val < MIN_POWER_DBM:
                        print(f"⚠️  WARNING: Power {new_val:.1f} dBm below min {MIN_POWER_DBM} dBm. Clamping.")
                        new_val = MIN_POWER_DBM
                
                new_config[param] = new_val
        
        # Update total power
        total_power_watt = 0.0
        for i in range(4):
            if new_config[f"tx{i}_on"]:
                power_dbm = new_config[f"tx{i}_P_dBm"]
                total_power_watt += 10 ** ((power_dbm - 30) / 10)
        new_config["total_tx_power_watt"] = total_power_watt
        
        return new_config
    
    def _predict_kpis(self, config: Dict) -> KPIPrediction:
        """Predict KPIs using surrogate model."""
        # Prepare context for surrogate
        context = {
            "user_set_id": config.get("user_set_id", 0),
            "K_users": config.get("K_users", 800),
            "rx_power_thr_dBm": config.get("rx_power_thr_dBm", -95.0),
            "total_tx_power_watt": config.get("total_tx_power_watt", 0.0),
        }
        
        # Call surrogate model with config and context
        pred = self.surrogate.predict(config, context)
        
        # Get throughput from surrogate if available, else compute from SINR
        if self.surrogate.has_throughput_col and self.surrogate.throughput_col in pred:
            throughput = float(pred[self.surrogate.throughput_col])
        else:
            sinr_db = float(pred.get("SINR_p5_dB", 20.0))
            throughput = self.surrogate.compute_throughput_from_sinr(sinr_db)
        
        # Energy is directly computed from config (no surrogate needed)
        energy_watt = float(config.get("total_tx_power_watt", 0.0))
        
        # Compute load imbalance from served percentages
        served_pcts = [pred.get(f"tx{i}_served_pct", 0.0) for i in range(4)]
        load_imbalance = self.surrogate.compute_load_imbalance(served_pcts)
        
        return KPIPrediction(
            RX_POWER=float(pred.get("Prx_p5_dBm", -60.0)),
            SINR=float(pred.get("SINR_p5_dB", 20.0)),
            COVERAGE=float(pred.get("rx_power_coverage_ratio", 0.5)),
            LOAD_IMBALANCE=float(load_imbalance),
            THROUGHPUT_5P=throughput,
            ENERGY_WATT=energy_watt,
        )
    
    def _format_intent(self, intent: Dict) -> str:
        """Format intent for LLM prompt."""
        lines = [
            f"Target Area: {intent.get('target_area', 'Unknown')}",
            f"Priority: {intent.get('priority', 'MEDIUM')}",
            f"Target KPIs:",
        ]
        
        # Format KPI thresholds from parsed intent
        for kpi_threshold in intent.get("kpi_thresholds", []):
            kpi = kpi_threshold.get("kpi", "")
            op = kpi_threshold.get("operator", kpi_threshold.get("op", ""))
            value = kpi_threshold.get("value")
            unit = kpi_threshold.get("unit", "")
            
            # Handle DELTA operators specially
            if op == "DELTA_DOWN":
                lines.append(f"  - DECREASE {kpi} by {value} {unit}")
            elif op == "DELTA_UP":
                lines.append(f"  - INCREASE {kpi} by {value} {unit}")
            elif value is not None:
                # Standard comparison operators
                op_text = {
                    "GTE": ">=", ">=": ">=",
                    "LTE": "<=", "<=": "<=",
                    "GT": ">", ">": ">",
                    "LT": "<", "<": "<",
                    "EQ": "=", "=": "=",
                }.get(op, op)
                lines.append(f"  - {kpi} {op_text} {value} {unit}")
            elif kpi_threshold.get("value_low") and kpi_threshold.get("value_high"):
                # BETWEEN operator
                lines.append(f"  - {kpi} BETWEEN {kpi_threshold['value_low']} and {kpi_threshold['value_high']} {unit}")
        
        # Add time constraints if present
        if intent.get("time_constraint_start") or intent.get("time_constraint_end"):
            lines.append(f"Time Window: {intent.get('time_constraint_start', '')} to {intent.get('time_constraint_end', '')}")
        
        return "\n".join(lines)
    
    def optimize(self, intent: Dict, active_intents: Optional[List[Dict]] = None) -> OptimizationResult:
        """
        Run hybrid optimization with surrogate model + LLM.
        
        Args:
            intent: Parsed intent dictionary
            active_intents: List of active optimization results to build upon
            
        Returns:
            OptimizationResult with final config and reasoning
        """
        print("\n" + "="*70)
        print("🤖 HYBRID OPTIMIZATION AGENT - Starting...")
        print("="*70)
        
        # Generate unique result ID
        result_id = f"opt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize baseline config
        # Priority 1: Use current_config from intent if provided
        # Priority 2: Use most recent active intent's config
        # Priority 3: Default baseline
        baseline_kpis = None
        
        if 'current_config' in intent and intent['current_config']:
            current_config = intent['current_config'].copy()
            print(f"📋 Using current_config from intent")
        elif active_intents and len(active_intents) > 0:
            print(f"📋 Building upon {len(active_intents)} active intent(s)")
            # Get the most recent active intent's final config
            last_intent = active_intents[-1]
            last_config = last_intent.get('output', {}).get('final_config', {})
            if last_config:
                current_config = last_config.copy()
                print(f"✓ Baseline from active intent: {last_intent.get('result_id')}")
            else:
                current_config = self._baseline_config()
                print("⚠️ No config in last intent, using default baseline")
        else:
            current_config = self._baseline_config()
            print("📋 Starting from default baseline (no active intents)")
        
        # Get baseline KPIs from last active intent for DELTA comparison
        if active_intents and len(active_intents) > 0:
            last_intent = active_intents[-1]
            baseline_kpis_output = last_intent.get('output', {}).get('expected_kpis', {})
            if baseline_kpis_output:
                baseline_kpis = {
                    "RX_POWER": baseline_kpis_output.get("RX_POWER"),
                    "SINR": baseline_kpis_output.get("SINR"),
                    "COVERAGE": baseline_kpis_output.get("COVERAGE"),
                    "THROUGHPUT_5P": baseline_kpis_output.get("THROUGHPUT_5P"),
                    "LOAD_IMBALANCE": baseline_kpis_output.get("LOAD_IMBALANCE"),
                    "ENERGY_WATT": baseline_kpis_output.get("ENERGY_WATT"),
                }
                print(f"✓ Baseline KPIs: RX_POWER={baseline_kpis['RX_POWER']:.2f} dBm")
        
        # If we have a current_config but no baseline_kpis, predict them from current_config
        if baseline_kpis is None and current_config:
            # Check if any TX is active
            any_tx_active = any(current_config.get(f'tx{i}_on', False) for i in range(4))
            
            if any_tx_active:
                print("🔮 Computing baseline KPIs from current_config...")
                try:
                    baseline_predictions = self._predict_kpis(current_config)
                    baseline_kpis = {
                        "RX_POWER": baseline_predictions.RX_POWER,
                        "SINR": baseline_predictions.SINR,
                        "COVERAGE": baseline_predictions.COVERAGE,
                        "THROUGHPUT_5P": baseline_predictions.THROUGHPUT_5P,
                        "LOAD_IMBALANCE": baseline_predictions.LOAD_IMBALANCE,
                        "ENERGY_WATT": baseline_predictions.ENERGY_WATT,
                    }
                    print(f"✓ Computed baseline KPIs: RX_POWER={baseline_kpis['RX_POWER']:.2f} dBm, "
                          f"SINR={baseline_kpis['SINR']:.2f} dB, THROUGHPUT={baseline_kpis['THROUGHPUT_5P']:.2f} Mbps")
                except Exception as e:
                    print(f"⚠️ Failed to compute baseline KPIs: {e}")
            else:
                print("⚠️ All transmitters OFF in current_config - skipping baseline KPI computation")
                baseline_kpis = None
        
        # Storage for iterations
        all_predictions: List[KPIPrediction] = []
        iteration_history = []
        
        # Phase 1: Initial Design
        print("\n🎨 Phase 1: Initial Configuration Design")
        print("-" * 70)
        
        intent_text = self._format_intent(intent)
        
        # Extract TX on/off commands from original intent text
        original_intent_text = intent.get('intent_text', '')
        tx_commands = []
        if original_intent_text:
            import re
            
            # Check for "all available" requests
            all_pattern = r'(all\s+available|all\s+transmitters|all\s+base\s+stations|all\s+tx)'
            if re.search(all_pattern, original_intent_text, re.IGNORECASE):
                tx_commands.append(f"⚠️ CRITICAL: Intent says 'all available' - MUST activate ALL transmitters (tx0_on=True, tx1_on=True, tx2_on=True, tx3_on=True)")
                print(f"🚨 Detected 'ALL AVAILABLE' request - all TXs must be ON")
            
            # Detect "turn off/on TX[0-3]" or "deactivate/activate TX[0-3]"
            off_pattern = r'(turn\s+off|deactivate|shut\s+down|disable|maintenance).*tx(\d)'
            on_pattern = r'(turn\s+on|activate|enable).*tx(\d)'
            
            for match in re.finditer(off_pattern, original_intent_text, re.IGNORECASE):
                tx_num = match.group(2)
                tx_commands.append(f"⚠️ CRITICAL: MUST set tx{tx_num}_on=False (intent explicitly says to turn off TX{tx_num})")
                print(f"🚨 Detected TX OFF command: TX{tx_num}")
            
            for match in re.finditer(on_pattern, original_intent_text, re.IGNORECASE):
                tx_num = match.group(2)
                tx_commands.append(f"⚠️ CRITICAL: MUST set tx{tx_num}_on=True (intent explicitly says to turn on TX{tx_num})")
                print(f"🚨 Detected TX ON command: TX{tx_num}")
        
        tx_commands_text = "\n".join(tx_commands) if tx_commands else ""
        
        # Format current baseline config for LLM
        baseline_description = []
        for i in range(4):
            tx_on = current_config.get(f'tx{i}_on', False)
            tx_p = current_config.get(f'tx{i}_P_dBm', 30.0)
            tx_az = current_config.get(f'tx{i}_dAz', 0.0)
            tx_el = current_config.get(f'tx{i}_dEl', 0.0)
            status = "ON" if tx_on else "OFF"
            baseline_description.append(f"  tx{i}: {status}, P={tx_p:.1f} dBm, Az={tx_az:+.1f}°, El={tx_el:+.1f}°")
        
        baseline_text = "\n".join(baseline_description)
        
        # Count active TXs in baseline
        active_tx_count = sum(1 for i in range(4) if current_config.get(f'tx{i}_on', False))
        
        # Add baseline KPIs if available
        baseline_kpis_text = ""
        if baseline_kpis:
            sinr_quality = "EXCELLENT ✅" if baseline_kpis['SINR'] > 10 else "GOOD ✅" if baseline_kpis['SINR'] > 5 else "FAIR ⚠️" if baseline_kpis['SINR'] > 0 else "POOR ❌ (interference!)"
            baseline_kpis_text = f"""
⚠️ CURRENT BASELINE KPIs ({active_tx_count} TXs active):
  RX_POWER: {baseline_kpis['RX_POWER']:.2f} dBm
  SINR: {baseline_kpis['SINR']:.2f} dB ({sinr_quality})
  COVERAGE: {baseline_kpis['COVERAGE']*100:.1f}%
  THROUGHPUT: {baseline_kpis['THROUGHPUT_5P']:.2f} Mbps
  ENERGY: {baseline_kpis['ENERGY_WATT']:.2f} W

⚠️ CRITICAL: If baseline SINR > 10 dB, be VERY careful! Small changes only (±2-3 dBm).
Don't break good SINR by adding too much power - interference will kill throughput!
"""
        
        design_prompt = f"""
Design an initial configuration for this intent:

{intent_text}
{tx_commands_text}
{baseline_kpis_text}
CURRENT BASELINE CONFIGURATION:
{baseline_text}

Provide configuration CHANGES from this baseline.
For parameters you want to change, specify the DELTA (change amount).
For parameters you want to keep, don't include them.

IMPORTANT: If the intent says "DECREASE RX_POWER by X dBm", you must REDUCE the power to make RX_POWER more negative (weaker signal).
If the intent says "INCREASE RX_POWER by X dBm", you must INCREASE the power to make RX_POWER less negative (stronger signal).
"""
        
        print(f"💬 Prompting LLM for initial design...")
        
        design_response = self.design_agent.run(design_prompt)
        
        # Extract design
        if hasattr(design_response, 'content'):
            content = design_response.content
        elif hasattr(design_response, 'messages') and design_response.messages:
            content = design_response.messages[0].content
        else:
            content = {}
        
        # Parse JSON if string
        if isinstance(content, str):
            if not content or content.strip() == "" or "cancel" in content.lower():
                raise ValueError("LLM returned empty or cancelled response! Check your GROQ_API_KEY and model availability.")
            
            content = content.replace("```json", "").replace("```", "").strip()
            
            try:
                design = ConfigDesign(**json.loads(content))
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parse Error: {e}")
                print(f"Content that failed to parse: {content}")
                raise
        elif isinstance(content, dict):
            design = ConfigDesign(**content)
        else:
            # Assume it's already a ConfigDesign object
            design = content
        
        print(f"\n✅ Initial design: {len(design.changes)} changes proposed")
        print(f"   Reasoning: {design.reasoning}")
        print(f"   Expected Outcome: {design.expected_outcome}")
        print(f"\n📋 Proposed Changes (JSON):")
        changes_json = [{"parameter": c.parameter, "before": c.before, "change": c.change, "unit": c.unit} for c in design.changes]
        print(json.dumps(changes_json, indent=2))
        
        # Apply initial design
        current_config = self._apply_changes(current_config, design.changes)
        print(f"\n✓ Configuration updated with {len(design.changes)} changes")
        
        # Predict KPIs with surrogate
        print(f"\n🔮 Predicting KPIs with surrogate model...")
        predictions = self._predict_kpis(current_config)
        all_predictions.append(predictions)
        
        print(f"\n📊 Predicted KPIs from Surrogate Model:")
        print(f"   • RX_POWER: {predictions.RX_POWER:.2f} dBm")
        print(f"   • SINR: {predictions.SINR:.2f} dB")
        print(f"   • COVERAGE: {predictions.COVERAGE*100:.1f}%")
        print(f"   • THROUGHPUT: {predictions.THROUGHPUT_5P:.2f} Mbps")
        print(f"   • LOAD_IMBALANCE: {predictions.LOAD_IMBALANCE:.3f}")
        print(f"   • ENERGY: {predictions.ENERGY_WATT:.2f} W")
        
        iteration_history.append({
            "iteration": 0,
            "changes": design.changes,
            "predictions": predictions,
            "reasoning": design.reasoning,
        })
        
        # Phase 2: Iterative Refinement
        print("\n🔄 Phase 2: Iterative Refinement")
        print("-" * 70)
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n📍 Iteration {iteration}/{self.max_iterations}")
            
            # Evaluate current predictions
            # Build baseline comparison
            baseline_comparison = ""
            if baseline_kpis:
                sinr_change = predictions.SINR - baseline_kpis['SINR']
                throughput_change = predictions.THROUGHPUT_5P - baseline_kpis['THROUGHPUT_5P']
                coverage_change = predictions.COVERAGE - baseline_kpis['COVERAGE']
                
                sinr_status = "✅ GOOD" if sinr_change >= 0 else f"❌ DEGRADED by {abs(sinr_change):.1f} dB!"
                throughput_status = "✅" if throughput_change >= 0 else f"❌ DROPPED by {abs(throughput_change):.2f} Mbps"
                
                if baseline_kpis['SINR'] > 10 and predictions.SINR < 0:
                    baseline_comparison = f"""
⚠️⚠️⚠️ CRITICAL WARNING: YOU BROKE A GOOD BASELINE! ⚠️⚠️⚠️
Baseline had EXCELLENT SINR ({baseline_kpis['SINR']:.1f} dB) but current is NEGATIVE ({predictions.SINR:.1f} dB)!
Throughput: {baseline_kpis['THROUGHPUT_5P']:.2f} → {predictions.THROUGHPUT_5P:.2f} Mbps ({throughput_status})
ACTION REQUIRED: REDUCE power significantly or go back closer to baseline config!
"""
                else:
                    baseline_comparison = f"""
Comparison vs Baseline:
  SINR: {baseline_kpis['SINR']:.1f} → {predictions.SINR:.1f} dB ({sinr_status})
  THROUGHPUT: {baseline_kpis['THROUGHPUT_5P']:.2f} → {predictions.THROUGHPUT_5P:.2f} Mbps ({throughput_status})
  COVERAGE: {baseline_kpis['COVERAGE']*100:.0f} → {predictions.COVERAGE*100:.0f}% (change: {coverage_change*100:+.0f}%)
"""
            
            refinement_prompt = f"""
Evaluate the current configuration results:

TARGET (from intent):
{intent_text}
{tx_commands_text}
{baseline_comparison}
CURRENT PREDICTIONS (from surrogate model trained on 1M+ simulations):
- RX_POWER: {predictions.RX_POWER:.2f} dBm
- SINR: {predictions.SINR:.2f} dB
- COVERAGE: {predictions.COVERAGE*100:.1f}%
- THROUGHPUT: {predictions.THROUGHPUT_5P:.2f} Mbps
- LOAD_IMBALANCE: {predictions.LOAD_IMBALANCE:.3f}
- ENERGY: {predictions.ENERGY_WATT:.2f} W

CURRENT CONFIGURATION:
{json.dumps(current_config, indent=2)}

Does this meet the target KPIs? If not, propose specific refinements.
"""
            
            print(f"\n💬 Prompting LLM for evaluation...")
            
            refinement_response = self.refinement_agent.run(refinement_prompt)
            
            # Extract decision
            if hasattr(refinement_response, 'content'):
                content = refinement_response.content
            elif hasattr(refinement_response, 'messages') and refinement_response.messages:
                content = refinement_response.messages[0].content
            else:
                content = {}
            
            if isinstance(content, str):
                if not content or content.strip() == "" or "cancel" in content.lower():
                    print(f"⚠️ LLM request cancelled or empty response, stopping refinement.")
                    break
                
                content = content.replace("```json", "").replace("```", "").strip()
                
                try:
                    decision = RefinementDecision(**json.loads(content))
                except json.JSONDecodeError as e:
                    print(f"❌ JSON Parse Error: {e}")
                    print(f"Content that failed to parse: {content}")
                    break
            elif isinstance(content, dict):
                decision = RefinementDecision(**content)
            else:
                decision = content
            
            print(f"\n✅ Evaluation Complete:")
            print(f"   • Target met: {decision.target_met}")
            print(f"   • Refinement needed: {decision.refinement_needed}")
            print(f"   • Reasoning: {decision.reasoning}")
            
            # Check if target is met
            if decision.target_met or not decision.refinement_needed:
                print(f"🎯 Target achieved! Stopping refinement.")
                break
            
            # Apply refinements
            if decision.next_changes:
                print(f"\n🔧 Applying {len(decision.next_changes)} refinements...")
                print(f"\n📋 Proposed Refinements (JSON):")
                refinement_changes_json = [{"parameter": c.parameter, "before": c.before, "change": c.change, "unit": c.unit} for c in decision.next_changes]
                print(json.dumps(refinement_changes_json, indent=2))
                
                current_config = self._apply_changes(current_config, decision.next_changes)
                print(f"\n✓ Configuration updated with {len(decision.next_changes)} changes")
                
                # Predict again
                print(f"\n🔮 Predicting KPIs with surrogate model...")
                predictions = self._predict_kpis(current_config)
                all_predictions.append(predictions)
                
                print(f"\n📊 New Predicted KPIs from Surrogate Model:")
                print(f"   • RX_POWER: {predictions.RX_POWER:.2f} dBm")
                print(f"   • SINR: {predictions.SINR:.2f} dB")
                print(f"   • COVERAGE: {predictions.COVERAGE*100:.1f}%")
                print(f"   • THROUGHPUT: {predictions.THROUGHPUT_5P:.2f} Mbps")
                print(f"   • LOAD_IMBALANCE: {predictions.LOAD_IMBALANCE:.3f}")
                print(f"   • ENERGY: {predictions.ENERGY_WATT:.2f} W")
                
                iteration_history.append({
                    "iteration": iteration,
                    "changes": decision.next_changes,
                    "predictions": predictions,
                    "reasoning": decision.reasoning,
                })
            else:
                print(f"⚠️ No refinements proposed, stopping.")
                break
        
        # Phase 3: Final Result
        print("\n" + "="*70)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*70)
        
        # Check if target was met
        final_predictions = all_predictions[-1]
        target_met = self._check_target_met(intent, final_predictions, baseline_kpis)
        
        # Format final changes (from baseline)
        final_changes = []
        baseline = self._baseline_config()
        for param, value in current_config.items():
            if param in baseline and baseline[param] != value:
                # Calculate change amount
                before_val = baseline[param]
                if isinstance(value, bool):
                    change_val = str(value)  # Boolean as string
                else:
                    change_val = value - before_val  # Numeric delta
                
                final_changes.append({
                    "parameter": param,
                    "before": before_val,
                    "change": change_val,
                })
        
        # Create complete reasoning
        reasoning_parts = []
        for hist in iteration_history:
            reasoning_parts.append(
                f"Iteration {hist['iteration']}: {hist['reasoning']}\n"
                f"  Predicted: RX_POWER={hist['predictions'].RX_POWER:.2f} dBm, "
                f"SINR={hist['predictions'].SINR:.2f} dB"
            )
        
        complete_reasoning = "\n\n".join(reasoning_parts)
        
        # Build result with current_config and current_kpis
        # Store the initial baseline config as current_config
        initial_baseline = self._baseline_config()
        if active_intents and len(active_intents) > 0:
            last_intent = active_intents[-1]
            last_config = last_intent.get('output', {}).get('final_config', {})
            if last_config:
                initial_baseline = last_config.copy()
        
        result = OptimizationResult(
            test_name=intent.get("intent_text", "optimization"),
            passed=target_met,
            input=intent,
            output={
                "selected_config_id": 1,
                "changes": final_changes,
                "final_config": current_config,  # Include final config for next iteration
                "current_config": initial_baseline,  # Configuration before optimization
                "expected_kpis": {
                    "RX_POWER": final_predictions.RX_POWER,
                    "SINR": final_predictions.SINR,
                    "COVERAGE": final_predictions.COVERAGE,
                    "THROUGHPUT_5P": final_predictions.THROUGHPUT_5P,
                    "LOAD_IMBALANCE": final_predictions.LOAD_IMBALANCE,
                    "ENERGY_WATT": final_predictions.ENERGY_WATT,
                },
                "current_kpis": {
                    "RX_POWER": baseline_kpis.get("RX_POWER") if baseline_kpis else None,
                    "SINR": baseline_kpis.get("SINR") if baseline_kpis else None,
                    "COVERAGE": baseline_kpis.get("COVERAGE") if baseline_kpis else None,
                    "THROUGHPUT_5P": baseline_kpis.get("THROUGHPUT_5P") if baseline_kpis else None,
                    "LOAD_IMBALANCE": baseline_kpis.get("LOAD_IMBALANCE") if baseline_kpis else None,
                    "ENERGY_WATT": baseline_kpis.get("ENERGY_WATT") if baseline_kpis else None,
                },
                "constraints_satisfied": target_met,
            },
            reasoning=complete_reasoning,
            result_id=result_id,
            priority=intent.get("priority", "MEDIUM"),
            iterations=len(iteration_history),
            surrogate_predictions=all_predictions,
        )
        
        print(f"📝 Final Result:")
        print(f"   Result ID: {result_id}")
        print(f"   Passed: {target_met}")
        print(f"   Iterations: {len(iteration_history)}")
        print(f"   Final KPIs: RX_POWER={final_predictions.RX_POWER:.2f} dBm")
        
        return result
    
    def _check_target_met(self, intent: Dict, predictions: KPIPrediction, baseline_kpis: Optional[Dict] = None) -> bool:
        """Check if target KPIs are met."""
        for kpi_threshold in intent.get("kpi_thresholds", []):
            # Support both "kpi" and "kpi_name" fields
            kpi_name = kpi_threshold.get("kpi", kpi_threshold.get("kpi_name", ""))
            operator = kpi_threshold.get("op", kpi_threshold.get("operator", "GTE"))
            target_value = kpi_threshold.get("value")
            
            if target_value is None:
                continue
            
            target_value = float(target_value)
            
            # Map KPI name to prediction field
            if kpi_name == "RX_POWER":
                predicted_value = predictions.RX_POWER
            elif kpi_name == "SINR":
                predicted_value = predictions.SINR
            elif kpi_name == "COVERAGE":
                predicted_value = predictions.COVERAGE
            elif kpi_name == "THROUGHPUT_5P" or kpi_name == "THROUGHPUT":
                predicted_value = predictions.THROUGHPUT_5P
            elif kpi_name == "LOAD_IMBALANCE":
                predicted_value = predictions.LOAD_IMBALANCE
            elif kpi_name == "ENERGY_WATT" or kpi_name == "ENERGY":
                predicted_value = predictions.ENERGY_WATT
            else:
                continue
            
            # Handle DELTA operators
            if operator == "DELTA_DOWN":
                # target_value is the amount to decrease
                # Need baseline to compare
                if baseline_kpis and kpi_name in baseline_kpis:
                    baseline_val = baseline_kpis[kpi_name]
                    # For RX_POWER: decreasing means going more negative
                    # baseline -53, delta 5 → target -58 (more negative)
                    # predicted should be <= target (i.e., <= -58)
                    target_threshold = baseline_val - target_value
                    if predicted_value > target_threshold:
                        return False
                else:
                    # No baseline, can't verify DELTA - assume met
                    continue
            elif operator == "DELTA_UP":
                # target_value is the amount to increase
                if baseline_kpis and kpi_name in baseline_kpis:
                    baseline_val = baseline_kpis[kpi_name]
                    # For RX_POWER: increasing means going less negative
                    # baseline -58, delta 5 → target -53 (less negative)
                    # predicted should be >= target (i.e., >= -53)
                    target_threshold = baseline_val + target_value
                    if predicted_value < target_threshold:
                        return False
                else:
                    # No baseline, can't verify DELTA - assume met
                    continue
            # Standard comparison operators
            elif operator in [">=", "GTE"]:
                if predicted_value < target_value:
                    return False
            elif operator in ["<=", "LTE"]:
                if predicted_value > target_value:
                    return False
            elif operator in [">", "GT"]:
                if predicted_value <= target_value:
                    return False
            elif operator in ["<", "LT"]:
                if predicted_value >= target_value:
                    return False
        
        return True


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

def run_hybrid_optimization(
    intent: Dict,
    surrogate_model_path: Optional[str] = None,
    active_intents: Optional[List[Dict]] = None,
) -> Dict:
    """
    Main function to run hybrid optimization.
    
    Args:
        intent: Parsed intent dictionary
        surrogate_model_path: Path to surrogate model
        active_intents: List of active optimization results to build upon
        
    Returns:
        Optimization result as dictionary
    """
    print("\n" + "🔬 HYBRID OPTIMIZATION (Surrogate Model + LLM)")
    print("="*70)
    
    # Initialize agent
    agent = HybridOptimizationAgent(
        surrogate_model_path=surrogate_model_path,
        max_iterations=3,
    )
    
    # Run optimization
    result = agent.optimize(intent, active_intents=active_intents)
    
    # Convert to dict
    return result.model_dump()


if __name__ == "__main__":
    # Test with example intent
    test_intent = {
        "intent_text": "Increase power to at least -55 dBm in Kartal region. Priority is high.",
        "target_area": "Kartal",
        "priority": "HIGH",
        "target_kpis": ["RX_POWER"],
        "kpi_thresholds": [
            {"kpi_name": "RX_POWER", "operator": ">=", "value": -55.0}
        ],
    }
    
    result = run_hybrid_optimization(test_intent)
    
    print("\n" + "="*70)
    print("📊 FINAL RESULT:")
    print("="*70)
    print(json.dumps(result, indent=2))
