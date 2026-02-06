#!/usr/bin/env python3
"""
Optimization Agent (Agno-based)
Complete working optimization agent for telecom base-station configuration using Agno framework.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq

# Import the core optimization logic
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from optimization_agent_v2 import OptimizationAgent as CoreOptimizationAgent, SurrogateModel

load_dotenv()


# ============================================================================
# RESPONSE SCHEMA
# ============================================================================

class KPIValues(BaseModel):
    """KPI values for network performance."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    RX_POWER: Optional[float] = Field(None, description="Received power in dBm")
    SINR: Optional[float] = Field(None, description="Signal to Interference + Noise Ratio in dB")
    THROUGHPUT_5P: Optional[float] = Field(None, description="5th percentile throughput in Mbps")
    LOAD_IMBALANCE: Optional[float] = Field(None, description="Load imbalance metric")
    RX_COVERAGE_RATIO: Optional[float] = Field(None, description="RX coverage ratio")


class ConfigChange(BaseModel):
    """Single configuration parameter change."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    param: str = Field(..., description="Parameter name (e.g., tx0_P_dBm)")
    before: Optional[Any] = Field(None, description="Value before change")
    change: Any = Field(..., description="Change value (delta for numeric, new value for boolean)")
    unit: Optional[str] = Field(None, description="Unit of measurement (dBm, deg, etc.)")


class OptimizationOutput(BaseModel):
    """Output from optimization process."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    selected_config_id: int = Field(..., description="ID of selected configuration")
    current_config_id: Optional[int] = Field(None, description="ID of current configuration")
    changes: List[ConfigChange] = Field(..., description="List of configuration changes")
    expected_kpis: KPIValues = Field(..., description="Expected KPIs after applying changes")
    current_kpis: Optional[KPIValues] = Field(None, description="Current KPIs before changes")
    constraints_satisfied: bool = Field(..., description="Whether all constraints are satisfied")


class OptimizationResult(BaseModel):
    """Complete optimization result with reasoning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    test_name: str = Field(..., description="Test/optimization name")
    passed: bool = Field(..., description="Whether optimization passed constraints")
    input: Dict = Field(..., description="Original intent input")
    output: OptimizationOutput = Field(..., description="Optimization output")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the optimization decisions")
    result_id: str = Field(..., description="Unique result ID")
    priority: str = Field(default="MEDIUM", description="Priority level: LOW, MEDIUM, HIGH, CRITICAL")


# ============================================================================
# AGENT INSTRUCTIONS
# ============================================================================

INSTRUCTIONS = [
    "You are an expert optimization agent for cellular/mobile network configuration.",
    "Your task: Generate optimal base-station configurations to achieve user's network performance goals.",
    "",
    "## Your Responsibilities:",
    "1. Analyze the parsed intent with target KPIs and constraints",
    "2. Use the surrogate model to predict network performance for different configurations",
    "3. Search for optimal configuration that satisfies all constraints",
    "4. Ensure changes are safe and within operational bounds (guardrails)",
    "5. Provide clear reasoning for configuration decisions",
    "",
    "## Configuration Parameters (per transmitter tx0-tx3):",
    "- tx{i}_on: Boolean - transmitter on/off state",
    "- tx{i}_P_dBm: Float - transmission power in dBm (range: 20-43 dBm)",
    "- tx{i}_dAz: Float - azimuth angle delta in degrees (range: -30 to +30)",
    "- tx{i}_dEl: Float - elevation/tilt angle delta in degrees (range: -5 to +5)",
    "",
    "## KPI Mapping:",
    "- RX_POWER: Received signal power at 5th percentile (Prx_p5_dBm)",
    "- SINR: Signal to Interference + Noise Ratio at 5th percentile",
    "- THROUGHPUT_5P: User throughput at 5th percentile",
    "- SERVED_USERS: Load balance across transmitters (lower imbalance = better)",
    "",
    "## Constraint Operators:",
    "- GTE (>=): Value must be greater than or equal to threshold",
    "- GT (>): Value must be greater than threshold",
    "- LTE (<=): Value must be less than or equal to threshold",
    "- LT (<): Value must be less than threshold",
    "- BETWEEN: Value must be between two thresholds",
    "- DELTA_UP: Value must increase by at least delta from current",
    "- DELTA_DOWN: Value must decrease by at least delta from current",
    "- TARGET: Optimize towards target value",
    "",
    "## Optimization Strategy:",
    "1. For DELTA_DOWN (power reduction) intents:",
    "   - Minimize total TX power while maintaining coverage",
    "   - Explore low-power configurations",
    "   - Consider turning off some transmitters if possible",
    "",
    "2. For coverage/capacity improvement intents:",
    "   - Increase power and optimize angles for better coverage",
    "   - Balance load across transmitters",
    "   - Maximize RX power and SINR",
    "",
    "3. Always apply guardrails to limit changes:",
    "   - Power: ±3 dB max change",
    "   - Elevation: ±2° max change",
    "   - Azimuth: ±10° max change",
    "",
    "## Reasoning Guidelines:",
    "- Explain WHY you chose specific configuration changes",
    "- Describe trade-offs between competing objectives",
    "- Justify how the solution meets constraints",
    "- Note any limitations or risks",
    "- Be concise but informative (2-4 sentences)",
    "",
    "## Important:",
    "- You receive a pre-computed optimization result from the core engine",
    "- Your role is to validate it and provide reasoning/context",
    "- Focus on explaining the WHAT and WHY of the decisions",
    "- If constraints are not satisfied, explain why and suggest alternatives",
]


# ============================================================================
# AGENT DEFINITION
# ============================================================================

optimization_agent = Agent(
    name="OptimizationAgent",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=INSTRUCTIONS,
    markdown=False,
)


# ============================================================================
# HELPER FUNCTION TO RUN OPTIMIZATION
# ============================================================================

def run_optimization(intent: Dict[str, Any], surrogate_model_path: str = None) -> OptimizationResult:
    """
    Run optimization for given intent.
    
    Args:
        intent: Parsed intent dictionary with target_kpis, kpi_thresholds, etc.
        surrogate_model_path: Path to surrogate model (optional, defaults to models/surrogate.joblib)
    
    Returns:
        OptimizationResult with configuration and reasoning
    """
    print("\n" + "="*70)
    print("🤖 OPTIMIZATION AGENT - Starting...")
    print("="*70)
    
    try:
        # Load surrogate model
        if surrogate_model_path is None:
            surrogate_model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "models", 
                "surrogate.joblib"
            )
        
        print(f"📂 Loading surrogate model from: {surrogate_model_path}")
    
        if not os.path.exists(surrogate_model_path):
            raise FileNotFoundError(f"Surrogate model not found: {surrogate_model_path}")
        
        print("✅ Model loaded successfully")
        surrogate = SurrogateModel.load(surrogate_model_path)
        
        # Initialize core optimization agent
        print("🔧 Initializing core optimization engine...")
        core_agent = CoreOptimizationAgent(
            surrogate=surrogate,
            search_iterations=1000,
            random_seed=42
        )
        
        # Run optimization (core algorithm)
        print("🔍 Running optimization search...")
        core_result = core_agent.optimize(intent)
        print("✅ Core optimization completed")
        
        # Generate unique result ID
        import datetime
        result_id = f"opt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract priority from intent
        priority = intent.get("priority", "MEDIUM")
        
        # Add result_id and priority to result
        core_result["result_id"] = result_id
        core_result["priority"] = priority
        
        # Prepare prompt for LLM to generate reasoning
        print("🧠 Generating reasoning with LLM...")
        
        # Format current KPIs if available
        current_kpis_text = ""
        if core_result['output']['current_kpis']:
            current_kpis_text = f"Current KPIs:\n{_format_kpis(core_result['output']['current_kpis'])}\n"
        
        prompt = f"""Analyze this optimization result and provide clear reasoning:

INPUT INTENT:
- Target Area: {intent.get('target_area')}
- Target KPIs: {', '.join(intent.get('target_kpis', []))}
- Priority: {priority}
- Constraints: {len(intent.get('kpi_thresholds', []))} constraint(s)

OPTIMIZATION RESULT:
- Passed: {core_result['passed']}
- Configuration Changes: {len(core_result['output']['changes'])} parameter(s) changed
- Constraints Satisfied: {core_result['output']['constraints_satisfied']}

Expected KPIs:
{_format_kpis(core_result['output']['expected_kpis'])}

{current_kpis_text}
Provide the complete OptimizationResult with reasoning that explains:
1. Why these configuration changes were selected
2. How they achieve the target KPIs
3. Any trade-offs or limitations

The result should include all fields from the input with added reasoning."""

        # Run agent to generate reasoning
        response = optimization_agent.run(prompt, stream=False)
        print("✅ LLM reasoning generated")
        
        # Agent returns text response, we use core result with added reasoning
        # Extract reasoning from LLM response
        reasoning_text = ""
        if hasattr(response, 'content'):
            reasoning_text = str(response.content)
        elif isinstance(response, str):
            reasoning_text = response
        else:
            reasoning_text = "Optimization completed successfully."
        
        # Return core result with LLM reasoning
        print("\n✅ OPTIMIZATION AGENT - Completed Successfully!")
        print("="*70)
        return OptimizationResult(**core_result, reasoning=reasoning_text)
    
    except Exception as e:
        print("\n❌ OPTIMIZATION AGENT - ERROR!")
        print("="*70)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        import traceback
        print("\nFull Traceback:")
        traceback.print_exc()
        print("="*70)
        raise RuntimeError(f"OptimizationAgent failed: {str(e)}") from e


def _format_kpis(kpis: Dict) -> str:
    """Format KPIs for display."""
    if not kpis:
        return "N/A"
    
    lines = []
    if kpis.get('RX_POWER') is not None:
        lines.append(f"  - RX_POWER: {kpis['RX_POWER']:.2f} dBm")
    if kpis.get('SINR') is not None:
        lines.append(f"  - SINR: {kpis['SINR']:.2f} dB")
    if kpis.get('THROUGHPUT_5P') is not None:
        lines.append(f"  - THROUGHPUT_5P: {kpis['THROUGHPUT_5P']:.2f} Mbps")
    if kpis.get('LOAD_IMBALANCE') is not None:
        lines.append(f"  - LOAD_IMBALANCE: {kpis['LOAD_IMBALANCE']:.4f}")
    
    return '\n'.join(lines) if lines else "N/A"


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    test_intent = {
        "target_area": "test_site_001",
        "target_kpis": ["RX_POWER", "SINR"],
        "kpi_thresholds": [
            {"kpi": "RX_POWER", "op": "GTE", "value": -95.0, "unit": "dBm"}
        ],
        "priority": "HIGH",
        "confidence": 0.95,
        "current_config": {
            'tx0_on': True, 'tx0_P_dBm': 30.0, 'tx0_dAz': 0.0, 'tx0_dEl': 0.0,
            'tx1_on': True, 'tx1_P_dBm': 30.0, 'tx1_dAz': 0.0, 'tx1_dEl': 0.0,
            'tx2_on': True, 'tx2_P_dBm': 30.0, 'tx2_dAz': 0.0, 'tx2_dEl': 0.0,
            'tx3_on': True, 'tx3_P_dBm': 30.0, 'tx3_dAz': 0.0, 'tx3_dEl': 0.0
        },
        "k_users": 800,
        "user_set_id": 0,
        "test_name": "test_optimization"
    }
    
    result = run_optimization(test_intent)
    print(f"\n✅ Optimization Result:")
    print(f"   Test: {result.test_name}")
    print(f"   Passed: {result.passed}")
    print(f"   Result ID: {result.result_id}")
    print(f"   Priority: {result.priority}")
    if result.reasoning:
        print(f"\n📝 Reasoning:\n{result.reasoning}")
