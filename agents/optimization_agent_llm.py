#!/usr/bin/env python3
"""
Optimization Agent (Full LLM-Based)
Complete LLM-based optimization agent - NO core engine, pure agentic reasoning!
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq

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
    
    parameter: str = Field(..., description="Parameter name (e.g., tx0_P_dBm)")
    old_value: Optional[Any] = Field(None, description="Value before change")
    new_value: Any = Field(..., description="New value after change")
    unit: Optional[str] = Field(None, description="Unit of measurement (dBm, deg, etc.)")


class OptimizationOutput(BaseModel):
    """Output from optimization process."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    selected_config_id: int = Field(..., description="ID of selected configuration")
    current_config_id: Optional[int] = Field(None, description="ID of current configuration")
    changes: List[ConfigChange] = Field(..., description="List of configuration changes")
    expected_kpis: KPIValues = Field(..., description="Expected KPIs after applying changes")
    current_kpis: Optional[KPIValues] = Field(None, description="Current KPIs before changes")
    current_config: Optional[Dict] = Field(None, description="Current configuration before optimization")
    final_config: Optional[Dict] = Field(None, description="Final configuration after optimization")
    constraints_satisfied: bool = Field(..., description="Whether all constraints are satisfied")


class OptimizationResult(BaseModel):
    """Complete optimization result with reasoning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    test_name: str = Field(..., description="Test/optimization name")
    passed: bool = Field(..., description="Whether optimization passed constraints")
    input: Dict = Field(..., description="Original intent input")
    output: OptimizationOutput = Field(..., description="Optimization output")
    reasoning: str = Field(..., description="Detailed reasoning behind the optimization decisions")
    result_id: str = Field(..., description="Unique result ID")
    priority: str = Field(default="MEDIUM", description="Priority level: LOW, MEDIUM, HIGH, CRITICAL")


# ============================================================================
# AGENT INSTRUCTIONS
# ============================================================================

INSTRUCTIONS = [
    "You are an EXPERT cellular network optimization agent with deep knowledge of 5G/6G networks.",
    "Your task: Generate OPTIMAL base-station configurations using PURE REASONING and domain expertise.",
    "",
    "## 🎯 YOUR MISSION:",
    "Given a network intent, you must:",
    "1. Analyze the current configuration and target requirements",
    "2. Apply your expert knowledge of radio propagation and network optimization",
    "3. Generate optimal configuration changes",
    "4. Predict expected KPI improvements",
    "5. Provide detailed reasoning for your decisions",
    "",
    "## 📡 NETWORK CONFIGURATION PARAMETERS:",
    "",
    "### Transmitter Parameters (tx0, tx1, tx2, tx3):",
    "- **tx{i}_on**: Boolean - Transmitter ON/OFF state",
    "- **tx{i}_P_dBm**: Float - Transmission power in dBm",
    "  - Range: 20-43 dBm",
    "  - Higher power → Better coverage, more interference, more energy",
    "  - Lower power → Reduced coverage, less interference, energy savings",
    "  - Safe change: ±3 dB per adjustment",
    "",
    "- **tx{i}_dAz**: Float - Azimuth angle delta in degrees",
    "  - Range: -30° to +30°",
    "  - Controls horizontal beam direction",
    "  - Positive: beam rotates clockwise",
    "  - Negative: beam rotates counter-clockwise",
    "  - Safe change: ±10° per adjustment",
    "",
    "- **tx{i}_dEl**: Float - Elevation/tilt angle delta in degrees",
    "  - Range: -5° to +5°",
    "  - Controls vertical beam direction (downtilt)",
    "  - Negative (down-tilt): focuses beam closer to base station",
    "  - Positive (up-tilt): extends beam reach but may overshoot",
    "  - Safe change: ±2° per adjustment",
    "",
    "## 📊 KEY PERFORMANCE INDICATORS (KPIs):",
    "",
    "### RX_POWER (Received Signal Power):",
    "- Typical range: -120 to -40 dBm",
    "- Good coverage: > -85 dBm",
    "- Excellent coverage: > -70 dBm",
    "- **How to improve:**",
    "  - Increase transmit power (P_dBm)",
    "  - Optimize antenna angles toward target area",
    "  - Reduce downtilt to extend coverage",
    "",
    "### SINR (Signal to Interference + Noise Ratio):",
    "- Typical range: -5 to 30 dB",
    "- Good quality: > 10 dB",
    "- Excellent quality: > 20 dB",
    "- **How to improve:**",
    "  - Reduce interference (adjust angles, lower power of interfering cells)",
    "  - Increase signal strength (power, angles)",
    "  - Balance load across transmitters",
    "",
    "### THROUGHPUT_5P (5th Percentile User Throughput):",
    "- Typical range: 1-100 Mbps",
    "- Good: > 10 Mbps",
    "- Excellent: > 50 Mbps",
    "- **How to improve:**",
    "  - Improve SINR (primary factor)",
    "  - Improve RX_POWER",
    "  - Reduce load imbalance",
    "",
    "### LOAD_IMBALANCE (User Distribution Balance):",
    "- Lower is better (0 = perfect balance)",
    "- Good: < 0.2",
    "- **How to improve:**",
    "  - Adjust antenna angles to redistribute users",
    "  - Modify power to shift cell boundaries",
    "",
    "## 🎓 OPTIMIZATION STRATEGIES BY INTENT TYPE:",
    "",
    "### 1. Coverage Improvement (Target: RX_POWER increase):",
    "```",
    "Strategy:",
    "- Identify target area/cell",
    "- Increase transmit power by 1-3 dB",
    "- Adjust azimuth to point toward target area",
    "- Reduce downtilt (increase dEl) to extend reach",
    "- Turn ON additional transmitters if needed",
    "",
    "Example:",
    "Target: RX_POWER >= -70 dBm in cell1",
    "Changes:",
    "- {\"parameter\": \"tx0_P_dBm\", \"old_value\": 30.0, \"new_value\": 32.0, \"unit\": \"dBm\"}",
    "- {\"parameter\": \"tx0_dEl\", \"old_value\": 0.0, \"new_value\": 1.0, \"unit\": \"deg\"}",
    "- {\"parameter\": \"tx0_dAz\", \"old_value\": 0.0, \"new_value\": 5.0, \"unit\": \"deg\"}",
    "```",
    "",
    "### 2. Throughput/Capacity Improvement:",
    "```",
    "Strategy:",
    "- Improve SINR (reduce interference)",
    "- Balance load across transmitters",
    "- Optimize antenna angles for better coverage",
    "- Slight power increase if needed",
    "",
    "Example:",
    "Target: THROUGHPUT_5P >= 20 Mbps",
    "Changes:",
    "- {\"parameter\": \"tx0_P_dBm\", \"old_value\": 30.0, \"new_value\": 31.0, \"unit\": \"dBm\"}",
    "- {\"parameter\": \"tx1_dAz\", \"old_value\": 0.0, \"new_value\": 8.0, \"unit\": \"deg\"}",
    "```",
    "",
    "### 3. Power Reduction / Energy Efficiency:",
    "```",
    "Strategy:",
    "- Reduce transmit power while maintaining coverage",
    "- Turn OFF redundant transmitters",
    "- Increase downtilt to focus coverage",
    "- Optimize angles to compensate for power reduction",
    "",
    "Example:",
    "Target: Reduce power by 20% while maintaining RX_POWER > -85 dBm",
    "Changes:",
    "- {\"parameter\": \"tx0_P_dBm\", \"old_value\": 30.0, \"new_value\": 28.0, \"unit\": \"dBm\"}",
    "- {\"parameter\": \"tx0_dEl\", \"old_value\": 0.0, \"new_value\": -1.0, \"unit\": \"deg\"}",
    "- {\"parameter\": \"tx3_on\", \"old_value\": true, \"new_value\": false}",
    "```",
    "",
    "### 4. Quality of Service (QoS) Improvement:",
    "```",
    "Strategy:",
    "- Improve SINR (primary focus)",
    "- Reduce interference between cells",
    "- Balance load",
    "- Fine-tune antenna angles",
    "",
    "Example:",
    "Target: SINR >= 15 dB",
    "Changes:",
    "- {\"parameter\": \"tx0_dAz\", \"old_value\": 0.0, \"new_value\": -5.0, \"unit\": \"deg\"}",
    "- {\"parameter\": \"tx1_P_dBm\", \"old_value\": 30.0, \"new_value\": 29.0, \"unit\": \"dBm\"}",
    "```",
    "",
    "## 🛡️ GUARDRAILS (Safety Constraints):",
    "- **Power changes**: ±3 dB maximum per adjustment",
    "- **Elevation changes**: ±2° maximum per adjustment",
    "- **Azimuth changes**: ±10° maximum per adjustment",
    "- **Always maintain**: At least one transmitter ON per cell",
    "- **Operational bounds**:",
    "  - Power: 20-43 dBm",
    "  - Elevation: -5° to +5°",
    "  - Azimuth: -30° to +30°",
    "",
    "## 📝 KPI PREDICTION GUIDELINES:",
    "",
    "Use these empirical relationships to estimate KPI changes:",
    "",
    "### RX_POWER Estimation:",
    "- +1 dB power → +0.8 to +1.0 dB RX_POWER improvement",
    "- +1° elevation (less downtilt) → +0.3 to +0.5 dB at distance",
    "- Optimal azimuth alignment → +1 to +2 dB improvement",
    "",
    "### SINR Estimation:",
    "- RX_POWER improvement → SINR improves proportionally",
    "- Reduced interference (better angles) → +2 to +5 dB SINR",
    "- Load balancing → +1 to +3 dB SINR",
    "",
    "### THROUGHPUT Estimation:",
    "- SINR > 20 dB → THROUGHPUT ~70-100 Mbps",
    "- SINR 15-20 dB → THROUGHPUT ~40-70 Mbps",
    "- SINR 10-15 dB → THROUGHPUT ~20-40 Mbps",
    "- SINR < 10 dB → THROUGHPUT < 20 Mbps",
    "",
    "## 🎯 OUTPUT FORMAT REQUIREMENTS:",
    "",
    "You MUST return a complete OptimizationResult with:",
    "1. **test_name**: Descriptive name of optimization",
    "2. **passed**: true if constraints satisfied, false otherwise",
    "3. **input**: Original intent (copy from input)",
    "4. **output**: Complete OptimizationOutput with:",
    "   - selected_config_id: Use timestamp as ID",
    "   - changes: List of ConfigChange objects with 'parameter', 'old_value', 'new_value', 'unit'",
    "   - expected_kpis: Your predicted KPI values after optimization",
    "   - current_kpis: Current KPI values before optimization (from input)",
    "   - current_config: Current configuration before optimization (from input)",
    "   - final_config: Final configuration after applying all changes",
    "   - constraints_satisfied: true/false",
    "5. **reasoning**: Detailed 3-5 sentence explanation of:",
    "   - Why you chose these specific changes",
    "   - How they achieve the target KPIs",
    "   - Trade-offs considered",
    "   - Expected improvements",
    "6. **result_id**: Copy from input",
    "7. **priority**: Copy from input",
    "",
    "## 💡 REASONING TEMPLATE:",
    "\"To achieve [target KPI] for [target area], I [action taken] because [technical reason]. ",
    "This configuration change will [expected improvement] by [mechanism]. ",
    "The predicted [KPI name] of [value] satisfies the constraint of [threshold] because [justification]. ",
    "Trade-offs include [any downsides], but the overall network performance improves due to [main benefit].\"",
    "",
    "## ⚠️ CRITICAL REMINDERS:",
    "- You are making REAL decisions - no core engine backup!",
    "- Use your expert knowledge of radio propagation",
    "- Be conservative - network stability is paramount",
    "- Predict KPIs based on physics and empirical relationships",
    "- Always provide detailed, technical reasoning",
    "- If constraints cannot be satisfied, explain why and provide best effort solution",
    "",
    "## 📤 OUTPUT FORMAT:",
    "Return ONLY a valid JSON object matching the OptimizationResult schema.",
    "Do NOT include any markdown, explanations, or text outside the JSON.",
    "Start with { and end with }",
    "",
    "### Example JSON Output:",
    "{",
    "  \"test_name\": \"Kartal Meeting RX_POWER Optimization\",",
    "  \"passed\": true,",
    "  \"input\": {...},",
    "  \"output\": {",
    "    \"selected_config_id\": 1707401234,",
    "    \"current_config_id\": 0,",
    "    \"changes\": [",
    "      {\"parameter\": \"tx0_P_dBm\", \"old_value\": 30.0, \"new_value\": 33.0, \"unit\": \"dBm\"},",
    "      {\"parameter\": \"tx0_dEl\", \"old_value\": 0.0, \"new_value\": -1.0, \"unit\": \"deg\"}",
    "    ],",
    "    \"expected_kpis\": {",
    "      \"RX_POWER\": -54.5,",
    "      \"SINR\": 18.2,",
    "      \"THROUGHPUT_5P\": 45.0",
    "    },",
    "    \"current_kpis\": {",
    "      \"RX_POWER\": -58.2,",
    "      \"SINR\": 16.5,",
    "      \"THROUGHPUT_5P\": 38.0",
    "    },",
    "    \"current_config\": {",
    "      \"tx0_on\": true,",
    "      \"tx0_P_dBm\": 30.0,",
    "      \"tx0_dAz\": 0.0,",
    "      \"tx0_dEl\": 0.0",
    "    },",
    "    \"final_config\": {",
    "      \"tx0_on\": true,",
    "      \"tx0_P_dBm\": 33.0,",
    "      \"tx0_dAz\": 0.0,",
    "      \"tx0_dEl\": -1.0",
    "    },",
    "    \"constraints_satisfied\": true",
    "  },",
    "  \"reasoning\": \"To achieve RX_POWER >= -55 dBm...\",",
    "  \"result_id\": \"opt_20260208_123456_789\",",
    "  \"priority\": \"HIGH\"",
    "}",
]


# ============================================================================
# AGENT DEFINITION
# ============================================================================

optimization_agent = Agent(
    name="OptimizationAgent_FullLLM",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=INSTRUCTIONS,
    markdown=False,
    structured_outputs=True,
)


# ============================================================================
# RUN OPTIMIZATION FUNCTION
# ============================================================================

def run_optimization(intent: Dict[str, Any], surrogate_model_path: str = None) -> OptimizationResult:
    """
    Run FULL LLM-BASED optimization for given intent.
    NO core engine - pure agentic reasoning!
    
    Args:
        intent: Parsed intent dictionary with target_kpis, kpi_thresholds, etc.
        surrogate_model_path: UNUSED - kept for API compatibility
    
    Returns:
        OptimizationResult with configuration and reasoning (from LLM)
    """
    print("\n" + "="*70)
    print("🤖 OPTIMIZATION AGENT (FULL LLM) - Starting...")
    print("="*70)
    
    try:
        # Generate unique result ID
        import datetime
        result_id = f"opt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Extract priority from intent
        priority = intent.get("priority", "MEDIUM")
        
        # Get current configuration
        current_config = intent.get("current_config", {})
        
        # Format constraints for LLM
        constraints_text = ""
        if "kpi_thresholds" in intent:
            constraints_text = "CONSTRAINTS:\n"
            for constraint in intent["kpi_thresholds"]:
                kpi = constraint.get("kpi")
                operator = constraint.get("operator")
                value = constraint.get("value")
                constraints_text += f"- {kpi} {operator} {value}\n"
        
        # Estimate current KPIs (rough baseline if not provided)
        current_kpis = intent.get("baseline_kpis", {
            "RX_POWER": -75.0,
            "SINR": 12.0,
            "THROUGHPUT_5P": 25.0,
            "LOAD_IMBALANCE": 0.15,
            "RX_COVERAGE_RATIO": 0.85
        })
        
        # Build comprehensive prompt for LLM
        prompt = f"""You are optimizing a 6G cellular network configuration.

TARGET INTENT:
- Area: {intent.get('target_area', 'general')}
- Target KPIs: {', '.join(intent.get('target_kpis', []))}
- Priority: {priority}

{constraints_text}

CURRENT CONFIGURATION:
{_format_config(current_config)}

CURRENT KPIs (Baseline):
{_format_kpis(current_kpis)}

YOUR TASK:
1. Analyze the intent and current state
2. Design optimal configuration changes
3. Predict expected KPI improvements
4. Provide detailed technical reasoning

REQUIREMENTS:
- Apply your expert knowledge of radio propagation
- Use the optimization strategies and KPI prediction guidelines from your instructions
- Stay within guardrails (±3dB power, ±2° elevation, ±10° azimuth)
- Predict realistic KPI values based on your changes
- Ensure constraints are satisfied

Generate the complete OptimizationResult JSON with:
- result_id: "{result_id}"
- priority: "{priority}"
- test_name: "optimization_{intent.get('target_area', 'general')}"
- input: (copy the intent below)
- output: (your configuration changes and predictions)
- reasoning: (detailed technical explanation)

Intent to copy to input field:
{json.dumps(intent, indent=2)}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanations."""

        # Run LLM agent
        print("🧠 LLM is analyzing and generating optimal configuration...")
        print(f"📍 Target: {intent.get('target_area')}")
        print(f"🎯 KPIs: {', '.join(intent.get('target_kpis', []))}")
        print(f"⚡ Priority: {priority}")
        
        response = optimization_agent.run(prompt, stream=False)
        
        # Parse JSON response
        import json as json_module
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Clean markdown if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result_dict = json_module.loads(response_text)
        
        # Convert to Pydantic model
        result = OptimizationResult(**result_dict)
        
        print("✅ LLM optimization completed!")
        print(f"📊 Configuration changes: {len(result.output.changes)} parameters")
        print(f"✓ Constraints satisfied: {result.output.constraints_satisfied}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def _format_config(config: Dict) -> str:
    """Format configuration dict for display."""
    if not config:
        return "Default baseline configuration"
    
    lines = []
    for i in range(4):
        tx_on = config.get(f'tx{i}_on', True)
        tx_p = config.get(f'tx{i}_P_dBm', 30.0)
        tx_az = config.get(f'tx{i}_dAz', 0.0)
        tx_el = config.get(f'tx{i}_dEl', 0.0)
        
        status = "ON" if tx_on else "OFF"
        lines.append(f"TX{i}: {status} | Power: {tx_p:.1f} dBm | Azimuth: {tx_az:+.1f}° | Elevation: {tx_el:+.1f}°")
    
    return "\n".join(lines)


def _format_kpis(kpis: Dict) -> str:
    """Format KPI dict for display."""
    if not kpis:
        return "No KPI data available"
    
    lines = []
    if "RX_POWER" in kpis:
        lines.append(f"RX_POWER: {kpis['RX_POWER']:.2f} dBm")
    if "SINR" in kpis:
        lines.append(f"SINR: {kpis['SINR']:.2f} dB")
    if "THROUGHPUT_5P" in kpis:
        lines.append(f"THROUGHPUT: {kpis['THROUGHPUT_5P']:.2f} Mbps")
    if "LOAD_IMBALANCE" in kpis:
        lines.append(f"LOAD_IMBALANCE: {kpis['LOAD_IMBALANCE']:.3f}")
    if "RX_COVERAGE_RATIO" in kpis:
        lines.append(f"COVERAGE_RATIO: {kpis['RX_COVERAGE_RATIO']:.3f}")
    
    return "\n".join(lines)


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================

if __name__ == "__main__":
    # Test with sample intent
    test_intent = {
        "target_area": "cell1",
        "target_kpis": ["RX_POWER"],
        "kpi_thresholds": [
            {"kpi": "RX_POWER", "operator": "GTE", "value": -70.0}
        ],
        "priority": "HIGH",
        "current_config": {
            'tx0_on': True, 'tx0_P_dBm': 30.0, 'tx0_dAz': 0.0, 'tx0_dEl': 0.0,
            'tx1_on': True, 'tx1_P_dBm': 30.0, 'tx1_dAz': 0.0, 'tx1_dEl': 0.0,
            'tx2_on': True, 'tx2_P_dBm': 30.0, 'tx2_dAz': 0.0, 'tx2_dEl': 0.0,
            'tx3_on': True, 'tx3_P_dBm': 30.0, 'tx3_dAz': 0.0, 'tx3_dEl': 0.0
        }
    }
    
    result = run_optimization(test_intent)
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(f"Passed: {result.passed}")
    print(f"Changes: {len(result.output.changes)}")
    print(f"Expected KPIs: {result.output.expected_kpis}")
    print(f"Reasoning: {result.reasoning}")
