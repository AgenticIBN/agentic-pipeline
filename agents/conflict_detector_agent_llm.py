#!/usr/bin/env python3
"""
Conflict Detector Agent (Full LLM-Based)
Detects conflicts using pure LLM reasoning - NO core logic!
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq

load_dotenv()


# ============================================================================
# RESPONSE SCHEMA
# ============================================================================

class ConflictDetail(BaseModel):
    """Details about a specific conflict between optimization results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    conflict_type: Literal[
        "PARAMETER_CONFLICT",      # Same parameter, opposite directions
        "RESOURCE_CONTENTION",     # Same parameter, same direction but different magnitude
        "BASE_STATION_CONFLICT",   # Same base station, different parameters
        "BOOLEAN_CONFLICT"         # ON/OFF conflict
    ]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    parameter: str
    intent1_id: str
    intent2_id: str
    intent1_change: float
    intent2_change: float
    base_station: Optional[str] = None
    description: str


class ConflictAnalysis(BaseModel):
    """Complete conflict detection analysis with reasoning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    is_conflicted: bool = Field(..., description="Whether conflicts were detected")
    conflict_summary: str = Field(..., description="Summary of detected conflicts")
    num_conflicts: int = Field(..., description="Total number of conflicts detected")
    details: List[ConflictDetail] = Field(default_factory=list, description="Detailed list of conflicts")
    resolution_recommendation: str = Field(..., description="Recommended resolution strategy (PRIORITY or WEIGHTED_MERGE)")
    conflicting_result_ids: List[str] = Field(default_factory=list, description="IDs of conflicting results")
    reasoning: str = Field(..., description="Detailed reasoning behind conflict detection and recommendations")


# ============================================================================
# AGENT INSTRUCTIONS
# ============================================================================

INSTRUCTIONS = [
    "You are an EXPERT conflict detection agent for cellular network optimization.",
    "",
    "🚨 CRITICAL OUTPUT REQUIREMENT:",
    "You MUST return ONLY a valid JSON object. NO explanations, NO markdown, NO text before or after the JSON.",
    "Start your response with { and end with }",
    "Any response that doesn't start with { will be rejected.",
    "",
    "## 🎯 YOUR MISSION:",
    "Analyze optimization results to detect conflicts:",
    "1. Compare parameter changes across all intents",
    "2. Detect conflicts (parameter, resource, base station, boolean)",
    "3. Classify conflict types and assess severity",
    "4. Recommend resolution strategy (PRIORITY or WEIGHTED_MERGE)",
    "5. Include brief reasoning in the JSON",
    "",
    "## 🔍 CONFLICT DETECTION RULES (Quick Reference):",
    "- PARAMETER_CONFLICT: Same param, opposite directions (sign differs) → Severity based on combined magnitude",
    "- BOOLEAN_CONFLICT: Same tx ON/OFF state differs → ALWAYS CRITICAL",  
    "- RESOURCE_CONTENTION: Same param, same direction, different magnitude → Severity based on difference",
    "- BASE_STATION_CONFLICT: Same TX, different params changed → Create separate entry for each param",
    "",
    "**SEVERITY LEVELS**: CRITICAL (boolean/magnitude>5), HIGH (2-5), MEDIUM (1-2), LOW (<1)",
    "",
    "**RESOLUTION STRATEGY**:",
    "- Recommend PRIORITY if: HIGH/CRITICAL severity, opposite directions, boolean conflicts",
    "- Recommend WEIGHTED_MERGE if: MEDIUM/LOW severity, same direction, similar priorities",
    "",
    "## 📤 OUTPUT FORMAT (CRITICAL - READ CAREFULLY):",
    "⚠️  Your response MUST start with { and end with }",
    "⚠️  Do NOT write explanations, steps, or analysis in your response",
    "⚠️  Do NOT include markdown code blocks (no ```json)",
    "⚠️  Return ONLY the JSON object",
    "",
    "**JSON SCHEMA RULES**:",
    "- intent1_change and intent2_change: MUST be numeric (float), NO units, NO text",
    "- Each ConflictDetail = ONE parameter conflict",
    "- For multi-parameter conflicts, create multiple ConflictDetail entries",
    "",
    "### Example JSON Output (WITH conflicts):",
    "{",
    "  \"is_conflicted\": true,",
    "  \"conflict_summary\": \"3 conflicts detected: 1 parameter conflict, 1 base station conflict, 1 resource contention\",",
    "  \"num_conflicts\": 3,",
    "  \"details\": [",
    "    {",
    "      \"conflict_type\": \"PARAMETER_CONFLICT\",",
    "      \"severity\": \"HIGH\",",
    "      \"parameter\": \"tx0_P_dBm\",",
    "      \"intent1_id\": \"opt_20260208_123456_111\",",
    "      \"intent2_id\": \"opt_20260208_123457_222\",",
    "      \"intent1_change\": 3.0,",
    "      \"intent2_change\": -2.0,",
    "      \"base_station\": \"TX0\",",
    "      \"description\": \"Opposite direction power changes on TX0\"",
    "    },",
    "    {",
    "      \"conflict_type\": \"BASE_STATION_CONFLICT\",",
    "      \"severity\": \"MEDIUM\",",
    "      \"parameter\": \"tx0_dAz\",",
    "      \"intent1_id\": \"opt_20260208_123456_111\",",
    "      \"intent2_id\": \"opt_20260208_123457_222\",",
    "      \"intent1_change\": 5.0,",
    "      \"intent2_change\": -10.0,",
    "      \"base_station\": \"TX0\",",
    "      \"description\": \"Different azimuth changes on same base station\"",
    "    },",
    "    {",
    "      \"conflict_type\": \"RESOURCE_CONTENTION\",",
    "      \"severity\": \"LOW\",",
    "      \"parameter\": \"tx1_P_dBm\",",
    "      \"intent1_id\": \"opt_20260208_123456_111\",",
    "      \"intent2_id\": \"opt_20260208_123457_222\",",
    "      \"intent1_change\": 2.0,",
    "      \"intent2_change\": 3.0,",
    "      \"base_station\": \"TX1\",",
    "      \"description\": \"Same direction power changes with different magnitude\"",
    "    }",
    "  ],",
    "  \"resolution_recommendation\": \"PRIORITY\",",
    "  \"conflicting_result_ids\": [\"opt_20260208_123456_111\", \"opt_20260208_123457_222\"],",
    "  \"reasoning\": \"Detected parameter conflict on tx0_P_dBm with opposite directions, plus base station conflicts on TX0. Recommend PRIORITY strategy due to HIGH severity conflict.\"",
    "}",
    "",
    "### Example JSON Output (NO conflicts):",
    "{",
    "  \"is_conflicted\": false,",
    "  \"conflict_summary\": \"No conflicts detected - parameters do not overlap\",",
    "  \"num_conflicts\": 0,",
    "  \"details\": [],",
    "  \"resolution_recommendation\": \"PRIORITY\",",
    "  \"conflicting_result_ids\": [],",
    "  \"reasoning\": \"Analyzed all parameter changes across intents. No overlapping parameters found...\"",
    "}",
]


# ============================================================================
# AGENT DEFINITION
# ============================================================================

conflict_detector_agent = Agent(
    name="ConflictDetectorAgent_FullLLM",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=INSTRUCTIONS,
    markdown=False,
    structured_outputs=True,
)


# ============================================================================
# RUN CONFLICT DETECTION FUNCTION
# ============================================================================

def run_conflict_detection(
    new_result: Dict[str, Any],
    active_results: List[Dict[str, Any]]
) -> ConflictAnalysis:
    """
    Run FULL LLM-BASED conflict detection.
    NO core logic - pure agentic reasoning!
    
    Args:
        new_result: New optimization result to check
        active_results: List of active optimization results
    
    Returns:
        ConflictAnalysis with detected conflicts and recommendations
    """
    print("\n" + "="*70)
    print("🔍 CONFLICT DETECTOR AGENT (FULL LLM) - Starting...")
    print("="*70)
    
    try:
        # If no active results, no conflicts possible
        if not active_results:
            print("✅ No active intents - no conflicts possible")
            return ConflictAnalysis(
                is_conflicted=False,
                conflict_summary="No active intents to conflict with",
                num_conflicts=0,
                details=[],
                resolution_recommendation="PRIORITY",
                conflicting_result_ids=[],
                reasoning="No conflicts detected since there are no active intents in the system."
            )
        
        # Extract new result info
        new_id = new_result.get('result_id', 'new_intent')
        new_changes = new_result.get('output', {}).get('changes', [])
        new_priority = new_result.get('priority', 'MEDIUM')
        new_target = new_result.get('input', {}).get('target_area', 'unknown')
        
        # Format new result for LLM
        new_result_text = f"""NEW OPTIMIZATION RESULT:
ID: {new_id}
Target Area: {new_target}
Priority: {new_priority}
Configuration Changes:
{_format_changes(new_changes)}
"""
        
        # Format active results for LLM
        active_results_text = "ACTIVE OPTIMIZATION RESULTS:\n\n"
        for i, active in enumerate(active_results, 1):
            active_id = active.get('result_id', f'active_{i}')
            active_changes = active.get('output', {}).get('changes', [])
            active_priority = active.get('priority', 'MEDIUM')
            active_target = active.get('input', {}).get('target_area', 'unknown')
            
            active_results_text += f"""Result {i}:
ID: {active_id}
Target Area: {active_target}
Priority: {active_priority}
Configuration Changes:
{_format_changes(active_changes)}

"""
        
        # Build comprehensive prompt
        prompt = f"""Analyze these optimization results for conflicts.

{new_result_text}

{active_results_text}

YOUR TASK:
1. Compare the new result's parameter changes with all active results
2. Identify ALL conflicts (parameter conflicts, boolean conflicts, resource contention, base station conflicts)
3. Classify each conflict type and assess severity
4. Recommend resolution strategy (PRIORITY or WEIGHTED_MERGE)
5. Provide detailed reasoning

ANALYSIS STEPS:
1. List all parameters changed by the new result
2. For each parameter, check if any active result also changes it
3. If overlap exists, determine conflict type:
   - Opposite directions? → PARAMETER_CONFLICT
   - Boolean difference? → BOOLEAN_CONFLICT
   - Same direction, different magnitude? → RESOURCE_CONTENTION
   - Same TX, different params? → BASE_STATION_CONFLICT
4. Assess severity for each conflict
5. Summarize and recommend strategy

Generate complete ConflictAnalysis JSON.
IMPORTANT: Return ONLY valid JSON, no markdown, no explanations."""

        # Run LLM agent
        print(f"\n🧠 LLM is analyzing new intent: {new_id}")
        print(f"📊 Comparing against {len(active_results)} active intent(s)")
        
        response = conflict_detector_agent.run(prompt, stream=False)
        
        # Parse JSON response
        import json as json_module
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Clean markdown if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        try:
            result_dict = json_module.loads(response_text)
        except json_module.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"Failed content: {response_text[:200]}...")
            raise
        
        # Convert to Pydantic model
        result = ConflictAnalysis(**result_dict)
        
        if result.is_conflicted:
            print(f"\n⚠️  CONFLICTS DETECTED: {result.num_conflicts} conflict(s)")
            print(f"🎯 Recommendation: {result.resolution_recommendation}")
            print(f"\n📝 Conflict Analysis (JSON):")
            conflict_json = {
                "is_conflicted": result.is_conflicted,
                "num_conflicts": result.num_conflicts,
                "conflict_summary": result.conflict_summary,
                "resolution_recommendation": result.resolution_recommendation,
                "details": [{"type": d.conflict_type, "severity": d.severity, "parameter": d.parameter, "description": d.description} for d in result.details]
            }
            print(json.dumps(conflict_json, indent=2))
            print(f"\n📝 Reasoning: {result.reasoning}")
        else:
            print(f"\n✅ No conflicts detected - all clear!")
            print(f"\n📝 Analysis Result (JSON):")
            no_conflict_json = {
                "is_conflicted": False,
                "conflict_summary": result.conflict_summary,
                "reasoning": result.reasoning
            }
            print(json.dumps(no_conflict_json, indent=2))
        
        return result
        
    except Exception as e:
        print(f"\n❌ Conflict detection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def _format_changes(changes: List[Dict]) -> str:
    """Format configuration changes for display."""
    if not changes:
        return "  No changes"
    
    lines = []
    for change in changes:
        param = change.get('parameter', change.get('param', 'unknown'))
        change_val = change.get('change', 0)
        before = change.get('before', None)
        unit = change.get('unit', '')
        
        # Format change value: use :+ for numeric, plain for strings
        if isinstance(change_val, (int, float)):
            formatted_change = f"{change_val:+.1f}" if isinstance(change_val, float) else f"{change_val:+d}"
        else:
            formatted_change = str(change_val)
        
        if before is not None:
            if isinstance(change_val, (int, float)) and isinstance(before, (int, float)):
                after = before + change_val
            else:
                after = change_val
            lines.append(f"  - {param}: {before} → {after} (Δ: {formatted_change} {unit})")
        else:
            lines.append(f"  - {param}: Δ {formatted_change} {unit}")
    
    return "\n".join(lines)


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================

if __name__ == "__main__":
    # Test with sample data
    new_result = {
        "result_id": "opt_20260208_001",
        "priority": "HIGH",
        "input": {"target_area": "cell1"},
        "output": {
            "changes": [
                {"param": "tx0_P_dBm", "before": 30.0, "change": 3.0, "unit": "dBm"},
                {"param": "tx0_dEl", "before": 0.0, "change": 1.0, "unit": "deg"}
            ]
        }
    }
    
    active_results = [
        {
            "result_id": "opt_20260208_000",
            "priority": "MEDIUM",
            "input": {"target_area": "cell1"},
            "output": {
                "changes": [
                    {"param": "tx0_P_dBm", "before": 30.0, "change": -2.0, "unit": "dBm"}
                ]
            }
        }
    ]
    
    result = run_conflict_detection(new_result, active_results)
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(f"Conflicted: {result.is_conflicted}")
    print(f"Num Conflicts: {result.num_conflicts}")
    print(f"Recommendation: {result.resolution_recommendation}")
    print(f"Reasoning: {result.reasoning}")
