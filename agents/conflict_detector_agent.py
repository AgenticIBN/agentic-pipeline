#!/usr/bin/env python3
"""
Conflict Detector Agent (Agno-based)
Detects conflicts between optimization results using Agno framework.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq

# Import core conflict detection logic
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from conflict_detector_agent import detect_conflicts as core_detect_conflicts, ConflictReport as CoreConflictReport

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
    resolution_recommendation: Optional[str] = Field(None, description="Recommended resolution strategy")
    conflicting_result_ids: List[str] = Field(default_factory=list, description="IDs of conflicting results")
    reasoning: Optional[str] = Field(None, description="Reasoning behind conflict detection and recommendations")


# ============================================================================
# AGENT INSTRUCTIONS
# ============================================================================

INSTRUCTIONS = [
    "You are an expert conflict detector for cellular/mobile network optimization.",
    "Your task: Analyze optimization results to detect conflicts and recommend resolution strategies.",
    "",
    "## Conflict Types:",
    "1. PARAMETER_CONFLICT:",
    "   - Same parameter modified in opposite directions",
    "   - Example: Intent A increases tx0_P_dBm by +3dB, Intent B decreases by -2dB",
    "   - Usually HIGH or CRITICAL severity",
    "",
    "2. RESOURCE_CONTENTION:",
    "   - Same parameter modified in same direction but different magnitudes",
    "   - Example: Both intents increase power but by different amounts",
    "   - Usually MEDIUM severity",
    "",
    "3. BASE_STATION_CONFLICT:",
    "   - Same base station (transmitter) affected by different parameter changes",
    "   - Example: Intent A changes tx0 power, Intent B changes tx0 angles",
    "   - Severity depends on parameter interaction",
    "",
    "4. BOOLEAN_CONFLICT:",
    "   - Conflicting ON/OFF states for transmitters",
    "   - Example: One intent turns tx0 ON, another turns it OFF",
    "   - Usually CRITICAL severity",
    "",
    "## Severity Assessment:",
    "- CRITICAL: Boolean conflicts, opposite direction changes >5 units",
    "- HIGH: Opposite direction changes 2-5 units, large magnitude differences",
    "- MEDIUM: Small opposite changes, moderate contention",
    "- LOW: Minor contention, same direction with small differences",
    "",
    "## Resolution Recommendations:",
    "1. For CRITICAL/HIGH severity conflicts:",
    "   - Recommend PRIORITY strategy (select highest priority intent)",
    "   - Explain which intent should take precedence and why",
    "",
    "2. For MEDIUM/LOW severity conflicts:",
    "   - Recommend WEIGHTED_MERGE strategy (merge based on priority weights)",
    "   - Explain how weighted merge can balance competing objectives",
    "",
    "3. For no conflicts:",
    "   - Recommend NO_CONFLICT (apply new intent safely)",
    "   - Confirm all intents can coexist",
    "",
    "## Reasoning Guidelines:",
    "- Explain WHAT conflicts were detected and WHY they matter",
    "- Describe potential impact on network performance",
    "- Justify recommended resolution strategy",
    "- Consider priority levels of conflicting intents",
    "- Be concise but informative (2-4 sentences)",
    "",
    "## Important:",
    "- You receive pre-computed conflict details from core detection engine",
    "- Your role is to validate, contextualize, and provide reasoning",
    "- Focus on explaining the implications and recommendations",
    "- Consider operational impact when assessing severity",
]


# ============================================================================
# AGENT DEFINITION
# ============================================================================

conflict_detector_agent = Agent(
    name="ConflictDetectorAgent",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=INSTRUCTIONS,
    markdown=False,
)


# ============================================================================
# HELPER FUNCTION TO RUN CONFLICT DETECTION
# ============================================================================

def run_conflict_detection(new_result: dict, active_results: list) -> ConflictAnalysis:
    """
    Run conflict detection between new result and active results.
    
    Args:
        new_result: New optimization result to check
        active_results: List of currently active optimization results
    
    Returns:
        ConflictAnalysis with detected conflicts and reasoning
    """
    print("\n" + "="*70)
    print("🤖 CONFLICT DETECTOR AGENT - Starting...")
    print("="*70)
    
    try:
        print(f"📊 Checking new result against {len(active_results)} active result(s)")
        
        # Run core conflict detection
        print("🔍 Running core conflict detection algorithm...")
        core_report = core_detect_conflicts(
            new_result=new_result,
            active_results=active_results
        )
        print(f"✅ Core detection completed: {core_report.num_conflicts} conflict(s) found")
    
        # Convert to dict for processing
        report_dict = core_report.model_dump()
        
        # Prepare prompt for LLM to generate reasoning
        print("🧠 Generating conflict analysis reasoning with LLM...")
        prompt = f"""Analyze this conflict detection result and provide clear reasoning:

NEW RESULT:
- Result ID: {new_result.get('result_id', 'N/A')}
- Priority: {new_result.get('priority', new_result.get('input', {}).get('priority', 'MEDIUM'))}
- Changes: {len(new_result.get('output', {}).get('changes', []))} parameter(s)

ACTIVE RESULTS: {len(active_results)} intent(s)

CONFLICT DETECTION:
- Conflicts Found: {report_dict['is_conflicted']}
- Number of Conflicts: {report_dict['num_conflicts']}
- Summary: {report_dict['conflict_summary']}

{_format_conflict_details(report_dict.get('details', []))}

Provide the complete ConflictAnalysis with reasoning that explains:
1. What conflicts were detected and their significance
2. Potential impact on network performance
3. Why the recommended resolution strategy is appropriate

Include all fields from the detection result with added reasoning."""

        # Run agent to generate reasoning
        response = conflict_detector_agent.run(prompt, stream=False)
        print("✅ LLM reasoning generated")
        
        # Extract reasoning from LLM response
        reasoning_text = ""
        if hasattr(response, 'content'):
            reasoning_text = str(response.content)
        elif isinstance(response, str):
            reasoning_text = response
        else:
            reasoning_text = "Conflict detection completed. See details for specific conflicts."
        
        # Return core report with LLM reasoning
        print("\n✅ CONFLICT DETECTOR AGENT - Completed Successfully!")
        print("="*70)
        return ConflictAnalysis(
            **report_dict,
            reasoning=reasoning_text
        )
    
    except Exception as e:
        print("\n❌ CONFLICT DETECTOR AGENT - ERROR!")
        print("="*70)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        import traceback
        print("\nFull Traceback:")
        traceback.print_exc()
        print("="*70)
        raise RuntimeError(f"ConflictDetectorAgent failed: {str(e)}") from e


def _format_conflict_details(details: List[dict]) -> str:
    """Format conflict details for display."""
    if not details:
        return "No conflict details available."
    
    lines = ["CONFLICT DETAILS:"]
    for i, detail in enumerate(details[:5], 1):  # Show first 5
        lines.append(f"  {i}. {detail.get('conflict_type')} (Severity: {detail.get('severity')})")
        lines.append(f"     Parameter: {detail.get('parameter')}")
        lines.append(f"     {detail.get('intent1_id')}: {detail.get('intent1_change')}")
        lines.append(f"     {detail.get('intent2_id')}: {detail.get('intent2_change')}")
        lines.append(f"     {detail.get('description')}")
        lines.append("")
    
    if len(details) > 5:
        lines.append(f"  ... and {len(details) - 5} more conflict(s)")
    
    return '\n'.join(lines)


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    new_result = {
        "result_id": "opt_20260207_120000",
        "priority": "HIGH",
        "output": {
            "changes": [
                {"param": "tx0_P_dBm", "before": 30.0, "change": 3.0, "unit": "dBm"}
            ]
        }
    }
    
    active_results = [
        {
            "result_id": "opt_20260207_100000",
            "priority": "MEDIUM",
            "output": {
                "changes": [
                    {"param": "tx0_P_dBm", "before": 30.0, "change": -2.0, "unit": "dBm"}
                ]
            }
        }
    ]
    
    analysis = run_conflict_detection(new_result, active_results)
    
    print(f"\n🔍 Conflict Detection Result:")
    print(f"   Conflicted: {analysis.is_conflicted}")
    print(f"   Conflicts: {analysis.num_conflicts}")
    print(f"   Summary: {analysis.conflict_summary}")
    if analysis.reasoning:
        print(f"\n📝 Reasoning:\n{analysis.reasoning}")
