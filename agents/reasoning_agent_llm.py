#!/usr/bin/env python3
"""
Reasoning Agent (Full LLM-Based)
Synthesizes inputs from Optimization, Conflict Detection, and Resolution agents
into a cohesive strategic narrative.
"""
from __future__ import annotations

import json
from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq

load_dotenv()

# ============================================================================
# RESPONSE SCHEMA
# ============================================================================

class StrategicReport(BaseModel):
    """Final strategic reasoning report combining all workflow steps."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    decision_headline: str = Field(..., description="One sentence summary of the final action (e.g. 'Priority Override Applied for Kartal Coverage').")
    executive_summary: str = Field(..., description="A coherent paragraph explaining WHAT was done and WHY, combining technical and strategic reasons.")
    final_technical_state: str = Field(..., description="Summary of the final configuration state (power levels, active TXs).")
    expected_impact: str = Field(..., description="Predicted impact on network KPIs.")
    process_narrative: str = Field(..., description="Step-by-step reasoning: 'First optimized for X, then conflict Y detected, resolved by Z'.")

# ============================================================================
# AGENT DEFINITION
# ============================================================================

INSTRUCTIONS = [
    "You are the Strategic Reasoning Agent for a 6G Network Automation System.",
    "Your task is to CONCATENATE and SYNTHESIZE outputs from previous agents into a final report.",
    "",
    "## 📥 INPUTS YOU WILL RECEIVE:",
    "1. **User Intent**: What the user originally wanted.",
    "2. **Optimization Reasoning**: Why the technical configuration was chosen.",
    "3. **Conflict Detection**: Whether conflicts occurred and why.",
    "4. **Resolution Reasoning**: How conflicts were resolved (Priority vs Merge).",
    "",
    "## 🧠 YOUR JOB:",
    "Weave these inputs into a story (The Process Narrative).",
    "- If Conflict Detected: Explain clearly what conflicted (e.g. 'Optimization wanted +3dBm but Safety Rule limited it').",
    "- Explain how the Resolution Agent fixed it.",
    "",
    "## 📝 OUTPUT STYLE:",
    "- Professional, Executive, Concise.",
    "- Explain the 'Why' behind the final decision.",
]

reasoning_agent = Agent(
    name="ReasoningAgent",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=INSTRUCTIONS,
    output_schema=StrategicReport,
    markdown=False,
    structured_outputs=True,
)

# ============================================================================
# RUN FUNCTION
# ============================================================================

def run_reasoning_agent(
    intent_text: str,
    optimization_result: Dict[str, Any],
    conflict_result: Dict[str, Any],
    resolution_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs the reasoning agent to synthesize the final report.
    """
    print("\n" + "="*70)
    print("🧠 REASONING AGENT - Synthesizing Final Report...")
    print("="*70)

    # 1. Extract Data safely
    opt_reasoning = optimization_result.get('reasoning', 'Optimization completed.')
    
    conf_reasoning = conflict_result.get('reasoning', 'No specific conflict reasoning.')
    is_conflicted = conflict_result.get('is_conflicted', False)
    conf_summary = conflict_result.get('conflict_summary', 'No conflicts.')
    
    res_reasoning = resolution_result.get('reasoning', 'No resolution details.')
    strategy = resolution_result.get('resolution_strategy', 'NONE')
    
    # Extract KPIs based on strategy
    final_kpis = {}
    if strategy == "PRIORITY":
        final_kpis = resolution_result.get('winning_config', {}).get('output', {}).get('expected_kpis', {})
    elif strategy == "WEIGHTED_MERGE":
        final_kpis = resolution_result.get('merged_config', {}).get('output', {}).get('expected_kpis', {})
    else:
        final_kpis = optimization_result.get('output', {}).get('expected_kpis', {})

    # 2. Create Prompt
    prompt = f"""
    Generate a Strategic Report based on this workflow:

    ORIGINAL INTENT: "{intent_text}"

    STEP 1: OPTIMIZATION AGENT:
    "{opt_reasoning}"

    STEP 2: CONFLICT DETECTION AGENT:
    Status: {"Conflicted" if is_conflicted else "Clean"}
    Summary: "{conf_summary}"
    Reasoning: "{conf_reasoning}"

    STEP 3: RESOLUTION AGENT ({strategy}):
    "{res_reasoning}"

    FINAL KPI PREDICTIONS:
    {json.dumps(final_kpis, indent=2)}
    """

    # 3. Run Agent
    try:
        response = reasoning_agent.run(prompt, stream=False)
        
        if hasattr(response, 'content') and not isinstance(response.content, str):
            return response.content.model_dump()
        else:
            content = str(response.content)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)

    except Exception as e:
        print(f"❌ Reasoning Agent Failed: {e}")
        return {
            "decision_headline": "Optimization Completed",
            "executive_summary": "System processed request (Reasoning generation failed).",
            "final_technical_state": "N/A",
            "expected_impact": "N/A",
            "process_narrative": f"Error: {str(e)}"
        }