#!/usr/bin/env python3
"""
Priority Based Resolution Agent (Agno-based)
Resolves conflicts by selecting highest priority intent using Agno framework.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq

# Import core resolution logic
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from priority_based_resolution_agent import (
    resolve_by_priority as core_resolve_by_priority,
    PriorityBasedResolutionOutput as CoreResolutionOutput
)

load_dotenv()


# ============================================================================
# RESPONSE SCHEMA
# ============================================================================

class PriorityResolutionResult(BaseModel):
    """Result of priority-based conflict resolution with reasoning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    conflict_detected: bool = Field(..., description="Whether conflict was detected")
    resolution_strategy: str = Field(..., description="Resolution strategy used")
    winning_result_id: Optional[str] = Field(None, description="ID of selected result")
    winning_priority: Optional[str] = Field(None, description="Priority of winning result")
    winning_config: Optional[Dict] = Field(None, description="Full winning optimization result")
    rejected_result_ids: List[str] = Field(default_factory=list, description="IDs of rejected results")
    resolution_notes: str = Field(..., description="Notes about resolution decision")
    conflict_summary: Optional[str] = Field(None, description="Summary of original conflict")
    reasoning: Optional[str] = Field(None, description="Detailed reasoning for resolution decision")


# ============================================================================
# AGENT INSTRUCTIONS
# ============================================================================

INSTRUCTIONS = [
    "You are an expert conflict resolution agent using PRIORITY-based strategy.",
    "Your task: Resolve conflicts by selecting the result with highest priority.",
    "",
    "## Priority Levels (highest to lowest):",
    "1. CRITICAL - Emergency situations, critical failures",
    "2. HIGH - Important operational needs, significant issues",
    "3. MEDIUM - Standard optimization requests",
    "4. LOW - Optional improvements, nice-to-have changes",
    "",
    "## Resolution Strategy:",
    "1. Identify all conflicting results with their priorities",
    "2. Select result with HIGHEST priority",
    "3. If priorities are equal:",
    "   - Prefer ACTIVE results (already applied to network)",
    "   - This minimizes disruption to existing configuration",
    "4. Reject all other conflicting results",
    "",
    "## When to Use Priority Strategy:",
    "- HIGH or CRITICAL severity conflicts",
    "- Boolean conflicts (ON/OFF states)",
    "- Opposite direction parameter changes",
    "- Clear priority differences between intents",
    "- When one intent must take full precedence",
    "",
    "## Reasoning Guidelines:",
    "- Explain WHY the winning result was selected",
    "- Describe what makes it higher priority than alternatives",
    "- Note any important rejected results and implications",
    "- Discuss operational impact of the decision",
    "- Consider network stability and safety",
    "- Be concise but informative (2-4 sentences)",
    "",
    "## Important:",
    "- You receive pre-computed resolution from core engine",
    "- Your role is to validate and provide reasoning/context",
    "- Focus on explaining the decision rationale",
    "- Consider broader operational context",
]


# ============================================================================
# AGENT DEFINITION
# ============================================================================

priority_resolution_agent = Agent(
    name="PriorityResolutionAgent",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=INSTRUCTIONS,
    markdown=False,
)


# ============================================================================
# HELPER FUNCTION TO RUN PRIORITY RESOLUTION
# ============================================================================

def run_priority_resolution(
    conflict_report: dict,
    new_result: dict,
    active_results: list
) -> PriorityResolutionResult:
    """
    Run priority-based conflict resolution.
    
    Args:
        conflict_report: Conflict detection report dict
        new_result: New optimization result
        active_results: List of active optimization results
    
    Returns:
        PriorityResolutionResult with resolution decision and reasoning
    """
    print("\n" + "="*70)
    print("🤖 PRIORITY RESOLUTION AGENT - Starting...")
    print("="*70)
    
    try:
        print(f"🎯 Resolving conflicts using PRIORITY strategy")
        print(f"📊 New result vs {len(active_results)} active result(s)")
        
        # Run core priority resolution
        print("🔍 Running core priority resolution algorithm...")
        core_result = core_resolve_by_priority(
            conflict_report=conflict_report,
            new_result=new_result,
            active_results=active_results
        )
        print(f"✅ Core resolution completed: Winner selected")
    
        # Convert to dict for processing
        result_dict = core_result.model_dump()
        
        # Prepare prompt for LLM to generate reasoning
        print("🧠 Generating resolution reasoning with LLM...")
        prompt = f"""Analyze this priority-based resolution result and provide clear reasoning:

CONFLICT DETECTED: {result_dict['conflict_detected']}

NEW RESULT:
- Result ID: {new_result.get('result_id', 'N/A')}
- Priority: {new_result.get('priority', new_result.get('input', {}).get('priority', 'MEDIUM'))}

ACTIVE RESULTS: {len(active_results)} intent(s)
{_format_active_results(active_results)}

RESOLUTION DECISION:
- Strategy: {result_dict['resolution_strategy']}
- Winner: {result_dict['winning_result_id']} (Priority: {result_dict['winning_priority']})
- Rejected: {len(result_dict['rejected_result_ids'])} result(s)
- Notes: {result_dict['resolution_notes']}

Provide the complete PriorityResolutionResult with reasoning that explains:
1. Why the winning result was selected over others
2. What makes its priority justify overriding other intents
3. Operational impact of rejecting other results

Include all fields from the resolution result with added reasoning."""

        # Run agent to generate reasoning
        response = priority_resolution_agent.run(prompt, stream=False)
        print("✅ LLM reasoning generated")
        
        # Extract reasoning from LLM response
        reasoning_text = ""
        if hasattr(response, 'content'):
            reasoning_text = str(response.content)
        elif isinstance(response, str):
            reasoning_text = response
        else:
            reasoning_text = "Priority-based resolution completed. Highest priority result selected."
        
        # Return core result with LLM reasoning
        print("\n✅ PRIORITY RESOLUTION AGENT - Completed Successfully!")
        print("="*70)
        return PriorityResolutionResult(
            **result_dict,
            reasoning=reasoning_text
        )
    
    except Exception as e:
        print("\n❌ PRIORITY RESOLUTION AGENT - ERROR!")
        print("="*70)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        import traceback
        print("\nFull Traceback:")
        traceback.print_exc()
        print("="*70)
        raise RuntimeError(f"PriorityResolutionAgent failed: {str(e)}") from e


def _format_active_results(active_results: List[dict]) -> str:
    """Format active results for display."""
    if not active_results:
        return "  (none)"
    
    lines = []
    for i, result in enumerate(active_results[:3], 1):  # Show first 3
        result_id = result.get('result_id', 'N/A')
        priority = result.get('priority', result.get('input', {}).get('priority', 'MEDIUM'))
        lines.append(f"  {i}. {result_id} (Priority: {priority})")
    
    if len(active_results) > 3:
        lines.append(f"  ... and {len(active_results) - 3} more")
    
    return '\n'.join(lines) if lines else "  (none)"


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    conflict_report = {
        "is_conflicted": True,
        "conflict_summary": "Parameter conflict detected",
        "num_conflicts": 1
    }
    
    new_result = {
        "result_id": "opt_20260207_120000",
        "priority": "HIGH",
        "output": {"changes": []}
    }
    
    active_results = [
        {
            "result_id": "opt_20260207_100000",
            "priority": "MEDIUM",
            "output": {"changes": []}
        }
    ]
    
    result = run_priority_resolution(conflict_report, new_result, active_results)
    
    print(f"\n🎯 Priority Resolution Result:")
    print(f"   Winner: {result.winning_result_id}")
    print(f"   Priority: {result.winning_priority}")
    print(f"   Rejected: {len(result.rejected_result_ids)}")
    if result.reasoning:
        print(f"\n📝 Reasoning:\n{result.reasoning}")
