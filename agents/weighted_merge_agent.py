#!/usr/bin/env python3
"""
Weighted Merge Resolution Agent (Agno-based)
Resolves conflicts by merging configurations using priority-based weights via Agno framework.
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
from weighted_merge_resolution_agent import (
    resolve_by_weighted_merge as core_resolve_by_weighted_merge,
    WeightedMergeOutput as CoreMergeOutput
)

load_dotenv()


# ============================================================================
# RESPONSE SCHEMA
# ============================================================================

class ContributingResult(BaseModel):
    """Information about a result contributing to the merge."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(..., description="Result ID")
    priority: str = Field(..., description="Priority level")
    weight: float = Field(..., description="Weight in merge (0-1)")


class MergeDetail(BaseModel):
    """Details about how a specific parameter was merged."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    parameter: str = Field(..., description="Parameter name")
    merged_value: float = Field(..., description="Final merged value")
    contributing_values: List[Dict[str, Any]] = Field(default_factory=list, description="Values from each result")
    merge_method: str = Field(..., description="How values were combined")


class WeightedMergeResult(BaseModel):
    """Result of weighted merge conflict resolution with reasoning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    conflict_detected: bool = Field(..., description="Whether conflict was detected")
    resolution_strategy: str = Field(..., description="Resolution strategy used")
    merged_result_id: str = Field(..., description="ID of merged result")
    contributing_results: List[ContributingResult] = Field(default_factory=list, description="Results contributing to merge")
    merged_config: Dict = Field(..., description="Merged optimization result")
    merge_details: List[Dict] = Field(default_factory=list, description="Details of parameter merging")
    resolution_notes: str = Field(..., description="Notes about merge process")
    conflict_summary: Optional[str] = Field(None, description="Summary of original conflict")
    reasoning: Optional[str] = Field(None, description="Detailed reasoning for merge decisions")


# ============================================================================
# AGENT INSTRUCTIONS
# ============================================================================

INSTRUCTIONS = [
    "You are an expert conflict resolution agent using WEIGHTED MERGE strategy.",
    "Your task: Resolve conflicts by intelligently merging configurations based on priority weights.",
    "",
    "## Priority Weights:",
    "- CRITICAL: Weight = 4",
    "- HIGH: Weight = 3",
    "- MEDIUM: Weight = 2",
    "- LOW: Weight = 1",
    "",
    "## Merge Strategy:",
    "1. Collect all conflicting optimization results",
    "2. Assign weight to each result based on its priority",
    "3. For each parameter conflict:",
    "   - Calculate weighted average of changes",
    "   - Weight = priority_weight / sum_of_all_weights",
    "   - Merged_value = Σ(value_i × weight_i)",
    "4. Combine into single merged configuration",
    "",
    "## When to Use Weighted Merge:",
    "- MEDIUM or LOW severity conflicts",
    "- Resource contention (same direction changes)",
    "- Multiple valid competing objectives",
    "- When compromise is acceptable",
    "- Balancing multiple stakeholder needs",
    "",
    "## Merge Benefits:",
    "- Preserves contributions from all intents",
    "- Respects priority hierarchy through weights",
    "- Smoother network operation (less disruption)",
    "- Balanced approach for competing objectives",
    "- Fair compromise when priorities are close",
    "",
    "## Merge Limitations:",
    "- May not fully satisfy any single intent",
    "- Not suitable for boolean conflicts (ON/OFF)",
    "- Not ideal for opposite direction changes",
    "- Weighted average may not be optimal configuration",
    "",
    "## Reasoning Guidelines:",
    "- Explain HOW the merge balances competing intents",
    "- Describe the weight distribution and its impact",
    "- Note which intents influenced the result most",
    "- Discuss trade-offs in the merged solution",
    "- Consider operational benefits of compromise",
    "- Be concise but informative (2-4 sentences)",
    "",
    "## Important:",
    "- You receive pre-computed merge from core engine",
    "- Your role is to validate and provide reasoning/context",
    "- Focus on explaining the merge rationale and benefits",
    "- Consider how well the compromise serves all objectives",
]


# ============================================================================
# AGENT DEFINITION
# ============================================================================

weighted_merge_agent = Agent(
    name="WeightedMergeAgent",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=INSTRUCTIONS,
    markdown=False,
)


# ============================================================================
# HELPER FUNCTION TO RUN WEIGHTED MERGE
# ============================================================================

def run_weighted_merge(
    conflict_report: dict,
    new_result: dict,
    active_results: list
) -> WeightedMergeResult:
    """
    Run weighted merge conflict resolution.
    
    Args:
        conflict_report: Conflict detection report dict
        new_result: New optimization result
        active_results: List of active optimization results
    
    Returns:
        WeightedMergeResult with merged configuration and reasoning
    """
    print("\n" + "="*70)
    print("🤖 WEIGHTED MERGE AGENT - Starting...")
    print("="*70)
    
    try:
        print(f"⚖️  Resolving conflicts using WEIGHTED MERGE strategy")
        print(f"📊 Merging new result with {len(active_results)} active result(s)")
        
        # Run core weighted merge
        print("🔍 Running core weighted merge algorithm...")
        core_result = core_resolve_by_weighted_merge(
            conflict_report=conflict_report,
            new_result=new_result,
            active_results=active_results
        )
        print(f"✅ Core merge completed: Configuration merged")
    
        # Convert to dict for processing
        result_dict = core_result.model_dump()
        
        # Prepare prompt for LLM to generate reasoning
        print("🧠 Generating merge reasoning with LLM...")
        prompt = f"""Analyze this weighted merge resolution result and provide clear reasoning:

CONFLICT DETECTED: {result_dict['conflict_detected']}

NEW RESULT:
- Result ID: {new_result.get('result_id', 'N/A')}
- Priority: {new_result.get('priority', new_result.get('input', {}).get('priority', 'MEDIUM'))}

ACTIVE RESULTS: {len(active_results)} intent(s)
{_format_active_results(active_results)}

MERGE RESULT:
- Strategy: {result_dict['resolution_strategy']}
- Merged ID: {result_dict['merged_result_id']}
- Contributing Results: {len(result_dict['contributing_results'])}
{_format_contributing_results(result_dict['contributing_results'])}
- Merge Details: {len(result_dict['merge_details'])} parameter(s) merged
- Notes: {result_dict['resolution_notes']}

Provide the complete WeightedMergeResult with reasoning that explains:
1. How the weighted merge balances competing objectives
2. Which intents influenced the result most (and why)
3. Benefits and trade-offs of this compromise solution

Include all fields from the merge result with added reasoning."""

        # Run agent to generate reasoning
        response = weighted_merge_agent.run(prompt, stream=False)
        print("✅ LLM reasoning generated")
        
        # Extract reasoning from LLM response
        reasoning_text = ""
        if hasattr(response, 'content'):
            reasoning_text = str(response.content)
        elif isinstance(response, str):
            reasoning_text = response
        else:
            reasoning_text = "Weighted merge completed. Configuration balances all intent priorities."
        
        # Return core result with LLM reasoning
        print("\n✅ WEIGHTED MERGE AGENT - Completed Successfully!")
        print("="*70)
        return WeightedMergeResult(
            **result_dict,
            reasoning=reasoning_text
        )
    
    except Exception as e:
        print("\n❌ WEIGHTED MERGE AGENT - ERROR!")
        print("="*70)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        import traceback
        print("\nFull Traceback:")
        traceback.print_exc()
        print("="*70)
        raise RuntimeError(f"WeightedMergeAgent failed: {str(e)}") from e


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


def _format_contributing_results(contributing: List[dict]) -> str:
    """Format contributing results with weights."""
    if not contributing:
        return "  (none)"
    
    lines = []
    for i, contrib in enumerate(contributing, 1):
        lines.append(f"  {i}. {contrib.get('id')} - Priority: {contrib.get('priority')} - Weight: {contrib.get('weight'):.2%}")
    
    return '\n'.join(lines)


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    conflict_report = {
        "is_conflicted": True,
        "conflict_summary": "Resource contention detected",
        "num_conflicts": 1
    }
    
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
                    {"param": "tx0_P_dBm", "before": 30.0, "change": 1.5, "unit": "dBm"}
                ]
            }
        }
    ]
    
    result = run_weighted_merge(conflict_report, new_result, active_results)
    
    print(f"\n⚖️  Weighted Merge Result:")
    print(f"   Merged ID: {result.merged_result_id}")
    print(f"   Contributing: {len(result.contributing_results)} result(s)")
    print(f"   Parameters Merged: {len(result.merge_details)}")
    if result.reasoning:
        print(f"\n📝 Reasoning:\n{result.reasoning}")
