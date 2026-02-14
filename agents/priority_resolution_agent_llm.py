#!/usr/bin/env python3
"""
Priority Resolution Agent (Full LLM-Based)
Resolves conflicts using pure LLM reasoning - NO core logic!
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq

load_dotenv()


# ============================================================================
# RESPONSE SCHEMA
# ============================================================================

class PriorityResolutionResult(BaseModel):
    """Result of priority-based conflict resolution with reasoning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    conflict_detected: bool = Field(..., description="Whether conflict was detected")
    resolution_strategy: str = Field(default="PRIORITY", description="Resolution strategy used")
    winning_result_id: str = Field(..., description="ID of selected result")
    winning_priority: str = Field(..., description="Priority of winning result")
    winning_config: Dict = Field(..., description="Full winning optimization result")
    rejected_result_ids: List[str] = Field(default_factory=list, description="IDs of rejected results")
    resolution_notes: str = Field(..., description="Notes about resolution decision")
    conflict_summary: str = Field(..., description="Summary of original conflict")
    conflict_resolution_summary: str = Field(default="", description="Summary of conflict resolution outcome")
    reasoning: str = Field(..., description="Detailed reasoning for resolution decision")
    tie_breaking_rule_used: Optional[str] = Field(None, description="Rule used if priorities were equal (e.g., 'prefer_active')")


# ============================================================================
# AGENT INSTRUCTIONS
# ============================================================================

INSTRUCTIONS = [
    "You are an EXPERT conflict resolution agent using PRIORITY-based strategy.",
    "Your task: Resolve conflicts by selecting the highest priority intent using PURE REASONING.",
    "",
    "## 🎯 YOUR MISSION:",
    "Given conflicting optimization results, you must:",
    "1. Compare priority levels of all conflicting intents",
    "2. Select the result with highest priority",
    "3. Handle tie-breaking scenarios intelligently",
    "4. Reject lower priority results",
    "5. Provide detailed reasoning for your decision",
    "",
    "## 📊 PRIORITY HIERARCHY (Highest to Lowest):",
    "",
    "### 1. CRITICAL (Priority Level 4):",
    "- Emergency network situations",
    "- Critical failures or outages",
    "- Safety-critical operations",
    "- Regulatory compliance requirements",
    "- **Always takes precedence over all others**",
    "",
    "### 2. HIGH (Priority Level 3):",
    "- Important operational needs",
    "- Significant service degradation",
    "- Major customer complaints",
    "- SLA violations",
    "- **Takes precedence over MEDIUM and LOW**",
    "",
    "### 3. MEDIUM (Priority Level 2):",
    "- Standard optimization requests",
    "- Routine network improvements",
    "- Proactive optimizations",
    "- Scheduled maintenance",
    "- **Takes precedence over LOW only**",
    "",
    "### 4. LOW (Priority Level 1):",
    "- Optional improvements",
    "- Nice-to-have enhancements",
    "- Experimental optimizations",
    "- Long-term planning",
    "- **Always defers to higher priorities**",
    "",
    "## 🎯 RESOLUTION PROCESS:",
    "",
    "### Step 1: Identify All Conflicting Results",
    "List all optimization results involved in the conflict:",
    "```",
    "Conflicting Results:",
    "- Result A: ID=opt_001, Priority=HIGH",
    "- Result B: ID=opt_002, Priority=MEDIUM",
    "- Result C: ID=opt_003, Priority=HIGH",
    "```",
    "",
    "### Step 2: Find Highest Priority Level",
    "Determine the maximum priority among all conflicting results:",
    "```",
    "Priorities: [HIGH, MEDIUM, HIGH]",
    "Highest: HIGH",
    "```",
    "",
    "### Step 3: Filter Results by Highest Priority",
    "Select all results with the highest priority:",
    "```",
    "Candidates: [Result A (HIGH), Result C (HIGH)]",
    "```",
    "",
    "### Step 4: Handle Tie-Breaking",
    "If multiple results share highest priority, use these tie-breakers:",
    "",
    "#### Tie-Breaker 1: Result Status",
    "- **ACTIVE results** (already applied) have preference",
    "- Reason: Minimizes network disruption",
    "- If new vs active → Choose ACTIVE",
    "",
    "#### Tie-Breaker 2: Timestamp",
    "- **Earlier timestamp** has preference",
    "- Reason: First-come-first-served fairness",
    "- Parse result_id timestamp if available",
    "",
    "#### Tie-Breaker 3: Severity/Impact",
    "- Consider conflict severity",
    "- Choose result that addresses more CRITICAL issues",
    "- Consider network stability impact",
    "",
    "### Step 5: Select Winner",
    "Choose ONE result as winner:",
    "```",
    "Winner: Result A",
    "Reason: Highest priority (HIGH), already ACTIVE",
    "```",
    "",
    "### Step 6: Reject Others",
    "All non-winning results are rejected:",
    "```",
    "Rejected: [Result B (lower priority), Result C (tie-lost)]",
    "```",
    "",
    "## 💡 DECISION RATIONALE GUIDELINES:",
    "",
    "### When Winner is Clear (Different Priorities):",
    "\"Selected Result [ID] with [PRIORITY] priority over [N] lower-priority results. ",
    "This intent addresses [purpose] which takes precedence due to [reason - e.g., emergency, SLA, safety]. ",
    "Rejected results include [IDs] with [their priorities and purposes].\"",
    "",
    "### When Tie-Breaking is Needed (Same Priorities):",
    "\"Multiple intents share [PRIORITY] priority. Selected Result [ID] because [tie-breaker reason - e.g., active status, earlier timestamp, higher impact]. ",
    "This minimizes [disruption/risk] while addressing [primary concern]. ",
    "Rejected Result [ID] despite equal priority due to [tie-breaker reason].\"",
    "",
    "### When CRITICAL Priority Involved:",
    "\"CRITICAL priority intent takes absolute precedence. Result [ID] addresses [emergency/safety issue] which cannot be compromised. ",
    "All other intents, regardless of merit, must be deferred to ensure [critical outcome]. ",
    "Network stability and [safety/compliance] require immediate [action].\"",
    "",
    "## ⚖️ SPECIAL CONSIDERATIONS:",
    "",
    "### Network Stability:",
    "- Prefer results that minimize configuration changes",
    "- Consider cumulative impact of changes",
    "- Avoid frequent flip-flopping between states",
    "",
    "### Operational Impact:",
    "- Consider user impact of rejection",
    "- Note dependencies between intents",
    "- Mention if rejected intents should be re-evaluated",
    "",
    "### Future Actions:",
    "- Suggest when rejected intents could be reconsidered",
    "- Note if conflicts might resolve over time",
    "- Recommend monitoring after resolution",
    "",
    "## 📝 OUTPUT REQUIREMENTS:",
    "",
    "You MUST return a complete PriorityResolutionResult with:",
    "1. **conflict_detected**: true (always true when this agent runs)",
    "2. **resolution_strategy**: \"PRIORITY\"",
    "3. **winning_result_id**: ID of selected result",
    "4. **winning_priority**: Priority level of winner",
    "5. **winning_config**: Complete optimization result of winner",
    "6. **rejected_result_ids**: List of all rejected result IDs",
    "7. **resolution_notes**: Brief summary of decision",
    "8. **conflict_summary**: Description of original conflict",
    "9. **reasoning**: Detailed 4-6 sentence explanation including:",
    "   - Why winner was selected",
    "   - What makes it higher priority or better tie-breaker",
    "   - Impact of rejecting other results",
    "   - Network stability considerations",
    "",
    "## ⚠️ CRITICAL REMINDERS:",
    "- You are making REAL resolution decisions affecting live network",
    "- No core engine backup - rely on your expert judgment",
    "- Priority hierarchy is STRICT - CRITICAL always wins",
    "- Tie-breaking must be consistent and fair",
    "- Consider long-term network health, not just immediate resolution",
    "- Be decisive - exactly ONE winner must be selected",
    "",
    "## 📤 OUTPUT FORMAT:",
    "Return ONLY a valid JSON object matching the PriorityResolutionResult schema.",
    "Do NOT include any markdown, explanations, or text outside the JSON.",
    "Start with { and end with }",
]


# ============================================================================
# AGENT DEFINITION
# ============================================================================

priority_resolution_agent = Agent(
    name="PriorityResolutionAgent_FullLLM",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=INSTRUCTIONS,
    output_schema=PriorityResolutionResult,
    markdown=False,
    structured_outputs=True,
)


# ============================================================================
# RUN RESOLUTION FUNCTION
# ============================================================================

def run_priority_resolution(
    conflict_report: Dict[str, Any],
    new_result: Dict[str, Any],
    active_results: List[Dict[str, Any]]
) -> PriorityResolutionResult:
    """
    Run FULL LLM-BASED priority resolution.
    NO core logic - pure agentic reasoning!
    
    Args:
        conflict_report: Conflict detection report
        new_result: New optimization result
        active_results: List of active results involved in conflict
    
    Returns:
        PriorityResolutionResult with winner selection and reasoning
    """
    print("\n" + "="*70)
    print("⚖️  PRIORITY RESOLUTION AGENT (FULL LLM) - Starting...")
    print("="*70)
    
    try:
        # Get conflicting result IDs from conflict report
        conflicting_ids = conflict_report.get('conflicting_result_ids', [])
        
        # Filter active results to only conflicting ones
        conflicting_results = []
        for active in active_results:
            if active.get('result_id') in conflicting_ids:
                conflicting_results.append(active)
        
        # Add new result to conflicting results
        all_results = [new_result] + conflicting_results
        
        # Format conflict summary
        conflict_summary = conflict_report.get('conflict_summary', 'Configuration conflicts detected')
        num_conflicts = conflict_report.get('num_conflicts', len(conflicting_ids))
        
        # Format all conflicting results for LLM
        results_text = "CONFLICTING OPTIMIZATION RESULTS:\n\n"
        for i, result in enumerate(all_results, 1):
            result_id = result.get('result_id', f'result_{i}')
            priority = result.get('priority', 'MEDIUM')
            target_area = result.get('input', {}).get('target_area', 'unknown')
            target_kpis = result.get('input', {}).get('target_kpis', [])
            changes = result.get('output', {}).get('changes', [])
            
            # Determine if result is new or active
            status = "NEW" if result == new_result else "ACTIVE"
            
            results_text += f"""Result {i} [{status}]:
ID: {result_id}
Priority: {priority}
Target Area: {target_area}
Target KPIs: {', '.join(target_kpis)}
Configuration Changes ({len(changes)} parameters):
{_format_changes(changes)}

"""
        
        # Build comprehensive prompt
        prompt = f"""Resolve this priority-based conflict.

CONFLICT SUMMARY:
{conflict_summary}
Number of conflicts: {num_conflicts}

{results_text}

YOUR TASK:
1. Identify all priority levels present
2. Select the result with HIGHEST priority
3. If tie (equal priorities), use tie-breaking rules:
   - Prefer ACTIVE status (minimizes disruption)
   - Consider timestamp (earlier wins)
   - Assess network stability impact
4. Reject all other results
5. Provide detailed reasoning

INSTRUCTIONS:
- Be decisive - select exactly ONE winner
- Explain why winner was chosen
- Note what happens to rejected intents
- Consider network stability
- Follow strict priority hierarchy

Generate complete PriorityResolutionResult JSON.
IMPORTANT: Return ONLY valid JSON, no markdown, no explanations."""

        # Run LLM agent
        print(f"\n🧠 LLM is analyzing {len(all_results)} conflicting intent(s)")
        print(f"⚖️  Priorities: {[r.get('priority', 'MEDIUM') for r in all_results]}")
        
        response = priority_resolution_agent.run(prompt, stream=False)
        
        # Extract result from response
        # Agent has output_schema set, so response.content is already PriorityResolutionResult
        if isinstance(response.content, PriorityResolutionResult):
            result = response.content
        elif hasattr(response, 'content') and isinstance(response.content, dict):
            result = PriorityResolutionResult(**response.content)
        else:
            # Fallback: try to parse as JSON string
            import json as json_module
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean markdown if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result_dict = json_module.loads(response_text)
            result = PriorityResolutionResult(**result_dict)
        
        # Find the actual winning result from all_results
        winning_result = None
        for r in all_results:
            if r.get('result_id') == result.winning_result_id:
                winning_result = r
                break
        
        if not winning_result:
            raise ValueError(f"Could not find winning result with ID: {result.winning_result_id}")
        
        # Update result with actual winning optimization result (not just config)
        result.winning_config = winning_result
        
        print(f"\n✅ Resolution complete!")
        print(f"🏆 Winner: {result.winning_result_id} (Priority: {result.winning_priority})")
        print(f"❌ Rejected: {len(result.rejected_result_ids)} intent(s)")
        print(f"\n📝 Resolution Result (JSON):")
        resolution_json = {
            "winning_result_id": result.winning_result_id,
            "winning_priority": result.winning_priority,
            "rejected_result_ids": result.rejected_result_ids,
            "tie_breaking_rule_used": result.tie_breaking_rule_used,
            "conflict_resolution_summary": result.conflict_resolution_summary,
            "reasoning": result.reasoning
        }
        print(json.dumps(resolution_json, indent=2))
        
        return result
        
    except Exception as e:
        print(f"\n❌ Priority resolution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def _format_changes(changes: List[Dict]) -> str:
    """Format configuration changes for display."""
    if not changes:
        return "  No changes"
    
    lines = []
    for change in changes[:5]:  # Limit to first 5 for display
        param = change.get('parameter', change.get('param', 'unknown'))
        change_val = change.get('change', 0)
        unit = change.get('unit', '')
        
        # Format change value: use :+ for numeric, plain for strings
        if isinstance(change_val, (int, float)):
            formatted_change = f"{change_val:+.1f}" if isinstance(change_val, float) else f"{change_val:+d}"
        else:
            formatted_change = str(change_val)
        
        lines.append(f"  - {param}: {formatted_change} {unit}")
    
    if len(changes) > 5:
        lines.append(f"  ... and {len(changes) - 5} more changes")
    
    return "\n".join(lines)


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================

if __name__ == "__main__":
    # Test with sample data
    conflict_report = {
        "is_conflicted": True,
        "conflict_summary": "Parameter conflict detected on tx0_P_dBm",
        "num_conflicts": 1,
        "conflicting_result_ids": ["opt_20260208_000"]
    }
    
    new_result = {
        "result_id": "opt_20260208_001",
        "priority": "HIGH",
        "input": {
            "target_area": "cell1",
            "target_kpis": ["RX_POWER"]
        },
        "output": {
            "changes": [
                {"param": "tx0_P_dBm", "change": 3.0, "unit": "dBm"}
            ]
        }
    }
    
    active_results = [
        {
            "result_id": "opt_20260208_000",
            "priority": "MEDIUM",
            "input": {
                "target_area": "cell1",
                "target_kpis": ["POWER_EFFICIENCY"]
            },
            "output": {
                "changes": [
                    {"param": "tx0_P_dBm", "change": -2.0, "unit": "dBm"}
                ]
            }
        }
    ]
    
    result = run_priority_resolution(conflict_report, new_result, active_results)
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(f"Winner: {result.winning_result_id}")
    print(f"Priority: {result.winning_priority}")
    print(f"Rejected: {result.rejected_result_ids}")
    print(f"Reasoning: {result.reasoning}")
