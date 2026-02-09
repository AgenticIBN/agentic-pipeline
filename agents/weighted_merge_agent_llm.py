#!/usr/bin/env python3
"""
Weighted Merge Resolution Agent (Full LLM-Based)
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
    merged_change: Any = Field(..., description="Final merged change value (can be float or bool)")
    merged_value: Optional[Any] = Field(None, description="Final merged value (optional, for new config)")
    contributing_values: List[Dict[str, Any]] = Field(default_factory=list, description="Values from each result")
    merge_method: Optional[str] = Field(None, description="How values were combined (e.g., 'weighted_average', 'weighted_voting')")


class WeightedMergeResult(BaseModel):
    """Result of weighted merge conflict resolution with reasoning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    conflict_detected: bool = Field(default=True, description="Whether conflict was detected")
    resolution_strategy: str = Field(default="WEIGHTED_MERGE", description="Resolution strategy used")
    merged_result_id: str = Field(..., description="ID of merged result")
    contributing_results: List[ContributingResult] = Field(default_factory=list, description="Results contributing to merge")
    merged_config: Dict = Field(..., description="Merged optimization result")
    merge_details: List[MergeDetail] = Field(default_factory=list, description="Details of parameter merging")
    resolution_notes: str = Field(..., description="Notes about merge process")
    conflict_summary: str = Field(..., description="Summary of original conflict")
    reasoning: str = Field(..., description="Detailed reasoning for merge decisions")


# ============================================================================
# AGENT INSTRUCTIONS
# ============================================================================

INSTRUCTIONS = [
    "You are an EXPERT conflict resolution agent using WEIGHTED MERGE strategy.",
    "Your task: Resolve conflicts by intelligently merging configurations using PURE REASONING.",
    "",
    "## 🎯 YOUR MISSION:",
    "Given conflicting optimization results, you must:",
    "1. Calculate weights for each result based on priority",
    "2. Merge parameter changes using weighted averaging",
    "3. Create unified configuration that balances all intents",
    "4. Provide detailed merge methodology",
    "5. Explain reasoning for merge decisions",
    "",
    "## ⚖️ PRIORITY-BASED WEIGHTS:",
    "",
    "### Weight Calculation:",
    "```",
    "Priority  → Raw Weight",
    "CRITICAL  → 4",
    "HIGH      → 3",
    "MEDIUM    → 2",
    "LOW       → 1",
    "",
    "Normalized Weight = Raw Weight / Sum of All Raw Weights",
    "```",
    "",
    "### Example:",
    "```",
    "Intent A: HIGH (weight=3)",
    "Intent B: MEDIUM (weight=2)",
    "Intent C: LOW (weight=1)",
    "",
    "Sum = 3 + 2 + 1 = 6",
    "",
    "Normalized:",
    "- Intent A: 3/6 = 0.50 (50%)",
    "- Intent B: 2/6 = 0.33 (33%)",
    "- Intent C: 1/6 = 0.17 (17%)",
    "```",
    "",
    "## 🔧 MERGE ALGORITHMS:",
    "",
    "### 1. Numeric Parameter Merging (Power, Angles):",
    "**Method**: Weighted average of changes",
    "",
    "```",
    "Example: tx0_P_dBm merging",
    "",
    "Intent A (HIGH, weight=0.5): Change +3 dB",
    "Intent B (MEDIUM, weight=0.3): Change +2 dB",
    "Intent C (LOW, weight=0.2): Change +1 dB",
    "",
    "Merged Change = (3 × 0.5) + (2 × 0.3) + (1 × 0.2)",
    "              = 1.5 + 0.6 + 0.2",
    "              = 2.3 dB",
    "",
    "Final: tx0_P_dBm change = +2.3 dB",
    "```",
    "",
    "**Special Case**: Opposite directions",
    "```",
    "Intent A (HIGH, weight=0.6): Change +3 dB",
    "Intent B (MEDIUM, weight=0.4): Change -2 dB",
    "",
    "Merged Change = (3 × 0.6) + (-2 × 0.4)",
    "              = 1.8 - 0.8",
    "              = +1.0 dB",
    "",
    "Result: Positive direction wins (weighted)",
    "```",
    "",
    "### 2. Boolean Parameter Merging (ON/OFF States):",
    "**Method**: Weighted voting",
    "",
    "```",
    "Example: tx0_on merging",
    "",
    "Intent A (HIGH, weight=0.5): tx0_on = True (vote: 0.5)",
    "Intent B (MEDIUM, weight=0.3): tx0_on = False (vote: 0.0)",
    "Intent C (LOW, weight=0.2): tx0_on = True (vote: 0.2)",
    "",
    "Total vote for True = 0.5 + 0.2 = 0.7 (70%)",
    "Total vote for False = 0.3 (30%)",
    "",
    "Decision: tx0_on = True (majority vote)",
    "Threshold: >50% to enable",
    "```",
    "",
    "### 3. Non-Conflicting Parameters:",
    "**Method**: Include all without merging",
    "",
    "```",
    "If only one intent changes a parameter:",
    "- Use that intent's value directly",
    "- Apply its full weight (no dilution)",
    "```",
    "",
    "## 📊 MERGE PROCESS:",
    "",
    "### Step 1: Calculate Weights",
    "For each conflicting result:",
    "1. Get priority level",
    "2. Map to raw weight (1-4)",
    "3. Calculate normalized weight",
    "4. Store in contributing_results list",
    "",
    "### Step 2: Identify All Parameters",
    "Collect all unique parameters changed across all intents:",
    "```",
    "Intent A changes: [tx0_P_dBm, tx0_dEl]",
    "Intent B changes: [tx0_P_dBm, tx1_P_dBm]",
    "",
    "All parameters: [tx0_P_dBm, tx0_dEl, tx1_P_dBm]",
    "```",
    "",
    "### Step 3: Merge Each Parameter",
    "For each parameter:",
    "1. Find all intents that change it",
    "2. Extract their change values and weights",
    "3. Apply appropriate merge algorithm (numeric/boolean)",
    "4. Calculate merged value",
    "5. Document in merge_details",
    "",
    "### Step 4: Build Merged Configuration",
    "Create new optimization result with:",
    "- Merged result ID",
    "- All merged parameter changes",
    "- Combined expected KPIs (weighted average)",
    "- Constraints check",
    "",
    "### Step 5: Predict Merged KPIs",
    "Estimate KPI values for merged configuration:",
    "- Use weighted average of expected KPIs from all intents",
    "- Consider interaction effects",
    "- Validate against constraints",
    "",
    "## 💡 KPI MERGING GUIDELINES:",
    "",
    "```",
    "Example: RX_POWER prediction",
    "",
    "Intent A (weight=0.5): Expected RX_POWER = -65 dBm",
    "Intent B (weight=0.3): Expected RX_POWER = -70 dBm",
    "Intent C (weight=0.2): Expected RX_POWER = -68 dBm",
    "",
    "Merged RX_POWER = (-65 × 0.5) + (-70 × 0.3) + (-68 × 0.2)",
    "                = -32.5 - 21.0 - 13.6",
    "                = -67.1 dBm",
    "```",
    "",
    "## 🎯 WHEN TO USE WEIGHTED MERGE:",
    "",
    "### Ideal Scenarios:",
    "- ✅ Resource contention (same direction, different magnitudes)",
    "- ✅ Multiple stakeholder objectives",
    "- ✅ MEDIUM or LOW severity conflicts",
    "- ✅ Priorities are similar (within 1 level)",
    "- ✅ Compromise is acceptable",
    "- ✅ Gradual optimization preferred",
    "",
    "### Less Ideal (Consider PRIORITY instead):",
    "- ⚠️  HIGH or CRITICAL severity conflicts",
    "- ⚠️  Boolean conflicts with strong disagreement",
    "- ⚠️  Large priority differences (CRITICAL vs LOW)",
    "- ⚠️  Emergency situations",
    "- ⚠️  Safety-critical decisions",
    "",
    "## 📝 MERGED RESULT FORMAT:",
    "",
    "The merged_config must be a complete optimization result:",
    "```json",
    "{",
    '  "result_id": "merged_<timestamp>",',
    '  "priority": "<highest_priority>",',
    '  "test_name": "merged_optimization",',
    '  "passed": true,',
    '  "input": {',
    '    "target_area": "merged",',
    '    "target_kpis": [...all unique KPIs...],',
    '    "priority": "<highest_priority>"',
    "  },",
    '  "output": {',
    '    "selected_config_id": <new_id>,',
    '    "changes": [...merged changes...],',
    '    "expected_kpis": {...merged KPI predictions...},',
    '    "constraints_satisfied": true',
    "  },",
    '  "reasoning": "Merged configuration balancing..."',
    "}",
    "```",
    "",
    "## 💭 REASONING TEMPLATE:",
    "",
    "\"Merged [N] conflicting intents using priority-weighted averaging. ",
    "Contributing intents have weights: [list weights]. ",
    "For parameter [X], merged value [Y] balances Intent A's [change1] (weight [w1]) with Intent B's [change2] (weight [w2]). ",
    "This compromise [achieves/approximates] all target objectives while respecting priority hierarchy. ",
    "Expected merged KPIs show [improvement/balance] across [metrics]. ",
    "[Any caveats or trade-offs noted].\"",
    "",
    "## 📝 OUTPUT REQUIREMENTS:",
    "",
    "You MUST return a complete WeightedMergeResult with:",
    "1. **conflict_detected**: true",
    "2. **resolution_strategy**: \"WEIGHTED_MERGE\"",
    "3. **merged_result_id**: New unique ID for merged result (e.g., 'merged_20260209_123456')",
    "4. **contributing_results**: List with id, priority, weight for each intent",
    "5. **merged_config**: Complete optimization result (full format above)",
    "6. **merge_details**: List of MergeDetail for each merged parameter with:",
    "   - parameter: name (e.g., 'tx0_P_dBm')",
    "   - merged_change: final change value (e.g., +2.5 or true)",
    "   - merge_method: 'weighted_average' or 'weighted_voting' (OPTIONAL)",
    "   - contributing_values: list of {id, value, weight} for each source",
    "7. **resolution_notes**: Brief summary of merge process",
    "8. **conflict_summary**: Description of original conflict",
    "9. **reasoning**: Detailed 4-6 sentence explanation including:",
    "   - How weights were calculated",
    "   - How each parameter was merged",
    "   - Expected benefits of compromise",
    "   - Any trade-offs or limitations",
    "",
    "## 📋 EXAMPLE OUTPUT:",
    "```json",
    "{",
    '  "conflict_detected": true,',
    '  "resolution_strategy": "WEIGHTED_MERGE",',
    '  "merged_result_id": "merged_20260209_123456",',
    '  "contributing_results": [',
    '    {"id": "opt_123", "priority": "HIGH", "weight": 0.6},',
    '    {"id": "opt_456", "priority": "MEDIUM", "weight": 0.4}',
    "  ],",
    '  "merge_details": [',
    '    {',
    '      "parameter": "tx0_P_dBm",',
    '      "merged_change": 2.6,',
    '      "merge_method": "weighted_average",',
    '      "contributing_values": [',
    '        {"id": "opt_123", "value": 3.0, "weight": 0.6},',
    '        {"id": "opt_456", "value": 2.0, "weight": 0.4}',
    "      ]",
    "    }",
    "  ],",
    '  "merged_config": {...full optimization result...},',
    '  "resolution_notes": "Merged 2 intents...",',
    '  "conflict_summary": "Conflicts on 3 parameters",',
    '  "reasoning": "Calculated weights..."',
    "}",
    "```",
    "",
    "## ⚠️ CRITICAL REMINDERS:",
    "- You are making REAL merge decisions affecting live network",
    "- No core engine backup - rely on your mathematical reasoning",
    "- Calculate weights precisely using priority levels",
    "- Use weighted averaging for numeric values",
    "- Use weighted voting for boolean values",
    "- Predict realistic merged KPIs",
    "- Ensure merged configuration is operationally safe",
    "- Document methodology clearly in merge_details",
    "",
    "## 📤 OUTPUT FORMAT:",
    "Return ONLY a valid JSON object matching the WeightedMergeResult schema.",
    "Do NOT include any markdown, explanations, or text outside the JSON.",
    "Start with { and end with }",
]


# ============================================================================
# AGENT DEFINITION
# ============================================================================

weighted_merge_agent = Agent(
    name="WeightedMergeAgent_FullLLM",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=INSTRUCTIONS,
    markdown=False,
    structured_outputs=True,
)


# ============================================================================
# RUN MERGE FUNCTION
# ============================================================================

def run_weighted_merge(
    conflict_report: Dict[str, Any],
    new_result: Dict[str, Any],
    active_results: List[Dict[str, Any]]
) -> WeightedMergeResult:
    """
    Run FULL LLM-BASED weighted merge resolution.
    NO core logic - pure agentic reasoning!
    
    Args:
        conflict_report: Conflict detection report
        new_result: New optimization result
        active_results: List of active results involved in conflict
    
    Returns:
        WeightedMergeResult with merged configuration and reasoning
    """
    print("\n" + "="*70)
    print("🔀 WEIGHTED MERGE AGENT (FULL LLM) - Starting...")
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
        results_text = "CONFLICTING OPTIMIZATION RESULTS TO MERGE:\n\n"
        for i, result in enumerate(all_results, 1):
            result_id = result.get('result_id', f'result_{i}')
            priority = result.get('priority', 'MEDIUM')
            target_area = result.get('input', {}).get('target_area', 'unknown')
            target_kpis = result.get('input', {}).get('target_kpis', [])
            changes = result.get('output', {}).get('changes', [])
            expected_kpis = result.get('output', {}).get('expected_kpis', {})
            
            results_text += f"""Result {i}:
ID: {result_id}
Priority: {priority}
Target Area: {target_area}
Target KPIs: {', '.join(target_kpis)}
Configuration Changes:
{_format_changes(changes)}
Expected KPIs:
{_format_kpis(expected_kpis)}

"""
        
        # Build comprehensive prompt
        prompt = f"""Resolve this conflict using weighted merge strategy.

CONFLICT SUMMARY:
{conflict_summary}
Number of conflicts: {num_conflicts}

{results_text}

YOUR TASK:
1. Calculate weights for each result based on priority:
   - CRITICAL → 4, HIGH → 3, MEDIUM → 2, LOW → 1
   - Normalize: weight = raw / sum(all_raw)

2. For each conflicting parameter:
   - Extract all change values
   - Apply weighted averaging (numeric) or voting (boolean)
   - Document in merge_details

3. Create merged configuration result with:
   - New merged_result_id
   - All merged parameter changes
   - Weighted average of expected KPIs
   - Complete optimization result format

4. Provide detailed reasoning explaining:
   - Weight calculations
   - Merge methodology for each parameter
   - Expected outcomes
   - Trade-offs

INSTRUCTIONS:
- Calculate precise weights using priority hierarchy
- Use weighted math for all merges
- Predict realistic merged KPIs
- Ensure merged config is safe and valid
- Document methodology clearly

Generate complete WeightedMergeResult JSON with merged_config containing full optimization result structure.
IMPORTANT: Return ONLY valid JSON, no markdown, no explanations."""

        # Run LLM agent
        print(f"🧠 LLM is merging {len(all_results)} conflicting intent(s)")
        print(f"⚖️  Priorities: {[r.get('priority', 'MEDIUM') for r in all_results]}")
        
        response = weighted_merge_agent.run(prompt, stream=False)
        
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
        result = WeightedMergeResult(**result_dict)
        
        print(f"\n✅ Merge complete!")
        print(f"🔀 Merged result: {result.merged_result_id}")
        print(f"📊 Contributors: {len(result.contributing_results)} intent(s)")
        print(f"🔧 Merged parameters: {len(result.merge_details)}")
        print(f"\n📝 Merge Result (JSON):")
        # Convert Pydantic models to dict for JSON serialization
        merge_result_json = result.model_dump()
        print(json_module.dumps(merge_result_json, indent=2, default=str))
        
        return result
        
    except Exception as e:
        print(f"\n❌ Weighted merge failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def _format_changes(changes: List[Dict]) -> str:
    """Format configuration changes for display."""
    if not changes:
        return "  No changes"
    
    lines = []
    for change in changes[:5]:  # Limit to first 5 for display
        # Support both 'param' and 'parameter' keys
        param = change.get('parameter') or change.get('param', 'unknown')
        change_val = change.get('change', 0)
        unit = change.get('unit', '')
        
        # Format change value: use :+ for numeric, plain string for boolean/string
        if isinstance(change_val, (int, float)):
            formatted_val = f"{change_val:+.1f}" if isinstance(change_val, float) else f"{change_val:+d}"
        else:
            formatted_val = str(change_val)
        
        lines.append(f"  - {param}: {formatted_val} {unit}")
    
    if len(changes) > 5:
        lines.append(f"  ... and {len(changes) - 5} more changes")
    
    return "\n".join(lines)


def _format_kpis(kpis: Dict) -> str:
    """Format KPI dict for display."""
    if not kpis:
        return "  No KPI data"
    
    lines = []
    for key, value in list(kpis.items())[:5]:
        if isinstance(value, (int, float)):
            lines.append(f"  - {key}: {value:.2f}")
        else:
            lines.append(f"  - {key}: {value}")
    
    return "\n".join(lines) if lines else "  No KPI data"


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================

if __name__ == "__main__":
    # Test with sample data
    conflict_report = {
        "is_conflicted": True,
        "conflict_summary": "Resource contention on tx0_P_dBm",
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
            ],
            "expected_kpis": {"RX_POWER": -65.0}
        }
    }
    
    active_results = [
        {
            "result_id": "opt_20260208_000",
            "priority": "MEDIUM",
            "input": {
                "target_area": "cell1",
                "target_kpis": ["RX_POWER"]
            },
            "output": {
                "changes": [
                    {"param": "tx0_P_dBm", "change": 2.0, "unit": "dBm"}
                ],
                "expected_kpis": {"RX_POWER": -68.0}
            }
        }
    ]
    
    result = run_weighted_merge(conflict_report, new_result, active_results)
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(f"Merged ID: {result.merged_result_id}")
    print(f"Contributors: {len(result.contributing_results)}")
    print(f"Merge Details: {len(result.merge_details)}")
    print(f"Reasoning: {result.reasoning}")
