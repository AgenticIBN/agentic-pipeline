#!/usr/bin/env python3
"""
Priority Based Resolution Agent V2
Pure deterministic resolution based on priorities
Works with conflict_detector_agent.py outputs
"""
from __future__ import annotations

import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# SCHEMAS
# ============================================================================

class PriorityBasedResolutionOutput(BaseModel):
    """Output from priority-based resolution."""
    conflict_detected: bool
    resolution_strategy: str
    winning_result_id: Optional[str] = None
    winning_priority: Optional[str] = None
    winning_config: Optional[Dict[str, Any]] = None  # The full optimization result
    rejected_result_ids: List[str] = Field(default_factory=list)
    resolution_notes: str
    conflict_summary: Optional[str] = None


# ============================================================================
# PRIORITY RANKING
# ============================================================================

PRIORITY_RANK = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1
}


# ============================================================================
# MAIN RESOLUTION FUNCTION
# ============================================================================

def resolve_by_priority(
    conflict_report: Dict[str, Any],
    new_result: Dict[str, Any],
    active_results: List[Dict[str, Any]]
) -> PriorityBasedResolutionOutput:
    """
    Priority-based resolution: Select result with highest priority.
    
    Priority order: CRITICAL > HIGH > MEDIUM > LOW
    If same priority, prefer active results (already applied).
    
    Args:
        conflict_report: Dict from ConflictReport.model_dump()
        new_result: New optimization result
        active_results: List of active optimization results
        
    Returns:
        PriorityBasedResolutionOutput with winning result
    """
    
    # Check if conflict exists
    if not conflict_report.get("is_conflicted", False):
        return PriorityBasedResolutionOutput(
            conflict_detected=False,
            resolution_strategy="NO_CONFLICT",
            winning_result_id=get_result_id(new_result),
            winning_priority=get_priority(new_result),
            winning_config=new_result,
            rejected_result_ids=[],
            resolution_notes="No conflicts detected. New result can be applied safely.",
            conflict_summary=conflict_report.get("conflict_summary", "")
        )
    
    # Collect all results with their priorities
    all_results = [new_result] + active_results
    result_priorities = []
    
    for result in all_results:
        result_id = get_result_id(result)
        priority = get_priority(result)
        priority_rank = PRIORITY_RANK.get(priority, 1)
        is_active = result in active_results
        
        result_priorities.append({
            "result": result,
            "result_id": result_id,
            "priority": priority,
            "priority_rank": priority_rank,
            "is_active": is_active
        })
    
    # Sort by priority (descending), then prefer active results
    sorted_results = sorted(
        result_priorities,
        key=lambda x: (x["priority_rank"], x["is_active"]),
        reverse=True
    )
    
    # Winner is highest priority
    winner = sorted_results[0]
    rejected = [r["result_id"] for r in sorted_results[1:]]
    
    # Generate resolution notes
    notes = []
    notes.append(f"Conflicts: {conflict_report.get('num_conflicts', 0)}")
    notes.append(f"Strategy: PRIORITY_SELECTION")
    notes.append(f"Winner: {winner['result_id']} (Priority: {winner['priority']})")
    
    if rejected:
        notes.append(f"Rejected: {len(rejected)} result(s)")
    
    # Check if winner is the new result or an active one
    if winner["result"] == new_result:
        notes.append("Decision: Apply new result")
    else:
        notes.append("Decision: Keep active result, reject new result")
    
    return PriorityBasedResolutionOutput(
        conflict_detected=True,
        resolution_strategy="PRIORITY_SELECTION",
        winning_result_id=winner["result_id"],
        winning_priority=winner["priority"],
        winning_config=winner["result"],
        rejected_result_ids=rejected,
        resolution_notes=" | ".join(notes),
        conflict_summary=conflict_report.get("conflict_summary", "")
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_result_id(result: Dict[str, Any]) -> str:
    """Get unique identifier for an optimization result."""
    if "output" in result and "selected_config_id" in result["output"]:
        return str(result["output"]["selected_config_id"])
    elif "test_name" in result:
        return result["test_name"]
    else:
        return f"result_{id(result)}"


def get_priority(result: Dict[str, Any]) -> str:
    """Get priority from optimization result input."""
    return result.get("input", {}).get("priority", "MEDIUM")


# ============================================================================
# MAIN TEST/CLI FUNCTION
# ============================================================================

def main():
    """Test priority-based resolution with test_conflict_detection_result.json"""
    import sys
    
    # Load conflict detection result
    result_file = "test_conflict_detection_result.json"
    
    try:
        with open(result_file, "r") as f:
            detection_result = json.load(f)
    except FileNotFoundError:
        print(f"Error: {result_file} not found. Run test_conflict_detection.py first.")
        sys.exit(1)
    
    print("="*80)
    print("PRIORITY-BASED RESOLUTION TEST")
    print("="*80)
    print()
    
    # Extract data
    conflict_report = detection_result.get("conflict_report", {})
    new_result = detection_result.get("new_intent")
    active_results = detection_result.get("active_intents", [])
    
    print(f"Input Data:")
    print(f"  Conflict Report: {conflict_report.get('num_conflicts', 0)} conflicts")
    print(f"  New Result ID: {get_result_id(new_result)}")
    print(f"  New Result Priority: {get_priority(new_result)}")
    print(f"  Active Results: {len(active_results)}")
    for ar in active_results:
        print(f"    - {get_result_id(ar)} (Priority: {get_priority(ar)})")
    print()
    
    # Run resolution
    print("Running Priority-Based Resolution...")
    print()
    
    resolution = resolve_by_priority(
        conflict_report=conflict_report,
        new_result=new_result,
        active_results=active_results
    )
    
    # Display results
    print("="*80)
    print("PRIORITY-BASED RESOLUTION RESULT:")
    print("="*80)
    print(f"Conflict Detected: {resolution.conflict_detected}")
    print(f"Strategy: {resolution.resolution_strategy}")
    print()
    
    if resolution.conflict_detected:
        print(f"Winner:")
        print(f"  Result ID: {resolution.winning_result_id}")
        print(f"  Priority: {resolution.winning_priority}")
        print()
        
        if resolution.rejected_result_ids:
            print(f"Rejected Results:")
            for rid in resolution.rejected_result_ids:
                print(f"  - {rid}")
            print()
    
    print(f"Resolution Notes:")
    print(f"  {resolution.resolution_notes}")
    print()
    
    print(f"Conflict Summary:")
    print(f"  {resolution.conflict_summary}")
    print()
    
    print("="*80)
    
    # Show winning configuration
    if resolution.winning_config:
        print()
        print("WINNING CONFIGURATION:")
        print("="*80)
        changes = resolution.winning_config.get("output", {}).get("changes", [])
        print(f"Number of changes: {len(changes)}")
        for change in changes:
            param = change.get("param")
            change_val = change.get("change")
            unit = change.get("unit", "")
            print(f"  {param}: {change_val:+.1f} {unit}".strip())
        print()
        
        expected_kpis = resolution.winning_config.get("output", {}).get("expected_kpis", {})
        print(f"Expected KPIs:")
        for kpi, value in expected_kpis.items():
            print(f"  {kpi}: {value}")
        print("="*80)
    
    # Save result
    output_file = "test_priority_based_resolution_result.json"
    with open(output_file, "w") as f:
        json.dump(resolution.model_dump(), f, indent=2)
    
    print()
    print(f"Full result saved to: {output_file}")


if __name__ == "__main__":
    main()
