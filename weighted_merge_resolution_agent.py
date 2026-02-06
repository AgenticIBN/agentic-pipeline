#!/usr/bin/env python3
"""
Weighted Merge Resolution Agent
Creates merged configuration using priority-based weights
Works with conflict_detector_agent.py outputs
"""
from __future__ import annotations

import json
from typing import List, Optional, Dict, Any, Set
from pydantic import BaseModel, Field


# ============================================================================
# SCHEMAS
# ============================================================================

class WeightedMergeOutput(BaseModel):
    """Output from weighted merge resolution."""
    conflict_detected: bool
    resolution_strategy: str
    merged_result_id: str
    contributing_results: List[Dict[str, Any]] = Field(default_factory=list)  # [{"id": "...", "priority": "...", "weight": ...}]
    merged_config: Dict[str, Any]  # The merged optimization result
    merge_details: List[Dict[str, Any]] = Field(default_factory=list)  # Details per parameter
    resolution_notes: str
    conflict_summary: Optional[str] = None


# ============================================================================
# PRIORITY RANKING & WEIGHTS
# ============================================================================

PRIORITY_WEIGHT = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1
}


# ============================================================================
# MAIN RESOLUTION FUNCTION
# ============================================================================

def resolve_by_weighted_merge(
    conflict_report: Dict[str, Any],
    new_result: Dict[str, Any],
    active_results: List[Dict[str, Any]]
) -> WeightedMergeOutput:
    """
    Weighted merge resolution: Merge all results using priority-based weights.
    
    Priority weights: CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1
    Equal priorities get equal weights.
    
    Args:
        conflict_report: Dict from ConflictReport.model_dump()
        new_result: New optimization result
        active_results: List of active optimization results
        
    Returns:
        WeightedMergeOutput with merged configuration
    """
    
    # Check if conflict exists
    if not conflict_report.get("is_conflicted", False):
        # No conflict - just return new result as-is
        return WeightedMergeOutput(
            conflict_detected=False,
            resolution_strategy="NO_CONFLICT",
            merged_result_id=get_result_id(new_result),
            contributing_results=[{
                "id": get_result_id(new_result),
                "priority": get_priority(new_result),
                "weight": 1.0
            }],
            merged_config=new_result,
            merge_details=[],
            resolution_notes="No conflicts detected. New result used as-is.",
            conflict_summary=conflict_report.get("conflict_summary", "")
        )
    
    # Collect all results
    all_results = [new_result] + active_results
    
    # Calculate weights for each result
    result_info = []
    total_weight = 0
    
    for result in all_results:
        result_id = get_result_id(result)
        priority = get_priority(result)
        weight = PRIORITY_WEIGHT.get(priority, 1)
        
        result_info.append({
            "result": result,
            "id": result_id,
            "priority": priority,
            "weight": weight
        })
        total_weight += weight
    
    # Normalize weights (optional, for percentage display)
    for info in result_info:
        info["weight_normalized"] = info["weight"] / total_weight
    
    # Extract all parameters from all results
    all_params = set()
    result_changes = {}
    
    for info in result_info:
        changes = extract_changes_from_result(info["result"])
        result_changes[info["id"]] = changes
        all_params.update(changes.keys())
    
    # Merge parameters using weighted average
    merged_changes = {}
    merge_details = []
    
    for param in all_params:
        param_contributions = []
        weighted_sum = 0
        weight_sum = 0
        
        for info in result_info:
            changes = result_changes[info["id"]]
            if param in changes:
                change_value = changes[param]
                weight = info["weight"]
                
                weighted_sum += change_value * weight
                weight_sum += weight
                
                param_contributions.append({
                    "result_id": info["id"],
                    "priority": info["priority"],
                    "weight": weight,
                    "value": change_value
                })
        
        # Calculate weighted average
        if weight_sum > 0:
            merged_value = weighted_sum / weight_sum
            merged_changes[param] = merged_value
            
            # Build formula string
            formula_parts = [f"{c['value']}*{c['weight']}" for c in param_contributions]
            formula_str = f"({' + '.join(formula_parts)}) / {weight_sum}"
            
            merge_details.append({
                "parameter": param,
                "merged_value": round(merged_value, 2),
                "contributions": param_contributions,
                "formula": formula_str
            })
    
    # Build merged result structure
    merged_result = create_merged_result(
        new_result,
        merged_changes,
        result_info
    )
    
    # Generate resolution notes
    notes = []
    notes.append(f"Conflicts: {conflict_report.get('num_conflicts', 0)}")
    notes.append(f"Strategy: WEIGHTED_MERGE")
    notes.append(f"Contributing results: {len(result_info)}")
    for info in result_info:
        notes.append(f"  - {info['id']} (Priority: {info['priority']}, Weight: {info['weight']}, {info['weight_normalized']*100:.1f}%)")
    notes.append(f"Merged parameters: {len(merged_changes)}")
    
    contributing_results = [
        {
            "id": info["id"],
            "priority": info["priority"],
            "weight": info["weight"],
            "weight_pct": f"{info['weight_normalized']*100:.1f}%"
        }
        for info in result_info
    ]
    
    return WeightedMergeOutput(
        conflict_detected=True,
        resolution_strategy="WEIGHTED_MERGE",
        merged_result_id=f"merged_{len(result_info)}_results",
        contributing_results=contributing_results,
        merged_config=merged_result,
        merge_details=merge_details,
        resolution_notes=" | ".join(notes),
        conflict_summary=conflict_report.get("conflict_summary", "")
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_changes_from_result(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract parameter changes from optimization result."""
    changes_dict = {}
    
    if "output" in result and "changes" in result["output"]:
        for change_item in result["output"]["changes"]:
            param = change_item.get("param")
            change_value = change_item.get("change")
            
            if param and change_value is not None:
                # Handle boolean changes
                if isinstance(change_value, bool):
                    change_value = 1.0 if change_value else 0.0
                else:
                    change_value = float(change_value)
                
                changes_dict[param] = change_value
    
    return changes_dict


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


def create_merged_result(
    template_result: Dict[str, Any],
    merged_changes: Dict[str, float],
    result_info: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create a merged optimization result structure.
    
    Args:
        template_result: Use as template for structure
        merged_changes: Dict of merged parameter values
        result_info: List of contributing results with their info
    
    Returns:
        Merged result in optimization_agent_v2 format
    """
    # Build changes list
    changes_list = []
    for param, value in merged_changes.items():
        # Determine unit from parameter name
        if "P_dBm" in param:
            unit = "dBm"
        elif "dAz" in param or "dEl" in param:
            unit = "deg"
        elif "_on" in param:
            unit = None
        else:
            unit = None
        
        changes_list.append({
            "param": param,
            "before": 0.0,  # We don't track original values in merge
            "change": round(value, 2),
            "unit": unit
        })
    
    # Sort by parameter name for consistency
    changes_list.sort(key=lambda x: x["param"])
    
    # Merge KPIs (use weighted average of expected KPIs)
    merged_kpis = merge_kpis(result_info)
    
    # Create merged result
    merged = {
        "test_name": "weighted_merge",
        "passed": True,
        "input": {
            "target_area": template_result.get("input", {}).get("target_area", "merged_area"),
            "target_kpis": list(merged_kpis.keys()),
            "priority": "MERGED",
        },
        "output": {
            "selected_config_id": "merged_config",
            "current_config_id": None,
            "changes": changes_list,
            "expected_kpis": merged_kpis,
            "constraints_satisfied": True,
        }
    }
    
    return merged


def merge_kpis(result_info: List[Dict[str, Any]]) -> Dict[str, float]:
    """Merge expected KPIs using weighted average."""
    all_kpis = set()
    
    # Collect all KPI names
    for info in result_info:
        result = info["result"]
        kpis = result.get("output", {}).get("expected_kpis", {})
        all_kpis.update(kpis.keys())
    
    # Merge each KPI
    merged_kpis = {}
    total_weight = sum(info["weight"] for info in result_info)
    
    for kpi_name in all_kpis:
        weighted_sum = 0
        weight_sum = 0
        
        for info in result_info:
            result = info["result"]
            kpis = result.get("output", {}).get("expected_kpis", {})
            
            if kpi_name in kpis:
                kpi_value = kpis[kpi_name]
                weight = info["weight"]
                
                weighted_sum += kpi_value * weight
                weight_sum += weight
        
        if weight_sum > 0:
            merged_kpis[kpi_name] = round(weighted_sum / weight_sum, 2)
    
    return merged_kpis


# ============================================================================
# MAIN TEST/CLI FUNCTION
# ============================================================================

def main():
    """Test weighted merge resolution with test_conflict_detection_result.json"""
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
    print("WEIGHTED MERGE RESOLUTION TEST")
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
    print("Running Weighted Merge Resolution...")
    print()
    
    resolution = resolve_by_weighted_merge(
        conflict_report=conflict_report,
        new_result=new_result,
        active_results=active_results
    )
    
    # Display results
    print("="*80)
    print("WEIGHTED MERGE RESOLUTION RESULT:")
    print("="*80)
    print(f"Conflict Detected: {resolution.conflict_detected}")
    print(f"Strategy: {resolution.resolution_strategy}")
    print()
    
    if resolution.conflict_detected:
        print(f"Contributing Results:")
        for contrib in resolution.contributing_results:
            print(f"  - {contrib['id']}: Priority={contrib['priority']}, Weight={contrib['weight']} ({contrib['weight_pct']})")
        print()
        
        print(f"Merge Details (Parameter-by-Parameter):")
        print("-" * 80)
        for detail in resolution.merge_details:
            print(f"\n  Parameter: {detail['parameter']}")
            print(f"  Merged Value: {detail['merged_value']:+.2f}")
            print(f"  Contributions:")
            for contrib in detail['contributions']:
                print(f"    - Result {contrib['result_id']} ({contrib['priority']}): {contrib['value']:+.2f} × weight {contrib['weight']}")
            print(f"  Formula: {detail['formula']}")
        print()
    
    print(f"Resolution Notes:")
    print(f"  {resolution.resolution_notes}")
    print()
    
    print(f"Conflict Summary:")
    print(f"  {resolution.conflict_summary}")
    print()
    
    print("="*80)
    
    # Show merged configuration
    if resolution.merged_config:
        print()
        print("MERGED CONFIGURATION:")
        print("="*80)
        changes = resolution.merged_config.get("output", {}).get("changes", [])
        print(f"Number of parameters: {len(changes)}")
        for change in changes:
            param = change.get("param")
            change_val = change.get("change")
            unit = change.get("unit", "")
            print(f"  {param}: {change_val:+.2f} {unit}".strip())
        print()
        
        expected_kpis = resolution.merged_config.get("output", {}).get("expected_kpis", {})
        print(f"Expected KPIs (weighted average):")
        for kpi, value in expected_kpis.items():
            print(f"  {kpi}: {value}")
        print("="*80)
    
    # Save result
    output_file = "test_weighted_merge_resolution_result.json"
    with open(output_file, "w") as f:
        json.dump(resolution.model_dump(), f, indent=2)
    
    print()
    print(f"Full result saved to: {output_file}")


if __name__ == "__main__":
    main()
