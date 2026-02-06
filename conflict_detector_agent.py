from __future__ import annotations

import json
import os
from typing import List, Optional, Literal, Set, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# SCHEMAS
# ============================================================================

class ConflictDetail(BaseModel):
    """Details about a specific conflict between optimization results."""
    conflict_type: Literal[
        "PARAMETER_CONFLICT",      # Same parameter, opposite directions
        "RESOURCE_CONTENTION",     # Same parameter, same direction but different magnitude
        "BASE_STATION_CONFLICT",   # Same base station, different parameters
        "BOOLEAN_CONFLICT"         # ON/OFF conflict
    ]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    parameter: str
    intent1_id: str  # ID from optimization output
    intent2_id: str  # ID from optimization output
    intent1_change: float
    intent2_change: float
    base_station: Optional[str] = None  # tx0, tx1, tx2, tx3
    description: str


class ConflictReport(BaseModel):
    """Report of conflicts between optimization results."""
    is_conflicted: bool
    conflict_summary: str
    num_conflicts: int
    details: List[ConflictDetail] = Field(default_factory=list)
    resolution_recommendation: Optional[str] = None
    conflicting_result_ids: List[str] = Field(default_factory=list)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_changes_from_result(result: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract parameter changes from optimization_agent_v2 result.
    
    Args:
        result: Dict from optimization_agent_v2.optimize():
            {
                "test_name": str,
                "output": {
                    "changes": [{"param": "tx0_P_dBm", "change": 3.0, ...}]
                }
            }
    
    Returns:
        Dict mapping parameter to change value: {"tx0_P_dBm": 3.0, ...}
    """
    changes_dict = {}
    
    if "output" in result and "changes" in result["output"]:
        for change_item in result["output"]["changes"]:
            param = change_item.get("param")
            change_value = change_item.get("change")
            
            if param and change_value is not None:
                # Handle boolean changes (e.g., tx3_on: True -> 1.0)
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


def get_target_area(result: Dict[str, Any]) -> str:
    """Get target area from optimization result input."""
    return result.get("input", {}).get("target_area", "unknown")


def get_base_station_from_param(param: str) -> Optional[str]:
    """Extract base station ID from parameter name (e.g., tx0_P_dBm -> tx0)."""
    if param.startswith("tx") and len(param) > 2:
        end_idx = param.find("_")
        if end_idx > 0:
            return param[:end_idx]
    return None


def analyze_parameter_conflict(
    param: str,
    change1: float,
    change2: float,
    result1_id: str,
    result2_id: str,
    same_target_area: bool
) -> Optional[ConflictDetail]:
    """
    Analyze conflict between two parameter changes.
    
    Args:
        param: Parameter name (e.g., tx0_P_dBm)
        change1: Change value from result1
        change2: Change value from result2
        result1_id: ID of first optimization result
        result2_id: ID of second optimization result
        same_target_area: Whether results target same area (increases severity)
    
    Returns:
        ConflictDetail if conflict exists, None otherwise
    """
    base_station = get_base_station_from_param(param)
    
    # Skip if identical changes (agreement, not conflict)
    if abs(change1 - change2) < 0.01:
        return None
    
    # 1. BOOLEAN CONFLICT (ON/OFF parameters)
    if "on" in param.lower():
        # Different boolean values
        if (change1 > 0.5) != (change2 > 0.5):  # One is ON, other is OFF
            severity = "CRITICAL" if same_target_area else "HIGH"
            return ConflictDetail(
                conflict_type="BOOLEAN_CONFLICT",
                severity=severity,
                parameter=param,
                intent1_id=result1_id,
                intent2_id=result2_id,
                intent1_change=change1,
                intent2_change=change2,
                base_station=base_station,
                description=f"Boolean conflict on {param}: Result1 wants {'ON' if change1 > 0.5 else 'OFF'}, Result2 wants {'ON' if change2 > 0.5 else 'OFF'}"
            )
        else:
            return None  # Both want same state (agreement)
    
    # 2. OPPOSITE DIRECTIONS (Most Critical)
    if change1 * change2 < 0:  # Different signs
        magnitude = abs(change1) + abs(change2)
        
        # Severity based on magnitude and area overlap
        if magnitude >= 10:
            severity = "CRITICAL"
        elif magnitude >= 5:
            severity = "CRITICAL" if same_target_area else "HIGH"
        elif magnitude >= 2:
            severity = "HIGH" if same_target_area else "MEDIUM"
        else:
            severity = "MEDIUM" if same_target_area else "LOW"
        
        return ConflictDetail(
            conflict_type="PARAMETER_CONFLICT",
            severity=severity,
            parameter=param,
            intent1_id=result1_id,
            intent2_id=result2_id,
            intent1_change=change1,
            intent2_change=change2,
            base_station=base_station,
            description=f"Direct opposition on {param}: Result1 changes by {change1:+.1f}, Result2 by {change2:+.1f}"
        )
    
    # 3. SAME DIRECTION but large difference (Resource Contention)
    elif abs(change1 - change2) > 5:
        severity = "HIGH" if same_target_area else "MEDIUM"
        
        return ConflictDetail(
            conflict_type="RESOURCE_CONTENTION",
            severity=severity,
            parameter=param,
            intent1_id=result1_id,
            intent2_id=result2_id,
            intent1_change=change1,
            intent2_change=change2,
            base_station=base_station,
            description=f"Large magnitude difference on {param}: Result1 changes by {change1:+.1f}, Result2 by {change2:+.1f}"
        )
    
    # 4. SAME DIRECTION, smaller difference (Coordination needed)
    else:
        severity = "MEDIUM" if same_target_area else "LOW"
        
        return ConflictDetail(
            conflict_type="RESOURCE_CONTENTION",
            severity=severity,
            parameter=param,
            intent1_id=result1_id,
            intent2_id=result2_id,
            intent1_change=change1,
            intent2_change=change2,
            base_station=base_station,
            description=f"Same direction on {param}: Result1 {change1:+.1f}, Result2 {change2:+.1f}. Coordination may be needed."
        )


# ============================================================================
# MAIN CONFLICT DETECTION FUNCTION
# ============================================================================

def detect_conflicts(
    new_result: Dict[str, Any],
    active_results: List[Dict[str, Any]]
) -> ConflictReport:
    """
    Detect conflicts between a new optimization result and active optimization results.
    
    This is the MAIN function - only works with optimization_agent_v2 outputs.
    No intent parsing or LLM calls - pure deterministic conflict detection.
    
    Args:
        new_result: New optimization result from optimization_agent_v2.optimize()
        active_results: List of active optimization results
        
    Returns:
        ConflictReport with details of conflicts
    """
    
    # Extract changes and metadata from new result
    new_changes = extract_changes_from_result(new_result)
    new_id = get_result_id(new_result)
    new_priority = get_priority(new_result)
    new_area = get_target_area(new_result)
    
    conflicts = []
    conflicting_ids = set()
    
    # Compare with each active result
    for active_result in active_results:
        active_changes = extract_changes_from_result(active_result)
        active_id = get_result_id(active_result)
        active_priority = get_priority(active_result)
        active_area = get_target_area(active_result)
        
        same_area = (new_area == active_area)
        
        # Find common parameters
        common_params = set(new_changes.keys()) & set(active_changes.keys())
        
        if not common_params:
            # No overlapping parameters - check if same base stations
            new_bs = {get_base_station_from_param(p) for p in new_changes.keys()}
            active_bs = {get_base_station_from_param(p) for p in active_changes.keys()}
            new_bs.discard(None)
            active_bs.discard(None)
            
            common_bs = new_bs & active_bs
            if common_bs and same_area:
                # Same base stations, different parameters, same area
                conflicts.append(ConflictDetail(
                    conflict_type="BASE_STATION_CONFLICT",
                    severity="MEDIUM" if same_area else "LOW",
                    parameter="multiple",
                    intent1_id=new_id,
                    intent2_id=active_id,
                    intent1_change=0.0,
                    intent2_change=0.0,
                    base_station=", ".join(sorted(common_bs)),
                    description=f"Both results modify {', '.join(sorted(common_bs))} but different parameters"
                ))
                conflicting_ids.update([new_id, active_id])
            continue
        
        # Analyze each common parameter
        for param in common_params:
            conflict = analyze_parameter_conflict(
                param,
                new_changes[param],
                active_changes[param],
                new_id,
                active_id,
                same_area
            )
            
            if conflict:
                conflicts.append(conflict)
                conflicting_ids.update([new_id, active_id])
    
    # Generate report
    if conflicts:
        param_list = [c.parameter for c in conflicts if c.parameter != "multiple"]
        conflict_summary = f"Found {len(conflicts)} conflict(s) between new result ({new_id}) and {len(active_results)} active result(s). "
        if param_list:
            conflict_summary += f"Conflicting parameters: {', '.join(sorted(set(param_list)))}"
        
        # Generate recommendation based on priorities
        priority_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        new_p_level = priority_map.get(new_priority, 2)
        
        # Get highest priority from active results
        active_priorities = [get_priority(r) for r in active_results]
        max_active_p = max([priority_map.get(p, 2) for p in active_priorities])
        
        if new_p_level > max_active_p:
            recommendation = f"New result has higher priority ({new_priority}). Consider prioritizing new result or applying conflict resolution."
        elif new_p_level < max_active_p:
            recommendation = f"Active results have higher priority. Consider rejecting new result or applying conflict resolution."
        else:
            recommendation = f"Equal priorities ({new_priority}). Apply conflict resolution: weighted merge, sequential application, or user arbitration."
        
        return ConflictReport(
            is_conflicted=True,
            conflict_summary=conflict_summary,
            num_conflicts=len(conflicts),
            details=conflicts,
            resolution_recommendation=recommendation,
            conflicting_result_ids=sorted(list(conflicting_ids))
        )
    else:
        return ConflictReport(
            is_conflicted=False,
            conflict_summary=f"No conflicts detected. New result ({new_id}) can be applied safely with {len(active_results)} active result(s).",
            num_conflicts=0,
            details=[],
            resolution_recommendation="Proceed with applying new result.",
            conflicting_result_ids=[]
        )
