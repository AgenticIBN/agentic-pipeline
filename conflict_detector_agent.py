from __future__ import annotations

import json
import os
from typing import List, Optional, Literal, Set, Dict
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Import schemas from optimization_agent
from optimization_agent import OptimizationPlan, ParamChange, IntentParse

load_dotenv()

# -----------------------------
# 1) SHARED SCHEMAS (For Meta-Agent Handoff)
# -----------------------------

class ConflictDetail(BaseModel):
    conflict_type: Literal["RESOURCE_CONTENTION", "DIRECT_OPPOSITION", "SPATIAL_OVERLAP", "BASE_STATION_CONFLICT"]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    conflicting_param: Optional[str] = None
    conflicting_base_station: Optional[str] = None  # NEW: tx0, tx1, tx2, tx3
    description: str
    conflicting_intent_id: str

class ConflictReport(BaseModel):
    is_conflicted: bool
    conflict_summary: str
    details: List[ConflictDetail] = Field(default_factory=list)
    resolution_recommendation: Optional[str] = None

class AgentProposal(BaseModel):
    """
    Wraps an intent and its proposed plan.
    Used by Meta-Agent to evaluate trade-offs.
    """
    agent_id: str
    intent: IntentParse
    plan: OptimizationPlan

class MetaArbitrationInput(BaseModel):
    """
    The full package required by the Meta-Agent to resolve conflicts.
    This is now the output format of the Conflict Detector.
    """
    conflict_report: ConflictReport
    proposals: List[AgentProposal]
    current_config_id: Optional[int] = None

# -----------------------------
# 2) HELPER LOGIC
# -----------------------------

def _extract_base_station_from_param(param: str) -> Optional[str]:
    """
    Extract base station ID from parameter name.
    Example: tx0_P_dBm -> tx0, tx1_dAz -> tx1
    """
    if param.startswith("tx"):
        parts = param.split("_")
        if len(parts) > 0:
            return parts[0]  # tx0, tx1, tx2, tx3
    return None

def _check_time_overlap(intent_a: IntentParse, intent_b: IntentParse) -> Optional[ConflictDetail]:
    """
    Check if two intents have overlapping time constraints.
    Returns ConflictDetail if overlap exists, None otherwise.
    
    Time format expected: ISO 8601 (e.g., "2026-02-05T10:00:00")
    """
    from datetime import datetime
    
    # If either intent has no time constraints, consider no time conflict
    if not (intent_a.time_constraint_start and intent_a.time_constraint_end):
        return None
    if not (intent_b.time_constraint_start and intent_b.time_constraint_end):
        return None
    
    try:
        # Parse time strings
        a_start = datetime.fromisoformat(intent_a.time_constraint_start)
        a_end = datetime.fromisoformat(intent_a.time_constraint_end)
        b_start = datetime.fromisoformat(intent_b.time_constraint_start)
        b_end = datetime.fromisoformat(intent_b.time_constraint_end)
        
        # Check for overlap: A and B overlap if (A.start < B.end) AND (B.start < A.end)
        has_overlap = (a_start < b_end) and (b_start < a_end)
        
        if has_overlap:
            # Calculate overlap duration
            overlap_start = max(a_start, b_start)
            overlap_end = min(a_end, b_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds() / 3600  # hours
            
            return ConflictDetail(
                conflict_type="RESOURCE_CONTENTION",
                severity="HIGH",
                description=f"Time overlap detected. Intent A: {a_start} to {a_end}, Intent B: {b_start} to {b_end}. Overlap: {overlap_duration:.2f} hours.",
                conflicting_intent_id="time_conflict"
            )
        else:
            return None
            
    except (ValueError, AttributeError) as e:
        # Invalid time format, skip time conflict check
        return None

def _get_base_station_params(plan: OptimizationPlan) -> Dict[str, List[str]]:
    """
    Group parameters by base station.
    Returns: {tx0: [tx0_P_dBm, tx0_dAz], tx1: [...], ...}
    """
    bs_params = {}
    for change in plan.changes:
        bs = _extract_base_station_from_param(change.param)
        if bs:
            if bs not in bs_params:
                bs_params[bs] = []
            bs_params[bs].append(change.param)
    return bs_params

def _analyze_parameter_conflict(
    param: str, 
    change_a: ParamChange, 
    change_b: ParamChange,
    same_target_area: bool
) -> Optional[ConflictDetail]:
    """
    Analyzes conflict between two parameter changes.
    In Optimization Agent logic, 'after' for numeric values represents the DELTA.
    
    same_target_area: If True, increases severity (more critical)
    """
    
    bs = _extract_base_station_from_param(param)
    
    # Severity multiplier based on target area overlap
    severity_boost = 1 if same_target_area else 0
    
    # 1. Boolean Conflict (e.g., One sets ON, other sets OFF)
    if "on" in param.lower():
        val_a = change_a.change
        val_b = change_b.change
        if val_a != val_b:
            # TRUE CONFLICT: One wants ON, other wants OFF
            severity = "CRITICAL" if same_target_area else "HIGH"
            return ConflictDetail(
                conflict_type="DIRECT_OPPOSITION",
                severity=severity,
                conflicting_param=param,
                conflicting_base_station=bs,
                description=f"Intent A wants {val_a}, Intent B wants {val_b} for {param}. Same target area: {same_target_area}",
                conflicting_intent_id="active_intent"
            )
        else:
            # NO CONFLICT: Both want the same state (e.g., both ON)
            # This is agreement, not conflict
            return None

    # 2. Numeric Conflict (Power, Tilt, Azimuth)
    # 'change' is a delta value (e.g., +3.0 or -5.0)
    try:
        delta_a = float(change_a.change)
        delta_b = float(change_b.change)
    except (ValueError, TypeError):
        return None

    # Opposite Directions (One Increases, One Decreases) -> CRITICAL/HIGH
    if (delta_a > 0 and delta_b < 0) or (delta_a < 0 and delta_b > 0):
        severity = "CRITICAL" if same_target_area else "HIGH"
        return ConflictDetail(
            conflict_type="DIRECT_OPPOSITION",
            severity=severity,
            conflicting_param=param,
            conflicting_base_station=bs,
            description=f"Intent A modifies {param} by {delta_a}, Intent B by {delta_b} (opposite directions). Same area: {same_target_area}",
            conflicting_intent_id="active_intent"
        )
    
    # Same Direction (Both Increase) -> Resource Contention / Saturation Risk
    severity = "HIGH" if same_target_area else "MEDIUM"
    return ConflictDetail(
        conflict_type="RESOURCE_CONTENTION",
        severity=severity,
        conflicting_param=param,
        conflicting_base_station=bs,
        description=f"Both intents modify {param} in same direction. A: {delta_a}, B: {delta_b}. Risk of over-saturation. Same area: {same_target_area}",
        conflicting_intent_id="active_intent"
    )

# -----------------------------
# 3) TOOL DEFINITION
# -----------------------------

def detect_conflicts(
    new_intent_json: str, 
    new_plan_json: str, 
    active_intents_data: str
) -> str:
    """
    Analyzes conflicts at BASE STATION level and prepares the full input package for the Meta-Agent.
    
    KEY CHANGE: Conflict detection is based on which BASE STATIONS are affected,
    not just target_area. Since all intents affect the same 4 base stations (tx0-tx3),
    conflicts are detected when:
    1. Same base station is modified by multiple intents
    2. Same parameter on same base station is modified
    3. Severity is increased if target_area also overlaps
    
    Args:
        new_intent_json: JSON string of the new IntentParse.
        new_plan_json: JSON string of the proposed OptimizationPlan.
        active_intents_data: JSON string list of dicts [{"intent": IntentParse, "plan": OptimizationPlan}, ...].
        
    Returns:
        JSON string of MetaArbitrationInput (containing the report AND all proposals).
    """
    
    # 1. Parse Inputs
    new_intent = IntentParse.model_validate_json(new_intent_json)
    new_plan = OptimizationPlan.model_validate_json(new_plan_json)
    
    try:
        active_list = json.loads(active_intents_data)
    except:
        return json.dumps({"error": "Invalid active_intents_data format"})

    conflicts = []
    proposals = []

    # 2. Prepare Proposal List (Active + New)
    proposals.append(AgentProposal(
        agent_id="new_intent_agent",
        intent=new_intent,
        plan=new_plan
    ))

    # Extract base station parameters from new plan
    new_changes_map = {c.param: c for c in new_plan.changes}
    new_bs_params = _get_base_station_params(new_plan)

    for idx, item in enumerate(active_list):
        active_intent = IntentParse(**item["intent"])
        active_plan = OptimizationPlan(**item["plan"])
        
        # Add to proposals list for Meta-Agent
        proposals.append(AgentProposal(
            agent_id=f"active_intent_{idx}_{active_plan.selected_config_id}",
            intent=active_intent,
            plan=active_plan
        ))

        active_changes_map = {c.param: c for c in active_plan.changes}
        active_bs_params = _get_base_station_params(active_plan)
        
        # Check if target areas overlap (used for severity boosting)
        same_target_area = (new_intent.target_area == active_intent.target_area)
        
        # --- TIME OVERLAP CHECK (Priority Check) ---
        time_conflict = _check_time_overlap(new_intent, active_intent)
        if time_conflict:
            time_conflict.conflicting_intent_id = str(active_plan.selected_config_id)
            conflicts.append(time_conflict)
        
        # --- A. BASE STATION LEVEL CONFLICT CHECK ---
        common_base_stations = set(new_bs_params.keys()) & set(active_bs_params.keys())
        
        if not common_base_stations:
            # Different base stations
            if same_target_area:
                # Same target area but different BSs - Low coordination needed
                conflicts.append(ConflictDetail(
                    conflict_type="SPATIAL_OVERLAP",
                    severity="LOW",
                    description=f"Same target area ({new_intent.target_area}) but different base stations. New: {list(new_bs_params.keys())}, Active: {list(active_bs_params.keys())}. May need coordination.",
                    conflicting_intent_id=str(active_plan.selected_config_id)
                ))
            # else: Different BS + Different Area = No conflict at all, skip
            continue
        
        # --- B. PARAMETER LEVEL CONFLICT CHECK (on common base stations) ---
        common_params = set(new_changes_map.keys()) & set(active_changes_map.keys())
        
        if not common_params:
            # Same base stations, different parameters
            severity = "MEDIUM" if same_target_area else "LOW"
            conflicts.append(ConflictDetail(
                conflict_type="BASE_STATION_CONFLICT",
                severity=severity,
                description=f"Both intents modify same base stations {list(common_base_stations)} but different parameters. Same target area: {same_target_area}",
                conflicting_intent_id=str(active_plan.selected_config_id)
            ))
        else:
            # Same base stations, same parameters - Detailed conflict analysis
            for param in common_params:
                detail = _analyze_parameter_conflict(
                    param, 
                    new_changes_map[param], 
                    active_changes_map[param],
                    same_target_area
                )
                if detail:
                    detail.conflicting_intent_id = str(active_plan.selected_config_id)
                    conflicts.append(detail)

    # 3. Create Report
    is_conflicted = len(conflicts) > 0
    max_severity = "LOW"
    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    
    if is_conflicted:
        current_max = 0
        for c in conflicts:
            if severity_rank[c.severity] > current_max:
                current_max = severity_rank[c.severity]
                max_severity = c.severity
        
        summary = f"Detected {len(conflicts)} conflicts. Max severity: {max_severity}. Base station-level analysis."
    else:
        summary = "No conflicts detected. Intents operate on completely disjoint resources."

    conflict_report = ConflictReport(
        is_conflicted=is_conflicted,
        conflict_summary=summary,
        details=conflicts,
        resolution_recommendation=_get_recommendation(max_severity, is_conflicted)
    )
    
    # 4. Construct Final Meta-Agent Input Package
    meta_input = MetaArbitrationInput(
        conflict_report=conflict_report,
        proposals=proposals,
        current_config_id=new_plan.current_config_id
    )
    
    return meta_input.model_dump_json(indent=2)

def _get_recommendation(max_severity: str, is_conflicted: bool) -> str:
    """Generate resolution recommendation based on severity"""
    if not is_conflicted:
        return "Proceed with execution."
    
    if max_severity == "CRITICAL":
        return "CRITICAL conflicts detected. Meta-Agent must resolve immediately. Consider rejecting or sequencing intents."
    elif max_severity == "HIGH":
        return "HIGH conflicts detected. Meta-Agent should evaluate trade-offs and potentially merge or sequence plans."
    elif max_severity == "MEDIUM":
        return "MEDIUM conflicts detected. Coordinate changes on same base stations. Merge if compatible."
    else:
        return "LOW conflicts detected. Consider coordination but parallel execution may be safe."

# -----------------------------
# 4) AGENT DEFINITION
# -----------------------------

CONFLICT_INSTRUCTIONS = [
    "You are a BASE STATION LEVEL Conflict Detection Agent (CDA) for a 6G Network Management System.",
    "Your Goal: Analyze proposed Optimization Plans at the BASE STATION level against Active Intents.",
    "",
    "CRITICAL: This system uses 4 shared base stations (tx0, tx1, tx2, tx3).",
    "Even if intents target different geographic areas, they may conflict if they modify the SAME base stations.",
    "",
    "Conflict Detection Logic:",
    "1. TIME OVERLAP CHECK (First Priority):",
    "   - If time_constraint_start/end overlap → HIGH severity conflict",
    "   - No time constraints = no time conflict",
    "",
    "2. BASE STATION OVERLAP: Do plans modify the same base station (tx0-tx3)?",
    "   - Different BS + Different Area → NO CONFLICT (skip)",
    "   - Different BS + Same Area → LOW severity (coordination)",
    "   - Same BS, different params → LOW-MEDIUM severity",
    "   - Same BS, same param → Detailed analysis",
    "",
    "3. PARAMETER CONFLICT: For common base stations and parameters:",
    "   - Boolean params (tx_on): Same value → NO CONFLICT (agreement)",
    "   - Boolean params (tx_on): Different values → HIGH/CRITICAL",
    "   - Numeric params: Opposite directions (+/-) → HIGH/CRITICAL",
    "   - Numeric params: Same direction → MEDIUM/HIGH (saturation risk)",
    "",
    "4. SEVERITY BOOSTING: If target_area also matches, increase severity by one level.",
    "   Example: HIGH → CRITICAL, MEDIUM → HIGH",
    "",
    "Output: MetaArbitrationInput with detailed ConflictReport and all AgentProposals.",
]

conflict_detector_agent = Agent(
    name="Base Station Level Conflict Detector",
    description="Analyzes network optimization plans for conflicts at base station level.",
    model=Groq(id=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")),
    tools=[detect_conflicts],
    output_schema=MetaArbitrationInput,
    instructions=CONFLICT_INSTRUCTIONS,
)
