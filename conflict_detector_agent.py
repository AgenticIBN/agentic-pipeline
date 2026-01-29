from __future__ import annotations

import json
import os
from typing import List, Optional, Literal, Set, Dict
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
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
        val_a = change_a.after
        val_b = change_b.after
        if val_a != val_b:
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
            # Both want the same state (e.g., both ON) - Low severity
            severity = "MEDIUM" if same_target_area else "LOW"
            return ConflictDetail(
                conflict_type="RESOURCE_CONTENTION",
                severity=severity,
                conflicting_param=param,
                conflicting_base_station=bs,
                description=f"Both intents want to set {param} to {val_a}.",
                conflicting_intent_id="active_intent"
            )

    # 2. Numeric Conflict (Power, Tilt, Azimuth)
    # 'after' is a delta value (e.g., +3.0 or -5.0)
    try:
        delta_a = float(change_a.after)
        delta_b = float(change_b.after)
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
        
        # --- A. BASE STATION LEVEL CONFLICT CHECK ---
        common_base_stations = set(new_bs_params.keys()) & set(active_bs_params.keys())
        
        if not common_base_stations:
            # Different base stations - Very low conflict (coordination only)
            conflicts.append(ConflictDetail(
                conflict_type="SPATIAL_OVERLAP",
                severity="LOW",
                description=f"Intents modify different base stations. New: {list(new_bs_params.keys())}, Active: {list(active_bs_params.keys())}. Target areas: {new_intent.target_area} vs {active_intent.target_area}",
                conflicting_intent_id=str(active_plan.selected_config_id)
            ))
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
    "1. BASE STATION OVERLAP: Do plans modify the same base station (tx0-tx3)?",
    "   - Different base stations → LOW severity (minimal conflict)",
    "   - Same base station, different params → LOW-MEDIUM severity",
    "   - Same base station, same param → Detailed analysis",
    "",
    "2. PARAMETER CONFLICT: For common base stations and parameters:",
    "   - Boolean params (tx_on): Different values → HIGH/CRITICAL",
    "   - Numeric params: Opposite directions (+/-) → HIGH/CRITICAL",
    "   - Numeric params: Same direction → MEDIUM/HIGH (saturation risk)",
    "",
    "3. SEVERITY BOOSTING: If target_area also matches, increase severity by one level.",
    "   Example: HIGH → CRITICAL, MEDIUM → HIGH",
    "",
    "Output: MetaArbitrationInput with detailed ConflictReport and all AgentProposals.",
]

conflict_detector_agent = Agent(
    name="Base Station Level Conflict Detector",
    description="Analyzes network optimization plans for conflicts at base station level.",
    model=Gemini(id=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")),
    tools=[detect_conflicts],
    output_schema=MetaArbitrationInput,
    instructions=CONFLICT_INSTRUCTIONS,
)
