from __future__ import annotations

import json
import os
from typing import List, Optional, Literal
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
    conflict_type: Literal["RESOURCE_CONTENTION", "DIRECT_OPPOSITION", "SPATIAL_OVERLAP"]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    conflicting_param: Optional[str] = None
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

def _analyze_parameter_conflict(
    param: str, 
    change_a: ParamChange, 
    change_b: ParamChange
) -> Optional[ConflictDetail]:
    """
    Analyzes conflict between two parameter changes.
    In Optimization Agent logic, 'after' for numeric values represents the DELTA.
    """
    
    # 1. Boolean Conflict (e.g., One sets ON, other sets OFF)
    if "on" in param:
        val_a = change_a.after
        val_b = change_b.after
        if val_a != val_b:
            return ConflictDetail(
                conflict_type="DIRECT_OPPOSITION",
                severity="CRITICAL",
                conflicting_param=param,
                description=f"Intent A wants {val_a}, Intent B wants {val_b}.",
                conflicting_intent_id="active_intent"
            )
        else:
            # Both want the same state (e.g., both ON) - Low severity
            return ConflictDetail(
                conflict_type="RESOURCE_CONTENTION",
                severity="LOW",
                conflicting_param=param,
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

    # Opposite Directions (One Increases, One Decreases) -> CRITICAL
    if (delta_a > 0 and delta_b < 0) or (delta_a < 0 and delta_b > 0):
        return ConflictDetail(
            conflict_type="DIRECT_OPPOSITION",
            severity="CRITICAL",
            conflicting_param=param,
            description=f"Intent A modifies by {delta_a} (Increase), Intent B modifies by {delta_b} (Decrease).",
            conflicting_intent_id="active_intent"
        )
    
    # Same Direction (Both Increase) -> Resource Contention / Saturation Risk
    return ConflictDetail(
        conflict_type="RESOURCE_CONTENTION",
        severity="HIGH",
        conflicting_param=param,
        description=f"Both intents increase/decrease parameter. A: {delta_a}, B: {delta_b}. Risk of over-saturation.",
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
    Analyzes conflicts and prepares the full input package for the Meta-Agent.
    
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
    
    # Add the NEW proposal first
    proposals.append(AgentProposal(
        agent_id="new_intent_agent",
        intent=new_intent,
        plan=new_plan
    ))

    # Add ACTIVE proposals and check for conflicts
    new_changes_map = {c.param: c for c in new_plan.changes}

    for idx, item in enumerate(active_list):
        active_intent = IntentParse(**item["intent"])
        active_plan = OptimizationPlan(**item["plan"])
        
        # Add to proposals list for Meta-Agent
        proposals.append(AgentProposal(
            agent_id=f"active_intent_{idx}_{active_plan.selected_config_id}",
            intent=active_intent,
            plan=active_plan
        ))

        # --- A. SPATIAL OVERLAP CHECK ---
        if new_intent.target_area == active_intent.target_area:
            active_changes_map = {c.param: c for c in active_plan.changes}
            
            # --- B. PARAMETER CONFLICT CHECK ---
            common_params = set(new_changes_map.keys()) & set(active_changes_map.keys())
            
            if not common_params:
                conflicts.append(ConflictDetail(
                    conflict_type="SPATIAL_OVERLAP",
                    severity="LOW",
                    description=f"Both intents target {new_intent.target_area} but modify different parameters.",
                    conflicting_intent_id=str(active_plan.selected_config_id)
                ))
            else:
                for param in common_params:
                    detail = _analyze_parameter_conflict(
                        param, 
                        new_changes_map[param], 
                        active_changes_map[param]
                    )
                    if detail:
                        detail.conflicting_intent_id = str(active_plan.selected_config_id)
                        conflicts.append(detail)
        else:
            # Different areas -> No conflict
            continue

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
        
        summary = f"Detected {len(conflicts)} conflicts. Max severity: {max_severity}."
    else:
        summary = "No conflicts detected. Intents operate on disjoint resources or areas."

    conflict_report = ConflictReport(
        is_conflicted=is_conflicted,
        conflict_summary=summary,
        details=conflicts,
        resolution_recommendation="Review CRITICAL conflicts. If compatible, merge HIGH conflicts." if is_conflicted else "Proceed with execution."
    )
    
    # 4. Construct Final Meta-Agent Input Package
    meta_input = MetaArbitrationInput(
        conflict_report=conflict_report,
        proposals=proposals,
        current_config_id=new_plan.current_config_id
    )
    
    return meta_input.model_dump_json(indent=2)

# -----------------------------
# 4) AGENT DEFINITION
# -----------------------------

CONFLICT_INSTRUCTIONS = [
    "You are a Conflict Detection Agent (CDA) for a 6G Network Management System.",
    "Your Goal: Analyze a proposed Optimization Plan against Active Intents and prepare data for the Meta-Agent.",
    "Input: New Intent/Plan and a list of Active Intents/Plans.",
    "Conflict Logic:",
    "1. SPATIAL OVERLAP: Do intents target the same area?",
    "2. RESOURCE CONTENTION: Do plans modify the same parameter (e.g. tx0_P_dBm)?",
    "3. DIRECT OPPOSITION: Does one agent Increase (+) and another Decrease (-) the same parameter? This is CRITICAL.",
    "Output: A structured JSON (MetaArbitrationInput) containing the ConflictReport AND all AgentProposals."
]

conflict_detector_agent = Agent(
    name="Conflict Detector Agent",
    description="Analyzes network optimization plans for conflicts and packages proposals for arbitration.",
    model=Gemini(id=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")),
    tools=[detect_conflicts],
    output_schema=MetaArbitrationInput, # Output schema updated
    instructions=CONFLICT_INSTRUCTIONS,
)