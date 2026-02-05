from __future__ import annotations

import json
import os
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Import schemas
from conflict_detector_agent import MetaArbitrationInput, AgentProposal, ConflictReport
from optimization_agent import OptimizationPlan, ParamChange, IntentParse, KpiSnapshot

load_dotenv()

# -----------------------------
# 1) OUTPUT SCHEMA
# -----------------------------

class ConflictResolutionOutput(BaseModel):
    """
    Complete output from the Conflict Resolution Agent
    Simple priority-based selection strategy
    """
    conflict_detected: bool
    resolution_applied: bool
    winning_intent_id: Optional[str] = None  # ID of the winning intent
    winning_priority: Optional[str] = None  # Priority level of winner
    selected_plan: Optional[OptimizationPlan] = None  # The plan from winning intent
    rejected_intents: List[str] = Field(default_factory=list)  # IDs of rejected intents
    resolution_notes: str

# -----------------------------
# 2) PRIORITY RANKING
# -----------------------------

PRIORITY_RANK = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1
}

# -----------------------------
# 3) PRIORITY SELECTION LOGIC
# -----------------------------

def select_highest_priority_intent(
    proposals: List[AgentProposal]
) -> AgentProposal:
    """
    Simple priority-based selection: Choose the intent with highest priority.
    
    Priority order: CRITICAL > HIGH > MEDIUM > LOW
    
    If multiple intents have same priority, choose first one (FIFO).
    
    Args:
        proposals: List of agent proposals to choose from
        
    Returns:
        The winning proposal
    """
    if not proposals:
        raise ValueError("No proposals provided for selection")
    
    # Sort by priority rank (descending)
    sorted_proposals = sorted(
        proposals, 
        key=lambda p: PRIORITY_RANK.get(p.intent.priority, 0),
        reverse=True
    )
    
    # Return highest priority
    return sorted_proposals[0]

# -----------------------------
# 4) MAIN RESOLUTION TOOL
# -----------------------------

def resolve_conflicts(meta_input_json: str) -> str:
    """
    Simple conflict resolution: Select the intent with highest priority.
    
    Strategy: CRITICAL > HIGH > MEDIUM > LOW
    No weighted merging, no complex calculations - just pick the winner.
    
    Args:
        meta_input_json: JSON string of MetaArbitrationInput from Conflict Detector
        
    Returns:
        JSON string of ConflictResolutionOutput
    """
    
    # Parse input
    meta_input = MetaArbitrationInput.model_validate_json(meta_input_json)
    conflict_report = meta_input.conflict_report
    proposals = meta_input.proposals
    
    # If no conflict, return indication that all can proceed
    if not conflict_report.is_conflicted:
        return ConflictResolutionOutput(
            conflict_detected=False,
            resolution_applied=False,
            winning_intent_id=None,
            winning_priority=None,
            selected_plan=None,
            rejected_intents=[],
            resolution_notes="No conflicts detected. All intents can proceed independently."
        ).model_dump_json(indent=2)
    
    # Select highest priority intent
    winner = select_highest_priority_intent(proposals)
    
    # Collect rejected intent IDs
    rejected = [p.agent_id for p in proposals if p.agent_id != winner.agent_id]
    
    # Generate resolution notes
    notes = []
    notes.append(f"Conflict detected: {len(conflict_report.details)} conflict(s)")
    notes.append(f"Resolution strategy: PRIORITY_SELECTION")
    notes.append(f"Winner: {winner.agent_id} (Priority: {winner.intent.priority})")
    notes.append(f"Rejected: {len(rejected)} intent(s) - {', '.join(rejected)}")
    
    # Get max conflict severity
    max_severity = "LOW"
    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    for detail in conflict_report.details:
        if severity_rank[detail.severity] > severity_rank[max_severity]:
            max_severity = detail.severity
    notes.append(f"Max conflict severity: {max_severity}")
    
    return ConflictResolutionOutput(
        conflict_detected=True,
        resolution_applied=True,
        winning_intent_id=winner.agent_id,
        winning_priority=winner.intent.priority,
        selected_plan=winner.plan,
        rejected_intents=rejected,
        resolution_notes=" | ".join(notes)
    ).model_dump_json(indent=2)

# -----------------------------
# 5) AGENT DEFINITION
# -----------------------------

RESOLUTION_INSTRUCTIONS = [
    "You are a Conflict Resolution Agent for a 6G Network Management System.",
    "Your role: Resolve conflicts by selecting the highest priority intent.",
    "",
    "INPUT: MetaArbitrationInput containing conflict report and agent proposals.",
    "",
    "RESOLUTION STRATEGY - SIMPLE PRIORITY SELECTION:",
    "1. When conflicts exist, select the intent with HIGHEST PRIORITY",
    "2. Priority order: CRITICAL > HIGH > MEDIUM > LOW",
    "3. Reject all other conflicting intents",
    "4. No merging, no weighted averaging - winner takes all",
    "",
    "EXAMPLE:",
    "   Intent A: Priority CRITICAL → WINNER",
    "   Intent B: Priority HIGH → REJECTED",
    "   Intent C: Priority MEDIUM → REJECTED",
    "   Result: Only Intent A's configuration is used",
    "",
    "OUTPUT: ConflictResolutionOutput with winning intent and rejected intents.",
]

conflict_resolution_agent = Agent(
    name="Conflict Resolution Agent",
    description="Resolves conflicts by selecting the highest priority intent (CRITICAL > HIGH > MEDIUM > LOW).",
    model=Groq(id=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")),
    tools=[resolve_conflicts],
    output_schema=ConflictResolutionOutput,
    instructions=RESOLUTION_INSTRUCTIONS,
)
