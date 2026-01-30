from __future__ import annotations

import json
import os
from typing import List, Optional, Dict, Any
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

class MergedConfiguration(BaseModel):
    """
    The final merged configuration from conflict resolution.
    Similar format to OptimizationPlan but with conflict resolution metadata.
    """
    merged_config_id: str  # Timestamp-based unique ID
    contributing_intents: List[str]  # List of intent agent_ids that contributed
    priority_weights: Dict[str, float]  # Weights assigned to each contributing intent
    changes: List[ParamChange]  # Final parameter changes
    expected_kpis: KpiSnapshot  # Expected KPIs after applying merged config
    resolution_strategy: str  # How conflicts were resolved
    conflict_summary: str  # Summary of resolved conflicts
    constraints_satisfied: bool

class ConflictResolutionOutput(BaseModel):
    """
    Complete output from the Conflict Resolution Agent/Team
    """
    conflict_detected: bool
    resolution_applied: bool
    merged_configuration: Optional[MergedConfiguration] = None
    original_proposals: List[AgentProposal]
    resolution_notes: str

# -----------------------------
# 2) WEIGHT CALCULATION
# -----------------------------

PRIORITY_WEIGHTS = {
    "CRITICAL": 4.0,
    "HIGH": 3.0,
    "MEDIUM": 2.0,
    "LOW": 1.0
}

def calculate_intent_weight(
    proposal: AgentProposal,
    conflict_severity: str
) -> float:
    """
    Calculate weight for an intent based on:
    1. Priority (primary factor)
    2. Confidence score
    3. KPI improvement magnitude
    4. Conflict severity (if intent is involved in high severity conflicts, reduce weight)
    """
    intent = proposal.intent
    plan = proposal.plan
    
    # Base weight from priority
    base_weight = PRIORITY_WEIGHTS.get(intent.priority, 2.0)
    
    # Confidence multiplier (0.5 to 1.0)
    confidence_mult = 0.5 + (intent.confidence * 0.5)
    
    # KPI improvement score
    kpi_improvement = 0.0
    if plan.expected_kpis and plan.current_kpis:
        # Calculate normalized improvement across target KPIs
        for kpi_name in intent.target_kpis:
            expected = getattr(plan.expected_kpis, kpi_name, None)
            current = getattr(plan.current_kpis, kpi_name, None)
            
            if expected is not None and current is not None:
                if kpi_name in ["RX_POWER", "SINR", "THROUGHPUT_5P"]:
                    # Higher is better
                    improvement = (expected - current) / (abs(current) + 1e-6)
                elif kpi_name == "LOAD_IMBALANCE":
                    # Lower is better
                    improvement = (current - expected) / (abs(current) + 1e-6)
                else:
                    improvement = 0.0
                
                kpi_improvement += max(0, improvement)  # Only count positive improvements
    
    # Normalize KPI improvement (cap at 1.0)
    kpi_mult = min(1.0, kpi_improvement / len(intent.target_kpis) if intent.target_kpis else 0.5)
    
    # Conflict severity penalty
    severity_penalty = {
        "LOW": 1.0,
        "MEDIUM": 0.9,
        "HIGH": 0.7,
        "CRITICAL": 0.5
    }.get(conflict_severity, 1.0)
    
    final_weight = base_weight * confidence_mult * (0.7 + 0.3 * kpi_mult) * severity_penalty
    
    return float(final_weight)

# -----------------------------
# 3) PARAMETER MERGING LOGIC
# -----------------------------

def merge_parameter_changes(
    proposals: List[AgentProposal],
    weights: Dict[str, float],
    conflict_report: ConflictReport
) -> List[ParamChange]:
    """
    Merge parameter changes from multiple proposals using weighted averaging.
    
    Strategy:
    1. For each parameter, collect all proposed changes
    2. Apply weighted average based on intent weights
    3. Handle boolean parameters with majority voting (weighted)
    4. For conflicting parameters (CRITICAL severity), use highest priority intent
    """
    
    # Collect all parameters and their proposed changes
    param_proposals: Dict[str, List[tuple[str, ParamChange, float]]] = {}
    
    for proposal in proposals:
        agent_id = proposal.agent_id
        weight = weights.get(agent_id, 1.0)
        
        for change in proposal.plan.changes:
            if change.param not in param_proposals:
                param_proposals[change.param] = []
            param_proposals[change.param].append((agent_id, change, weight))
    
    # Get critical conflicts to handle specially
    critical_params = set()
    for detail in conflict_report.details:
        if detail.severity == "CRITICAL" and detail.conflicting_param:
            critical_params.add(detail.conflicting_param)
    
    merged_changes = []
    
    for param, proposals_list in param_proposals.items():
        if len(proposals_list) == 1:
            # Only one proposal for this parameter, use it directly
            _, change, _ = proposals_list[0]
            merged_changes.append(change)
            continue
        
        # Multiple proposals for this parameter
        if param in critical_params:
            # CRITICAL conflict: Use highest weight proposal only
            proposals_list.sort(key=lambda x: x[2], reverse=True)
            _, change, _ = proposals_list[0]
            merged_changes.append(change)
        else:
            # Non-critical: Weighted merge
            _, first_change, _ = proposals_list[0]
            
            # Check if boolean parameter
            if isinstance(first_change.after, bool):
                # Boolean: Weighted majority voting
                true_weight = sum(w for _, c, w in proposals_list if c.after is True)
                false_weight = sum(w for _, c, w in proposals_list if c.after is False)
                
                merged_value = true_weight > false_weight
                merged_changes.append(ParamChange(
                    param=param,
                    before=first_change.before,
                    after=merged_value,
                    unit=first_change.unit
                ))
            else:
                # Numeric: Weighted average
                try:
                    total_weight = sum(w for _, _, w in proposals_list)
                    weighted_sum = sum(float(c.after) * w for _, c, w in proposals_list)
                    merged_value = weighted_sum / total_weight
                    
                    merged_changes.append(ParamChange(
                        param=param,
                        before=first_change.before,
                        after=merged_value,
                        unit=first_change.unit
                    ))
                except (ValueError, TypeError):
                    # If conversion fails, use highest weight proposal
                    proposals_list.sort(key=lambda x: x[2], reverse=True)
                    _, change, _ = proposals_list[0]
                    merged_changes.append(change)
    
    return merged_changes

def merge_kpi_expectations(
    proposals: List[AgentProposal],
    weights: Dict[str, float]
) -> KpiSnapshot:
    """
    Merge expected KPIs using weighted average
    """
    kpi_values = {
        "RX_POWER": [],
        "SINR": [],
        "THROUGHPUT_5P": [],
        "LOAD_IMBALANCE": [],
        "RX_COVERAGE_RATIO": []
    }
    
    for proposal in proposals:
        weight = weights.get(proposal.agent_id, 1.0)
        kpis = proposal.plan.expected_kpis
        
        if kpis.RX_POWER is not None:
            kpi_values["RX_POWER"].append((kpis.RX_POWER, weight))
        if kpis.SINR is not None:
            kpi_values["SINR"].append((kpis.SINR, weight))
        if kpis.THROUGHPUT_5P is not None:
            kpi_values["THROUGHPUT_5P"].append((kpis.THROUGHPUT_5P, weight))
        if kpis.LOAD_IMBALANCE is not None:
            kpi_values["LOAD_IMBALANCE"].append((kpis.LOAD_IMBALANCE, weight))
        if kpis.RX_COVERAGE_RATIO is not None:
            kpi_values["RX_COVERAGE_RATIO"].append((kpis.RX_COVERAGE_RATIO, weight))
    
    def weighted_avg(values_weights):
        if not values_weights:
            return None
        total_weight = sum(w for _, w in values_weights)
        return sum(v * w for v, w in values_weights) / total_weight
    
    return KpiSnapshot(
        RX_POWER=weighted_avg(kpi_values["RX_POWER"]),
        SINR=weighted_avg(kpi_values["SINR"]),
        THROUGHPUT_5P=weighted_avg(kpi_values["THROUGHPUT_5P"]),
        LOAD_IMBALANCE=weighted_avg(kpi_values["LOAD_IMBALANCE"]),
        RX_COVERAGE_RATIO=weighted_avg(kpi_values["RX_COVERAGE_RATIO"])
    )

# -----------------------------
# 4) MAIN RESOLUTION TOOL
# -----------------------------

def resolve_conflicts(meta_input_json: str) -> str:
    """
    Main conflict resolution function.
    Takes MetaArbitrationInput from Conflict Detector and produces merged configuration.
    
    Args:
        meta_input_json: JSON string of MetaArbitrationInput from Conflict Detector
        
    Returns:
        JSON string of ConflictResolutionOutput
    """
    
    # Parse input
    meta_input = MetaArbitrationInput.model_validate_json(meta_input_json)
    conflict_report = meta_input.conflict_report
    proposals = meta_input.proposals
    
    # If no conflict, return all proposals as-is (no merging needed)
    if not conflict_report.is_conflicted:
        return ConflictResolutionOutput(
            conflict_detected=False,
            resolution_applied=False,
            merged_configuration=None,
            original_proposals=proposals,
            resolution_notes="No conflicts detected. All intents can proceed independently."
        ).model_dump_json(indent=2)
    
    # Calculate weights for each proposal
    weights = {}
    max_severity = "LOW"
    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    
    for detail in conflict_report.details:
        if severity_rank[detail.severity] > severity_rank[max_severity]:
            max_severity = detail.severity
    
    for proposal in proposals:
        weight = calculate_intent_weight(proposal, max_severity)
        weights[proposal.agent_id] = weight
    
    # Normalize weights (sum to number of proposals)
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: (v / total_weight) * len(proposals) for k, v in weights.items()}
    
    # Merge parameter changes
    merged_changes = merge_parameter_changes(proposals, weights, conflict_report)
    
    # Merge KPI expectations
    merged_kpis = merge_kpi_expectations(proposals, weights)
    
    # Check if constraints are satisfied (use weighted voting)
    constraints_votes = sum(
        weights.get(p.agent_id, 1.0) 
        for p in proposals 
        if p.plan.constraints_satisfied
    )
    total_votes = sum(weights.values())
    constraints_satisfied = constraints_votes > (total_votes / 2)
    
    # Determine resolution strategy
    strategy = "WEIGHTED_MERGE"
    if max_severity == "CRITICAL":
        strategy = "PRIORITY_OVERRIDE"  # Critical conflicts use highest priority only
    elif max_severity == "HIGH":
        strategy = "WEIGHTED_MERGE_CONSERVATIVE"  # More conservative averaging
    
    # Create merged configuration
    merged_config = MergedConfiguration(
        merged_config_id=f"merged_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        contributing_intents=[p.agent_id for p in proposals],
        priority_weights=weights,
        changes=merged_changes,
        expected_kpis=merged_kpis,
        resolution_strategy=strategy,
        conflict_summary=conflict_report.conflict_summary,
        constraints_satisfied=constraints_satisfied
    )
    
    # Generate resolution notes
    notes = []
    notes.append(f"Resolved {len(conflict_report.details)} conflicts using {strategy} strategy.")
    notes.append(f"Combined {len(proposals)} proposals with weights: {weights}")
    notes.append(f"Max conflict severity: {max_severity}")
    
    return ConflictResolutionOutput(
        conflict_detected=True,
        resolution_applied=True,
        merged_configuration=merged_config,
        original_proposals=proposals,
        resolution_notes=" | ".join(notes)
    ).model_dump_json(indent=2)

# -----------------------------
# 5) AGENT DEFINITION
# -----------------------------

RESOLUTION_INSTRUCTIONS = [
    "You are a Conflict Resolution Agent (Meta-Agent) for a 6G Network Management System.",
    "Your role: Resolve conflicts between multiple optimization intents and create a unified configuration.",
    "",
    "INPUT: MetaArbitrationInput containing conflict report and agent proposals.",
    "",
    "RESOLUTION STRATEGY:",
    "1. PRIORITY-BASED WEIGHTING:",
    "   - CRITICAL priority: 4.0x weight",
    "   - HIGH priority: 3.0x weight",
    "   - MEDIUM priority: 2.0x weight",
    "   - LOW priority: 1.0x weight",
    "",
    "2. TIE-BREAKING (Same Priority):",
    "   - Confidence score (higher confidence gets more weight)",
    "   - KPI improvement magnitude (bigger improvements get more weight)",
    "   - Conflict involvement (intents in CRITICAL conflicts get penalty)",
    "",
    "3. MERGE STRATEGIES:",
    "   - CRITICAL conflicts: Use highest priority intent only",
    "   - HIGH conflicts: Weighted merge with conservative averaging",
    "   - MEDIUM/LOW conflicts: Full weighted merge",
    "   - Boolean params: Weighted majority voting",
    "   - Numeric params: Weighted averaging",
    "",
    "OUTPUT: ConflictResolutionOutput with merged configuration and resolution metadata.",
]

conflict_resolution_agent = Agent(
    name="Conflict Resolution Meta-Agent",
    description="Resolves conflicts between network optimization intents using priority-based weighted merging.",
    model=Groq(id=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")),
    tools=[resolve_conflicts],
    output_schema=ConflictResolutionOutput,
    instructions=RESOLUTION_INSTRUCTIONS,
)
