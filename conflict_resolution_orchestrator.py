from __future__ import annotations

import json
import os
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import all necessary modules
from optimization_agent import optimize_from_intent, IntentParse, OptimizationPlan
from conflict_detector_agent import detect_conflicts, MetaArbitrationInput, AgentProposal
from conflict_resolution_agent import resolve_conflicts, ConflictResolutionOutput

load_dotenv()

# -----------------------------
# ORCHESTRATOR INPUT/OUTPUT
# -----------------------------

class NewIntentInput(BaseModel):
    """
    Input when a new intent arrives and needs to be checked against active intents
    """
    new_intent: IntentParse  # Yeni gelen intent
    active_intents: List[Dict[str, Any]] = Field(default_factory=list)  # Sistemde çalışan intentler (zaten optimize edilmiş)
    orchestration_id: str = Field(default_factory=lambda: f"orch_{datetime.now().strftime('%Y%m%d%H%M%S')}")

class MultiIntentInput(BaseModel):
    """
    DEPRECATED: Use NewIntentInput instead
    Input for the orchestrator: multiple intents to be processed
    """
    intents: List[IntentParse]
    orchestration_id: str = Field(default_factory=lambda: f"orch_{datetime.now().strftime('%Y%m%d%H%M%S')}")

class IntentOptimizationResult(BaseModel):
    """
    Result from a single optimization agent
    """
    intent_id: str
    intent: IntentParse
    optimization_plan: OptimizationPlan
    success: bool
    error_message: str = ""

class OrchestrationOutput(BaseModel):
    """
    Complete output from the orchestrator
    """
    orchestration_id: str
    total_intents: int
    successful_optimizations: int
    failed_optimizations: int
    
    # Individual optimization results
    optimization_results: List[IntentOptimizationResult]
    
    # Conflict analysis
    conflict_detected: bool
    conflict_report: Dict[str, Any] = Field(default_factory=dict)
    
    # Final resolution (if conflicts exist)
    resolution_applied: bool
    final_configuration: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution strategy
    execution_strategy: str = "UNKNOWN"  # SINGLE, PARALLEL, MERGED
    
    # Notes and metadata
    notes: List[str] = Field(default_factory=list)


class ConflictResolutionOrchestrator:
    """
    Team leader that coordinates optimization agents and performs conflict resolution when a new intent arrives.
    
    CORRECTED Workflow:
    1. System has active intents running (already optimized)
    2. NEW intent arrives
    3. Optimize ONLY the new intent
    4. Run conflict detection (new vs active)
    5. If NO conflict: Add new intent to system (parallel execution)
    6. If conflict EXISTS: Team/Meta-agent resolves using priority-based merging
    7. Return final configuration
    
    Orchestrates multiple optimization agents and resolves conflicts using a team approach.
    
    Workflow:
    1. Receive multiple intents
    2. Run optimization agent for each intent (simulates team members)
    3. Collect all optimization results
    4. Run conflict detection
    5. If conflicts: Run conflict resolution (meta-agent)
    6. Return final merged configuration or parallel execution plan
    """
    
    def __init__(self):
        self.orchestration_id = None
        self.results = []
        
    def optimize_single_intent(self, intent: IntentParse, intent_id: str) -> IntentOptimizationResult:
        """
        Call optimization agent for a single intent (simulates a team member)
        """
        try:
            # Call optimization agent tool directly
            result_json = optimize_from_intent(intent.model_dump_json())
            result = json.loads(result_json)
            
            # Parse result to OptimizationPlan
            plan = OptimizationPlan(**result)
            
            return IntentOptimizationResult(
                intent_id=intent_id,
                intent=intent,
                optimization_plan=plan,
                success=True
            )
            
        except Exception as e:
            return IntentOptimizationResult(
                intent_id=intent_id,
                intent=intent,
                optimization_plan=OptimizationPlan(
                    selected_config_id=0,
                    changes=[],
                    expected_kpis={},
                    constraints_satisfied=False
                ),
                success=False,
                error_message=str(e)
            )
    
    def process_new_intent(self, new_intent_input: NewIntentInput) -> OrchestrationOutput:
        """
        Process a NEW intent against ACTIVE intents (correct workflow)
        """
        self.orchestration_id = new_intent_input.orchestration_id
        new_intent = new_intent_input.new_intent
        active_intents_data = new_intent_input.active_intents
        
        output = OrchestrationOutput(
            orchestration_id=self.orchestration_id,
            total_intents=1 + len(active_intents_data),
            successful_optimizations=0,
            failed_optimizations=0,
            optimization_results=[],
            conflict_detected=False,
            resolution_applied=False,
            execution_strategy="PARALLEL"
        )
        
        print(f"\n{'='*80}")
        print(f"ORCHESTRATION {self.orchestration_id}: New Intent Arrival")
        print(f"Active Intents: {len(active_intents_data)}")
        print(f"{'='*80}\n")
        
        # STEP 1: Optimize ONLY the new intent
        print(f"[NEW] Optimizing New Intent: {new_intent.target_area} (Priority: {new_intent.priority})")
        
        new_result = self.optimize_single_intent(new_intent, f"new_intent_{new_intent.priority}")
        
        if new_result.success:
            output.successful_optimizations += 1
            print(f"  ✅ Success - Config ID: {new_result.optimization_plan.selected_config_id}")
            print(f"     Changes: {len(new_result.optimization_plan.changes)} parameters")
        else:
            output.failed_optimizations += 1
            print(f"  ❌ Failed - {new_result.error_message}")
            output.notes.append("New intent optimization failed. Cannot proceed.")
            return output
        
        output.optimization_results.append(new_result)
        
        # If no active intents, just return the new intent
        if len(active_intents_data) == 0:
            output.execution_strategy = "SINGLE"
            output.notes.append("No active intents. New intent will be executed.")
            output.final_configuration = new_result.optimization_plan.model_dump()
            return output
        
        # STEP 2: Prepare active intents (already optimized)
        print(f"\n[ACTIVE] {len(active_intents_data)} active intent(s) in system:")
        for i, active_data in enumerate(active_intents_data):
            intent_info = active_data.get('intent', {})
            plan_info = active_data.get('plan', {})
            print(f"  [{i+1}] {intent_info.get('target_area', 'Unknown')} - Priority: {intent_info.get('priority', 'Unknown')}")
            print(f"      Config ID: {plan_info.get('selected_config_id', 'Unknown')}")
            print(f"      Changes: {len(plan_info.get('changes', []))} parameters")
        
        # STEP 3: Run conflict detection (new vs active)
        print(f"\n{'='*80}")
        print("STEP 2: Conflict Detection (New vs Active)")
        print(f"{'='*80}\n")
        
        conflict_result_json = detect_conflicts(
            new_intent_json=new_result.intent.model_dump_json(),
            new_plan_json=new_result.optimization_plan.model_dump_json(),
            active_intents_data=json.dumps(active_intents_data)
        )
        
        conflict_result = json.loads(conflict_result_json)
        output.conflict_report = conflict_result
        output.conflict_detected = conflict_result['conflict_report']['is_conflicted']
        
        print(f"Conflicts Detected: {output.conflict_detected}")
        if output.conflict_detected:
            print(f"Conflict Summary: {conflict_result['conflict_report']['conflict_summary']}")
        
        # STEP 4: Decision - Conflict Resolution or Parallel Execution
        if output.conflict_detected:
            print(f"\n{'='*80}")
            print("STEP 3: Conflict Resolution (Team/Meta-Agent)")
            print(f"{'='*80}\n")
            
            resolution_result_json = resolve_conflicts(conflict_result_json)
            resolution_result = json.loads(resolution_result_json)
            
            output.resolution_applied = resolution_result['resolution_applied']
            output.final_configuration = resolution_result.get('merged_configuration', {})
            output.execution_strategy = "MERGED"
            
            if resolution_result['merged_configuration']:
                merged = resolution_result['merged_configuration']
                print(f"✅ Resolution Applied: {merged['resolution_strategy']}")
                print(f"   Contributing Intents: {len(merged['contributing_intents'])}")
                print(f"   Parameter Changes: {len(merged['changes'])}")
                print(f"   Constraints Satisfied: {merged['constraints_satisfied']}")
                
                output.notes.append(f"Conflicts resolved using {merged['resolution_strategy']} strategy")
                output.notes.append(f"Merged {len(merged['contributing_intents'])} intents into single configuration")
            else:
                output.notes.append("Resolution attempted but no merged configuration produced")
        else:
            # No conflicts - add new intent to parallel execution
            output.execution_strategy = "PARALLEL"
            output.notes.append("No conflicts detected. New intent added to parallel execution.")
            
            # Store all configurations (active + new)
            all_configs = []
            for active_data in active_intents_data:
                all_configs.append({
                    "intent_id": active_data.get('intent', {}).get('target_area', 'Unknown'),
                    "plan": active_data.get('plan', {})
                })
            
            all_configs.append({
                "intent_id": new_result.intent_id,
                "plan": new_result.optimization_plan.model_dump()
            })
            
            output.final_configuration = {
                "execution_mode": "parallel",
                "configurations": all_configs
            }
        
        return output
    
    def run_orchestration(self, multi_intent_input: MultiIntentInput) -> OrchestrationOutput:
        """
        Main orchestration logic
        """
        self.orchestration_id = multi_intent_input.orchestration_id
        intents = multi_intent_input.intents
        
        output = OrchestrationOutput(
            orchestration_id=self.orchestration_id,
            total_intents=len(intents),
            successful_optimizations=0,
            failed_optimizations=0,
            optimization_results=[],
            conflict_detected=False,
            resolution_applied=False,
            execution_strategy="PARALLEL"
        )
        
        # STEP 1: Run optimization for each intent (Team Members)
        print(f"\n{'='*80}")
        print(f"ORCHESTRATION {self.orchestration_id}: Processing {len(intents)} intents")
        print(f"{'='*80}\n")
        
        optimization_results = []
        for i, intent in enumerate(intents):
            intent_id = f"intent_{i}_{intent.priority}"
            print(f"[{i+1}/{len(intents)}] Optimizing Intent: {intent.target_area} (Priority: {intent.priority})")
            
            result = self.optimize_single_intent(intent, intent_id)
            optimization_results.append(result)
            
            if result.success:
                output.successful_optimizations += 1
                print(f"  ✅ Success - Config ID: {result.optimization_plan.selected_config_id}")
                print(f"     Changes: {len(result.optimization_plan.changes)} parameters")
            else:
                output.failed_optimizations += 1
                print(f"  ❌ Failed - {result.error_message}")
        
        output.optimization_results = optimization_results
        
        # If all failed, return early
        if output.successful_optimizations == 0:
            output.notes.append("All optimizations failed. No conflict resolution needed.")
            return output
        
        # If only one succeeded, no conflict possible
        if output.successful_optimizations == 1:
            output.execution_strategy = "SINGLE"
            output.notes.append("Only one intent succeeded. No conflict resolution needed.")
            successful_result = [r for r in optimization_results if r.success][0]
            output.final_configuration = successful_result.optimization_plan.model_dump()
            return output
        
        # STEP 2: Prepare proposals for conflict detection
        print(f"\n{'='*80}")
        print("STEP 2: Conflict Detection")
        print(f"{'='*80}\n")
        
        successful_results = [r for r in optimization_results if r.success]
        
        # Use first intent as "new" and rest as "active"
        new_result = successful_results[0]
        active_results = successful_results[1:]
        
        active_proposals = [
            {
                "intent": r.intent.model_dump(),
                "plan": r.optimization_plan.model_dump()
            }
            for r in active_results
        ]
        
        # Run conflict detection
        conflict_result_json = detect_conflicts(
            new_intent_json=new_result.intent.model_dump_json(),
            new_plan_json=new_result.optimization_plan.model_dump_json(),
            active_intents_data=json.dumps(active_proposals)
        )
        
        conflict_result = json.loads(conflict_result_json)
        output.conflict_report = conflict_result
        output.conflict_detected = conflict_result['conflict_report']['is_conflicted']
        
        print(f"Conflicts Detected: {output.conflict_detected}")
        if output.conflict_detected:
            print(f"Conflict Summary: {conflict_result['conflict_report']['conflict_summary']}")
        
        # STEP 3: Conflict Resolution (if needed)
        if output.conflict_detected:
            print(f"\n{'='*80}")
            print("STEP 3: Conflict Resolution (Meta-Agent)")
            print(f"{'='*80}\n")
            
            resolution_result_json = resolve_conflicts(conflict_result_json)
            resolution_result = json.loads(resolution_result_json)
            
            output.resolution_applied = resolution_result['resolution_applied']
            output.final_configuration = resolution_result.get('merged_configuration', {})
            output.execution_strategy = "MERGED"
            
            if resolution_result['merged_configuration']:
                merged = resolution_result['merged_configuration']
                print(f"✅ Resolution Applied: {merged['resolution_strategy']}")
                print(f"   Contributing Intents: {len(merged['contributing_intents'])}")
                print(f"   Parameter Changes: {len(merged['changes'])}")
                print(f"   Constraints Satisfied: {merged['constraints_satisfied']}")
                
                output.notes.append(f"Conflicts resolved using {merged['resolution_strategy']} strategy")
                output.notes.append(f"Merged {len(merged['contributing_intents'])} intents into single configuration")
            else:
                output.notes.append("Resolution attempted but no merged configuration produced")
        else:
            # No conflicts - all intents can run in parallel
            output.execution_strategy = "PARALLEL"
            output.notes.append("No conflicts detected. All intents can execute in parallel.")
            
            # Store all successful plans
            output.final_configuration = {
                "execution_mode": "parallel",
                "configurations": [
                    {
                        "intent_id": r.intent_id,
                        "plan": r.optimization_plan.model_dump()
                    }
                    for r in successful_results
                ]
            }
        
        return output

# -----------------------------
# CONVENIENCE FUNCTIONS
# -----------------------------

def run_multi_intent_optimization(intents_json: str) -> str:
    """
    Main entry point for multi-intent optimization with conflict resolution.
    
    Args:
        intents_json: JSON string of MultiIntentInput or list of IntentParse
        
    Returns:
        JSON string of OrchestrationOutput
    """
    
    try:
        # Try to parse as MultiIntentInput
        multi_input = MultiIntentInput.model_validate_json(intents_json)
    except:
        # Try to parse as list of intents
        intents_list = json.loads(intents_json)
        intents = [IntentParse(**i) for i in intents_list]
        multi_input = MultiIntentInput(intents=intents)
    
    orchestrator = ConflictResolutionOrchestrator()
    result = orchestrator.run_orchestration(multi_input)
    
    return result.model_dump_json(indent=2)

def save_orchestration_result(output: OrchestrationOutput, filename: str = None):
    """
    Save orchestration result to file
    """
    if filename is None:
        filename = f"results/orchestration_{output.orchestration_id}.json"
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(output.model_dump(), f, indent=2)
    
    return filename

def process_new_intent(new_intent_json: str, active_intents_json: str = "[]") -> str:
    """
    MAIN ENTRY POINT: Process a new intent against active intents (CORRECT workflow)
    
    Args:
        new_intent_json: JSON string of the new IntentParse
        active_intents_json: JSON string of active intents (already optimized)
                            Format: [{"intent": {...}, "plan": {...}}, ...]
        
    Returns:
        JSON string of OrchestrationOutput
    """
    
    new_intent = IntentParse.model_validate_json(new_intent_json)
    active_intents = json.loads(active_intents_json)
    
    new_intent_input = NewIntentInput(
        new_intent=new_intent,
        active_intents=active_intents
    )
    
    orchestrator = ConflictResolutionOrchestrator()
    result = orchestrator.process_new_intent(new_intent_input)
    
    return result.model_dump_json(indent=2)
