#!/usr/bin/env python3
"""
Agno Workflow: 6G Network Optimization Pipeline
Sequential workflow with individual agent steps

Run: python workflow_steps.py --playground
Open: http://localhost:7777 → Workflows tab
"""

from agno.os import AgentOS
from agno.workflow import Workflow, Step
from agno.agent import Agent
from agno.models.groq import Groq
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json

load_dotenv()

# Import existing agents and tools
from intent_parser.intent_parser_agent import intent_parser_agent, IntentParse
from optimization_agent import optimize_from_intent, OptimizationPlan
from conflict_detector_agent import detect_conflicts, AgentProposal
from conflict_resolution_agent import resolve_conflicts, ConflictResolutionOutput

# =====================================================
# WORKFLOW STATE MODEL
# =====================================================

class OptimizationWorkflowState(BaseModel):
    """State that flows through the workflow steps"""
    natural_language_intent: str
    active_intents: List[Dict[str, Any]] = []
    
    # Step outputs
    parsed_intent: Optional[IntentParse] = None
    optimization_plan: Optional[OptimizationPlan] = None
    conflict_report: Optional[Dict[str, Any]] = None
    resolution_result: Optional[ConflictResolutionOutput] = None
    execution_strategy: Optional[str] = None
    final_configuration: Optional[Dict[str, Any]] = None
    
    # Tracking
    current_step: str = "INITIAL"
    conflict_detected: bool = False
    errors: List[str] = []

# =====================================================
# STORAGE
# =====================================================

ACTIVE_INTENTS_FILE = "active_intents_workflow.json"

def load_active_intents():
    import os
    if os.path.exists(ACTIVE_INTENTS_FILE):
        with open(ACTIVE_INTENTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_active_intents(intents):
    with open(ACTIVE_INTENTS_FILE, 'w') as f:
        json.dump(intents, f, indent=2)

def clear_active_intents():
    """Clear all active intents"""
    import os
    if os.path.exists(ACTIVE_INTENTS_FILE):
        os.remove(ACTIVE_INTENTS_FILE)
    return {"status": "cleared"}

# =====================================================
# WORKFLOW STEP FUNCTIONS
# =====================================================

def step_1_parse_intent(step_input):
    """Step 1: Parse natural language intent into structured JSON"""
    print(f"\n{'='*60}")
    print("📋 STEP 1: Intent Parsing")
    print(f"{'='*60}")
    
    state = None
    try:
        # Get the state from step input
        if isinstance(step_input.input, dict):
            state = OptimizationWorkflowState(**step_input.input)
        else:
            state = step_input.input
        
        # Call intent parser agent
        parse_response = intent_parser_agent.run(state.natural_language_intent)
        state.parsed_intent = parse_response.content
        state.current_step = "PARSED"
        
        print(f"✅ Intent Parsed:")
        print(f"   Target Area: {state.parsed_intent.target_area}")
        print(f"   Target KPIs: {state.parsed_intent.target_kpis}")
        print(f"   Priority: {state.parsed_intent.priority}")
        print(f"   Confidence: {state.parsed_intent.confidence:.2f}")
        
        return state.model_dump()
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        if state:
            state.errors.append(f"Parse error: {str(e)}")
        raise

def step_2_optimize(step_input):
    """Step 2: Find optimal base station configuration"""
    print(f"\n{'='*60}")
    print("🔧 STEP 2: Optimization")
    print(f"{'='*60}")
    
    state = None
    try:
        # Get state from previous step (parse JSON if string)
        prev_content = step_input.previous_step_content
        print(f"DEBUG: prev_content type = {type(prev_content)}")
        print(f"DEBUG: prev_content = {prev_content[:200] if isinstance(prev_content, str) else prev_content}")
        
        if isinstance(prev_content, str):
            prev_content = json.loads(prev_content)
        state = OptimizationWorkflowState(**prev_content)
        state.current_step = "OPTIMIZED"
        
        if not state.parsed_intent:
            raise ValueError("No parsed intent available")
        
        # Call optimization tool
        opt_result_json = optimize_from_intent(state.parsed_intent.model_dump_json())
        opt_result_dict = json.loads(opt_result_json)
        
        # Convert to OptimizationPlan
        state.optimization_plan = OptimizationPlan(**opt_result_dict)
        
        print(f"✅ Optimization Complete:")
        print(f"   Config ID: {state.optimization_plan.selected_config_id}")
        print(f"   Changes: {len(state.optimization_plan.changes)} parameters")
        print(f"   Constraints Satisfied: {state.optimization_plan.constraints_satisfied}")
        
        return state.model_dump()
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        if state:
            state.errors.append(f"Optimization error: {str(e)}")
        raise

def step_3_detect_conflicts(step_input):
    """Step 3: Check for conflicts with active intents"""
    print(f"\n{'='*60}")
    print("🔍 STEP 3: Conflict Detection")
    print(f"{'='*60}")
    
    state = None
    try:
        # Get state from previous step (parse JSON if string)
        prev_content = step_input.previous_step_content
        if isinstance(prev_content, str):
            prev_content = json.loads(prev_content)
        state = OptimizationWorkflowState(**prev_content)
        state.current_step = "CONFLICT_CHECKED"
        
        # If no active intents, skip conflict detection
        if len(state.active_intents) == 0:
            print("ℹ️  No active intents - skipping conflict detection")
            state.conflict_detected = False
            state.conflict_report = {
                "is_conflicted": False,
                "conflict_summary": "No active intents in system"
            }
            return state.model_dump()
        
        if not state.parsed_intent or not state.optimization_plan:
            raise ValueError("Missing intent or plan for conflict detection")
        
        # Call conflict detector
        conflict_result_json = detect_conflicts(
            state.parsed_intent.model_dump_json(),
            state.optimization_plan.model_dump_json(),
            json.dumps(state.active_intents)
        )
        conflict_result = json.loads(conflict_result_json)
        
        state.conflict_report = conflict_result
        state.conflict_detected = conflict_result['conflict_report']['is_conflicted']
        
        status = "🔴 CONFLICT DETECTED" if state.conflict_detected else "🟢 NO CONFLICT"
        print(f"{status}")
        print(f"   Summary: {conflict_result['conflict_report']['conflict_summary']}")
        
        if state.conflict_detected:
            print(f"   Conflicts: {len(conflict_result['conflict_report']['details'])}")
            for detail in conflict_result['conflict_report']['details']:
                print(f"      - {detail['conflict_type']} (Severity: {detail['severity']})")
        
        return state.model_dump()
    except Exception as e:
        if state:
            state.errors.append(f"Conflict detection error: {str(e)}")
        raise

def step_4_resolve_conflicts(step_input):
    """Step 4: Resolve conflicts using priority-based selection"""
    print(f"\n{'='*60}")
    print("🎯 STEP 4: Conflict Resolution")
    print(f"{'='*60}")
    
    state = None
    try:
        # Get state from previous step (parse JSON if string)
        prev_content = step_input.previous_step_content
        if isinstance(prev_content, str):
            prev_content = json.loads(prev_content)
        state = OptimizationWorkflowState(**prev_content)
        state.current_step = "RESOLVED"
        
        if not state.conflict_detected:
            # No conflict - add to parallel execution
            state.execution_strategy = "PARALLEL"
            state.final_configuration = {
                "execution_mode": "parallel",
                "configurations": state.active_intents + [{
                    "intent": state.parsed_intent.model_dump() if state.parsed_intent else None,
                    "plan": state.optimization_plan.model_dump() if state.optimization_plan else None
                }]
            }
            print(f"✅ No Conflict - Parallel Execution")
            print(f"   Total Running Intents: {len(state.final_configuration['configurations'])}")
            
            # Save to storage
            if state.parsed_intent and state.optimization_plan:
                new_entry = {
                    "intent": state.parsed_intent.model_dump(),
                    "plan": state.optimization_plan.model_dump()
                }
                active = load_active_intents()
                active.append(new_entry)
                save_active_intents(active)
                print(f"💾 Saved. Total active intents: {len(active)}")
            
            return state.model_dump()
        
        # Conflict detected - use resolution agent
        print(f"🔴 Conflict Detected - Running Resolution Agent")
        
        # Prepare input for resolution agent
        meta_input_json = json.dumps(state.conflict_report)
        
        # Call resolution agent tool
        resolution_json = resolve_conflicts(meta_input_json)
        resolution_dict = json.loads(resolution_json)
        
        state.resolution_result = ConflictResolutionOutput(**resolution_dict)
        
        # Determine execution strategy
        if state.resolution_result.resolution_applied:
            state.execution_strategy = "PRIORITY_SELECTION"
            
            print(f"✅ Resolution Applied:")
            print(f"   Strategy: {state.execution_strategy}")
            print(f"   Winner: {state.resolution_result.winning_intent_id}")
            print(f"   Priority: {state.resolution_result.winning_priority}")
            print(f"   Rejected: {len(state.resolution_result.rejected_intents)} intent(s)")
            
            # Final configuration is the winning plan
            if state.resolution_result.selected_plan:
                state.final_configuration = {
                    "execution_mode": "priority_selection",
                    "winning_intent": state.resolution_result.winning_intent_id,
                    "plan": state.resolution_result.selected_plan.model_dump()
                }
            
            # Update active intents - keep only winner
            # Find which intent is the winner
            winner_id = state.resolution_result.winning_intent_id
            
            if winner_id and winner_id.startswith("new_intent"):
                # New intent won - replace all active intents
                if state.parsed_intent and state.optimization_plan:
                    new_active = [{
                        "intent": state.parsed_intent.model_dump(),
                        "plan": state.optimization_plan.model_dump()
                    }]
                    save_active_intents(new_active)
                    print(f"💾 New intent won - replaced all active intents")
            else:
                # Existing intent won - keep only that one
                active = load_active_intents()
                # Keep existing intents (they already won)
                print(f"💾 Existing intent won - keeping active intents")
        else:
            state.execution_strategy = "UNKNOWN"
            print(f"⚠️  Resolution not applied")
        
        return state.model_dump()
        
    except Exception as e:
        print(f"❌ Resolution failed: {e}")
        if state:
            state.errors.append(f"Resolution error: {str(e)}")
        raise

# =====================================================
# WORKFLOW DEFINITION
# =====================================================

optimization_workflow = Workflow(
    name="6G Network Optimization",
    description="End-to-end intent-based optimization with conflict resolution",
    steps=[
        Step(
            name="Parse Intent",
            executor=step_1_parse_intent,
            description="Parse natural language into structured IntentParse JSON"
        ),
        Step(
            name="Optimize Configuration",
            executor=step_2_optimize,
            description="Find optimal base station configuration using Optimization Agent"
        ),
        Step(
            name="Detect Conflicts",
            executor=step_3_detect_conflicts,
            description="Check for conflicts with active intents using Conflict Detector"
        ),
        Step(
            name="Resolve Conflicts",
            executor=step_4_resolve_conflicts,
            description="Resolve conflicts using priority-based selection (CRITICAL > HIGH > MEDIUM > LOW)"
        )
    ]
)

# =====================================================
# AGNO OS SETUP
# =====================================================

def setup_workflow_system():
    """Setup AgentOS with the workflow"""
    
    agent_os = AgentOS(
        name="6G Network Optimization System",
        workflows=[optimization_workflow]
    )
    
    print("✅ Workflow System Initialized")
    print("   Workflow: 6G Network Optimization")
    print("   Steps: 4 (Parse → Optimize → Detect → Resolve)")
    print("   Resolution Strategy: Priority Selection (CRITICAL > HIGH > MEDIUM > LOW)")
    
    return agent_os

# Create global app
agent_os = setup_workflow_system()
app = agent_os.get_app()

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    import sys
    
    if "--playground" in sys.argv:
        print("\n🎮 Starting Workflow Playground...")
        print("🌐 Open: http://localhost:7777")
        print("\n📊 Go to 'Workflows' tab to see the workflow!")
        print("   Steps:")
        print("   1. Parse Intent → Intent Parser Agent")
        print("   2. Optimize → Optimization Agent")
        print("   3. Detect Conflicts → Conflict Detector Agent")
        print("   4. Resolve Conflicts → Priority Selection Agent")
        print("\n   Resolution: CRITICAL > HIGH > MEDIUM > LOW (Winner takes all)")
        print("\nPress Ctrl+C to stop\n")
        agent_os.serve(app="workflow_steps:app", reload=True, port=7777)
    else:
        # Demo run
        print("💡 Run with --playground:")
        print("   python workflow_steps.py --playground")
        print("\nOr test workflow directly:")
        
        # Load active intents
        active = load_active_intents()
        print(f"\n📊 Current active intents: {len(active)}")
        
        # Create initial state
        initial_state = OptimizationWorkflowState(
            natural_language_intent="Increase coverage in Kadikoy region. RX power must be at least -95 dBm. Priority is HIGH.",
            active_intents=active
        )
        
        # Run workflow
        print("\n🚀 Running workflow...")
        result = optimization_workflow.run(initial_state)
        
        print(f"\n{'='*60}")
        print("✅ WORKFLOW COMPLETED")
        print(f"{'='*60}")
        print(f"Execution Strategy: {result.content.execution_strategy}")
        print(f"Conflict Detected: {result.content.conflict_detected}")
        if result.content.resolution_result:
            print(f"Winner: {result.content.resolution_result.winning_intent_id}")
            print(f"Priority: {result.content.resolution_result.winning_priority}")
        print(f"Errors: {result.content.errors if result.content.errors else 'None'}")

