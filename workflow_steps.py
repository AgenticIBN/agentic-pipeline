#!/usr/bin/env python3
"""
Agno Workflow: 6G Network Optimization Pipeline
True workflow structure with 4 sequential steps

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
from optimization_agent import optimize_from_intent
from conflict_detector_agent import detect_conflicts
from conflict_resolution_orchestrator import process_new_intent

# =====================================================
# WORKFLOW STATE MODEL
# =====================================================

class OptimizationWorkflowState(BaseModel):
    """State that flows through the workflow steps"""
    natural_language_intent: str
    active_intents: List[Dict[str, Any]] = []
    
    # Step outputs
    parsed_intent: Optional[IntentParse] = None
    optimization_plan: Optional[Dict[str, Any]] = None
    conflict_report: Optional[Dict[str, Any]] = None
    execution_strategy: Optional[str] = None
    final_configuration: Optional[Dict[str, Any]] = None
    
    # Tracking
    current_step: str = "INITIAL"
    conflict_detected: bool = False
    errors: List[str] = []

# =====================================================
# STORAGE
# =====================================================

ACTIVE_INTENTS_FILE = "active_intents_workflow_steps.json"

def load_active_intents():
    import os
    if os.path.exists(ACTIVE_INTENTS_FILE):
        with open(ACTIVE_INTENTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_active_intents(intents):
    with open(ACTIVE_INTENTS_FILE, 'w') as f:
        json.dump(intents, f, indent=2)

# =====================================================
# WORKFLOW STEP FUNCTIONS
# =====================================================

# =====================================================
# WORKFLOW STEP FUNCTIONS
# =====================================================

def step_1_parse_intent(step_input):
    """Step 1: Parse natural language intent into structured JSON"""
    print(f"\n{'='*60}")
    print("📋 STEP 1: Intent Parsing")
    print(f"{'='*60}")
    
    try:
        # Get the state from step input
        if isinstance(step_input.input, dict):
            state = OptimizationWorkflowState(**step_input.input)
        else:
            state = step_input.input
        
        # Call intent parser agent
        parse_response = intent_parser_agent.run(state.natural_language_intent)
        state.parsed_intent = parse_response.content
        state.current_step = "PARSING"
        
        print(f"✅ Intent Parsed:")
        print(f"   Target Area: {state.parsed_intent.target_area}")
        print(f"   Target KPIs: {state.parsed_intent.target_kpis}")
        print(f"   Priority: {state.parsed_intent.priority}")
        
        return state.model_dump()
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        raise

def step_2_optimize(step_input):
    """Step 2: Find optimal base station configuration"""
    print(f"\n{'='*60}")
    print("🔧 STEP 2: Optimization")
    print(f"{'='*60}")
    
    try:
        # Get state from previous step
        state = OptimizationWorkflowState(**step_input.previous_step_content)
        state.current_step = "OPTIMIZING"
        
        if not state.parsed_intent:
            raise ValueError("No parsed intent available")
        
        # Call optimization tool
        opt_result_json = optimize_from_intent(state.parsed_intent.model_dump_json())
        opt_result = json.loads(opt_result_json)
        state.optimization_plan = opt_result
        
        print(f"✅ Optimization Complete:")
        print(f"   Config ID: {opt_result['selected_config_id']}")
        print(f"   Changes: {len(opt_result['changes'])} parameters")
        
        return state.model_dump()
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        raise

def step_3_detect_conflicts(step_input):
    """Step 3: Check for conflicts with active intents"""
    print(f"\n{'='*60}")
    print("🔍 STEP 3: Conflict Detection")
    print(f"{'='*60}")
    
    try:
        # Get state from previous step
        state = OptimizationWorkflowState(**step_input.previous_step_content)
        state.current_step = "CONFLICT_DETECTION"
        
        if len(state.active_intents) == 0:
            print("ℹ️ No active intents - skipping conflict detection")
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
            json.dumps(state.optimization_plan),
            json.dumps(state.active_intents)
        )
        conflict_result = json.loads(conflict_result_json)
        state.conflict_report = conflict_result['conflict_report']
        state.conflict_detected = state.conflict_report['is_conflicted']
        
        status = "🔴 CONFLICT DETECTED" if state.conflict_detected else "🟢 NO CONFLICT"
        print(f"{status}")
        print(f"   Summary: {state.conflict_report['conflict_summary']}")
        
        return state.model_dump()
    except Exception as e:
        print(f"❌ Conflict detection failed: {e}")
        raise

def step_4_orchestrate(step_input):
    """Step 4: Resolve conflicts and determine execution strategy"""
    print(f"\n{'='*60}")
    print("🎯 STEP 4: Orchestration")
    print(f"{'='*60}")
    
    try:
        # Get state from previous step
        state = OptimizationWorkflowState(**step_input.previous_step_content)
        state.current_step = "ORCHESTRATION"
        
        if not state.conflict_detected:
            # No conflict - parallel execution
            state.execution_strategy = "PARALLEL"
            state.final_configuration = {
                "execution_mode": "parallel",
                "configurations": state.active_intents + [{
                    "intent": state.parsed_intent.model_dump() if state.parsed_intent else None,
                    "plan": state.optimization_plan
                }]
            }
            print(f"✅ Execution Strategy: PARALLEL")
            print(f"   Total Running Intents: {len(state.final_configuration['configurations'])}")
            
            # Save to storage
            if state.parsed_intent and state.optimization_plan:
                new_entry = {
                    "intent": state.parsed_intent.model_dump(),
                    "plan": state.optimization_plan
                }
                active = load_active_intents()
                active.append(new_entry)
                save_active_intents(active)
                print(f"💾 Saved. Total active intents: {len(active)}")
            
            return state.model_dump()
        
        # Conflict detected - use orchestrator
        orch_result_json = process_new_intent(
            state.parsed_intent.model_dump_json(),
            json.dumps(state.active_intents)
        )
        orch_result = json.loads(orch_result_json)
        
        state.execution_strategy = orch_result.get('execution_strategy', 'MERGED')
        state.final_configuration = orch_result.get('final_configuration', {})
        
        print(f"✅ Execution Strategy: {state.execution_strategy}")
        if state.execution_strategy == "MERGED":
            merged = state.final_configuration
            print(f"   Resolution: {merged.get('resolution_strategy')}")
            print(f"   Contributing Intents: {len(merged.get('contributing_intents', []))}")
        
        # Save to storage
        if state.parsed_intent and state.optimization_plan:
            new_entry = {
                "intent": state.parsed_intent.model_dump(),
                "plan": state.optimization_plan
            }
            active = load_active_intents()
            active.append(new_entry)
            save_active_intents(active)
            print(f"💾 Saved. Total active intents: {len(active)}")
        
        return state.model_dump()
    except Exception as e:
        print(f"❌ Orchestration failed: {e}")
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
            description="Find optimal base station configuration"
        ),
        Step(
            name="Detect Conflicts",
            executor=step_3_detect_conflicts,
            description="Check for conflicts with active intents"
        ),
        Step(
            name="Orchestrate Resolution",
            executor=step_4_orchestrate,
            description="Resolve conflicts and determine execution strategy"
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
    print("   Steps: 4 (Parse → Optimize → Detect → Orchestrate)")
    
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
        print("   Steps: Parse → Optimize → Detect Conflicts → Orchestrate")
        print("\nPress Ctrl+C to stop\n")
        agent_os.serve(app="workflow_steps:app", reload=True, port=7777)
    else:
        # Demo run
        print("💡 Run with --playground:")
        print("   python workflow_steps.py --playground")
        print("\nOr test workflow directly:")
        
        # Load active intents
        active = load_active_intents()
        
        # Create initial state
        initial_state = OptimizationWorkflowState(
            natural_language_intent="Increase coverage in Kadikoy region. RX power must be at least -95 dBm. Priority is high.",
            active_intents=active
        )
        
        # Run workflow
        result = optimization_workflow.run(initial_state)
        
        print(f"\n{'='*60}")
        print("✅ WORKFLOW COMPLETED")
        print(f"{'='*60}")
        print(f"Execution Strategy: {result.content.execution_strategy}")
        print(f"Conflict Detected: {result.content.conflict_detected}")
        print(f"Errors: {result.content.errors if result.content.errors else 'None'}")

