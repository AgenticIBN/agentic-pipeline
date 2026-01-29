#!/usr/bin/env python3
"""
Agno Workflow: Complete 6G Intent-Based Optimization Pipeline
Multi-agent system with conflict resolution

Run: python workflow_main.py
Then open: http://localhost:7777 (Agno Playground)
"""

from agno.os import AgentOS
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json

load_dotenv()

# Import EXISTING agents and tools
from intent_parser.intent_parser_agent import intent_parser_agent
from optimization_agent import optimize_from_intent, IntentParse, OptimizationPlan
from conflict_detector_agent import detect_conflicts
from conflict_resolution_orchestrator import process_new_intent

# =====================================================
# WORKFLOW STATE
# =====================================================

class ConflictReport(BaseModel):
    """Conflict detection result"""
    is_conflicted: bool
    conflict_summary: str
    details: List[Dict[str, Any]] = Field(default_factory=list)
    resolution_recommendation: str = ""

class WorkflowState(BaseModel):
    """Shared state across workflow steps"""
    # Input
    natural_language_intent: str
    active_intents: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Step outputs
    parsed_intent: Optional[IntentParse] = None
    optimization_plan: Optional[OptimizationPlan] = None
    conflict_report: Optional[ConflictReport] = None
    final_configuration: Optional[Dict[str, Any]] = None
    
    # Metadata
    current_step: str = "START"
    execution_strategy: str = "UNKNOWN"
    notes: List[str] = Field(default_factory=list)

# =====================================================
# USE EXISTING AGENTS
# =====================================================

# Agent 1: Intent Parser (already defined in intent_parser_agent.py)
# intent_parser_agent - imported above

# Create a workflow execution tool
def execute_workflow_tool(natural_language_intent: str, active_intents_json: str = "[]") -> str:
    """
    Execute the complete 6G optimization workflow.
    
    Args:
        natural_language_intent: User's natural language intent
        active_intents_json: JSON string of currently active intents (default: empty list)
    
    Returns:
        JSON string with workflow execution results
    """
    import json
    
    try:
        active_intents = json.loads(active_intents_json)
    except:
        active_intents = []
    
    result = run_workflow(natural_language_intent, active_intents)
    return json.dumps(result, indent=2)

# Agents are now just wrappers/proxies since the real logic is in the imported tools

# =====================================================
# WORKFLOW DEFINITION
# =====================================================

def run_workflow(natural_language: str, active_intents: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute the complete workflow
    
    Args:
        natural_language: User's natural language intent
        active_intents: List of currently active intents in system
    
    Returns:
        RunResponse with final configuration
    """
    
    if active_intents is None:
        active_intents = []
    
    state = WorkflowState(
        natural_language_intent=natural_language,
        active_intents=active_intents
    )
    
    print("\n" + "="*80)
    print("🚀 WORKFLOW EXECUTION STARTED")
    print("="*80)
    print(f"Input: {natural_language}")
    print(f"Active Intents: {len(active_intents)}")
    
    # ============================================
    # STEP 1: INTENT PARSING
    # ============================================
    state.current_step = "INTENT_PARSING"
    print(f"\n{'='*80}")
    print(f"📋 STEP 1: Intent Parser Agent")
    print(f"{'='*80}")
    
    # Create a message for the agent with workflow context
    parse_message = f"""
🎯 **WORKFLOW STEP 1/4: INTENT PARSING**

Parse this natural language intent:
```
{natural_language}
```

Active intents in system: {len(active_intents)}

Please parse this into structured IntentParse JSON.
"""
    
    parse_response = intent_parser_agent.run(parse_message)
    
    # Extract IntentParse from response
    try:
        parsed_data = json.loads(parse_response.content.model_dump_json())
        state.parsed_intent = IntentParse(**parsed_data)
        print(f"✅ Intent Parsed:")
        print(f"   Target Area: {state.parsed_intent.target_area}")
        print(f"   Target KPIs: {state.parsed_intent.target_kpis}")
        print(f"   Priority: {state.parsed_intent.priority}")
        print(f"   Confidence: {state.parsed_intent.confidence}")
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        return {
            "error": f"Failed at intent parsing: {e}",
            "step": "INTENT_PARSING"
        }
    
    # ============================================
    # STEP 2: OPTIMIZATION
    # ============================================
    state.current_step = "OPTIMIZATION"
    print(f"\n{'='*80}")
    print(f"🔧 STEP 2: Optimization Agent (Tool)")
    print(f"{'='*80}")
    
    # Use the existing optimization_agent tool directly
    try:
        opt_result_json = optimize_from_intent(state.parsed_intent.model_dump_json())
        opt_result = json.loads(opt_result_json)
        
        state.optimization_plan = OptimizationPlan(**opt_result)
        print(f"✅ Optimization Complete:")
        print(f"   Config ID: {state.optimization_plan.selected_config_id}")
        print(f"   Changes: {len(state.optimization_plan.changes)}")
        print(f"   Constraints Satisfied: {state.optimization_plan.constraints_satisfied}")
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return {
            "error": f"Failed at optimization: {e}",
            "step": "OPTIMIZATION"
        }
    
    # ============================================
    # STEP 3: CONFLICT DETECTION
    # ============================================
    state.current_step = "CONFLICT_DETECTION"
    print(f"\n{'='*80}")
    print(f"🔍 STEP 3: Conflict Detector (Tool)")
    print(f"{'='*80}")
    
    if len(active_intents) == 0:
        print("ℹ️ No active intents in system - skipping conflict detection")
        state.execution_strategy = "SINGLE"
        state.conflict_report = ConflictReport(
            is_conflicted=False,
            conflict_summary="No active intents",
            resolution_recommendation="Execute new intent directly"
        )
    else:
        # Use the existing conflict detector tool directly
        try:
            conflict_result_json = detect_conflicts(
                new_intent_json=state.parsed_intent.model_dump_json(),
                new_plan_json=state.optimization_plan.model_dump_json(),
                active_intents_data=json.dumps(active_intents)
            )
            conflict_result = json.loads(conflict_result_json)
            conflict_data = conflict_result['conflict_report']
            
            state.conflict_report = ConflictReport(**conflict_data)
            print(f"{'🔴 CONFLICT DETECTED' if state.conflict_report.is_conflicted else '🟢 NO CONFLICT'}")
            print(f"   Summary: {state.conflict_report.conflict_summary}")
        except Exception as e:
            print(f"❌ Conflict detection failed: {e}")
            # Default to no conflict
            state.conflict_report = ConflictReport(
                is_conflicted=False,
                conflict_summary="Detection failed, defaulting to no conflict",
                resolution_recommendation="Proceed with caution"
            )
    
    # ============================================
    # STEP 4: ORCHESTRATION
    # ============================================
    state.current_step = "ORCHESTRATION"
    print(f"\n{'='*80}")
    print(f"🎯 STEP 4: Orchestrator (Tool)")
    print(f"{'='*80}")
    
    if not state.conflict_report.is_conflicted:
        # No conflict - parallel execution
        state.execution_strategy = "PARALLEL"
        state.final_configuration = {
            "execution_mode": "parallel",
            "configurations": active_intents + [{
                "intent": state.parsed_intent.model_dump(),
                "plan": state.optimization_plan.model_dump()
            }]
        }
        print(f"✅ Execution Strategy: PARALLEL")
        print(f"   Total Running Intents: {len(state.final_configuration['configurations'])}")
    else:
        # Conflict detected - use orchestrator tool directly
        try:
            orch_result_json = process_new_intent(
                new_intent_json=state.parsed_intent.model_dump_json(),
                active_intents_json=json.dumps(active_intents)
            )
            orch_result = json.loads(orch_result_json)
            
            state.execution_strategy = orch_result.get('execution_strategy', 'MERGED')
            state.final_configuration = orch_result.get('final_configuration', {})
            state.notes = orch_result.get('notes', [])
            
            print(f"✅ Execution Strategy: {state.execution_strategy}")
            if state.execution_strategy == "MERGED":
                merged = state.final_configuration
                print(f"   Resolution Strategy: {merged.get('resolution_strategy')}")
                print(f"   Contributing Intents: {len(merged.get('contributing_intents', []))}")
                print(f"   Priority Weights: {merged.get('priority_weights')}")
        except Exception as e:
            print(f"❌ Orchestration failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            state.execution_strategy = "FAILED"
            state.final_configuration = {"error": str(e)}
    
    # ============================================
    # WORKFLOW COMPLETE
    # ============================================
    print(f"\n{'='*80}")
    print(f"✅ WORKFLOW EXECUTION COMPLETED")
    print(f"{'='*80}")
    print(f"Final Strategy: {state.execution_strategy}")
    print(f"Notes: {state.notes}")
    
    return {
        "final_configuration": state.final_configuration,
        "execution_strategy": state.execution_strategy,
        "conflict_detected": state.conflict_report.is_conflicted if state.conflict_report else False,
        "total_steps": 4,
        "notes": state.notes
    }

# =====================================================
# AGNO OS SETUP
# =====================================================

def setup_agno_os():
    """Setup AgentOS for playground visualization"""
    
    # Create a Workflow Agent that can execute the full pipeline
    from agno.agent import Agent
    from agno.models.google import Gemini
    
    workflow_agent = Agent(
        name="6G Workflow Orchestrator",
        model=Gemini(id="gemini-2.5-flash"),
        description="Complete 6G network optimization workflow: Intent Parsing → Optimization → Conflict Detection → Resolution",
        instructions=[
            "You execute the complete 6G network optimization workflow.",
            "When the user provides a natural language intent, call execute_workflow_tool to run all 4 steps:",
            "1. Intent Parsing",
            "2. Optimization", 
            "3. Conflict Detection",
            "4. Orchestration & Resolution",
            "You will see progress updates for each step.",
            "After workflow completes, summarize the final configuration and execution strategy."
        ],
        tools=[execute_workflow_tool],
        markdown=True
    )
    
    # Create AgentOS instance with the workflow agent
    agent_os = AgentOS(
        name="6G Intent Optimization System",
        agents=[workflow_agent]
    )
    
    return agent_os

# Create global app for serve
agent_os = setup_agno_os()
app = agent_os.get_app()

# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    import sys
    
    # Check if running in playground mode
    if "--playground" in sys.argv:
        print("🎮 Starting Agno Playground...")
        print("🌐 Open browser: http://localhost:7777")
        print("Press Ctrl+C to stop")
        agent_os.serve(app="workflow_main:app", reload=True, port=7777)
    else:
        # Run demo scenarios
        print("💡 Tip: Run with --playground flag to start Agno Playground")
        print("   python workflow_main.py --playground")
        print("\n" + "="*80)
        print("Running Demo Scenarios...")
        print("="*80)
        
        # Scenario 1: New CRITICAL intent vs Active LOW intent
        print("\n\n" + "="*80)
        print("📋 SCENARIO 1: CRITICAL Intent vs Active LOW Intent")
        print("="*80)
        
        active_low = [{
            "intent": {
                "target_area": "Kadıköy",
                "target_kpis": ["RX_POWER"],
                "priority": "LOW",
                "confidence": 0.85
            },
            "plan": {
                "selected_config_id": 12345,
                "changes": [
                    {"param": "tx0_P_dBm", "before": 40.0, "after": 35.0, "unit": "dBm"}
                ],
                "expected_kpis": {"RX_POWER": -82.0},
                "constraints_satisfied": True
            }
        }]
        
        result1 = run_workflow(
            "Kadıköy bölgesinde SINR'ı 10 dB üzerine çıkar, CRITICAL öncelik",
            active_intents=active_low
        )
        
        # Scenario 2: Different areas (no conflict)
        print("\n\n" + "="*80)
        print("📋 SCENARIO 2: Different Areas - No Conflict")
        print("="*80)
        
        active_kadikoy = [{
            "intent": {
                "target_area": "Kadıköy",
                "target_kpis": ["RX_POWER"],
                "priority": "HIGH",
                "confidence": 0.9
            },
            "plan": {
                "selected_config_id": 67890,
                "changes": [{"param": "tx0_P_dBm", "before": 42.0, "after": 40.0}],
                "expected_kpis": {"RX_POWER": -80.0},
                "constraints_satisfied": True
            }
        }]
        
        result2 = run_workflow(
            "Ümraniye bölgesinde throughput'u artır, MEDIUM öncelik",
            active_intents=active_kadikoy
        )
        
        print("\n\n" + "="*80)
        print("✅ DEMO COMPLETED")
        print("="*80)
        print("\n🎮 To visualize in Agno Playground:")
        print("   python workflow_main.py --playground")
        print("   Then open: http://localhost:7777")
