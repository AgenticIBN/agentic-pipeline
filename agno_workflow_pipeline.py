#!/usr/bin/env python3
"""
AgentOS Workflow Pipeline
Complete 6G Network Optimization Pipeline using AgentOS Workflow

Architecture:
1. Intent Parser Agent → Parse natural language intent
2. Optimization Agent V2 → Generate optimal configuration
3. Conflict Detector Agent → Detect conflicts with active intents
4. Resolution Agent → Resolve conflicts (Priority or Weighted Merge)
5. Final Output → Combined configuration and execution log

Run: python agno_workflow_pipeline.py --playground
Open: http://localhost:7777 → Workflows tab
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from agno.os import AgentOS
from agno.workflow import Workflow, Step
from agno.agent import Agent
from agno.models.groq import Groq

# Import existing agents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "intent_parser"))
from intent_parser.intent_parser_agent import intent_parser_agent, IntentParse

# Import helper modules
from optimization_agent_v2 import OptimizationAgent
from conflict_detector_agent import detect_conflicts
from priority_based_resolution_agent import resolve_by_priority, PriorityBasedResolutionOutput
from weighted_merge_resolution_agent import resolve_by_weighted_merge, WeightedMergeOutput

load_dotenv()

# ============================================================================
# WORKFLOW STATE MODEL
# ============================================================================

class PipelineState(BaseModel):
    """Complete state that flows through the workflow pipeline."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra='allow'
    )
    
    # Input
    natural_language_intent: str = Field(..., description="User's natural language intent")
    resolution_strategy: str = Field(
        default="PRIORITY",
        description="Conflict resolution strategy to use (PRIORITY or WEIGHTED_MERGE)"
    )
    
    # Step 1: Intent Parser Output
    parsed_intent: Any = None
    
    # Step 2: Optimization Output
    optimization_result: Any = None
    
    # Step 3: Conflict Detection Output
    conflict_report: Any = None
    has_conflict: bool = False
    
    # Step 4: Resolution Output (if conflict exists)
    resolution_output: Any = None
    
    # Final Output
    final_configuration: Any = None
    execution_log: Any = Field(default_factory=list)
    
    # Metadata
    workflow_id: str = Field(default_factory=lambda: f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    current_step: str = "INITIAL"


# ============================================================================
# STORAGE FOR ACTIVE INTENTS
# ============================================================================

ACTIVE_INTENTS_FILE = "active_intents_workflow.json"

def load_active_intents() -> list:
    """Load active optimization results from storage."""
    if os.path.exists(ACTIVE_INTENTS_FILE):
        with open(ACTIVE_INTENTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_active_intents(intents: list) -> None:
    """Save active optimization results to storage."""
    with open(ACTIVE_INTENTS_FILE, 'w') as f:
        json.dump(intents, f, indent=2)

def add_active_intent(result: dict) -> None:
    """Add a new optimization result to active intents."""
    active = load_active_intents()
    active.append(result)
    save_active_intents(active)


# ============================================================================
# WORKFLOW STEP 1: INTENT PARSING
# ============================================================================

def step_1_parse_intent(step_input) -> PipelineState:
    """
    Step 1: Parse natural language intent using Intent Parser Agent
    
    Input: PipelineState with natural_language_intent
    Output: PipelineState with parsed_intent
    """
    state = step_input.input if hasattr(step_input, 'input') else step_input
    
    print("\n" + "="*70)
    print("📋 STEP 1: INTENT PARSING")
    print("="*70)
    print(f"Input: {state.natural_language_intent}")
    
    try:
        # Run intent parser agent
        parse_response = intent_parser_agent.run(state.natural_language_intent)
        state.parsed_intent = parse_response.content
        state.current_step = "INTENT_PARSED"
        state.execution_log.append(f"✅ Intent parsed successfully")
        
        print(f"\n✅ Intent Parsed:")
        
        # Handle both Pydantic model and dict
        if hasattr(state.parsed_intent, 'target_area'):
            # Pydantic model
            print(f"   Target Area: {state.parsed_intent.target_area}")
            print(f"   Target KPIs: {state.parsed_intent.target_kpis}")
            print(f"   Priority: {state.parsed_intent.priority}")
            print(f"   Confidence: {state.parsed_intent.confidence}")
            
            if state.parsed_intent.kpi_thresholds:
                print(f"   Thresholds:")
                for th in state.parsed_intent.kpi_thresholds:
                    if hasattr(th, 'kpi'):
                        print(f"      - {th.kpi}: {th.op} {th.value}")
                    else:
                        print(f"      - {th.get('kpi')}: {th.get('op')} {th.get('value')}")
        else:
            # Dict
            print(f"   Target Area: {state.parsed_intent.get('target_area')}")
            print(f"   Target KPIs: {state.parsed_intent.get('target_kpis')}")
            print(f"   Priority: {state.parsed_intent.get('priority')}")
            print(f"   Confidence: {state.parsed_intent.get('confidence')}")
            
            if state.parsed_intent.get('kpi_thresholds'):
                print(f"   Thresholds:")
                for th in state.parsed_intent.get('kpi_thresholds', []):
                    print(f"      - {th.get('kpi')}: {th.get('op')} {th.get('value')}")
        
    except Exception as e:
        state.execution_log.append(f"❌ Intent parsing failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise
    
    return state


# ============================================================================
# WORKFLOW STEP 2: OPTIMIZATION
# ============================================================================

def step_2_optimize_configuration(step_input) -> PipelineState:
    """
    Step 2: Generate optimal configuration using Optimization Agent V2
    
    Input: PipelineState with parsed_intent
    Output: PipelineState with optimization_result
    """
    state = step_input.input if hasattr(step_input, 'input') else step_input
    
    print("\n" + "="*70)
    print("🔧 STEP 2: CONFIGURATION OPTIMIZATION")
    print("="*70)
    
    if not state.parsed_intent:
        raise ValueError("No parsed intent available for optimization")
    
    try:
        # Initialize optimization agent
        surrogate_path = "models/surrogate.joblib"
        if not os.path.exists(surrogate_path):
            raise FileNotFoundError(f"Surrogate model not found at {surrogate_path}")
        
        # Load surrogate model first
        from optimization_agent_v2 import SurrogateModel
        surrogate = SurrogateModel.load(surrogate_path)
        
        # Initialize optimization agent with loaded surrogate
        opt_agent = OptimizationAgent(surrogate=surrogate)
        
        # Convert parsed intent to optimization input
        if hasattr(state.parsed_intent, 'model_dump'):
            intent_dict = state.parsed_intent.model_dump()
        elif isinstance(state.parsed_intent, dict):
            intent_dict = state.parsed_intent
        else:
            intent_dict = dict(state.parsed_intent)
        
        # Add default current_config if not present
        if 'current_config' not in intent_dict or not intent_dict['current_config']:
            # Default TX configuration (baseline) - matches surrogate model parameters
            intent_dict['current_config'] = {
                'tx0_on': True, 'tx0_P_dBm': 30.0, 'tx0_dAz': 0.0, 'tx0_dEl': 0.0,
                'tx1_on': True, 'tx1_P_dBm': 30.0, 'tx1_dAz': 0.0, 'tx1_dEl': 0.0,
                'tx2_on': True, 'tx2_P_dBm': 30.0, 'tx2_dAz': 0.0, 'tx2_dEl': 0.0,
                'tx3_on': True, 'tx3_P_dBm': 30.0, 'tx3_dAz': 0.0, 'tx3_dEl': 0.0
            }
        
        # Run optimization
        result = opt_agent.optimize(intent_dict)
        state.optimization_result = result
        state.current_step = "OPTIMIZED"
        state.execution_log.append(f"✅ Optimization completed")
        
        print(f"\n✅ Optimization Result:")
        print(f"   Result ID: {result.get('result_id', 'N/A')}")
        print(f"   Status: {result.get('status', 'N/A')}")
        
        if 'config_changes' in result:
            changes = result['config_changes']
            print(f"   Configuration Changes ({len(changes)}):")
            for param, change in list(changes.items())[:5]:  # Show first 5
                print(f"      - {param}: {change:.4f}")
        
        if 'predicted_kpis' in result:
            kpis = result['predicted_kpis']
            print(f"   Predicted KPIs:")
            for kpi, value in kpis.items():
                print(f"      - {kpi}: {value:.4f}")
        
    except Exception as e:
        state.execution_log.append(f"❌ Optimization failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise
    
    return state


# ============================================================================
# WORKFLOW STEP 3: CONFLICT DETECTION
# ============================================================================

def step_3_detect_conflicts(step_input) -> PipelineState:
    """
    Step 3: Detect conflicts with active intents using Conflict Detector Agent
    
    Input: PipelineState with optimization_result
    Output: PipelineState with conflict_report
    """
    state = step_input.input if hasattr(step_input, 'input') else step_input
    
    print("\n" + "="*70)
    print("🔍 STEP 3: CONFLICT DETECTION")
    print("="*70)
    
    if not state.optimization_result:
        raise ValueError("No optimization result available for conflict detection")
    
    try:
        # Load active intents
        active_results = load_active_intents()
        print(f"Active intents count: {len(active_results)}")
        
        # Detect conflicts using the function (not a class)
        conflict_report = detect_conflicts(
            new_result=state.optimization_result,
            active_results=active_results
        )
        
        # Convert ConflictReport to dict
        state.conflict_report = conflict_report.model_dump()
        state.has_conflict = conflict_report.is_conflicted
        state.current_step = "CONFLICT_CHECKED"
        
        if state.has_conflict:
            state.execution_log.append(f"⚠️  Conflicts detected: {conflict_report.num_conflicts}")
            print(f"\n⚠️  CONFLICTS DETECTED!")
            print(f"   Number of conflicts: {conflict_report.num_conflicts}")
            print(f"   Summary: {conflict_report.conflict_summary}")
            
            if conflict_report.details:
                print(f"   Conflict Details:")
                for i, detail in enumerate(conflict_report.details[:3], 1):  # Show first 3
                    print(f"      {i}. {detail.conflict_type} - {detail.description}")
        else:
            state.execution_log.append(f"✅ No conflicts detected")
            print(f"\n✅ NO CONFLICTS - Configuration can be applied directly")
        
    except Exception as e:
        state.execution_log.append(f"❌ Conflict detection failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise
    
    return state


# ============================================================================
# WORKFLOW STEP 4: CONFLICT RESOLUTION (CONDITIONAL)
# ============================================================================

def step_4_resolve_conflicts(step_input) -> PipelineState:
    """
    Step 4: Resolve conflicts using selected strategy (PRIORITY or WEIGHTED_MERGE)
    
    Input: PipelineState with conflict_report
    Output: PipelineState with resolution_output
    """
    state = step_input.input if hasattr(step_input, 'input') else step_input
    
    print("\n" + "="*70)
    print("⚖️  STEP 4: CONFLICT RESOLUTION")
    print("="*70)
    
    # Skip if no conflicts
    if not state.has_conflict:
        print("No conflicts to resolve, skipping resolution step")
        state.execution_log.append("⏭️  Skipped: No conflicts to resolve")
        state.current_step = "RESOLUTION_SKIPPED"
        return state
    
    if not state.conflict_report:
        raise ValueError("No conflict report available for resolution")
    
    try:
        active_results = load_active_intents()
        
        print(f"Resolution Strategy: {state.resolution_strategy}")
        
        # Apply resolution based on strategy
        if state.resolution_strategy == "PRIORITY":
            resolution = resolve_by_priority(
                conflict_report=state.conflict_report,
                new_result=state.optimization_result,
                active_results=active_results
            )
            state.resolution_output = resolution.model_dump()
            
            print(f"\n✅ Priority-Based Resolution:")
            print(f"   Winning Result: {resolution.winning_result_id}")
            print(f"   Winning Priority: {resolution.winning_priority}")
            print(f"   Rejected Results: {len(resolution.rejected_result_ids)}")
            print(f"   Notes: {resolution.resolution_notes}")
            
        else:  # WEIGHTED_MERGE
            resolution = resolve_by_weighted_merge(
                conflict_report=state.conflict_report,
                new_result=state.optimization_result,
                active_results=active_results
            )
            state.resolution_output = resolution.model_dump()
            
            print(f"\n✅ Weighted Merge Resolution:")
            print(f"   Merged Result ID: {resolution.merged_result_id}")
            print(f"   Contributing Results: {len(resolution.contributing_results)}")
            print(f"   Notes: {resolution.resolution_notes}")
            
            if resolution.merge_details:
                print(f"   Merge Details (sample):")
                for detail in resolution.merge_details[:3]:  # Show first 3
                    print(f"      - {detail.get('parameter')}: {detail.get('merged_value')}")
        
        state.current_step = "RESOLVED"
        state.execution_log.append(f"✅ Conflicts resolved using {state.resolution_strategy}")
        
    except Exception as e:
        state.execution_log.append(f"❌ Conflict resolution failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise
    
    return state


# ============================================================================
# WORKFLOW STEP 5: FINALIZE CONFIGURATION
# ============================================================================

def step_5_finalize_configuration(step_input) -> PipelineState:
    """
    Step 5: Finalize configuration and update active intents
    
    Input: PipelineState with resolution_output (or optimization_result if no conflicts)
    Output: PipelineState with final_configuration
    """
    state = step_input.input if hasattr(step_input, 'input') else step_input
    
    print("\n" + "="*70)
    print("✨ STEP 5: FINALIZE CONFIGURATION")
    print("="*70)
    
    try:
        # Determine final configuration based on conflict resolution
        if state.has_conflict and state.resolution_output:
            if state.resolution_strategy == "PRIORITY":
                # Use winning configuration
                state.final_configuration = state.resolution_output.get('winning_config')
                print(f"Using winning configuration from priority-based resolution")
                
            else:  # WEIGHTED_MERGE
                # Use merged configuration
                state.final_configuration = state.resolution_output.get('merged_config')
                print(f"Using merged configuration from weighted merge resolution")
        else:
            # No conflicts, use optimization result directly
            state.final_configuration = state.optimization_result
            print(f"Using optimization result directly (no conflicts)")
        
        # Add to active intents
        if state.final_configuration:
            add_active_intent(state.final_configuration)
            print(f"✅ Configuration added to active intents")
        
        state.current_step = "COMPLETED"
        state.execution_log.append(f"✅ Workflow completed successfully")
        
        # Print summary
        print(f"\n" + "="*70)
        print("📊 WORKFLOW SUMMARY")
        print("="*70)
        print(f"Workflow ID: {state.workflow_id}")
        print(f"Timestamp: {state.timestamp}")
        print(f"Final Status: {state.current_step}")
        print(f"\nExecution Log:")
        for i, log in enumerate(state.execution_log, 1):
            print(f"  {i}. {log}")
        
        if state.final_configuration:
            print(f"\n📋 Final Configuration:")
            print(f"   ID: {state.final_configuration.get('result_id', 'N/A')}")
            print(f"   Status: {state.final_configuration.get('status', 'N/A')}")
            if 'config_changes' in state.final_configuration:
                print(f"   Total Changes: {len(state.final_configuration['config_changes'])}")
        
    except Exception as e:
        state.execution_log.append(f"❌ Finalization failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise
    
    return state


# ============================================================================
# WORKFLOW DEFINITION
# ============================================================================

# Create workflow
optimization_workflow = Workflow(
    name="6G_Network_Optimization_Pipeline",
    description="Complete pipeline: Intent → Optimize → Detect Conflicts → Resolve → Execute",
    steps=[
        Step(
            name="parse_intent",
            description="Parse natural language intent into structured format",
            executor=step_1_parse_intent
        ),
        Step(
            name="optimize_configuration",
            description="Generate optimal base-station configuration",
            executor=step_2_optimize_configuration
        ),
        Step(
            name="detect_conflicts",
            description="Detect conflicts with active intents",
            executor=step_3_detect_conflicts
        ),
        Step(
            name="resolve_conflicts",
            description="Resolve conflicts using priority or weighted merge",
            executor=step_4_resolve_conflicts
        ),
        Step(
            name="finalize_configuration",
            description="Finalize and store configuration",
            executor=step_5_finalize_configuration
        )
    ]
)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="6G Network Optimization Workflow")
    parser.add_argument("--playground", action="store_true", help="Start AgentOS playground")
    parser.add_argument("--intent", type=str, help="Natural language intent to process")
    parser.add_argument("--strategy", type=str, choices=["PRIORITY", "WEIGHTED_MERGE"], 
                       default="PRIORITY", help="Conflict resolution strategy")
    parser.add_argument("--clear", action="store_true", help="Clear active intents before running")
    
    args = parser.parse_args()
    
    # Clear active intents if requested
    if args.clear:
        if os.path.exists(ACTIVE_INTENTS_FILE):
            os.remove(ACTIVE_INTENTS_FILE)
            print("🗑️  Cleared active intents")
    
    # Start playground mode
    if args.playground:
        print("\n" + "="*70)
        print("🚀 Starting AgentOS Playground")
        print("="*70)
        print("🌐 Playground URL: http://localhost:7777")
        print("📊 Navigate to: Workflows tab → 6G_Network_Optimization_Pipeline")
        print("💡 Tips:")
        print("   - Her intent'i ayrı ayrı test edebilirsiniz")
        print("   - Workflow step'lerini görsel olarak görebilirsiniz")
        print("   - Execution log'ları takip edebilirsiniz")
        print("   - Active intents'i kontrol edin: cat active_intents_workflow.json")
        print("="*70 + "\n")
        
        # Create AgentOS instance and add workflow
        agent_os = AgentOS(
            name="6G_Network_Optimizer",
            workflows=[optimization_workflow]
        )
        
        # Start playground
        print("⏳ Starting playground server...")
        print("✅ Server will start at: http://localhost:7777")
        print("\n💡 Open your browser and navigate to the Workflows tab\n")
        
        # Get the FastAPI app and serve it
        app = agent_os.get_app()
        agent_os.serve(app=app, host="localhost", port=7777)
    
    # Run with specific intent
    elif args.intent:
        print("\n" + "="*70)
        print("🚀 Running Optimization Workflow")
        print("="*70)
        
        # Create initial state
        initial_state = PipelineState(
            natural_language_intent=args.intent,
            resolution_strategy=args.strategy
        )
        
        print(f"Intent: {args.intent}")
        print(f"Strategy: {args.strategy}")
        print("="*70)
        
        # Run workflow
        try:
            result = optimization_workflow.run(input=initial_state)
            
            # Save result to file
            result_file = f"workflow_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w') as f:
                # Convert to dict for JSON serialization
                if hasattr(result, 'model_dump'):
                    result_dict = result.model_dump()
                else:
                    result_dict = result
                json.dump(result_dict, f, indent=2, default=str)
            
            print(f"\n✅ Workflow completed successfully!")
            print(f"📁 Result saved to: {result_file}")
            
        except Exception as e:
            print(f"\n❌ Workflow failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    else:
        print("Usage:")
        print("  Playground mode: python agno_workflow_pipeline.py --playground")
        print("  Direct run:      python agno_workflow_pipeline.py --intent 'improve coverage in cell1' --strategy PRIORITY")
        print("  Clear intents:   python agno_workflow_pipeline.py --clear --intent 'your intent'")
