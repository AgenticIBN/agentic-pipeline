#!/usr/bin/env python3
"""
AgentOS Workflow Pipeline (Full Agent-Based)
Complete 6G Network Optimization Pipeline using AgentOS Workflow with ALL agents

Architecture (ALL Agent-Based):
1. Intent Parser Agent → Parse natural language intent (Agno Agent)
2. Optimization Agent → Generate optimal configuration (Agno Agent)
3. Conflict Detector Agent → Detect conflicts with active intents (Agno Agent)
4. Resolution Agent → Resolve conflicts - Priority or Weighted Merge (Agno Agent)
5. Final Output → Combined configuration and execution log

All components now use Agno Agent structure with LLM reasoning!

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

# Import new Agno-based agents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
from agents.optimization_agent import run_optimization
from agents.conflict_detector_agent import run_conflict_detection
from agents.priority_resolution_agent import run_priority_resolution
from agents.weighted_merge_agent import run_weighted_merge

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

def get_current_system_config() -> dict:
    """
    Get the current system configuration from the last finalized intent.
    If no active intents exist, return default configuration.
    
    Returns:
        Current configuration dict with tx parameters
    """
    active_intents = load_active_intents()
    
    if not active_intents:
        # No active intents - return default baseline configuration
        print("📌 No active intents found - using default baseline config")
        return {
            'tx0_on': True, 'tx0_P_dBm': 30.0, 'tx0_dAz': 0.0, 'tx0_dEl': 0.0,
            'tx1_on': True, 'tx1_P_dBm': 30.0, 'tx1_dAz': 0.0, 'tx1_dEl': 0.0,
            'tx2_on': True, 'tx2_P_dBm': 30.0, 'tx2_dAz': 0.0, 'tx2_dEl': 0.0,
            'tx3_on': True, 'tx3_P_dBm': 30.0, 'tx3_dAz': 0.0, 'tx3_dEl': 0.0
        }
    
    # Get the last active intent (most recent)
    last_intent = active_intents[-1]
    
    # Extract final configuration from the last intent
    config = extract_final_config_from_result(last_intent)
    
    print(f"📌 Loaded current config from last intent (ID: {last_intent.get('result_id', 'N/A')})")
    return config

def extract_final_config_from_result(result: dict) -> dict:
    """
    Extract the final configuration from an optimization result.
    Reconstructs the full config by applying changes to base config.
    
    Args:
        result: Optimization result dict
    
    Returns:
        Complete configuration dict
    """
    # Try to get output section
    output = result.get('output', {})
    
    # Get changes list
    changes = output.get('changes', [])
    
    # Start with default config (or stored final if available)
    if 'final_config' in result:
        return result['final_config']
    
    # Reconstruct from changes
    input_data = result.get('input', {})
    base_config = input_data.get('current_config', {})
    
    if not base_config:
        # Use default
        base_config = {
            'tx0_on': True, 'tx0_P_dBm': 30.0, 'tx0_dAz': 0.0, 'tx0_dEl': 0.0,
            'tx1_on': True, 'tx1_P_dBm': 30.0, 'tx1_dAz': 0.0, 'tx1_dEl': 0.0,
            'tx2_on': True, 'tx2_P_dBm': 30.0, 'tx2_dAz': 0.0, 'tx2_dEl': 0.0,
            'tx3_on': True, 'tx3_P_dBm': 30.0, 'tx3_dAz': 0.0, 'tx3_dEl': 0.0
        }
    
    # Apply changes to base config
    final_config = dict(base_config)
    
    for change in changes:
        param = change.get('param')
        change_val = change.get('change')
        before = change.get('before')
        
        if param and change_val is not None:
            # For boolean params (tx_on), change is the new value
            if param.endswith('_on'):
                final_config[param] = bool(change_val)
            # For numeric params, change is delta
            elif before is not None:
                final_config[param] = float(before) + float(change_val)
            else:
                # No before value, treat change as absolute
                final_config[param] = float(change_val)
    
    return final_config


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
    Step 2: Generate optimal configuration using Optimization Agent (Agno-based)
    
    Input: PipelineState with parsed_intent
    Output: PipelineState with optimization_result
    """
    state = step_input.input if hasattr(step_input, 'input') else step_input
    
    print("\n" + "="*70)
    print("🔧 STEP 2: CONFIGURATION OPTIMIZATION (Agent-based)")
    print("="*70)
    
    if not state.parsed_intent:
        raise ValueError("No parsed intent available for optimization")
    
    try:
        # Convert parsed intent to optimization input
        if hasattr(state.parsed_intent, 'model_dump'):
            intent_dict = state.parsed_intent.model_dump()
        elif isinstance(state.parsed_intent, dict):
            intent_dict = state.parsed_intent
        else:
            intent_dict = dict(state.parsed_intent)
        
        # Get current system configuration (from last finalized intent or default)
        current_config = get_current_system_config()
        intent_dict['current_config'] = current_config
        
        # Load surrogate model to compute current KPIs
        surrogate_path = "models/surrogate.joblib"
        if not os.path.exists(surrogate_path):
            raise FileNotFoundError(f"Surrogate model not found at {surrogate_path}")
        
        from optimization_agent_v2 import SurrogateModel
        surrogate = SurrogateModel.load(surrogate_path)
        
        # Compute current KPIs
        context = {
            "user_set_id": 0,
            "K_users": 800,
            "rx_power_thr_dBm": -95.0,
            "total_tx_power_watt": 0.0,
        }
        current_kpis = surrogate.predict(current_config, context)
        
        # Compute throughput if needed
        if "SINR_p5_dB" in current_kpis:
            if surrogate.has_throughput_col and surrogate.throughput_col in current_kpis:
                throughput = current_kpis[surrogate.throughput_col]
            else:
                throughput = surrogate.compute_throughput_from_sinr(current_kpis["SINR_p5_dB"])
            current_kpis["THROUGHPUT_5P"] = throughput
        
        print(f"\n📊 Current System State:")
        print(f"   Configuration:")
        print(f"      TX0: {'ON' if current_config['tx0_on'] else 'OFF'} | P={current_config['tx0_P_dBm']:.1f}dBm | Az={current_config['tx0_dAz']:.1f}° | El={current_config['tx0_dEl']:.1f}°")
        print(f"      TX1: {'ON' if current_config['tx1_on'] else 'OFF'} | P={current_config['tx1_P_dBm']:.1f}dBm | Az={current_config['tx1_dAz']:.1f}° | El={current_config['tx1_dEl']:.1f}°")
        print(f"      TX2: {'ON' if current_config['tx2_on'] else 'OFF'} | P={current_config['tx2_P_dBm']:.1f}dBm | Az={current_config['tx2_dAz']:.1f}° | El={current_config['tx2_dEl']:.1f}°")
        print(f"      TX3: {'ON' if current_config['tx3_on'] else 'OFF'} | P={current_config['tx3_P_dBm']:.1f}dBm | Az={current_config['tx3_dAz']:.1f}° | El={current_config['tx3_dEl']:.1f}°")
        print(f"   Current KPIs:")
        print(f"      RX_POWER:     {current_kpis.get('Prx_p5_dBm', 0):.2f} dBm")
        print(f"      SINR:         {current_kpis.get('SINR_p5_dB', 0):.2f} dB")
        print(f"      THROUGHPUT:   {current_kpis.get('THROUGHPUT_5P', 0):.2f} Mbps")
        if 'rx_power_coverage_ratio' in current_kpis:
            print(f"      COVERAGE:     {current_kpis.get('rx_power_coverage_ratio', 0):.2%}")
        
        # Add test_name if not present
        if 'test_name' not in intent_dict:
            intent_dict['test_name'] = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add k_users and user_set_id if not present
        intent_dict.setdefault('k_users', 800)
        intent_dict.setdefault('user_set_id', 0)
        
        # Run optimization using Agno-based agent
        print(f"\n🔍 Running optimization from current state...")
        result = run_optimization(intent_dict, surrogate_model_path=surrogate_path)
        
        # Convert OptimizationResult to dict for state storage
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        else:
            result_dict = dict(result)
        
        state.optimization_result = result_dict
        state.current_step = "OPTIMIZED"
        state.execution_log.append(f"✅ Optimization completed (Agent-based)")
        
        print(f"\n✅ Optimization Result (from Agent):")
        print(f"   Result ID: {result_dict.get('result_id', 'N/A')}")
        print(f"   Priority: {result_dict.get('priority', 'N/A')}")
        print(f"   Passed: {result_dict.get('passed', 'N/A')}")
        
        if result.reasoning:
            print(f"\n📝 Agent Reasoning:")
            print(f"   {result.reasoning}")
        
        output = result_dict.get('output', {})
        if 'changes' in output:
            changes = output['changes']
            print(f"\n   Configuration Changes ({len(changes)}):")
            for change in changes[:5]:  # Show first 5
                param = change.get('param')
                change_val = change.get('change')
                unit = change.get('unit', '')
                print(f"      - {param}: {change_val} {unit}")
        
        if 'expected_kpis' in output:
            kpis = output['expected_kpis']
            print(f"\n   Expected KPIs:")
            for kpi, value in kpis.items():
                if value is not None:
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
    Step 3: Detect conflicts with active intents using Conflict Detector Agent (Agno-based)
    
    Input: PipelineState with optimization_result
    Output: PipelineState with conflict_report
    """
    state = step_input.input if hasattr(step_input, 'input') else step_input
    
    print("\n" + "="*70)
    print("🔍 STEP 3: CONFLICT DETECTION (Agent-based)")
    print("="*70)
    
    if not state.optimization_result:
        raise ValueError("No optimization result available for conflict detection")
    
    try:
        # Load active intents
        active_results = load_active_intents()
        print(f"Active intents count: {len(active_results)}")
        
        # Run conflict detection using Agno-based agent
        conflict_analysis = run_conflict_detection(
            new_result=state.optimization_result,
            active_results=active_results
        )
        
        # Convert ConflictAnalysis to dict for state storage
        if hasattr(conflict_analysis, 'model_dump'):
            analysis_dict = conflict_analysis.model_dump()
        else:
            analysis_dict = dict(conflict_analysis)
        
        state.conflict_report = analysis_dict
        state.has_conflict = conflict_analysis.is_conflicted
        state.current_step = "CONFLICT_CHECKED"
        
        if state.has_conflict:
            state.execution_log.append(f"⚠️  Conflicts detected: {conflict_analysis.num_conflicts} (Agent-based)")
            print(f"\n⚠️  CONFLICTS DETECTED (by Agent)!")
            print(f"   Number of conflicts: {conflict_analysis.num_conflicts}")
            print(f"   Summary: {conflict_analysis.conflict_summary}")
            
            if conflict_analysis.reasoning:
                print(f"\n📝 Agent Reasoning:")
                print(f"   {conflict_analysis.reasoning}")
            
            if conflict_analysis.details:
                print(f"\n   Conflict Details:")
                for i, detail in enumerate(conflict_analysis.details[:3], 1):  # Show first 3
                    print(f"      {i}. {detail.conflict_type} - {detail.description}")
            
            if conflict_analysis.resolution_recommendation:
                print(f"\n💡 Recommended Strategy: {conflict_analysis.resolution_recommendation}")
        else:
            state.execution_log.append(f"✅ No conflicts detected (Agent-based)")
            print(f"\n✅ NO CONFLICTS - Configuration can be applied directly")
            
            if conflict_analysis.reasoning:
                print(f"\n📝 Agent Reasoning:")
                print(f"   {conflict_analysis.reasoning}")
        
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
    Step 4: Resolve conflicts using selected strategy (PRIORITY or WEIGHTED_MERGE) - Agent-based
    
    Input: PipelineState with conflict_report
    Output: PipelineState with resolution_output
    """
    state = step_input.input if hasattr(step_input, 'input') else step_input
    
    print("\n" + "="*70)
    print("⚖️  STEP 4: CONFLICT RESOLUTION (Agent-based)")
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
        
        # Apply resolution based on strategy using Agno-based agents
        if state.resolution_strategy == "PRIORITY":
            resolution = run_priority_resolution(
                conflict_report=state.conflict_report,
                new_result=state.optimization_result,
                active_results=active_results
            )
            
            # Convert to dict for state storage
            if hasattr(resolution, 'model_dump'):
                resolution_dict = resolution.model_dump()
            else:
                resolution_dict = dict(resolution)
            
            state.resolution_output = resolution_dict
            
            print(f"\n✅ Priority-Based Resolution (by Agent):")
            print(f"   Winning Result: {resolution.winning_result_id}")
            print(f"   Winning Priority: {resolution.winning_priority}")
            print(f"   Rejected Results: {len(resolution.rejected_result_ids)}")
            print(f"   Notes: {resolution.resolution_notes}")
            
            if resolution.reasoning:
                print(f"\n📝 Agent Reasoning:")
                print(f"   {resolution.reasoning}")
            
        else:  # WEIGHTED_MERGE
            resolution = run_weighted_merge(
                conflict_report=state.conflict_report,
                new_result=state.optimization_result,
                active_results=active_results
            )
            
            # Convert to dict for state storage
            if hasattr(resolution, 'model_dump'):
                resolution_dict = resolution.model_dump()
            else:
                resolution_dict = dict(resolution)
            
            state.resolution_output = resolution_dict
            
            print(f"\n✅ Weighted Merge Resolution (by Agent):")
            print(f"   Merged Result ID: {resolution.merged_result_id}")
            print(f"   Contributing Results: {len(resolution.contributing_results)}")
            print(f"   Notes: {resolution.resolution_notes}")
            
            if resolution.reasoning:
                print(f"\n📝 Agent Reasoning:")
                print(f"   {resolution.reasoning}")
            
            if resolution.merge_details:
                print(f"\n   Merge Details (sample):")
                for detail in resolution.merge_details[:3]:  # Show first 3
                    print(f"      - {detail.get('parameter')}: {detail.get('merged_value')}")
        
        state.current_step = "RESOLVED"
        state.execution_log.append(f"✅ Conflicts resolved using {state.resolution_strategy} (Agent-based)")
        
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
        
        # Extract and store final config for next intent
        if state.final_configuration:
            # Extract the actual configuration parameters
            final_config = extract_final_config_from_result(state.final_configuration)
            
            # Store it in the result for easy access
            state.final_configuration['final_config'] = final_config
            
            # Add to active intents (this becomes the current config for next intent)
            add_active_intent(state.final_configuration)
            
            print(f"\n✅ Configuration added to active intents")
            print(f"\n📊 New System State (will be used for next intent):")
            print(f"   TX0: {'ON' if final_config['tx0_on'] else 'OFF'} | P={final_config['tx0_P_dBm']:.1f}dBm | Az={final_config['tx0_dAz']:.1f}° | El={final_config['tx0_dEl']:.1f}°")
            print(f"   TX1: {'ON' if final_config['tx1_on'] else 'OFF'} | P={final_config['tx1_P_dBm']:.1f}dBm | Az={final_config['tx1_dAz']:.1f}° | El={final_config['tx1_dEl']:.1f}°")
            print(f"   TX2: {'ON' if final_config['tx2_on'] else 'OFF'} | P={final_config['tx2_P_dBm']:.1f}dBm | Az={final_config['tx2_dAz']:.1f}° | El={final_config['tx2_dEl']:.1f}°")
            print(f"   TX3: {'ON' if final_config['tx3_on'] else 'OFF'} | P={final_config['tx3_P_dBm']:.1f}dBm | Az={final_config['tx3_dAz']:.1f}° | El={final_config['tx3_dEl']:.1f}°")
        
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
            
            # Extract the final state from workflow result
            if hasattr(result, 'input') and isinstance(result.input, PipelineState):
                final_state = result.input
            else:
                final_state = initial_state
            
            # Create a clean, readable summary
            summary = {
                "workflow_id": final_state.workflow_id,
                "timestamp": final_state.timestamp,
                "status": final_state.current_step,
                "intent": {
                    "natural_language": final_state.natural_language_intent,
                    "priority": final_state.parsed_intent.priority if final_state.parsed_intent else None,
                    "target_area": final_state.parsed_intent.target_area if final_state.parsed_intent else None,
                    "target_kpis": final_state.parsed_intent.target_kpis if final_state.parsed_intent else None,
                    "time_constraint": {
                        "start": final_state.parsed_intent.time_constraint_start if final_state.parsed_intent else None,
                        "end": final_state.parsed_intent.time_constraint_end if final_state.parsed_intent else None
                    }
                },
                "optimization": None,
                "conflicts": None,
                "resolution": None,
                "final_configuration": None,
                "execution_log": final_state.execution_log
            }
            
            # Add optimization results if available
            if final_state.optimization_result:
                opt = final_state.optimization_result
                if isinstance(opt, dict) and 'output' in opt:
                    summary["optimization"] = {
                        "config_id": opt['output'].get('selected_config_id'),
                        "changes": opt['output'].get('changes', []),
                        "expected_kpis": opt['output'].get('expected_kpis', {}),
                        "current_kpis": opt['output'].get('current_kpis', {}),
                        "constraints_satisfied": opt['output'].get('constraints_satisfied', False)
                    }
            
            # Add conflict information if available
            if final_state.conflict_report:
                summary["conflicts"] = {
                    "is_conflicted": final_state.conflict_report.get('is_conflicted', False),
                    "num_conflicts": final_state.conflict_report.get('num_conflicts', 0),
                    "summary": final_state.conflict_report.get('conflict_summary', ''),
                    "conflicting_ids": final_state.conflict_report.get('conflicting_result_ids', [])
                }
            
            # Add resolution information if available
            if final_state.resolution_output:
                res = final_state.resolution_output
                if isinstance(res, dict):
                    summary["resolution"] = {
                        "strategy": final_state.resolution_strategy,
                        "selected_config_id": res.get('selected_config_id'),
                        "reasoning": res.get('reasoning', '')
                    }
            
            # Add final configuration
            if final_state.final_configuration:
                final_config = final_state.final_configuration
                if isinstance(final_config, dict) and 'output' in final_config:
                    summary["final_configuration"] = {
                        "config_id": final_config['output'].get('selected_config_id'),
                        "changes": final_config['output'].get('changes', []),
                        "expected_kpis": final_config['output'].get('expected_kpis', {})
                    }
            
            # Save clean summary to file
            result_file = f"workflow_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
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
