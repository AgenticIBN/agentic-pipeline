#!/usr/bin/env python3
"""
Hybrid Workflow Pipeline (Surrogate Model + LLM)

Complete agentic workflow using hybrid optimization agent:
- Surrogate Model: Accurate KPI predictions
- LLM: Reasoning, explanations, and iterative refinement

Steps:
1. Intent Parsing (LLM)
2. Hybrid Optimization (Surrogate + LLM)
3. Conflict Detection (LLM)
4. Conflict Resolution (LLM)
5. Finalization (LLM)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

from agno.os import AgentOS
from agno.workflow import Workflow, Step
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Add agents to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
sys.path.insert(0, os.path.dirname(__file__))

from intent_parser.intent_parser_agent import IntentParse, run_intent_parser
from agents.optimization_agent_hybrid import run_hybrid_optimization
from agents.conflict_detector_agent_llm import run_conflict_detection
from agents.priority_resolution_agent_llm import run_priority_resolution
from agents.weighted_merge_agent_llm import run_weighted_merge

load_dotenv()


# ============================================================================
# ACTIVE INTENTS STORAGE
# ============================================================================

ACTIVE_INTENTS_FILE = "active_intents_workflow_hybrid.json"

def load_active_intents() -> list:
    """Load active optimization results from storage."""
    if os.path.exists(ACTIVE_INTENTS_FILE):
        with open(ACTIVE_INTENTS_FILE, 'r') as f:
            data = json.load(f)
            # Handle both old format (list) and new format (dict with all_intents)
            if isinstance(data, list):
                return data
            return data.get('all_intents', [])
    return []


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
        # Başlangıç durumu: Tüm TX'ler açık, düşük power (35 dBm), açılar merkezi (0 derece)
        print("📌 No active intents found - using default baseline config")
        return {
            'tx0_on': True, 'tx0_P_dBm': 35.0, 'tx0_dAz': 0.0, 'tx0_dEl': 0.0,
            'tx1_on': True, 'tx1_P_dBm': 35.0, 'tx1_dAz': 0.0, 'tx1_dEl': 0.0,
            'tx2_on': True, 'tx2_P_dBm': 35.0, 'tx2_dAz': 0.0, 'tx2_dEl': 0.0,
            'tx3_on': True, 'tx3_P_dBm': 35.0, 'tx3_dAz': 0.0, 'tx3_dEl': 0.0
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
    
    # Check if final_config is directly available
    if 'final_config' in output:
        return output['final_config']
    
    # Get changes list
    changes = output.get('changes', [])
    
    # Reconstruct from changes
    input_data = result.get('input', {})
    base_config = input_data.get('current_config', {})
    
    if not base_config:
        # Use default
        base_config = {
            'tx0_on': True, 'tx0_P_dBm': 35.0, 'tx0_dAz': 0.0, 'tx0_dEl': 0.0,
            'tx1_on': True, 'tx1_P_dBm': 35.0, 'tx1_dAz': 0.0, 'tx1_dEl': 0.0,
            'tx2_on': True, 'tx2_P_dBm': 35.0, 'tx2_dAz': 0.0, 'tx2_dEl': 0.0,
            'tx3_on': True, 'tx3_P_dBm': 35.0, 'tx3_dAz': 0.0, 'tx3_dEl': 0.0
        }
    
    # Apply changes to base config
    final_config = dict(base_config)
    
    for change in changes:
        # Support both 'param' and 'parameter' keys
        param = change.get('parameter') or change.get('param')
        change_val = change.get('change')
        before = change.get('before')
        
        if param and change_val is not None:
            # For boolean params (tx_on), change is the new value
            if param.endswith('_on'):
                if isinstance(change_val, str):
                    final_config[param] = change_val.lower() == 'true'
                else:
                    final_config[param] = bool(change_val)
            # For numeric params, change is delta
            elif before is not None:
                final_config[param] = float(before) + float(change_val)
            else:
                # No before value, treat change as absolute
                final_config[param] = float(change_val)
    
    return final_config

def save_active_intents(intents: list) -> None:
    """Save active optimization results to storage."""
    output = {
        "workflow_id": f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "resolution_strategy": "HYBRID",
        "all_intents": intents,
    }
    with open(ACTIVE_INTENTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)

def add_active_intent(result: dict) -> None:
    """Add a new optimization result to active intents."""
    active = load_active_intents()
    active.append(result)
    save_active_intents(active)

def remove_rejected_intents(rejected_ids: list) -> None:
    """Remove rejected intents from active intents."""
    active = load_active_intents()
    active = [intent for intent in active if intent.get('result_id') not in rejected_ids]
    save_active_intents(active)


# ============================================================================
# WORKFLOW DEFINITION
# ============================================================================

def create_hybrid_workflow() -> Workflow:
    """
    Create hybrid workflow with 5 steps.
    """
    
    # Step 1: Intent Parsing
    step1 = Step(
        name="intent_parse",
        executor=run_intent_parser,
    )
    
    # Step 2: Hybrid Optimization (Surrogate + LLM)
    step2 = Step(
        name="hybrid_optimization",
        executor=run_hybrid_optimization,
    )
    
    # Step 3: Conflict Detection
    step3 = Step(
        name="conflict_detection",
        executor=run_conflict_detection,
    )
    
    # Step 4: Conflict Resolution (strategy-based)
    def resolution_router(conflict_report: Dict, new_result: Dict, active_results: List[Dict], strategy: str) -> Dict:
        """Route to appropriate resolution strategy."""
        if strategy == "PRIORITY":
            return run_priority_resolution(conflict_report, new_result, active_results)
        elif strategy == "WEIGHTED_MERGE":
            return run_weighted_merge(conflict_report, new_result, active_results)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    step4 = Step(
        name="conflict_resolution",
        executor=resolution_router,
    )
    
    # Step 5: Finalization
    def finalize_workflow(
        resolution_result: Dict,
        workflow_id: str,
        output_file: str = "active_intents_workflow_hybrid.json",
    ) -> Dict:
        """Finalize workflow and save results."""
        print("\n" + "="*70)
        print("📝 STEP 5: FINALIZATION")
        print("="*70)
        
        # Detect strategy and extract appropriate fields
        strategy = None
        contributing = []  # Initialize for both strategies
        
        if hasattr(resolution_result, 'resolution_strategy'):
            strategy = resolution_result.resolution_strategy
        else:
            strategy = resolution_result.get('resolution_strategy', 'UNKNOWN')
        
        # Handle PRIORITY strategy
        if strategy == "PRIORITY":
            if hasattr(resolution_result, 'winning_result_id'):
                # Pydantic model
                result_id = resolution_result.winning_result_id
                priority = resolution_result.winning_priority
                full_result = resolution_result.winning_config
                rejected_ids = resolution_result.rejected_result_ids
            else:
                # Dict
                result_id = resolution_result.get('winning_result_id')
                priority = resolution_result.get('winning_priority')
                full_result = resolution_result.get('winning_config', {})
                rejected_ids = resolution_result.get('rejected_result_ids', [])
            
            print(f"🎯 Resolution Strategy: PRIORITY")
            print(f"🏆 Winner: {result_id} (Priority: {priority})")
            
            # Remove rejected intents
            if rejected_ids:
                print(f"🗑️  Removing {len(rejected_ids)} rejected intent(s)...")
                remove_rejected_intents(rejected_ids)
            
            # Add winning intent
            if full_result:
                print(f"➕ Adding winning intent: {result_id}")
                add_active_intent(full_result)
        
        # Handle WEIGHTED_MERGE strategy
        elif strategy == "WEIGHTED_MERGE":
            if hasattr(resolution_result, 'merged_result_id'):
                # Pydantic model
                result_id = resolution_result.merged_result_id
                full_result = resolution_result.merged_config
                contributing = resolution_result.contributing_results
            else:
                # Dict
                result_id = resolution_result.get('merged_result_id')
                full_result = resolution_result.get('merged_config', {})
                contributing = resolution_result.get('contributing_results', [])
            
            priority = "MERGED"  # Merged results have combined priority
            rejected_ids = []  # No rejections in merge strategy
            
            print(f"🎯 Resolution Strategy: WEIGHTED_MERGE")
            print(f"🔀 Merged Result: {result_id}")
            print(f"📊 Contributing Intents: {len(contributing)}")
            for contrib in contributing:
                if isinstance(contrib, dict):
                    print(f"   • {contrib.get('id')}: {contrib.get('priority')} (weight: {contrib.get('weight', 0):.2f})")
                else:
                    print(f"   • {contrib.id}: {contrib.priority} (weight: {contrib.weight:.2f})")
            
            # Remove ALL active intents that were merged
            active_intents = load_active_intents()
            merged_ids = [c.get('id') if isinstance(c, dict) else c.id for c in contributing]
            if merged_ids:
                print(f"🔄 Replacing {len(merged_ids)} merged intent(s) with single merged result...")
                remove_rejected_intents(merged_ids)
            
            # Add merged result as new active intent
            if full_result:
                print(f"➕ Adding merged intent: {result_id}")
                add_active_intent(full_result)
        
        else:
            raise ValueError(f"Unknown resolution strategy: {strategy}")
        
        # Load current active intents
        all_intents = load_active_intents()
        
        # Extract final config
        final_config = full_result.get('output', {}).get('final_config', {}) if full_result else {}
        
        # Create final output
        final_output = {
            "workflow_id": workflow_id,
            "timestamp": datetime.now().isoformat(),
            "resolution_strategy": strategy,
            "selected_intent": result_id,
            "selected_priority": priority,
            "selected_config": final_config,
            "all_intents": all_intents,
            "rejected_intents": rejected_ids if strategy == "PRIORITY" else [],
            "merged_intents": [c.get('id') if isinstance(c, dict) else c.id for c in contributing] if strategy == "WEIGHTED_MERGE" else [],
            "execution_log": [
                "✅ Intent parsed by LLM Agent",
                "✅ Optimization by Hybrid Agent (Surrogate Model + LLM)",
                "✅ Conflicts detected by LLM Agent",
                f"✅ Resolution by {strategy} strategy",
                f"✅ Active intents updated ({len(all_intents)} total)",
                "✅ Workflow finalized and saved",
            ],
        }
        
        # Save to file
        save_active_intents(all_intents)
        
        print(f"✅ Active intents updated:")
        print(f"   Total active: {len(all_intents)}")
        print(f"   Selected: {result_id}")
        if strategy == "PRIORITY":
            print(f"   Rejected: {len(rejected_ids)}")
        elif strategy == "WEIGHTED_MERGE":
            print(f"   Merged from: {len(contributing)} intent(s)")
        print(f"   Strategy: {strategy}")
        
        return final_output
    
    step5 = Step(
        name="finalization",
        executor=finalize_workflow,
    )
    
    # Create workflow with step dependencies
    workflow = Workflow(
        name="HybridNetworkOptimization",
        steps=[step1, step2, step3, step4, step5],
    )
    
    return workflow


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run hybrid workflow (Surrogate Model + LLM) for network optimization"
    )
    parser.add_argument(
        "--intent",
        type=str,
        required=True,
        help="Intent text to process"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="PRIORITY",
        choices=["PRIORITY", "WEIGHTED_MERGE"],
        help="Conflict resolution strategy"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="active_intents_workflow_hybrid.json",
        help="Output file for workflow results"
    )
    parser.add_argument(
        "--surrogate-model",
        type=str,
        default=None,
        help="Path to surrogate model (default: ./models/surrogate.joblib)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 HYBRID WORKFLOW PIPELINE (Surrogate Model + LLM)")
    print("="*70)
    print(f"Intent: {args.intent}")
    print(f"Strategy: {args.strategy}")
    print(f"Output: {args.output}")
    
    # Generate workflow ID
    workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create workflow
    workflow = create_hybrid_workflow()
    
    # ========================================================================
    # STEP 1: INTENT PARSING
    # ========================================================================
    print("\n" + "="*70)
    print("📝 STEP 1: INTENT PARSING")
    print("="*70)
    
    intent_text = args.intent
    parsed_intent_output = run_intent_parser(intent_text)
    
    # Parse output (handle RunOutput or dict)
    if hasattr(parsed_intent_output, 'content'):
        intent_content = parsed_intent_output.content
    elif hasattr(parsed_intent_output, 'messages') and parsed_intent_output.messages:
        intent_content = parsed_intent_output.messages[0].content
    else:
        intent_content = parsed_intent_output
    
    # Clean and parse JSON
    if isinstance(intent_content, str):
        intent_content = intent_content.replace("```json", "").replace("```", "").strip()
        parsed_intent = json.loads(intent_content)
    else:
        parsed_intent = intent_content if isinstance(intent_content, dict) else intent_content.model_dump()
    
    print(f"✅ Intent parsed:")
    print(f"   Target Area: {parsed_intent.get('target_area')}")
    print(f"   Priority: {parsed_intent.get('priority')}")
    print(f"   Target KPIs: {', '.join(parsed_intent.get('target_kpis', []))}")
    
    # Add original intent text to parsed intent for optimization agent
    parsed_intent['intent_text'] = intent_text
    
    # ========================================================================
    # STEP 2: HYBRID OPTIMIZATION (Surrogate + LLM)
    # ========================================================================
    print("\n" + "="*70)
    print("🔬 STEP 2: HYBRID OPTIMIZATION (Surrogate Model + LLM)")
    print("="*70)
    
    # Get current system configuration (from last finalized intent or default)
    current_config = get_current_system_config()
    parsed_intent['current_config'] = current_config
    
    # Display current system state
    print(f"\n📊 Current System State:")
    print(f"   Configuration:")
    print(f"      TX0: {'ON' if current_config['tx0_on'] else 'OFF'} | P={current_config['tx0_P_dBm']:.1f}dBm | Az={current_config['tx0_dAz']:.1f}° | El={current_config['tx0_dEl']:.1f}°")
    print(f"      TX1: {'ON' if current_config['tx1_on'] else 'OFF'} | P={current_config['tx1_P_dBm']:.1f}dBm | Az={current_config['tx1_dAz']:.1f}° | El={current_config['tx1_dEl']:.1f}°")
    print(f"      TX2: {'ON' if current_config['tx2_on'] else 'OFF'} | P={current_config['tx2_P_dBm']:.1f}dBm | Az={current_config['tx2_dAz']:.1f}° | El={current_config['tx2_dEl']:.1f}°")
    print(f"      TX3: {'ON' if current_config['tx3_on'] else 'OFF'} | P={current_config['tx3_P_dBm']:.1f}dBm | Az={current_config['tx3_dAz']:.1f}° | El={current_config['tx3_dEl']:.1f}°")
    
    # Load active intents before optimization to build upon them
    active_intents_for_opt = load_active_intents()
    
    optimization_result = run_hybrid_optimization(
        parsed_intent,
        surrogate_model_path=args.surrogate_model,
        active_intents=active_intents_for_opt,
    )
    
    print(f"✅ Optimization complete:")
    print(f"   Result ID: {optimization_result.get('result_id')}")
    print(f"   Iterations: {optimization_result.get('iterations')}")
    print(f"   Passed: {optimization_result.get('passed')}")
    print(f"   Expected KPIs:")
    print(f"     RX_POWER={optimization_result['output']['expected_kpis']['RX_POWER']:.2f} dBm, ", end="")
    print(f"SINR={optimization_result['output']['expected_kpis']['SINR']:.2f} dB, ", end="")
    print(f"THROUGHPUT={optimization_result['output']['expected_kpis']['THROUGHPUT_5P']:.2f} Mbps")
    
    # ========================================================================
    # STEP 3: CONFLICT DETECTION
    # ========================================================================
    print("\n" + "="*70)
    print("🔍 STEP 3: CONFLICT DETECTION")
    print("="*70)
    
    # Load active intents for conflict detection
    active_intents = load_active_intents()
    print(f"📂 Loaded {len(active_intents)} active intent(s)")
    
    conflict_result = run_conflict_detection(optimization_result, active_intents)
    
    # Parse conflict output
    if hasattr(conflict_result, 'content'):
        conflict_content = conflict_result.content
    elif hasattr(conflict_result, 'messages') and conflict_result.messages:
        conflict_content = conflict_result.messages[0].content
    else:
        conflict_content = conflict_result
    
    if isinstance(conflict_content, str):
        conflict_content = conflict_content.replace("```json", "").replace("```", "").strip()
        conflicts = json.loads(conflict_content)
    elif hasattr(conflict_content, 'model_dump'):
        conflicts = conflict_content.model_dump()
    else:
        conflicts = conflict_content if isinstance(conflict_content, dict) else {}
    
    print(f"✅ Conflict detection complete:")
    if isinstance(conflicts, dict):
        is_conflicted = conflicts.get('is_conflicted', False)
        num_conflicts = conflicts.get('num_conflicts', 0)
        print(f"   Conflicts Found: {is_conflicted}")
        print(f"   Number of Conflicts: {num_conflicts}")
    else:
        print(f"   Result: {conflicts}")
    
    # ========================================================================
    # STEP 4: CONFLICT RESOLUTION
    # ========================================================================
    print("\n" + "="*70)
    print("⚖️ STEP 4: CONFLICT RESOLUTION")
    print("="*70)
    
    # Call resolution with correct parameters
    resolution_result = workflow.steps[3].executor(
        conflicts,           # conflict_report
        optimization_result, # new_result
        active_intents,      # active_results
        args.strategy        # strategy
    )
    
    # Parse resolution output
    if hasattr(resolution_result, 'content'):
        resolution_content = resolution_result.content
    elif hasattr(resolution_result, 'messages') and resolution_result.messages:
        resolution_content = resolution_result.messages[0].content
    else:
        resolution_content = resolution_result
    
    if isinstance(resolution_content, str):
        resolution_content = resolution_content.replace("```json", "").replace("```", "").strip()
        resolution = json.loads(resolution_content)
    else:
        resolution = resolution_content if isinstance(resolution_content, dict) else resolution_content.model_dump()
    
    print(f"✅ Resolution complete:")
    print(f"   Strategy: {resolution.get('resolution_strategy', 'UNKNOWN')}")
    print(f"   Selected Intent: {resolution.get('winning_result_id', 'None')}")
    
    # ========================================================================
    # STEP 5: FINALIZATION
    # ========================================================================
    final_output = workflow.steps[4].executor(
        resolution,
        workflow_id,
        args.output,
    )
    
    # Save full workflow result
    workflow_output_file = f"workflow_result_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    workflow_output_path = os.path.join(os.path.dirname(__file__), workflow_output_file)
    
    full_result = {
        "workflow_id": workflow_id,
        "workflow_type": "HYBRID (Surrogate Model + LLM)",
        "timestamp": datetime.now().isoformat(),
        "input": {
            "intent_text": intent_text,
            "strategy": args.strategy,
        },
        "step1_intent_parse": parsed_intent,
        "step2_hybrid_optimization": optimization_result,
        "step3_conflict_detection": conflicts,
        "step4_resolution": resolution,
        "step5_final_output": final_output,
    }
    
    with open(workflow_output_path, 'w') as f:
        json.dump(full_result, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ WORKFLOW COMPLETE!")
    print("="*70)
    print(f"📊 Full result saved to: {workflow_output_path}")
    print(f"📝 Active intents saved to: {args.output}")
    print()
    print("🎯 Key Results:")
    print(f"   - Workflow ID: {workflow_id}")
    print(f"   - Optimization Method: Hybrid (Surrogate Model + LLM)")
    print(f"   - Iterations: {optimization_result.get('iterations')}")
    print(f"   - Target Met: {optimization_result.get('passed')}")
    print(f"   - Final KPIs:")
    print(f"     * RX_POWER: {optimization_result['output']['expected_kpis']['RX_POWER']:.2f} dBm")
    print(f"     * SINR: {optimization_result['output']['expected_kpis']['SINR']:.2f} dB")
    print(f"     * COVERAGE: {optimization_result['output']['expected_kpis']['COVERAGE']*100:.1f}%")
    print(f"     * THROUGHPUT: {optimization_result['output']['expected_kpis']['THROUGHPUT_5P']:.2f} Mbps")
    print(f"     * LOAD_IMBALANCE: {optimization_result['output']['expected_kpis']['LOAD_IMBALANCE']:.3f}")
    print(f"     * ENERGY: {optimization_result['output']['expected_kpis']['ENERGY_WATT']:.2f} W")
    print()
    print("💡 This workflow used:")
    print("   ✓ Surrogate Model for accurate KPI predictions (trained on 1M+ simulations)")
    print("   ✓ LLM for reasoning, explanations, and iterative refinement")
    print("   ✓ Best of both worlds: Accuracy + Explainability")


if __name__ == "__main__":
    main()
