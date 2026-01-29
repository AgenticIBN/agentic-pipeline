#!/usr/bin/env python3
"""
Test Conflict Resolution Orchestrator (Team Approach)
Tests the full pipeline: Multiple Intents → Optimization Agents → Conflict Detection → Resolution
"""
import json
from datetime import datetime
from conflict_resolution_orchestrator import (
    run_multi_intent_optimization, 
    ConflictResolutionOrchestrator,
    MultiIntentInput,
    save_orchestration_result
)
from optimization_agent import IntentParse

# ==========================================
# TEST SCENARIOS
# ==========================================

test_scenarios = [
    {
        "name": "Scenario 1: CRITICAL vs LOW Priority Conflict",
        "description": "Two intents targeting same area with opposite parameter changes",
        "intents": [
            {
                "target_area": "Kadıköy",
                "target_kpis": ["RX_POWER", "SINR"],
                "kpi_thresholds": [
                    {"kpi": "RX_POWER", "op": "GTE", "value": -85.0, "unit": "dBm"},
                    {"kpi": "SINR", "op": "GTE", "value": 10.0, "unit": "dB"}
                ],
                "priority": "CRITICAL",
                "confidence": 0.9,
                "user_set_id": 0,
                "k_users": 800,
                "current_config": {
                    "tx0_on": True,
                    "tx0_P_dBm": 40.0,
                    "tx0_dAz": -20.0,
                    "tx0_dEl": -5.0,
                    "tx1_on": True,
                    "tx1_P_dBm": 40.0,
                    "tx1_dAz": 20.0,
                    "tx1_dEl": -5.0,
                    "tx2_on": True,
                    "tx2_P_dBm": 40.0,
                    "tx2_dAz": 20.0,
                    "tx2_dEl": -5.0,
                    "tx3_on": True,
                    "tx3_P_dBm": 40.0,
                    "tx3_dAz": -20.0,
                    "tx3_dEl": -5.0
                }
            },
            {
                "target_area": "Kadıköy",
                "target_kpis": ["THROUGHPUT_5P"],
                "kpi_thresholds": [
                    {"kpi": "THROUGHPUT_5P", "op": "GTE", "value": 20.0, "unit": "Mbps"}
                ],
                "priority": "LOW",
                "confidence": 0.7,
                "user_set_id": 0,
                "k_users": 800,
                "current_config": {
                    "tx0_on": True,
                    "tx0_P_dBm": 45.0,
                    "tx0_dAz": 0.0,
                    "tx0_dEl": -3.0,
                    "tx1_on": True,
                    "tx1_P_dBm": 45.0,
                    "tx1_dAz": 0.0,
                    "tx1_dEl": -3.0,
                    "tx2_on": True,
                    "tx2_P_dBm": 45.0,
                    "tx2_dAz": 0.0,
                    "tx2_dEl": -3.0,
                    "tx3_on": True,
                    "tx3_P_dBm": 45.0,
                    "tx3_dAz": 0.0,
                    "tx3_dEl": -3.0
                }
            }
        ],
        "expected_outcome": "CRITICAL priority should dominate. Merged config should favor first intent."
    },
    {
        "name": "Scenario 2: Same Priority, Different Confidence",
        "description": "Three intents with HIGH priority but different confidence levels",
        "intents": [
            {
                "target_area": "Beşiktaş",
                "target_kpis": ["RX_POWER"],
                "kpi_thresholds": [
                    {"kpi": "RX_POWER", "op": "GTE", "value": -88.0, "unit": "dBm"}
                ],
                "priority": "HIGH",
                "confidence": 0.95,  # Highest confidence
                "user_set_id": 0,
                "k_users": 800,
                "current_config": {
                    "tx0_on": True,
                    "tx0_P_dBm": 42.0,
                    "tx0_dAz": -15.0,
                    "tx0_dEl": -4.0,
                    "tx1_on": True,
                    "tx1_P_dBm": 42.0,
                    "tx1_dAz": 15.0,
                    "tx1_dEl": -4.0,
                    "tx2_on": True,
                    "tx2_P_dBm": 42.0,
                    "tx2_dAz": 15.0,
                    "tx2_dEl": -4.0,
                    "tx3_on": True,
                    "tx3_P_dBm": 42.0,
                    "tx3_dAz": -15.0,
                    "tx3_dEl": -4.0
                }
            },
            {
                "target_area": "Beşiktaş",
                "target_kpis": ["SINR"],
                "kpi_thresholds": [
                    {"kpi": "SINR", "op": "GTE", "value": 8.0, "unit": "dB"}
                ],
                "priority": "HIGH",
                "confidence": 0.75,  # Medium confidence
                "user_set_id": 0,
                "k_users": 800,
                "current_config": {
                    "tx0_on": True,
                    "tx0_P_dBm": 43.0,
                    "tx0_dAz": -10.0,
                    "tx0_dEl": -3.0,
                    "tx1_on": True,
                    "tx1_P_dBm": 43.0,
                    "tx1_dAz": 10.0,
                    "tx1_dEl": -3.0,
                    "tx2_on": True,
                    "tx2_P_dBm": 43.0,
                    "tx2_dAz": 10.0,
                    "tx2_dEl": -3.0,
                    "tx3_on": True,
                    "tx3_P_dBm": 43.0,
                    "tx3_dAz": -10.0,
                    "tx3_dEl": -3.0
                }
            },
            {
                "target_area": "Beşiktaş",
                "target_kpis": ["THROUGHPUT_5P"],
                "kpi_thresholds": [
                    {"kpi": "THROUGHPUT_5P", "op": "GTE", "value": 22.0, "unit": "Mbps"}
                ],
                "priority": "HIGH",
                "confidence": 0.65,  # Lowest confidence
                "user_set_id": 0,
                "k_users": 800,
                "current_config": {
                    "tx0_on": True,
                    "tx0_P_dBm": 44.0,
                    "tx0_dAz": -5.0,
                    "tx0_dEl": -2.0,
                    "tx1_on": True,
                    "tx1_P_dBm": 44.0,
                    "tx1_dAz": 5.0,
                    "tx1_dEl": -2.0,
                    "tx2_on": True,
                    "tx2_P_dBm": 44.0,
                    "tx2_dAz": 5.0,
                    "tx2_dEl": -2.0,
                    "tx3_on": True,
                    "tx3_P_dBm": 44.0,
                    "tx3_dAz": -5.0,
                    "tx3_dEl": -2.0
                }
            }
        ],
        "expected_outcome": "Highest confidence (0.95) intent should get more weight in merged config."
    },
    {
        "name": "Scenario 3: No Conflict - Different Areas",
        "description": "Two intents targeting completely different areas",
        "intents": [
            {
                "target_area": "Kadıköy",
                "target_kpis": ["RX_POWER"],
                "kpi_thresholds": [
                    {"kpi": "RX_POWER", "op": "GTE", "value": -87.0, "unit": "dBm"}
                ],
                "priority": "HIGH",
                "confidence": 0.85,
                "user_set_id": 0,
                "k_users": 800,
                "current_config": {
                    "tx0_on": True,
                    "tx0_P_dBm": 41.0,
                    "tx0_dAz": -18.0,
                    "tx0_dEl": -4.5,
                    "tx1_on": True,
                    "tx1_P_dBm": 41.0,
                    "tx1_dAz": 18.0,
                    "tx1_dEl": -4.5,
                    "tx2_on": True,
                    "tx2_P_dBm": 41.0,
                    "tx2_dAz": 18.0,
                    "tx2_dEl": -4.5,
                    "tx3_on": True,
                    "tx3_P_dBm": 41.0,
                    "tx3_dAz": -18.0,
                    "tx3_dEl": -4.5
                }
            },
            {
                "target_area": "Ümraniye",
                "target_kpis": ["THROUGHPUT_5P"],
                "kpi_thresholds": [
                    {"kpi": "THROUGHPUT_5P", "op": "GTE", "value": 25.0, "unit": "Mbps"}
                ],
                "priority": "MEDIUM",
                "confidence": 0.8,
                "user_set_id": 0,
                "k_users": 800,
                "current_config": {
                    "tx0_on": True,
                    "tx0_P_dBm": 43.0,
                    "tx0_dAz": 0.0,
                    "tx0_dEl": -3.0,
                    "tx1_on": True,
                    "tx1_P_dBm": 43.0,
                    "tx1_dAz": 0.0,
                    "tx1_dEl": -3.0,
                    "tx2_on": True,
                    "tx2_P_dBm": 43.0,
                    "tx2_dAz": 0.0,
                    "tx2_dEl": -3.0,
                    "tx3_on": True,
                    "tx3_P_dBm": 43.0,
                    "tx3_dAz": 0.0,
                    "tx3_dEl": -3.0
                }
            }
        ],
        "expected_outcome": "No conflicts. Both configurations should execute in parallel."
    }
]

def run_orchestrator_tests():
    print("="*100)
    print("CONFLICT RESOLUTION ORCHESTRATOR - TEAM APPROACH TEST SUITE")
    print("="*100)
    print("\nThis tests the full pipeline:")
    print("  1. Multiple Intents → Optimization Agents (Team Members)")
    print("  2. Conflict Detection")
    print("  3. Conflict Resolution (Meta-Agent)")
    print("  4. Final Merged Configuration\n")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*100}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*100}")
        print(f"Description: {scenario['description']}")
        print(f"Number of Intents: {len(scenario['intents'])}")
        print(f"Expected Outcome: {scenario['expected_outcome']}")
        print()
        
        # Create intents
        intents = [IntentParse(**intent_data) for intent_data in scenario['intents']]
        
        # Create multi-intent input
        multi_input = MultiIntentInput(intents=intents)
        
        # Run orchestration
        result_json = run_multi_intent_optimization(multi_input.model_dump_json())
        result = json.loads(result_json)
        
        # Print results
        print(f"\n{'='*50}")
        print("ORCHESTRATION RESULTS")
        print(f"{'='*50}")
        print(f"Orchestration ID: {result['orchestration_id']}")
        print(f"Total Intents: {result['total_intents']}")
        print(f"Successful Optimizations: {result['successful_optimizations']}")
        print(f"Failed Optimizations: {result['failed_optimizations']}")
        
        print(f"\nExecution Strategy: {result['execution_strategy']}")
        print(f"Conflict Detected: {result['conflict_detected']}")
        print(f"Resolution Applied: {result['resolution_applied']}")
        
        if result['conflict_detected'] and result['conflict_report']:
            conflict_report = result['conflict_report']['conflict_report']
            print(f"\nConflict Summary: {conflict_report['conflict_summary']}")
            print(f"Number of Conflicts: {len(conflict_report['details'])}")
        
        if result['final_configuration']:
            print(f"\n{'='*50}")
            print("FINAL CONFIGURATION")
            print(f"{'='*50}")
            
            if result['execution_strategy'] == "MERGED":
                merged = result['final_configuration']
                print(f"Config ID: {merged['merged_config_id']}")
                print(f"Resolution Strategy: {merged['resolution_strategy']}")
                print(f"Contributing Intents: {len(merged['contributing_intents'])}")
                
                print("\nPriority Weights:")
                for agent_id, weight in merged['priority_weights'].items():
                    print(f"  {agent_id}: {weight:.3f}")
                
                print(f"\nParameter Changes: {len(merged['changes'])}")
                for change in merged['changes'][:8]:  # Show first 8
                    print(f"  {change['param']}: {change['before']} → {change['after']} {change['unit'] or ''}")
                
                print(f"\nExpected KPIs:")
                for kpi, value in merged['expected_kpis'].items():
                    if value is not None:
                        print(f"  {kpi}: {value:.2f}")
                        
            elif result['execution_strategy'] == "PARALLEL":
                configs = result['final_configuration'].get('configurations', [])
                print(f"Parallel Execution: {len(configs)} independent configurations")
                for config in configs:
                    print(f"  - {config['intent_id']}: {len(config['plan']['changes'])} changes")
        
        print(f"\nNotes:")
        for note in result['notes']:
            print(f"  • {note}")
        
        # Save results
        output_file = f"results/orchestration_scenario_{i}.json"
        os.makedirs("results", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "scenario": scenario,
                "orchestration_result": result
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"\n{'='*100}\n")

if __name__ == "__main__":
    import os
    run_orchestrator_tests()
