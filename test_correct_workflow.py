#!/usr/bin/env python3
"""
Test CORRECT Workflow: Active Intent + New Intent → Conflict Detection → Resolution
"""
import json
from conflict_resolution_orchestrator import process_new_intent
from optimization_agent import IntentParse, OptimizationPlan, ParamChange, KpiSnapshot

def simulate_active_intent(area: str, priority: str, changes: list) -> dict:
    """
    Simulate an already optimized and running intent
    """
    intent = IntentParse(
        target_area=area,
        target_kpis=["RX_POWER"],
        priority=priority,
        confidence=0.85
    )
    
    plan = OptimizationPlan(
        selected_config_id=12345,
        current_config_id=0,
        changes=[ParamChange(**c) for c in changes],
        expected_kpis=KpiSnapshot(
            RX_POWER=-82.0,
            SINR=8.0,
            THROUGHPUT_5P=20.0,
            LOAD_IMBALANCE=5.0,
            RX_COVERAGE_RATIO=0.75
        ),
        constraints_satisfied=True
    )
    
    return {
        "intent": intent.model_dump(),
        "plan": plan.model_dump()
    }

# ==========================================
# TEST SCENARIOS - CORRECT WORKFLOW
# ==========================================

test_scenarios = [
    {
        "name": "Scenario 1: New CRITICAL Intent vs Active LOW Intent",
        "description": "System has LOW priority intent running. CRITICAL intent arrives.",
        "active_intents": [
            simulate_active_intent(
                area="Kadıköy",
                priority="LOW",
                changes=[
                    {"param": "tx0_P_dBm", "before": 40.0, "after": 3.0, "unit": "dBm"},
                    {"param": "tx1_dAz", "before": 0.0, "after": -5.0, "unit": "deg"}
                ]
            )
        ],
        "new_intent": {
            "target_area": "Kadıköy",
            "target_kpis": ["RX_POWER", "SINR"],
            "kpi_thresholds": [
                {"kpi": "RX_POWER", "op": "GTE", "value": -85.0, "unit": "dBm"},
                {"kpi": "SINR", "op": "GTE", "value": 10.0, "unit": "dB"}
            ],
            "priority": "CRITICAL",
            "confidence": 0.95,
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
        "expected": "CRITICAL should dominate. Conflict resolution should favor new intent."
    },
    {
        "name": "Scenario 2: New HIGH Intent vs Active HIGH Intent (Different Confidence)",
        "description": "System has HIGH priority (conf=0.7) running. New HIGH (conf=0.95) arrives.",
        "active_intents": [
            simulate_active_intent(
                area="Beşiktaş",
                priority="HIGH",
                changes=[
                    {"param": "tx1_P_dBm", "before": 43.0, "after": 2.0, "unit": "dBm"},
                    {"param": "tx2_dEl", "before": -3.0, "after": 1.0, "unit": "deg"}
                ]
            )
        ],
        "new_intent": {
            "target_area": "Beşiktaş",
            "target_kpis": ["THROUGHPUT_5P"],
            "kpi_thresholds": [
                {"kpi": "THROUGHPUT_5P", "op": "GTE", "value": 25.0, "unit": "Mbps"}
            ],
            "priority": "HIGH",
            "confidence": 0.95,  # Higher confidence than active
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                "tx0_on": True,
                "tx0_P_dBm": 44.0,
                "tx0_dAz": -10.0,
                "tx0_dEl": -3.0,
                "tx1_on": True,
                "tx1_P_dBm": 44.0,
                "tx1_dAz": 10.0,
                "tx1_dEl": -3.0,
                "tx2_on": True,
                "tx2_P_dBm": 44.0,
                "tx2_dAz": 10.0,
                "tx2_dEl": -3.0,
                "tx3_on": True,
                "tx3_P_dBm": 44.0,
                "tx3_dAz": -10.0,
                "tx3_dEl": -3.0
            }
        },
        "expected": "Higher confidence new intent should get more weight in merge."
    },
    {
        "name": "Scenario 3: New Intent - No Active Intents",
        "description": "System is empty. First intent arrives.",
        "active_intents": [],  # Sistem boş
        "new_intent": {
            "target_area": "Şişli",
            "target_kpis": ["RX_POWER"],
            "kpi_thresholds": [
                {"kpi": "RX_POWER", "op": "GTE", "value": -87.0, "unit": "dBm"}
            ],
            "priority": "MEDIUM",
            "confidence": 0.8,
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                "tx0_on": True,
                "tx0_P_dBm": 41.0,
                "tx0_dAz": -12.0,
                "tx0_dEl": -4.5,
                "tx1_on": True,
                "tx1_P_dBm": 41.0,
                "tx1_dAz": 12.0,
                "tx1_dEl": -4.5,
                "tx2_on": True,
                "tx2_P_dBm": 41.0,
                "tx2_dAz": 12.0,
                "tx2_dEl": -4.5,
                "tx3_on": True,
                "tx3_P_dBm": 41.0,
                "tx3_dAz": -12.0,
                "tx3_dEl": -4.5
            }
        },
        "expected": "No conflict. New intent should be executed directly."
    },
    {
        "name": "Scenario 4: New Intent - Different Area (No Conflict)",
        "description": "Active intent in Kadıköy. New intent in Ümraniye (different area).",
        "active_intents": [
            simulate_active_intent(
                area="Kadıköy",
                priority="HIGH",
                changes=[
                    {"param": "tx0_P_dBm", "before": 40.0, "after": 4.0, "unit": "dBm"}
                ]
            )
        ],
        "new_intent": {
            "target_area": "Ümraniye",  # Farklı alan
            "target_kpis": ["SINR"],
            "kpi_thresholds": [
                {"kpi": "SINR", "op": "GTE", "value": 8.0, "unit": "dB"}
            ],
            "priority": "MEDIUM",
            "confidence": 0.85,
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
        },
        "expected": "No/Low conflict. Both should run in parallel."
    }
]

def run_correct_workflow_tests():
    print("="*100)
    print("CORRECT WORKFLOW TEST: Active Intent + New Intent → Conflict Detection → Resolution")
    print("="*100)
    print("\nThis tests the REAL system workflow:")
    print("  1. System has active intent(s) running (already optimized)")
    print("  2. NEW intent arrives")
    print("  3. Optimize ONLY the new intent")
    print("  4. Conflict Detection (new vs active)")
    print("  5. If conflict: Team resolves it")
    print("  6. Final configuration\n")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*100}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*100}")
        print(f"Description: {scenario['description']}")
        print(f"Active Intents in System: {len(scenario['active_intents'])}")
        print(f"Expected Outcome: {scenario['expected']}")
        print()
        
        # Create new intent
        new_intent = IntentParse(**scenario['new_intent'])
        
        # Run the CORRECT workflow
        result_json = process_new_intent(
            new_intent_json=new_intent.model_dump_json(),
            active_intents_json=json.dumps(scenario['active_intents'])
        )
        
        result = json.loads(result_json)
        
        # Print results
        print(f"\n{'='*50}")
        print("ORCHESTRATION RESULTS")
        print(f"{'='*50}")
        print(f"Orchestration ID: {result['orchestration_id']}")
        print(f"Total Intents: {result['total_intents']} (Active: {len(scenario['active_intents'])}, New: 1)")
        print(f"New Intent Optimization: {'✅ Success' if result['successful_optimizations'] > 0 else '❌ Failed'}")
        
        print(f"\nExecution Strategy: {result['execution_strategy']}")
        print(f"Conflict Detected: {result['conflict_detected']}")
        print(f"Resolution Applied: {result['resolution_applied']}")
        
        if result['conflict_detected'] and result['conflict_report']:
            conflict_report = result['conflict_report']['conflict_report']
            print(f"\nConflict Summary: {conflict_report['conflict_summary']}")
        
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
                for change in merged['changes'][:5]:
                    print(f"  {change['param']}: {change['before']} → {change['after']} {change['unit'] or ''}")
                
            elif result['execution_strategy'] == "PARALLEL":
                configs = result['final_configuration'].get('configurations', [])
                print(f"Parallel Execution: {len(configs)} independent configurations")
                for config in configs:
                    plan = config.get('plan', {})
                    print(f"  - {config['intent_id']}: {len(plan.get('changes', []))} changes")
            
            elif result['execution_strategy'] == "SINGLE":
                print("Single intent execution (no active intents)")
        
        print(f"\nNotes:")
        for note in result['notes']:
            print(f"  • {note}")
        
        # Save results
        import os
        output_file = f"results/correct_workflow_scenario_{i}.json"
        os.makedirs("results", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "scenario": scenario,
                "result": result
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"\n{'='*100}\n")

if __name__ == "__main__":
    run_correct_workflow_tests()
