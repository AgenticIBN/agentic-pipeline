#!/usr/bin/env python3
"""
Test Conflict Resolution Agent
Tests the meta-agent's ability to merge conflicting optimization plans
"""
import json
from datetime import datetime
from conflict_detector_agent import detect_conflicts
from conflict_resolution_agent import resolve_conflicts
from optimization_agent import OptimizationPlan, ParamChange, IntentParse, KpiSnapshot

def create_test_proposal(
    area: str, 
    priority: str, 
    confidence: float,
    bs_changes: dict,
    expected_kpis: dict,
    current_kpis: dict = None
) -> dict:
    """Helper to create a complete proposal"""
    
    intent = IntentParse(
        target_area=area,
        target_kpis=list(expected_kpis.keys()),
        priority=priority,
        confidence=confidence
    )
    
    changes = [
        ParamChange(
            param=param, 
            before=0 if not isinstance(value, bool) else False, 
            after=value,
            unit="dBm" if "P_dBm" in param else ("deg" if "dAz" in param or "dEl" in param else None)
        )
        for param, value in bs_changes.items()
    ]
    
    plan = OptimizationPlan(
        selected_config_id=int(datetime.now().strftime("%Y%m%d%H%M%S%f")[:14]),
        current_config_id=0,
        changes=changes,
        expected_kpis=KpiSnapshot(**expected_kpis),
        current_kpis=KpiSnapshot(**current_kpis) if current_kpis else None,
        constraints_satisfied=True
    )
    
    return {
        "intent": intent.model_dump(),
        "plan": plan.model_dump()
    }

# ==========================================
# TEST SCENARIOS
# ==========================================

test_scenarios = [
    {
        "name": "Scenario 1: CRITICAL vs LOW Priority - Same Parameter Conflict",
        "description": "CRITICAL intent should dominate over LOW priority",
        "proposals": [
            {
                "area": "Kadıköy",
                "priority": "CRITICAL",
                "confidence": 0.9,
                "changes": {"tx0_P_dBm": 5.0, "tx1_dAz": -10.0},
                "expected_kpis": {
                    "RX_POWER": -80.0,
                    "SINR": 10.0,
                    "THROUGHPUT_5P": 25.0,
                    "LOAD_IMBALANCE": 5.0,
                    "RX_COVERAGE_RATIO": 0.8
                },
                "current_kpis": {
                    "RX_POWER": -90.0,
                    "SINR": 5.0,
                    "THROUGHPUT_5P": 15.0,
                    "LOAD_IMBALANCE": 10.0,
                    "RX_COVERAGE_RATIO": 0.6
                }
            },
            {
                "area": "Kadıköy",
                "priority": "LOW",
                "confidence": 0.7,
                "changes": {"tx0_P_dBm": -3.0, "tx2_dEl": 2.0},  # Opposite direction on tx0_P_dBm
                "expected_kpis": {
                    "RX_POWER": -85.0,
                    "SINR": 7.0,
                    "THROUGHPUT_5P": 18.0,
                    "LOAD_IMBALANCE": 8.0,
                    "RX_COVERAGE_RATIO": 0.65
                },
                "current_kpis": {
                    "RX_POWER": -88.0,
                    "SINR": 6.0,
                    "THROUGHPUT_5P": 16.0,
                    "LOAD_IMBALANCE": 9.0,
                    "RX_COVERAGE_RATIO": 0.62
                }
            }
        ],
        "expected_behavior": "CRITICAL intent should get much higher weight. tx0_P_dBm should be close to +5.0"
    },
    {
        "name": "Scenario 2: Same Priority, Different Confidence",
        "description": "Higher confidence should get more weight when priorities are equal",
        "proposals": [
            {
                "area": "Beşiktaş",
                "priority": "HIGH",
                "confidence": 0.95,  # High confidence
                "changes": {"tx1_P_dBm": 4.0, "tx1_dAz": -5.0},
                "expected_kpis": {
                    "RX_POWER": -82.0,
                    "SINR": 12.0,
                    "THROUGHPUT_5P": 28.0,
                    "LOAD_IMBALANCE": 4.0,
                    "RX_COVERAGE_RATIO": 0.85
                },
                "current_kpis": {
                    "RX_POWER": -88.0,
                    "SINR": 8.0,
                    "THROUGHPUT_5P": 20.0,
                    "LOAD_IMBALANCE": 7.0,
                    "RX_COVERAGE_RATIO": 0.75
                }
            },
            {
                "area": "Beşiktaş",
                "priority": "HIGH",
                "confidence": 0.65,  # Lower confidence
                "changes": {"tx1_P_dBm": 2.0, "tx2_dAz": 10.0},
                "expected_kpis": {
                    "RX_POWER": -84.0,
                    "SINR": 9.0,
                    "THROUGHPUT_5P": 22.0,
                    "LOAD_IMBALANCE": 6.0,
                    "RX_COVERAGE_RATIO": 0.78
                },
                "current_kpis": {
                    "RX_POWER": -87.0,
                    "SINR": 8.5,
                    "THROUGHPUT_5P": 21.0,
                    "LOAD_IMBALANCE": 6.5,
                    "RX_COVERAGE_RATIO": 0.76
                }
            }
        ],
        "expected_behavior": "Higher confidence proposal should dominate. tx1_P_dBm should be closer to 4.0"
    },
    {
        "name": "Scenario 3: Multi-Intent Complex Merge",
        "description": "Three intents with different priorities and overlapping parameters",
        "proposals": [
            {
                "area": "Şişli",
                "priority": "CRITICAL",
                "confidence": 0.9,
                "changes": {"tx0_P_dBm": 6.0, "tx0_dAz": -8.0, "tx1_on": True},
                "expected_kpis": {
                    "RX_POWER": -78.0,
                    "SINR": 14.0,
                    "THROUGHPUT_5P": 30.0,
                    "LOAD_IMBALANCE": 3.0,
                    "RX_COVERAGE_RATIO": 0.9
                },
                "current_kpis": {
                    "RX_POWER": -92.0,
                    "SINR": 6.0,
                    "THROUGHPUT_5P": 18.0,
                    "LOAD_IMBALANCE": 9.0,
                    "RX_COVERAGE_RATIO": 0.55
                }
            },
            {
                "area": "Şişli",
                "priority": "HIGH",
                "confidence": 0.85,
                "changes": {"tx0_P_dBm": 4.0, "tx2_P_dBm": 3.0, "tx1_on": True},
                "expected_kpis": {
                    "RX_POWER": -82.0,
                    "SINR": 11.0,
                    "THROUGHPUT_5P": 26.0,
                    "LOAD_IMBALANCE": 5.0,
                    "RX_COVERAGE_RATIO": 0.82
                },
                "current_kpis": {
                    "RX_POWER": -88.0,
                    "SINR": 7.0,
                    "THROUGHPUT_5P": 20.0,
                    "LOAD_IMBALANCE": 8.0,
                    "RX_COVERAGE_RATIO": 0.68
                }
            },
            {
                "area": "Pendik",  # Different area
                "priority": "MEDIUM",
                "confidence": 0.75,
                "changes": {"tx3_P_dBm": 2.0, "tx3_dEl": 1.0},
                "expected_kpis": {
                    "RX_POWER": -85.0,
                    "SINR": 8.0,
                    "THROUGHPUT_5P": 20.0,
                    "LOAD_IMBALANCE": 7.0,
                    "RX_COVERAGE_RATIO": 0.7
                },
                "current_kpis": {
                    "RX_POWER": -89.0,
                    "SINR": 7.0,
                    "THROUGHPUT_5P": 18.0,
                    "LOAD_IMBALANCE": 8.0,
                    "RX_COVERAGE_RATIO": 0.65
                }
            }
        ],
        "expected_behavior": "CRITICAL intent dominates tx0_P_dBm. Boolean tx1_on should be True (weighted majority). Other params weighted merge."
    },
    {
        "name": "Scenario 4: Boolean Parameter Conflict",
        "description": "Test weighted majority voting for boolean parameters",
        "proposals": [
            {
                "area": "Kartal",
                "priority": "HIGH",
                "confidence": 0.9,
                "changes": {"tx2_on": True, "tx2_P_dBm": 5.0},
                "expected_kpis": {
                    "RX_POWER": -80.0,
                    "SINR": 12.0,
                    "THROUGHPUT_5P": 27.0,
                    "LOAD_IMBALANCE": 4.0,
                    "RX_COVERAGE_RATIO": 0.85
                },
                "current_kpis": {
                    "RX_POWER": -90.0,
                    "SINR": 6.0,
                    "THROUGHPUT_5P": 15.0,
                    "LOAD_IMBALANCE": 10.0,
                    "RX_COVERAGE_RATIO": 0.6
                }
            },
            {
                "area": "Kartal",
                "priority": "MEDIUM",
                "confidence": 0.7,
                "changes": {"tx2_on": False, "tx2_dAz": -5.0},
                "expected_kpis": {
                    "RX_POWER": -88.0,
                    "SINR": 8.0,
                    "THROUGHPUT_5P": 19.0,
                    "LOAD_IMBALANCE": 7.0,
                    "RX_COVERAGE_RATIO": 0.68
                },
                "current_kpis": {
                    "RX_POWER": -89.0,
                    "SINR": 7.5,
                    "THROUGHPUT_5P": 17.0,
                    "LOAD_IMBALANCE": 8.0,
                    "RX_COVERAGE_RATIO": 0.65
                }
            }
        ],
        "expected_behavior": "HIGH priority should win. tx2_on should be True due to higher weight."
    }
]

def run_resolution_tests():
    print("="*100)
    print("CONFLICT RESOLUTION META-AGENT - TEST SUITE")
    print("="*100)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*100}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*100}")
        print(f"Description: {scenario['description']}")
        print(f"Expected Behavior: {scenario['expected_behavior']}")
        print()
        
        # Create proposals
        active_list = []
        for prop_data in scenario['proposals']:
            proposal = create_test_proposal(
                area=prop_data['area'],
                priority=prop_data['priority'],
                confidence=prop_data['confidence'],
                bs_changes=prop_data['changes'],
                expected_kpis=prop_data['expected_kpis'],
                current_kpis=prop_data.get('current_kpis')
            )
            active_list.append(proposal)
        
        # Step 1: Run Conflict Detection
        print("STEP 1: Conflict Detection")
        print("-" * 50)
        
        if len(active_list) < 2:
            print("⚠️  Need at least 2 proposals for conflict detection")
            continue
        
        new_proposal = active_list[0]
        active_proposals = active_list[1:]
        
        conflict_result_json = detect_conflicts(
            new_intent_json=json.dumps(new_proposal['intent']),
            new_plan_json=json.dumps(new_proposal['plan']),
            active_intents_data=json.dumps(active_proposals)
        )
        
        conflict_result = json.loads(conflict_result_json)
        conflict_report = conflict_result['conflict_report']
        
        print(f"Conflicts Detected: {conflict_report['is_conflicted']}")
        print(f"Conflict Summary: {conflict_report['conflict_summary']}")
        print(f"Total Conflicts: {len(conflict_report['details'])}")
        
        if conflict_report['details']:
            print("\nConflict Details:")
            for detail in conflict_report['details'][:3]:  # Show first 3
                print(f"  • {detail['conflict_type']} ({detail['severity']})")
                print(f"    Param: {detail.get('conflicting_param', 'N/A')}, BS: {detail.get('conflicting_base_station', 'N/A')}")
        
        # Step 2: Run Conflict Resolution
        print(f"\n{'='*50}")
        print("STEP 2: Conflict Resolution")
        print("-" * 50)
        
        resolution_result_json = resolve_conflicts(conflict_result_json)
        resolution_result = json.loads(resolution_result_json)
        
        print(f"Conflict Detected: {resolution_result['conflict_detected']}")
        print(f"Resolution Applied: {resolution_result['resolution_applied']}")
        print(f"Resolution Notes: {resolution_result['resolution_notes']}")
        
        if resolution_result['merged_configuration']:
            merged = resolution_result['merged_configuration']
            
            print(f"\n{'='*50}")
            print("MERGED CONFIGURATION")
            print("-" * 50)
            print(f"Config ID: {merged['merged_config_id']}")
            print(f"Resolution Strategy: {merged['resolution_strategy']}")
            print(f"Contributing Intents: {len(merged['contributing_intents'])}")
            
            print("\nPriority Weights:")
            for agent_id, weight in merged['priority_weights'].items():
                intent_idx = int(agent_id.split('_')[-1]) if 'active' in agent_id else 0
                priority = scenario['proposals'][intent_idx]['priority']
                print(f"  {agent_id}: {weight:.3f} (Priority: {priority})")
            
            print(f"\nParameter Changes ({len(merged['changes'])}):")
            for change in merged['changes'][:10]:  # Show first 10
                print(f"  {change['param']}: {change['before']} → {change['after']} {change['unit'] or ''}")
            
            print(f"\nExpected KPIs:")
            kpis = merged['expected_kpis']
            for kpi, value in kpis.items():
                if value is not None:
                    print(f"  {kpi}: {value:.2f}")
            
            print(f"\nConstraints Satisfied: {merged['constraints_satisfied']}")
            
            # Save detailed results
            output_file = f"results/conflict_resolution_scenario_{i}.json"
            os.makedirs("results", exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    "scenario": scenario['name'],
                    "description": scenario['description'],
                    "input_proposals": scenario['proposals'],
                    "conflict_detection": conflict_result,
                    "resolution_output": resolution_result
                }, f, indent=2)
            
            print(f"\n💾 Detailed results saved to: {output_file}")
        
        print(f"\n{'='*100}\n")

if __name__ == "__main__":
    import os
    run_resolution_tests()
