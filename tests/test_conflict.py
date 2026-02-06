#!/usr/bin/env python3
"""
Test Base Station Level Conflict Detection
Compares OLD (target_area based) vs NEW (base station based) approaches
"""
import json
from conflict_detector_agent import detect_conflicts
from optimization_agent import OptimizationPlan, ParamChange, IntentParse

def create_test_intent(area: str, bs_changes: dict) -> tuple[IntentParse, OptimizationPlan]:
    """Helper to create intent and plan"""
    intent = IntentParse(
        target_area=area,
        target_kpis=["RX_POWER"],
        priority="HIGH",
        confidence=0.9
    )
    
    changes = [
        ParamChange(param=param, before=0, change=value)
        for param, value in bs_changes.items()
    ]
    
    plan = OptimizationPlan(
        selected_config_id=1,
        current_config_id=0,
        changes=changes,
        expected_kpis={"RX_POWER": -85.0},
        constraints_satisfied=True
    )
    
    return intent, plan

# Test Cases
test_cases = [
    {
        "name": "Test 1: Different Areas, Different Base Stations",
        "new": ("Kadıköy", {"tx0_P_dBm": 5.0, "tx1_dAz": -10.0}),
        "active": [("Pendik", {"tx2_P_dBm": 3.0, "tx3_on": True})],
        "expected_severity": "LOW",
        "expected_conflicts": True,  # Minimal conflict
        "explanation": "Farklı BS'ler, minimal koordinasyon gerekli"
    },
    {
        "name": "Test 2: Different Areas, Same BS, Different Params",
        "new": ("Kadıköy", {"tx0_P_dBm": 5.0}),
        "active": [("Pendik", {"tx0_dAz": -10.0})],
        "expected_severity": "LOW-MEDIUM",
        "expected_conflicts": True,
        "explanation": "Aynı BS (tx0) ama farklı parametreler"
    },
    {
        "name": "Test 3: Different Areas, Same BS+Param, Opposite Direction",
        "new": ("Kadıköy", {"tx0_P_dBm": 5.0}),
        "active": [("Pendik", {"tx0_P_dBm": -3.0})],
        "expected_severity": "HIGH",
        "expected_conflicts": True,
        "explanation": "Aynı BS, aynı param, zıt yön - Same area olsaydı CRITICAL"
    },
    {
        "name": "Test 4: Same Area, Same BS+Param, Opposite Direction",
        "new": ("Kadıköy", {"tx0_P_dBm": 5.0}),
        "active": [("Kadıköy", {"tx0_P_dBm": -3.0})],
        "expected_severity": "CRITICAL",
        "expected_conflicts": True,
        "explanation": "CRITICAL conflict - Severity boost from same area"
    },
    {
        "name": "Test 5: Same Area, Same BS+Param, Same Direction",
        "new": ("Kadıköy", {"tx0_P_dBm": 5.0}),
        "active": [("Kadıköy", {"tx0_P_dBm": 3.0})],
        "expected_severity": "HIGH",
        "expected_conflicts": True,
        "explanation": "Over-saturation risk with severity boost"
    },
    {
        "name": "Test 6: Multiple Active Intents with Complex Conflicts",
        "new": ("Kadıköy", {"tx0_P_dBm": 5.0, "tx1_dAz": -10.0, "tx2_on": True}),
        "active": [
            ("Pendik", {"tx0_P_dBm": -3.0, "tx3_dEl": 2.0}),  # tx0 conflict
            ("Beşiktaş", {"tx1_dAz": 15.0, "tx2_on": False}),  # tx1, tx2 conflicts
        ],
        "expected_severity": "CRITICAL",
        "expected_conflicts": True,
        "explanation": "Multiple conflicts: tx0 (opposite), tx1 (same param), tx2 (boolean)"
    },
]

def run_tests():
    print("="*80)
    print("BASE STATION LEVEL CONFLICT DETECTION - TEST SUITE")
    print("="*80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*80}")
        print(f"Expected: {test['expected_severity']} severity")
        print(f"Explanation: {test['explanation']}")
        print()
        
        # Create intents and plans
        new_intent, new_plan = create_test_intent(test['new'][0], test['new'][1])
        
        active_list = []
        for active_area, active_changes in test['active']:
            active_intent, active_plan = create_test_intent(active_area, active_changes)
            active_list.append({
                "intent": active_intent.model_dump(),
                "plan": active_plan.model_dump()
            })
        
        # Run conflict detection
        result_json = detect_conflicts(
            new_intent_json=new_intent.model_dump_json(),
            new_plan_json=new_plan.model_dump_json(),
            active_intents_data=json.dumps(active_list)
        )
        
        result = json.loads(result_json)
        conflict_report = result['conflict_report']
        
        # Print results
        print(f"Result:")
        print(f"  Conflicted: {conflict_report['is_conflicted']}")
        print(f"  Max Severity: {conflict_report.get('conflict_summary', 'N/A')}")
        print(f"  Total Conflicts: {len(conflict_report['details'])}")
        print()
        
        print("Conflict Details:")
        for detail in conflict_report['details']:
            print(f"  - Type: {detail['conflict_type']}")
            print(f"    Severity: {detail['severity']}")
            print(f"    Base Station: {detail.get('conflicting_base_station', 'N/A')}")
            print(f"    Parameter: {detail.get('conflicting_param', 'N/A')}")
            print(f"    Description: {detail['description']}")
            print()
        
        print(f"Recommendation: {conflict_report['resolution_recommendation']}")
        
        # Validation
        if conflict_report['is_conflicted'] == test['expected_conflicts']:
            print(f"✅ PASS: Conflict detection matches expected")
        else:
            print(f"❌ FAIL: Expected conflicts={test['expected_conflicts']}, got {conflict_report['is_conflicted']}")

if __name__ == "__main__":
    run_tests()
