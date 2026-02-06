#!/usr/bin/env python3
"""
Simple test: 2 conflicted intents with different priorities
"""
import json
from workflow_steps import optimization_workflow, OptimizationWorkflowState, save_active_intents

def test_scenario_1():
    """
    Scenario: CRITICAL vs HIGH priority conflict
    Expected: CRITICAL wins
    """
    print("\n" + "="*70)
    print("TEST SCENARIO 1: CRITICAL vs HIGH Priority")
    print("="*70)
    
    # Setup: Active intent (HIGH priority)
    active_intents = [
        {
            "intent": {
                "target_area": "Kadikoy",
                "target_kpis": ["RX_POWER"],
                "priority": "HIGH",
                "confidence": 0.85,
                "constraints": {
                    "max_power_increase": 3.0,
                    "max_tilt_change": 5.0
                }
            },
            "plan": {
                "selected_config_id": 1001,
                "current_config_id": 0,
                "changes": [
                    {
                        "param": "tx0_P_dBm",
                        "before": 40.0,
                        "after": 2.5,  # Delta: +2.5 dBm
                        "justification": "Increase coverage in Kadikoy"
                    },
                    {
                        "param": "tx0_dAz",
                        "before": 0.0,
                        "after": -3.0,  # Delta: -3 degrees
                        "justification": "Adjust azimuth for Kadikoy"
                    }
                ],
                "expected_kpis": {
                    "RX_POWER": -82.0,
                    "SINR": 8.0,
                    "THROUGHPUT_5P": 18.0,
                    "LOAD_IMBALANCE": 6.0,
                    "RX_COVERAGE_RATIO": 0.72
                },
                "current_kpis": {
                    "RX_POWER": -90.0,
                    "SINR": 6.0,
                    "THROUGHPUT_5P": 12.0,
                    "LOAD_IMBALANCE": 8.0,
                    "RX_COVERAGE_RATIO": 0.60
                }
            }
        }
    ]
    
    # Save active intents
    save_active_intents(active_intents)
    print(f"✅ Setup: 1 active HIGH priority intent in Kadikoy")
    
    # New intent: CRITICAL priority (SAME area, SAME base station)
    new_intent_nl = """
    CRITICAL: Emergency coverage needed in Kadikoy. 
    Increase tx0 power to at least -80 dBm. 
    This is urgent due to network outage.
    """
    
    print(f"📥 New Intent: CRITICAL priority, Kadikoy (same as active)")
    
    # Create initial state
    initial_state = OptimizationWorkflowState(
        natural_language_intent=new_intent_nl,
        active_intents=active_intents
    )
    
    # Run workflow
    print("\n🚀 Running workflow...")
    result = optimization_workflow.run(initial_state)
    
    # Print results
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    state = result.content
    print(f"Conflict Detected: {state.conflict_detected}")
    print(f"Execution Strategy: {state.execution_strategy}")
    
    if state.resolution_result:
        res = state.resolution_result
        print(f"\n🏆 WINNER:")
        print(f"   Intent ID: {res.winning_intent_id}")
        print(f"   Priority: {res.winning_priority}")
        print(f"\n❌ REJECTED:")
        for rejected in res.rejected_intents:
            print(f"   - {rejected}")
        print(f"\nResolution Notes:")
        print(f"   {res.resolution_notes}")
    
    print("\n✅ Expected: CRITICAL intent wins, HIGH intent rejected")
    
    return state

def test_scenario_2():
    """
    Scenario: MEDIUM vs LOW priority conflict
    Expected: MEDIUM wins
    """
    print("\n" + "="*70)
    print("TEST SCENARIO 2: MEDIUM vs LOW Priority")
    print("="*70)
    
    # Setup: Active intent (LOW priority)
    active_intents = [
        {
            "intent": {
                "target_area": "Besiktas",
                "target_kpis": ["THROUGHPUT_5P"],
                "priority": "LOW",
                "confidence": 0.75,
                "constraints": {}
            },
            "plan": {
                "selected_config_id": 2001,
                "current_config_id": 0,
                "changes": [
                    {
                        "param": "tx1_P_dBm",
                        "before": 38.0,
                        "after": 1.0,  # Delta: +1 dBm
                        "justification": "Minor throughput optimization"
                    }
                ],
                "expected_kpis": {
                    "RX_POWER": -85.0,
                    "SINR": 7.5,
                    "THROUGHPUT_5P": 16.0,
                    "LOAD_IMBALANCE": 7.0,
                    "RX_COVERAGE_RATIO": 0.68
                },
                "current_kpis": {
                    "RX_POWER": -88.0,
                    "SINR": 7.0,
                    "THROUGHPUT_5P": 14.0,
                    "LOAD_IMBALANCE": 7.5,
                    "RX_COVERAGE_RATIO": 0.65
                }
            }
        }
    ]
    
    # Save active intents
    save_active_intents(active_intents)
    print(f"✅ Setup: 1 active LOW priority intent in Besiktas")
    
    # New intent: MEDIUM priority (SAME area, SAME base station)
    new_intent_nl = """
    Improve load balancing in Besiktas region. 
    Target: Load imbalance below 5%. 
    Medium priority.
    """
    
    print(f"📥 New Intent: MEDIUM priority, Besiktas (same as active)")
    
    # Create initial state
    initial_state = OptimizationWorkflowState(
        natural_language_intent=new_intent_nl,
        active_intents=active_intents
    )
    
    # Run workflow
    print("\n🚀 Running workflow...")
    result = optimization_workflow.run(initial_state)
    
    # Print results
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    state = result.content
    print(f"Conflict Detected: {state.conflict_detected}")
    print(f"Execution Strategy: {state.execution_strategy}")
    
    if state.resolution_result:
        res = state.resolution_result
        print(f"\n🏆 WINNER:")
        print(f"   Intent ID: {res.winning_intent_id}")
        print(f"   Priority: {res.winning_priority}")
        print(f"\n❌ REJECTED:")
        for rejected in res.rejected_intents:
            print(f"   - {rejected}")
        print(f"\nResolution Notes:")
        print(f"   {res.resolution_notes}")
    
    print("\n✅ Expected: MEDIUM intent wins, LOW intent rejected")
    
    return state

if __name__ == "__main__":
    # Clear any previous state
    save_active_intents([])
    
    # Run tests
    print("\n🧪 Testing Priority-Based Conflict Resolution")
    print("="*70)
    
    try:
        test_scenario_1()
        print("\n" + "="*70 + "\n")
        test_scenario_2()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED")
        print("="*70)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
