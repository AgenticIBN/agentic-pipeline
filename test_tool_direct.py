#!/usr/bin/env python3
"""
Direct test of optimize_from_intent tool (NO AGENT, NO API calls)
"""
from optimization_agent import optimize_from_intent
import json

# Test Case: Simple Coverage Improvement
test_input = {
    "target_area": "Kadıköy",
    "target_kpis": ["RX_POWER"],
    "kpi_thresholds": [
        {
            "kpi": "RX_POWER",
            "op": "GTE",
            "value": -95.0,
            "unit": "dBm"
        }
    ],
    "priority": "HIGH",
    "confidence": 0.9,
    "user_set_id": 0,  # Dataset has user_set_id=0
    "k_users": 800      # Dataset has K_users=800
}

print("=" * 80)
print("OPTIMIZATION TOOL - DIRECT TEST (NO GEMINI API)")
print("=" * 80)
print("\n📥 INPUT:")
print(json.dumps(test_input, indent=2))
print("\n🔄 Running optimization tool directly...")
print("-" * 80)

try:
    # Call the tool function directly (no agent, no API)
    result_json = optimize_from_intent(json.dumps(test_input))
    
    print("\n✅ OUTPUT:")
    print("-" * 80)
    
    # Parse and pretty print
    result = json.loads(result_json)
    print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"  Selected Config ID: {result['selected_config_id']}")
    print(f"  Baseline Config ID: {result.get('baseline_config_id', 'N/A')}")
    print(f"  Number of Changes: {len(result['changes'])}")
    print(f"  Constraints Satisfied: {result['constraints_satisfied']}")
    print(f"  Warnings: {len(result['warnings'])}")
    
    if result['expected_kpis']:
        print(f"\n  Expected KPIs:")
        for k, v in result['expected_kpis'].items():
            if v is not None:
                print(f"    - {k}: {v}")
    
except Exception as e:
    print("\n❌ ERROR:")
    print("-" * 80)
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 80)
    print("❌ TEST FAILED")
    print("=" * 80)
