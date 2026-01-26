#!/usr/bin/env python3
"""
Direct test of optimize_from_intent tool (NO AGENT, NO API calls)
"""
from optimization_agent import optimize_from_intent
import json
import os
from datetime import datetime
from pathlib import Path

def save_optimization_result(input_data, output_data, output_dir="results"):
    """
    Save optimization result as individual JSON file with unique timestamp ID.
    """
    # Create output directory if not exists
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate unique ID from timestamp
    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"optimization_{unique_id}.json")
    
    # Create record with metadata
    record = {
        "id": unique_id,
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "output": output_data
    }
    
    # Save as pretty-printed JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    
    return json_path

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
    
    # Save result to JSONL file
    saved_path = save_optimization_result(test_input, result)
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n💾 Result saved to: {saved_path}")
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"  Selected Config ID: {result['selected_config_id']}")
    print(f"  Current Config ID: {result.get('current_config_id', 'N/A')}")
    print(f"  Number of Changes: {len(result['changes'])}")
    print(f"  Constraints Satisfied: {result['constraints_satisfied']}")
    
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
