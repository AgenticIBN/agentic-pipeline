#!/usr/bin/env python3
"""
Comprehensive Optimization Agent Test Bench
Tests agent's ability to improve configurations based on intents
"""
from optimization_agent import optimize_from_intent
import json
from pathlib import Path
from datetime import datetime

# Test cases: Each has a "bad" current config and an intent requiring improvement
TEST_CASES = [
    {
        "name": "Test 1: RX_POWER Improvement",
        "description": "Current RX_POWER is poor (-95 dBm), intent requires >= -90 dBm",
        "input": {
            "target_area": "Kadıköy",
            "target_kpis": ["RX_POWER"],
            "kpi_thresholds": [
                {
                    "kpi": "RX_POWER",
                    "op": "GTE",
                    "value": -90.0,
                    "unit": "dBm"
                }
            ],
            "priority": "HIGH",
            "confidence": 0.9,
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                # Poor RX_POWER config: low power, bad angles
                "tx0_on": True,
                "tx0_P_dBm": 40.0,  # Low power
                "tx0_dAz": -25.0,
                "tx0_dEl": -3.0,
                "tx1_on": True,
                "tx1_P_dBm": 40.0,
                "tx1_dAz": 25.0,
                "tx1_dEl": -3.0,
                "tx2_on": True,
                "tx2_P_dBm": 40.0,
                "tx2_dAz": 25.0,
                "tx2_dEl": -3.0,
                "tx3_on": True,
                "tx3_P_dBm": 40.0,
                "tx3_dAz": -25.0,
                "tx3_dEl": -3.0,
            }
        },
        "expected_improvement": "RX_POWER should increase from ~-95 to >= -90 dBm"
    },
    {
        "name": "Test 2: SINR Improvement",
        "description": "Current SINR is poor (-8 dB), intent requires >= -5 dB",
        "input": {
            "target_area": "Beşiktaş",
            "target_kpis": ["SINR"],
            "kpi_thresholds": [
                {
                    "kpi": "SINR",
                    "op": "GTE",
                    "value": -5.0,
                    "unit": "dB"
                }
            ],
            "priority": "HIGH",
            "confidence": 0.85,
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                # Poor SINR: interference issues from bad beam angles
                "tx0_on": True,
                "tx0_P_dBm": 46.0,
                "tx0_dAz": 0.0,  # All beams pointing same direction (interference)
                "tx0_dEl": -2.0,
                "tx1_on": True,
                "tx1_P_dBm": 46.0,
                "tx1_dAz": 5.0,  # Too close to tx0
                "tx1_dEl": -2.0,
                "tx2_on": True,
                "tx2_P_dBm": 46.0,
                "tx2_dAz": -5.0,
                "tx2_dEl": -2.0,
                "tx3_on": True,
                "tx3_P_dBm": 46.0,
                "tx3_dAz": 0.0,
                "tx3_dEl": -2.0,
            }
        },
        "expected_improvement": "SINR should increase from ~-8 to >= -5 dB"
    },
    {
        "name": "Test 3: Throughput Improvement",
        "description": "Current throughput is low (~2 Mbps), intent requires >= 4 Mbps",
        "input": {
            "target_area": "Şişli",
            "target_kpis": ["THROUGHPUT_5P"],
            "kpi_thresholds": [
                {
                    "kpi": "THROUGHPUT_5P",
                    "op": "GTE",
                    "value": 4.0,
                    "unit": "Mbps"
                }
            ],
            "priority": "CRITICAL",
            "confidence": 0.95,
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                # Low throughput: poor SINR and power
                "tx0_on": True,
                "tx0_P_dBm": 42.0,
                "tx0_dAz": -20.0,
                "tx0_dEl": -6.0,  # Too steep tilt
                "tx1_on": True,
                "tx1_P_dBm": 42.0,
                "tx1_dAz": 20.0,
                "tx1_dEl": -6.0,
                "tx2_on": True,
                "tx2_P_dBm": 42.0,
                "tx2_dAz": 20.0,
                "tx2_dEl": -6.0,
                "tx3_on": True,
                "tx3_P_dBm": 42.0,
                "tx3_dAz": -20.0,
                "tx3_dEl": -6.0,
            }
        },
        "expected_improvement": "Throughput should increase from ~2 to >= 4 Mbps"
    },
    {
        "name": "Test 4: Coverage Ratio Improvement",
        "description": "Current coverage ratio is poor (0.4), intent requires >= 0.7",
        "input": {
            "target_area": "Üsküdar",
            "target_kpis": ["RX_COVERAGE_RATIO"],
            "kpi_thresholds": [
                {
                    "kpi": "RX_COVERAGE_RATIO",
                    "op": "GTE",
                    "value": 0.7,
                    "unit": None
                },
                {
                    "kpi": "RX_POWER",
                    "op": "GTE",
                    "value": -95.0,
                    "unit": "dBm"
                }
            ],
            "priority": "HIGH",
            "confidence": 0.88,
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                # Poor coverage: low power, narrow beam coverage
                "tx0_on": True,
                "tx0_P_dBm": 38.0,  # Very low power
                "tx0_dAz": -30.0,  # Extreme angles
                "tx0_dEl": -2.0,
                "tx1_on": False,  # One sector off
                "tx1_P_dBm": 0.0,
                "tx1_dAz": 0.0,
                "tx1_dEl": 0.0,
                "tx2_on": True,
                "tx2_P_dBm": 38.0,
                "tx2_dAz": 30.0,
                "tx2_dEl": -2.0,
                "tx3_on": True,
                "tx3_P_dBm": 38.0,
                "tx3_dAz": 0.0,
                "tx3_dEl": -2.0,
            }
        },
        "expected_improvement": "Coverage ratio should increase from ~0.4 to >= 0.7"
    },
    {
        "name": "Test 5: Multi-KPI Optimization",
        "description": "Current config fails multiple KPIs, intent requires all improvements",
        "input": {
            "target_area": "Sarıyer",
            "target_kpis": ["RX_POWER", "SINR", "THROUGHPUT_5P"],
            "kpi_thresholds": [
                {
                    "kpi": "RX_POWER",
                    "op": "GTE",
                    "value": -92.0,
                    "unit": "dBm"
                },
                {
                    "kpi": "SINR",
                    "op": "GTE",
                    "value": -6.0,
                    "unit": "dB"
                },
                {
                    "kpi": "THROUGHPUT_5P",
                    "op": "GTE",
                    "value": 3.5,
                    "unit": "Mbps"
                }
            ],
            "priority": "CRITICAL",
            "confidence": 0.92,
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                # All KPIs are bad
                "tx0_on": True,
                "tx0_P_dBm": 41.0,
                "tx0_dAz": -15.0,
                "tx0_dEl": -4.5,
                "tx1_on": True,
                "tx1_P_dBm": 41.0,
                "tx1_dAz": 15.0,
                "tx1_dEl": -4.5,
                "tx2_on": True,
                "tx2_P_dBm": 41.0,
                "tx2_dAz": 15.0,
                "tx2_dEl": -4.5,
                "tx3_on": True,
                "tx3_P_dBm": 41.0,
                "tx3_dAz": -15.0,
                "tx3_dEl": -4.5,
            }
        },
        "expected_improvement": "All KPIs should meet thresholds"
    },
    {
        "name": "Test 6: Load Balancing (SERVED_USERS)",
        "description": "Current config has high load imbalance, intent requires better distribution",
        "input": {
            "target_area": "Bakırköy",
            "target_kpis": ["SERVED_USERS"],
            "kpi_thresholds": [
                {
                    "kpi": "SERVED_USERS",
                    "op": "LTE",
                    "value": 8.0,  # Lower imbalance is better
                    "unit": None
                }
            ],
            "priority": "MEDIUM",
            "confidence": 0.80,
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                # Unbalanced: some sectors high power, others low
                "tx0_on": True,
                "tx0_P_dBm": 46.0,  # Max power
                "tx0_dAz": -30.0,
                "tx0_dEl": -3.0,
                "tx1_on": True,
                "tx1_P_dBm": 38.0,  # Low power
                "tx1_dAz": 30.0,
                "tx1_dEl": -3.0,
                "tx2_on": True,
                "tx2_P_dBm": 46.0,
                "tx2_dAz": 30.0,
                "tx2_dEl": -3.0,
                "tx3_on": True,
                "tx3_P_dBm": 38.0,  # Low power
                "tx3_dAz": -30.0,
                "tx3_dEl": -3.0,
            }
        },
        "expected_improvement": "Load imbalance should decrease (better distribution)"
    },
    {
        "name": "Test 7: Combined RX_POWER + Coverage",
        "description": "Current has poor power and coverage, intent requires both improvements",
        "input": {
            "target_area": "Maltepe",
            "target_kpis": ["RX_POWER", "RX_COVERAGE_RATIO"],
            "kpi_thresholds": [
                {
                    "kpi": "RX_POWER",
                    "op": "GTE",
                    "value": -88.0,
                    "unit": "dBm"
                },
                {
                    "kpi": "RX_COVERAGE_RATIO",
                    "op": "GTE",
                    "value": 0.65,
                    "unit": None
                }
            ],
            "priority": "HIGH",
            "confidence": 0.87,
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                # Poor power and coverage
                "tx0_on": True,
                "tx0_P_dBm": 39.0,
                "tx0_dAz": -28.0,
                "tx0_dEl": -5.5,
                "tx1_on": True,
                "tx1_P_dBm": 39.0,
                "tx1_dAz": 28.0,
                "tx1_dEl": -5.5,
                "tx2_on": True,
                "tx2_P_dBm": 39.0,
                "tx2_dAz": 28.0,
                "tx2_dEl": -5.5,
                "tx3_on": True,
                "tx3_P_dBm": 39.0,
                "tx3_dAz": -28.0,
                "tx3_dEl": -5.5,
            }
        },
        "expected_improvement": "Both RX_POWER and coverage ratio should improve"
    },
    {
        "name": "Test 8: Extreme Tilt Correction",
        "description": "Current has wrong tilt angles causing poor coverage",
        "input": {
            "target_area": "Kartal",
            "target_kpis": ["RX_COVERAGE_RATIO", "THROUGHPUT_5P"],
            "kpi_thresholds": [
                {
                    "kpi": "RX_COVERAGE_RATIO",
                    "op": "GTE",
                    "value": 0.60,
                    "unit": None
                },
                {
                    "kpi": "THROUGHPUT_5P",
                    "op": "GTE",
                    "value": 3.8,
                    "unit": "Mbps"
                }
            ],
            "priority": "HIGH",
            "confidence": 0.83,
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                # Extreme tilt causing coverage issues
                "tx0_on": True,
                "tx0_P_dBm": 44.0,
                "tx0_dAz": -25.0,
                "tx0_dEl": -7.0,  # Too steep
                "tx1_on": True,
                "tx1_P_dBm": 44.0,
                "tx1_dAz": 25.0,
                "tx1_dEl": -1.0,  # Too shallow
                "tx2_on": True,
                "tx2_P_dBm": 44.0,
                "tx2_dAz": 25.0,
                "tx2_dEl": -7.0,
                "tx3_on": True,
                "tx3_P_dBm": 44.0,
                "tx3_dAz": -25.0,
                "tx3_dEl": -1.0,
            }
        },
        "expected_improvement": "Tilt angles should be optimized for better coverage"
    },
    {
        "name": "Test 9: Power Efficiency",
        "description": "Current uses max power inefficiently, optimize for better SINR with less power",
        "input": {
            "target_area": "Ataşehir",
            "target_kpis": ["SINR", "RX_POWER"],
            "kpi_thresholds": [
                {
                    "kpi": "SINR",
                    "op": "GTE",
                    "value": -4.0,
                    "unit": "dB"
                },
                {
                    "kpi": "RX_POWER",
                    "op": "GTE",
                    "value": -89.0,
                    "unit": "dBm"
                }
            ],
            "priority": "MEDIUM",
            "confidence": 0.78,
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                # High power but poor SINR due to interference
                "tx0_on": True,
                "tx0_P_dBm": 46.0,  # Max power
                "tx0_dAz": -10.0,
                "tx0_dEl": -3.5,
                "tx1_on": True,
                "tx1_P_dBm": 46.0,
                "tx1_dAz": 10.0,
                "tx1_dEl": -3.5,
                "tx2_on": True,
                "tx2_P_dBm": 46.0,
                "tx2_dAz": 10.0,
                "tx2_dEl": -3.5,
                "tx3_on": True,
                "tx3_P_dBm": 46.0,
                "tx3_dAz": -10.0,
                "tx3_dEl": -3.5,
            }
        },
        "expected_improvement": "Better SINR with optimized angles and potentially lower power"
    },
    {
        "name": "Test 10: Sector Activation Optimization",
        "description": "One sector is off, causing coverage gaps",
        "input": {
            "target_area": "Pendik",
            "target_kpis": ["RX_COVERAGE_RATIO", "RX_POWER"],
            "kpi_thresholds": [
                {
                    "kpi": "RX_COVERAGE_RATIO",
                    "op": "GTE",
                    "value": 0.68,
                    "unit": None
                },
                {
                    "kpi": "RX_POWER",
                    "op": "GTE",
                    "value": -91.0,
                    "unit": "dBm"
                }
            ],
            "priority": "CRITICAL",
            "confidence": 0.91,
            "user_set_id": 0,
            "k_users": 800,
            "current_config": {
                # One sector off + suboptimal configuration
                "tx0_on": True,
                "tx0_P_dBm": 43.0,
                "tx0_dAz": -22.0,
                "tx0_dEl": -4.0,
                "tx1_on": False,  # Sector OFF
                "tx1_P_dBm": 0.0,
                "tx1_dAz": 0.0,
                "tx1_dEl": 0.0,
                "tx2_on": True,
                "tx2_P_dBm": 43.0,
                "tx2_dAz": 22.0,
                "tx2_dEl": -4.0,
                "tx3_on": True,
                "tx3_P_dBm": 43.0,
                "tx3_dAz": 0.0,
                "tx3_dEl": -4.0,
            }
        },
        "expected_improvement": "Coverage should improve by activating sector and optimizing"
    }
]


def print_section(title, char="=", width=100):
    """Print a formatted section header"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def print_kpi_comparison(current_kpis, expected_kpis, thresholds):
    """Print KPI comparison table"""
    print(f"\n{'KPI':<20} {'Current':<15} {'Expected':<15} {'Threshold':<20} {'Status':<10}")
    print("-" * 80)
    
    kpi_map = {
        "RX_POWER": ("RX_POWER", "dBm"),
        "SINR": ("SINR", "dB"),
        "THROUGHPUT_5P": ("THROUGHPUT_5P", "Mbps"),
        "RX_COVERAGE_RATIO": ("RX_COVERAGE_RATIO", ""),
        "LOAD_IMBALANCE": ("LOAD_IMBALANCE", "")
    }
    
    # Build threshold dict
    thr_dict = {}
    for thr in thresholds:
        thr_dict[thr["kpi"]] = f"{thr['op']} {thr['value']}"
    
    for kpi_name, (key, unit) in kpi_map.items():
        curr_val = current_kpis.get(key) if current_kpis else None
        exp_val = expected_kpis.get(key) if expected_kpis else None
        thr_str = thr_dict.get(kpi_name, "-")
        
        if curr_val is not None and exp_val is not None:
            curr_str = f"{curr_val:.2f} {unit}".strip()
            exp_str = f"{exp_val:.2f} {unit}".strip()
            
            # Check if improved
            if kpi_name == "LOAD_IMBALANCE":
                improved = exp_val < curr_val  # Lower is better
            else:
                improved = exp_val > curr_val  # Higher is better
            
            status = "✅ Improved" if improved else "⚠️  Same/Worse"
            print(f"{kpi_name:<20} {curr_str:<15} {exp_str:<15} {thr_str:<20} {status:<10}")


def print_config_comparison(current_config, changes):
    """Print configuration changes"""
    print(f"\n{'Parameter':<20} {'Current':<15} {'Change':<15} {'Status':<10}")
    print("-" * 60)
    
    for change in changes:
        param = change["param"]
        before = change["before"]
        after = change["after"]
        unit = change.get("unit", "")
        
        if isinstance(after, bool):
            status = "🔄 Changed" if before != after else "- Same"
            print(f"{param:<20} {str(before):<15} {str(after):<15} {status:<10}")
        else:
            # For numeric: after is delta
            if before is not None:
                before_str = f"{before:.2f} {unit}".strip()
                delta_str = f"{after:+.2f} {unit}".strip() if after != 0 else "No change"
                status = "🔄 Changed" if after != 0 else "- Same"
            else:
                before_str = "N/A"
                delta_str = f"{after:.2f} {unit}".strip()
                status = "🆕 New"
            
            print(f"{param:<20} {before_str:<15} {delta_str:<15} {status:<10}")


def run_test(test_case, test_num, total_tests):
    """Run a single test case"""
    print_section(f"TEST {test_num}/{total_tests}: {test_case['name']}", char="=")
    
    print(f"📝 Description: {test_case['description']}")
    print(f"🎯 Expected: {test_case['expected_improvement']}\n")
    
    # Print INPUT
    print("📥 INPUT INTENT:")
    print("-" * 100)
    input_data = test_case["input"]
    print(f"  Target Area: {input_data['target_area']}")
    print(f"  Target KPIs: {', '.join(input_data['target_kpis'])}")
    print(f"  Priority: {input_data['priority']}")
    print(f"  User Set: {input_data['user_set_id']}, K_users: {input_data['k_users']}")
    print(f"\n  KPI Thresholds:")
    for thr in input_data["kpi_thresholds"]:
        unit_str = f" {thr['unit']}" if thr.get('unit') else ""
        print(f"    - {thr['kpi']}: {thr['op']} {thr['value']}{unit_str}")
    
    print(f"\n  Current Configuration (before optimization):")
    curr_cfg = input_data["current_config"]
    for i in range(4):
        on_status = "ON" if curr_cfg[f"tx{i}_on"] else "OFF"
        print(f"    tx{i}: {on_status:3s} | P={curr_cfg[f'tx{i}_P_dBm']:.1f}dBm, "
              f"Az={curr_cfg[f'tx{i}_dAz']:+.1f}°, El={curr_cfg[f'tx{i}_dEl']:+.1f}°")
    
    # Run optimization
    print(f"\n🔄 Running optimization agent...")
    print("-" * 100)
    
    try:
        result_json = optimize_from_intent(json.dumps(input_data))
        result = json.loads(result_json)
        
        # Print OUTPUT
        print("\n✅ OUTPUT OPTIMIZATION PLAN:")
        print("-" * 100)
        print(f"  Selected Config ID: {result['selected_config_id']}")
        print(f"  Current Config ID: {result.get('current_config_id', 'N/A (manual config)')}")
        print(f"  Constraints Satisfied: {'✅ YES' if result['constraints_satisfied'] else '❌ NO'}")
        
        # KPI Comparison
        print_kpi_comparison(
            result.get("current_kpis"),
            result.get("expected_kpis"),
            input_data["kpi_thresholds"]
        )
        
        # Configuration Changes
        if result["changes"]:
            print(f"\n📊 CONFIGURATION CHANGES ({len(result['changes'])} parameters):")
            print_config_comparison(curr_cfg, result["changes"])
        else:
            print(f"\n⚠️  No configuration changes (no current config baseline)")
        
        # Test Result
        print(f"\n{'='*100}")
        if result['constraints_satisfied']:
            print(f"✅ TEST PASSED: Constraints satisfied!")
        else:
            print(f"⚠️  TEST WARNING: Constraints not fully satisfied (best-effort result)")
        print(f"{'='*100}")
        
        return {
            "test_name": test_case["name"],
            "passed": result['constraints_satisfied'],
            "input": input_data,
            "output": result
        }
        
    except Exception as e:
        print(f"\n❌ ERROR during optimization:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "test_name": test_case["name"],
            "passed": False,
            "error": str(e)
        }


def save_results(results, output_dir="testbench_results"):
    """Save each test result as individual JSON file with unique ID"""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = []
    for i, result in enumerate(results, 1):
        # Generate unique ID: timestamp + test number
        unique_id = f"{timestamp}_test{i:02d}"
        json_path = f"{output_dir}/optimization_{unique_id}.json"
        
        # Save as pretty-printed JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        saved_files.append(json_path)
    
    return saved_files


def main():
    print_section("OPTIMIZATION AGENT TEST BENCH", char="█", width=100)
    print(f"Running {len(TEST_CASES)} test cases...\n")
    
    results = []
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(TEST_CASES, 1):
        result = run_test(test_case, i, len(TEST_CASES))
        results.append(result)
        
        if result.get("passed", False):
            passed += 1
        else:
            failed += 1
        
        print()  # Spacing between tests
    
    # Final Summary
    print_section("TEST BENCH SUMMARY", char="█", width=100)
    print(f"Total Tests: {len(TEST_CASES)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed/Warning: {failed}")
    print(f"Success Rate: {(passed/len(TEST_CASES)*100):.1f}%\n")
    
    # Save results
    saved_files = save_results(results)
    print(f"💾 Results saved to {len(saved_files)} individual JSON files:")
    for fpath in saved_files:
        print(f"   - {fpath}")
    print("=" * 100)


if __name__ == "__main__":
    main()
