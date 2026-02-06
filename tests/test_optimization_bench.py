#!/usr/bin/env python3
"""
Optimization Agent Test Bench
Tests agent with various scenarios and validates output correctness
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from optimization_agent import optimize_from_intent

# Test Results Directory
TEST_RESULTS_DIR = "test_results"
Path(TEST_RESULTS_DIR).mkdir(exist_ok=True)


class TestCase:
    """Test case with input, expected behavior, and validation"""
    def __init__(
        self,
        name: str,
        input_data: Dict[str, Any],
        current_config_id: int = None,
        expected_behavior: Dict[str, Any] = None,
    ):
        self.name = name
        self.input_data = input_data
        if current_config_id is not None:
            self.input_data["current_config_id"] = current_config_id
        self.expected_behavior = expected_behavior or {}
    
    def validate_output(self, output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate output against expected behavior"""
        errors = []
        
        # Basic structure checks
        required_fields = [
            "selected_config_id", "changes", "expected_kpis", "constraints_satisfied"
        ]
        for field in required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")
        
        # Current config ID check
        if self.input_data.get("current_config_id"):
            if output.get("current_config_id") != self.input_data["current_config_id"]:
                errors.append(f"Current config ID mismatch")
            
            # Should have current_kpis when current_config exists
            if output.get("current_kpis") is None:
                errors.append("Missing current_kpis when current_config_id is provided")
            
            # Changes should show deltas (before != None)
            has_proper_changes = False
            for change in output.get("changes", []):
                if change.get("before") is not None:
                    has_proper_changes = True
                    break
            if not has_proper_changes and len(output.get("changes", [])) > 0:
                errors.append("Changes should have 'before' values when current_config exists")
        else:
            # No current config → before should be None
            for change in output.get("changes", []):
                if change.get("before") is not None:
                    errors.append(f"Change {change['param']} has before != None without current_config")
                    break
        
        # Constraint satisfaction check
        if self.expected_behavior.get("constraints_satisfied") is not None:
            expected = self.expected_behavior["constraints_satisfied"]
            actual = output.get("constraints_satisfied", False)
            if expected != actual:
                errors.append(f"Constraints satisfied mismatch: expected={expected}, actual={actual}")
        
        # KPI threshold validation
        if "kpi_checks" in self.expected_behavior:
            for kpi_name, check in self.expected_behavior["kpi_checks"].items():
                kpi_value = output.get("expected_kpis", {}).get(kpi_name)
                if kpi_value is None:
                    errors.append(f"Missing KPI: {kpi_name}")
                    continue
                
                if "min" in check and kpi_value < check["min"]:
                    errors.append(f"{kpi_name}={kpi_value:.2f} < min={check['min']}")
                if "max" in check and kpi_value > check["max"]:
                    errors.append(f"{kpi_name}={kpi_value:.2f} > max={check['max']}")
        
        return (len(errors) == 0, errors)


# ============================================================================
# TEST CASES
# ============================================================================

test_cases = [
    # Test 1: Coverage Improvement (No Current Config)
    TestCase(
        name="T1_Coverage_Improvement_No_Current",
        input_data={
            "target_area": "Kadıköy",
            "target_kpis": ["RX_POWER"],
            "kpi_thresholds": [
                {"kpi": "RX_POWER", "op": "GTE", "value": -95.0, "unit": "dBm"}
            ],
            "priority": "HIGH",
            "confidence": 0.9,
            "user_set_id": 0,
            "k_users": 800,
        },
        expected_behavior={
            "constraints_satisfied": True,
            "kpi_checks": {
                "RX_POWER": {"min": -95.0},
            }
        }
    ),
    
    # Test 2: SINR Optimization with Current Config
    TestCase(
        name="T2_SINR_Optimization_With_Current",
        input_data={
            "target_area": "Beşiktaş",
            "target_kpis": ["SINR"],
            "kpi_thresholds": [
                {"kpi": "SINR", "op": "GTE", "value": 5.0, "unit": "dB"}
            ],
            "priority": "CRITICAL",
            "confidence": 0.95,
            "user_set_id": 0,
            "k_users": 800,
        },
        current_config_id=1,  # Assumes config_id=1 exists in dataset
        expected_behavior={
            "constraints_satisfied": True,
            "kpi_checks": {
                "SINR": {"min": 5.0},
            }
        }
    ),
    
    # Test 3: Throughput Maximization
    TestCase(
        name="T3_Throughput_Maximization",
        input_data={
            "target_area": "Şişli",
            "target_kpis": ["THROUGHPUT_5P"],
            "kpi_thresholds": [
                {"kpi": "THROUGHPUT_5P", "op": "GTE", "value": 10.0, "unit": "Mbps"}
            ],
            "priority": "MEDIUM",
            "confidence": 0.85,
            "user_set_id": 0,
            "k_users": 800,
        },
        expected_behavior={
            "kpi_checks": {
                "THROUGHPUT_5P": {"min": 10.0},
            }
        }
    ),
    
    # Test 4: Multi-KPI Optimization
    TestCase(
        name="T4_Multi_KPI_Optimization",
        input_data={
            "target_area": "Üsküdar",
            "target_kpis": ["RX_POWER", "SINR", "THROUGHPUT_5P"],
            "kpi_thresholds": [
                {"kpi": "RX_POWER", "op": "GTE", "value": -90.0, "unit": "dBm"},
                {"kpi": "SINR", "op": "GTE", "value": 3.0, "unit": "dB"},
            ],
            "priority": "HIGH",
            "confidence": 0.88,
            "user_set_id": 0,
            "k_users": 800,
        },
        current_config_id=5,
        expected_behavior={
            "kpi_checks": {
                "RX_POWER": {"min": -90.0},
                "SINR": {"min": 3.0},
            }
        }
    ),
    
    # Test 5: Coverage Ratio Target
    TestCase(
        name="T5_Coverage_Ratio_Target",
        input_data={
            "target_area": "Fatih",
            "target_kpis": ["RX_COVERAGE_RATIO"],
            "kpi_thresholds": [
                {"kpi": "RX_COVERAGE_RATIO", "op": "GTE", "value": 0.8}
            ],
            "priority": "CRITICAL",
            "confidence": 0.92,
            "user_set_id": 0,
            "k_users": 800,
        },
        expected_behavior={
            "kpi_checks": {
                "RX_COVERAGE_RATIO": {"min": 0.8, "max": 1.0},
            }
        }
    ),
    
    # Test 6: Low Priority Optimization
    TestCase(
        name="T6_Low_Priority_RX_Power",
        input_data={
            "target_area": "Eyüp",
            "target_kpis": ["RX_POWER"],
            "kpi_thresholds": [
                {"kpi": "RX_POWER", "op": "GTE", "value": -100.0, "unit": "dBm"}
            ],
            "priority": "LOW",
            "confidence": 0.75,
            "user_set_id": 0,
            "k_users": 800,
        },
        expected_behavior={
            "constraints_satisfied": True,
        }
    ),
    
    # Test 7: Between Constraint
    TestCase(
        name="T7_SINR_Between_Range",
        input_data={
            "target_area": "Sarıyer",
            "target_kpis": ["SINR"],
            "kpi_thresholds": [
                {"kpi": "SINR", "op": "BETWEEN", "value_low": 0.0, "value_high": 10.0}
            ],
            "priority": "MEDIUM",
            "confidence": 0.8,
            "user_set_id": 0,
            "k_users": 800,
        },
        current_config_id=10,
        expected_behavior={
            "kpi_checks": {
                "SINR": {"min": 0.0, "max": 10.0},
            }
        }
    ),
]


# ============================================================================
# TEST EXECUTION
# ============================================================================

def run_test_bench():
    """Run all test cases and generate report"""
    print("=" * 80)
    print("OPTIMIZATION AGENT - TEST BENCH")
    print("=" * 80)
    print(f"\nTotal Test Cases: {len(test_cases)}")
    print(f"Results Directory: {TEST_RESULTS_DIR}/")
    print("\n" + "-" * 80)
    
    results = []
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Running: {test_case.name}")
        print("-" * 80)
        
        try:
            # Run optimization
            input_json = json.dumps(test_case.input_data)
            output_json = optimize_from_intent(input_json)
            output = json.loads(output_json)
            
            # Validate output
            is_valid, errors = test_case.validate_output(output)
            
            # Store result
            test_result = {
                "test_name": test_case.name,
                "timestamp": datetime.now().isoformat(),
                "input": test_case.input_data,
                "output": output,
                "validation": {
                    "passed": is_valid,
                    "errors": errors,
                },
                "expected_behavior": test_case.expected_behavior,
            }
            results.append(test_result)
            
            # Print summary
            if is_valid:
                passed += 1
                print(f"✅ PASSED")
            else:
                failed += 1
                print(f"❌ FAILED")
                for error in errors:
                    print(f"   - {error}")
            
            # Print key metrics
            print(f"\n   Selected Config ID: {output['selected_config_id']}")
            print(f"   Current Config ID: {output.get('current_config_id', 'N/A')}")
            print(f"   Changes: {len(output.get('changes', []))}")
            print(f"   Constraints Satisfied: {output.get('constraints_satisfied')}")
            
            if output.get("expected_kpis"):
                print(f"\n   Expected KPIs:")
                for kpi, value in output["expected_kpis"].items():
                    if value is not None:
                        print(f"     - {kpi}: {value:.4f}")
            
        except Exception as e:
            failed += 1
            print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
            test_result = {
                "test_name": test_case.name,
                "timestamp": datetime.now().isoformat(),
                "input": test_case.input_data,
                "error": str(e),
                "validation": {"passed": False, "errors": [str(e)]},
            }
            results.append(test_result)
    
    # Save results to JSONL
    results_file = os.path.join(TEST_RESULTS_DIR, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    with open(results_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("TEST BENCH SUMMARY")
    print("=" * 80)
    print(f"\nTotal Tests: {len(test_cases)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_cases)*100):.1f}%")
    print(f"\n💾 Results saved to: {results_file}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_test_bench()
