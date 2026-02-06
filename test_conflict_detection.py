#!/usr/bin/env python3
"""
Test conflict detection agent with two conflicting intents.
"""
import json
from conflict_detector_agent import detect_conflicts

def main():
    print("="*80)
    print("CONFLICT DETECTION TEST")
    print("="*80)
    print()
    
    # Intent 1: Increase RX_POWER (needs to increase transmit power)
    intent1 = {
        "test_name": "increase_rx_power",
        "passed": True,
        "input": {
            "target_area": "sector_A",
            "target_kpis": ["RX_POWER"],
            "priority": "HIGH",
        },
        "output": {
            "selected_config_id": 20260206001,
            "current_config_id": None,
            "changes": [
                {"param": "tx0_P_dBm", "before": 43.0, "change": 3.0, "unit": "dBm"},
                {"param": "tx0_dAz", "before": 0.0, "change": -5.0, "unit": "deg"},
                {"param": "tx1_P_dBm", "before": 43.0, "change": 3.0, "unit": "dBm"},
                {"param": "tx1_dAz", "before": 120.0, "change": -5.0, "unit": "deg"},
                {"param": "tx2_P_dBm", "before": 43.0, "change": 2.0, "unit": "dBm"},
            ],
            "expected_kpis": {
                "RX_POWER": -52.5,
                "SINR": 21.2,
                "THROUGHPUT_5P": 3.8,
                "LOAD_IMBALANCE": 5.2,
            },
            "constraints_satisfied": True,
        }
    }
    
    # Intent 2: Reduce LOAD_IMBALANCE (conflicts with Intent 1 on tx0, tx1)
    intent2 = {
        "test_name": "reduce_load_imbalance",
        "passed": True,
        "input": {
            "target_area": "sector_A",
            "target_kpis": ["LOAD_IMBALANCE"],
            "priority": "CRITICAL",
        },
        "output": {
            "selected_config_id": 20260206002,
            "current_config_id": None,
            "changes": [
                {"param": "tx0_P_dBm", "before": 43.0, "change": -1.0, "unit": "dBm"},  # CONFLICT: tx0 power
                {"param": "tx0_dAz", "before": 0.0, "change": 10.0, "unit": "deg"},      # CONFLICT: tx0 azimuth
                {"param": "tx1_P_dBm", "before": 43.0, "change": -2.0, "unit": "dBm"},  # CONFLICT: tx1 power
                {"param": "tx1_dAz", "before": 120.0, "change": 5.0, "unit": "deg"},     # CONFLICT: tx1 azimuth
                {"param": "tx3_on", "before": False, "change": True, "unit": None},      # Turn ON tx3
            ],
            "expected_kpis": {
                "RX_POWER": -55.8,
                "SINR": 20.5,
                "THROUGHPUT_5P": 3.5,
                "LOAD_IMBALANCE": 2.1,
            },
            "constraints_satisfied": True,
        }
    }
    
    print("Intent 1: Increase RX_POWER")
    print(f"  Changes: {len(intent1['output']['changes'])} parameters")
    print(f"  Affected TXs: tx0_P_dBm (+3.0), tx0_dAz (-5.0), tx1_P_dBm (+3.0), tx1_dAz (-5.0), tx2_P_dBm (+2.0)")
    print(f"  Priority: {intent1['input']['priority']}")
    print()
    
    print("Intent 2: Reduce LOAD_IMBALANCE")
    print(f"  Changes: {len(intent2['output']['changes'])} parameters")
    print(f"  Affected TXs: tx0_P_dBm (-1.0), tx0_dAz (+10.0), tx1_P_dBm (-2.0), tx1_dAz (+5.0), tx3_on (ON)")
    print(f"  Priority: {intent2['input']['priority']}")
    print()
    
    print("="*80)
    print("Expected Conflicts:")
    print("  - tx0_P_dBm: Intent1 wants +3.0 dBm, Intent2 wants -1.0 dBm (OPPOSITE)")
    print("  - tx0_dAz: Intent1 wants -5.0 deg, Intent2 wants +10.0 deg (OPPOSITE)")
    print("  - tx1_P_dBm: Intent1 wants +3.0 dBm, Intent2 wants -2.0 dBm (OPPOSITE)")
    print("  - tx1_dAz: Intent1 wants -5.0 deg, Intent2 wants +5.0 deg (OPPOSITE)")
    print("="*80)
    print()
    
    # Run conflict detection
    print("Running Conflict Detection...")
    print()
    
    # Simulate: intent2 is already active, intent1 is new
    active_intents = [intent2]  # Intent 2 is already in the system
    new_intent = intent1  # Intent 1 is the new incoming intent
    
    report = detect_conflicts(new_intent, active_intents)
    
    print("="*80)
    print("CONFLICT DETECTION RESULT:")
    print("="*80)
    print(f"Conflicted: {report.is_conflicted}")
    print(f"Summary: {report.conflict_summary}")
    print(f"Total Conflicts: {report.num_conflicts}")
    print(f"Conflicting Result IDs: {report.conflicting_result_ids}")
    print()
    
    for i, conflict in enumerate(report.details, 1):
        print(f"Conflict {i}:")
        print(f"  Type: {conflict.conflict_type}")
        print(f"  Severity: {conflict.severity}")
        print(f"  Parameter: {conflict.parameter}")
        print(f"  Base Station: {conflict.base_station}")
        print(f"  Result1 ID: {conflict.intent1_id}")
        print(f"  Result2 ID: {conflict.intent2_id}")
        print(f"  Result1 change: {conflict.intent1_change:+.1f}")
        print(f"  Result2 change: {conflict.intent2_change:+.1f}")
        print(f"  Description: {conflict.description}")
        print()
    
    if report.resolution_recommendation:
        print(f"Recommendation: {report.resolution_recommendation}")
    
    print("="*80)
    
    # Test validation
    expected_conflicts = 4
    actual_conflicts = report.num_conflicts
    
    print()
    print(f"TEST RESULT: {'✓ PASS' if actual_conflicts == expected_conflicts else '✗ FAIL'}")
    print(f"Expected {expected_conflicts} conflicts, found {actual_conflicts}")
    print("="*80)
    
    # Save results
    output_file = "test_conflict_detection_result.json"
    with open(output_file, "w") as f:
        json.dump({
            "new_intent": intent1,
            "active_intents": active_intents,
            "conflict_report": report.model_dump()
        }, f, indent=2)
    
    print(f"\nFull result saved to: {output_file}")


if __name__ == "__main__":
    main()
