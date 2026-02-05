#!/usr/bin/env python3
"""
Test Conflict Detection Fixes
1. Boolean same value = No conflict
2. Different BS + Different Area = No conflict
3. Time overlap detection
"""
import json
from conflict_detector_agent import detect_conflicts, _check_time_overlap
from optimization_agent import OptimizationPlan, ParamChange, IntentParse, KpiSnapshot

def test_boolean_same_value():
    """Test: Both intents want tx0_on = True → Should NOT create conflict"""
    print("\n" + "="*60)
    print("TEST 1: Boolean Same Value (Both want ON)")
    print("="*60)
    
    new_intent = IntentParse(
        target_area="Kadıköy",
        target_kpis=["RX_POWER"],
        priority="HIGH",
        confidence=0.9
    )
    
    new_plan = OptimizationPlan(
        selected_config_id=12345,
        changes=[
            ParamChange(param="tx0_on", before=False, after=True),
            ParamChange(param="tx0_P_dBm", before=40.0, after=3.0, unit="dBm")
        ],
        expected_kpis=KpiSnapshot(RX_POWER=-80.0, SINR=10.0),
        constraints_satisfied=True
    )
    
    active_intent = IntentParse(
        target_area="Kadıköy",
        target_kpis=["SINR"],
        priority="MEDIUM",
        confidence=0.8
    )
    
    active_plan = OptimizationPlan(
        selected_config_id=54321,
        changes=[
            ParamChange(param="tx0_on", before=False, after=True),  # Same value!
            ParamChange(param="tx1_dAz", before=0.0, after=-5.0, unit="deg")
        ],
        expected_kpis=KpiSnapshot(RX_POWER=-85.0, SINR=12.0),
        constraints_satisfied=True
    )
    
    active_intents = [{"intent": active_intent.model_dump(), "plan": active_plan.model_dump()}]
    
    result_json = detect_conflicts(
        new_intent.model_dump_json(),
        new_plan.model_dump_json(),
        json.dumps(active_intents)
    )
    
    result = json.loads(result_json)
    conflicts = result['conflict_report']['details']
    
    # Check: tx0_on with same value should NOT appear in conflicts
    tx0_on_conflicts = [c for c in conflicts if c.get('conflicting_param') == 'tx0_on']
    
    print(f"Total conflicts detected: {len(conflicts)}")
    print(f"tx0_on conflicts: {len(tx0_on_conflicts)}")
    
    if len(tx0_on_conflicts) == 0:
        print("✅ PASS: Boolean same value does NOT create conflict")
    else:
        print("❌ FAIL: Boolean same value still creates conflict:")
        for c in tx0_on_conflicts:
            print(f"   - {c['conflict_type']}: {c['description']}")
    
    return len(tx0_on_conflicts) == 0

def test_different_bs_different_area():
    """Test: Different BS + Different Area → Should NOT create conflict"""
    print("\n" + "="*60)
    print("TEST 2: Different BS + Different Area")
    print("="*60)
    
    new_intent = IntentParse(
        target_area="Kadıköy",
        target_kpis=["RX_POWER"],
        priority="HIGH",
        confidence=0.9
    )
    
    new_plan = OptimizationPlan(
        selected_config_id=11111,
        changes=[
            ParamChange(param="tx0_P_dBm", before=40.0, after=3.0, unit="dBm")
        ],
        expected_kpis=KpiSnapshot(RX_POWER=-80.0),
        constraints_satisfied=True
    )
    
    active_intent = IntentParse(
        target_area="Beşiktaş",  # Different area!
        target_kpis=["SINR"],
        priority="MEDIUM",
        confidence=0.8
    )
    
    active_plan = OptimizationPlan(
        selected_config_id=22222,
        changes=[
            ParamChange(param="tx2_dAz", before=0.0, after=-5.0, unit="deg")  # Different BS!
        ],
        expected_kpis=KpiSnapshot(SINR=12.0),
        constraints_satisfied=True
    )
    
    active_intents = [{"intent": active_intent.model_dump(), "plan": active_plan.model_dump()}]
    
    result_json = detect_conflicts(
        new_intent.model_dump_json(),
        new_plan.model_dump_json(),
        json.dumps(active_intents)
    )
    
    result = json.loads(result_json)
    conflicts = result['conflict_report']['details']
    
    print(f"Total conflicts detected: {len(conflicts)}")
    
    if len(conflicts) == 0:
        print("✅ PASS: Different BS + Different Area = No conflict")
    else:
        print("❌ FAIL: Unnecessary conflicts detected:")
        for c in conflicts:
            print(f"   - {c['conflict_type']} ({c['severity']}): {c['description']}")
    
    return len(conflicts) == 0

def test_time_overlap():
    """Test: Time overlap detection"""
    print("\n" + "="*60)
    print("TEST 3: Time Overlap Detection")
    print("="*60)
    
    # Case 1: Overlapping times
    intent_a = IntentParse(
        target_area="Kadıköy",
        target_kpis=["RX_POWER"],
        priority="HIGH",
        confidence=0.9,
        time_constraint_start="2026-02-05T10:00:00",
        time_constraint_end="2026-02-05T14:00:00"
    )
    
    intent_b = IntentParse(
        target_area="Kadıköy",
        target_kpis=["SINR"],
        priority="MEDIUM",
        confidence=0.8,
        time_constraint_start="2026-02-05T12:00:00",  # Overlaps!
        time_constraint_end="2026-02-05T16:00:00"
    )
    
    conflict = _check_time_overlap(intent_a, intent_b)
    
    if conflict:
        print("✅ PASS: Time overlap detected")
        print(f"   Severity: {conflict.severity}")
        print(f"   Description: {conflict.description}")
    else:
        print("❌ FAIL: Time overlap NOT detected")
    
    # Case 2: No overlap
    intent_c = IntentParse(
        target_area="Kadıköy",
        target_kpis=["RX_POWER"],
        priority="HIGH",
        confidence=0.9,
        time_constraint_start="2026-02-05T08:00:00",
        time_constraint_end="2026-02-05T10:00:00"
    )
    
    conflict2 = _check_time_overlap(intent_a, intent_c)
    
    if not conflict2:
        print("✅ PASS: No overlap correctly detected")
    else:
        print("❌ FAIL: False positive - overlap detected where there is none")
    
    return conflict is not None and conflict2 is None

def test_same_bs_same_area():
    """Test: Same BS + Same Area + Different params → Should create LOW-MEDIUM conflict"""
    print("\n" + "="*60)
    print("TEST 4: Same BS + Same Area + Different Params")
    print("="*60)
    
    new_intent = IntentParse(
        target_area="Kadıköy",
        target_kpis=["RX_POWER"],
        priority="HIGH",
        confidence=0.9
    )
    
    new_plan = OptimizationPlan(
        selected_config_id=33333,
        changes=[
            ParamChange(param="tx0_P_dBm", before=40.0, after=3.0, unit="dBm")
        ],
        expected_kpis=KpiSnapshot(RX_POWER=-80.0),
        constraints_satisfied=True
    )
    
    active_intent = IntentParse(
        target_area="Kadıköy",  # Same area
        target_kpis=["SINR"],
        priority="MEDIUM",
        confidence=0.8
    )
    
    active_plan = OptimizationPlan(
        selected_config_id=44444,
        changes=[
            ParamChange(param="tx0_dAz", before=0.0, after=-5.0, unit="deg")  # Same BS, different param
        ],
        expected_kpis=KpiSnapshot(SINR=12.0),
        constraints_satisfied=True
    )
    
    active_intents = [{"intent": active_intent.model_dump(), "plan": active_plan.model_dump()}]
    
    result_json = detect_conflicts(
        new_intent.model_dump_json(),
        new_plan.model_dump_json(),
        json.dumps(active_intents)
    )
    
    result = json.loads(result_json)
    conflicts = result['conflict_report']['details']
    
    print(f"Total conflicts detected: {len(conflicts)}")
    
    if len(conflicts) > 0:
        print("✅ PASS: BASE_STATION_CONFLICT detected")
        for c in conflicts:
            print(f"   - {c['conflict_type']} ({c['severity']}): {c.get('description', 'N/A')[:80]}")
    else:
        print("❌ FAIL: No conflict detected (should have BASE_STATION_CONFLICT)")
    
    return len(conflicts) > 0

if __name__ == "__main__":
    print("\n" + "🧪" * 30)
    print("CONFLICT DETECTION FIX TESTS")
    print("🧪" * 30)
    
    results = []
    results.append(("Boolean Same Value", test_boolean_same_value()))
    results.append(("Different BS + Different Area", test_different_bs_different_area()))
    results.append(("Time Overlap", test_time_overlap()))
    results.append(("Same BS + Same Area", test_same_bs_same_area()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results, key=lambda x: x[1])
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
