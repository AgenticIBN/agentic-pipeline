import json
import sys
import os
from datetime import datetime

# Dosya yapısı gereği importları yapıyoruz
try:
    from conflict_detector_agent import detect_conflicts
    # optimization_agent.py dosyasındaki modelleri kullanıyoruz
except ImportError as e:
    print("ERROR: 'conflict_detector_agent.py' not found.")
    sys.exit(1)

def print_separator(title):
    print("\n" + "="*60)
    print(f"TEST SCENARIO: {title}")
    print("="*60)

def run_test_scenario(scenario_name, active_data, new_intent, new_plan):
    print_separator(scenario_name)
    
    # Verileri JSON string formatına çevir (Agent input formatı)
    active_json = json.dumps(active_data)
    new_intent_json = json.dumps(new_intent)
    new_plan_json = json.dumps(new_plan)
    
    print(f"Active Intents: {len(active_data)}")
    print(f"New Intent Target: {new_intent['target_area']} -> {new_intent['target_kpis']}")
    
    # Çakışma dedektörünü çalıştır
    try:
        result_json = detect_conflicts(new_intent_json, new_plan_json, active_json)
        result = json.loads(result_json)
        
        # --- EKRANA YAZDIRMA ---
        conflict_report = result["conflict_report"]
        proposals = result["proposals"]
        
        print("\n--- CONFLICT REPORT (Input for Meta-Agent) ---")
        if conflict_report["is_conflicted"]:
            print(f"❌ CONFLICT DETECTED! ({conflict_report['conflict_summary']})")
            for detail in conflict_report["details"]:
                print(f"   ► TYPE: {detail['conflict_type']}")
                print(f"   ► SEVERITY: {detail['severity']}")
                print(f"   ► PARAMETER: {detail['conflicting_param']}")
                print(f"   ► DESCRIPTION: {detail['description']}")
                print("   ---")
        else:
            print("✅ NO CONFLICT. (Safe to execute)")
            
        print(f"\n--- DATA PACKAGE STATS ---")
        print(f"Total Proposals Packed: {len(proposals)}")
        
        # --- DOSYAYA KAYDETME (BURASI EKLENDİ) ---
        # Dosya ismi oluştur: conflict_report_SenaryoAdi_Tarih.json
        safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conflict_report_{safe_name}_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f"\n💾 SAVED TO: {filename}")
            
    except Exception as e:
        print(f"💥 ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# SCENARIO 1: CRITICAL CONFLICT
# ==========================================
def test_critical_conflict():
    active_item = {
        "intent": {
            "target_area": "Kadikoy",
            "target_kpis": ["RX_COVERAGE_RATIO"],
            "kpi_thresholds": [],
            "priority": "HIGH",
            "confidence": 1.0,
            "affected_sectors": ["tx0"],
            "configuration_change": [] 
        },
        "plan": {
            "selected_config_id": 101,
            "changes": [
                {"param": "tx0_P_dBm", "before": 40.0, "after": 3.0, "unit": "dBm"}
            ],
            "expected_kpis": {},
            "constraints_satisfied": True
        }
    }
    
    new_intent = {
        "target_area": "Kadikoy",
        "target_kpis": ["RX_POWER"],
        "kpi_thresholds": [],
        "priority": "MEDIUM",
        "confidence": 0.9,
        "affected_sectors": ["tx0"],
        "configuration_change": []
    }
    
    new_plan = {
        "selected_config_id": 202,
        "changes": [
            {"param": "tx0_P_dBm", "before": 40.0, "after": -5.0, "unit": "dBm"}
        ],
        "expected_kpis": {},
        "constraints_satisfied": True,
        "current_config_id": 101
    }
    
    run_test_scenario("1_CRITICAL_CONFLICT", [active_item], new_intent, new_plan)

# ==========================================
# SCENARIO 2: SPATIAL OVERLAP
# ==========================================
def test_spatial_overlap():
    active_item = {
        "intent": {
            "target_area": "Besiktas",
            "target_kpis": ["SINR"],
            "kpi_thresholds": [],
            "priority": "MEDIUM",
            "confidence": 1.0,
            "affected_sectors": ["tx1"],
            "configuration_change": []
        },
        "plan": {
            "selected_config_id": 303,
            "changes": [
                {"param": "tx1_dEl", "before": -2.0, "after": 2.0, "unit": "deg"}
            ],
            "expected_kpis": {},
            "constraints_satisfied": True
        }
    }
    
    new_intent = {
        "target_area": "Besiktas",
        "target_kpis": ["THROUGHPUT_5P"],
        "kpi_thresholds": [],
        "priority": "LOW",
        "confidence": 0.8,
        "affected_sectors": ["tx1"],
        "configuration_change": []
    }
    
    new_plan = {
        "selected_config_id": 404,
        "changes": [
            {"param": "tx1_dAz", "before": 0.0, "after": 10.0, "unit": "deg"}
        ],
        "expected_kpis": {},
        "constraints_satisfied": True,
        "current_config_id": 303
    }
    
    run_test_scenario("2_SPATIAL_OVERLAP", [active_item], new_intent, new_plan)

# ==========================================
# SCENARIO 3: RESOURCE CONTENTION
# ==========================================
def test_resource_contention():
    active_item = {
        "intent": {
            "target_area": "Uskudar",
            "target_kpis": ["RX_POWER"],
            "kpi_thresholds": [],
            "priority": "HIGH",
            "confidence": 1.0,
            "affected_sectors": ["tx2"],
            "configuration_change": []
        },
        "plan": {
            "selected_config_id": 505,
            "changes": [
                {"param": "tx2_P_dBm", "before": 40.0, "after": 2.0, "unit": "dBm"}
            ],
            "expected_kpis": {},
            "constraints_satisfied": True
        }
    }
    
    new_intent = {
        "target_area": "Uskudar",
        "target_kpis": ["SINR"],
        "kpi_thresholds": [],
        "priority": "HIGH",
        "confidence": 0.9,
        "affected_sectors": ["tx2"],
        "configuration_change": []
    }
    
    new_plan = {
        "selected_config_id": 606,
        "changes": [
            {"param": "tx2_P_dBm", "before": 40.0, "after": 4.0, "unit": "dBm"}
        ],
        "expected_kpis": {},
        "constraints_satisfied": True,
        "current_config_id": 505
    }
    
    run_test_scenario("3_RESOURCE_CONTENTION", [active_item], new_intent, new_plan)

if __name__ == "__main__":
    test_critical_conflict()
    test_spatial_overlap()
    test_resource_contention()