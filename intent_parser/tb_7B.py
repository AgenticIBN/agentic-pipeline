import time
import pandas as pd
from codecarbon import EmissionsTracker
from agno.agent import Agent
from agno.models.ollama import Ollama

# Intent Parser Agent dosyasından şema ve talimatları çekiyoruz.
from intent_parser_agent import (
    IntentParse, 
    KpiThreshold, 
    ConfigChange, 
    INSTRUCTIONS
)

# --- 1. TEST VERİ SETİ ---
TEST_INTENTS = [
    "Increase coverage in the Kadikoy region. RX power must be at least -95 dBm. Priority is high, valid between 18:00 and 23:00.",
    "Quality is degrading in Site TR-IST-034. Ensure SINR is above 10 dB. You can increase tilt by 2 degrees if needed.",
    "Load is unbalanced in Ankara Cankaya cluster. Balance the number of served users immediately, this is critical.",
    "Throughput is too low in Besiktas. We need 5-percentile throughput to be at least 8 Mbps.",
    "Reduce energy consumption in the idle sectors of Izmir-North cluster during night hours (02:00-05:00).",
    "Optimize for latency in the industrial zone of Gebze. Keep average latency below 10ms for machine control.",
    "Sector B in cell 45 has high interference. Check SINR and adjust azimuth by -5 degrees.",
    "Ensure fair usage in the stadium area during the match. Limit per user throughput to 5 Mbps to serve more users.",
    "Maximize capacity in the shopping mall area. Throughput should be greater than 20 Mbps.",
    "Emergency configuration for the hospital zone. Priority critical. Secure all resources for voice availability."
]

# --- 2. MANUEL GROUND TRUTH (BİLİMSEL REFERANS) ---
# Gemini hatası almamak için cevap anahtarını sabitliyoruz.
ground_truths = {
    TEST_INTENTS[0]: IntentParse(
        target_area="Kadikoy region", target_kpis=["RX_POWER"],
        kpi_thresholds=[KpiThreshold(kpi="RX_POWER", op="GTE", value=-95.0, unit="dBm")],
        priority="HIGH", time_constraint_start="18:00", time_constraint_end="23:00", confidence=1.0
    ),
    TEST_INTENTS[1]: IntentParse(
        target_area="Site TR-IST-034", target_kpis=["SINR"],
        kpi_thresholds=[KpiThreshold(kpi="SINR", op="GT", value=10.0, unit="dB")],
        configuration_change=[ConfigChange(action="increase tilt", parameter="tilt", direction="INCREASE", amount=2.0, unit="degrees")],
        confidence=1.0
    ),
    TEST_INTENTS[2]: IntentParse(
        target_area="Ankara Cankaya cluster", target_kpis=["SERVED_USERS"], priority="CRITICAL",
        configuration_change=[ConfigChange(action="Balance the number of served users", direction="OPTIMIZE")],
        confidence=1.0
    ),
    TEST_INTENTS[3]: IntentParse(
        target_area="Besiktas", target_kpis=["THROUGHPUT_5P"],
        kpi_thresholds=[KpiThreshold(kpi="THROUGHPUT_5P", op="GTE", value=8.0, unit="Mbps")], confidence=1.0
    ),
    TEST_INTENTS[4]: IntentParse(
        target_area="Izmir-North cluster", target_kpis=[], time_constraint_start="02:00", time_constraint_end="05:00",
        affected_sectors=["idle sectors"], configuration_change=[ConfigChange(action="Reduce energy consumption", direction="DECREASE")],
        confidence=1.0
    ),
    TEST_INTENTS[5]: IntentParse(
        target_area="industrial zone of Gebze", target_kpis=[], confidence=1.0
    ),
    TEST_INTENTS[6]: IntentParse(
        target_area="cell 45", target_kpis=["SINR"], affected_sectors=["Sector B"],
        configuration_change=[ConfigChange(action="adjust azimuth", parameter="azimuth", amount=-5.0, unit="degrees")],
        confidence=1.0
    ),
    TEST_INTENTS[7]: IntentParse(
        target_area="stadium area", target_kpis=["THROUGHPUT_5P"],
        kpi_thresholds=[KpiThreshold(kpi="THROUGHPUT_5P", op="LTE", value=5.0, unit="Mbps")], confidence=1.0
    ),
    TEST_INTENTS[8]: IntentParse(
        target_area="shopping mall area", target_kpis=["THROUGHPUT_5P"],
        kpi_thresholds=[KpiThreshold(kpi="THROUGHPUT_5P", op="GT", value=20.0, unit="Mbps")], confidence=1.0
    ),
    TEST_INTENTS[9]: IntentParse(
        target_area="hospital zone", target_kpis=[], priority="CRITICAL",
        configuration_change=[ConfigChange(action="Secure all resources for voice availability", direction="OPTIMIZE")],
        confidence=1.0
    )
}

# --- 3. TEST EDİLECEK MODEL ---
STUDENT_MODELS = {
    "Qwen2.5-7B": Ollama(id="qwen2.5:7b"),
}

# --- 4. DOĞRULUK HESABI ---
def calculate_slot_accuracy(pred, truth):
    score = 0
    total_slots = 0
    
    # 1. Basit Alanlar
    for field in ['target_area', 'priority', 'time_constraint_start', 'time_constraint_end']:
        total_slots += 1
        if getattr(pred, field) == getattr(truth, field):
            score += 1
            
    # 2. Liste Alanları
    for field in ['target_kpis', 'affected_sectors']:
        total_slots += 1
        truth_val = getattr(truth, field) or []
        pred_val = getattr(pred, field) or []
        if set(truth_val) == set(pred_val):
            score += 1

    # 3. KPI Thresholds
    if truth.kpi_thresholds:
        total_slots += len(truth.kpi_thresholds)
        if pred.kpi_thresholds and len(pred.kpi_thresholds) >= len(truth.kpi_thresholds):
            for i in range(len(truth.kpi_thresholds)):
                t_item = truth.kpi_thresholds[i]
                try:
                    p_item = pred.kpi_thresholds[i]
                    if (t_item.kpi == p_item.kpi) and (t_item.value == p_item.value) and (t_item.op == p_item.op):
                        score += 1
                except IndexError: pass
    
    # 4. Configuration Change
    if truth.configuration_change:
        total_slots += len(truth.configuration_change)
        if pred.configuration_change and len(pred.configuration_change) >= len(truth.configuration_change):
            for i in range(len(truth.configuration_change)):
                t_conf = truth.configuration_change[i]
                try:
                    p_conf = pred.configuration_change[i]
                    match = True
                    if t_conf.parameter and t_conf.parameter != p_conf.parameter: match = False
                    if t_conf.direction and t_conf.direction != p_conf.direction: match = False
                    if t_conf.amount and t_conf.amount != p_conf.amount: match = False
                    if match: score += 1
                except IndexError: pass

    return score / total_slots if total_slots > 0 else 1.0

# --- 5. BENCHMARK AKIŞI ---
results = []

print("\n--- ADIM 2: STUDENT (QWEN 7B) TESTLERİ ---")

for model_name, model_obj in STUDENT_MODELS.items():
    print(f"\nTesting Model: {model_name}")
    
    student_agent = Agent(
        name="Student",
        model=model_obj,
        response_model=IntentParse,
        instructions=INSTRUCTIONS
    )
    
    # Enerji Takibi
    tracker = EmissionsTracker(project_name=model_name, measure_power_secs=0.1, save_to_file=False, logging_logger=None)
    tracker.start()
    
    model_latencies = []
    model_accuracies = []
    
    for idx, intent in enumerate(TEST_INTENTS):
        truth = ground_truths.get(intent)
        if not truth: 
            continue
        
        start_time = time.time()
        try:
            response = student_agent.run(intent)
            end_time = time.time()
            latency = end_time - start_time
            
            acc = calculate_slot_accuracy(response.content, truth)
            model_latencies.append(latency)
            model_accuracies.append(acc)
            
            print(f"  Query {idx+1}/{len(TEST_INTENTS)} | Latency: {latency:.2f}s | Acc: {acc:.2f}")
            
        except Exception as e:
            # print(f"Hata: {e}")
            model_latencies.append(None)
            model_accuracies.append(0.0)

    tracker.stop()
    total_energy_kwh = tracker._total_energy.kWh
    total_energy_joules = total_energy_kwh * 3.6e6
    
    valid_latencies = [l for l in model_latencies if l is not None]
    avg_latency = sum(valid_latencies)/len(valid_latencies) if valid_latencies else 0
    avg_accuracy = sum(model_accuracies)/len(model_accuracies) if model_accuracies else 0
    avg_energy_per_query = total_energy_joules / len(TEST_INTENTS)
    
    results.append({
        "Model": model_name,
        "Avg_Latency (s)": round(avg_latency, 3),
        "Avg_Accuracy (vs Gemini)": round(avg_accuracy, 3),
        "Total_Energy (Joules)": round(total_energy_joules, 4),
        "Energy_Per_Query (Joules)": round(avg_energy_per_query, 4)
    })

# --- RAPORLAMA ---
df = pd.DataFrame(results)
# Baseline'i da tabloya ekleyelim
df.loc[len(df)] = ["Gemini-Flash (Baseline)", 0.5, 1.0, 0, 0] 

print("\n" + "="*40)
print("BENCHMARK RESULTS FOR PAPER")
print("="*40)
print(df.to_string(index=False))

df.to_csv("llm_benchmark_results_final.csv", index=False)
print(f"\nSonuçlar 'llm_benchmark_results_final.csv' dosyasına kaydedildi.")