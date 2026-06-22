import time
import pandas as pd
import json
import os
from codecarbon import EmissionsTracker
from typing import List
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.ollama import Ollama

# Intent Parser Agent dosyasından (senin oluşturduğun) şema ve talimatları çekiyoruz
from intent_parser_agent import IntentParse, INSTRUCTIONS


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

# --- 2. MODELLERİN TANIMLANMASI ---
MODELS = {
    "Gemini-Flash (Baseline)": Gemini(id="gemini-2.5-flash"),
    #"Qwen2.5-0.5B": Ollama(id="qwen2.5:0.5b"),
    #"Qwen2.5-1.5B": Ollama(id="qwen2.5:1.5b"),
    #"Llama3.2-1B": Ollama(id="llama3.2:1b"),
    #"Gemma2-2B": Ollama(id="gemma2:2b"),
    "Llama3.2-3B": Ollama(id="llama3.2:3b"),
}

# --- 3. GÜNCELLENMİŞ DOĞRULUK HESABI ---
def calculate_slot_accuracy(pred, truth):
    """
    Yeni IntentParse şemasına göre detaylı doğruluk hesabı yapar.
    Basit alanlar, listeler ve iç içe objeler (Thresholds, ConfigChange) kontrol edilir.
    """
    score = 0
    total_slots = 0
    
    # 1. Basit Alanlar (String / Enum)
    simple_fields = ['target_area', 'priority', 'time_constraint_start', 'time_constraint_end']
    for field in simple_fields:
        total_slots += 1
        # None vs None eşitliği veya değer eşitliği
        if getattr(pred, field) == getattr(truth, field):
            score += 1
            
    # 2. Liste Alanları (Sırasız eşitlik kontrolü - Set ile)
    # target_kpis ve affected_sectors liste olarak gelir
    list_fields = ['target_kpis', 'affected_sectors']
    for field in list_fields:
        total_slots += 1
        truth_val = getattr(truth, field) or []
        pred_val = getattr(pred, field) or []
        # Listelerin içeriği aynı mı? (Sıra önemsiz)
        if set(truth_val) == set(pred_val):
            score += 1

    # 3. Karmaşık Obje: KPI Thresholds
    # Burası önemli: Hem sayı tutmalı hem de KPI tipi.
    if truth.kpi_thresholds:
        total_slots += len(truth.kpi_thresholds)
        # Eğer tahmin edilen liste uzunluğu en az gerçek kadar ise kontrol et
        if len(pred.kpi_thresholds) >= len(truth.kpi_thresholds):
            for i in range(len(truth.kpi_thresholds)):
                t_item = truth.kpi_thresholds[i]
                # Basit bir eşleştirme mantığı: Sıradaki eleman tutuyor mu?
                # Daha gelişmiş versiyonda KPI adına göre arama yapılabilir ama bu yeterli.
                try:
                    p_item = pred.kpi_thresholds[i]
                    # KPI tipi ve Hedef Değer tutuyor mu?
                    if (t_item.kpi == p_item.kpi) and (t_item.value == p_item.value) and (t_item.op == p_item.op):
                        score += 1
                except IndexError:
                    pass
    
    # 4. Karmaşık Obje: Configuration Change
    # Yeni şemadaki "Action", "Amount" vb. kontrolü
    if truth.configuration_change:
        total_slots += len(truth.configuration_change)
        if len(pred.configuration_change) >= len(truth.configuration_change):
            for i in range(len(truth.configuration_change)):
                t_conf = truth.configuration_change[i]
                try:
                    p_conf = pred.configuration_change[i]
                    # Action stringi veya parametre tutuyor mu?
                    # Action serbest metin olduğu için birebir tutmayabilir, parametre ve direction'a bakalım.
                    match = True
                    if t_conf.parameter and t_conf.parameter != p_conf.parameter: match = False
                    if t_conf.direction and t_conf.direction != p_conf.direction: match = False
                    if t_conf.amount and t_conf.amount != p_conf.amount: match = False
                    
                    if match:
                        score += 1
                except IndexError:
                    pass

    return score / total_slots if total_slots > 0 else 1.0

# --- 4. BENCHMARK DÖNGÜSÜ ---

results = []
ground_truths = {}

print("--- 1. ADIM: GROUND TRUTH OLUŞTURULUYOR (GEMINI) ---")
teacher_agent = Agent(
    name="Teacher",
    model=MODELS["Gemini-Flash (Baseline)"],
    response_model=IntentParse,
    instructions=INSTRUCTIONS
)

for intent in TEST_INTENTS:
    print(f"Processing GT for: {intent[:40]}...")
    try:
        response = teacher_agent.run(intent)
        ground_truths[intent] = response.content
        # Debug için GT'yi görelim (İstersen yorum satırı yapabilirsin)
        # print(f"GT: {response.content.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"Error generating GT: {e}")
        ground_truths[intent] = None

print("\n--- 2. ADIM: MİNİ-LLM TESTLERİ VE ENERJİ ÖLÇÜMÜ ---")

for model_name, model_obj in MODELS.items():
    # Baseline'ı (Gemini) tekrar ölçmüyoruz, sadece local modeller
    if "Gemini" in model_name: continue 
    
    print(f"\nTesting Model: {model_name}")
    
    student_agent = Agent(
        name="Student",
        model=model_obj,
        response_model=IntentParse,
        instructions=INSTRUCTIONS
    )
    
    # CodeCarbon ile enerji takibi başlat
    tracker = EmissionsTracker(project_name=model_name, measure_power_secs=0.1, save_to_file=False, logging_logger=None)
    tracker.start()
    
    model_latencies = []
    model_accuracies = []
    
    for idx, intent in enumerate(TEST_INTENTS):
        if ground_truths[intent] is None: continue
        
        start_time = time.time()
        try:
            # --- INFERENCE ---
            response = student_agent.run(intent)
            # -----------------
            
            end_time = time.time()
            latency = end_time - start_time
            model_latencies.append(latency)
            
            # Güncellenmiş Accuracy fonksiyonunu kullan
            acc = calculate_slot_accuracy(response.content, ground_truths[intent])
            model_accuracies.append(acc)
            
            print(f"  Query {idx+1} | Latency: {latency:.2f}s | Acc: {acc:.2f}")
            
        except Exception as e:
            print(f"  Error on query {idx}: {e}")
            model_latencies.append(None)
            model_accuracies.append(0.0)

    # Enerji ölçümünü bitir
    tracker.stop()
    total_energy_kwh = tracker._total_energy.kWh
    total_energy_joules = total_energy_kwh * 3.6e6  # kWh -> Joule
    
    # İstatistikler
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

# --- 5. RAPORLAMA ---
df = pd.DataFrame(results)

# Gemini referans satırını ekle (Enerji 0 kabul ediyoruz cloud olduğu için)
df.loc[len(df)] = ["Gemini-Flash (Baseline)", 0.5, 1.0, 0, 0] 

# Sonuçları göster
print("\n" + "="*40)
print("BENCHMARK RESULTS FOR PAPER")
print("="*40)
print(df.to_string(index=False))

# CSV'ye kaydet
csv_filename = "llm_benchmark_results_v2.csv"
df.to_csv(csv_filename, index=False)
print(f"\nSonuçlar '{csv_filename}' dosyasına kaydedildi.")