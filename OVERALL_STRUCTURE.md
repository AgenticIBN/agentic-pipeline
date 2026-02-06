# 6G Ağ Optimizasyon Sistemi - Genel Yapı ve Dokümantasyon

## 📋 İçindekiler
1. [Sistem Mimarisi](#sistem-mimarisi)
2. [Agent Yapısı (Agno Framework)](#agent-yapısı)
3. [Surrogate Model Eğitimi](#surrogate-model-eğitimi)
4. [Sistem Parametreleri ve KPI'lar](#sistem-parametreleri-ve-kpilar)
5. [Workflow Pipeline](#workflow-pipeline)
6. [Kullanımda Olan Dosyalar](#kullanımda-olan-dosyalar)
7. [Sistemi Çalıştırma](#sistemi-çalıştırma)

---

## 🏗️ Sistem Mimarisi

Bu sistem, 6G ağ optimizasyonu için çoklu intent'leri yöneten, çakışmaları tespit eden ve çözen ajansal (agent-based) bir pipeline'dır. Tüm bileşenler **Agno framework** kullanılarak agent yapısında oluşturulmuştur.

### Ana Bileşenler:

```
┌─────────────────────────────────────────────────────────────┐
│                    AGNO WORKFLOW PIPELINE                   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Step 1:      │    │ Step 2:      │    │ Step 3:      │
│ Intent       │───▶│ Optimization │───▶│ Conflict     │
│ Parser       │    │ Agent        │    │ Detection    │
└──────────────┘    └──────────────┘    └──────────────┘
                                                │
                        ┌───────────────────────┘
                        ▼
                ┌──────────────┐
                │ Step 4:      │
                │ Conflict     │
                │ Resolution   │
                └──────────────┘
                        │
                        ▼
                ┌──────────────┐
                │ Step 5:      │
                │ Finalization │
                └──────────────┘
```

---

## 🤖 Agent Yapısı

Sistemdeki tüm kritik bileşenler **Agno Agent** yapısındadır ve **Groq LLM (llama-3.3-70b-versatile)** kullanarak akıl yürütme (reasoning) sağlar.

### 1. Intent Parser Agent
**Dosya:** `intent_parser/intent_parser_agent.py`

**Görev:** Doğal dildeki intent'i yapılandırılmış formata çevirir.

**Çıktı:**
```python
{
    "target_area": "Kartal",
    "target_kpis": ["RX_POWER"],
    "kpi_thresholds": [
        {"kpi": "RX_POWER", "operator": "GTE", "value": -52, "unit": "dBm"}
    ],
    "time_constraint_start": "2026-02-06T13:00:00",
    "time_constraint_end": "2026-02-06T15:00:00",
    "priority": "HIGH",
    "confidence": 0.95
}
```

**Özellikler:**
- Türkçe ve İngilizce destekler
- Priority seviyeleri: LOW, MEDIUM, HIGH, CRITICAL
- Operatörler: GTE (>=), LTE (<=), GT (>), LT (<), DELTA_UP, DELTA_DOWN
- LLM reasoning ile belirsizlikleri açıklar

---

### 2. Optimization Agent
**Dosya:** `agents/optimization_agent.py`

**Görev:** Surrogate model kullanarak base station konfigürasyonunu optimize eder.

**İşleyiş:**
1. Mevcut sistem konfigürasyonunu alır (default: tüm TX'ler 30dBm, 0° açı)
2. Intent'teki KPI hedeflerine göre seed konfigürasyonlar üretir
3. Local search ile en iyi konfigürasyonu bulur
4. LLM reasoning ile optimizasyon kararlarını açıklar

**Çıktı Yapısı:**
```python
{
    "selected_config_id": 20260206224109,
    "changes": [
        {"param": "tx0_P_dBm", "before": 30.0, "change": 3.0, "unit": "dBm"},
        {"param": "tx0_dAz", "before": 0.0, "change": -10.0, "unit": "deg"},
        {"param": "tx1_on", "before": True, "change": False, "unit": "bool"}
    ],
    "expected_kpis": {
        "RX_POWER": -51.84,
        "SINR": 15.25,
        "THROUGHPUT_5P": 3.09,
        "LOAD_IMBALANCE": 0.89,
        "RX_COVERAGE_RATIO": 0.52
    },
    "constraints_satisfied": True
}
```

**Core Logic:** `optimization_agent_v2.py` dosyasındaki `OptimizationAgent` sınıfını wrapper'lar.

---

### 3. Conflict Detector Agent
**Dosya:** `agents/conflict_detector_agent.py`

**Görev:** Yeni optimization sonucunu aktif intent'lerle karşılaştırıp çakışmaları tespit eder.

**Çakışma Tipleri:**
- **PARAMETER_CONFLICT:** Aynı parametre, zıt yönler (birisi artır, diğeri azalt)
- **RESOURCE_CONTENTION:** Aynı parametre, aynı yön ama farklı büyüklük
- **BASE_STATION_CONFLICT:** Aynı TX, farklı parametreler
- **BOOLEAN_CONFLICT:** ON/OFF çakışması

**Severity Seviyeleri:** LOW, MEDIUM, HIGH, CRITICAL

**Örnek Çıktı:**
```python
{
    "is_conflicted": True,
    "num_conflicts": 3,
    "conflict_summary": "3 conflicts detected between new result and 1 active result(s)",
    "details": [
        {
            "conflict_type": "PARAMETER_CONFLICT",
            "severity": "HIGH",
            "parameter": "tx0_P_dBm",
            "intent1_id": "20260206224109",
            "intent2_id": "20260206224234",
            "intent1_change": 3.0,
            "intent2_change": -5.0,
            "description": "TX0 power: one wants +3.0dBm, other wants -5.0dBm"
        }
    ],
    "conflicting_result_ids": ["20260206224109"]
}
```

**Core Logic:** `conflict_detector_agent.py` dosyasındaki core fonksiyonu wrapper'lar.

---

### 4. Priority Resolution Agent
**Dosya:** `agents/priority_resolution_agent.py`

**Görev:** Çakışmaları priority'ye göre çözer - en yüksek priority kazanır.

**Priority Sıralaması:** CRITICAL > HIGH > MEDIUM > LOW

**Çıktı:**
```python
{
    "selected_config_id": "20260206224234",
    "selected_intent_priority": "CRITICAL",
    "rejected_intent_ids": ["20260206224109"],
    "reasoning": "Selected CRITICAL priority intent over HIGH priority. 
                  CRITICAL intents take precedence in conflict resolution."
}
```

---

### 5. Weighted Merge Agent
**Dosya:** `agents/weighted_merge_agent.py`

**Görev:** Çakışan intent'leri priority ağırlıklarına göre birleştirir.

**Ağırlık Sistemi:**
- CRITICAL: 4.0
- HIGH: 3.0
- MEDIUM: 2.0
- LOW: 1.0

**İşleyiş:**
- Her parametrenin değişimi ağırlıklı ortalama ile hesaplanır
- Örnek: HIGH (+3dBm) ve CRITICAL (-5dBm) → ağırlıklı merge

**Çıktı:**
```python
{
    "merged_config_id": "MERGED_20260206224300",
    "merged_changes": [
        {"param": "tx0_P_dBm", "before": 30.0, "change": -2.14, "unit": "dBm"}
    ],
    "source_configs": ["20260206224109", "20260206224234"],
    "reasoning": "Merged 2 conflicting intents using weighted average..."
}
```

---

## 🎓 Surrogate Model Eğitimi

Optimization agent'in temelinde **LightGBM** tabanlı bir surrogate model bulunur. Bu model, base station konfigürasyonundan KPI'ları tahmin eder.

### Model Eğitim Süreci

**Script:** `train_surrogate_v2.py`

#### 1. Veri Hazırlama

Eğitim verisi bir simulator'dan gelir ve şu bilgileri içerir:

**Giriş Özellikleri (Features):**
```python
feature_cols = [
    # Senaryo parametreleri
    "user_set_id",              # Kullanıcı dağılımı senaryosu
    "K_users",                  # Toplam kullanıcı sayısı
    "rx_power_thr_dBm",         # RX power threshold
    "total_tx_power_watt",      # Toplam TX gücü (watt)
    
    # Base Station TX0 parametreleri
    "tx0_on",                   # TX0 açık/kapalı (boolean)
    "tx0_P_dBm",                # TX0 transmit gücü (dBm)
    "tx0_dAz",                  # TX0 azimuth açı değişimi (derece)
    "tx0_dEl",                  # TX0 elevation açı değişimi (derece)
    
    # TX1, TX2, TX3 için aynı parametreler
    "tx1_on", "tx1_P_dBm", "tx1_dAz", "tx1_dEl",
    "tx2_on", "tx2_P_dBm", "tx2_dAz", "tx2_dEl",
    "tx3_on", "tx3_P_dBm", "tx3_dAz", "tx3_dEl"
]
```

**Çıktı Hedefleri (Targets - KPI'lar):**
```python
target_cols = [
    "Prx_p5_dBm",               # 5th percentile alınan güç (RX_POWER)
    "SINR_p5_dB",               # 5th percentile SINR
    "rx_power_coverage_ratio",  # RX power kapsama oranı
    "tx0_served_pct",           # TX0 tarafından servis edilen kullanıcı %'si
    "tx1_served_pct",           # TX1 tarafından servis edilen kullanıcı %'si
    "tx2_served_pct",           # TX2 tarafından servis edilen kullanıcı %'si
    "tx3_served_pct"            # TX3 tarafından servis edilen kullanıcı %'si
]
```

#### 2. Feature Engineering

Model performansını artırmak için ek özellikler türetilir:

```python
# Güç istatistikleri
"avg_power"           # Aktif TX'lerin ortalama gücü
"power_std"           # Güç varyansı (heterojenlik)

# Açı istatistikleri
"avg_azimuth"         # Ortalama azimuth
"avg_elevation"       # Ortalama elevation
"azimuth_range"       # Azimuth değişim aralığı
"elevation_range"     # Elevation değişim aralığı

# Etkileşim özellikleri (her TX için)
"tx0_P_x_Az"         # Güç × Azimuth
"tx0_P_x_El"         # Güç × Elevation
"tx0_Az_x_El"        # Azimuth × Elevation

# Sistem özellikleri
"n_active_tx"        # Aktif TX sayısı
"azimuth_uniformity" # Açı dağılımının düzgünlüğü
```

#### 3. Model Eğitimi

**Kullanılan Algoritma:** LightGBM (Gradient Boosting)

**Model Parametreleri:**
```python
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'max_depth': 15,
    'verbose': -1
}
```

**Her KPI için ayrı model:**
- Model_RX_POWER → Prx_p5_dBm tahmin eder
- Model_SINR → SINR_p5_dB tahmin eder
- Model_COVERAGE → rx_power_coverage_ratio tahmin eder
- LOAD_IMBALANCE → tx_served_pct'lerden hesaplanır
- THROUGHPUT_5P → SINR'den Shannon formülü ile hesaplanır

#### 4. Eğitimi Çalıştırma

```bash

# Tüm veriyi kullan (çok büyük dataset için)
python3 train_surrogate_v2.py
```

**Parametre Açıklaması:**
- `--sample-size`: Eğitimde kullanılacak örnek sayısı
- Belirtilmezse: Tüm veri kullanılır
- CSV veya Parquet formatı desteklenir

**Model Kaydedilme:**
Eğitilen model `models/surrogate.joblib` olarak kaydedilir ve optimization agent tarafından yüklenir.

#### 5. Model Performansı

Eğitim sonunda her KPI için performans metrikleri gösterilir:

```
✅ Model trained successfully!

Performance Metrics:
--------------------
KPI: Prx_p5_dBm
  Train RMSE: 1.23 dBm
  Test RMSE:  1.45 dBm
  R² Score:   0.94

KPI: SINR_p5_dB
  Train RMSE: 0.87 dB
  Test RMSE:  1.02 dB
  R² Score:   0.91
...
```

---

## 📊 Sistem Parametreleri ve KPI'lar

### Base Station Kontrol Parametreleri

Her base station (TX0, TX1, TX2, TX3) için optimize edilen parametreler:

| Parametre | Açıklama | Aralık | Birim |
|-----------|----------|--------|-------|
| `tx{i}_on` | TX açık/kapalı | True/False | boolean |
| `tx{i}_P_dBm` | Transmit gücü | 0 - 50 | dBm |
| `tx{i}_dAz` | Azimuth açı değişimi | -30 - +30 | derece |
| `tx{i}_dEl` | Elevation açı değişimi | -15 - +15 | derece |

**Toplam:** 16 kontrol parametresi (4 TX × 4 parametre)

### Anahtar Performans Göstergeleri (KPI'lar)

| KPI | Açıklama | İyi Değer | Birim |
|-----|----------|-----------|-------|
| **RX_POWER** | Alınan sinyal gücü (5th percentile) | > -70 dBm | dBm |
| **SINR** | Sinyal-Gürültü Oranı (5th percentile) | > 10 dB | dB |
| **THROUGHPUT_5P** | Veri hızı (5th percentile) | Yüksek | Mbps |
| **RX_COVERAGE_RATIO** | Kapsama oranı | > 0.8 | 0-1 |
| **LOAD_IMBALANCE** | Yük dengesizliği | Düşük | 0-100 |

**KPI Hesaplama:**
- **RX_POWER, SINR, COVERAGE:** Surrogate model tarafından doğrudan tahmin edilir
- **THROUGHPUT:** Shannon formülü ile SINR'den hesaplanır:
  ```python
  throughput = BW * log2(1 + 10^(SINR/10))
  ```
- **LOAD_IMBALANCE:** TX'ler arası servis edilen kullanıcı % farkından:
  ```python
  load_imbalance = std([tx0_served_pct, tx1_served_pct, tx2_served_pct, tx3_served_pct])
  ```

---

## 🔄 Workflow Pipeline

### Pipeline Adımları

**Ana Dosya:** `agno_workflow_pipeline.py`

```python
optimization_workflow = Workflow(
    id="6g_network_optimization_pipeline",
    name="6G_Network_Optimization_Pipeline",
    steps=[
        Step(name="parse_intent", executor=step_1_parse_intent),
        Step(name="optimize_configuration", executor=step_2_optimize_configuration),
        Step(name="detect_conflicts", executor=step_3_detect_conflicts),
        Step(name="resolve_conflicts", executor=step_4_resolve_conflicts),
        Step(name="finalize_configuration", executor=step_5_finalize_configuration)
    ]
)
```

### Adım Detayları

#### Step 1: Parse Intent
- Intent Parser Agent çağrılır
- Doğal dil → Yapılandırılmış intent
- Current config: `active_intents_workflow.json`'dan yüklenir

#### Step 2: Optimize Configuration
- Optimization Agent çağrılır
- Mevcut configden başlayarak optimal config bulunur
- Surrogate model ile KPI'lar tahmin edilir
- **Önemli:** Her intent, bir önceki intent'in final config'ini current config olarak kullanır

#### Step 3: Detect Conflicts
- Conflict Detector Agent çağrılır
- Yeni result vs. aktif intent'ler karşılaştırılır
- Çakışmalar tespit edilir ve kategorize edilir

#### Step 4: Resolve Conflicts (Koşullu)
- Çakışma yoksa: Skip
- Çakışma varsa:
  - **PRIORITY stratejisi:** Priority Resolution Agent
  - **WEIGHTED_MERGE stratejisi:** Weighted Merge Agent

#### Step 5: Finalize Configuration
- Final config belirlenir (yeni veya çözümlenmiş)
- `active_intents_workflow.json` güncellenir
- Sonuçlar kaydedilir

### State Management

**Dosya:** `active_intents_workflow.json`

Sistemdeki tüm aktif intent'lerin final konfigürasyonlarını saklar:

```json
{
  "intents": [
    {
      "config_id": "20260206224109",
      "priority": "HIGH",
      "target_area": "Kartal",
      "time_constraint_start": "2026-02-06T13:00:00",
      "time_constraint_end": "2026-02-06T15:00:00",
      "final_config": {
        "tx0_on": true,
        "tx0_P_dBm": 33.0,
        "tx0_dAz": -10.0,
        "tx0_dEl": -2.0,
        ...
      },
      "expected_kpis": {
        "RX_POWER": -51.84,
        "SINR": 15.25,
        ...
      }
    }
  ]
}
```

**Fonksiyonlar:**
- `get_current_system_config()`: Son finalized config'i döner (yoksa default)
- `extract_final_config_from_result()`: Optimization/resolution result'tan final config çıkarır
- `save_finalized_result()`: Yeni intent'i state'e ekler

---

## 📁 Kullanımda Olan Dosyalar

### 🟢 Aktif Agent Dosyaları

| Dosya | Görev | Kullanım |
|-------|-------|----------|
| `agents/optimization_agent.py` | Optimization Agent (Agno) | Step 2 |
| `agents/conflict_detector_agent.py` | Conflict Detection Agent (Agno) | Step 3 |
| `agents/priority_resolution_agent.py` | Priority Resolution Agent (Agno) | Step 4 (PRIORITY) |
| `agents/weighted_merge_agent.py` | Weighted Merge Agent (Agno) | Step 4 (WEIGHTED_MERGE) |
| `intent_parser/intent_parser_agent.py` | Intent Parser Agent (Agno) | Step 1 |

### 🟢 Core Logic Dosyaları

| Dosya | Görev | Kullanım |
|-------|-------|----------|
| `optimization_agent_v2.py` | Core optimization algorithm + surrogate | agents/optimization_agent.py tarafından import edilir |
| `conflict_detector_agent.py` (kök) | Core conflict detection logic | agents/conflict_detector_agent.py tarafından import edilir |
| `priority_based_resolution_agent.py` (kök) | Core priority resolution logic | agents/priority_resolution_agent.py tarafından import edilir |
| `weighted_merge_resolution_agent.py` (kök) | Core weighted merge logic | agents/weighted_merge_agent.py tarafından import edilir |

### 🟢 Pipeline ve State

| Dosya | Görev |
|-------|-------|
| `agno_workflow_pipeline.py` | Ana workflow orchestrator |
| `workflow_steps.py` | Step executor fonksiyonları |
| `active_intents_workflow.json` | Aktif intent state management |
| `models/surrogate.joblib` | Eğitilmiş surrogate model |

### 🟢 Training ve Utilities

| Dosya | Görev |
|-------|-------|
| `train_surrogate_v2.py` | Surrogate model eğitim script'i |
| `server.py` | Agno playground server |
| `start_playground.py` | Playground launcher |

### 🟢 Test Dosyaları

| Dosya | Görev |
|-------|-------|
| `tests/test_workflow_simple.py` | Basit workflow test |
| `tests/test_conflict_resolution.py` | Conflict resolution test |
| `tests/test_optimization_bench.py` | Optimization benchmark |

### 🔴 Kullanılmayan Eski Dosyalar (Silinebilir)

| Dosya | Durum |
|-------|-------|
| `agentos_optimization.py` | ❌ Eski non-agent versiyon |
| `run_full_pipeline.py` | ❌ Eski pipeline |
| `test_conflict_detection.py` | ❌ Eski test |
| `test_power_optimization.py` | ❌ Eski test |
| `test_workflow_pipeline.py` | ❌ Eski test |

---

## 🚀 Sistemi Çalıştırma

### 1. Ortam Hazırlığı

```bash
# Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# Bağımlılıkları kur
pip install agno groq python-dotenv pydantic numpy joblib lightgbm

# .env dosyasını oluştur
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### 2. Surrogate Model Eğitimi (İlk Kurulum)

```bash
# Modeli eğit (simulator verisi gerekli)
python3 train_surrogate_v2.py

# Model models/surrogate.joblib olarak kaydedilir
```

**Not:** Eğer simulator veriniz yoksa, mevcut `models/surrogate.joblib` dosyasını kullanabilirsiniz.

### 3. Agno Playground Modu (İnteraktif) - henüz implement edilemedi

```bash
# Playground server'ı başlat
python3 agno_workflow_pipeline.py --playground

# Veya start_playground.py kullan
python3 start_playground.py
```

**Tarayıcıda:** http://localhost:7777
- Workflows sekmesine git
- "6G Network Optimization Pipeline" seç
- UI üzerinden intent gir ve çalıştır

### 4. Komut Satırı Modu (CLI)

#### Tek Intent (Çakışma Yok)

```bash
python3 agno_workflow_pipeline.py \
  --intent "Kartal bölgesinde etkinlik var. RX gücü en az -52 dBm olsun. Priority HIGH" \
  --strategy PRIORITY
```

#### İkinci Intent (Çakışma Tespiti)

```bash
# Önce bir intent çalıştır (yukarıdaki)
# Sonra çakışan bir intent çalıştır:

python3 agno_workflow_pipeline.py \
  --intent "Kartal'da toplantı var, gücü düşür enerji tasarrufu için. Priority CRITICAL" \
  --strategy PRIORITY

# Çıktı: Conflict detected → Priority resolution → CRITICAL wins
```

#### Weighted Merge Stratejisi

```bash
python3 agno_workflow_pipeline.py \
  --intent "Reduce power in Kartal region for energy saving. Priority MEDIUM" \
  --strategy WEIGHTED_MERGE

# Çakışma varsa: Ağırlıklı merge yapılır
```

### 5. Sonuçları Görüntüleme

**Terminal Çıktısı:**
```
======================================================================
🚀 Starting 6G Network Optimization Workflow
======================================================================
Intent: Kartal bölgesinde etkinlik var. RX gücü en az -52 dBm olsun...
Strategy: PRIORITY
======================================================================

📊 Current System Configuration:
   └─ Based on: DEFAULT (No previous intents)
   
📊 Current System KPIs:
      TX0: ON | P=30.0dBm | Az=0.0° | El=0.0°
      TX1: ON | P=30.0dBm | Az=0.0° | El=0.0°
      TX2: ON | P=30.0dBm | Az=0.0° | El=0.0°
      TX3: ON | P=30.0dBm | Az=0.0° | El=0.0°
      
      RX_POWER:     -55.74 dBm
      SINR:         15.21 dB
      THROUGHPUT:   3.08 Mbps
      LOAD_IMB:     3.59
      COVERAGE:     0.33

Step 1: Parse Intent
✅ Intent parsed successfully
   Priority: HIGH
   Target Area: Kartal
   Target KPIs: ['RX_POWER']
   Threshold: RX_POWER >= -52 dBm

Step 2: Optimize Configuration
✅ Optimization completed
   Config ID: 20260206224109
   Changes: 12 parameters modified
   Expected RX_POWER: -51.84 dBm ✓
   Constraints: ✅ Satisfied

Step 3: Detect Conflicts
✅ No conflicts detected
   Active intents: 0

Step 4: Resolve Conflicts
⏭️  Skipped (no conflicts)

Step 5: Finalize Configuration
✅ Configuration finalized
   Final Config ID: 20260206224109
   TX0: ON | P=33.0dBm | Az=-10.0° | El=-2.0°
   TX1: ON | P=33.0dBm | Az=-10.0° | El=-2.0°
   ...

✅ Workflow completed successfully!
📁 Result saved to: workflow_result_20260206_224110.json
```

**JSON Sonuç Dosyası:**

Workflow tamamlandığında `workflow_result_TIMESTAMP.json` dosyası oluşturulur:

```json
{
  "workflow_id": "workflow_20260206_224106",
  "timestamp": "2026-02-06T22:41:06",
  "status": "COMPLETED",
  "intent": {
    "natural_language": "Kartal bölgesinde etkinlik var...",
    "priority": "HIGH",
    "target_area": "Kartal",
    "target_kpis": ["RX_POWER"],
    "time_constraint": {
      "start": "2026-02-06T13:00:00",
      "end": "2026-02-06T15:00:00"
    }
  },
  "optimization": {
    "config_id": "20260206224109",
    "changes": [...],
    "expected_kpis": {...},
    "current_kpis": {...},
    "constraints_satisfied": true
  },
  "conflicts": {
    "is_conflicted": false,
    "num_conflicts": 0,
    "summary": "No conflicts detected"
  },
  "resolution": null,
  "final_configuration": {...},
  "execution_log": [
    "✅ Intent parsed successfully",
    "✅ Optimization completed",
    "✅ No conflicts detected",
    "⏭️  Skipped: No conflicts to resolve",
    "✅ Workflow completed successfully"
  ]
}
```

### 6. State Dosyası Kontrolü

```bash
# Aktif intent'leri göster
cat active_intents_workflow.json

# Pretty print
python3 -m json.tool active_intents_workflow.json

# State'i sıfırla (yeni başlangıç)
echo '{"intents": []}' > active_intents_workflow.json
```

### 7. Debug ve Troubleshooting

**Log seviyesini artır:**
```bash
export DEBUG=1
python3 agno_workflow_pipeline.py --intent "..." --strategy PRIORITY
```

**Surrogate model test et:**
```python
from optimization_agent_v2 import SurrogateModel

surrogate = SurrogateModel("models/surrogate.joblib")
config = {
    'tx0_on': True, 'tx0_P_dBm': 30.0, 'tx0_dAz': 0.0, 'tx0_dEl': 0.0,
    'tx1_on': True, 'tx1_P_dBm': 30.0, 'tx1_dAz': 0.0, 'tx1_dEl': 0.0,
    'tx2_on': True, 'tx2_P_dBm': 30.0, 'tx2_dAz': 0.0, 'tx2_dEl': 0.0,
    'tx3_on': True, 'tx3_P_dBm': 30.0, 'tx3_dAz': 0.0, 'tx3_dEl': 0.0
}
kpis = surrogate.predict_kpis(config)
print(kpis)
```

**Agent'ları ayrı test et:**
```bash
# Optimization agent test
python3 -c "from agents.optimization_agent import run_optimization; print(run_optimization(...))"

# Conflict detector test
python3 tests/test_conflict_resolution.py
```

---

## 📝 Örnek Senaryolar

### Senaryo 1: Tek Intent (Çakışma Yok)

```bash
python3 agno_workflow_pipeline.py \
  --intent "There is an event in Kadıköy between 2-4 PM. Increase RX power to at least -50 dBm. Priority HIGH" \
  --strategy PRIORITY
```

**Sonuç:**
- Intent parse edilir
- Optimal config bulunur (TX güçleri artırılır)
- Çakışma tespit edilmez
- Config finalize edilir ve state'e kaydedilir

### Senaryo 2: Çakışan İki Intent (Priority Resolution)

```bash
# 1. Intent
python3 agno_workflow_pipeline.py \
  --intent "Event in Kartal, increase RX power to -52 dBm. Priority HIGH" \
  --strategy PRIORITY

# 2. Intent (çakışan)
python3 agno_workflow_pipeline.py \
  --intent "Meeting in Kartal, reduce power for energy saving. Priority CRITICAL" \
  --strategy PRIORITY
```

**Sonuç:**
- İki intent çakışır (birisi artır, diğeri azalt)
- CRITICAL > HIGH olduğu için 2. intent kazanır
- 1. intent state'den kaldırılır
- 2. intent finalize edilir

### Senaryo 3: Çoklu Intent (Weighted Merge)

```bash
# 1. Intent
python3 agno_workflow_pipeline.py \
  --intent "Increase power in Kartal. Priority HIGH" \
  --strategy WEIGHTED_MERGE

# 2. Intent
python3 agno_workflow_pipeline.py \
  --intent "Decrease power in Kartal. Priority MEDIUM" \
  --strategy WEIGHTED_MERGE
```

**Sonuç:**
- İki intent çakışır
- Ağırlıklı merge yapılır: HIGH (3.0) + MEDIUM (2.0)
- Sonuç: Hafif artış (HIGH'ın etkisi daha fazla)
- Merged config state'e kaydedilir

---

## 🎯 Sistem Özellikleri

### ✅ Tamamlanan Özellikler

1. **Tam Agent-Based Mimari:** Tüm bileşenler Agno Agent yapısında
2. **LLM Reasoning:** Her agent kararlarını açıklıyor
3. **State Management:** Intent'ler arası config takibi
4. **Conflict Detection:** 4 tip çakışma tespiti
5. **Dual Resolution:** Priority ve Weighted Merge stratejileri
6. **Surrogate Model:** LightGBM tabanlı hızlı KPI tahmini
7. **Multi-KPI Optimization:** 5 farklı KPI'nın optimize edilmesi
8. **Türkçe/İngilizce Destek:** Intent parser her iki dili destekler
9. **CLI ve Playground:** İki farklı kullanım modu
10. **JSON Result Export:** Okunabilir sonuç kayıtları

### 🔮 Gelecek Geliştirmeler (Potansiyel)

1. **Real-time Monitoring:** Canlı KPI takibi
2. **Time-based Scheduling:** Zaman tabanlı intent aktivasyonu
3. **Multi-area Support:** Farklı bölgeler için eş zamanlı optimizasyon
4. **Advanced Resolution:** Pareto optimal çözümler
5. **Feedback Loop:** Gerçek ölçümlerle model güncelleme
6. **Web UI:** Agno playground'dan bağımsız web arayüzü

---

## 📞 Destek ve İletişim

Bu sistem **Agno framework** ve **6G ağ optimizasyonu** araştırması için geliştirilmiştir.

**Teknik Detaylar:**
- Agno Docs: https://docs.agno.com
- Groq API: https://groq.com

**Proje Yapısı:**
```
agentic-pipeline/
├── agents/                         # Agno agent'ları
│   ├── optimization_agent.py
│   ├── conflict_detector_agent.py
│   ├── priority_resolution_agent.py
│   └── weighted_merge_agent.py
├── intent_parser/                  # Intent parser agent
│   └── intent_parser_agent.py
├── models/                         # Eğitilmiş modeller
│   └── surrogate.joblib
├── agno_workflow_pipeline.py       # Ana workflow
├── optimization_agent_v2.py        # Core optimization
├── train_surrogate_v2.py           # Model eğitim
├── active_intents_workflow.json    # State management
└── OVERALL_STRUCTURE.md            # Bu doküman
```

---

**Son Güncelleme:** 7 Şubat 2026  
**Versiyon:** 2.0 (Full Agent-Based)  
**Durum:** ✅ Üretim Hazır
