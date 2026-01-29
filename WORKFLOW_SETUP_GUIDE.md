# Agno Workflow Setup Guide

## 🎯 Workflow Yapısı

Sisteminiz artık **mevcut agent'ları kullanarak** Agno playground ile görselleştirilebilir!

### Kullanılan Mevcut Agent'lar:

1. **Intent Parser Agent** (`intent_parser/intent_parser_agent.py`)
   - Tam agent tanımı mevcut
   - Agno playground'da görünür

2. **Optimization Tool** (`optimization_agent.py`)
   - `optimize_from_intent()` fonksiyonu
   - Workflow içinde tool olarak çağrılır

3. **Conflict Detector Tool** (`conflict_detector_agent.py`)
   - `detect_conflicts()` fonksiyonu
   - Workflow içinde tool olarak çağrılır

4. **Orchestrator Tool** (`conflict_resolution_orchestrator.py`)
   - `process_new_intent()` fonksiyonu
   - Workflow içinde tool olarak çağrılır

---

## 🚀 Çalıştırma

### 1. Normal Demo Modu (Terminal'de)

```bash
python workflow_main.py
```

Bu mod:
- ✅ 2 senaryo çalıştırır (CRITICAL vs LOW, Different areas)
- ✅ Her adımı terminalde gösterir
- ✅ Agent flow'u detaylı print eder

**Çıktı:**
```
================================================================================
🚀 WORKFLOW EXECUTION STARTED
================================================================================
Input: Kadıköy bölgesinde SINR'ı 10 dB üzerine çıkar, CRITICAL öncelik
Active Intents: 1

================================================================================
📋 STEP 1: Intent Parser Agent
================================================================================
✅ Intent Parsed:
   Target Area: Kadıköy
   Target KPIs: ['SINR']
   Priority: CRITICAL
   Confidence: 0.95

================================================================================
🔧 STEP 2: Optimization Agent (Tool)
================================================================================
✅ Optimization Complete:
   Config ID: 20260130123456
   Changes: 15
   Constraints Satisfied: True

================================================================================
🔍 STEP 3: Conflict Detector (Tool)
================================================================================
🔴 CONFLICT DETECTED
   Summary: Detected 2 conflicts. Max severity: LOW

================================================================================
🎯 STEP 4: Orchestrator (Tool)
================================================================================
✅ Execution Strategy: MERGED
   Resolution Strategy: WEIGHTED_MERGE
   Contributing Intents: 2
   Priority Weights: {'new_CRITICAL': 1.616, 'active_LOW': 0.383}
```

---

### 2. Playground Modu (Web UI)

```bash
python workflow_main.py --playground
```

Bu mod:
- 🌐 `http://localhost:7777` adresinde web UI açar
- 🎮 Agent'ı görselleştirir
- 📊 Workflow step-by-step takip edilebilir
- 🔍 Intent verip real-time sonuç görebilirsin

**Playground'da:**

1. **Sol Panel**: Agent listesi
   - Intent Parser Agent görünür

2. **Orta Panel**: Chat interface
   - Natural language intent gir:
     ```
     Kadıköy bölgesinde SINR'ı 10 dB üzerine çıkar, CRITICAL öncelik
     ```

3. **Sağ Panel**: Agent response
   - Parsed intent JSON
   - Optimization sonucu
   - Conflict report
   - Final configuration

4. **Timeline/History**: 
   - Her adımı görebilirsin
   - Tool calls görünür
   - Input/output tracking

---

## 📋 Test Senaryoları

### Scenario 1: Conflict Detection (CRITICAL vs LOW)

**Input:**
```python
natural_language = "Kadıköy bölgesinde SINR'ı 10 dB üzerine çıkar, CRITICAL öncelik"

active_intents = [{
    "intent": {
        "target_area": "Kadıköy",
        "priority": "LOW"
    },
    "plan": {
        "changes": [{"param": "tx0_P_dBm", "before": 40.0, "after": 35.0}]
    }
}]
```

**Expected:**
- ✅ Conflict detected: YES
- ✅ Execution strategy: MERGED
- ✅ CRITICAL intent dominant (weight: ~1.6 vs ~0.4)

---

### Scenario 2: No Conflict (Different Areas)

**Input:**
```python
natural_language = "Ümraniye bölgesinde throughput'u artır, MEDIUM öncelik"

active_intents = [{
    "intent": {
        "target_area": "Kadıköy",  # Farklı alan
        "priority": "HIGH"
    },
    "plan": {...}
}]
```

**Expected:**
- ✅ Conflict detected: NO
- ✅ Execution strategy: PARALLEL
- ✅ Both intents run independently

---

## 🎮 Playground Kullanımı

### Adım 1: Başlat

```bash
python workflow_main.py --playground
```

Terminal çıktısı:
```
🎮 Starting Agno Playground...
🌐 Open browser: http://localhost:7777
Press Ctrl+C to stop
```

### Adım 2: Browser'da Aç

`http://localhost:7777` adresine git

### Adım 3: Intent Gir

Chat interface'de:
```
Kadıköy bölgesinde SINR'ı 10 dB üzerine çıkar, bu CRITICAL bir istek
```

### Adım 4: Workflow İzle

1. **Intent Parser** çalışır → IntentParse JSON üretir
2. **Optimization Tool** çalışır → OptimizationPlan üretir
3. **Conflict Detector** çalışır → ConflictReport üretir
4. **Orchestrator** çalışır → Final Configuration üretir

Her adımda:
- ✅ Tool call görünür
- ✅ Input/output görünür
- ✅ Süre görünür
- ✅ Sonuç görünür

---

## 📊 Workflow State Tracking

Workflow execution sırasında state:

```python
{
    "current_step": "ORCHESTRATION",
    "parsed_intent": {...},
    "optimization_plan": {...},
    "conflict_report": {
        "is_conflicted": true,
        "conflict_summary": "Detected 1 conflicts..."
    },
    "execution_strategy": "MERGED",
    "final_configuration": {
        "priority_weights": {...},
        "changes": [...],
        "resolution_strategy": "WEIGHTED_MERGE"
    }
}
```

---

## 🔧 Troubleshooting

### Problem: Playground açılmıyor

**Çözüm:**
```bash
# Port'u değiştir
# workflow_main.py içinde:
agent_os.start_playground(port=8888)
```

### Problem: Agent görünmüyor

**Çözüm:**
- Intent parser agent zaten tanımlı
- Diğer agent'lar tool olarak çalışır (normal)
- Playground'da sadece intent_parser_agent görünür

### Problem: Tool çağrıları başarısız

**Çözüm:**
```bash
# Import'ları kontrol et
python -c "from optimization_agent import optimize_from_intent; print('OK')"
python -c "from conflict_detector_agent import detect_conflicts; print('OK')"
python -c "from conflict_resolution_orchestrator import process_new_intent; print('OK')"
```

---

## 📝 Dosya Yapısı

```
agentic-pipeline/
├── workflow_main.py              # 🆕 Ana workflow dosyası
├── intent_parser/
│   └── intent_parser_agent.py    # ✅ Mevcut agent
├── optimization_agent.py         # ✅ Mevcut tool
├── conflict_detector_agent.py    # ✅ Mevcut tool
└── conflict_resolution_orchestrator.py  # ✅ Mevcut tool
```

---

## 🎯 Özet

✅ **Mevcut agent'lar kullanılıyor** - Yeni agent tanımı yok!
✅ **Workflow tam fonksiyonel** - Tüm pipeline çalışıyor
✅ **Playground ready** - Web UI ile görselleştirilebilir
✅ **Step-by-step tracking** - Her adım izlenebilir
✅ **Conflict resolution** - Priority-based merging çalışıyor

**Çalıştırmak için:**
```bash
# Terminal demo
python workflow_main.py

# Web playground
python workflow_main.py --playground
```

**Playground'da göreceğin şeyler:**
- 🟢 Intent Parser Agent (main)
- 🔧 Optimization Tool (background)
- 🔍 Conflict Detector Tool (background)
- 🎯 Orchestrator Tool (background)
- 📊 Final Configuration (merged veya parallel)
