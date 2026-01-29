# Conflict Resolution Team Architecture

## 📐 Mimari Genel Bakış

```
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR (Team Leader)                     │
│              conflict_resolution_orchestrator.py                 │
└──────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │  STEP 1: Parallel Optimization Agents   │
        │  (Her intent için ayrı agent çağrısı)   │
        └──────────────┬──────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
   ┌─────────┐                  ┌─────────┐
   │ Agent 1 │                  │ Agent N │
   │ Intent 1│                  │ Intent N│
   └────┬────┘                  └────┬────┘
        │                             │
        └──────────────┬──────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │  STEP 2: Conflict Detection             │
        │  conflict_detector_agent.py             │
        └──────────────┬──────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────────┐
        │  STEP 3: Conflict Resolution            │
        │  conflict_resolution_agent.py           │
        │  (Meta-Agent - Weight-based Merging)    │
        └──────────────┬──────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────────┐
        │  FINAL OUTPUT                            │
        │  - MERGED config (conflicts var)         │
        │  - PARALLEL configs (conflict yok)       │
        └─────────────────────────────────────────┘
```

## 🔧 Bileşenler

### 1. **Orchestrator (Team Leader)**
- **Dosya:** `conflict_resolution_orchestrator.py`
- **Rol:** Tüm süreci yönetir, team member'ları (optimization agentları) koordine eder
- **Input:** `MultiIntentInput` (birden fazla intent)
- **Output:** `OrchestrationOutput` (final merged veya parallel config)

**İş Akışı:**
```python
orchestrator = ConflictResolutionOrchestrator()
result = orchestrator.run_orchestration(multi_intent_input)
```

### 2. **Optimization Agents (Team Members)**
- **Dosya:** `optimization_agent.py`
- **Rol:** Her intent için ayrı optimization yapar
- **Her agent:** Kendi intent'i için en iyi config'i bulur

**Paralel Çalışma:**
```python
for intent in intents:
    result = optimize_single_intent(intent)  # Her biri ayrı agent gibi
    results.append(result)
```

### 3. **Conflict Detector**
- **Dosya:** `conflict_detector_agent.py`
- **Rol:** Base station seviyesinde conflict analizi
- **Output:** Conflict raporu + severity levels

### 4. **Conflict Resolution Meta-Agent**
- **Dosya:** `conflict_resolution_agent.py`
- **Rol:** Conflict'leri çözümler, optimal merge yapar
- **Strateji:** Priority-based weighted merging

## 📊 Weight Sistemi

### Priority Weights (Ana Faktör)
```python
PRIORITY_WEIGHTS = {
    "CRITICAL": 4.0,  # En yüksek öncelik
    "HIGH": 3.0,
    "MEDIUM": 2.0,
    "LOW": 1.0
}
```

### Aynı Priority'de Tie-Breaking
```python
final_weight = base_weight * confidence * kpi_improvement * severity_penalty

# Örnek:
# Intent A: CRITICAL, confidence=0.9, improvement=0.8
# Weight = 4.0 * 0.95 * 0.94 = 3.57

# Intent B: CRITICAL, confidence=0.7, improvement=0.5
# Weight = 4.0 * 0.85 * 0.85 = 2.89

# Intent A daha yüksek weight alır!
```

**Tie-Breaking Faktörleri:**
1. **Confidence Score** (0.5-1.0 multiplier)
2. **KPI Improvement** (ne kadar iyileştirme sağlıyor)
3. **Conflict Severity Penalty** (CRITICAL conflict'e karışırsa penalty)

## 🔀 Merge Stratejileri

### 1. CRITICAL Conflicts
```python
strategy = "PRIORITY_OVERRIDE"
# En yüksek weight'e sahip intent'in değeri kullanılır
# Örnek: Intent A (weight=3.57) vs Intent B (weight=2.89)
# Sonuç: Intent A'nın değeri (override)
```

### 2. HIGH Conflicts
```python
strategy = "WEIGHTED_MERGE_CONSERVATIVE"
# Weighted average ama daha conservative
# Örnek: tx0_P_dBm
#   Intent A: +5.0 (weight=3.0)
#   Intent B: +2.0 (weight=1.5)
# Sonuç: (5.0*3.0 + 2.0*1.5) / (3.0+1.5) = 4.0
```

### 3. MEDIUM/LOW Conflicts
```python
strategy = "WEIGHTED_MERGE"
# Tam weighted average
```

### 4. Boolean Parametreler
```python
# Weighted majority voting
# Örnek: tx2_on
#   Intent A: True (weight=3.0)
#   Intent B: False (weight=1.5)
# Sonuç: True (3.0 > 1.5)
```

## 🚀 Kullanım

### Basit Kullanım
```python
from conflict_resolution_orchestrator import run_multi_intent_optimization

# Multiple intents hazırla
intents_json = json.dumps([
    {
        "target_area": "Kadıköy",
        "target_kpis": ["RX_POWER"],
        "priority": "CRITICAL",
        "confidence": 0.9,
        # ... diğer alanlar
    },
    {
        "target_area": "Kadıköy",
        "target_kpis": ["THROUGHPUT_5P"],
        "priority": "LOW",
        "confidence": 0.7,
        # ... diğer alanlar
    }
])

# Orchestrator'ı çalıştır
result_json = run_multi_intent_optimization(intents_json)
result = json.loads(result_json)

# Sonuçları incele
print(f"Execution Strategy: {result['execution_strategy']}")
print(f"Conflict Detected: {result['conflict_detected']}")

if result['final_configuration']:
    merged = result['final_configuration']
    print(f"Merged Config ID: {merged['merged_config_id']}")
    print(f"Parameter Changes: {len(merged['changes'])}")
```

### Test Çalıştırma
```bash
# Tam test suite
python test_orchestrator.py

# Sonuçlar: results/orchestration_scenario_*.json
```

## 📤 Output Formatı

### Conflict Varsa (MERGED)
```json
{
  "orchestration_id": "orch_20260129...",
  "execution_strategy": "MERGED",
  "conflict_detected": true,
  "resolution_applied": true,
  "final_configuration": {
    "merged_config_id": "merged_20260129...",
    "contributing_intents": ["intent_0_CRITICAL", "intent_1_LOW"],
    "priority_weights": {
      "intent_0_CRITICAL": 3.57,
      "intent_1_LOW": 0.89
    },
    "changes": [
      {"param": "tx0_P_dBm", "before": 40.0, "after": 4.2, "unit": "dBm"},
      {"param": "tx1_dAz", "before": 0.0, "after": -8.5, "unit": "deg"}
    ],
    "expected_kpis": {
      "RX_POWER": -82.5,
      "SINR": 9.2,
      "THROUGHPUT_5P": 24.8
    },
    "resolution_strategy": "PRIORITY_OVERRIDE",
    "constraints_satisfied": true
  }
}
```

### Conflict Yoksa (PARALLEL)
```json
{
  "orchestration_id": "orch_20260129...",
  "execution_strategy": "PARALLEL",
  "conflict_detected": false,
  "resolution_applied": false,
  "final_configuration": {
    "execution_mode": "parallel",
    "configurations": [
      {
        "intent_id": "intent_0_HIGH",
        "plan": { /* Intent 1'in optimization plan'ı */ }
      },
      {
        "intent_id": "intent_1_MEDIUM",
        "plan": { /* Intent 2'nin optimization plan'ı */ }
      }
    ]
  }
}
```

## 🎯 Ana Farklar (Eski vs Yeni)

### ❌ Eski Yaklaşım (Yanlış)
```python
# Tek agent, sadece hazır proposals'ları merge ediyor
resolve_conflicts(meta_input_json)
# Input: Hazır proposals
# Problem: Optimization agents çağrılmıyor!
```

### ✅ Yeni Yaklaşım (Doğru - Team)
```python
# Orchestrator: Her intent için optimization agent çağırır
orchestrator.run_orchestration(multi_intent_input)

# Adım 1: Intent 1 → Optimization Agent 1 → Output 1
# Adım 2: Intent 2 → Optimization Agent 2 → Output 2
# Adım 3: Conflict Detection (Output 1 vs Output 2)
# Adım 4: Conflict Resolution (Weight-based merge)
# Sonuç: Merged Configuration
```

## 🧪 Test Senaryoları

1. **CRITICAL vs LOW Priority:** CRITICAL dominates
2. **Same Priority, Different Confidence:** Higher confidence wins
3. **No Conflict - Different Areas:** Parallel execution

Her test sonucu `results/` klasörüne kaydedilir.

## 📝 Notlar

- **Base Station Level Detection:** tx0-tx3 seviyesinde conflict kontrolü
- **Automatic Weight Calculation:** Priority + Confidence + KPI Improvement
- **Severity-Based Strategy:** CRITICAL → Override, LOW → Full merge
- **Parallel Execution:** Conflict yoksa her intent bağımsız çalışır
