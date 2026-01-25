# Optimization Agent Test Cases

Playgroundda (http://localhost:8001) bu test case'leri kullanabilirsin.

## Test Case 1: Basit Coverage İyileştirme

```json
{
  "target_area": "Kadıköy",
  "target_kpis": ["RX_POWER"],
  "kpi_thresholds": [
    {
      "kpi": "RX_POWER",
      "op": "GTE",
      "value": -95.0,
      "unit": "dBm"
    }
  ],
  "priority": "HIGH",
  "confidence": 0.9,
  "user_set_id": 1,
  "k_users": 100
}
```

**Açıklama:** Kadıköy bölgesinde RX power en az -95 dBm olsun. Yüksek öncelik.

---

## Test Case 2: Quality İyileştirme + Config Change

```json
{
  "target_area": "TR-IST-034",
  "target_kpis": ["SINR"],
  "kpi_thresholds": [
    {
      "kpi": "SINR",
      "op": "GT",
      "value": 10.0,
      "unit": "dB"
    }
  ],
  "priority": "MEDIUM",
  "configuration_change": [
    {
      "action": "increase tilt by 2 degrees",
      "parameter": "tilt",
      "direction": "INCREASE",
      "amount": 2.0,
      "unit": "degrees"
    }
  ],
  "confidence": 0.95,
  "current_config_id": 1500,
  "user_set_id": 1,
  "k_users": 100
}
```

**Açıklama:** Site TR-IST-034'te SINR 10 dB üstüne çıksın. Mevcut config_id 1500'den başlayarak güvenli değişiklik.

---

## Test Case 3: Load Balancing (CRITICAL)

```json
{
  "target_area": "Ankara Çankaya",
  "target_kpis": ["SERVED_USERS"],
  "kpi_thresholds": [
    {
      "kpi": "SERVED_USERS",
      "op": "TARGET",
      "value": 0.0
    }
  ],
  "priority": "CRITICAL",
  "confidence": 0.85,
  "user_set_id": 2,
  "k_users": 200
}
```

**Açıklama:** Ankara Çankaya'da yük dengesiz, kullanıcı dağılımını dengele. Kritik öncelik.

---

## Test Case 4: Throughput Artırma

```json
{
  "target_area": "Beşiktaş",
  "target_kpis": ["THROUGHPUT_5P"],
  "kpi_thresholds": [
    {
      "kpi": "THROUGHPUT_5P",
      "op": "GTE",
      "value": 8.0,
      "unit": "Mbps"
    }
  ],
  "priority": "HIGH",
  "confidence": 0.9,
  "user_set_id": 1,
  "k_users": 150
}
```

**Açıklama:** Beşiktaş'ta 5-percentile throughput en az 8 Mbps olsun.

---

## Test Case 5: Multi-KPI (Coverage + Quality)

```json
{
  "target_area": "Istanbul City Center",
  "target_kpis": ["RX_POWER", "SINR"],
  "kpi_thresholds": [
    {
      "kpi": "RX_POWER",
      "op": "GTE",
      "value": -90.0,
      "unit": "dBm"
    },
    {
      "kpi": "SINR",
      "op": "GTE",
      "value": 8.0,
      "unit": "dB"
    }
  ],
  "priority": "HIGH",
  "time_constraint_start": "18:00",
  "time_constraint_end": "23:00",
  "confidence": 0.92,
  "current_config_id": 2000,
  "user_set_id": 1,
  "k_users": 100
}
```

**Açıklama:** İstanbul merkezde hem coverage hem quality iyileştir. Akşam 18:00-23:00 arası.

---

## Test Case 6: BETWEEN Constraint

```json
{
  "target_area": "Ümraniye",
  "target_kpis": ["RX_POWER"],
  "kpi_thresholds": [
    {
      "kpi": "RX_POWER",
      "op": "BETWEEN",
      "value_low": -95.0,
      "value_high": -85.0,
      "unit": "dBm"
    }
  ],
  "priority": "MEDIUM",
  "confidence": 0.88,
  "user_set_id": 1,
  "k_users": 120
}
```

**Açıklama:** Ümraniye'de RX power -95 ile -85 dBm arasında olsun.

---

## Test Case 7: DELTA_UP (Baseline'dan artış)

```json
{
  "target_area": "Maltepe",
  "target_kpis": ["THROUGHPUT_5P"],
  "kpi_thresholds": [
    {
      "kpi": "THROUGHPUT_5P",
      "op": "DELTA_UP",
      "delta": 2.0,
      "unit": "Mbps"
    }
  ],
  "priority": "HIGH",
  "confidence": 0.9,
  "current_config_id": 1800,
  "user_set_id": 1,
  "k_users": 100
}
```

**Açıklama:** Maltepe'de mevcut throughput'a göre 2 Mbps artış sağla.

---

## 🎯 Playground'da Nasıl Test Edilir?

1. **http://localhost:8001** aç
2. Chat alanına yukarıdaki JSON'lardan birini yapıştır
3. **ÖNEMLİ:** DATA_PATH environment variable'ı .env dosyasında tanımlı olmalı
4. Agent optimize_from_intent tool'unu çağırır
5. OptimizationPlan JSON döner

---

## ⚙️ .env Dosyası Kontrolü

Aşağıdaki değişkenin .env dosyasında olması gerekli:

```bash
GOOGLE_API_KEY=your_api_key_here
DATA_PATH=/path/to/your/dataset.parquet  # veya .csv
GEMINI_MODEL=gemini-2.5-flash
```

---

## 📊 Beklenen Output Formatı

```json
{
  "selected_config_id": 2345,
  "baseline_config_id": 1500,
  "changes": [
    {
      "param": "tx0_P_dBm",
      "before": 20.0,
      "after": 23.0,
      "unit": "dBm"
    },
    {
      "param": "tx1_dEl",
      "before": 5.0,
      "after": 3.0,
      "unit": "deg"
    }
  ],
  "expected_kpis": {
    "RX_POWER": -92.5,
    "SINR": 11.2,
    "THROUGHPUT_5P": 9.8,
    "LOAD_IMBALANCE": 3.2,
    "RX_COVERAGE_RATIO": 0.85
  },
  "baseline_kpis": {
    "RX_POWER": -95.0,
    "SINR": 9.5,
    "THROUGHPUT_5P": 7.5,
    "LOAD_IMBALANCE": 5.1,
    "RX_COVERAGE_RATIO": 0.78
  },
  "constraints_satisfied": true,
  "warnings": [],
  "rationale": "Selected the highest-scoring feasible configuration..."
}
```
