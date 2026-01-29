# Agno Multi-Agent Workflow: 6G Network Intent-Based Optimization

## 🎯 Pipeline Özet

```
Doğal Dil Intent
    ↓
[1] Intent Parser Agent
    ↓
IntentParse JSON
    ↓
[2] Optimization Agent
    ↓
OptimizationPlan JSON
    ↓
[3] Conflict Detection Agent (New vs Active)
    ↓
    ├─→ No Conflict → [4a] Parallel Execution
    │                      (Add to system, run independently)
    │
    └─→ Conflict Detected → [4b] Orchestration Team
                                   ↓
                            Conflict Resolution Agent
                                   ↓
                            Merged Configuration
```

---

## 📋 Agent Detayları

### 1️⃣ Intent Parser Agent
**Dosya**: `intent_parser/intent_parser_agent.py`

**Görev**: Doğal dil intent'i → Structured IntentParse JSON

**Input**:
```python
"Kadıköy bölgesindeki kullanıcılar için SINR'ı 10 dB üzerine çıkar, CRITICAL öncelik"
```

**Output**:
```json
{
  "target_area": "Kadıköy",
  "target_kpis": ["SINR"],
  "kpi_thresholds": [
    {"kpi": "SINR", "op": "GTE", "value": 10.0, "unit": "dB"}
  ],
  "priority": "CRITICAL",
  "confidence": 0.95
}
```

**Agno Tool**: `parse_intent(natural_language: str) -> IntentParse`

---

### 2️⃣ Optimization Agent
**Dosya**: `optimization_agent.py`

**Görev**: IntentParse → Optimum base station configuration

**Input**: IntentParse JSON

**Output**:
```json
{
  "selected_config_id": 12345,
  "changes": [
    {"param": "tx0_P_dBm", "before": 42.0, "after": 40.0, "unit": "dBm"},
    {"param": "tx1_dAz", "before": 0.0, "after": -5.0, "unit": "deg"}
  ],
  "expected_kpis": {
    "RX_POWER": -82.0,
    "SINR": 8.5,
    "THROUGHPUT_5P": 22.0
  },
  "constraints_satisfied": true
}
```

**Agno Tool**: `optimize_from_intent(intent_json: str) -> OptimizationPlan`

---

### 3️⃣ Conflict Detection Agent
**Dosya**: `conflict_detector_agent.py`

**Görev**: Yeni intent vs Active intentler → Conflict analizi

**Input**:
- New intent + plan
- Active intents + plans (running in system)

**Output**:
```json
{
  "conflict_report": {
    "is_conflicted": true,
    "conflict_summary": "Detected 2 conflicts. Max severity: HIGH",
    "details": [
      {
        "conflict_type": "DIRECT_OPPOSITION",
        "severity": "HIGH",
        "conflicting_base_station": "tx0",
        "description": "New intent increases tx0_P_dBm while active decreases it"
      }
    ]
  }
}
```

**Agno Tool**: `detect_conflicts(new_intent_json, new_plan_json, active_intents_data) -> ConflictReport`

---

### 4️⃣ Orchestrator & Resolution Agent

#### 4a. Parallel Execution (No Conflict)
**Dosya**: `conflict_resolution_orchestrator.py`

**Görev**: Add new intent to system for parallel execution

**Output**:
```json
{
  "execution_strategy": "PARALLEL",
  "final_configuration": {
    "execution_mode": "parallel",
    "configurations": [
      {"intent_id": "active_1", "plan": {...}},
      {"intent_id": "new_intent", "plan": {...}}
    ]
  }
}
```

#### 4b. Team Resolution (Conflict Detected)
**Dosya**: `conflict_resolution_agent.py` + `conflict_resolution_orchestrator.py`

**Görev**: Merge conflicting intents using priority-based weights

**Output**:
```json
{
  "execution_strategy": "MERGED",
  "final_configuration": {
    "merged_config_id": "merged_20260129233112",
    "contributing_intents": ["new_CRITICAL", "active_LOW"],
    "priority_weights": {
      "new_CRITICAL": 1.6166,
      "active_LOW": 0.3834
    },
    "changes": [
      {"param": "tx0_P_dBm", "before": 42.0, "after": 40.5, "unit": "dBm"}
    ],
    "resolution_strategy": "WEIGHTED_MERGE"
  }
}
```

---

## 🚀 Agno Playground Workflow Yapısı

### Adım 1: Workspace Oluştur

```python
# workspace_config.py
from agno import Workspace

workspace = Workspace(
    name="6g-intent-optimization",
    description="Multi-agent 6G network intent-based optimization system",
    agents=[
        intent_parser_agent,
        optimization_agent,
        conflict_detector_agent,
        orchestrator_agent
    ]
)
```

### Adım 2: Agent Tanımları

```python
# agents.py
from agno.agent import Agent
from agno.models.google import Gemini

# 1. Intent Parser Agent
intent_parser_agent = Agent(
    name="Intent Parser",
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Parses natural language intents into structured JSON",
    instructions=[
        "Parse user intent into IntentParse schema",
        "Extract target_area, KPIs, thresholds, priority",
        "Return structured JSON with confidence score"
    ],
    response_model=IntentParse,
    tools=[parse_intent_tool]
)

# 2. Optimization Agent
optimization_agent = Agent(
    name="Optimization Agent",
    model=Gemini(id="gemini-2.5-flash"),
    description="Finds optimal base station configuration for given intent",
    instructions=[
        "Given IntentParse, find best config from surrogate model",
        "Return OptimizationPlan with changes and expected KPIs",
        "Ensure constraints are satisfied"
    ],
    tools=[optimize_from_intent],
    response_model=OptimizationPlan
)

# 3. Conflict Detector Agent
conflict_detector_agent = Agent(
    name="Conflict Detector",
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Detects conflicts between new and active intents at base station level",
    instructions=[
        "Compare new intent changes vs active intent changes",
        "Check base station parameter conflicts (tx0-tx3)",
        "Classify conflict type and severity",
        "Return detailed conflict report"
    ],
    tools=[detect_conflicts],
    response_model=ConflictReport
)

# 4. Orchestrator Agent (Team Leader)
orchestrator_agent = Agent(
    name="Orchestrator",
    model=Gemini(id="gemini-2.5-flash"),
    description="Coordinates optimization agents and resolves conflicts",
    instructions=[
        "Process new intent against active intents",
        "If no conflict: Add to parallel execution",
        "If conflict: Use resolution agent to merge with priority weights",
        "Return final configuration"
    ],
    tools=[process_new_intent, resolve_conflicts],
    team=[optimization_agent, conflict_detector_agent]
)
```

### Adım 3: Workflow Definition

```python
# workflow.py
from agno.workflow import Workflow, WorkflowStep

workflow = Workflow(
    name="intent-optimization-workflow",
    description="End-to-end intent-based 6G network optimization",
    steps=[
        WorkflowStep(
            name="parse_intent",
            agent=intent_parser_agent,
            input_key="user_intent",
            output_key="parsed_intent"
        ),
        WorkflowStep(
            name="optimize",
            agent=optimization_agent,
            input_key="parsed_intent",
            output_key="optimization_plan"
        ),
        WorkflowStep(
            name="check_conflicts",
            agent=conflict_detector_agent,
            input_key=["optimization_plan", "active_intents"],
            output_key="conflict_report",
            condition="has_active_intents"
        ),
        WorkflowStep(
            name="orchestrate",
            agent=orchestrator_agent,
            input_key=["optimization_plan", "conflict_report", "active_intents"],
            output_key="final_configuration",
            branches={
                "no_conflict": "add_to_parallel",
                "conflict_detected": "team_resolution"
            }
        )
    ]
)
```

### Adım 4: Playground ile Çalıştırma

```bash
# Terminal'de
agno playground start
```

**Playground'da**:

1. **Workspace seç**: `6g-intent-optimization`

2. **Workflow başlat**:
   ```json
   {
     "user_intent": "Kadıköy bölgesinde SINR'ı 10 dB üzerine çıkar, CRITICAL öncelik",
     "active_intents": [
       {
         "intent": {"target_area": "Kadıköy", "priority": "LOW"},
         "plan": {"changes": [...]}
       }
     ]
   }
   ```

3. **Agent akışını izle**:
   - 🟢 Intent Parser → IntentParse
   - 🔵 Optimization Agent → OptimizationPlan
   - 🟠 Conflict Detector → ConflictReport (is_conflicted: true)
   - 🔴 Orchestrator → Team Resolution → MergedConfiguration

4. **Final Output görüntüle**:
   ```json
   {
     "execution_strategy": "MERGED",
     "final_configuration": {
       "priority_weights": {...},
       "changes": [...],
       "resolution_strategy": "WEIGHTED_MERGE"
     }
   }
   ```

---

## 🎮 Playground Görselleştirme

### Agent Graph View
```
┌─────────────────┐
│ Intent Parser   │
└────────┬────────┘
         │ IntentParse
         ↓
┌─────────────────┐
│ Optimization    │
└────────┬────────┘
         │ OptimizationPlan
         ↓
┌─────────────────┐     ┌──────────────┐
│ Conflict        │────→│ Active       │
│ Detector        │     │ Intents DB   │
└────────┬────────┘     └──────────────┘
         │
    ┌────┴────┐
    │         │
No Conflict  Conflict
    │         │
    ↓         ↓
┌────────┐  ┌─────────────────┐
│Parallel│  │ Team Resolution │
│Execute │  │ (Orchestrator)  │
└────────┘  └────────┬────────┘
                     │
                     ↓
              Merged Config
```

---

## 📊 Test Scenarios (Playground)

### Scenario 1: No Conflict
```python
# Input
{
  "user_intent": "Ümraniye bölgesinde throughput'u artır",
  "active_intents": [
    {"target_area": "Kadıköy", "priority": "MEDIUM"}  # Farklı alan
  ]
}

# Expected: execution_strategy = "PARALLEL"
```

### Scenario 2: LOW Conflict
```python
# Input
{
  "user_intent": "Kadıköy'de SINR'ı 12 dB üzerine çıkar, CRITICAL",
  "active_intents": [
    {"target_area": "Kadıköy", "priority": "LOW"}  # Aynı alan, düşük priority
  ]
}

# Expected: execution_strategy = "MERGED"
# CRITICAL intent dominant olur (weight: 1.6 vs 0.4)
```

### Scenario 3: HIGH Conflict
```python
# Input
{
  "user_intent": "tx0'ın gücünü artır, HIGH priority",
  "active_intents": [
    {"plan": {"changes": [{"param": "tx0_P_dBm", "after": 35.0}]}, "priority": "HIGH"}
  ]
}

# Expected: resolution_strategy = "CONSERVATIVE_MERGE"
# İki HIGH intent → Dikkatli merge
```

---

## 🛠️ Implementasyon Adımları

### 1. Tool Wrapper'ları Oluştur

```python
# tools/intent_parser_tool.py
from agno.tools import tool
from intent_parser.intent_parser_agent import intent_parser_agent

@tool(name="parse_intent")
def parse_intent_tool(natural_language: str) -> dict:
    """Parse natural language intent into structured JSON"""
    response = intent_parser_agent.run(natural_language)
    return response.content.model_dump()
```

```python
# tools/optimization_tool.py
from agno.tools import tool
from optimization_agent import optimize_from_intent

@tool(name="optimize_intent")
def optimize_intent_tool(intent_json: str) -> dict:
    """Find optimal configuration for given intent"""
    result_json = optimize_from_intent(intent_json)
    return json.loads(result_json)
```

```python
# tools/orchestrator_tool.py
from agno.tools import tool
from conflict_resolution_orchestrator import process_new_intent

@tool(name="orchestrate_intent")
def orchestrate_intent_tool(new_intent_json: str, active_intents_json: str) -> dict:
    """Process new intent with conflict resolution"""
    result_json = process_new_intent(new_intent_json, active_intents_json)
    return json.loads(result_json)
```

### 2. Playground Config Oluştur

```python
# playground_config.py
from agno.playground import PlaygroundConfig

config = PlaygroundConfig(
    workspace_dir="./",
    agents_dir="./",
    tools_dir="./tools",
    workflows_dir="./workflows",
    port=7777,
    auto_reload=True,
    log_level="INFO"
)
```

### 3. Çalıştır

```bash
# 1. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Dependencies
pip install agno pydantic python-dotenv

# 3. Start playground
agno playground start --config playground_config.py

# 4. Browser'da aç
# http://localhost:7777
```

---

## 📈 Monitoring & Debugging

Playground'da şunları görebilirsiniz:

1. **Agent Chain**: Her agent'ın sırayla çağrılması
2. **Tool Calls**: Hangi tool'lar çağrıldı, ne döndü
3. **Intermediate Outputs**: Her adımın çıktısı
4. **Error Traces**: Hata oluşursa detaylı stack trace
5. **Performance Metrics**: Her agent'ın response time'ı
6. **Conflict Resolution Details**: Weight hesaplamaları, merge stratejisi

---

## 🎯 Özet

**Agno Playground ile**:
- ✅ Multi-agent pipeline'ı görselleştir
- ✅ Her agent'ın input/output'unu izle
- ✅ Conflict resolution sürecini step-by-step gör
- ✅ Test scenarioları ile hızlı debug
- ✅ Production'a geçmeden önce validate et

**Workflow**:
```
Natural Language → IntentParse → Optimize → Conflict Check → Orchestrate → Final Config
```

**Key Files**:
- `intent_parser/intent_parser_agent.py`
- `optimization_agent.py`
- `conflict_detector_agent.py`
- `conflict_resolution_orchestrator.py`
- `conflict_resolution_agent.py`

Tüm agentlar hazır, sadece Agno playground integration yapılacak!
