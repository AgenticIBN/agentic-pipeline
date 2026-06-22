# Architecture and Usage

This document explains the system architecture, agent responsibilities, possible workflow variants, expected file structure, and common commands. It is written for the next student or developer who will continue the project.

## 1. Architecture overview

The project is intended as an agentic workflow for 6G network optimization.

```text
┌──────────────────────────────────────────────────────────────┐
│ Input: natural-language network intent                       │
│ Example: "Increase RX power in Kartal to at least -55 dBm"   │
└──────────────────────────────┬───────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 1 — Intent Parser Agent                                 │
│ Converts free text into structured fields: area, KPI,         │
│ threshold, time window, and priority.                         │
└──────────────────────────────┬───────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 2 — Optimization Agent                                  │
│ Proposes configuration changes for TX power, angle, and       │
│ ON/OFF state. This is the component that currently needs      │
│ debugging and validation.                                     │
└──────────────────────────────┬───────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 3 — Conflict Detector Agent                             │
│ Compares the new optimization result with active intents.      │
└──────────────────────────────┬───────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 4 — Conflict Resolution                                 │
│ Resolves conflicts using either PRIORITY or WEIGHTED_MERGE.   │
└──────────────────────────────┬───────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 5 — Finalization                                        │
│ Saves the final configuration, updates active intent state,    │
│ and writes a workflow result JSON.                            │
└──────────────────────────────────────────────────────────────┘
```

## 2. Which version is active?

The existing documentation describes three overlapping versions. Do not assume the filename alone identifies the latest version.

| Version described in old docs | What it claims | Candidate active files | Notes |
|---|---|---|---|
| Hybrid full-agentic version | LLM agents design/refine configurations while a LightGBM surrogate predicts KPIs. | `agno_workflow_pipeline_hybrid.py`, `agents/optimization_agent_hybrid.py`, `active_intents_workflow_hybrid.json` | Best match for a project that has datasets, surrogate training, and coverage evaluation. |
| Pure LLM full-agentic version | LLM agents do optimization and KPI prediction without a surrogate model or rule engine. | `agno_workflow_pipeline_full_agentic.py`, `agents/optimization_agent_llm.py`, `playground_launcher_agentic.py` | Useful as an experimental branch, but risky for quantitative KPI claims unless validated. |
| Earlier Agno + surrogate version | Agno agents wrap core logic files, including `optimization_agent_v2.py`. | `agno_workflow_pipeline.py`, `workflow_steps.py`, `agents/optimization_agent.py`, `active_intents_workflow.json` | More conservative and easier to validate through the surrogate model. |

Recommended documentation stance until the code is verified:

> The project contains an agentic 6G optimization pipeline with both hybrid and pure-LLM variants in its history. The currently active workflow should be confirmed from imports and smoke tests before reporting final results.

Use these commands to verify the active path:

```bash
grep -R "Workflow(" . --include="*.py"
grep -R "optimization_workflow" . --include="*.py"
grep -R "optimization_agent_hybrid\|optimization_agent_llm\|agents.optimization_agent" . --include="*.py"
grep -R "active_intents_workflow" . --include="*.py"
```

## 3. Agent responsibilities

### 3.1 Intent Parser Agent

Candidate file:

```text
intent_parser/intent_parser_agent.py
```

Purpose:

- Parse the user request.
- Extract target area, target KPI, threshold/operator, time window, priority, and confidence.
- Produce structured output that can be consumed by the optimization step.

Example output:

```json
{
  "target_area": "Kartal",
  "target_kpis": ["RX_POWER"],
  "kpi_thresholds": [
    {"kpi": "RX_POWER", "operator": "GTE", "value": -55.0, "unit": "dBm"}
  ],
  "time_constraint_start": "2026-02-09T15:00:00",
  "time_constraint_end": "2026-02-09T17:00:00",
  "priority": "HIGH",
  "confidence": 0.95
}
```

### 3.2 Optimization Agent

Candidate files:

```text
agents/optimization_agent_hybrid.py
agents/optimization_agent_llm.py
agents/optimization_agent.py
optimization_agent_v2.py
```

Purpose:

- Convert the parsed intent into concrete configuration changes.
- Typical parameters: `tx{i}_on`, `tx{i}_P_dBm`, `tx{i}_dAz`, `tx{i}_dEl` for TX0–TX3.
- Predict or estimate KPIs such as RX power, SINR, throughput, coverage ratio, and load imbalance.

Current issue:

> This component is currently not reliable. Treat all optimization results as tentative until the active implementation is identified, the model path is confirmed, and benchmark checks pass.

Expected output shape:

```json
{
  "result_id": "opt_YYYYMMDD_HHMMSS",
  "passed": true,
  "config_changes": [
    {"parameter": "tx0_P_dBm", "before": 30.0, "change": 3.0, "unit": "dBm"},
    {"parameter": "tx0_dAz", "before": 0.0, "change": -10.0, "unit": "deg"}
  ],
  "predicted_kpis": {
    "RX_POWER": -51.8,
    "SINR": 15.2,
    "THROUGHPUT_5P": 3.1,
    "RX_COVERAGE_RATIO": 0.52,
    "LOAD_IMBALANCE": 0.89
  },
  "reasoning": "..."
}
```

### 3.3 Conflict Detector Agent

Candidate files:

```text
agents/conflict_detector_agent_llm.py
agents/conflict_detector_agent.py
conflict_detector_agent.py
```

Purpose:

- Compare a new optimization result with active intents.
- Identify conflicts on the same parameter, same transmitter, resource direction, or ON/OFF state.

Conflict types:

| Type | Meaning |
|---|---|
| `PARAMETER_CONFLICT` | Same parameter, opposite directions. |
| `RESOURCE_CONTENTION` | Same parameter, same direction, different magnitude. |
| `BASE_STATION_CONFLICT` | Same transmitter, different parameter changes that may interact. |
| `BOOLEAN_CONFLICT` | ON/OFF conflict. |

### 3.4 Priority Resolution Agent

Candidate files:

```text
agents/priority_resolution_agent_llm.py
agents/priority_resolution_agent.py
priority_based_resolution_agent.py
```

Purpose:

- Resolve conflicts by selecting the highest-priority intent.
- Priority order: `CRITICAL > HIGH > MEDIUM > LOW`.
- If priorities tie, document the tie-breaking rule used by the active implementation.

### 3.5 Weighted Merge Agent

Candidate files:

```text
agents/weighted_merge_agent_llm.py
agents/weighted_merge_agent.py
weighted_merge_resolution_agent.py
```

Purpose:

- Merge compatible or low/medium-severity conflicts using priority weights.
- Suggested weights: `CRITICAL=4`, `HIGH=3`, `MEDIUM=2`, `LOW=1`.

Example:

```text
Intent A: HIGH,   tx0_P_dBm change = +5, weight = 3
Intent B: MEDIUM, tx0_P_dBm change = +3, weight = 2
Merged change = (3*5 + 2*3) / (3+2) = +4.2 dBm
```

## 4. State and results

Candidate state files:

```text
active_intents_workflow.json
active_intents_workflow_hybrid.json
```

Typical state content:

```json
{
  "intents": [
    {
      "config_id": "20260206224109",
      "priority": "HIGH",
      "target_area": "Kartal",
      "final_config": {
        "tx0_on": true,
        "tx0_P_dBm": 33.0,
        "tx0_dAz": -10.0,
        "tx0_dEl": -2.0
      },
      "expected_kpis": {
        "RX_POWER": -51.84,
        "SINR": 15.25
      }
    }
  ]
}
```

Workflow outputs are usually saved as:

```text
workflow_result_YYYYMMDD_HHMMSS.json
```

Do not commit large or repeated workflow result files. Keep one small example only if it is useful for documentation.

## 5. Recommended clean repository structure

```text
agentic-pipeline/
├── README.md
├── docs/
│   ├── ARCHITECTURE_AND_USAGE.md
│   └── AGENT_TRAINING_EVALUATION.md
├── agents/
│   ├── optimization_agent_hybrid.py          # if hybrid is active
│   ├── optimization_agent_llm.py             # if pure LLM branch is kept
│   ├── conflict_detector_agent_llm.py
│   ├── priority_resolution_agent_llm.py
│   └── weighted_merge_agent_llm.py
├── intent_parser/
│   └── intent_parser_agent.py
├── models/
│   └── surrogate.joblib                      # optional; use Drive/LFS if large
├── tests/
├── train_surrogate_v2.py
├── optimization_agent_v2.py                  # if hybrid/surrogate path is active
├── requirements.txt
├── .env.example
└── .gitignore
```

## 6. Setup commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Example `.env`:

```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
DATA_PATH=/absolute/path/to/dataset.csv
MODEL_PATH=models/surrogate.joblib
RX_POWER_THRESHOLD_DBM=-85
```

## 7. Running the workflow

After confirming the active file, use one of these patterns.

### Hybrid workflow

```bash
python3 agno_workflow_pipeline_hybrid.py \
  --intent "There is a meeting in Kartal between 3 PM and 5 PM. Increase RX power to at least -55 dBm. Priority HIGH." \
  --strategy PRIORITY
```

### Pure LLM workflow

```bash
python3 agno_workflow_pipeline_full_agentic.py \
  --intent "Improve coverage in Kartal to at least -70 dBm. Priority HIGH." \
  --strategy PRIORITY
```

### Earlier Agno workflow

```bash
python3 agno_workflow_pipeline.py \
  --intent "Event in Kartal. Increase RX power to at least -52 dBm. Priority HIGH." \
  --strategy PRIORITY
```

## 8. Resetting active state

Use the state file that matches the active workflow.

```bash
echo '{"intents": []}' > active_intents_workflow.json
echo '{"intents": []}' > active_intents_workflow_hybrid.json
```

Only run the command for the state file actually used by the workflow.

## 9. Test and smoke-check commands

```bash
python3 -m compileall .
python3 -m pytest tests/ -q
```

Minimal workflow smoke test:

```bash
python3 [TODO_ACTIVE_WORKFLOW_FILE] \
  --intent "Test intent: increase RX power in Kartal to at least -70 dBm. Priority LOW." \
  --strategy PRIORITY
```

Then check:

```bash
ls -lh workflow_result_*.json
python3 -m json.tool workflow_result_*.json | head -n 80
```

## 10. Cleanup checklist

Before pushing to GitHub:

```bash
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete
```

Do not commit:

```text
.env
.venv/
workflow_result_*.json
*.log
large *.csv / *.parquet files
large *.joblib files
.ipynb_checkpoints/
```

Before deleting old files, run:

```bash
grep -R "old_file_name_without_py" . --include="*.py" --include="*.md"
```
