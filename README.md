# 6G Network Optimization Agentic Pipeline

This repository contains a research prototype for agent-based 6G network optimization. The system takes a natural-language network management request, converts it into a structured intent, proposes base-station configuration changes, checks whether the new request conflicts with active intents, and resolves conflicts using either priority-based selection or weighted merging.

> **Current status:** this is research code, not production software. The Optimization Agent is known to be unreliable in the current working copy and should be treated as a component that needs debugging before its outputs are trusted.

## Documentation map

The repository is documented with three README-style files:

| File | Purpose |
|---|---|
| `README.md` | Entry point: project overview, setup, quick start, external links, and current status. |
| `docs/ARCHITECTURE_AND_USAGE.md` | System architecture, agent responsibilities, possible workflow variants, file structure, commands, and cleanup guidance. |
| `docs/AGENT_TRAINING_EVALUATION.md` | Dataset links, surrogate model training, agent behavior, coverage notebook usage, evaluation notes, and known issues. |

## High-level pipeline

```text
Natural Language Intent
        ↓
Intent Parser Agent
        ↓
Optimization Agent
        ↓
Conflict Detector Agent
        ↓
Priority Resolution / Weighted Merge Agent
        ↓
Final Configuration + Workflow Result JSON
```

The intended workflow has five stages:

1. **Intent parsing** — convert a natural-language request into a structured JSON-like intent.
2. **Optimization** — propose TX power, antenna angle, or ON/OFF changes.
3. **Conflict detection** — compare the new result with active intents.
4. **Conflict resolution** — choose a winner by priority or merge compatible changes.
5. **Finalization** — save the selected configuration and workflow output.

## Important repository status

There are multiple documentation files and several workflow variants in the current project history. They describe slightly different versions of the system:

| Variant | Main idea | Candidate files |
|---|---|---|
| Hybrid agentic workflow | LLM agents design or reason; a LightGBM surrogate model predicts KPIs. | `agno_workflow_pipeline_hybrid.py`, `agents/optimization_agent_hybrid.py`, `active_intents_workflow_hybrid.json` |
| Pure LLM / full-agentic workflow | LLM agents perform optimization, KPI prediction, conflict detection, and resolution without a surrogate model. | `agno_workflow_pipeline_full_agentic.py`, `agents/optimization_agent_llm.py`, `playground_launcher_agentic.py` |
| Earlier Agno + surrogate workflow | Agno agents wrap core optimization, conflict detection, and resolution logic. | `agno_workflow_pipeline.py`, `workflow_steps.py`, `optimization_agent_v2.py`, `active_intents_workflow.json` |

Before relying on the commands below, confirm which workflow is actually active in the codebase.

```bash
grep -R "optimization_workflow\|Workflow(" . --include="*.py"
grep -R "optimization_agent_hybrid\|optimization_agent_llm\|optimization_agent.py" . --include="*.py"
python3 agno_workflow_pipeline_hybrid.py --help 2>/dev/null || true
python3 agno_workflow_pipeline_full_agentic.py --help 2>/dev/null || true
python3 agno_workflow_pipeline.py --help 2>/dev/null || true
```

After confirmation, replace the placeholders below:

```text
Active workflow file:        [TODO_ACTIVE_WORKFLOW_FILE]
Active optimization agent:   [TODO_ACTIVE_OPTIMIZATION_AGENT]
Active state file:           [TODO_ACTIVE_STATE_FILE]
Active model path:           [TODO_MODEL_PATH]
Active dataset path/link:    [TODO_DATASET_PATH_OR_DRIVE_LINK]
Coverage notebook link:      [TODO_COVERAGE_NOTEBOOK_LINK]
```

## External data, model, and notebook links

Large files should not be committed directly to GitHub. Upload them to Google Drive or another shared storage location and put the links here.

| Artifact | Link | Notes |
|---|---|---|
| Raw simulator dataset | `[TODO_RAW_DATASET_DRIVE_LINK]` | CSV/Parquet data used for training or evaluation. |
| Processed training dataset | `[TODO_PROCESSED_DATASET_DRIVE_LINK]` | Optional cleaned or feature-engineered version. |
| Trained surrogate model | `[TODO_SURROGATE_MODEL_DRIVE_LINK]` | Example: `models/surrogate.joblib`. |
| Coverage calculation notebook | `[TODO_COVERAGE_NOTEBOOK_LINK]` | Notebook that manually computes the coverage percentage in the environment. |
| Benchmark/results folder | `[TODO_RESULTS_DRIVE_LINK]` | Large CSV, logs, plots, or workflow outputs. |

## Setup

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

Do **not** commit `.env`.

## Quick start

Use the command that matches the verified active workflow.

### Hybrid workflow candidate

```bash
python3 agno_workflow_pipeline_hybrid.py \
  --intent "There is a meeting in Kartal between 3 PM and 5 PM. Increase RX power to at least -55 dBm. Priority is HIGH." \
  --strategy PRIORITY
```

### Full-agentic workflow candidate

```bash
python3 agno_workflow_pipeline_full_agentic.py \
  --intent "Improve coverage in Kartal to at least -70 dBm. Priority is HIGH." \
  --strategy PRIORITY
```

### Earlier workflow candidate

```bash
python3 agno_workflow_pipeline.py \
  --intent "There is an event in Kartal. RX power should be at least -52 dBm. Priority HIGH." \
  --strategy PRIORITY
```

## Repository cleanup guidance

Keep source code, lightweight configs, tests, and documentation. Move large or generated artifacts to Drive.

Recommended to keep:

```text
README.md
docs/
agents/
intent_parser/
models/                 # only if model files are small enough for GitHub
optimization_agent_v2.py
train_surrogate_v2.py
requirements.txt
.env.example
tests/
```

Recommended to exclude from GitHub:

```text
.env
.venv/
__pycache__/
*.pyc
workflow_result_*.json
active_intents_workflow*.json   # unless a small example state file is intentionally included
*.log
*.csv
*.parquet
*.joblib                       # if large; otherwise keep a small model or use Git LFS
.ipynb_checkpoints/
```

Before deleting duplicate or old Python files, check imports first:

```bash
grep -R "from .* import\|import " . --include="*.py" | grep -E "optimization|conflict|resolution|workflow"
```

## Known issue: Optimization Agent

The Optimization Agent is currently documented as unreliable. Do not present its output as a validated network decision until the following are checked:

1. The active workflow file imports the intended optimization agent.
2. The expected input/output schema matches the downstream conflict detector.
3. The surrogate model path exists if the hybrid version is used.
4. KPI predictions are checked against benchmark or simulator outputs.
5. A simple smoke test passes end-to-end and writes a valid workflow result JSON.

See `docs/AGENT_TRAINING_EVALUATION.md` for the debugging checklist.
