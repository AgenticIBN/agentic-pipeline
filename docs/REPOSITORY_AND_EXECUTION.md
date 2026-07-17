# Repository Structure and Execution

## 1. Repository tree

```text
agentic-ibn-repo/
├── .env.example
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── run_workflow.py
├── train_surrogate.py
├── benchmark_surrogates.py
├── config/
│   └── model_registry.example.json
├── examples/
│   └── parsed_intent.json
├── agentic_ibn/
│   ├── config.py
│   ├── constraints.py
│   ├── intent_adapter.py
│   ├── schemas.py
│   ├── agents/
│   │   ├── runtime.py
│   │   ├── intent_parser_agent.py
│   │   ├── optimization_agent.py
│   │   ├── conflict_detector_agent.py
│   │   ├── meta_agent.py
│   │   ├── priority_resolution_agent.py
│   │   ├── weighted_merge_agent.py
│   │   └── reasoning_agent.py
│   ├── orchestration/
│   │   └── workflow.py
│   ├── optimization/
│   │   ├── objective.py
│   │   └── validator.py
│   ├── surrogate/
│   │   ├── features.py
│   │   ├── model.py
│   │   └── training.py
│   └── storage/
│       ├── result_store.py
│       └── state_store.py
├── docs/
│   ├── ARCHITECTURE_AND_AGENTS.md
│   ├── REPOSITORY_AND_EXECUTION.md
│   └── DATASETS_MODELS_COLAB_AND_EVALUATION.md
├── notebooks/
└── tests/
```

## 2. Main modules

### Command-line entry points

```text
run_workflow.py
train_surrogate.py
benchmark_surrogates.py
```

### Agent modules

```text
agentic_ibn/agents/
```

These files define the seven Agno agent factories, production wrappers, structured-output handling, and deterministic offline alternatives.

### Orchestration

```text
agentic_ibn/orchestration/workflow.py
```

This module builds the Agno Workflow, executes its named steps, records history, and coordinates persistence.

### Optimization and surrogate modules

```text
agentic_ibn/optimization/
agentic_ibn/surrogate/
```

These modules validate candidate configurations, score intent objectives, build model features, train surrogate regressors, load artifacts, and perform KPI inference.

### Storage

```text
agentic_ibn/storage/
```

The storage layer manages active intents, effective configuration, run directories, JSON artifacts, and failure records.

## 3. Environment variables

Create the local environment file:

```bash
cp .env.example .env
```

Required for the Agno runtime:

```text
GROQ_API_KEY
```

Supported settings:

```text
GROQ_MODEL
AGNO_RUNTIME
DEFAULT_SCENARIO
MODEL_REGISTRY_FILE
RESULTS_DIR
ACTIVE_STATE_FILE
MAX_OPTIMIZATION_ITERATIONS
```


## 4. Model registry

Create a local registry:

```bash
cp config/model_registry.example.json config/model_registry.json
```


```json
{
  "urban_area": "models/urban_area_surrogate.joblib",
  "open_area": "models/open_area_surrogate.joblib"
}
```

The registry maps each scenario name to its surrogate artifact. The scenario requested at runtime must match the scenario metadata stored in the artifact.

## 5. Local data and model layout

Datasets and model artifacts are intentionally stored outside version control:

```text
datasets/
├── urban_area/
└── open_area/

models/
├── urban_area_surrogate.joblib
├── urban_area_surrogate.joblib.metrics.json
├── open_area_surrogate.joblib
└── open_area_surrogate.joblib.metrics.json
```

The repository can be distributed without these directories. Each user can place the datasets and trained artifacts locally and configure their paths through commands or the model registry.

## 7. Train a surrogate

### Urban-area model

```bash
python train_surrogate.py \
  --data datasets/urban_area \
  --scenario urban_area \
  --model-path models/urban_area_surrogate.joblib \
  --model-family lightgbm \
  --sample-size 10000000 \
  --split-strategy auto \
  --num-threads 4
```

### Open-area model

```bash
python train_surrogate.py \
  --data datasets/open_area \
  --scenario open_area \
  --model-path models/open_area_surrogate.joblib \
  --model-family lightgbm \
  --sample-size 10000000 \
  --split-strategy auto \
  --num-threads 4
```

## 8. Run the Agno workflow

Basic command:

```bash
python run_workflow.py \
  --intent "Improve coverage in Kadikoy and keep RX power above -95 dBm" \
  --scenario urban_area
```

The default runtime is:

```text
--runtime agno
```

This mode invokes the Agno Workflow and all applicable Agno Agents.

### Explicit model path

```bash
python run_workflow.py \
  --intent "Turn off TX2 for maintenance while preserving coverage" \
  --scenario urban_area \
  --model-path models/urban_area_surrogate.joblib
```

### Select a conflict strategy

```bash
python run_workflow.py \
  --intent "Reduce total transmit power in the park" \
  --scenario urban_area \
  --strategy PRIORITY
```

Supported strategy values are defined by the command-line interface and schema layer. `AUTO` allows the Meta Agent and deterministic guards to select an eligible policy.

### Reset state before the run

```bash
python run_workflow.py \
  --intent "Improve RX power in the park" \
  --scenario urban_area \
  --reset-state
```

## 9. Active state

Default active-state file:

```text
state/active_intents.json
```

The file stores active and suppressed intents together with the effective network configuration. It is created automatically when needed.

To start a run from an empty state, use:

```text
--reset-state
```

## 10. Run outputs

Each execution creates a timestamped directory:

```text
results/YYYY-MM-DD_HH-MM-SS_runN/
├── input.json
├── agent_history.json
├── active_intents_before.json
├── active_intents_after.json
└── final_result.json
```

On failure:

```text
results/YYYY-MM-DD_HH-MM-SS_runN/error.json
```

### Output roles

- `input.json` records the intent, scenario, runtime, strategy, and surrogate metadata.
- `agent_history.json` records step inputs, outputs, timestamps, and status.
- `active_intents_before.json` captures state before execution.
- `active_intents_after.json` captures committed state after resolution.
- `final_result.json` contains the complete `WorkflowResult`.
- `error.json` contains the exception type and message for unsuccessful runs.
