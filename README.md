# Agentic IBN — Conflict-Aware 6G Network Orchestration

This repository implements a hybrid intent-based networking system for translating natural-language network goals into validated transmitter configurations. The system uses **Agno Agents** for semantic reasoning and decision-facing tasks, an **Agno Workflow** for orchestration, deterministic components for numerical and safety-critical operations, and scenario-specific surrogate models for fast KPI prediction.

```text
Natural-language intent
        ↓
Agno Intent Parser Agent
        ↓
Agno Optimization Agent
        ↓
Constraint validation + surrogate feedback loop
        ↓
Agno Conflict Detector Agent + deterministic predicate verification
        ↓
Dynamic Meta Agent
   ├── Priority Resolution Agent (CRS-1)
   └── Weighted Merge Agent (CRS-2)
        ↓
Final surrogate validation
        ↓
Agno Reasoning Agent
        ↓
Persistent state and per-run history
```

## Core design

The implementation separates responsibilities according to their reliability requirements:

- Agno Agents interpret intents, propose network configurations, select conflict-resolution policies, audit decisions, and generate operator-facing explanations.
- Deterministic modules enforce hard transmitter commands, configuration bounds, objective scoring, conflict predicates, conflict-resolution arithmetic, and persistence rules.
- The surrogate model predicts wireless KPIs for each validated configuration.
- The Agno Workflow executes the required stages in a fixed, traceable order.

## Documentation

1. [`docs/ARCHITECTURE_AND_AGENTS.md`](docs/ARCHITECTURE_AND_AGENTS.md) — system architecture, Agno agent responsibilities, deterministic safety boundaries, orchestration, and state flow.
2. [`docs/REPOSITORY_AND_EXECUTION.md`](docs/REPOSITORY_AND_EXECUTION.md) — repository structure, installation, configuration, training, execution, testing, and generated outputs.
3. [`docs/DATASETS_MODELS_COLAB_AND_EVALUATION.md`](docs/DATASETS_MODELS_COLAB_AND_EVALUATION.md) — dataset schema, surrogate training, model evaluation, and Colab-based simulator validation.

## Main entry points

```text
run_workflow.py          Run the end-to-end Agentic IBN workflow
train_surrogate.py       Train a scenario-specific surrogate artifact
benchmark_surrogates.py Compare supported tabular model families
```

## Minimal setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
cp config/model_registry.example.json config/model_registry.json
```

Set `GROQ_API_KEY` in `.env` and register the local surrogate artifact paths in `config/model_registry.json`.

## Train a surrogate model

```bash
python train_surrogate.py \
  --data datasets/urban_area \
  --scenario urban_area \
  --model-path models/urban_area_surrogate.joblib \
  --model-family lightgbm \
  --sample-size 2000000 \
  --split-strategy auto \
  --num-threads 4
```
After trying with 2 million samples, you can increment sample size to 10 million etc.

## Run the workflow

```bash
python run_workflow.py \
  --intent "Improve RX power in Kadikoy to at least -95 dBm" \
  --scenario urban_area
```

The default runtime is `agno`. It invokes the Agno Workflow and all applicable Agno Agents. The `deterministic` runtime is provided for offline testing and requires a prepared `ParsedIntent` JSON file.

## Generated outputs

Each run creates a timestamped directory under `results/` containing the input, agent history, state snapshots, and final result. Active intents are maintained in `state/active_intents.json`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -ra
```

