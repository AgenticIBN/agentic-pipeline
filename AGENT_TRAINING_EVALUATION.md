# Agent Training, Surrogate Model, Coverage, and Evaluation

This document explains the data, model training, agent behavior, coverage notebook, evaluation workflow, and known issues.

## 1. What “agent training” means in this project

The phrase “agent training” can be misleading. In this repository, it may refer to two different things:

1. **Surrogate model training**
   - A LightGBM-based model is trained on simulator or measurement data.
   - It predicts KPI values from base-station configuration parameters.
   - This is actual machine-learning training.

2. **LLM agent behavior design**
   - Intent parsing, conflict detection, and resolution may be implemented through prompts, schemas, wrappers, and rules.
   - Unless there is a separate fine-tuning dataset and training script, these LLM agents are not fine-tuned.
   - Document this as agent configuration or prompt engineering rather than model training.

Recommended wording:

> The trained component is the surrogate KPI prediction model. The LLM agents are configured with task-specific prompts, structured output schemas, and workflow logic.

## 2. External artifact links

Add Drive links for data and notebooks here.

| Artifact | Link | Description |
|---|---|---|
| Raw simulator dataset | `[TODO_RAW_DATASET_DRIVE_LINK]` | Original simulator output or measurement data. |
| Processed training dataset | `[TODO_PROCESSED_DATASET_DRIVE_LINK]` | Optional cleaned or feature-engineered data. |
| Trained surrogate model | `[TODO_SURROGATE_MODEL_DRIVE_LINK]` | Example: `models/surrogate.joblib`. |
| Coverage calculation notebook | `[TODO_COVERAGE_NOTEBOOK_LINK]` | Manual coverage-percentage notebook. |
| Benchmark results | `[TODO_BENCHMARK_RESULTS_DRIVE_LINK]` | CSV/log/plot outputs from tests. |

## 3. Surrogate model training

The hybrid version uses a surrogate model to estimate network KPIs from TX configuration. The older documentation names `train_surrogate_v2.py` as the training script and `models/surrogate.joblib` as the model output. Verify these paths in the current code before finalizing this README.

Verification commands:

```bash
grep -R "LightGBM\|LGBM\|lgb\|joblib.dump\|surrogate" . --include="*.py"
grep -R "MODEL_PATH\|DATA_PATH\|surrogate.joblib" . --include="*.py" --include="*.md"
```

Fill these fields after verification:

```text
Active training script: [TODO_TRAINING_SCRIPT]
Input dataset path:     [TODO_DATASET_PATH]
Model output path:      [TODO_MODEL_OUTPUT_PATH]
```

### 3.1 Typical input features

The surrogate model is expected to use scenario features and TX configuration features.

```python
feature_cols = [
    "user_set_id",
    "K_users",
    "rx_power_thr_dBm",
    "total_tx_power_watt",

    "tx0_on", "tx0_P_dBm", "tx0_dAz", "tx0_dEl",
    "tx1_on", "tx1_P_dBm", "tx1_dAz", "tx1_dEl",
    "tx2_on", "tx2_P_dBm", "tx2_dAz", "tx2_dEl",
    "tx3_on", "tx3_P_dBm", "tx3_dAz", "tx3_dEl"
]
```

### 3.2 Typical target KPIs

```python
target_cols = [
    "Prx_p5_dBm",
    "SINR_p5_dB",
    "rx_power_coverage_ratio",
    "tx0_served_pct",
    "tx1_served_pct",
    "tx2_served_pct",
    "tx3_served_pct"
]
```

Common KPI mapping:

| Internal KPI | Dataset column / computation |
|---|---|
| `RX_POWER` | `Prx_p5_dBm` |
| `SINR` | `SINR_p5_dB` |
| `RX_COVERAGE_RATIO` | `rx_power_coverage_ratio` |
| `LOAD_IMBALANCE` | computed from TX served percentages |
| `THROUGHPUT_5P` | derived from SINR, if implemented |

### 3.3 Training command template

```bash
python3 [TODO_TRAINING_SCRIPT] \
  --data-path [TODO_DATASET_PATH] \
  --model-path [TODO_MODEL_OUTPUT_PATH] \
  --sample-size 50000
```

If the active script does not support these arguments, replace the command with the actual interface.

### 3.4 Model validation checklist

After training, record:

| Metric | Value | Notes |
|---|---:|---|
| RX_POWER MAE/RMSE | `[TODO]` | dB |
| SINR MAE/RMSE | `[TODO]` | dB |
| Coverage MAE/RMSE | `[TODO]` | ratio or percentage |
| Train/test split | `[TODO]` | random, scenario-based, or time-based |
| Dataset size | `[TODO]` | number of rows |

Do not claim a specific error value unless it is reproduced from the current training run.

## 4. Coverage calculation notebook

The project also has a notebook that manually computes coverage percentage in the environment. Put the Drive link here:

```text
Coverage notebook: [TODO_COVERAGE_NOTEBOOK_LINK]
```

Recommended notebook instructions:

1. Open the notebook in Jupyter or Google Colab.
2. Set the data path or upload the required CSV/Parquet file.
3. Set the RX power threshold used for the experiment.
4. Run all cells.
5. Record the final coverage percentage and the threshold used.

Example coverage formula:

```python
RX_POWER_THRESHOLD_DBM = -85
covered = df["rx_power_dBm"] >= RX_POWER_THRESHOLD_DBM
coverage_percent = covered.mean() * 100
print(f"Coverage: {coverage_percent:.2f}%")
```

If the notebook computes coverage per cell, region, or scenario, document the grouping column explicitly:

```python
coverage_by_region = (
    df.assign(covered=df["rx_power_dBm"] >= RX_POWER_THRESHOLD_DBM)
      .groupby("region")["covered"]
      .mean()
      .mul(100)
      .reset_index(name="coverage_percent")
)
```

## 5. Agent behavior documentation

### 5.1 Intent Parser Agent

Expected role:

- Parse natural-language requests.
- Extract target area, KPI, threshold, operator, time window, and priority.
- Return structured output with confidence.

Recommended test cases:

```text
"Increase RX power in Kartal to at least -55 dBm. Priority HIGH."
"Reduce power consumption in cell1 by 20 percent. Priority MEDIUM."
"Improve SINR above 15 dB in Kadikoy between 2 PM and 4 PM."
```

### 5.2 Optimization Agent

Expected role:

- Propose network configuration changes.
- Predict or retrieve KPI estimates.
- Explain why the proposed changes should satisfy the intent.

Current known issue:

> The Optimization Agent is not currently reliable. The likely failure modes are wrong active file, mismatched output schema, missing model path, inconsistent KPI naming, or unrealistic LLM-generated KPI predictions.

Debug checklist:

```bash
# 1. Identify active optimization imports
grep -R "optimization_agent" . --include="*.py"

# 2. Check expected output field names
grep -R "config_changes\|changes\|predicted_kpis\|expected_kpis" . --include="*.py"

# 3. Check model loading path for hybrid version
grep -R "surrogate.joblib\|MODEL_PATH\|joblib.load" . --include="*.py"

# 4. Run a minimal smoke test
python3 [TODO_ACTIVE_WORKFLOW_FILE] \
  --intent "Test: increase RX power in Kartal to at least -70 dBm. Priority LOW." \
  --strategy PRIORITY
```

### 5.3 Conflict Detector Agent

Expected role:

- Detect conflicts between the new result and active results.
- Classify conflict types.
- Assign severity.
- Recommend `PRIORITY` or `WEIGHTED_MERGE` when applicable.

Recommended validation:

- Create one active intent that increases `tx0_P_dBm`.
- Submit a new intent that decreases `tx0_P_dBm`.
- Confirm that the detector returns `PARAMETER_CONFLICT`.

### 5.4 Resolution Agents

Priority resolution should select the highest-priority intent:

```text
CRITICAL > HIGH > MEDIUM > LOW
```

Weighted merge should combine numeric changes using priority weights:

```python
weights = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
merged_change = sum(weight_i * change_i) / sum(weight_i)
```

Boolean changes should use weighted voting or a documented deterministic rule.

## 6. Evaluation workflow

A minimal evaluation should include:

1. **Intent parsing checks** — verify structured output for a small set of English and Turkish intents.
2. **Optimization sanity checks** — verify that proposed changes are within allowed ranges.
3. **KPI validation** — compare predicted KPIs against surrogate, simulator, or benchmark data.
4. **Conflict detection tests** — manually create opposite-direction and same-direction conflicts.
5. **Resolution tests** — verify priority winner and weighted merge calculations.
6. **End-to-end smoke test** — run a full workflow and inspect the saved JSON.

Suggested results table:

| Test | Input | Expected | Actual | Status |
|---|---|---|---|---|
| Intent parser | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Optimization smoke test | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Conflict detection | `[TODO]` | `PARAMETER_CONFLICT` | `[TODO]` | `[TODO]` |
| Priority resolution | `[TODO]` | CRITICAL wins | `[TODO]` | `[TODO]` |
| Weighted merge | `[TODO]` | weighted average | `[TODO]` | `[TODO]` |
| Coverage notebook | `[TODO]` | coverage percentage | `[TODO]` | `[TODO]` |

## 7. Known issues and TODOs

### 7.1 Optimization Agent reliability

Status: **Known issue**

Action items:

- Confirm active optimization implementation.
- Standardize output schema across optimization, conflict detection, and resolution.
- Decide whether the final system should be hybrid or pure LLM.
- If hybrid, make sure the surrogate model is loaded and validated.
- If pure LLM, do not claim quantitative KPI accuracy without simulator or benchmark validation.

### 7.2 Conflicting old documentation

Status: **Known issue**

The old documents describe different active files and different architecture assumptions. Use this README set as the cleaned structure, then update the TODOs after verifying the code.

### 7.3 Large artifacts

Status: **Repository hygiene**

Move large datasets, generated workflow outputs, logs, and notebooks to Drive unless they are small and intentionally included as examples.
