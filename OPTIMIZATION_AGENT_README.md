# Telecom Base-Station Optimization Agent

A complete ML-based optimization system for telecom base-station configuration using surrogate models and local search.

## Overview

This implementation provides:

1. **Surrogate Model Training** - Train LightGBM models on large datasets to predict KPIs from configuration parameters
2. **Optimization Agent** - Use local search on surrogate predictions to find optimal configurations
3. **Constraint Satisfaction** - Ensure all hard constraints are met while optimizing objectives
4. **Full Pipeline** - Train, test, and deploy in one unified system

## Architecture

```
Dataset (CSV/Parquet)
    ↓
train_surrogate_v2.py → Trains LightGBM models
    ↓
models/surrogate.joblib (saved model)
    ↓
optimization_agent_v2.py → Loads model & runs optimization
    ↓
Optimized Configuration (JSON)
```

## Features

### Surrogate Model (`train_surrogate_v2.py`)

- **Scalable Data Loading**: Chunked reading for datasets with tens of millions of rows
- **Sampling Support**: Train on subset for faster iteration (e.g., 1M rows)
- **Multiple Model Backends**: LightGBM (preferred), XGBoost, or sklearn fallback
- **Automatic Feature Engineering**: Handles TX on/off states, clipping, sentinel values
- **Target KPIs**:
  - `Prx_p5_dBm` (RX_POWER)
  - `SINR_p5_dB` (SINR)
  - `ThrRR_p5` or computed from SINR (THROUGHPUT_5P)
  - `rx_power_coverage_ratio` (RX_COVERAGE_RATIO)
  - `tx0..tx3_served_pct` → `LOAD_IMBALANCE` (computed)

### Optimization Agent (`optimization_agent_v2.py`)

- **SurrogateModel Class**: Clean interface for KPI prediction
- **OptimizationAgent Class**: Local search optimizer with:
  - Bounded parameter deltas (power, azimuth, elevation)
  - Multi-objective scoring
  - Hard constraint checking
  - Guardrails (±3dB power, ±2° tilt, ±10° azimuth)
- **Generates NEW Configurations**: Does NOT simply pick existing dataset rows
- **Output Format**: Structured JSON with before/after/changes

## Installation

```bash
# Install dependencies
pip install pandas numpy scikit-learn joblib python-dotenv lightgbm

# Optional: For faster training
pip install pyarrow  # For Parquet support

# Optional: Alternative to LightGBM
pip install xgboost
```

## Configuration

Create a `.env` file with:

```bash
# Path to dataset (CSV or Parquet)
DATA_PATH=/path/to/kpi_K800.parquet

# Path to save/load trained model
MODEL_PATH=./models/surrogate.joblib

# Optional: Physical bounds for sanity checks
PRX_MIN=-140.0
PRX_MAX=-40.0
SINR_MIN=-20.0
SINR_MAX=40.0
```

## Usage

### 1. Train Surrogate Models

```bash
# Train on full dataset (may take time)
python train_surrogate_v2.py

# Train on 1M rows for faster iteration
python train_surrogate_v2.py --sample-size 1000000

# Specify paths manually
python train_surrogate_v2.py \
    --data-path /path/to/data.parquet \
    --model-path ./models/my_model.joblib \
    --sample-size 500000
```

### 2. Test Optimization Agent

```bash
# Run optimization with example scenarios
python optimization_agent_v2.py
```

### 3. Full Pipeline (Train + Test)

```bash
# Train and test
python run_full_pipeline.py --train --test --sample-size 1000000

# Train only
python run_full_pipeline.py --train --sample-size 2000000

# Test only (uses existing model)
python run_full_pipeline.py --test
```

## Input Format

Intent JSON structure:

```json
{
  "test_name": "improve_rx_power",
  "target_area": "sector_A",
  "target_kpis": ["RX_POWER", "THROUGHPUT_5P"],
  "kpi_thresholds": [
    {
      "kpi": "RX_POWER",
      "op": "GTE",
      "value": -95.0,
      "unit": "dBm"
    }
  ],
  "priority": "HIGH",
  "confidence": 0.95,
  "current_config": {
    "tx0_on": true,
    "tx0_P_dBm": 43.0,
    "tx0_dAz": 0.0,
    "tx0_dEl": 0.0,
    "tx1_on": true,
    "tx1_P_dBm": 43.0,
    "tx1_dAz": 120.0,
    "tx1_dEl": 0.0,
    "tx2_on": true,
    "tx2_P_dBm": 43.0,
    "tx2_dAz": -120.0,
    "tx2_dEl": 0.0,
    "tx3_on": false,
    "tx3_P_dBm": 0.0,
    "tx3_dAz": 0.0,
    "tx3_dEl": 0.0
  },
  "k_users": 800,
  "user_set_id": 0
}
```

### Supported KPIs

- `RX_POWER` - Received power (Prx_p5_dBm)
- `SINR` - Signal-to-interference-plus-noise ratio
- `THROUGHPUT_5P` - 5th percentile throughput
- `RX_COVERAGE_RATIO` - Coverage ratio
- `SERVED_USERS` - Load balancing (via LOAD_IMBALANCE)

### Supported Operators

- `GTE` - Greater than or equal
- `GT` - Greater than
- `LTE` - Less than or equal
- `LT` - Less than
- `BETWEEN` - Within range (requires value_low and value_high)
- `DELTA_UP` - Increase by delta from baseline
- `DELTA_DOWN` - Decrease by delta from baseline

## Output Format

```json
{
  "test_name": "improve_rx_power",
  "passed": true,
  "input": { /* echo of input intent */ },
  "output": {
    "selected_config_id": 20260206143052,
    "current_config_id": null,
    "changes": [
      {
        "param": "tx0_P_dBm",
        "before": 43.0,
        "change": 2.0,
        "unit": "dBm"
      }
    ],
    "expected_kpis": {
      "RX_POWER": -92.5,
      "SINR": 12.3,
      "THROUGHPUT_5P": 45.2,
      "LOAD_IMBALANCE": 8.5,
      "RX_COVERAGE_RATIO": 0.92
    },
    "current_kpis": {
      "RX_POWER": -94.2,
      "SINR": 11.8,
      "THROUGHPUT_5P": 42.1,
      "LOAD_IMBALANCE": 10.2,
      "RX_COVERAGE_RATIO": 0.89
    },
    "constraints_satisfied": true
  }
}
```

## Dataset Requirements

The CSV or Parquet dataset must contain:

### Configuration Features (per TX 0..3)
- `txi_on` - TX on/off (bool or 0/1)
- `txi_P_dBm` - Power in dBm
- `txi_dAz` - Azimuth offset in degrees
- `txi_dEl` - Elevation offset in degrees

### Context Features
- `user_set_id` - User distribution identifier
- `K_users` - Number of users (e.g., 800)
- `rx_power_thr_dBm` - RX power threshold
- `total_tx_power_watt` - Total TX power

### Target KPIs (from simulator)
- `Prx_p5_dBm` - 5th percentile RX power
- `SINR_p5_dB` - 5th percentile SINR
- `ThrRR_p5` or `Thr_p5_Mbps` - 5th percentile throughput (optional)
- `rx_power_coverage_ratio` - Coverage ratio
- `tx0_served_pct` .. `tx3_served_pct` - Percentage of users served by each TX

## Algorithm Details

### Training Phase

1. **Data Loading**: Chunked reading with optional sampling
2. **Feature Sanitization**: 
   - OFF transmitters → power/angles = 0
   - Remove sentinel values (-9999)
   - Clip to physical bounds
3. **Target Cleaning**:
   - Filter unrealistic values
   - Remove NaN rows
4. **Model Training**: Separate LightGBM regressor per KPI
5. **Save**: Joblib pickle with metadata

### Optimization Phase

1. **Initialization**: Load surrogate model, compute current KPIs
2. **Seed Generation**:
   - Use current config if available
   - Generate random seeds for diversity
3. **Local Search**: For each seed:
   - Apply random parameter deltas
   - Predict KPIs using surrogate
   - Score based on objectives + constraints
   - Keep best valid configuration
4. **Guardrails**: Limit changes from current config
5. **Output**: Best config with changes and expected KPIs

### Scoring Function

```python
score = Σ (priority_weight * constraint_margin) +
        α₁ * RX_POWER +
        α₂ * SINR +
        α₃ * THROUGHPUT_5P +
        α₄ * (-LOAD_IMBALANCE) +
        penalty_for_SINR_degradation
```

## Performance Tips

1. **Training Speed**:
   - Use `--sample-size 1000000` for initial experiments
   - Increase for production accuracy
   - Use Parquet format for faster I/O

2. **Optimization Speed**:
   - Reduce `search_iterations` (default 1000)
   - Limit number of seed configs

3. **Memory Usage**:
   - Adjust `chunk_size` in trainer (default 100k)
   - Process data in batches

## Files

- `train_surrogate_v2.py` - Surrogate model trainer
- `optimization_agent_v2.py` - Optimization agent with SurrogateModel class
- `run_full_pipeline.py` - End-to-end training and testing
- `models/surrogate.joblib` - Trained model (generated)
- `optimization_results/` - Test results (generated)

## Legacy Files

The workspace also contains older implementations:
- `train_surrogate.py` - Original SGDRegressor version
- `optimization_agent.py` - Original with Agno agent wrapper

The new `_v2` versions are recommended for production use.

## Example Workflow

```bash
# 1. Set up environment
cat > .env << EOF
DATA_PATH=/Users/burhan/Downloads/kpi_K800.parquet
MODEL_PATH=./models/surrogate.joblib
EOF

# 2. Train surrogate (1M samples for speed)
python train_surrogate_v2.py --sample-size 1000000

# 3. Test with example scenarios
python optimization_agent_v2.py

# 4. Or run full pipeline
python run_full_pipeline.py --train --test --sample-size 1000000
```

## Troubleshooting

### Model file not found
```
Error: Model file not found: ./models/surrogate.joblib
```
**Solution**: Train the model first using `python train_surrogate_v2.py`

### DATA_PATH not set
```
ValueError: DATA_PATH must be set in .env file
```
**Solution**: Create `.env` file with `DATA_PATH=/path/to/data.parquet`

### Missing columns
```
ValueError: Dataset missing columns
```
**Solution**: Check your dataset has all required feature and target columns

### LightGBM not available
The code automatically falls back to XGBoost or sklearn if LightGBM is not installed.

## License

MIT

## Author

Optimization Agent v2 - February 2026
