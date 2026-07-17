# Datasets, Surrogate Models, Colab Validation, and Evaluation
## 1. Dataset schema

The uploaded open-area and urban-area sample files share 53 columns.

### Configuration and context features

```text
config_id
K_users
user_set_id
rx_power_thr_dBm
total_tx_power_watt

tx0_on ... tx3_on
tx0_P_dBm ... tx3_P_dBm
tx0_dAz ... tx3_dAz
tx0_dEl ... tx3_dEl
```

### Association outputs

```text
tx0_served_pct ... tx3_served_pct
```

### KPI targets

```text
Prx_min_dBm, Prx_p5_dBm, Prx_median_dBm, Prx_mean_dBm, Prx_max_dBm
SINR_min_dB, SINR_p5_dB, SINR_median_dB, SINR_mean_dB, SINR_max_dB
Thr_min_Mbps, Thr_p5_Mbps, Thr_median_Mbps, Thr_mean_Mbps, Thr_max_Mbps
ThrRR_min_Mbps, ThrRR_p5_Mbps, ThrRR_median_Mbps, ThrRR_mean_Mbps, ThrRR_max_Mbps
RSSI_min_dBm, RSSI_p5_dBm, RSSI_median_dBm, RSSI_mean_dBm, RSSI_max_dBm
rx_power_coverage_ratio
rx_power_cov_n
rx_power_out_n
```

The surrogate trainer uses both base features and engineered interactions such as active-TX count, active power statistics, angle spread, and power-angle products.

## 2. Open-area and urban-area models

The environments can have different propagation distributions, blockage behavior, coverage thresholds, and useful parameter ranges. The default design therefore trains:

```text
models/open_area_surrogate.joblib
models/urban_area_surrogate.joblib
```

A single combined model is possible, but it should include an explicit scenario representation and must be compared against separate models on held-out scenarios. A unified model should not be assumed better merely because it has more rows.


## 3. The current model family: LightGBM GBDT

The recommended first model is LightGBM using gradient-boosted decision trees. Surrogate training procedure trains one regressor per KPI.

## 4. Surrogate training procedure

### Step 1 — Discover files

The trainer accepts repeated files or directories and reads CSV/Parquet data.

### Step 2 — Sample across files

For very large data, `--sample-size` distributes a cap across the discovered files. This keeps different K-user files represented instead of taking only the beginning of one giant file.

### Step 3 — Sanitize configurations

- `tx_on` values become booleans.
- OFF TX sentinel powers such as `-9999` become zero-valued model features.
- total TX power is recomputed from the TX configuration.
- all-OFF configurations are removed.

### Step 4 — Preserve meaningful outage cases

Current implementation does not automatically trim the 0.1% and 99.9% KPI tails. Very low received power or SINR can be physically meaningful in outage scenarios. It removes non-finite values and invalid KPI domains instead.

### Step 5 — Engineer features

Examples:

```text
number of active TXs
mean and standard deviation of active-TX power
active angle means, ranges, and standard deviations
power × azimuth
power × elevation
azimuth × elevation
```

### Step 6 — Train one model per KPI

LightGBM uses early stopping on a validation set. The final test set is not used to choose the stopping point.

### Step 7 — Save model and metrics

```text
<model>.joblib
<model>.joblib.metrics.json
```

## 5. Colab notebook requirements

Notebook included in this repository:

```text
notebooks/Config_level_dataset_and_map_generation_code.ipynb
```

The uploaded notebook expects this archive in the Colab runtime:

```text
/content/boston_small_dm_export.zip
```

It unzips to:

```text
/content/deepmimo_scenarios
```

and loads:

```text
/content/deepmimo_scenarios/boston_small_export
```

### How to upload the ZIP

1. Open the notebook in Google Colab.
2. Open the Files panel on the left.
3. Upload `boston_small_dm_export.zip` directly under `/content/`.
4. Confirm the exact filename; otherwise edit the unzip cell.

Mounting Google Drive in cell 0 is optional unless the ZIP or outputs are stored in Drive.

## 6. Colab execution order

Run cells sequentially through the section:

```text
# Config-level coverage plotting and KPI stats
```

The notebook performs:

1. DeepMIMO installation;
2. scenario extraction and loading;
3. receiver filtering;
4. removal/reindexing of the satellite TX so four TXs remain;
5. antenna, OFDM, bandwidth, and channel-parameter setup;
6. baseline received-power analysis;
7. definition of the configuration-level KPI functions.

After the function-definition cell, use the `Example usage` block.

## 7. Entering an agent-produced configuration in Colab

The notebook uses:

```python
# Order: TX0, TX1, TX2, TX3
# Each angle tuple: (dAz, dEl)
# Use None for an OFF transmitter.

deltas_list = [
    (tx0_dAz, tx0_dEl),
    (tx1_dAz, tx1_dEl),
    (tx2_dAz, tx2_dEl),
    (tx3_dAz, tx3_dEl),
]

P_list_raw = [
    tx0_power_or_None,
    tx1_power_or_None,
    tx2_power_or_None,
    tx3_power_or_None,
]
```

Example:

```python
deltas_list = [(-15, 10), (15, 10), (0, 0), (10, 10)]
P_list_raw = [43, 43, None, 43]
```

Then run:

```python
out = plot_and_store_one_config(
    dataset_dm=dataset_dm,
    ch_params_list=ch_params_list,
    base_rot_list=base_rot_list,
    deltas_list=deltas_list,
    P_list_raw=P_list_raw,
    coverage_threshold_dBm=-60,
    BW=BW,
    NF_dB=7.0,
    N0_dBm_perHz=-174.0,
    s=6,
    save_dir="snapshots",
    tag="cfgA",
)
```

The function produces:

- a coverage map;
- coverage percentage;
- serving TX per receiver;
- RX power, SINR, throughput, round-robin throughput, and RSSI arrays;
- one configuration-level row;
- optional Parquet snapshot.

Important output fields:

```python
out["config_row"]["rx_power_coverage_ratio"]
out["config_row"]["SINR_mean_dB"]
out["config_row"]["Prx_p5_dBm"]
out["config_row"]["total_tx_power_watt"]
```

## 8. Comparing agent and Colab results

For each final configuration, create an evaluation record containing:

```text
run_id
scenario
intent
final configuration
surrogate-predicted KPIs
Colab/DeepMIMO KPIs
absolute error per KPI
whether the requested target is met under Colab recomputation
```