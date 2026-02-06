#!/usr/bin/env python3
"""
train_surrogate_v2.py

Train surrogate models for telecom base-station configuration optimization.
Uses LightGBM for accurate KPI prediction from configuration parameters.

Targets:
- Prx_p5_dBm (RX_POWER)
- SINR_p5_dB (SINR)
- ThrRR_p5 (THROUGHPUT_5P) - if available, else computed from SINR
- rx_power_coverage_ratio (RX_COVERAGE_RATIO)
- LOAD_IMBALANCE - computed from tx0..tx3_served_pct

Usage:
    python train_surrogate_v2.py --sample-size 1000000
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available, falling back to XGBoost")
    try:
        import xgboost as xgb
        HAS_XGBOOST = True
    except ImportError:
        HAS_XGBOOST = False
        print("Warning: XGBoost not available, falling back to sklearn")
        from sklearn.ensemble import HistGradientBoostingRegressor

load_dotenv()

# Configuration
BW_HZ_DEFAULT = 10_000_000  # 10 MHz for throughput calculation

# Physical bounds for sanity checks
PRX_MIN = float(os.getenv("PRX_MIN", "-140.0"))
PRX_MAX = float(os.getenv("PRX_MAX", "-40.0"))
SINR_MIN = float(os.getenv("SINR_MIN", "-20.0"))
SINR_MAX = float(os.getenv("SINR_MAX", "40.0"))
THR_MIN = float(os.getenv("THR_MIN", "0.0"))
THR_MAX = float(os.getenv("THR_MAX", "1000.0"))  # Mbps


class SurrogateTrainer:
    """Train surrogate models for KPI prediction."""
    
    def __init__(
        self,
        data_path: str,
        sample_size: int | None = None,
        random_seed: int = 42,
        chunk_size: int = 100000,
    ):
        """
        Initialize surrogate trainer.
        
        Args:
            data_path: Path to CSV or Parquet dataset
            sample_size: Number of rows to sample for training (None = all)
            random_seed: Random seed for reproducibility
            chunk_size: Chunk size for reading large files
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.chunk_size = chunk_size
        self.rng = np.random.default_rng(random_seed)
        
        # Feature columns (configuration parameters)
        self.feature_cols = [
            "user_set_id",
            "K_users",
            "rx_power_thr_dBm",
            "total_tx_power_watt",
            "tx0_on", "tx1_on", "tx2_on", "tx3_on",
            "tx0_P_dBm", "tx1_P_dBm", "tx2_P_dBm", "tx3_P_dBm",
            "tx0_dAz", "tx1_dAz", "tx2_dAz", "tx3_dAz",
            "tx0_dEl", "tx1_dEl", "tx2_dEl", "tx3_dEl",
        ]
        
        # Target columns (KPIs from simulator)
        self.target_cols = [
            "Prx_p5_dBm",
            "SINR_p5_dB",
            "rx_power_coverage_ratio",
            "tx0_served_pct",
            "tx1_served_pct",
            "tx2_served_pct",
            "tx3_served_pct",
        ]
        
        # Check for ThrRR_p5 column (may be named differently)
        self.has_throughput_col = False
        self.throughput_col = None
        
        self.models: Dict[str, any] = {}
        self.param_ranges: Dict[str, Dict[str, float]] = {}
        self.feature_names = None  # Will be set after feature engineering
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features to capture nonlinear interactions."""
        df = df.copy()
        
        # Average power across all ON transmitters
        on_mask = (df[[f"tx{i}_on" for i in range(4)]] == 1).values
        powers = df[[f"tx{i}_P_dBm" for i in range(4)]].values
        df["avg_power"] = (powers * on_mask).sum(axis=1) / (on_mask.sum(axis=1) + 1e-6)
        
        # Power variance (heterogeneity)
        df["power_std"] = df[[f"tx{i}_P_dBm" for i in range(4)]].std(axis=1)
        
        # Angle statistics
        df["avg_azimuth"] = df[[f"tx{i}_dAz" for i in range(4)]].mean(axis=1)
        df["avg_elevation"] = df[[f"tx{i}_dEl" for i in range(4)]].mean(axis=1)
        df["azimuth_range"] = df[[f"tx{i}_dAz" for i in range(4)]].max(axis=1) - df[[f"tx{i}_dAz" for i in range(4)]].min(axis=1)
        df["elevation_range"] = df[[f"tx{i}_dEl" for i in range(4)]].max(axis=1) - df[[f"tx{i}_dEl" for i in range(4)]].min(axis=1)
        
        # Power * angle interactions (for each TX)
        for i in range(4):
            df[f"tx{i}_P_x_Az"] = df[f"tx{i}_P_dBm"] * df[f"tx{i}_dAz"]
            df[f"tx{i}_P_x_El"] = df[f"tx{i}_P_dBm"] * df[f"tx{i}_dEl"]
            df[f"tx{i}_Az_x_El"] = df[f"tx{i}_dAz"] * df[f"tx{i}_dEl"]
        
        # Number of active transmitters
        df["n_active_tx"] = df[[f"tx{i}_on" for i in range(4)]].sum(axis=1)
        
        # Angle uniformity (all same = good coverage)
        df["azimuth_uniformity"] = -df[[f"tx{i}_dAz" for i in range(4)]].std(axis=1)
        
        return df
        
    def _sanitize_tx_params(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean TX parameters: OFF tx -> zero power/angles."""
        df = df.copy()
        for i in range(4):
            on_col = f"tx{i}_on"
            p_col = f"tx{i}_P_dBm"
            az_col = f"tx{i}_dAz"
            el_col = f"tx{i}_dEl"
            
            # Convert boolean to int if needed
            if df[on_col].dtype == bool:
                df[on_col] = df[on_col].astype(np.int8)
            else:
                df[on_col] = df[on_col].fillna(0).astype(np.int8)
            
            # If OFF, set params to 0
            off_mask = df[on_col] == 0
            df.loc[off_mask, p_col] = 0.0
            df.loc[off_mask, az_col] = 0.0
            df.loc[off_mask, el_col] = 0.0
            
            # Remove sentinel values (e.g., -9999)
            df.loc[df[p_col] < -1000, p_col] = 0.0
            
            # Clip to safe bounds
            df[p_col] = df[p_col].clip(lower=0.0, upper=100.0)
            df[az_col] = df[az_col].clip(lower=-180.0, upper=180.0)
            df[el_col] = df[el_col].clip(lower=-90.0, upper=90.0)
        
        return df
    
    def _clean_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter rows with valid target values."""
        df = df.copy()
        initial_count = len(df)
        
        # Convert to numeric
        for col in self.target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove extreme sentinel values
        sentinel_mask = np.ones(len(df), dtype=bool)
        for col in self.target_cols:
            if col in df.columns:
                col_mask = (df[col] >= -1e6) & (df[col] <= 1e6)
                sentinel_mask &= col_mask
        sentinel_removed = (~sentinel_mask).sum()
        df = df[sentinel_mask].copy()
        
        # Apply physical bounds
        mask = np.ones(len(df), dtype=bool)
        
        # Track what gets filtered
        filter_reasons = {}
        
        if "Prx_p5_dBm" in df.columns:
            prx_mask = df["Prx_p5_dBm"].between(PRX_MIN, PRX_MAX)
            filter_reasons["Prx_p5_dBm out of bounds"] = (~prx_mask).sum()
            mask &= prx_mask
        
        if "SINR_p5_dB" in df.columns:
            sinr_mask = df["SINR_p5_dB"].between(SINR_MIN, SINR_MAX)
            filter_reasons["SINR_p5_dB out of bounds"] = (~sinr_mask).sum()
            mask &= sinr_mask
        
        if "rx_power_coverage_ratio" in df.columns:
            cov_mask = df["rx_power_coverage_ratio"].between(0.0, 1.0)
            filter_reasons["rx_power_coverage_ratio out of bounds"] = (~cov_mask).sum()
            mask &= cov_mask
        
        for i in range(4):
            col = f"tx{i}_served_pct"
            if col in df.columns:
                served_mask = df[col].between(0.0, 100.0)
                filter_reasons[f"{col} out of bounds"] = (~served_mask).sum()
                mask &= served_mask
        
        # Check for throughput column
        if self.throughput_col and self.throughput_col in df.columns:
            thr_mask = df[self.throughput_col].between(THR_MIN, THR_MAX)
            filter_reasons[f"{self.throughput_col} out of bounds"] = (~thr_mask).sum()
            mask &= thr_mask
        
        # Drop rows with NaN in any target
        required_targets = [c for c in self.target_cols if c in df.columns]
        nan_mask = df[required_targets].notna().all(axis=1)
        filter_reasons["NaN in targets"] = (~nan_mask).sum()
        mask &= nan_mask
        
        # Apply final mask
        df_clean = df.loc[mask].reset_index(drop=True)
        
        # Print detailed cleaning report
        print("\n=== Data Cleaning Report ===")
        print(f"Initial rows: {initial_count:,}")
        if sentinel_removed > 0:
            print(f"  Removed sentinel values: {sentinel_removed:,}")
        for reason, count in filter_reasons.items():
            if count > 0:
                print(f"  {reason}: {count:,}")
        print(f"Final valid rows: {len(df_clean):,} ({len(df_clean)/initial_count*100:.2f}%)")
        print(f"Total removed: {initial_count - len(df_clean):,} ({(initial_count - len(df_clean))/initial_count*100:.2f}%)")
        print("=" * 30)
        
        return df_clean
    
    def _compute_param_ranges(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute parameter ranges from dataset."""
        ranges = {}
        
        # Power ranges (only for ON transmitters)
        p_vals = []
        for i in range(4):
            on_mask = df[f"tx{i}_on"] == 1
            p_vals.extend(df.loc[on_mask, f"tx{i}_P_dBm"].dropna().tolist())
        if p_vals:
            ranges["power"] = {"min": float(np.min(p_vals)), "max": float(np.max(p_vals))}
        else:
            ranges["power"] = {"min": 30.0, "max": 50.0}  # Default
        
        # Azimuth ranges
        az_vals = []
        for i in range(4):
            az_vals.extend(df[f"tx{i}_dAz"].dropna().tolist())
        if az_vals:
            ranges["dAz"] = {"min": float(np.min(az_vals)), "max": float(np.max(az_vals))}
        else:
            ranges["dAz"] = {"min": -180.0, "max": 180.0}
        
        # Elevation ranges
        el_vals = []
        for i in range(4):
            el_vals.extend(df[f"tx{i}_dEl"].dropna().tolist())
        if el_vals:
            ranges["dEl"] = {"min": float(np.min(el_vals)), "max": float(np.max(el_vals))}
        else:
            ranges["dEl"] = {"min": -10.0, "max": 10.0}
        
        return ranges
    
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess dataset with chunked reading and sampling."""
        print(f"Loading data from: {self.data_path}")
        
        file_ext = os.path.splitext(self.data_path.lower())[1]
        
        # Detect throughput column name
        if file_ext == ".parquet":
            sample_df = pd.read_parquet(self.data_path, engine='pyarrow')
            sample_cols = sample_df.columns.tolist()
        else:
            sample_df = pd.read_csv(self.data_path, nrows=100)
            sample_cols = sample_df.columns.tolist()
        
        # Check for throughput column variants
        thr_variants = ["ThrRR_p5_Mbps", "ThrRR_p5", "Thr_p5_Mbps", "throughput_p5", "thr_rr_p5"]
        for variant in thr_variants:
            if variant in sample_cols:
                self.throughput_col = variant
                self.has_throughput_col = True
                if variant not in self.target_cols:
                    self.target_cols.append(variant)
                print(f"Found throughput column: {variant}")
                break
        
        # Check which columns are available
        available_features = [c for c in self.feature_cols if c in sample_cols]
        available_targets = [c for c in self.target_cols if c in sample_cols]
        
        missing_features = set(self.feature_cols) - set(available_features)
        missing_targets = set(self.target_cols) - set(available_targets)
        
        if missing_features:
            print(f"Warning: Missing feature columns: {missing_features}")
        if missing_targets:
            print(f"Warning: Missing target columns: {missing_targets}")
        
        all_cols = available_features + available_targets
        
        # Load data with chunking and sampling
        dfs = []
        total_rows = 0
        kept_rows = 0
        
        if file_ext == ".parquet":
            # For Parquet, we can use pandas directly
            df_full = pd.read_parquet(self.data_path, columns=all_cols, engine='pyarrow')
            total_rows = len(df_full)
            
            if self.sample_size and self.sample_size < total_rows:
                print(f"Sampling {self.sample_size} rows from {total_rows} total rows")
                indices = self.rng.choice(total_rows, size=self.sample_size, replace=False)
                df_full = df_full.iloc[indices]
            
            dfs.append(df_full)
        
        else:
            # For CSV, use chunked reading
            chunks = pd.read_csv(
                self.data_path,
                usecols=all_cols,
                chunksize=self.chunk_size,
            )
            
            for chunk in chunks:
                total_rows += len(chunk)
                
                # Sample from chunk if needed
                if self.sample_size and kept_rows >= self.sample_size:
                    break
                
                if self.sample_size:
                    # Reservoir sampling logic
                    chunk_sample_size = min(len(chunk), self.sample_size - kept_rows)
                    if chunk_sample_size < len(chunk):
                        chunk = chunk.sample(n=chunk_sample_size, random_state=self.random_seed)
                
                dfs.append(chunk)
                kept_rows += len(chunk)
                
                if total_rows % 1000000 == 0:
                    print(f"  Processed {total_rows:,} rows, kept {kept_rows:,}")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df):,} rows from {total_rows:,} total")
        
        # Sanitize features
        print("Sanitizing TX parameters...")
        df = self._sanitize_tx_params(df)
        
        # Engineer features
        print("Engineering features (interactions, statistics)...")
        df = self._engineer_features(df)
        
        # Clean targets
        df = self._clean_targets(df)
        
        # Compute parameter ranges
        self.param_ranges = self._compute_param_ranges(df)
        print(f"Parameter ranges: {self.param_ranges}")
        
        # Update feature/target cols to only available ones
        self.feature_cols = available_features
        self.target_cols = available_targets
        
        return df
    
    def train_models(self, df: pd.DataFrame) -> None:
        """Train separate regression models for each KPI target with train/val split."""
        print("\nTraining surrogate models...")
        
        # Use all available features (base + engineered)
        engineered_features = [
            "avg_power", "power_std", "avg_azimuth", "avg_elevation",
            "azimuth_range", "elevation_range", "n_active_tx", "azimuth_uniformity"
        ]
        
        # Add interaction features
        for i in range(4):
            engineered_features.extend([
                f"tx{i}_P_x_Az", f"tx{i}_P_x_El", f"tx{i}_Az_x_El"
            ])
        
        # Combine base + engineered features
        all_features = self.feature_cols + [f for f in engineered_features if f in df.columns]
        self.feature_names = all_features
        
        X = df[all_features].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Feature matrix shape: {X.shape} (base: {len(self.feature_cols)}, engineered: {len(all_features) - len(self.feature_cols)})")
        
        # Train/validation split (90/10)
        n_train = int(len(X) * 0.9)
        indices = np.arange(len(X))
        self.rng.shuffle(indices)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        X_train = X[train_idx]
        X_val = X[val_idx]
        
        print(f"Train size: {len(X_train):,}, Validation size: {len(X_val):,}")
        
        # Train separate model for each target
        for target in self.target_cols:
            if target not in df.columns:
                continue
            
            print(f"\nTraining model for {target}...")
            y = df[target].values.astype(np.float32)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            # Choose model based on availability
            if HAS_LIGHTGBM:
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 127,  # Increased from 63
                    'max_depth': 12,     # Added explicit depth
                    'learning_rate': 0.03,  # Slightly lower for better convergence
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_data_in_leaf': 50,
                    'lambda_l1': 0.1,    # Regularization
                    'lambda_l2': 0.1,
                    'verbose': -1,
                    'random_state': self.random_seed,
                }
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train with early stopping
                callbacks = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=100)
                ]
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'valid'],
                    callbacks=callbacks,
                )
                
            elif HAS_XGBOOST:
                model = xgb.XGBRegressor(
                    n_estimators=1000,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    random_state=self.random_seed,
                    n_jobs=-1,
                    early_stopping_rounds=50,
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False,
                )
                
            else:
                # Fallback to sklearn
                model = HistGradientBoostingRegressor(
                    max_iter=1000,
                    max_depth=8,
                    learning_rate=0.05,
                    random_state=self.random_seed,
                    early_stopping=True,
                    validation_fraction=0.1,
                )
                model.fit(X_train, y_train)
            
            self.models[target] = model
            
            # Validation metrics
            y_pred_train = self._predict_single_model(model, X_train[:5000])
            mae_train = np.mean(np.abs(y_pred_train - y_train[:5000]))
            
            y_pred_val = self._predict_single_model(model, X_val[:5000])
            mae_val = np.mean(np.abs(y_pred_val - y_val[:5000]))
            
            print(f"  ✓ Model trained.")
            print(f"    Train MAE: {mae_train:.4f}")
            print(f"    Val MAE:   {mae_val:.4f}")
            
            if mae_val > mae_train * 2.0:
                print(f"    ⚠ Warning: Validation error is 2x train error (possible overfitting)")
        
        print(f"\nTrained {len(self.models)} models successfully")
    
    def _predict_single_model(self, model, X: np.ndarray) -> np.ndarray:
        """Predict using a single model (handles different model types)."""
        if HAS_LIGHTGBM and isinstance(model, lgb.Booster):
            return model.predict(X)
        else:
            return model.predict(X)
    
    def save(self, output_path: str) -> None:
        """Save trained models and metadata."""
        artifact = {
            "feature_cols": self.feature_cols,
            "feature_names": self.feature_names,  # Save all feature names
            "target_cols": self.target_cols,
            "models": self.models,
            "param_ranges": self.param_ranges,
            "bw_hz": BW_HZ_DEFAULT,
            "has_throughput_col": self.has_throughput_col,
            "throughput_col": self.throughput_col,
            "version": "2.1",
            "notes": (
                "Surrogate models trained with LightGBM/XGBoost + feature engineering. "
                "Predicts: Prx_p5_dBm, SINR_p5_dB, rx_power_coverage_ratio, tx*_served_pct. "
                "Features include interactions (power*angle) and statistics. "
                "LOAD_IMBALANCE computed from served_pct at inference. "
                "Throughput computed from SINR via Shannon if not in dataset."
            ),
            "bounds": {
                "PRX_MIN": PRX_MIN,
                "PRX_MAX": PRX_MAX,
                "SINR_MIN": SINR_MIN,
                "SINR_MAX": SINR_MAX,
            },
        }
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        joblib.dump(artifact, output_path)
        print(f"\nSaved surrogate model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train surrogate models for base-station optimization")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset (CSV or Parquet). Defaults to DATA_PATH env var.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save trained model. Defaults to MODEL_PATH env var.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of rows to sample for training (default: use all)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Chunk size for reading large files (default: 100000)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Get paths from args or env
    data_path = args.data_path or os.getenv("DATA_PATH", "").strip()
    model_path = args.model_path or os.getenv("MODEL_PATH", "./models/surrogate.joblib").strip()
    
    if not data_path:
        raise ValueError("DATA_PATH must be provided via --data-path or DATA_PATH env var")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Configuration:")
    print(f"  Data path: {data_path}")
    print(f"  Model path: {model_path}")
    print(f"  Sample size: {args.sample_size or 'ALL'}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Random seed: {args.random_seed}")
    print()
    
    # Train surrogate
    trainer = SurrogateTrainer(
        data_path=data_path,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        chunk_size=args.chunk_size,
    )
    
    df = trainer.load_data()
    trainer.train_models(df)
    trainer.save(model_path)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
