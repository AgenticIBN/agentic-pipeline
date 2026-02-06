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
        
        # Convert to numeric
        for col in self.target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove extreme sentinel values
        for col in self.target_cols:
            if col in df.columns:
                df.loc[(df[col] < -1e6) | (df[col] > 1e6), col] = np.nan
        
        # Apply physical bounds
        mask = np.ones(len(df), dtype=bool)
        
        if "Prx_p5_dBm" in df.columns:
            mask &= df["Prx_p5_dBm"].between(PRX_MIN, PRX_MAX)
        
        if "SINR_p5_dB" in df.columns:
            mask &= df["SINR_p5_dB"].between(SINR_MIN, SINR_MAX)
        
        if "rx_power_coverage_ratio" in df.columns:
            mask &= df["rx_power_coverage_ratio"].between(0.0, 1.0)
        
        for i in range(4):
            col = f"tx{i}_served_pct"
            if col in df.columns:
                mask &= df[col].between(0.0, 100.0)
        
        # Check for throughput column
        if self.throughput_col and self.throughput_col in df.columns:
            mask &= df[self.throughput_col].between(THR_MIN, THR_MAX)
        
        # Drop rows with NaN in any target
        required_targets = [c for c in self.target_cols if c in df.columns]
        mask &= df[required_targets].notna().all(axis=1)
        
        return df.loc[mask].reset_index(drop=True)
    
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
        thr_variants = ["ThrRR_p5", "Thr_p5_Mbps", "throughput_p5", "thr_rr_p5"]
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
        
        # Clean targets
        print("Cleaning target values...")
        df = self._clean_targets(df)
        print(f"After cleaning: {len(df):,} valid rows")
        
        # Compute parameter ranges
        self.param_ranges = self._compute_param_ranges(df)
        print(f"Parameter ranges: {self.param_ranges}")
        
        # Update feature/target cols to only available ones
        self.feature_cols = available_features
        self.target_cols = available_targets
        
        return df
    
    def train_models(self, df: pd.DataFrame) -> None:
        """Train separate regression models for each KPI target."""
        print("\nTraining surrogate models...")
        
        X = df[self.feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Feature matrix shape: {X.shape}")
        
        # Train separate model for each target
        for target in self.target_cols:
            if target not in df.columns:
                continue
            
            print(f"\nTraining model for {target}...")
            y = df[target].values.astype(np.float32)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Choose model based on availability
            if HAS_LIGHTGBM:
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': self.random_seed,
                }
                
                train_data = lgb.Dataset(X, label=y)
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[train_data],
                    valid_names=['train'],
                )
                
            elif HAS_XGBOOST:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    random_state=self.random_seed,
                    n_jobs=-1,
                )
                model.fit(X, y)
                
            else:
                # Fallback to sklearn
                model = HistGradientBoostingRegressor(
                    max_iter=100,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=self.random_seed,
                )
                model.fit(X, y)
            
            self.models[target] = model
            
            # Quick validation
            y_pred = self._predict_single_model(model, X[:1000])
            mae = np.mean(np.abs(y_pred - y[:1000]))
            print(f"  Model trained. Sample MAE on first 1000 rows: {mae:.4f}")
        
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
            "target_cols": self.target_cols,
            "models": self.models,
            "param_ranges": self.param_ranges,
            "bw_hz": BW_HZ_DEFAULT,
            "has_throughput_col": self.has_throughput_col,
            "throughput_col": self.throughput_col,
            "version": "2.0",
            "notes": (
                "Surrogate models trained with LightGBM/XGBoost. "
                "Predicts: Prx_p5_dBm, SINR_p5_dB, rx_power_coverage_ratio, tx*_served_pct. "
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
