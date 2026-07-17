from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .features import BASE_FEATURES, DEFAULT_FEATURES, engineer_features, sanitize_configuration_frame


DEFAULT_TARGETS = [
    "Prx_p5_dBm",
    "Prx_mean_dBm",
    "SINR_p5_dB",
    "SINR_mean_dB",
    "Thr_p5_Mbps",
    "ThrRR_p5_Mbps",
    "rx_power_coverage_ratio",
    "tx0_served_pct",
    "tx1_served_pct",
    "tx2_served_pct",
    "tx3_served_pct",
]

RAW_CONFIGURATION_COLUMNS = [
    "K_users",
    "rx_power_thr_dBm",
    "user_set_id",
]
for _index in range(4):
    RAW_CONFIGURATION_COLUMNS.extend(
        [
            f"tx{_index}_on",
            f"tx{_index}_P_dBm",
            f"tx{_index}_dAz",
            f"tx{_index}_dEl",
        ]
    )


@dataclass
class DataSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    strategy: str


@dataclass
class ExternalPreparedData:
    batch_files: dict[str, list[tuple[Path, Path]]]
    row_counts: dict[str, int]
    valid_rows: int
    target_names: list[str]
    split_strategy: str
    parameter_bounds: dict[str, dict[str, float]]
    prepared_dir: Path


def _manifest_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def discover_files(inputs: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.parquet")))
            files.extend(sorted(path.rglob("*.csv")))
        elif path.suffix.lower() in {".csv", ".parquet"}:
            files.append(path)
        else:
            raise FileNotFoundError(f"Unsupported or missing dataset path: {path}")
    unique = list(dict.fromkeys(path.resolve() for path in files))
    if not unique:
        raise FileNotFoundError("No CSV or Parquet files were found")

    # Dataset folders often contain the same rows as both CSV and Parquet.
    # Keep one representation per stem and prefer Parquet when pyarrow exists.
    try:
        import pyarrow  # noqa: F401

        parquet_available = True
    except ImportError:
        parquet_available = False
    grouped: dict[tuple[Path, str], list[Path]] = {}
    for path in unique:
        grouped.setdefault((path.parent, path.stem), []).append(path)
    selected: list[Path] = []
    for variants in grouped.values():
        parquet = next((path for path in variants if path.suffix.lower() == ".parquet"), None)
        csv = next((path for path in variants if path.suffix.lower() == ".csv"), None)
        selected.append(parquet if parquet_available and parquet is not None else (csv or parquet))
    return sorted(selected)


def _file_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading Parquet requires pyarrow") from exc
    return list(pq.ParquetFile(path).schema.names)


def _requested_file_columns(path: Path, requested: Sequence[str] | None) -> list[str] | None:
    if requested is None:
        return None
    available = set(_file_columns(path))
    selected = [column for column in requested if column in available]
    return selected or None


def _iter_file_batches(
    path: Path,
    chunk_size: int,
    columns: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    selected = _requested_file_columns(path, columns)
    if path.suffix.lower() == ".csv":
        iterator = pd.read_csv(path, chunksize=chunk_size, usecols=selected)
        yield from iterator
        return

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading Parquet incrementally requires pyarrow") from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=chunk_size, columns=selected, use_threads=False):
        yield batch.to_pandas()


def _selected_positions(total_rows: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if cap >= total_rows:
        return np.arange(total_rows, dtype=np.int64)
    return np.sort(rng.choice(total_rows, size=cap, replace=False).astype(np.int64, copy=False))


def _take_positions_from_batches(
    batches: Iterator[pd.DataFrame],
    positions: np.ndarray,
) -> list[pd.DataFrame]:
    collected: list[pd.DataFrame] = []
    offset = 0
    cursor = 0
    for chunk in batches:
        end = offset + len(chunk)
        next_cursor = int(np.searchsorted(positions, end, side="left"))
        if next_cursor > cursor:
            local = positions[cursor:next_cursor] - offset
            collected.append(chunk.iloc[local].copy())
        cursor = next_cursor
        offset = end
        if cursor >= len(positions):
            break
    return collected


def _csv_row_count(path: Path, chunk_size: int) -> int:
    columns = _file_columns(path)
    if not columns:
        return 0
    first = columns[0]
    return sum(len(chunk) for chunk in pd.read_csv(path, usecols=[first], chunksize=chunk_size))


def _read_file_sample(
    path: Path,
    cap: int | None,
    seed: int,
    chunk_size: int,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if cap is None:
        collected = list(_iter_file_batches(path, chunk_size=chunk_size, columns=columns))
    elif path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading Parquet incrementally requires pyarrow") from exc
        total_rows = int(pq.ParquetFile(path).metadata.num_rows)
        positions = _selected_positions(total_rows, min(cap, total_rows), rng)
        collected = _take_positions_from_batches(
            _iter_file_batches(path, chunk_size=chunk_size, columns=columns),
            positions,
        )
    else:
        total_rows = _csv_row_count(path, chunk_size=chunk_size)
        positions = _selected_positions(total_rows, min(cap, total_rows), rng)
        collected = _take_positions_from_batches(
            _iter_file_batches(path, chunk_size=chunk_size, columns=columns),
            positions,
        )

    if not collected:
        raise ValueError(f"Dataset file contains no rows: {path}")
    frame = pd.concat(collected, ignore_index=True)
    frame["__source_file"] = path.name
    return frame


def load_datasets(
    paths: list[Path],
    sample_size: int | None,
    seed: int,
    chunk_size: int,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    per_file_cap = None if sample_size is None else max(1, math.ceil(sample_size / len(paths)))
    frames = [
        _read_file_sample(
            path,
            per_file_cap,
            seed + index,
            chunk_size,
            columns=columns,
        )
        for index, path in enumerate(paths)
    ]
    frame = pd.concat(frames, ignore_index=True)
    if sample_size is not None and len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return frame


def _sanitize_target_columns(df: pd.DataFrame, targets: list[str]) -> list[str]:
    available_targets = [target for target in targets if target in df.columns]
    if not available_targets:
        raise ValueError(f"None of the requested targets exist. Requested: {targets}")

    for target in available_targets:
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df.loc[~np.isfinite(df[target]), target] = np.nan

    if "rx_power_coverage_ratio" in available_targets:
        invalid = ~df["rx_power_coverage_ratio"].between(0.0, 1.0)
        df.loc[invalid, "rx_power_coverage_ratio"] = np.nan
    for index in range(4):
        target = f"tx{index}_served_pct"
        if target in available_targets:
            df.loc[~df[target].between(0.0, 100.0), target] = np.nan
    for target in ["Thr_p5_Mbps", "ThrRR_p5_Mbps"]:
        if target in available_targets:
            df.loc[df[target] < 0.0, target] = np.nan
    return available_targets


def _prepare_training_frame(
    frame: pd.DataFrame,
    targets: list[str],
    *,
    minimum_rows: int = 0,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    df = sanitize_configuration_frame(frame, copy=False)

    required_features = [name for name in BASE_FEATURES if name != "total_tx_power_watt"]
    missing = [column for column in required_features if column not in df.columns]
    if missing:
        raise ValueError(f"Missing configuration columns: {missing}")

    available_targets = _sanitize_target_columns(df, targets)
    active_count = df[[f"tx{i}_on" for i in range(4)]].sum(axis=1)
    df = df.loc[active_count > 0].dropna(subset=available_targets).reset_index(drop=True)
    if len(df) < minimum_rows:
        raise ValueError(
            f"Only {len(df)} valid rows remain. The sample files are schema examples and are not sufficient "
            "for a reliable surrogate."
        )

    df = engineer_features(df, sanitize=False, copy=False)
    for column in DEFAULT_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(np.float32)
    for target in available_targets:
        df[target] = df[target].astype(np.float32)

    keep = list(dict.fromkeys([*DEFAULT_FEATURES, "user_set_id", *available_targets, *extra_columns]))
    return df[[column for column in keep if column in df.columns]]


def clean_training_frame(frame: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    return _prepare_training_frame(frame, targets, minimum_rows=50)


def make_split(frame: pd.DataFrame, seed: int, test_size: float, strategy: str) -> DataSplit:
    indices = np.arange(len(frame))
    chosen = strategy
    group_column: str | None = None
    if strategy == "auto":
        if "user_set_id" in frame.columns and frame["user_set_id"].nunique() >= 5:
            chosen = "group_user_set"
            group_column = "user_set_id"
        else:
            chosen = "random"
    elif strategy == "group_user_set":
        group_column = "user_set_id"
        if group_column not in frame.columns or frame[group_column].nunique() < 3:
            raise ValueError("group_user_set split requires at least three user_set_id groups")

    if chosen == "group_user_set":
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_pos, test_pos = next(outer.split(indices, groups=frame[group_column]))
        train_val = indices[train_val_pos]
        test = indices[test_pos]
        inner_groups = frame.iloc[train_val][group_column].to_numpy()
        inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + 1)
        train_pos, validation_pos = next(inner.split(train_val, groups=inner_groups))
        train = train_val[train_pos]
        validation = train_val[validation_pos]
    else:
        train_val, test = train_test_split(indices, test_size=test_size, random_state=seed)
        train, validation = train_test_split(train_val, test_size=0.125, random_state=seed + 1)

    return DataSplit(
        train_indices=np.asarray(train),
        validation_indices=np.asarray(validation),
        test_indices=np.asarray(test),
        strategy=chosen,
    )


def create_model(family: str, seed: int, num_threads: int) -> Any:
    if family == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("Install lightgbm to use --model-family lightgbm") from exc
        return LGBMRegressor(
            objective="regression",
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            max_bin=127,
            force_col_wise=True,
            histogram_pool_size=1024,
            random_state=seed,
            n_jobs=num_threads,
            verbosity=-1,
        )
    if family == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("Install xgboost to use --model-family xgboost") from exc
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1200,
            max_depth=8,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=num_threads,
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=63,
            l2_regularization=0.1,
            early_stopping=True,
            random_state=seed,
        )
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=num_threads,
        )
    raise ValueError(f"Unsupported in-memory model family: {family}")


def fit_model(
    model: Any,
    family: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Any:
    if family == "lightgbm":
        import lightgbm as lgb

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
        )
    elif family == "xgboost":
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        model.fit(x_train, y_train)
    return model


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def parameter_bounds(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    bounds: dict[str, dict[str, float]] = {}
    for output_name, source_suffix in {
        "power": "P_dBm",
        "dAz": "dAz",
        "dEl": "dEl",
    }.items():
        minimum = math.inf
        maximum = -math.inf
        for index in range(4):
            on = frame[f"tx{index}_on"].astype(bool)
            values = frame.loc[on, f"tx{index}_{source_suffix}"].to_numpy(dtype=float)
            if values.size:
                minimum = min(minimum, float(np.min(values)))
                maximum = max(maximum, float(np.max(values)))
        if math.isfinite(minimum):
            bounds[output_name] = {"min": minimum, "max": maximum}
    return bounds


def _library_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    for name in ["lightgbm", "xgboost", "pyarrow"]:
        try:
            module = __import__(name)
            versions[name] = str(module.__version__)
        except ImportError:
            pass
    return versions


def _total_parquet_rows(paths: list[Path]) -> int | None:
    if any(path.suffix.lower() != ".parquet" for path in paths):
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    return sum(int(pq.ParquetFile(path).metadata.num_rows) for path in paths)


def _system_memory_gb() -> float | None:
    if platform.system() == "Darwin":
        try:
            output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(output) / (1024**3)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024**3)
    except (AttributeError, OSError, ValueError):
        return None


def _estimated_in_memory_gb(row_count: int, target_count: int) -> float:
    # Includes pandas raw/clean copies, engineered columns, float32 matrices,
    # split copies, labels, and model-side histogram structures.
    bytes_per_row = (len(DEFAULT_FEATURES) + target_count + 20) * 16
    return row_count * bytes_per_row / (1024**3)


def _guard_full_in_memory_load(
    paths: list[Path],
    target_count: int,
    *,
    force: bool,
    memory_budget_gb: float | None,
) -> None:
    if force:
        return
    rows = _total_parquet_rows(paths)
    if rows is None:
        raise RuntimeError(
            "Full in-memory loading is disabled because its RAM requirement cannot be estimated safely. "
            "Use --sample-size for LightGBM or --model-family xgboost_external --all-data."
        )
    estimate = _estimated_in_memory_gb(rows, target_count)
    system_gb = _system_memory_gb()
    budget = memory_budget_gb if memory_budget_gb is not None else (0.65 * system_gb if system_gb else 12.0)
    if estimate > budget:
        raise RuntimeError(
            f"Refusing an unsafe in-memory load of approximately {rows:,} rows. Estimated peak RAM is "
            f"about {estimate:.1f} GiB, above the {budget:.1f} GiB safety budget. Use a representative "
            "--sample-size or select --model-family xgboost_external --all-data."
        )


def _discover_available_targets(paths: list[Path], requested_targets: list[str]) -> list[str]:
    union: set[str] = set()
    for path in paths:
        union.update(_file_columns(path))
    available = [target for target in requested_targets if target in union]
    if not available:
        raise ValueError(f"None of the requested targets exist. Requested: {requested_targets}")
    return available


def _has_enough_user_groups(paths: list[Path], chunk_size: int, threshold: int = 5) -> bool:
    hashes: set[int] = set()
    for path in paths:
        if "user_set_id" not in _file_columns(path):
            continue
        for chunk in _iter_file_batches(path, chunk_size=chunk_size, columns=["user_set_id"]):
            values = pd.util.hash_pandas_object(chunk["user_set_id"], index=False).to_numpy(dtype=np.uint64)
            hashes.update(int(value) for value in np.unique(values))
            if len(hashes) >= threshold:
                return True
    return False


def _streaming_split_strategy(paths: list[Path], requested: str, chunk_size: int) -> str:
    enough_groups = _has_enough_user_groups(paths, chunk_size=chunk_size)
    if requested == "auto":
        return "group_user_set" if enough_groups else "random"
    if requested == "group_user_set" and not enough_groups:
        raise ValueError("group_user_set split requires at least five user_set_id groups")
    return requested


def _mix_uint64(values: np.ndarray, seed: int) -> np.ndarray:
    x = values.astype(np.uint64, copy=True) ^ np.uint64(seed)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def _streaming_split_masks(
    frame: pd.DataFrame,
    strategy: str,
    seed: int,
    test_size: float,
) -> dict[str, np.ndarray]:
    if strategy == "group_user_set":
        base = pd.util.hash_pandas_object(frame["user_set_id"], index=False).to_numpy(dtype=np.uint64)
    else:
        base = frame["__row_key"].to_numpy(dtype=np.uint64)
    hashed = _mix_uint64(base, seed)
    total = 1 << 64
    validation_fraction = (1.0 - test_size) * 0.125
    test_cut = int(test_size * total)
    validation_cut = int((test_size + validation_fraction) * total)
    test = hashed < np.uint64(test_cut)
    validation = (hashed >= np.uint64(test_cut)) & (hashed < np.uint64(validation_cut))
    train = ~(test | validation)
    return {"train": train, "validation": validation, "test": test}


def _update_streaming_bounds(
    bounds: dict[str, dict[str, float]],
    frame: pd.DataFrame,
) -> None:
    for output_name, source_suffix in {
        "power": "P_dBm",
        "dAz": "dAz",
        "dEl": "dEl",
    }.items():
        current = bounds.setdefault(output_name, {"min": math.inf, "max": -math.inf})
        for index in range(4):
            on = frame[f"tx{index}_on"].astype(bool).to_numpy()
            values = frame[f"tx{index}_{source_suffix}"].to_numpy(dtype=float)[on]
            if values.size:
                current["min"] = min(current["min"], float(np.min(values)))
                current["max"] = max(current["max"], float(np.max(values)))


def _prepare_external_batches(
    paths: list[Path],
    target_names: list[str],
    prepared_dir: Path,
    *,
    seed: int,
    test_size: float,
    split_strategy: str,
    chunk_size: int,
) -> ExternalPreparedData:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    split_dirs = {name: prepared_dir / name for name in ["train", "validation", "test"]}
    for directory in split_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    batch_files: dict[str, list[tuple[Path, Path]]] = {name: [] for name in split_dirs}
    row_counts = {name: 0 for name in split_dirs}
    batch_numbers = {name: 0 for name in split_dirs}
    bounds: dict[str, dict[str, float]] = {}
    valid_rows = 0
    columns = list(dict.fromkeys([*RAW_CONFIGURATION_COLUMNS, *target_names]))

    for file_index, path in enumerate(paths):
        raw_offset = 0
        file_hash = int.from_bytes(hashlib.sha256(str(path).encode()).digest()[:8], "little")
        for raw_chunk in _iter_file_batches(path, chunk_size=chunk_size, columns=columns):
            # Keep a consistent multi-target schema across files. Rows from a file
            # missing one of the selected targets are excluded by the common dropna.
            for target in target_names:
                if target not in raw_chunk.columns:
                    raw_chunk[target] = np.nan
            raw_length = len(raw_chunk)
            row_ids = np.arange(raw_offset, raw_offset + raw_length, dtype=np.uint64)
            raw_chunk["__row_key"] = _mix_uint64(row_ids ^ np.uint64(file_hash), seed + file_index)
            raw_offset += raw_length

            frame = _prepare_training_frame(
                raw_chunk,
                target_names,
                minimum_rows=0,
                extra_columns=["__row_key"],
            )
            if frame.empty:
                continue
            _update_streaming_bounds(bounds, frame)
            masks = _streaming_split_masks(frame, split_strategy, seed, test_size)
            x_all = frame[DEFAULT_FEATURES].to_numpy(dtype=np.float32, copy=False)
            y_all = frame[target_names].to_numpy(dtype=np.float32, copy=False)
            valid_rows += len(frame)

            for split_name, mask in masks.items():
                if not np.any(mask):
                    continue
                number = batch_numbers[split_name]
                x_path = split_dirs[split_name] / f"X_{number:06d}.npy"
                y_path = split_dirs[split_name] / f"Y_{number:06d}.npy"
                np.save(x_path, np.ascontiguousarray(x_all[mask], dtype=np.float32), allow_pickle=False)
                np.save(y_path, np.ascontiguousarray(y_all[mask], dtype=np.float32), allow_pickle=False)
                batch_files[split_name].append((x_path, y_path))
                rows = int(np.count_nonzero(mask))
                row_counts[split_name] += rows
                batch_numbers[split_name] += 1

    if valid_rows < 50:
        raise ValueError(f"Only {valid_rows} valid rows remain after streaming preprocessing")
    empty_splits = [name for name, count in row_counts.items() if count == 0]
    if empty_splits:
        raise ValueError(f"Streaming split produced no rows for: {empty_splits}")

    cleaned_bounds = {
        name: values
        for name, values in bounds.items()
        if math.isfinite(values["min"]) and math.isfinite(values["max"])
    }
    return ExternalPreparedData(
        batch_files=batch_files,
        row_counts=row_counts,
        valid_rows=valid_rows,
        target_names=target_names,
        split_strategy=split_strategy,
        parameter_bounds=cleaned_bounds,
        prepared_dir=prepared_dir,
    )


class _NpyExternalIterator:
    """Factory wrapper so xgboost is imported only when external mode is selected."""

    @staticmethod
    def create(
        batch_files: list[tuple[Path, Path]],
        target_index: int,
        cache_prefix: Path,
    ) -> Any:
        import xgboost as xgb

        class IteratorImpl(xgb.DataIter):
            def __init__(self) -> None:
                self._index = 0
                self._x: np.ndarray | None = None
                self._y: np.ndarray | None = None
                super().__init__(cache_prefix=str(cache_prefix), release_data=True)

            def reset(self) -> None:
                self._index = 0
                self._x = None
                self._y = None

            def next(self, input_data: Any) -> bool:
                if self._index >= len(batch_files):
                    return False
                x_path, y_path = batch_files[self._index]
                self._x = np.load(x_path, allow_pickle=False)
                y_matrix = np.load(y_path, allow_pickle=False)
                self._y = np.ascontiguousarray(y_matrix[:, target_index], dtype=np.float32)
                input_data(data=self._x, label=self._y)
                self._index += 1
                return True

        return IteratorImpl()


def _labels_from_prepared_batches(
    batch_files: list[tuple[Path, Path]],
    target_index: int,
    expected_rows: int,
) -> np.ndarray:
    labels = np.empty(expected_rows, dtype=np.float32)
    cursor = 0
    for _, y_path in batch_files:
        y_matrix = np.load(y_path, mmap_mode="r", allow_pickle=False)
        values = np.asarray(y_matrix[:, target_index], dtype=np.float32)
        end = cursor + len(values)
        labels[cursor:end] = values
        cursor = end
    if cursor != expected_rows:
        raise RuntimeError(f"Prepared label count mismatch: expected {expected_rows}, found {cursor}")
    return labels


def _best_iteration_end(booster: Any) -> int:
    try:
        return int(booster.best_iteration) + 1
    except (AttributeError, TypeError, ValueError):
        return 0


def _train_xgboost_external(
    paths: list[Path],
    scenario: str,
    model_path: str | Path,
    *,
    seed: int,
    test_size: float,
    split_strategy: str,
    targets: list[str],
    chunk_size: int,
    num_threads: int,
    cache_dir: str | Path | None,
    keep_cache: bool,
    max_bin: int,
) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("Install xgboost>=3.0 to use xgboost_external") from exc
    major = int(str(xgb.__version__).split(".")[0])
    if major < 3 or not hasattr(xgb, "ExtMemQuantileDMatrix"):
        raise RuntimeError("xgboost_external requires xgboost>=3.0")

    target_names = _discover_available_targets(paths, targets)
    chosen_split = _streaming_split_strategy(paths, split_strategy, chunk_size=chunk_size)
    cache_parent = Path(cache_dir) if cache_dir else Path(".cache") / "surrogate_training"
    cache_parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{scenario}_", dir=cache_parent))
    prepared_dir = run_dir / "prepared"
    xgb_cache_dir = run_dir / "xgboost_cache"
    xgb_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        prepared = _prepare_external_batches(
            paths,
            target_names,
            prepared_dir,
            seed=seed,
            test_size=test_size,
            split_strategy=chosen_split,
            chunk_size=chunk_size,
        )

        iterators = {
            split_name: _NpyExternalIterator.create(
                prepared.batch_files[split_name],
                target_index=0,
                cache_prefix=xgb_cache_dir / split_name,
            )
            for split_name in ["train", "validation", "test"]
        }
        dtrain = xgb.ExtMemQuantileDMatrix(
            iterators["train"], max_bin=max_bin, nthread=num_threads
        )
        dvalidation = xgb.ExtMemQuantileDMatrix(
            iterators["validation"], max_bin=max_bin, nthread=num_threads, ref=dtrain
        )
        dtest = xgb.ExtMemQuantileDMatrix(
            iterators["test"], max_bin=max_bin, nthread=num_threads, ref=dtrain
        )

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "grow_policy": "depthwise",
            "max_depth": 8,
            "eta": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "min_child_weight": 20.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "max_bin": max_bin,
            "seed": seed,
            "nthread": num_threads,
        }

        models: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        for target_index, target in enumerate(target_names):
            if target_index > 0:
                train_y = _labels_from_prepared_batches(
                    prepared.batch_files["train"], target_index, prepared.row_counts["train"]
                )
                validation_y = _labels_from_prepared_batches(
                    prepared.batch_files["validation"], target_index, prepared.row_counts["validation"]
                )
                test_y = _labels_from_prepared_batches(
                    prepared.batch_files["test"], target_index, prepared.row_counts["test"]
                )
                dtrain.set_label(train_y)
                dvalidation.set_label(validation_y)
                dtest.set_label(test_y)
            else:
                validation_y = np.asarray(dvalidation.get_label(), dtype=np.float32)
                test_y = np.asarray(dtest.get_label(), dtype=np.float32)

            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=1500,
                evals=[(dtrain, "train"), (dvalidation, "validation")],
                early_stopping_rounds=75,
                verbose_eval=False,
            )
            end = _best_iteration_end(booster)
            prediction_kwargs = {"iteration_range": (0, end)} if end else {}
            validation_pred = booster.predict(dvalidation, **prediction_kwargs)
            test_pred = booster.predict(dtest, **prediction_kwargs)
            models[target] = booster
            metrics[target] = {
                "best_iteration": int(end - 1) if end else None,
                "validation": regression_metrics(validation_y, validation_pred),
                "test": regression_metrics(test_y, test_pred),
            }
            del validation_pred, test_pred
            if target_index > 0:
                del train_y, validation_y, test_y
            gc.collect()

        artifact = {
            "artifact_version": "4.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario,
            "model_family": "xgboost_external",
            "training_mode": "external_memory",
            "feature_names": list(DEFAULT_FEATURES),
            "target_names": target_names,
            "models": models,
            "parameter_bounds": prepared.parameter_bounds,
            "metrics": metrics,
            "split": {
                "strategy": prepared.split_strategy,
                "train_rows": prepared.row_counts["train"],
                "validation_rows": prepared.row_counts["validation"],
                "test_rows": prepared.row_counts["test"],
                "random_seed": seed,
            },
            "training_data": {
                "files": [str(path) for path in paths],
                "sampled_rows": None,
                "valid_rows": prepared.valid_rows,
                "manifest_hash": _manifest_hash(paths),
            },
            "external_memory": {
                "chunk_size": chunk_size,
                "max_bin": max_bin,
                "cache_directory": str(run_dir),
                "cache_retained": keep_cache,
            },
            "library_versions": _library_versions(),
            "bandwidth_hz": 400_000_000.0,
            "notes": (
                "All valid rows were processed in bounded batches. XGBoost cached quantized feature pages on disk; "
                "labels were held one target at a time in memory."
            ),
        }
        _save_artifact(artifact, model_path)

        del dtrain, dvalidation, dtest, iterators
        gc.collect()
        return artifact
    finally:
        if not keep_cache:
            shutil.rmtree(run_dir, ignore_errors=True)


def _save_artifact(artifact: dict[str, Any], model_path: str | Path) -> None:
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    metrics_path = output.with_suffix(output.suffix + ".metrics.json")
    metrics_path.write_text(
        json.dumps({key: value for key, value in artifact.items() if key != "models"}, indent=2),
        encoding="utf-8",
    )


def train_surrogate(
    data_paths: list[str | Path],
    scenario: str,
    model_path: str | Path,
    model_family: str = "lightgbm",
    sample_size: int | None = 1_000_000,
    seed: int = 42,
    test_size: float = 0.2,
    split_strategy: str = "auto",
    targets: list[str] | None = None,
    chunk_size: int = 100_000,
    num_threads: int = 4,
    cache_dir: str | Path | None = None,
    keep_cache: bool = False,
    max_bin: int = 127,
    force_in_memory_all_data: bool = False,
    memory_budget_gb: float | None = None,
) -> dict[str, Any]:
    paths = discover_files(data_paths)
    requested_targets = targets or DEFAULT_TARGETS

    if model_family == "xgboost_external":
        if sample_size is not None:
            raise ValueError(
                "xgboost_external is the full-data path. Pass sample_size=None or use the CLI flag --all-data."
            )
        return _train_xgboost_external(
            paths,
            scenario,
            model_path,
            seed=seed,
            test_size=test_size,
            split_strategy=split_strategy,
            targets=requested_targets,
            chunk_size=chunk_size,
            num_threads=num_threads,
            cache_dir=cache_dir,
            keep_cache=keep_cache,
            max_bin=max_bin,
        )

    if sample_size is None:
        _guard_full_in_memory_load(
            paths,
            len(requested_targets),
            force=force_in_memory_all_data,
            memory_budget_gb=memory_budget_gb,
        )

    source_columns = list(dict.fromkeys([*RAW_CONFIGURATION_COLUMNS, *requested_targets]))
    raw = load_datasets(
        paths,
        sample_size=sample_size,
        seed=seed,
        chunk_size=chunk_size,
        columns=source_columns,
    )
    frame = clean_training_frame(raw, requested_targets)
    target_names = [target for target in requested_targets if target in frame.columns]
    split = make_split(frame, seed=seed, test_size=test_size, strategy=split_strategy)

    feature_names = list(DEFAULT_FEATURES)
    x = frame[feature_names].to_numpy(dtype=np.float32, copy=False)
    x_train = np.ascontiguousarray(x[split.train_indices])
    x_validation = np.ascontiguousarray(x[split.validation_indices])
    x_test = np.ascontiguousarray(x[split.test_indices])
    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {}

    for target in target_names:
        y = frame[target].to_numpy(dtype=np.float32, copy=False)
        y_train = np.ascontiguousarray(y[split.train_indices])
        y_validation = np.ascontiguousarray(y[split.validation_indices])
        y_test = np.ascontiguousarray(y[split.test_indices])
        model = create_model(model_family, seed, num_threads=num_threads)
        model = fit_model(
            model,
            model_family,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
        models[target] = model
        metrics[target] = {
            "validation": regression_metrics(y_validation, model.predict(x_validation)),
            "test": regression_metrics(y_test, model.predict(x_test)),
        }

    artifact = {
        "artifact_version": "4.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "model_family": model_family,
        "training_mode": "sampled_in_memory" if sample_size is not None else "full_in_memory",
        "feature_names": feature_names,
        "target_names": target_names,
        "models": models,
        "parameter_bounds": parameter_bounds(frame),
        "metrics": metrics,
        "split": {
            "strategy": split.strategy,
            "train_rows": int(len(split.train_indices)),
            "validation_rows": int(len(split.validation_indices)),
            "test_rows": int(len(split.test_indices)),
            "random_seed": seed,
        },
        "training_data": {
            "files": [str(path) for path in paths],
            "sampled_rows": int(len(raw)) if sample_size is not None else None,
            "valid_rows": int(len(frame)),
            "manifest_hash": _manifest_hash(paths),
        },
        "library_versions": _library_versions(),
        "bandwidth_hz": 400_000_000.0,
        "notes": (
            "One tabular regressor is trained per KPI target. Sampled mode draws rows across the complete files, "
            "rather than taking only their first chunks."
        ),
    }
    _save_artifact(artifact, model_path)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a scenario-specific KPI surrogate artifact")
    parser.add_argument("--data", action="append", required=True, help="CSV/Parquet file or directory; repeat as needed")
    parser.add_argument("--scenario", required=True, help="Scenario key, for example urban_area or open_area")
    parser.add_argument("--model-path", required=True, help="Output .joblib path")
    parser.add_argument(
        "--model-family",
        default="lightgbm",
        choices=["lightgbm", "xgboost", "xgboost_external", "hist_gradient_boosting", "random_forest"],
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1_000_000,
        help="Rows retained across all files for in-memory training (default: 1,000,000)",
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Use every valid row. Recommended only with --model-family xgboost_external.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=["auto", "random", "group_user_set"], default="auto")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--num-threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--cache-dir", default=None, help="Fast local disk for xgboost_external temporary files")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--max-bin", type=int, default=127)
    parser.add_argument("--memory-budget-gb", type=float, default=None)
    parser.add_argument(
        "--force-in-memory-all-data",
        action="store_true",
        help="Bypass the RAM safety guard. This can cause the OS to kill the process.",
    )
    parser.add_argument("--target", action="append", dest="targets", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sample_size = None if args.all_data else args.sample_size
    artifact = train_surrogate(
        data_paths=args.data,
        scenario=args.scenario,
        model_path=args.model_path,
        model_family=args.model_family,
        sample_size=sample_size,
        seed=args.random_seed,
        test_size=args.test_size,
        split_strategy=args.split_strategy,
        targets=args.targets,
        chunk_size=args.chunk_size,
        num_threads=max(1, args.num_threads),
        cache_dir=args.cache_dir,
        keep_cache=args.keep_cache,
        max_bin=args.max_bin,
        force_in_memory_all_data=args.force_in_memory_all_data,
        memory_budget_gb=args.memory_budget_gb,
    )
    print(
        json.dumps(
            {
                "model_path": args.model_path,
                "scenario": artifact["scenario"],
                "model_family": artifact["model_family"],
                "training_mode": artifact["training_mode"],
                "targets": artifact["target_names"],
                "split": artifact["split"],
                "metrics": artifact["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
