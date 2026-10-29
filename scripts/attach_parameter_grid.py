"""Add the knob grid (distinct power / azimuth / tilt values) to an existing surrogate artifact.

Artifacts trained before parameter_grid existed only store min/max per knob. The optimizer then
cannot snap proposals onto values the surrogate was trained on, and the benchmark cannot enumerate
the grid. Either read the grid from the training parquet files:

    python scripts/attach_parameter_grid.py --artifact models/urban_area.joblib \
        --dataset datasets/urban_area/kpi_K200.parquet

or give the values directly:

    python scripts/attach_parameter_grid.py --artifact models/urban_area.joblib \
        --power 43,46 --azimuth -30,-15,0,15,30 --tilt 5,10,15,20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_ibn.surrogate.training import parameter_grid


def _values(text: str | None) -> list[float] | None:
    return None if text is None else sorted(float(value) for value in text.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--dataset", action="append", default=[], help="Parquet file(s) the model was trained on")
    parser.add_argument("--power", help="Comma-separated power values in dBm")
    parser.add_argument("--azimuth", help="Comma-separated azimuth offsets in degrees")
    parser.add_argument("--tilt", help="Comma-separated elevation (tilt) offsets in degrees")
    parser.add_argument("--out", help="Output artifact (default: overwrite --artifact)")
    args = parser.parse_args()

    artifact = joblib.load(args.artifact)  # trusted, project-produced artifact
    grid: dict[str, list[float]] = {}
    columns = [f"tx{i}_{suffix}" for i in range(4) for suffix in ("on", "P_dBm", "dAz", "dEl")]
    for path in args.dataset:
        found = parameter_grid(pd.read_parquet(path, columns=columns))
        for name, values in found.items():
            grid[name] = sorted(set(grid.get(name, [])) | set(values))
    for name, text in {"power": args.power, "dAz": args.azimuth, "dEl": args.tilt}.items():
        values = _values(text)
        if values:
            grid[name] = values
    if not grid:
        raise SystemExit("Give --dataset or explicit --power/--azimuth/--tilt values")

    artifact["parameter_grid"] = grid
    out = Path(args.out or args.artifact)
    joblib.dump(artifact, out)
    print(f"parameter_grid = {grid}\nSaved to {out}")


if __name__ == "__main__":
    main()
