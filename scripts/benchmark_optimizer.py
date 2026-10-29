"""Benchmark optimizer variants against the exhaustive-search optimum on the surrogate.

For every test case the script
1. enumerates every configuration on the surrogate's training grid that respects the hard
   TX constraints, scores all of them with the same ObjectiveEvaluator and keeps the best one
   (the "answer key"), then
2. runs each optimizer variant on the same case and reports how close it gets.

Example (offline, no LLM):
    python scripts/benchmark_optimizer.py --model-path models/urban_area.joblib \
        --cases examples/benchmark_cases.json --runtime deterministic

Variants are "name,rounds,candidates_per_round,history[,stall]". The defaults are
    V0-last-1   3 rounds, 1 candidate, agent sees only the latest candidate (closest to the old loop)
    V1-full-1   3 rounds, 1 candidate, agent sees the whole archive
    V2-full-4   3 rounds, 4 candidates (explore, then learn/combine/free), whole archive
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_ibn.agents.optimization_agent import (
    AgnoCandidateGenerator,
    HeuristicCandidateGenerator,
    OptimizationLoop,
)
from agentic_ibn.constraints import extract_hard_constraints
from agentic_ibn.optimization.objective import ObjectiveEvaluator
from agentic_ibn.schemas import NetworkConfig, ParsedIntent
from agentic_ibn.surrogate.model import SurrogateModel

DEFAULT_VARIANTS = ["V0-last-1,3,1,last,nostall", "V1-full-1,3,1,full", "V2-full-4,3,4,full"]


def parse_variant(text: str) -> dict[str, Any]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError(f"Variant must be name,rounds,candidates,history[,nostall]: {text!r}")
    return {
        "name": parts[0],
        "rounds": int(parts[1]),
        "candidates": int(parts[2]),
        "history": parts[3],
        "stop_on_stall": not (len(parts) == 5 and parts[4] == "nostall"),
    }


def tx_options(surrogate: SurrogateModel, required_state: bool | None) -> list[tuple[bool, float, float, float]]:
    grids = [surrogate.grid_for(name) for name in ("power_dbm", "azimuth_delta_deg", "elevation_delta_deg")]
    if any(grid is None for grid in grids):
        raise SystemExit(
            "The surrogate artifact has no parameter_grid. Retrain it with this branch or run "
            "scripts/attach_parameter_grid.py first."
        )
    options: list[tuple[bool, float, float, float]] = []
    if required_state is not True:
        options.append((False, 0.0, 0.0, 0.0))
    if required_state is not False:
        options.extend((True, p, a, e) for p, a, e in itertools.product(*grids))  # type: ignore[arg-type]
    return options


def exhaustive_best(
    surrogate: SurrogateModel,
    intent: ParsedIntent,
    baseline: NetworkConfig,
    chunk_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    evaluator = ObjectiveEvaluator()
    constraints = extract_hard_constraints(intent.raw_text)
    per_tx = [np.array(tx_options(surrogate, constraints.tx_states.get(i)), dtype=float) for i in range(4)]
    sizes = [len(options) for options in per_tx]
    index_grid = np.stack(np.meshgrid(*[np.arange(size) for size in sizes], indexing="ij"), axis=-1).reshape(-1, 4)
    on_any = np.zeros(len(index_grid), dtype=bool)
    for tx in range(4):
        on_any |= per_tx[tx][index_grid[:, tx], 0] > 0.5
    index_grid = index_grid[on_any]

    baseline_prediction = surrogate.predict(baseline)
    best: dict[str, Any] | None = None
    targets_met = 0
    total = len(index_grid)
    for start in range(0, total, chunk_size):
        block = index_grid[start : start + chunk_size]
        columns: dict[str, Any] = {
            "K_users": np.full(len(block), baseline.k_users),
            "user_set_id": np.full(len(block), baseline.user_set_id),
            "rx_power_thr_dBm": np.full(len(block), baseline.coverage_threshold_dbm),
        }
        for tx in range(4):
            values = per_tx[tx][block[:, tx]]
            columns[f"tx{tx}_on"] = values[:, 0] > 0.5
            columns[f"tx{tx}_P_dBm"] = values[:, 1]
            columns[f"tx{tx}_dAz"] = values[:, 2]
            columns[f"tx{tx}_dEl"] = values[:, 3]
        flat = pd.DataFrame(columns)
        predictions = surrogate.predict_flat_frame(flat)
        for row, prediction in enumerate(predictions):
            evaluation = evaluator.evaluate(intent, prediction, baseline_prediction)
            targets_met += int(evaluation.all_numeric_targets_met)
            if best is None or evaluation.score > best["score"]:
                best = {
                    "score": evaluation.score,
                    "target_satisfied": evaluation.all_numeric_targets_met,
                    "row": flat.iloc[row].to_dict(),
                }
        print(f"    exhaustive: {min(start + chunk_size, total):,}/{total:,} configurations", flush=True)
    assert best is not None
    configuration = NetworkConfig.from_flat_dict({**best.pop("row")})
    return {
        **best,
        "configuration": configuration.model_dump(mode="json"),
        "configurations_searched": total,
        "configurations_meeting_all_targets": targets_met,
        "seconds": round(time.perf_counter() - started, 2),
    }


def run_variant(
    surrogate: SurrogateModel,
    generator: Any,
    variant: dict[str, Any],
    intent: ParsedIntent,
    baseline: NetworkConfig,
) -> dict[str, Any]:
    loop = OptimizationLoop(
        surrogate=surrogate,
        candidate_generator=generator,
        max_iterations=variant["rounds"],
        candidates_per_round=variant["candidates"],
        history=variant["history"],
        stop_on_stall=variant["stop_on_stall"],
    )
    started = time.perf_counter()
    try:
        result = loop.optimize(intent, baseline)
    except Exception as exc:  # noqa: BLE001 - a failed run is a data point, not a crash of the benchmark
        return {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - started, 2)}
    return {
        "score": result.best_score,
        "target_satisfied": result.target_satisfied,
        "candidates_evaluated": len(result.iterations),
        "candidates_rejected": sum(1 for entry in result.archive if entry.status == "rejected"),
        "generator_calls": result.generator_calls,
        "rounds_run": result.rounds_run,
        "configuration": result.final_configuration.model_dump(mode="json"),
        "seconds": round(time.perf_counter() - started, 2),
    }


def summarize(cases: list[dict[str, Any]], variant_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for name in variant_names:
        runs = [run for case in cases for run in case["variants"][name]]
        ok = [run for run in runs if "error" not in run]
        gaps = [run["gap_to_optimum"] for run in ok if "gap_to_optimum" in run]
        rows.append(
            {
                "variant": name,
                "runs": len(runs),
                "failed_runs": len(runs) - len(ok),
                "target_hit_rate": round(mean(run["target_satisfied"] for run in ok), 3) if ok else None,
                "optimum_found_rate": round(mean(gap <= 1e-6 for gap in gaps), 3) if gaps else None,
                "mean_gap_to_optimum": round(mean(gaps), 4) if gaps else None,
                "mean_candidates_evaluated": round(mean(run["candidates_evaluated"] for run in ok), 2) if ok else None,
                "mean_generator_calls": round(mean(run["generator_calls"] for run in ok), 2) if ok else None,
                "mean_seconds": round(mean(run["seconds"] for run in ok), 2) if ok else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Surrogate .joblib artifact (with parameter_grid)")
    parser.add_argument("--cases", required=True, help="JSON list of {name, intent, baseline?}")
    parser.add_argument("--runtime", choices=["deterministic", "agno"], default="deterministic")
    parser.add_argument("--model-id", default=None, help="Groq model id for --runtime agno (default: GROQ_MODEL)")
    parser.add_argument("--variant", action="append", type=parse_variant, default=None)
    parser.add_argument("--repeats", type=int, default=1, help="Runs per variant and case (LLM runs vary)")
    parser.add_argument("--skip-exhaustive", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--out", default=None, help="Output JSON (default: results/benchmark_<timestamp>.json)")
    args = parser.parse_args()

    surrogate = SurrogateModel.load(args.model_path)
    variants = args.variant or [parse_variant(text) for text in DEFAULT_VARIANTS]
    if args.runtime == "agno":
        from agentic_ibn.config import Settings

        generator: Any = AgnoCandidateGenerator(args.model_id or Settings.from_env().groq_model)
    else:
        generator = HeuristicCandidateGenerator()

    raw_cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for raw in raw_cases:
        intent = ParsedIntent.model_validate(raw["intent"])
        baseline = NetworkConfig.model_validate(raw.get("baseline") or {})
        print(f"Case {raw['name']}", flush=True)
        case: dict[str, Any] = {"name": raw["name"], "intent": intent.model_dump(mode="json"), "variants": {}}
        if not args.skip_exhaustive:
            case["exhaustive"] = exhaustive_best(surrogate, intent, baseline, args.chunk_size)
            print(f"    optimum score {case['exhaustive']['score']:.4f}", flush=True)
        for variant in variants:
            runs = []
            for _ in range(args.repeats):
                run = run_variant(surrogate, generator, variant, intent, baseline)
                if "score" in run and "exhaustive" in case:
                    run["gap_to_optimum"] = case["exhaustive"]["score"] - run["score"]
                runs.append(run)
            case["variants"][variant["name"]] = runs
            shown = runs[0].get("score", runs[0].get("error"))
            print(f"    {variant['name']}: {shown}", flush=True)
        cases.append(case)

    summary = summarize(cases, [variant["name"] for variant in variants])
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(Path(args.model_path).resolve()),
        "scenario": surrogate.scenario,
        "runtime": args.runtime,
        "variants": variants,
        "summary": summary,
        "cases": cases,
    }
    out_path = Path(args.out or f"results/benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\n" + pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
