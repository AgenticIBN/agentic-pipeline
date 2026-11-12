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
from agentic_ibn.optimization.archive import configuration_changes, describe_change
from agentic_ibn.optimization.objective import ObjectiveEvaluator
from agentic_ibn.schemas import NetworkConfig, ParsedIntent
from agentic_ibn.surrogate.model import SurrogateModel

# KPIs reported for the baseline and for each run's best configuration.
REPORTED_KPIS = (
    "coverage_ratio",
    "sinr_p5_db",
    "rx_power_p5_dbm",
    "throughput_p5_mbps",
    "load_balance",
    "total_tx_power_watt",
)

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
    call_log = getattr(generator, "call_log", None)
    if call_log is not None:
        call_log.clear()
    started = time.perf_counter()
    try:
        result = loop.optimize(intent, baseline)
    except Exception as exc:  # noqa: BLE001 - a failed run is a data point, not a crash of the benchmark
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.perf_counter() - started, 2),
            **llm_timing(call_log),
        }
    seconds = round(time.perf_counter() - started, 2)
    timing = llm_timing(call_log)
    return {
        "score": result.best_score,
        "target_satisfied": result.target_satisfied,
        "candidates_evaluated": len(result.iterations),
        "candidates_rejected": sum(1 for entry in result.archive if entry.status == "rejected"),
        "generator_calls": result.generator_calls,
        "rounds_run": result.rounds_run,
        "configuration": result.final_configuration.model_dump(mode="json"),
        "changes_vs_baseline": [
            describe_change(change)
            for change in configuration_changes(result.final_configuration, result.baseline_configuration)
        ],
        "baseline_kpis": kpi_subset(result.baseline_prediction),
        "best_kpis": kpi_subset(result.predicted_kpis),
        "best_rationale": best_rationale(result),
        "seconds": seconds,
        **timing,
        "other_seconds": round(seconds - timing["llm_seconds"], 2) if "llm_seconds" in timing else None,
    }


def kpi_subset(prediction: Any) -> dict[str, float | None]:
    return {name: getattr(prediction, name, None) for name in REPORTED_KPIS}


def best_rationale(result: Any) -> str | None:
    """Rationale the agent gave for the configuration that won (None when the baseline won)."""
    for entry in result.archive:
        if (
            entry.status == "evaluated"
            and entry.evaluation is not None
            and entry.evaluation.score == result.best_score
            and entry.configuration == result.final_configuration
        ):
            return entry.rationale
    return None


def llm_timing(call_log: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Totals over the LLM calls of one run; empty for the offline generator."""
    if not call_log:
        return {}

    def total(key: str) -> int | None:
        values = [call.get(key) for call in call_log if call.get(key) is not None]
        return int(sum(values)) if values else None

    return {
        "llm_calls": [dict(call) for call in call_log],
        "llm_seconds": round(sum(call["seconds"] for call in call_log), 2),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
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
                "mean_llm_seconds": mean_of(ok, "llm_seconds"),
                "mean_best_coverage": mean_of([run.get("best_kpis", {}) for run in ok], "coverage_ratio", 4),
            }
        )
    return rows


def mean_of(rows: list[dict[str, Any]], key: str, digits: int = 2) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(mean(values), digits) if values else None


def fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_case_table(case: dict[str, Any]) -> None:
    """Readable per-case view: score, gap, coverage before and after, time and what changed."""
    print(f"\n{case['name']}")
    if "exhaustive" in case:
        print(f"  true best score {fmt(case['exhaustive']['score'], 4)}")
    for name, runs in case["variants"].items():
        for number, run in enumerate(runs, start=1):
            label = name if len(runs) == 1 else f"{name} #{number}"
            if "error" in run:
                print(f"  {label}: ERROR {run['error'][:120]} ({run['seconds']} s)")
                continue
            base, best = run["baseline_kpis"], run["best_kpis"]
            print(
                f"  {label}: score {fmt(run['score'], 4)}, gap {fmt(run.get('gap_to_optimum'), 4)}, "
                f"coverage {fmt(base['coverage_ratio'])} -> {fmt(best['coverage_ratio'])}, "
                f"SINR p5 {fmt(base['sinr_p5_db'], 1)} -> {fmt(best['sinr_p5_db'], 1)} dB, "
                f"time {run['seconds']} s (LLM {run.get('llm_seconds', '-')} s, {len(run.get('llm_calls', []))} calls)"
            )
            print(f"      changes: {', '.join(run['changes_vs_baseline']) or 'none'}")
            if run.get("best_rationale"):
                print(f"      why (agent): {run['best_rationale'][:240]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Surrogate .joblib artifact (with parameter_grid)")
    parser.add_argument("--cases", required=True, help="JSON list of {name, intent, baseline?}")
    parser.add_argument("--runtime", choices=["deterministic", "agno"], default="deterministic")
    parser.add_argument("--model-id", default=None, help="Groq model id for --runtime agno (default: GROQ_MODEL)")
    parser.add_argument("--variant", action="append", type=parse_variant, default=None)
    parser.add_argument(
        "--prompt-token-budget",
        type=int,
        default=None,
        help="Cap on estimated prompt tokens for --runtime agno (default: OPTIMIZATION_PROMPT_TOKEN_BUDGET)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Groq reasoning effort for gpt-oss models (default: provider default)",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Runs per variant and case (LLM runs vary)")
    parser.add_argument("--skip-exhaustive", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--out", default=None, help="Output JSON (default: results/benchmark_<timestamp>.json)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep the finished cases already in --out and run only the missing ones",
    )
    args = parser.parse_args()

    surrogate = SurrogateModel.load(args.model_path)
    variants = args.variant or [parse_variant(text) for text in DEFAULT_VARIANTS]
    if args.runtime == "agno":
        from agentic_ibn.config import Settings

        settings = Settings.from_env()
        generator: Any = AgnoCandidateGenerator(
            args.model_id or settings.groq_model,
            prompt_token_budget=args.prompt_token_budget or settings.optimization_prompt_token_budget,
            reasoning_effort=args.reasoning_effort,
        )
    else:
        generator = HeuristicCandidateGenerator()

    out_path = Path(args.out or f"results/benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    variant_names = [variant["name"] for variant in variants]

    def save(cases: list[dict[str, Any]], complete: bool) -> None:
        """Written after every case, so a crash or an exhausted API quota keeps the finished cases."""
        output = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "complete": complete,
            "model_path": str(Path(args.model_path).resolve()),
            "scenario": surrogate.scenario,
            "runtime": args.runtime,
            "llm_model": getattr(generator, "model_id", None),
            "reasoning_effort": getattr(generator, "reasoning_effort", None),
            "prompt_token_budget": getattr(generator, "prompt_token_budget", None),
            "variants": variants,
            "summary": summarize(cases, variant_names),
            "cases": cases,
        }
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    cases: list[dict[str, Any]] = []
    if args.resume and out_path.exists():
        previous = json.loads(out_path.read_text(encoding="utf-8")).get("cases", [])
        cases = [case for case in previous if case_finished(case, variant_names)]
        print(f"Resuming: {len(cases)} finished case(s) kept from {out_path}", flush=True)
    done = {case["name"] for case in cases}

    stopped_reason = None
    raw_cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    for raw in raw_cases:
        if raw["name"] in done:
            continue
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
                if is_quota_error(run.get("error")):
                    stopped_reason = run["error"]
                    break
            case["variants"][variant["name"]] = runs
            shown = runs[0].get("score", runs[0].get("error"))
            print(f"    {variant['name']}: {shown}", flush=True)
            if stopped_reason:
                break
        if stopped_reason:
            # The interrupted case is not kept, so --resume runs it again from the start.
            break
        cases.append(case)
        save(cases, complete=False)

    save(cases, complete=stopped_reason is None)
    for case in cases:
        print_case_table(case)
    print("\n" + pd.DataFrame(summarize(cases, variant_names)).to_string(index=False))
    print(f"\nSaved benchmark to: {out_path}")
    if stopped_reason:
        print(
            "\nStopped early: the LLM API rate limit or quota was hit. Finished cases are saved; run the same command "
            f"again with --resume once the quota resets.\n{stopped_reason[:300]}"
        )


def case_finished(case: dict[str, Any], variant_names: list[str]) -> bool:
    runs = case.get("variants", {})
    return all(name in runs and runs[name] and not any("error" in run for run in runs[name]) for name in variant_names)


def is_quota_error(error: str | None) -> bool:
    """Groq answers 429 with "tokens per day (TPD)" or "requests per day (RPD)" once the daily quota is gone.

    agno can hand the 429 body back as the response content, so the run then fails as a CandidateBatch
    ValidationError whose (truncated) text still contains "rate_limit_exceeded".
    """
    if not error:
        return False
    text = error.lower()
    markers = ("per day", "(tpd)", "(rpd)", "insufficient_quota", "rate_limit_exceeded")
    return any(marker in text for marker in markers)


if __name__ == "__main__":
    main()
