#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_ibn.config import Settings
from agentic_ibn.orchestration.workflow import AgenticWorkflow
from agentic_ibn.schemas import ParsedIntent
from agentic_ibn.storage.result_store import ResultStore
from agentic_ibn.storage.state_store import ActiveStateStore
from agentic_ibn.surrogate.model import SurrogateModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the conflict-aware Agno IBN workflow")
    parser.add_argument("--intent", required=True, help="Natural-language network intent")
    parser.add_argument("--scenario", default=None, help="Model registry key, e.g. urban_area or open_area")
    parser.add_argument("--model-path", default=None, help="Override the registry with a specific .joblib artifact")
    parser.add_argument("--strategy", choices=["AUTO", "PRIORITY", "WEIGHTED_MERGE"], default="AUTO")
    parser.add_argument("--runtime", choices=["agno", "deterministic"], default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--parsed-intent-file",
        default=None,
        help="Optional ParsedIntent JSON. Required for deterministic runtime; bypasses the LLM parser.",
    )
    parser.add_argument("--reset-state", action="store_true", help="Reset active intent state before running")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    scenario = args.scenario or settings.default_scenario
    runtime_mode = args.runtime or settings.runtime_mode

    if runtime_mode == "agno" and not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for --runtime agno")

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        registry = settings.load_model_registry()
        if scenario not in registry:
            raise KeyError(
                f"No model registered for scenario {scenario!r}. Add it to {settings.model_registry_file} or pass --model-path."
            )
        model_path = Path(registry[scenario])

    state_store = ActiveStateStore(settings.state_file)
    if args.reset_state and settings.state_file.exists():
        settings.state_file.unlink()

    surrogate = SurrogateModel.load(model_path)
    if surrogate.scenario not in {"unknown", scenario}:
        raise ValueError(
            f"Scenario mismatch: CLI requested {scenario!r}, artifact contains {surrogate.scenario!r}."
        )

    parsed_intent = None
    if args.parsed_intent_file:
        parsed_intent = ParsedIntent.model_validate_json(
            Path(args.parsed_intent_file).read_text(encoding="utf-8")
        )
    elif runtime_mode == "deterministic":
        raise RuntimeError("--runtime deterministic requires --parsed-intent-file")

    workflow = AgenticWorkflow(
        surrogate=surrogate,
        state_store=state_store,
        result_store=ResultStore(settings.results_dir),
        model_id=settings.groq_model,
        runtime_mode=runtime_mode,
        max_iterations=args.max_iterations or settings.max_optimization_iterations,
    )
    result = workflow.run(
        intent_text=args.intent,
        scenario=scenario,
        requested_strategy=args.strategy,
        parsed_intent=parsed_intent,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print(f"\nSaved run history to: {result.result_directory}")


if __name__ == "__main__":
    main()
