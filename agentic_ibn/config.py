from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    groq_model: str
    runtime_mode: str
    results_dir: Path
    state_file: Path
    model_registry_file: Path
    max_optimization_iterations: int
    default_scenario: str

    @classmethod
    def from_env(cls) -> "Settings":
        runtime_mode = os.getenv("AGNO_RUNTIME", "agno").strip().lower()
        if runtime_mode not in {"agno", "deterministic"}:
            raise ValueError("AGNO_RUNTIME must be 'agno' or 'deterministic'")
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            runtime_mode=runtime_mode,
            results_dir=Path(os.getenv("RESULTS_DIR", "results")),
            state_file=Path(os.getenv("ACTIVE_STATE_FILE", "state/active_intents.json")),
            model_registry_file=Path(os.getenv("MODEL_REGISTRY_FILE", "config/model_registry.json")),
            max_optimization_iterations=int(os.getenv("MAX_OPTIMIZATION_ITERATIONS", "3")),
            default_scenario=os.getenv("DEFAULT_SCENARIO", "urban_area"),
        )

    def load_model_registry(self) -> dict[str, str]:
        if not self.model_registry_file.exists():
            return {}
        data = json.loads(self.model_registry_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Model registry must contain a JSON object: {self.model_registry_file}")
        return {str(key): str(value) for key, value in data.items()}
