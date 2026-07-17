from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..schemas import AgentHistoryEntry, WorkflowResult
from .state_store import ActiveState


class ResultStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def create_run_directory(self, now: datetime | None = None) -> tuple[str, Path]:
        now = now or datetime.now()
        self.root.mkdir(parents=True, exist_ok=True)
        prefix = now.strftime("%Y-%m-%d_%H-%M-%S")
        run_number = 1
        while True:
            run_id = f"{prefix}_run{run_number}"
            path = self.root / run_id
            try:
                path.mkdir(parents=False, exist_ok=False)
                return run_id, path
            except FileExistsError:
                run_number += 1

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [ResultStore._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): ResultStore._jsonable(item) for key, item in value.items()}
        return value

    @staticmethod
    def write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(ResultStore._jsonable(value), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def persist(
        self,
        directory: Path,
        input_payload: dict[str, Any],
        history: list[AgentHistoryEntry],
        state_before: ActiveState,
        state_after: ActiveState,
        result: WorkflowResult,
    ) -> None:
        self.write_json(directory / "input.json", input_payload)
        self.write_json(directory / "agent_history.json", history)
        self.write_json(directory / "active_intents_before.json", state_before)
        self.write_json(directory / "active_intents_after.json", state_after)
        self.write_json(directory / "final_result.json", result)
