from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field

from ..agents.meta_agent import optimization_to_record
from ..schemas import (
    ActiveIntentRecord,
    NetworkConfig,
    OptimizationResult,
    ResolutionResult,
    ResolutionStrategy,
    StrictModel,
)


class ActiveState(StrictModel):
    version: str = "2.0"
    updated_at: datetime
    effective_configuration: NetworkConfig
    intents: list[ActiveIntentRecord] = Field(default_factory=list)


def default_network_configuration() -> NetworkConfig:
    return NetworkConfig()


class ActiveStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def empty(self) -> ActiveState:
        return ActiveState(
            updated_at=datetime.now(timezone.utc),
            effective_configuration=default_network_configuration(),
            intents=[],
        )

    def load(self) -> ActiveState:
        if not self.path.exists():
            return self.empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            state = ActiveState.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Invalid active state file: {self.path}") from exc
        return self.prune_expired(state)

    def prune_expired(self, state: ActiveState, now: datetime | None = None) -> ActiveState:
        now = now or datetime.now(timezone.utc)
        records: list[ActiveIntentRecord] = []
        for record in state.intents:
            end = record.intent.time_constraint_end
            if end is not None:
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                else:
                    end = end.astimezone(timezone.utc)
                if end < now and record.status in {"ACTIVE", "SUPPRESSED"}:
                    record = record.model_copy(update={"status": "EXPIRED"})
            records.append(record)

        if not any(record.status in {"ACTIVE", "SUPPRESSED"} for record in records):
            effective = default_network_configuration()
        else:
            effective = state.effective_configuration
        return state.model_copy(update={"intents": records, "effective_configuration": effective})

    def save(self, state: ActiveState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = state.model_dump_json(indent=2)
        fd, temporary_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def apply_result(
        self,
        state: ActiveState,
        optimization: OptimizationResult,
        resolution: ResolutionResult,
    ) -> ActiveState:
        new_record = optimization_to_record(optimization)
        updated_records: list[ActiveIntentRecord] = []

        participant_ids = set(resolution.participant_result_ids)
        suppressed_ids = set(resolution.suppressed_result_ids)
        for record in state.intents:
            updates = {}
            if record.result_id in participant_ids and resolution.meta_agent_id:
                updates["resolution_group_id"] = resolution.meta_agent_id
            if record.result_id in suppressed_ids:
                updates["status"] = "SUPPRESSED"
            elif resolution.strategy == ResolutionStrategy.PRIORITY and record.result_id == resolution.winner_result_id:
                updates["status"] = "ACTIVE"
            if updates:
                record = record.model_copy(update=updates)
            updated_records.append(record)

        if new_record.result_id in suppressed_ids:
            new_record = new_record.model_copy(update={"status": "SUPPRESSED"})
        if resolution.meta_agent_id:
            new_record = new_record.model_copy(update={"resolution_group_id": resolution.meta_agent_id})
        updated_records.append(new_record)

        return ActiveState(
            updated_at=datetime.now(timezone.utc),
            effective_configuration=resolution.final_configuration,
            intents=updated_records,
        )
