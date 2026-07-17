from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..agents.conflict_detector_agent import run_conflict_detector_agent
from ..agents.intent_parser_agent import run_intent_parser
from ..agents.meta_agent import ConflictResolutionCoordinator
from ..agents.optimization_agent import (
    AgnoCandidateGenerator,
    CandidateGenerator,
    HeuristicCandidateGenerator,
    OptimizationLoop,
)
from ..agents.reasoning_agent import ReasoningNarrator
from ..intent_adapter import adapt_intent_parse
from ..schemas import (
    AgentHistoryEntry,
    ConflictReport,
    KPIPrediction,
    OptimizationResult,
    ParsedIntent,
    ResolutionResult,
    StrategicNarrative,
    WorkflowResult,
)
from ..storage.result_store import ResultStore
from ..storage.state_store import ActiveState, ActiveStateStore
from ..surrogate.model import SurrogateModel


@dataclass
class _RunContext:
    run_id: str
    run_directory: Path
    created_at: datetime
    intent_text: str
    scenario: str
    requested_strategy: str
    parsed_intent_override: ParsedIntent | None
    state_before: ActiveState
    input_payload: dict[str, Any]
    history: list[AgentHistoryEntry] = field(default_factory=list)
    intent: ParsedIntent | None = None
    optimization: OptimizationResult | None = None
    conflict_report: ConflictReport | None = None
    resolution: ResolutionResult | None = None
    final_prediction: KPIPrediction | None = None
    narrative: StrategicNarrative | None = None
    result: WorkflowResult | None = None


class AgenticWorkflow:
    """Agno Workflow orchestrator with a deterministic offline test mode."""

    def __init__(
        self,
        surrogate: SurrogateModel,
        state_store: ActiveStateStore,
        result_store: ResultStore,
        *,
        model_id: str | None = None,
        runtime_mode: str = "agno",
        max_iterations: int = 3,
        candidate_generator: CandidateGenerator | None = None,
        resolution_coordinator: ConflictResolutionCoordinator | None = None,
        reasoning_narrator: ReasoningNarrator | None = None,
    ):
        if runtime_mode not in {"agno", "deterministic"}:
            raise ValueError("runtime_mode must be 'agno' or 'deterministic'")
        if runtime_mode == "agno" and not model_id:
            raise ValueError("model_id is required for the Agno runtime")
        self.surrogate = surrogate
        self.state_store = state_store
        self.result_store = result_store
        self.model_id = model_id
        self.runtime_mode = runtime_mode

        if candidate_generator is None:
            candidate_generator = (
                AgnoCandidateGenerator(model_id)
                if runtime_mode == "agno" and model_id
                else HeuristicCandidateGenerator()
            )
        self.optimization_loop = OptimizationLoop(
            surrogate=surrogate,
            candidate_generator=candidate_generator,
            max_iterations=max_iterations,
        )
        self.resolution_coordinator = resolution_coordinator or ConflictResolutionCoordinator(
            model_id=model_id,
            use_llm=runtime_mode == "agno",
        )
        self.reasoning_narrator = reasoning_narrator or ReasoningNarrator(
            model_id=model_id,
            use_llm=runtime_mode == "agno",
        )

    @staticmethod
    def _serialize(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return value
        return {"value": value}

    @staticmethod
    def _record_step(
        context: _RunContext,
        step: str,
        step_input: dict[str, Any],
        function: Callable[[], Any],
    ) -> Any:
        started = datetime.now(timezone.utc)
        try:
            output = function()
            context.history.append(
                AgentHistoryEntry(
                    step=step,
                    started_at=started,
                    finished_at=datetime.now(timezone.utc),
                    status="success",
                    input=step_input,
                    output=AgenticWorkflow._serialize(output),
                )
            )
            return output
        except Exception as exc:
            context.history.append(
                AgentHistoryEntry(
                    step=step,
                    started_at=started,
                    finished_at=datetime.now(timezone.utc),
                    status="error",
                    input=step_input,
                    output=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise

    def _step_parse_intent(self, context: _RunContext) -> ParsedIntent:
        if context.parsed_intent_override is not None:
            intent = context.parsed_intent_override
            context.history.append(
                AgentHistoryEntry(
                    step="intent_parser_agent",
                    started_at=context.created_at,
                    finished_at=context.created_at,
                    status="success",
                    input={"intent_text": context.intent_text},
                    output={"mode": "preparsed", "intent": intent.model_dump(mode="json")},
                )
            )
        else:
            if self.runtime_mode != "agno" or not self.model_id:
                raise RuntimeError(
                    "Deterministic runtime requires --parsed-intent-file because it does not call an LLM parser."
                )
            parser_output = self._record_step(
                context,
                "intent_parser_agent",
                {"intent_text": context.intent_text},
                lambda: run_intent_parser(context.intent_text, self.model_id or ""),
            )
            intent = adapt_intent_parse(context.intent_text, parser_output, context.scenario)
        context.intent = intent
        return intent

    def _step_optimize(self, context: _RunContext) -> OptimizationResult:
        if context.intent is None:
            raise RuntimeError("Intent Parser Agent did not produce an intent")
        context.optimization = self._record_step(
            context,
            "optimization_agent",
            {
                "parsed_intent": context.intent.model_dump(mode="json"),
                "baseline_configuration": context.state_before.effective_configuration.model_dump(mode="json"),
                "runtime_mode": self.runtime_mode,
            },
            lambda: self.optimization_loop.optimize(
                context.intent,
                context.state_before.effective_configuration,
            ),
        )
        return context.optimization

    def _step_detect_conflicts(self, context: _RunContext) -> ConflictReport:
        if context.optimization is None:
            raise RuntimeError("Optimization Agent did not produce a result")
        active_records = [
            record for record in context.state_before.intents if record.status in {"ACTIVE", "SUPPRESSED"}
        ]
        context.conflict_report = self._record_step(
            context,
            "conflict_detector_agent",
            {
                "new_result_id": context.optimization.result_id,
                "active_result_ids": [record.result_id for record in active_records],
            },
            lambda: run_conflict_detector_agent(
                context.optimization,
                active_records,
                model_id=self.model_id,
                use_llm=self.runtime_mode == "agno",
            ),
        )
        return context.conflict_report

    def _step_resolve(self, context: _RunContext) -> ResolutionResult:
        if context.optimization is None or context.conflict_report is None:
            raise RuntimeError("Conflict resolution received incomplete upstream state")
        active_records = [
            record for record in context.state_before.intents if record.status in {"ACTIVE", "SUPPRESSED"}
        ]
        context.resolution = self._record_step(
            context,
            "meta_agent_conflict_resolution",
            {
                "recommended_strategy": context.conflict_report.recommended_strategy,
                "requested_strategy": context.requested_strategy,
                "conflict_detected": context.conflict_report.conflict_detected,
            },
            lambda: self.resolution_coordinator.resolve(
                new_result=context.optimization,
                active_records=active_records,
                report=context.conflict_report,
                requested_strategy=context.requested_strategy,
            ),
        )
        return context.resolution

    def _step_final_validation(self, context: _RunContext) -> KPIPrediction:
        if context.resolution is None:
            raise RuntimeError("Resolution step did not produce a final configuration")
        context.final_prediction = self._record_step(
            context,
            "surrogate_final_validation",
            {"final_configuration": context.resolution.final_configuration.model_dump(mode="json")},
            lambda: self.surrogate.predict(context.resolution.final_configuration),
        )
        return context.final_prediction

    def _step_reason_and_persist(self, context: _RunContext) -> WorkflowResult:
        if any(
            value is None
            for value in (
                context.optimization,
                context.conflict_report,
                context.resolution,
                context.final_prediction,
            )
        ):
            raise RuntimeError("Final reasoning received incomplete workflow state")
        optimization = context.optimization
        conflict_report = context.conflict_report
        resolution = context.resolution
        final_prediction = context.final_prediction
        assert optimization is not None
        assert conflict_report is not None
        assert resolution is not None
        assert final_prediction is not None

        context.narrative = self._record_step(
            context,
            "reasoning_agent",
            {
                "optimization_result_id": optimization.result_id,
                "resolution_strategy": resolution.strategy.value,
            },
            lambda: self.reasoning_narrator.explain(
                optimization,
                conflict_report,
                resolution,
                final_prediction,
            ),
        )
        state_after = self.state_store.apply_result(context.state_before, optimization, resolution)
        self.state_store.save(state_after)
        context.result = WorkflowResult(
            run_id=context.run_id,
            created_at=context.created_at,
            input_text=context.intent_text,
            scenario=context.scenario,
            optimization=optimization,
            conflict_report=conflict_report,
            resolution=resolution,
            final_prediction=final_prediction,
            reasoning=context.narrative,
            result_directory=str(context.run_directory),
        )
        self.result_store.persist(
            directory=context.run_directory,
            input_payload=context.input_payload,
            history=context.history,
            state_before=context.state_before,
            state_after=state_after,
            result=context.result,
        )
        return context.result

    def _run_deterministic(self, context: _RunContext) -> None:
        self._step_parse_intent(context)
        self._step_optimize(context)
        self._step_detect_conflicts(context)
        self._step_resolve(context)
        self._step_final_validation(context)
        self._step_reason_and_persist(context)

    def _run_agno_workflow(self, context: _RunContext) -> None:
        try:
            from agno.workflow import Step, Workflow
            from agno.workflow.step import StepOutput
        except ImportError as exc:
            raise RuntimeError(
                "Agno Workflow is unavailable. Install requirements.txt or use --runtime deterministic for offline tests."
            ) from exc

        def executor(function: Callable[[_RunContext], Any]) -> Callable[[Any], Any]:
            def run_step(_step_input: Any) -> Any:
                output = function(context)
                return StepOutput(content=self._serialize(output))

            return run_step

        workflow = Workflow(
            name="Conflict-Aware Agentic IBN Workflow",
            description="Sequential Agno workflow for parsing, optimization, conflict handling, validation, and explanation.",
            steps=[
                Step(name="Intent Parser Agent", executor=executor(self._step_parse_intent), max_retries=0, on_error="fail"),
                Step(name="Optimization Agent", executor=executor(self._step_optimize), max_retries=0, on_error="fail"),
                Step(name="Conflict Detector Agent", executor=executor(self._step_detect_conflicts), max_retries=0, on_error="fail"),
                Step(name="Meta Agent Resolution", executor=executor(self._step_resolve), max_retries=0, on_error="fail"),
                Step(name="Final Surrogate Validation", executor=executor(self._step_final_validation), max_retries=0, on_error="fail"),
                Step(name="Reasoning Agent and Persistence", executor=executor(self._step_reason_and_persist), max_retries=0, on_error="fail"),
            ],
        )
        workflow.run(input=context.input_payload, stream=False)

    def run(
        self,
        intent_text: str,
        scenario: str,
        requested_strategy: str = "AUTO",
        parsed_intent: ParsedIntent | None = None,
    ) -> WorkflowResult:
        created_at = datetime.now(timezone.utc)
        run_id, run_directory = self.result_store.create_run_directory()
        state_before = self.state_store.load()
        input_payload = {
            "intent_text": intent_text,
            "scenario": scenario,
            "requested_strategy": requested_strategy,
            "runtime_mode": self.runtime_mode,
            "surrogate_artifact": str(self.surrogate.source_path) if self.surrogate.source_path else None,
            "surrogate_model_family": self.surrogate.model_family,
        }
        context = _RunContext(
            run_id=run_id,
            run_directory=run_directory,
            created_at=created_at,
            intent_text=intent_text,
            scenario=scenario,
            requested_strategy=requested_strategy,
            parsed_intent_override=parsed_intent,
            state_before=state_before,
            input_payload=input_payload,
        )

        try:
            if self.runtime_mode == "agno":
                self._run_agno_workflow(context)
            else:
                self._run_deterministic(context)
            if context.result is None:
                raise RuntimeError("Workflow completed without a final result")
            return context.result
        except Exception as exc:
            self.result_store.write_json(run_directory / "input.json", input_payload)
            self.result_store.write_json(run_directory / "agent_history.json", context.history)
            self.result_store.write_json(
                run_directory / "error.json",
                {"error_type": type(exc).__name__, "message": str(exc)},
            )
            raise
