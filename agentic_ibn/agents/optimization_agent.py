from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from ..constraints import extract_hard_constraints
from ..optimization.objective import ObjectiveEvaluator
from ..optimization.validator import CandidateValidator
from ..schemas import (
    CandidateProposal,
    HardConstraints,
    KPIName,
    KPIPrediction,
    NetworkConfig,
    ObjectiveEvaluation,
    OptimizationIteration,
    OptimizationResult,
    ParsedIntent,
)
from ..surrogate.model import SurrogateModel
from .runtime import build_groq_agent, coerce_response_model

OPTIMIZATION_INSTRUCTIONS = [
    "You are the Optimization Agent in a conflict-aware 6G intent-based networking system.",
    "Return one complete absolute configuration for TX0 through TX3; never return parameter deltas.",
    "Use only the supplied surrogate predictions and objective evaluations as numerical evidence.",
    "Never invent KPI values or claim simulator validation.",
    "Respect every explicit TX ON/OFF hard constraint exactly.",
    "Keep all values within the supplied model-supported ranges.",
    "Prefer the smallest safe configuration change that improves the requested objective.",
    "When a transmitter is disabled for maintenance, compensate only with transmitters that are allowed to remain active.",
    "Provide a concise engineering rationale and list expected trade-offs.",
    "Return only schema-compliant structured data.",
]


def create_optimization_agent(model_id: str) -> Any:
    return build_groq_agent(
        model_id=model_id,
        name="Optimization Agent",
        role="Proposes and refines network configurations from surrogate feedback.",
        description="Agentic proposal component inside the deterministic optimization loop.",
        instructions=OPTIMIZATION_INSTRUCTIONS,
        output_schema=CandidateProposal,
        structured_outputs=True,
        markdown=False,
        retries=2,
    )


class CandidateGenerator(Protocol):
    def initial_proposal(
        self,
        intent: ParsedIntent,
        baseline: NetworkConfig,
        baseline_prediction: KPIPrediction,
        hard_constraints: HardConstraints,
        parameter_ranges: dict[str, tuple[float, float]],
    ) -> CandidateProposal: ...

    def refine_proposal(
        self,
        intent: ParsedIntent,
        baseline: NetworkConfig,
        current: NetworkConfig,
        current_prediction: KPIPrediction,
        evaluation: ObjectiveEvaluation,
        hard_constraints: HardConstraints,
        parameter_ranges: dict[str, tuple[float, float]],
        iteration: int,
    ) -> CandidateProposal: ...


class AgnoCandidateGenerator:
    """Calls the real Agno Optimization Agent for candidate generation."""

    def __init__(self, model_id: str):
        self.agent = create_optimization_agent(model_id)

    def _run(self, payload: dict[str, Any], task: str) -> CandidateProposal:
        response = self.agent.run(task + "\n\nJSON context:\n" + json.dumps(payload, indent=2))
        return coerce_response_model(response, CandidateProposal)

    def initial_proposal(
        self,
        intent: ParsedIntent,
        baseline: NetworkConfig,
        baseline_prediction: KPIPrediction,
        hard_constraints: HardConstraints,
        parameter_ranges: dict[str, tuple[float, float]],
    ) -> CandidateProposal:
        payload = {
            "intent": intent.model_dump(mode="json"),
            "baseline_configuration": baseline.model_dump(mode="json"),
            "baseline_surrogate_prediction": baseline_prediction.model_dump(mode="json"),
            "hard_constraints": hard_constraints.model_dump(mode="json"),
            "model_supported_ranges": parameter_ranges,
        }
        return self._run(payload, "Design the first complete candidate configuration.")

    def refine_proposal(
        self,
        intent: ParsedIntent,
        baseline: NetworkConfig,
        current: NetworkConfig,
        current_prediction: KPIPrediction,
        evaluation: ObjectiveEvaluation,
        hard_constraints: HardConstraints,
        parameter_ranges: dict[str, tuple[float, float]],
        iteration: int,
    ) -> CandidateProposal:
        payload = {
            "iteration": iteration,
            "intent": intent.model_dump(mode="json"),
            "baseline_configuration": baseline.model_dump(mode="json"),
            "current_configuration": current.model_dump(mode="json"),
            "current_surrogate_prediction": current_prediction.model_dump(mode="json"),
            "objective_evaluation": evaluation.model_dump(mode="json"),
            "hard_constraints": hard_constraints.model_dump(mode="json"),
            "model_supported_ranges": parameter_ranges,
        }
        return self._run(payload, "Refine the candidate using the latest surrogate feedback and target gaps.")


class HeuristicCandidateGenerator:
    """Offline test generator; not the production agentic decision path."""

    @staticmethod
    def _apply_hard_states(data: dict[str, Any], constraints: HardConstraints) -> None:
        for index, state in constraints.tx_states.items():
            data[f"tx{index}"]["on"] = state
            if not state:
                data[f"tx{index}"].update(
                    power_dbm=0.0,
                    azimuth_delta_deg=0.0,
                    elevation_delta_deg=0.0,
                )

    def _propose(
        self,
        intent: ParsedIntent,
        current: NetworkConfig,
        hard_constraints: HardConstraints,
        iteration: int,
    ) -> CandidateProposal:
        data = current.model_dump()
        target_set = set(intent.target_kpis)
        step = 1.5 + iteration

        if KPIName.TOTAL_TX_POWER in target_set:
            for index in range(4):
                tx = data[f"tx{index}"]
                if tx["on"] and index not in hard_constraints.tx_states:
                    tx["power_dbm"] = max(20.0, float(tx["power_dbm"]) - step)
            rationale = "Reduce active-transmitter power while preserving hard TX states."
            tradeoffs = ["Lower power can reduce RX power and coverage."]
        elif target_set & {KPIName.SINR, KPIName.THROUGHPUT_5P, KPIName.THROUGHPUT_RR_5P}:
            angles = [-15.0, 15.0, -5.0, 5.0]
            for index in range(4):
                tx = data[f"tx{index}"]
                if tx["on"]:
                    tx["power_dbm"] = min(46.0, max(35.0, float(tx["power_dbm"])))
                    tx["azimuth_delta_deg"] = angles[index]
            rationale = "Use moderate power and angular separation to improve quality."
            tradeoffs = ["Interference reduction may lower raw received power in some locations."]
        elif target_set & {KPIName.LOAD_BALANCE, KPIName.LOAD_IMBALANCE, KPIName.SERVED_USERS}:
            angles = [-30.0, -10.0, 10.0, 30.0]
            for index in range(4):
                tx = data[f"tx{index}"]
                if index not in hard_constraints.tx_states or hard_constraints.tx_states[index]:
                    tx["on"] = True
                    tx["power_dbm"] = 43.0
                    tx["azimuth_delta_deg"] = angles[index]
            rationale = "Use comparable power and spatially diverse beams to spread associations."
            tradeoffs = ["More active transmitters can increase interference and energy use."]
        else:
            angles = [-20.0, 20.0, -10.0, 10.0]
            for index in range(4):
                tx = data[f"tx{index}"]
                if index not in hard_constraints.tx_states or hard_constraints.tx_states[index]:
                    tx["on"] = True
                    tx["power_dbm"] = min(46.0, max(35.0, float(tx["power_dbm"]) + step))
                    tx["azimuth_delta_deg"] = angles[index]
            rationale = "Increase allowed active-TX power and spread beams for coverage-oriented KPIs."
            tradeoffs = ["Higher power can increase interference and energy consumption."]

        self._apply_hard_states(data, hard_constraints)
        return CandidateProposal(
            configuration=NetworkConfig.model_validate(data),
            rationale=rationale,
            expected_tradeoffs=tradeoffs,
        )

    def initial_proposal(
        self,
        intent: ParsedIntent,
        baseline: NetworkConfig,
        baseline_prediction: KPIPrediction,
        hard_constraints: HardConstraints,
        parameter_ranges: dict[str, tuple[float, float]],
    ) -> CandidateProposal:
        return self._propose(intent, baseline, hard_constraints, 0)

    def refine_proposal(
        self,
        intent: ParsedIntent,
        baseline: NetworkConfig,
        current: NetworkConfig,
        current_prediction: KPIPrediction,
        evaluation: ObjectiveEvaluation,
        hard_constraints: HardConstraints,
        parameter_ranges: dict[str, tuple[float, float]],
        iteration: int,
    ) -> CandidateProposal:
        return self._propose(intent, current, hard_constraints, iteration)


class OptimizationLoop:
    """Deterministic evaluation loop around the Agno Optimization Agent."""

    def __init__(
        self,
        surrogate: SurrogateModel,
        candidate_generator: CandidateGenerator,
        max_iterations: int = 3,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.surrogate = surrogate
        self.candidate_generator = candidate_generator
        self.max_iterations = max_iterations
        self.validator = CandidateValidator(surrogate)
        self.evaluator = ObjectiveEvaluator()

    def _parameter_ranges(self) -> dict[str, tuple[float, float]]:
        return {
            "power_dbm": self.surrogate.bounds_for("power_dbm", 0.0, 60.0),
            "azimuth_delta_deg": self.surrogate.bounds_for("azimuth_delta_deg", -180.0, 180.0),
            "elevation_delta_deg": self.surrogate.bounds_for("elevation_delta_deg", -90.0, 90.0),
        }

    def optimize(self, intent: ParsedIntent, baseline: NetworkConfig) -> OptimizationResult:
        created_at = datetime.now(timezone.utc)
        result_id = f"opt_{created_at.strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
        hard_constraints = extract_hard_constraints(intent.raw_text)
        baseline_prediction = self.surrogate.predict(baseline)
        parameter_ranges = self._parameter_ranges()

        iterations: list[OptimizationIteration] = []
        current_config = baseline
        current_prediction = baseline_prediction
        current_evaluation = self.evaluator.evaluate(intent, current_prediction, baseline_prediction)
        best_iteration: OptimizationIteration | None = None

        for iteration_number in range(self.max_iterations):
            if iteration_number == 0:
                proposal = self.candidate_generator.initial_proposal(
                    intent,
                    baseline,
                    baseline_prediction,
                    hard_constraints,
                    parameter_ranges,
                )
            else:
                proposal = self.candidate_generator.refine_proposal(
                    intent,
                    baseline,
                    current_config,
                    current_prediction,
                    current_evaluation,
                    hard_constraints,
                    parameter_ranges,
                    iteration_number,
                )

            validation = self.validator.validate(proposal.configuration, hard_constraints)
            if not validation.valid:
                raise ValueError("Candidate validation failed: " + "; ".join(validation.errors))

            current_config = validation.configuration
            current_prediction = self.surrogate.predict(current_config)
            current_evaluation = self.evaluator.evaluate(intent, current_prediction, baseline_prediction)
            trace = OptimizationIteration(
                iteration=iteration_number,
                proposal=proposal,
                validation=validation,
                prediction=current_prediction,
                evaluation=current_evaluation,
            )
            iterations.append(trace)
            if best_iteration is None or trace.evaluation.score > best_iteration.evaluation.score:
                best_iteration = trace
            if current_evaluation.all_numeric_targets_met:
                break

        if best_iteration is None:
            raise RuntimeError("Optimization produced no candidate")

        return OptimizationResult(
            result_id=result_id,
            created_at=created_at,
            intent=intent,
            baseline_configuration=baseline,
            baseline_prediction=baseline_prediction,
            final_configuration=best_iteration.validation.configuration,
            predicted_kpis=best_iteration.prediction,
            target_satisfied=best_iteration.evaluation.all_numeric_targets_met,
            best_score=best_iteration.evaluation.score,
            iterations=iterations,
            hard_constraints=hard_constraints,
            model_scenario=self.surrogate.scenario,
        )
