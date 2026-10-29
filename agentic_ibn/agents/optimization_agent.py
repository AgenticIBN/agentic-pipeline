from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from ..constraints import extract_hard_constraints
from ..optimization.archive import OptimizationArchive
from ..optimization.objective import ObjectiveEvaluator
from ..optimization.validator import CandidateValidator
from ..schemas import (
    CandidateBatch,
    CandidateProposal,
    CandidateRole,
    HardConstraints,
    KPIName,
    KPIPrediction,
    NetworkConfig,
    OptimizationIteration,
    OptimizationResult,
    ParsedIntent,
    RoleProposal,
)
from ..surrogate.model import SurrogateModel
from .runtime import build_groq_agent, coerce_response_model

OPTIMIZATION_INSTRUCTIONS = [
    "You are the Optimization Agent in a conflict-aware 6G intent-based networking system.",
    "Return a batch of complete absolute configurations for TX0 through TX3; never return parameter deltas.",
    "Return exactly the requested number of candidates, with the requested roles in the requested order.",
    "Use only the supplied surrogate predictions, archive rows and objective evaluations as numerical evidence.",
    "Never invent KPI values or claim simulator validation.",
    "The archive lists every candidate tried so far, sorted from worst to best score; the best one is last.",
    "Each archive row says which parameters changed relative to an earlier entry (compared_with_id) and how every KPI and the score moved; use these to judge which directions help and reverse directions that hurt.",
    "Never propose a configuration that already appears in the archive.",
    "Respect every explicit TX ON/OFF hard constraint exactly.",
    "Keep k_users, user_set_id and coverage_threshold_dbm equal to the baseline; they describe the environment, not knobs.",
    "Use only the allowed values when a list is given; otherwise stay within the given range.",
    "When a transmitter is disabled for maintenance, compensate only with transmitters that are allowed to remain active.",
    "Among equally promising candidates prefer smaller changes and lower total transmit power.",
    "Give each candidate a concise engineering rationale and its expected trade-offs.",
    "Return only schema-compliant structured data.",
]

_TX_KNOBS = ("power_dbm", "azimuth_delta_deg", "elevation_delta_deg")


def create_optimization_agent(model_id: str) -> Any:
    return build_groq_agent(
        model_id=model_id,
        name="Optimization Agent",
        role="Proposes batches of network configurations from the optimization archive.",
        description="Agentic proposal component inside the deterministic optimization loop.",
        instructions=OPTIMIZATION_INSTRUCTIONS,
        output_schema=CandidateBatch,
        structured_outputs=True,
        markdown=False,
        retries=2,
    )


def round_roles(round_index: int, candidates_per_round: int) -> list[CandidateRole]:
    """Job of each candidate in a round: explore first, then learn single changes, combine and roam."""
    n = max(1, candidates_per_round)
    if round_index == 0:
        return ["explore"] * n
    if n == 1:
        return ["free"]
    if n == 2:
        return ["learn", "combine"]
    return ["learn"] * (n - 2) + ["combine", "free"]


@dataclass
class ProposalRequest:
    """Everything a candidate generator gets for one round."""

    intent: ParsedIntent
    baseline: NetworkConfig
    baseline_prediction: KPIPrediction
    hard_constraints: HardConstraints
    allowed_values: dict[str, list[float] | tuple[float, float]]
    archive: OptimizationArchive
    archive_view: dict[str, Any]
    round_index: int
    max_rounds: int
    roles: list[CandidateRole]

    @property
    def start_configuration(self) -> NetworkConfig:
        return self.archive.reference().configuration


class CandidateGenerator(Protocol):
    def propose_batch(self, request: ProposalRequest) -> list[RoleProposal]: ...


def _role_task(request: ProposalRequest) -> str:
    n = len(request.roles)
    round_label = f"Round {request.round_index + 1} of {request.max_rounds}"
    if request.round_index == 0:
        return (
            f"{round_label} (explore). Propose {n} complete candidate configuration(s) with role 'explore'. "
            "Make them clearly different from each other (different power, tilt and azimuth strategies) so the "
            "next rounds learn which direction works for this intent."
        )
    reference = request.archive.reference()
    role_help = {
        "learn": (
            "- learn: change only 1-2 parameters of the start configuration, picking changes the archive has not "
            "tried yet, so their effect on each KPI can be measured."
        ),
        "combine": (
            "- combine: apply together the changes that improved the score in earlier rows (score_change > 0); "
            "this candidate may move 3-4 parameters by small steps."
        ),
        "free": "- free: the change you judge most promising given the remaining target gaps and the trade-offs.",
    }
    header = (
        f"{round_label} (exploit). Start from the best entry so far (id {reference.entry_id}; its full "
        "configuration is in start_configuration). Return exactly these roles in this order: "
        + ", ".join(request.roles)
        + "."
    )
    return "\n".join([header, *(text for role, text in role_help.items() if role in request.roles)])


class AgnoCandidateGenerator:
    """Calls the real Agno Optimization Agent for candidate generation."""

    def __init__(self, model_id: str):
        self.agent = create_optimization_agent(model_id)

    def propose_batch(self, request: ProposalRequest) -> list[RoleProposal]:
        payload = {
            "intent": request.intent.model_dump(mode="json"),
            "hard_constraints": request.hard_constraints.model_dump(mode="json"),
            "fixed_environment": {
                "k_users": request.baseline.k_users,
                "user_set_id": request.baseline.user_set_id,
                "coverage_threshold_dbm": request.baseline.coverage_threshold_dbm,
            },
            "allowed_values": {
                name: ({"choices": values} if isinstance(values, list) else {"min": values[0], "max": values[1]})
                for name, values in request.allowed_values.items()
            },
            "requested_roles": request.roles,
            "start_configuration": request.start_configuration.model_dump(mode="json"),
            "archive": request.archive_view,
        }
        task = _role_task(request)
        response = self.agent.run(task + "\n\nJSON context:\n" + json.dumps(payload, indent=2))
        return coerce_response_model(response, CandidateBatch).candidates


class HeuristicCandidateGenerator:
    """Offline test generator; not the production agentic decision path.

    Follows the same round plan as the agent (explore, then learn / combine / free around the best
    entry) with fixed rules, so the loop and the benchmark can run without an LLM.
    """

    _AZIMUTH_PATTERNS = (
        (-20.0, 20.0, -10.0, 10.0),
        (-15.0, 15.0, -5.0, 5.0),
        (-30.0, -10.0, 10.0, 30.0),
        (0.0, 0.0, 0.0, 0.0),
    )

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

    @staticmethod
    def _choices(allowed: list[float] | tuple[float, float]) -> list[float]:
        if isinstance(allowed, list):
            return list(allowed)
        low, high = allowed
        return [low + (high - low) * step / 4 for step in range(5)]

    def _explore(self, request: ProposalRequest, variant: int) -> NetworkConfig:
        data = request.baseline.model_dump()
        powers = self._choices(request.allowed_values["power_dbm"])
        tilts = self._choices(request.allowed_values["elevation_delta_deg"])
        target_set = set(request.intent.target_kpis)
        # Power-saving intents explore from the low end, everything else from the high end.
        power_order = powers if KPIName.TOTAL_TX_POWER in target_set else list(reversed(powers))
        azimuths = self._AZIMUTH_PATTERNS[variant % len(self._AZIMUTH_PATTERNS)]
        for index in range(4):
            tx = data[f"tx{index}"]
            if request.hard_constraints.tx_states.get(index) is False:
                continue
            tx["on"] = True
            tx["power_dbm"] = power_order[(variant // 2) % len(power_order)]
            tx["azimuth_delta_deg"] = azimuths[index]
            tx["elevation_delta_deg"] = tilts[(variant + index) % len(tilts)]
        self._apply_hard_states(data, request.hard_constraints)
        return NetworkConfig.model_validate(data)

    def _neighbours(self, request: ProposalRequest, start: NetworkConfig) -> list[NetworkConfig]:
        """Configurations one grid step away from `start` on a single knob, not yet in the archive."""
        found: list[NetworkConfig] = []
        for knob in _TX_KNOBS:
            choices = self._choices(request.allowed_values[knob])
            for index in range(4):
                tx = start.tx(index)
                if not tx.on:
                    continue
                current = float(getattr(tx, knob))
                position = min(range(len(choices)), key=lambda i: abs(choices[i] - current))
                for step in (1, -1):
                    other = position + step
                    if not 0 <= other < len(choices):
                        continue
                    candidate = start.with_tx(index, tx.model_copy(update={knob: choices[other]}))
                    if not request.archive.contains(candidate):
                        found.append(candidate)
        return found

    def _combine(self, request: ProposalRequest, start: NetworkConfig) -> NetworkConfig | None:
        """Apply every single change that improved the score in earlier rounds on top of `start`."""
        helpful = [
            entry
            for entry in request.archive.evaluated()
            if entry.role == "learn" and (entry.score_delta or 0.0) > 0 and entry.changes
        ]
        if not helpful:
            return None
        configuration = start
        for entry in helpful:
            for change in entry.changes:
                if change.parameter == "on":
                    continue
                tx = configuration.tx(change.tx)
                if tx.on:
                    configuration = configuration.with_tx(change.tx, tx.model_copy(update={change.parameter: change.after}))
        return None if request.archive.contains(configuration) else configuration

    def propose_batch(self, request: ProposalRequest) -> list[RoleProposal]:
        proposals: list[RoleProposal] = []
        if request.round_index == 0:
            for variant, role in enumerate(request.roles):
                proposals.append(
                    RoleProposal(
                        role=role,
                        configuration=self._explore(request, variant),
                        rationale=f"Exploration variant {variant}: distinct power, azimuth and tilt pattern.",
                    )
                )
            return proposals

        start = request.start_configuration
        neighbours = self._neighbours(request, start)
        rng = random.Random(request.round_index)
        used: set[int] = set()
        for role in request.roles:
            configuration: NetworkConfig | None = None
            rationale = ""
            if role == "learn" and neighbours:
                pick = next((i for i in range(len(neighbours)) if i not in used), None)
                if pick is not None:
                    used.add(pick)
                    configuration = neighbours[pick]
                    rationale = "Single-knob step from the best entry to measure its effect."
            elif role == "combine":
                configuration = self._combine(request, start)
                rationale = "Combines the single changes that improved the score earlier."
            if configuration is None and neighbours:
                pool = [i for i in range(len(neighbours)) if i not in used] or list(range(len(neighbours)))
                first = rng.choice(pool)
                used.add(first)
                configuration = neighbours[first]
                second = self._neighbours(request, configuration)
                if second:
                    configuration = rng.choice(second)
                rationale = rationale or "Two-knob move around the best entry."
            if configuration is not None:
                proposals.append(RoleProposal(role=role, configuration=configuration, rationale=rationale))
        return proposals


class OptimizationLoop:
    """Deterministic evaluation loop around the Agno Optimization Agent.

    Each round the generator proposes a batch of candidates; every candidate is validated, scored by
    the surrogate and stored in the archive, which the next round sees in full. The loop stops when
    every numeric target is met, when a round brings no improvement, or after max_iterations rounds.
    """

    def __init__(
        self,
        surrogate: SurrogateModel,
        candidate_generator: CandidateGenerator,
        max_iterations: int = 3,
        candidates_per_round: int = 4,
        history: str = "full",
        stop_on_stall: bool = True,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if candidates_per_round < 1:
            raise ValueError("candidates_per_round must be at least 1")
        if history not in {"full", "last"}:
            raise ValueError("history must be 'full' or 'last'")
        self.surrogate = surrogate
        self.candidate_generator = candidate_generator
        self.max_iterations = max_iterations
        self.candidates_per_round = candidates_per_round
        self.history = history
        self.stop_on_stall = stop_on_stall
        self.validator = CandidateValidator(surrogate)
        self.evaluator = ObjectiveEvaluator()

    def optimize(self, intent: ParsedIntent, baseline: NetworkConfig) -> OptimizationResult:
        created_at = datetime.now(timezone.utc)
        result_id = f"opt_{created_at.strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
        hard_constraints = extract_hard_constraints(intent.raw_text)
        baseline_prediction = self.surrogate.predict(baseline)
        baseline_evaluation = self.evaluator.evaluate(intent, baseline_prediction, baseline_prediction)
        archive = OptimizationArchive(intent, baseline, baseline_prediction, baseline_evaluation)
        allowed_values = self.validator.allowed_values()

        iterations: list[OptimizationIteration] = []
        generator_calls = 0
        rounds_run = 0

        for round_index in range(self.max_iterations):
            roles = round_roles(round_index, self.candidates_per_round)
            request = ProposalRequest(
                intent=intent,
                baseline=baseline,
                baseline_prediction=baseline_prediction,
                hard_constraints=hard_constraints,
                allowed_values=allowed_values,
                archive=archive,
                archive_view=archive.prompt_view(history=self.history),
                round_index=round_index,
                max_rounds=self.max_iterations,
                roles=roles,
            )
            proposals = self.candidate_generator.propose_batch(request)
            generator_calls += 1
            rounds_run += 1
            best_before = archive.best_score()

            for proposal in proposals[: len(roles)]:
                validation = self.validator.validate(proposal.configuration, hard_constraints, context=baseline)
                configuration = validation.configuration
                if archive.contains(configuration):
                    continue
                if not validation.valid:
                    archive.add_rejected(
                        round_index=round_index,
                        role=proposal.role,
                        configuration=configuration,
                        errors=validation.errors,
                        rationale=proposal.rationale,
                    )
                    continue
                prediction = self.surrogate.predict(configuration)
                evaluation = self.evaluator.evaluate(intent, prediction, baseline_prediction)
                archive.add_evaluated(
                    round_index=round_index,
                    role=proposal.role,
                    configuration=configuration,
                    prediction=prediction,
                    evaluation=evaluation,
                    rationale=proposal.rationale,
                )
                iterations.append(
                    OptimizationIteration(
                        iteration=len(iterations),
                        round=round_index,
                        role=proposal.role,
                        proposal=CandidateProposal(
                            configuration=proposal.configuration,
                            rationale=proposal.rationale,
                            expected_tradeoffs=proposal.expected_tradeoffs,
                        ),
                        validation=validation,
                        prediction=prediction,
                        evaluation=evaluation,
                    )
                )

            best = archive.best()
            if best is not None and best.evaluation.all_numeric_targets_met:  # type: ignore[union-attr]
                break
            best_after = archive.best_score()
            if (
                self.stop_on_stall
                and best_before is not None
                and best_after is not None
                and best_after <= best_before + 1e-9
            ):
                break

        best = archive.best()
        if best is None:
            reasons = sorted({error for entry in archive.rejected() for error in entry.errors})
            raise RuntimeError(
                "Optimization produced no valid candidate" + (": " + "; ".join(reasons) if reasons else "")
            )

        return OptimizationResult(
            result_id=result_id,
            created_at=created_at,
            intent=intent,
            baseline_configuration=baseline,
            baseline_prediction=baseline_prediction,
            final_configuration=best.configuration,
            predicted_kpis=best.prediction,  # type: ignore[arg-type]
            target_satisfied=best.evaluation.all_numeric_targets_met,  # type: ignore[union-attr]
            best_score=best.evaluation.score,  # type: ignore[union-attr]
            iterations=iterations,
            hard_constraints=hard_constraints,
            model_scenario=self.surrogate.scenario,
            archive=archive.entries,
            rounds_run=rounds_run,
            generator_calls=generator_calls,
        )
