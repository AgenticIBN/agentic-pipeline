from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_ibn.agents.optimization_agent import (
    AgnoCandidateGenerator,
    HeuristicCandidateGenerator,
    OptimizationLoop,
    ProposalRequest,
    round_roles,
)
from agentic_ibn.optimization.archive import (
    OptimizationArchive,
    configuration_changes,
    describe_change,
)
from agentic_ibn.optimization.objective import ObjectiveEvaluator
from agentic_ibn.optimization.validator import CandidateValidator
from agentic_ibn.schemas import (
    CandidateBatch,
    HardConstraints,
    NetworkConfig,
    RoleProposal,
    TxConfig,
)

from .conftest import POWERS, TILTS, grid_flat_frame


def _with(config: NetworkConfig, index: int, **updates) -> NetworkConfig:
    return config.with_tx(index, config.tx(index).model_copy(update=updates))


class RecordingGenerator:
    """Wraps another generator and keeps every request it received."""

    def __init__(self, inner):
        self.inner = inner
        self.requests: list[ProposalRequest] = []

    def propose_batch(self, request: ProposalRequest) -> list[RoleProposal]:
        self.requests.append(request)
        return self.inner.propose_batch(request)


def test_round_roles_explore_then_exploit():
    assert round_roles(0, 4) == ["explore"] * 4
    assert round_roles(1, 4) == ["learn", "learn", "combine", "free"]
    assert round_roles(2, 2) == ["learn", "combine"]
    assert round_roles(1, 1) == ["free"]


def test_changes_are_described_per_knob(baseline):
    candidate = _with(_with(baseline, 0, elevation_delta_deg=5.0), 2, on=False, power_dbm=0.0, azimuth_delta_deg=0.0, elevation_delta_deg=0.0)
    text = [describe_change(change) for change in configuration_changes(candidate, baseline)]
    assert text == ["TX0 tilt 10->5", "TX2 ON->OFF"]


def test_validator_snaps_to_grid_and_pins_environment(surrogate, baseline):
    validator = CandidateValidator(surrogate)
    proposal = _with(baseline, 1, power_dbm=45.0, elevation_delta_deg=14.0, azimuth_delta_deg=-40.0)
    proposal = proposal.model_copy(update={"k_users": 800})
    result = validator.validate(proposal, HardConstraints(), context=baseline)
    tx1 = result.configuration.tx1
    assert (tx1.power_dbm, tx1.elevation_delta_deg, tx1.azimuth_delta_deg) == (46.0, 10.0, -30.0)
    assert result.configuration.k_users == baseline.k_users
    assert {c.parameter for c in result.corrections} >= {"k_users", "tx1.power_dbm", "tx1.elevation_delta_deg"}
    assert validator.allowed_values()["power_dbm"] == POWERS


def test_batch_prediction_matches_single_prediction(surrogate):
    flat = grid_flat_frame().sample(n=5, random_state=1).reset_index(drop=True)
    batch = surrogate.predict_flat_frame(flat)
    for row, prediction in enumerate(batch):
        single = surrogate.predict(NetworkConfig.from_flat_dict(flat.iloc[row].to_dict()))
        assert prediction.coverage_ratio == pytest.approx(single.coverage_ratio)
        assert prediction.sinr_p5_db == pytest.approx(single.sinr_p5_db)
        assert prediction.load_balance == pytest.approx(single.load_balance)


def test_loop_runs_rounds_with_full_archive(surrogate, baseline, maintenance_intent):
    generator = RecordingGenerator(HeuristicCandidateGenerator())
    loop = OptimizationLoop(surrogate, generator, max_iterations=3, candidates_per_round=4, stop_on_stall=False)
    result = loop.optimize(maintenance_intent, baseline)

    assert result.generator_calls == result.rounds_run == 3
    assert result.archive[0].role == "baseline"
    assert len(result.iterations) == len([e for e in result.archive if e.status == "evaluated"])
    # Hard constraint from the text: TX2 stays off in every candidate.
    assert all(not entry.configuration.tx2.on for entry in result.archive[1:])
    # Every evaluated configuration is unique and on the grid.
    keys = {entry.configuration.model_dump_json() for entry in result.archive}
    assert len(keys) == len(result.archive)
    for entry in result.archive[1:]:
        for tx in (entry.configuration.tx(i) for i in range(4)):
            if tx.on:
                assert tx.power_dbm in POWERS and tx.elevation_delta_deg in TILTS

    # Round 2 sees round 1 sorted worst to best, with what changed and how KPIs moved.
    view = generator.requests[1].archive_view
    rows = view["tried_candidates_worst_to_best"]
    assert len(rows) == 4
    assert [row["score"] for row in rows] == sorted(row["score"] for row in rows)
    assert rows[-1]["id"] == view["best_candidate_id"]
    assert all("changes" in row and "kpi_change" in row for row in rows)
    assert generator.requests[1].roles == ["learn", "learn", "combine", "free"]
    # Best score never gets worse across rounds and the result is the archive best.
    assert result.best_score == max(e.evaluation.score for e in result.archive if e.status == "evaluated")


def test_history_last_shows_only_latest_candidate(surrogate, baseline, maintenance_intent):
    generator = RecordingGenerator(HeuristicCandidateGenerator())
    loop = OptimizationLoop(surrogate, generator, max_iterations=2, candidates_per_round=1, history="last", stop_on_stall=False)
    loop.optimize(maintenance_intent, baseline)
    assert len(generator.requests[1].archive_view["tried_candidates_worst_to_best"]) == 1


def test_invalid_candidate_is_archived_not_fatal(surrogate, baseline, maintenance_intent):
    all_off = NetworkConfig(**{f"tx{i}": TxConfig(on=False, power_dbm=0.0, azimuth_delta_deg=0.0, elevation_delta_deg=0.0) for i in range(4)})
    good = _with(baseline, 0, power_dbm=46.0)

    class FixedGenerator:
        def propose_batch(self, request):
            return [
                RoleProposal(role="explore", configuration=all_off, rationale="bad"),
                RoleProposal(role="explore", configuration=good, rationale="good"),
            ]

    loop = OptimizationLoop(surrogate, FixedGenerator(), max_iterations=1, candidates_per_round=2)
    result = loop.optimize(maintenance_intent, baseline)
    statuses = [entry.status for entry in result.archive]
    assert statuses == ["baseline", "rejected", "evaluated"]
    assert "All transmitters are OFF" in result.archive[1].errors[0]


def test_stops_after_a_round_without_improvement(surrogate, baseline, maintenance_intent):
    first = _with(baseline, 0, power_dbm=46.0)

    class RepeatingGenerator:
        def propose_batch(self, request):
            return [RoleProposal(role=request.roles[0], configuration=first, rationale="same")]

    loop = OptimizationLoop(surrogate, RepeatingGenerator(), max_iterations=5, candidates_per_round=1)
    result = loop.optimize(maintenance_intent, baseline)
    assert result.rounds_run == 2 and len(result.iterations) == 1


def test_agno_generator_sends_archive_and_roles(surrogate, baseline, maintenance_intent):
    evaluator = ObjectiveEvaluator()
    prediction = surrogate.predict(baseline)
    archive = OptimizationArchive(maintenance_intent, baseline, prediction, evaluator.evaluate(maintenance_intent, prediction, prediction))
    sent: list[str] = []

    class FakeAgent:
        def run(self, text):
            sent.append(text)
            return SimpleNamespace(content=CandidateBatch(candidates=[RoleProposal(role="learn", configuration=baseline, rationale="x")]))

    generator = AgnoCandidateGenerator.__new__(AgnoCandidateGenerator)
    generator.agent = FakeAgent()
    request = ProposalRequest(
        intent=maintenance_intent,
        baseline=baseline,
        baseline_prediction=prediction,
        hard_constraints=HardConstraints(tx_states={2: False}),
        allowed_values=CandidateValidator(surrogate).allowed_values(),
        archive=archive,
        archive_view=archive.prompt_view(),
        round_index=1,
        max_rounds=3,
        roles=["learn", "combine"],
    )
    assert generator.propose_batch(request)[0].role == "learn"
    text = sent[0]
    assert "Round 2 of 3 (exploit)" in text and "learn, combine" in text and "- free" not in text
    assert '"tried_candidates_worst_to_best"' in text and '"choices": [' in text


def test_exhaustive_benchmark_finds_at_least_the_loop_best(surrogate, baseline, maintenance_intent):
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_optimizer.py"
    spec = importlib.util.spec_from_file_location("benchmark_optimizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    optimum = module.exhaustive_best(surrogate, maintenance_intent, baseline, chunk_size=50_000)
    assert optimum["configurations_searched"] == 19**3 - 1  # TX2 forced off, all-off excluded
    assert optimum["configuration"]["tx2"]["on"] is False
    run = module.run_variant(
        surrogate,
        HeuristicCandidateGenerator(),
        module.parse_variant("V2,3,4,full"),
        maintenance_intent,
        baseline,
    )
    assert run["score"] <= optimum["score"] + 1e-9
