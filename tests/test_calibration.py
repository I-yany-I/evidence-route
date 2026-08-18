import pytest

from evidence_route.config import RoutingSettings, stable_hash
from evidence_route.evaluation.calibration import (
    CalibrationRuntimeCase,
    CalibrationScoredCase,
    CandidateOutcome,
    attach_calibration_gold,
    candidate_grid,
    choose_candidate,
    replay_candidate,
)

pytest_plugins = ["tests.fixtures.evaluation.factories"]


def test_runtime_case_rejects_gold_field(calibration_case: CalibrationScoredCase) -> None:
    payload = calibration_case.runtime.model_dump(mode="json")
    payload["gold_label"] = "Supported"
    with pytest.raises(ValueError):
        CalibrationRuntimeCase.model_validate(payload)


def test_replay_uses_saved_single_then_multi_on_escalation(
    calibration_case: CalibrationScoredCase,
) -> None:
    runtime = calibration_case.runtime.model_copy(
        update={
            "saved_llm_route": "single",
            "single_result": calibration_case.runtime.single_result.model_copy(
                update={"confidence": 0.50}
            ),
        }
    )
    calibration_case = calibration_case.model_copy(update={"runtime": runtime})
    outcome = replay_candidate(
        calibration_case,
        RoutingSettings(
            clear_multi_clauses=4,
            clear_single_min_sources=3,
            low_confidence=0.65,
            minimum_coverage=1.0,
        ),
    )
    assert outcome.final_result == calibration_case.runtime.multi_result
    assert outcome.executed_path == "single_escalated_multi"
    assert outcome.total_tokens == (
        calibration_case.runtime.router_usage.total_tokens
        + calibration_case.runtime.single_result.usage.total_tokens
        + calibration_case.runtime.multi_result.usage.total_tokens
    )


def test_candidate_prefers_token_saving_within_quality_floor() -> None:
    outcomes = [
        CandidateOutcome(
            config_hash="a" * 64,
            macro_f1=0.70,
            total_tokens=1000,
            llm_router_calls=16,
            llm_router_rate=0.5,
            simulated_cost_micro_cny=1_000_000,
            settings={},
        ),
        CandidateOutcome(
            config_hash="b" * 64,
            macro_f1=0.68,
            total_tokens=500,
            llm_router_calls=6,
            llm_router_rate=0.2,
            simulated_cost_micro_cny=500_000,
            settings={},
        ),
        CandidateOutcome(
            config_hash="c" * 64,
            macro_f1=0.66,
            total_tokens=100,
            llm_router_calls=0,
            llm_router_rate=0.0,
            simulated_cost_micro_cny=100_000,
            settings={},
        ),
    ]
    selected = choose_candidate(outcomes, tolerance=0.03)
    assert selected.config_hash == "b" * 64


def test_candidate_tie_breaks_by_router_rate_then_hash() -> None:
    outcomes = [
        CandidateOutcome(
            config_hash="b" * 64,
            macro_f1=0.70,
            total_tokens=500,
            llm_router_calls=6,
            llm_router_rate=0.2,
            simulated_cost_micro_cny=500_000,
            settings={},
        ),
        CandidateOutcome(
            config_hash="a" * 64,
            macro_f1=0.70,
            total_tokens=500,
            llm_router_calls=6,
            llm_router_rate=0.2,
            simulated_cost_micro_cny=500_000,
            settings={},
        ),
    ]
    assert choose_candidate(outcomes, tolerance=0.03).config_hash == "a" * 64


def test_candidate_grid_has_54_unique_configs() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 54
    assert len({stable_hash(item.model_dump(mode="json")) for item in candidates}) == 54


def test_attach_calibration_gold_requires_train_balance(
    calibration_case: CalibrationScoredCase,
) -> None:
    runtime_cases = []
    for i in range(32):
        claim_id = f"train-{i}"
        runtime_cases.append(
            calibration_case.runtime.model_copy(
                update={
                    "claim_id": claim_id,
                    "single_result": calibration_case.runtime.single_result.model_copy(
                        update={"claim_id": claim_id}
                    ),
                    "multi_result": calibration_case.runtime.multi_result.model_copy(
                        update={"claim_id": claim_id}
                    ),
                }
            )
        )
    aligned = [
        (runtime_case, calibration_case.gold_label)
        for runtime_case in runtime_cases
    ]
    with pytest.raises(ValueError, match="exactly eight"):
        attach_calibration_gold(runtime_cases, aligned)


def test_replay_does_not_mutate_saved_case(calibration_case: CalibrationScoredCase) -> None:
    before = calibration_case.runtime.model_dump(mode="json")
    replay_candidate(calibration_case, RoutingSettings())
    assert calibration_case.runtime.model_dump(mode="json") == before
