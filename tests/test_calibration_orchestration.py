import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evidence_route.contracts import Verdict
from evidence_route.evaluation.calibration import (
    CalibrationItemStatus,
    CalibrationPlan,
    CalibrationReplay,
    begin_calibration_case,
    build_calibration_plan,
    build_calibration_replay,
    build_calibration_runtime_case,
    build_calibration_state,
    derive_case_id,
    derive_run_id,
    load_calibration_cases,
    load_calibration_plan,
    load_calibration_state,
    persist_calibration_case,
    persist_calibration_plan,
    persist_calibration_state,
    plan_fingerprint,
    reconcile_calibration_state,
    runtime_case_fingerprint,
    state_fingerprint,
    verify_plan_fingerprint,
    verify_state_fingerprint,
    write_calibration_outputs,
    write_calibration_replay_view,
    write_canonical_json,
)

pytest_plugins = ["tests.fixtures.evaluation.factories"]


SHA = "a" * 64


def _plan():
    claims = [
        SimpleNamespace(claim_id=f"train-{index}", split="train")
        for index in range(32)
    ]
    return build_calibration_plan(
        claims,
        activity_id="gate-a",
        manifest_freeze_git_sha="1" * 40,
        runtime_manifest_sha256=SHA,
        corpus_preparation_receipt_sha256="b" * 64,
        prompt_bundle_sha256="c" * 64,
        config_sha256="d" * 64,
        pricing_sha256="e" * 64,
        endpoint_config_sha256="f" * 64,
        requirements_lock_sha256="0" * 64,
        requested_alias="relay-model",
        seed=20260817,
        cap_micro_cny=350_000_000,
    )


def test_build_plan_and_state_freeze_deterministic_case_order() -> None:
    plan = _plan()
    state = build_calibration_state(plan)

    verify_plan_fingerprint(plan)
    verify_state_fingerprint(state)
    assert [item.order for item in plan.items] == list(range(32))
    for item, state_item in zip(plan.items, state.items, strict=True):
        assert item.case_id == derive_case_id(plan.activity_id, item.claim_id)
        assert item.router_run_id == derive_run_id(item.case_id, "router")
        assert item.single_run_id == derive_run_id(item.case_id, "single")
        assert item.multi_run_id == derive_run_id(item.case_id, "multi")
        assert state_item.case_id == item.case_id
        assert state_item.artifact_relpath == f"calibration/cases/{item.case_id}.json"

    assert build_calibration_state(_plan()) == state


def test_plan_rejects_rehashed_nondeterministic_case_id() -> None:
    payload = _plan().model_dump(mode="json")
    payload["items"][0]["case_id"] = "9" * 64
    payload["plan_fingerprint"] = plan_fingerprint(payload)

    with pytest.raises(ValueError, match="deterministic"):
        CalibrationPlan.model_validate(payload)


def _runtime_cases(calibration_case, plan):
    cases = []
    for item in plan.items:
        single = calibration_case.runtime.single_result.model_copy(
            update={"claim_id": item.claim_id}
        )
        multi = calibration_case.runtime.multi_result.model_copy(
            update={"claim_id": item.claim_id}
        )
        payload = calibration_case.runtime.model_dump(mode="json")
        payload.update(
            {
                "claim_id": item.claim_id,
                "case_id": item.case_id,
                "router_run_id": item.router_run_id,
                "single_run_id": item.single_run_id,
                "multi_run_id": item.multi_run_id,
                "runtime_manifest_sha256": plan.runtime_manifest_sha256,
                "requested_alias": plan.requested_alias,
                "call_ids": [f"{item.case_id}-{index}" for index in range(7)],
                "single_result": single.model_dump(mode="json"),
                "multi_result": multi.model_dump(mode="json"),
                "artifact_sha256": "0" * 64,
            }
        )
        payload["artifact_sha256"] = runtime_case_fingerprint(payload)
        cases.append(type(calibration_case.runtime).model_validate(payload))
    return cases


def test_persisted_plan_state_and_cases_round_trip_with_verified_hashes(
    tmp_path, calibration_case
) -> None:
    plan = _plan()
    state = build_calibration_state(plan)
    plan_path = tmp_path / "calibration" / "plan.json"
    state_path = tmp_path / "calibration-state.json"
    persist_calibration_plan(plan_path, plan, fresh=True)
    persist_calibration_state(state_path, state)

    cases = _runtime_cases(calibration_case, plan)
    for index, case in enumerate(cases):
        persist_calibration_case(tmp_path, plan, state, case)
        assert state.items[index].status is CalibrationItemStatus.COMPLETE
    persist_calibration_state(state_path, state)

    assert load_calibration_plan(plan_path) == plan
    loaded_state = load_calibration_state(state_path, expected_plan=plan)
    assert load_calibration_cases(tmp_path, plan, loaded_state) == cases
    assert len({item.artifact_sha256 for item in loaded_state.items}) == 32


def test_runtime_case_builder_seals_plan_identity(calibration_case) -> None:
    plan = _plan()
    work = plan.items[0]

    case = build_calibration_runtime_case(
        plan=plan,
        work=work,
        call_ids=[f"call-{index}" for index in range(7)],
        features=calibration_case.runtime.features,
        saved_llm_route="single",
        router_usage=calibration_case.runtime.router_usage,
        router_actual_cost_micro_cny=12,
        single_result=calibration_case.runtime.single_result,
        multi_result=calibration_case.runtime.multi_result,
        response_model_ids_raw=["relay-model"] * 7,
    )

    assert case.case_id == work.case_id
    assert case.requested_alias == plan.requested_alias
    assert case.artifact_sha256 == runtime_case_fingerprint(case)


def test_loading_cases_rejects_tampered_file(tmp_path, calibration_case) -> None:
    plan = _plan()
    state = build_calibration_state(plan)
    case = _runtime_cases(calibration_case, plan)[0]
    persist_calibration_case(tmp_path, plan, state, case)
    path = tmp_path / state.items[0].artifact_relpath
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["saved_llm_route"] = "multi"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_calibration_cases(tmp_path, plan, state, require_complete=False)


def test_resume_reconciles_closed_case_before_state_update(tmp_path, calibration_case) -> None:
    plan = _plan()
    state = build_calibration_state(plan)
    case = _runtime_cases(calibration_case, plan)[0]
    state.items[0].status = CalibrationItemStatus.RUNNING
    state.state_sha256 = state_fingerprint(state)
    path = tmp_path / state.items[0].artifact_relpath
    write_canonical_json(path, case)

    reconciled = reconcile_calibration_state(tmp_path, plan, state)

    assert reconciled.items[0].status is CalibrationItemStatus.COMPLETE
    assert reconciled.items[0].artifact_sha256 == case.artifact_sha256
    assert all(
        item.status is CalibrationItemStatus.PENDING for item in reconciled.items[1:]
    )
    verify_state_fingerprint(reconciled)


def test_begin_case_persists_single_running_item_and_allows_interrupted_resume(
    tmp_path,
) -> None:
    plan = _plan()
    state = build_calibration_state(plan)
    state_path = tmp_path / "calibration-state.json"

    begin_calibration_case(plan, state, plan.items[0].case_id, state_path=state_path)
    with pytest.raises(ValueError, match="already RUNNING"):
        begin_calibration_case(plan, state, plan.items[1].case_id)
    state.items[0].status = CalibrationItemStatus.INTERRUPTED
    state.state_sha256 = state_fingerprint(state)
    begin_calibration_case(plan, state, plan.items[0].case_id, state_path=state_path)

    loaded = load_calibration_state(state_path, expected_plan=plan)
    assert loaded.items[0].status is CalibrationItemStatus.RUNNING
    assert sum(item.status is CalibrationItemStatus.RUNNING for item in loaded.items) == 1


def test_replay_view_is_regenerated_once_in_plan_order(tmp_path, calibration_case) -> None:
    plan = _plan()
    cases = _runtime_cases(calibration_case, plan)
    path = tmp_path / "calibration" / "runtime_cases.jsonl"

    write_calibration_replay_view(path, cases, plan=plan)
    write_calibration_replay_view(path, cases, plan=plan)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 32
    assert [row["claim_id"] for row in rows] == [item.claim_id for item in plan.items]


def test_replay_writes_all_candidates_and_changes_only_routing_config(
    tmp_path, calibration_case
) -> None:
    plan = _plan()
    runtime_cases = _runtime_cases(calibration_case, plan)
    scored = [
        SimpleNamespace(runtime=case, gold_label=list(Verdict)[index % 4])
        for index, case in enumerate(runtime_cases)
    ]
    replay = build_calibration_replay(
        scored,
        plan=plan,
        code_git_sha="2" * 40,
        collection_accounting={"actual_cost_micro_cny": 123_000},
    )

    source = Path("configs/default.yaml")
    output_config = tmp_path / "calibrated.yaml"
    output_report = tmp_path / "calibration_report.json"
    written = write_calibration_outputs(
        source_config=source,
        output_config=output_config,
        output_report=output_report,
        replay=replay,
    )

    before = yaml.safe_load(source.read_text(encoding="utf-8"))
    after = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    assert {key: value for key, value in after.items() if key != "routing"} == {
        key: value for key, value in before.items() if key != "routing"
    }
    assert after["routing"] == written.selected.settings
    report = json.loads(output_report.read_text(encoding="utf-8"))
    assert len(report["candidates"]) == 54
    assert set(report["baselines"]) == {"always_single", "always_multi"}
    assert report["selected"]["config_hash"] == written.selected.config_hash
    assert report["scope"] == "train_calibration_only"
    assert report["calibrated_config_sha256"]


def test_fixed_single_baseline_never_uses_adaptive_escalation(calibration_case) -> None:
    plan = _plan()
    runtime_cases = _runtime_cases(calibration_case, plan)
    first = runtime_cases[0].model_dump(mode="json")
    first["single_result"]["confidence"] = 0.1
    first["artifact_sha256"] = "0" * 64
    first["artifact_sha256"] = runtime_case_fingerprint(first)
    runtime_cases[0] = type(runtime_cases[0]).model_validate(first)
    scored = [
        SimpleNamespace(runtime=case, gold_label=list(Verdict)[index % 4])
        for index, case in enumerate(runtime_cases)
    ]

    replay = build_calibration_replay(
        scored,
        plan=plan,
        code_git_sha="2" * 40,
    )

    single = replay.baselines["always_single"].decisions[0]
    assert single.executed_path == "single_failed"
    assert not single.escalated


def test_fixed_baselines_use_selected_validation_thresholds(calibration_case) -> None:
    plan = _plan()
    runtime_cases = _runtime_cases(calibration_case, plan)
    adjusted = []
    for case in runtime_cases:
        payload = case.model_dump(mode="json")
        payload["single_result"]["confidence"] = 0.6
        payload["artifact_sha256"] = "0" * 64
        payload["artifact_sha256"] = runtime_case_fingerprint(payload)
        adjusted.append(type(case).model_validate(payload))
    scored = [
        SimpleNamespace(runtime=case, gold_label=list(Verdict)[index % 4])
        for index, case in enumerate(adjusted)
    ]

    replay = build_calibration_replay(scored, plan=plan, code_git_sha="2" * 40)

    assert replay.selected.low_confidence == 0.55
    assert replay.baselines["always_single"].decisions[0].executed_path == "single"


def test_replay_requires_balanced_train_gold(calibration_case) -> None:
    plan = _plan()
    scored = [
        SimpleNamespace(runtime=case, gold_label=Verdict.SUPPORTED)
        for case in _runtime_cases(calibration_case, plan)
    ]

    with pytest.raises(ValueError, match="exactly eight"):
        build_calibration_replay(scored, plan=plan, code_git_sha="2" * 40)


def test_report_rejects_non_preregistered_selected_candidate(calibration_case) -> None:
    plan = _plan()
    scored = [
        SimpleNamespace(runtime=case, gold_label=list(Verdict)[index % 4])
        for index, case in enumerate(_runtime_cases(calibration_case, plan))
    ]
    replay = build_calibration_replay(scored, plan=plan, code_git_sha="2" * 40)
    payload = replay.model_dump(mode="json")
    payload["selected"] = next(
        candidate
        for candidate in payload["candidates"]
        if candidate["config_hash"] != replay.selected.config_hash
    )

    with pytest.raises(ValueError, match="pre-registered tie-break"):
        CalibrationReplay.model_validate(payload)
