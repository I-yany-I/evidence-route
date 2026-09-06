from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence_route.artifacts import CallState, SQLiteRunStore
from evidence_route.budget import BudgetExceeded
from evidence_route.cli import ProductionServices
from evidence_route.config import stable_hash
from evidence_route.contracts import ResultStatus, Usage, Verdict, VerificationResult
from evidence_route.evaluation import production_calibration as production_calibration_module
from evidence_route.evaluation.activity import (
    ActivityRecord,
    CallBounds,
    CampaignPlan,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    LatencyBreakdown,
    RunArtifact,
    WorkStatus,
)
from evidence_route.evaluation.calibration import (
    CalibrationItemStatus,
    load_calibration_state,
    runtime_case_fingerprint,
    state_fingerprint,
    write_canonical_json,
)
from evidence_route.evaluation.lifecycle import load_activity, persist_activity, sha256_file
from evidence_route.evaluation.production_calibration import (
    _requeue_authorized_calibration_recovery,
)
from evidence_route.evaluation.production_evaluation import (
    _billing_recovery_items,
    _evaluation_activity_is_closed,
    _validate_selected_routing_policy,
)
from evidence_route.evaluation.runner import CampaignProcessInterruption
from evidence_route.execution import build_run_artifact, load_price_config
from evidence_route.llm import BillingUncertain, RawCompletion


@pytest.mark.parametrize("experiment_mode", [False, True])
def test_billing_recovery_activity_is_resumable(experiment_mode: bool) -> None:
    activity = SimpleNamespace(
        calibration_status=CampaignStatus.COMPLETE,
        billing_uncertain=True,
        stop_reason=CampaignStopReason.BILLING_UNCERTAIN,
    )

    assert _evaluation_activity_is_closed(
        activity,
        mode="resume",
        experiment_mode=experiment_mode,
    )


def test_billing_recovery_includes_running_item_after_process_kill() -> None:
    running = SimpleNamespace(status=WorkStatus.RUNNING, stop_reason=None)
    pending = SimpleNamespace(status=WorkStatus.PENDING, stop_reason=None)
    stopped = SimpleNamespace(
        status=WorkStatus.STOPPED,
        stop_reason=CampaignStopReason.BILLING_UNCERTAIN,
    )

    assert _billing_recovery_items(SimpleNamespace(items=[running, pending, stopped])) == [
        running,
        stopped,
    ]


def test_authorized_calibration_recovery_requeues_stopped_case() -> None:
    work = SimpleNamespace(
        case_id="case-1",
        router_run_id="router-1",
        single_run_id="single-1",
        multi_run_id="multi-1",
    )
    item = SimpleNamespace(
        case_id="case-1",
        status=CalibrationItemStatus.STOPPED,
        errors=["BILLING_UNCERTAIN"],
    )

    class Store:
        def unresolved_call_states(self, run_id: str) -> dict[str, CallState]:
            if run_id == "single-1":
                return {"call-1": CallState.RESERVED}
            return {}

        def get_billing_recovery_event(self, call_id: str) -> dict[str, str] | None:
            return {"call_id": call_id, "action": "authorized_retry"}

    recovered = _requeue_authorized_calibration_recovery(
        SimpleNamespace(items=[work]), SimpleNamespace(items=[item]), Store()
    )

    assert recovered == ["case-1"]
    assert item.status is CalibrationItemStatus.PENDING
    assert item.errors == []


def _write_runtime_inputs(tmp_path: Path, *, cap_cny: float = 350.0) -> dict[str, Path]:
    manifest_dir = tmp_path / "manifests"
    corpus_dir = tmp_path / "corpora"
    manifest_dir.mkdir()
    corpus_dir.mkdir()
    items = []
    for index in range(32):
        claim_id = f"train-{index}"
        claim = f"Alice and Bob and Carol agree in report {index}."
        text = f'{{"evidence_id":"av:{claim_id}:0","title":"Source","source_url":"https://example.org/source","text":"{claim}","snapshot_sha256":"PLACEHOLDER"}}\n'
        snapshot = hashlib.sha256(claim.encode()).hexdigest()
        text = text.replace("PLACEHOLDER", snapshot)
        payload = text.encode()
        (corpus_dir / f"{claim_id}.jsonl").write_bytes(payload)
        items.append(
            {
                "claim_id": claim_id,
                "original_id": index,
                "claim": claim,
                "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
                "split": "train",
                "corpus_relpath": f"{claim_id}.jsonl",
                "corpus_sha256": hashlib.sha256(payload).hexdigest(),
                "corpus_bytes": len(payload),
                "corpus_records": 1,
            }
        )
    manifest = manifest_dir / "runtime.json"
    raw = json.dumps(
        {
            "schema_version": "1",
            "dataset": "AVeriTeC",
            "revision": "a" * 40,
            "source_metadata_sha256": "b" * 64,
            "seed": 20260817,
            "items": items,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest.write_bytes(raw)
    manifest.with_suffix(".json.sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="ascii"
    )
    (corpus_dir.parent / "preparation_receipt.json").write_text(
        '{"prepared":true}\n', encoding="utf-8"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "llm:\n  temperature: 0.0\n"
        "routing:\n  low_confidence: 0.0\n  minimum_coverage: 0.0\n"
        f"budget:\n  estimated_cost_cap_cny: {cap_cny}\n",
        encoding="utf-8",
    )
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "provider: fake\ncurrency: CNY\ninput_per_million: 1\n"
        "output_per_million: 1\nprice_source: fixture\nstrict_evaluation: true\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "corpus": corpus_dir,
        "config": config,
        "pricing": pricing,
    }


def test_experiment_routing_policy_is_checked_against_parent_config(tmp_path: Path) -> None:
    report = tmp_path / "calibration-report.json"
    parent_routing = {"clear_multi_clauses": 3, "clear_single_min_sources": 1}
    report.write_text(
        json.dumps({"selected": {"config_hash": stable_hash(parent_routing)}}),
        encoding="utf-8",
    )
    parent_config = SimpleNamespace(
        routing=SimpleNamespace(model_dump=lambda mode="json": parent_routing)
    )
    experiment_config = SimpleNamespace(
        routing=SimpleNamespace(
            model_dump=lambda mode="json": {
                "clear_multi_clauses": 3,
                "clear_single_min_sources": 2,
            }
        )
    )

    _validate_selected_routing_policy(report, experiment_config, parent_config)


class _Transport:
    def __init__(self, activity_dir: Path) -> None:
        self.activity_dir = activity_dir
        self.calls = 0

    async def create(self, **request: object) -> RawCompletion:
        self.calls += 1
        if self.calls == 1:
            assert (self.activity_dir / "calibration-plan.json").is_file()
            assert (self.activity_dir / "calibration-state.json").is_file()
            assert (self.activity_dir / "activity.json").is_file()
        schema = request["schema"]
        name = getattr(schema, "__name__", "")
        if name == "RouterPayload":
            payload = {
                "route": "single",
                "reason_codes": ["fixture"],
                "explanation": "fixture route",
            }
        elif name == "DecompositionDraft":
            payload = {
                "tasks": [
                    {"task_id": "t0", "claim_unit_ids": ["u0"], "query": "verify"},
                    {"task_id": "t1", "claim_unit_ids": ["u1"], "query": "verify"},
                    {"task_id": "t2", "claim_unit_ids": ["u2"], "query": "verify"},
                ]
            }
        elif name == "WorkerDraft":
            payload = {"verdict": "Supported", "confidence": 0.9, "citations": []}
        else:
            payload = {
                "verdict": "Supported",
                "confidence": 0.9,
                "rationale": "fixture",
                "citations": [],
            }
        return RawCompletion(
            content=json.dumps(payload),
            response_model_id_raw="fixture-model",
            input_tokens=2,
            output_tokens=1,
        )


def test_collect_freezes_before_first_transport_and_closes_all_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path)
    activity_dir = tmp_path / "activity"
    transport = _Transport(activity_dir)
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")

    payload = ProductionServices(transport_factory=lambda settings: transport).calibrate_collect(
        runtime_manifest=paths["manifest"],
        config_path=paths["config"],
        pricing_path=paths["pricing"],
        corpus_dir=paths["corpus"],
        activity_dir=activity_dir,
        checkpoint_db=tmp_path / "checkpoints.sqlite3",
        run_store=tmp_path / "run-store.sqlite3",
        activity_id="activity",
        output_config=tmp_path / "ignored-config.yaml",
        output_report=tmp_path / "ignored-report.json",
        resume=False,
    )

    assert payload["status"] == "complete"
    assert payload["completed_cases"] == 32
    rows = (
        (activity_dir / "calibration" / "runtime-cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(rows) == 32
    assert len({json.loads(row)["case_id"] for row in rows}) == 32
    assert transport.calls == 32 * 7
    plan = json.loads((activity_dir / "calibration-plan.json").read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert plan["manifest_freeze_git_sha"] == head
    assert (
        plan["runtime_manifest_sha256"]
        == hashlib.sha256(paths["manifest"].read_bytes()).hexdigest()
    )
    activity = json.loads((activity_dir / "activity.json").read_text(encoding="utf-8"))
    assert activity["calibration_status"] == "complete"
    assert (
        activity["calibration_plan_sha256"]
        == hashlib.sha256((activity_dir / "calibration-plan.json").read_bytes()).hexdigest()
    )
    assert (
        activity["calibration_state_sha256"]
        == hashlib.sha256((activity_dir / "calibration-state.json").read_bytes()).hexdigest()
    )
    assert len(activity["calibration_artifacts"]) == 32
    assert len({item["case_id"] for item in activity["calibration_artifacts"]}) == 32


def test_collect_pauses_after_case_limit_and_resumes_completed_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path)
    activity_dir = tmp_path / "activity"
    transport = _Transport(activity_dir)
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")
    kwargs = _collect_kwargs(paths, tmp_path, activity_dir)

    paused = ProductionServices(transport_factory=lambda settings: transport).calibrate_collect(
        **kwargs, max_cases=1
    )

    assert paused["status"] == CampaignStatus.PAUSED.value
    assert paused["completed_cases"] == 1
    assert paused["paused"] is True
    assert transport.calls == 7

    resumed = ProductionServices(transport_factory=lambda settings: transport).calibrate_collect(
        **{**kwargs, "resume": True}, max_cases=40
    )

    assert resumed["status"] == "complete"
    assert resumed["completed_cases"] == 32
    assert transport.calls == 32 * 7


def test_collect_rejects_non_positive_case_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path)
    kwargs = _collect_kwargs(paths, tmp_path, tmp_path / "activity")
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")

    with pytest.raises(ValueError, match="max_cases"):
        ProductionServices(
            transport_factory=lambda settings: _Transport(kwargs["activity_dir"])
        ).calibrate_collect(
            **kwargs, max_cases=0
        )


def test_collect_seals_request_fingerprint_for_every_saved_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path)
    activity_dir = tmp_path / "activity"
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")
    ProductionServices(
        transport_factory=lambda settings: _Transport(activity_dir)
    ).calibrate_collect(**_collect_kwargs(paths, tmp_path, activity_dir))

    state = load_calibration_state(activity_dir / "calibration-state.json")
    case = json.loads((activity_dir / state.items[0].artifact_relpath).read_text(encoding="utf-8"))
    assert set(case["request_sha256_by_call_id"]) == set(case["call_ids"])
    assert all(
        len(request_sha256) == 64 for request_sha256 in case["request_sha256_by_call_id"].values()
    )


def test_collect_rejects_startup_budget_before_transport_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path, cap_cny=0.000001)
    activity_dir = tmp_path / "activity"
    factory_calls = 0

    def transport_factory(_settings: object) -> object:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("transport must not be constructed over the startup cap")

    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")

    with pytest.raises(BudgetExceeded, match="startup"):
        ProductionServices(transport_factory=transport_factory).calibrate_collect(
            **_collect_kwargs(paths, tmp_path, activity_dir)
        )

    assert factory_calls == 0


def test_collect_persists_internal_error_as_terminal_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path)
    activity_dir = tmp_path / "activity"
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")
    monkeypatch.setattr(
        "evidence_route.evaluation.production_calibration.build_calibration_runtime_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("internal collector bug")),
    )

    with pytest.raises(RuntimeError, match="internal collector bug"):
        ProductionServices(
            transport_factory=lambda settings: _Transport(activity_dir)
        ).calibrate_collect(**_collect_kwargs(paths, tmp_path, activity_dir))

    state = load_calibration_state(activity_dir / "calibration-state.json")
    assert state.items[0].status is CalibrationItemStatus.STOPPED
    activity = json.loads((activity_dir / "activity.json").read_text(encoding="utf-8"))
    assert activity["calibration_status"] == CampaignStatus.FAILED.value
    assert activity["status"] == CampaignStatus.FAILED.value
    assert activity["stop_reason"] == CampaignStopReason.INTERNAL_ERROR.value


def _collect_kwargs(
    paths: dict[str, Path], tmp_path: Path, activity_dir: Path
) -> dict[str, object]:
    return {
        "runtime_manifest": paths["manifest"],
        "config_path": paths["config"],
        "pricing_path": paths["pricing"],
        "corpus_dir": paths["corpus"],
        "activity_dir": activity_dir,
        "checkpoint_db": tmp_path / "checkpoints.sqlite3",
        "run_store": tmp_path / "run-store.sqlite3",
        "activity_id": "activity",
        "output_config": tmp_path / "ignored-config.yaml",
        "output_report": tmp_path / "ignored-report.json",
        "resume": False,
    }


def test_calibration_probe_uses_versioned_provider_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path)
    activity_dir = tmp_path / "activity"
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")
    model_root = tmp_path / "retrieval-model"
    model_root.mkdir()
    model_receipt = tmp_path / "retrieval-model-receipt.json"
    model_receipt.write_text('{"model_id":"fixture-model"}\n', encoding="utf-8")
    model_receipt_sha256 = hashlib.sha256(model_receipt.read_bytes()).hexdigest()
    monkeypatch.setenv("EVIDENCE_ROUTE_RETRIEVAL_MODEL_ROOT", str(model_root))
    monkeypatch.setenv("EVIDENCE_ROUTE_RETRIEVAL_MODEL_RECEIPT", str(model_receipt))
    with paths["config"].open("a", encoding="utf-8") as handle:
        handle.write(
            "evidence:\n"
            "  retrieval_mode: source_hybrid_v2\n"
            "  source_candidate_k: 2\n"
            "  passages_per_source: 2\n"
            "  dense_candidate_k: 4\n"
            "  final_per_source: 1\n"
            "  dense_model_id: BAAI/bge-small-en-v1.5\n"
            "  dense_model_revision: 52398278842ec682c6f32300af41344b1c0b0bb2\n"
            f"  dense_model_receipt_sha256: {model_receipt_sha256}\n"
        )

    def factory(corpus_dir: Path, settings: object) -> object:
        del corpus_dir, settings
        raise RuntimeError("calibration provider factory called")

    monkeypatch.setattr(
        production_calibration_module,
        "build_evidence_provider",
        factory,
    )
    with pytest.raises(RuntimeError, match="calibration provider factory called"):
        ProductionServices(transport_factory=lambda settings: object()).calibrate_collect(
            **_collect_kwargs(paths, tmp_path, activity_dir)
        )
    plan = json.loads((activity_dir / "calibration-plan.json").read_text(encoding="utf-8"))
    assert plan["retrieval_model_receipt_sha256"] == model_receipt_sha256


def _write_gold_manifest(runtime_path: Path, output_path: Path) -> Path:
    """Create a scorer-only manifest bound to the supplied runtime bytes."""

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_digest = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    labels = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
    items = []
    for index, item in enumerate(runtime["items"]):
        items.append(
            {
                "claim_id": item["claim_id"],
                "original_id": item["original_id"],
                "claim": item["claim"],
                "label": labels[index % 4],
                "questions": [],
                "justification": "fixture",
                "claim_types": [],
            }
        )
    payload = {
        "schema_version": "1",
        "dataset": "AVeriTeC",
        "revision": runtime["revision"],
        "source_metadata_sha256": runtime["source_metadata_sha256"],
        "runtime_manifest_sha256": runtime_digest,
        "seed": runtime["seed"],
        "items": items,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    output_path.write_bytes(raw)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="ascii"
    )
    return output_path


def _collect_and_replay_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Path], Path, _Transport]:
    paths = _write_runtime_inputs(tmp_path)
    activity_dir = tmp_path / "activity"
    transport = _Transport(activity_dir)
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")
    ProductionServices(transport_factory=lambda settings: transport).calibrate_collect(
        **_collect_kwargs(paths, tmp_path, activity_dir)
    )
    gold = _write_gold_manifest(paths["manifest"], tmp_path / "gold.json")
    return paths | {"gold": gold}, activity_dir, transport


def test_calibrate_replay_never_constructs_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, collected_transport = _collect_and_replay_inputs(tmp_path, monkeypatch)
    output_config = tmp_path / "calibrated.yaml"
    output_report = tmp_path / "calibration-report.json"

    def forbidden_transport(_settings: object) -> object:
        raise AssertionError("replay must not construct a transport")

    payload = ProductionServices(transport_factory=forbidden_transport).calibrate_replay(
        runtime_manifest=paths["manifest"],
        gold_manifest=paths["gold"],
        config_path=paths["config"],
        pricing_path=paths["pricing"],
        corpus_dir=paths["corpus"],
        activity_dir=activity_dir,
        checkpoint_db=tmp_path / "checkpoints.sqlite3",
        run_store=tmp_path / "run-store.sqlite3",
        activity_id="activity",
        output_config=output_config,
        output_report=output_report,
    )

    assert payload["candidate_count"] == 54
    assert collected_transport.calls == 32 * 7
    metadata = json.loads((activity_dir / "calibration-replay.json").read_text(encoding="utf-8"))
    assert (
        metadata["calibrated_config_sha256"]
        == hashlib.sha256(output_config.read_bytes()).hexdigest()
    )
    assert (
        metadata["calibration_report_sha256"]
        == hashlib.sha256(output_report.read_bytes()).hexdigest()
    )


def test_calibrate_replay_rejects_wrong_activity_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="activity ID"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            activity_id="wrong-activity",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_calibrate_replay_rejects_changed_nonrouting_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    changed = tmp_path / "changed-config.yaml"
    changed.write_text(
        paths["config"].read_text(encoding="utf-8") + "\n# changed after collection\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config"):
        ProductionServices(
            transport_factory=lambda settings: (_ for _ in ()).throw(
                AssertionError("replay must not construct a transport")
            )
        ).calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=changed,
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            activity_id="activity",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_fresh_duplicate_is_rejected_without_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path)
    activity_dir = tmp_path / "activity"
    first_transport = _Transport(activity_dir)
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")
    kwargs = _collect_kwargs(paths, tmp_path, activity_dir)
    ProductionServices(transport_factory=lambda settings: first_transport).calibrate_collect(
        **kwargs
    )

    duplicate_transport = _Transport(activity_dir)
    with pytest.raises(FileExistsError, match="already|contains a persisted plan"):
        ProductionServices(
            transport_factory=lambda settings: duplicate_transport
        ).calibrate_collect(**kwargs)
    assert duplicate_transport.calls == 0


def test_calibration_resume_requires_existing_run_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path)
    activity_dir = tmp_path / "activity"
    transport = _Transport(activity_dir)
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")
    kwargs = _collect_kwargs(paths, tmp_path, activity_dir)
    ProductionServices(transport_factory=lambda settings: transport).calibrate_collect(**kwargs)
    (tmp_path / "run-store.sqlite3").unlink()

    with pytest.raises(ValueError, match="run store"):
        ProductionServices(
            transport_factory=lambda settings: (_ for _ in ()).throw(
                AssertionError("resume must fail before transport")
            )
        ).calibrate_collect(**{**kwargs, "resume": True})
    assert transport.calls == 32 * 7


def test_calibration_resume_after_call_completion_before_case_close_reuses_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path)
    activity_dir = tmp_path / "activity"
    transport = _Transport(activity_dir)
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")
    kwargs = _collect_kwargs(paths, tmp_path, activity_dir)
    module = __import__(
        "evidence_route.evaluation.production_calibration",
        fromlist=["persist_calibration_case"],
    )
    original_persist = module.persist_calibration_case
    interrupted = True

    def fail_once(*args: object, **values: object) -> str:
        nonlocal interrupted
        if interrupted:
            interrupted = False
            raise KeyboardInterrupt("simulated close interruption")
        return original_persist(*args, **values)

    monkeypatch.setattr(module, "persist_calibration_case", fail_once)
    with pytest.raises(KeyboardInterrupt, match="simulated close interruption"):
        ProductionServices(transport_factory=lambda settings: transport).calibrate_collect(**kwargs)
    assert transport.calls == 7
    interrupted_state = load_calibration_state(activity_dir / "calibration-state.json")
    assert interrupted_state.items[0].status is CalibrationItemStatus.RUNNING

    kwargs["resume"] = True
    payload = ProductionServices(transport_factory=lambda settings: transport).calibrate_collect(
        **kwargs
    )
    assert payload["completed_cases"] == 32
    assert transport.calls == 32 * 7
    cases = list((activity_dir / "calibration" / "cases").glob("*.json"))
    assert len(cases) == 32


def test_resume_reconciles_case_written_before_activity_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path)
    activity_dir = tmp_path / "activity"
    transport = _Transport(activity_dir)
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")
    kwargs = _collect_kwargs(paths, tmp_path, activity_dir)
    module = __import__(
        "evidence_route.evaluation.production_calibration", fromlist=["persist_activity"]
    )
    original_persist = module.persist_activity
    writes = 0

    def interrupt_after_first_case(*args: object, **values: object):
        nonlocal writes
        writes += 1
        if writes == 3:
            raise KeyboardInterrupt("simulated process kill")
        return original_persist(*args, **values)

    monkeypatch.setattr(module, "persist_activity", interrupt_after_first_case)
    with pytest.raises(KeyboardInterrupt, match="simulated process kill"):
        ProductionServices(transport_factory=lambda settings: transport).calibrate_collect(**kwargs)
    assert transport.calls == 7

    kwargs["resume"] = True
    monkeypatch.setattr(module, "persist_activity", original_persist)
    payload = ProductionServices(transport_factory=lambda settings: transport).calibrate_collect(
        **kwargs
    )
    assert payload["completed_cases"] == 32
    assert transport.calls == 32 * 7


def test_replay_rejects_reserialized_runtime_manifest_bound_to_updated_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_runtime_inputs(tmp_path)
    activity_dir = tmp_path / "activity"
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")
    ProductionServices(
        transport_factory=lambda settings: _Transport(activity_dir)
    ).calibrate_collect(**_collect_kwargs(paths, tmp_path, activity_dir))

    manifest_payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    paths["manifest"].write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    manifest_digest = hashlib.sha256(paths["manifest"].read_bytes()).hexdigest()
    paths["manifest"].with_suffix(paths["manifest"].suffix + ".sha256").write_text(
        manifest_digest + "\n", encoding="ascii"
    )
    gold_manifest = _write_gold_manifest(paths["manifest"], tmp_path / "gold-reserialized.json")

    factory_calls = 0

    def transport_factory(_settings: object) -> object:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("replay must not construct a transport")

    replay_kwargs = {
        "runtime_manifest": paths["manifest"],
        "gold_manifest": gold_manifest,
        "config_path": paths["config"],
        "pricing_path": paths["pricing"],
        "corpus_dir": paths["corpus"],
        "activity_dir": activity_dir,
        "checkpoint_db": tmp_path / "checkpoints.sqlite3",
        "output_config": tmp_path / "calibrated.yaml",
        "output_report": tmp_path / "calibration-report.json",
        "run_store": tmp_path / "run-store.sqlite3",
    }
    with pytest.raises(ValueError, match="runtime manifest digest differs"):
        ProductionServices(transport_factory=transport_factory).calibrate_replay(**replay_kwargs)
    assert factory_calls == 0
    assert not replay_kwargs["output_config"].exists()
    assert not replay_kwargs["output_report"].exists()


def test_replay_rejects_activity_bound_to_stale_state_file_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    state_path = activity_dir / "calibration-state.json"
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    state_path.write_text(json.dumps(state_payload, indent=2) + "\n", encoding="utf-8")
    output_config = tmp_path / "calibrated.yaml"
    output_report = tmp_path / "calibration-report.json"

    with pytest.raises(ValueError, match="activity calibration state hash differs"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=output_config,
            output_report=output_report,
        )
    assert not output_config.exists()
    assert not output_report.exists()


def test_replay_rejects_resealed_case_and_state_with_stale_activity_artifact_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    state_path = activity_dir / "calibration-state.json"
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    case_path = activity_dir / state_payload["items"][0]["artifact_relpath"]
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case_payload["router_actual_cost_micro_cny"] += 1
    case_payload["artifact_sha256"] = runtime_case_fingerprint(case_payload)
    write_canonical_json(case_path, case_payload)
    state_payload["items"][0]["artifact_sha256"] = case_payload["artifact_sha256"]
    state_payload["state_sha256"] = state_fingerprint(state_payload)
    write_canonical_json(state_path, state_payload)
    activity_path = activity_dir / "activity.json"
    activity = load_activity(activity_path).model_copy(
        update={"calibration_state_sha256": sha256_file(state_path)}, deep=True
    )
    persist_activity(activity_path, activity)

    with pytest.raises(ValueError, match="activity calibration artifact hashes differ"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_replay_rejects_config_and_report_path_collision_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    collision = tmp_path / "replay-output"

    with pytest.raises(ValueError, match="replay output paths must be distinct"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=collision,
            output_report=collision,
        )
    assert not collision.exists()
    assert not (activity_dir / "calibration-replay.json").exists()


def test_replay_rejects_output_collision_with_replay_metadata_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    output_config = tmp_path / "calibrated.yaml"
    metadata_path = activity_dir / "calibration-replay.json"

    with pytest.raises(ValueError, match="replay output paths must be distinct"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=output_config,
            output_report=metadata_path,
        )
    assert not output_config.exists()
    assert not metadata_path.exists()


def test_replay_rejects_output_path_collision_with_calibration_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    state_path = activity_dir / "calibration-state.json"
    original_state = state_path.read_bytes()
    output_report = tmp_path / "calibration-report.json"

    with pytest.raises(ValueError, match="replay output path collides with input"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=state_path,
            output_report=output_report,
        )
    assert state_path.read_bytes() == original_state
    assert not output_report.exists()
    assert not (activity_dir / "calibration-replay.json").exists()


def test_replay_rejects_output_path_collision_with_calibration_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    state = load_calibration_state(activity_dir / "calibration-state.json")
    case_path = activity_dir / state.items[0].artifact_relpath
    original_case = case_path.read_bytes()
    output_config = tmp_path / "calibrated.yaml"

    with pytest.raises(ValueError, match="replay output path collides with input"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=output_config,
            output_report=case_path,
        )
    assert case_path.read_bytes() == original_case
    assert not output_config.exists()
    assert not (activity_dir / "calibration-replay.json").exists()


def test_replay_rejects_output_config_collision_with_checkpoint_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    checkpoint_db = tmp_path / "checkpoints.sqlite3"
    original_checkpoint = checkpoint_db.read_bytes()
    output_report = tmp_path / "calibration-report.json"

    with pytest.raises(ValueError, match="replay output path collides with input"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=checkpoint_db,
            run_store=tmp_path / "run-store.sqlite3",
            output_config=checkpoint_db,
            output_report=output_report,
        )
    assert checkpoint_db.read_bytes() == original_checkpoint
    assert not output_report.exists()
    assert not (activity_dir / "calibration-replay.json").exists()


def test_replay_rejects_output_report_collision_with_checkpoint_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    checkpoint_db = tmp_path / "checkpoints.sqlite3"
    original_checkpoint = checkpoint_db.read_bytes()
    output_config = tmp_path / "calibrated.yaml"

    with pytest.raises(ValueError, match="replay output path collides with input"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=checkpoint_db,
            run_store=tmp_path / "run-store.sqlite3",
            output_config=output_config,
            output_report=checkpoint_db,
        )
    assert checkpoint_db.read_bytes() == original_checkpoint
    assert not output_config.exists()
    assert not (activity_dir / "calibration-replay.json").exists()


def test_replay_rejects_missing_run_store_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    missing_store = tmp_path / "missing-run-store.sqlite3"
    output_config = tmp_path / "calibrated.yaml"
    output_report = tmp_path / "calibration-report.json"

    with pytest.raises(ValueError, match="run store input is missing"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=missing_store,
            output_config=output_config,
            output_report=output_report,
        )
    assert not missing_store.exists()
    assert not output_config.exists()
    assert not output_report.exists()


def test_replay_rejects_empty_replacement_run_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    replacement_store = tmp_path / "replacement-run-store.sqlite3"
    SQLiteRunStore(
        replacement_store,
        activity_id="activity",
        cap_cny=350.0,
        pricing=load_price_config(paths["pricing"]),
    )

    with pytest.raises(ValueError, match="calibration run store call IDs differ"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=replacement_store,
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_replay_rejects_invalid_existing_run_store_without_repairing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    invalid_store = tmp_path / "invalid-run-store.sqlite3"
    invalid_store.write_bytes(b"not a sqlite database")
    original_store = invalid_store.read_bytes()

    with pytest.raises(ValueError, match="run store is not a valid existing ledger"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=invalid_store,
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )
    assert invalid_store.read_bytes() == original_store


def test_replay_rejects_run_store_request_fingerprint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    state = load_calibration_state(activity_dir / "calibration-state.json")
    case = json.loads((activity_dir / state.items[0].artifact_relpath).read_text(encoding="utf-8"))
    call_id = case["call_ids"][0]
    with sqlite3.connect(tmp_path / "run-store.sqlite3") as connection:
        connection.execute(
            "UPDATE calls SET request_sha256 = ? WHERE call_id = ?",
            ("f" * 64, call_id),
        )

    with pytest.raises(ValueError, match="request fingerprint differs"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_replay_rejects_changed_pricing_bytes_before_using_run_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    changed_pricing = tmp_path / "changed-pricing.yaml"
    changed_pricing.write_text(
        paths["pricing"].read_text(encoding="utf-8") + "\n# changed bytes\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pricing digest differs"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=changed_pricing,
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_replay_rejects_activity_not_closed_for_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    activity_path = activity_dir / "activity.json"
    activity = load_activity(activity_path).model_copy(
        update={"calibration_status": CampaignStatus.RUNNING, "status": CampaignStatus.RUNNING},
        deep=True,
    )
    persist_activity(activity_path, activity)

    with pytest.raises(ValueError, match="calibration activity is not closed"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_replay_rejects_calibration_call_alias_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    state = load_calibration_state(activity_dir / "calibration-state.json")
    case = json.loads((activity_dir / state.items[0].artifact_relpath).read_text(encoding="utf-8"))
    with sqlite3.connect(tmp_path / "run-store.sqlite3") as connection:
        connection.execute(
            "UPDATE calls SET requested_alias = 'tampered-alias' WHERE call_id = ?",
            (case["call_ids"][0],),
        )

    with pytest.raises(ValueError, match="calibration call alias differs"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_replay_rejects_activity_model_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    activity_path = activity_dir / "activity.json"
    activity = load_activity(activity_path).model_copy(
        update={"observed_response_model_ids_raw": ["tampered-model"]}, deep=True
    )
    persist_activity(activity_path, activity)

    with pytest.raises(ValueError, match="activity model identity differs"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_replay_collection_accounting_excludes_unrelated_later_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    with sqlite3.connect(tmp_path / "run-store.sqlite3") as connection:
        connection.execute(
            """
            INSERT INTO calls (
                call_id, request_sha256, activity_id, run_id, node, task_id, logical_attempt,
                state, reserved_micro_cny, actual_micro_cny, payload_json, usage_json,
                usage_source, requested_alias, response_model_id_raw, identity_verified,
                transport_attempts, cache_hits, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', 1, 1, ?, ?, 'provider', ?, ?, 0, 1, 0, ?, ?)
            """,
            (
                "later-call",
                "a" * 64,
                "activity",
                "later-dev-run",
                "single",
                "later",
                0,
                json.dumps({"content": "{}"}),
                json.dumps(
                    {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1, "complete": True}
                ),
                "fixture-model",
                "fixture-model",
                "2020-01-02T00:00:00+00:00",
                "2020-01-02T00:00:00+00:00",
            ),
        )

    payload = ProductionServices().calibrate_replay(
        runtime_manifest=paths["manifest"],
        gold_manifest=paths["gold"],
        config_path=paths["config"],
        pricing_path=paths["pricing"],
        corpus_dir=paths["corpus"],
        activity_dir=activity_dir,
        checkpoint_db=tmp_path / "checkpoints.sqlite3",
        run_store=tmp_path / "run-store.sqlite3",
        output_config=tmp_path / "calibrated.yaml",
        output_report=tmp_path / "calibration-report.json",
    )
    report = json.loads(Path(payload["calibration_report"]).read_text(encoding="utf-8"))
    assert "later-call" not in report["collection_accounting"]["call_ids"]


def test_replay_rejects_noncompleted_calibration_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    state = load_calibration_state(activity_dir / "calibration-state.json")
    case = json.loads((activity_dir / state.items[0].artifact_relpath).read_text(encoding="utf-8"))
    with sqlite3.connect(tmp_path / "run-store.sqlite3") as connection:
        connection.execute(
            "UPDATE calls SET state = 'sent' WHERE call_id = ?",
            (case["call_ids"][0],),
        )

    with pytest.raises(ValueError, match="calibration run store call is not completed"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_replay_rejects_calibration_call_slot_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    state = load_calibration_state(activity_dir / "calibration-state.json")
    case = json.loads((activity_dir / state.items[0].artifact_relpath).read_text(encoding="utf-8"))
    with sqlite3.connect(tmp_path / "run-store.sqlite3") as connection:
        connection.execute(
            "UPDATE calls SET node = 'tampered-node' WHERE call_id = ?",
            (case["call_ids"][0],),
        )

    with pytest.raises(ValueError, match="calibration call slot identity differs"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_replay_rejects_calibration_run_usage_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    state = load_calibration_state(activity_dir / "calibration-state.json")
    case = json.loads((activity_dir / state.items[0].artifact_relpath).read_text(encoding="utf-8"))
    tampered_usage = {
        "input_tokens": 99,
        "output_tokens": 1,
        "total_tokens": 100,
        "complete": True,
    }
    with sqlite3.connect(tmp_path / "run-store.sqlite3") as connection:
        connection.execute(
            "UPDATE calls SET usage_json = ? WHERE call_id = ?",
            (json.dumps(tampered_usage), case["call_ids"][0]),
        )

    with pytest.raises(ValueError, match="calibration run store usage differs"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_replay_rejects_calibration_run_cost_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    state = load_calibration_state(activity_dir / "calibration-state.json")
    case = json.loads((activity_dir / state.items[0].artifact_relpath).read_text(encoding="utf-8"))
    with sqlite3.connect(tmp_path / "run-store.sqlite3") as connection:
        connection.execute(
            "UPDATE calls SET actual_micro_cny = actual_micro_cny + 1 WHERE call_id = ?",
            (case["call_ids"][0],),
        )

    with pytest.raises(ValueError, match="calibration run store cost differs"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def test_replay_rejects_extra_call_in_calibration_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, activity_dir, _ = _collect_and_replay_inputs(tmp_path, monkeypatch)
    state = load_calibration_state(activity_dir / "calibration-state.json")
    case = json.loads((activity_dir / state.items[0].artifact_relpath).read_text(encoding="utf-8"))
    extra_call_id = "extra-calibration-call"
    with sqlite3.connect(tmp_path / "run-store.sqlite3") as connection:
        connection.execute(
            """
            INSERT INTO calls (
                call_id, request_sha256, activity_id, run_id, node, task_id, logical_attempt,
                state, reserved_micro_cny, actual_micro_cny, payload_json, usage_json,
                usage_source, requested_alias, response_model_id_raw, identity_verified,
                transport_attempts, cache_hits, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', 1, 1, ?, ?, 'provider', ?, ?, 0, 1, 0, ?, ?)
            """,
            (
                extra_call_id,
                "a" * 64,
                "activity",
                case["router_run_id"],
                "router",
                "extra",
                0,
                json.dumps({"content": "{}"}),
                json.dumps(
                    {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1, "complete": True}
                ),
                "fixture-model",
                "fixture-model",
                "2020-01-01T00:00:00+00:00",
                "2020-01-01T00:00:00+00:00",
            ),
        )

    with pytest.raises(ValueError, match="calibration run store call IDs differ"):
        ProductionServices().calibrate_replay(
            runtime_manifest=paths["manifest"],
            gold_manifest=paths["gold"],
            config_path=paths["config"],
            pricing_path=paths["pricing"],
            corpus_dir=paths["corpus"],
            activity_dir=activity_dir,
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            run_store=tmp_path / "run-store.sqlite3",
            output_config=tmp_path / "calibrated.yaml",
            output_report=tmp_path / "calibration-report.json",
        )


def _write_evaluation_manifests(paths: dict[str, Path]) -> tuple[Path, Path]:
    items: list[dict[str, object]] = []
    for index in range(80):
        claim_id = f"dev-{index}"
        claim = f"Evaluation claim {index}."
        snapshot = hashlib.sha256(claim.encode()).hexdigest()
        payload = (
            '{"evidence_id":"av:'
            + claim_id
            + ':0","title":"Source","source_url":"https://example.org/source",'
            + '"text":"'
            + claim
            + '","snapshot_sha256":"'
            + snapshot
            + '"}\n'
        ).encode()
        (paths["corpus"] / f"{claim_id}.jsonl").write_bytes(payload)
        items.append(
            {
                "claim_id": claim_id,
                "original_id": index,
                "claim": claim,
                "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
                "split": "dev",
                "corpus_relpath": f"{claim_id}.jsonl",
                "corpus_sha256": hashlib.sha256(payload).hexdigest(),
                "corpus_bytes": len(payload),
                "corpus_records": 1,
            }
        )

    def write_manifest(path: Path, selected: list[dict[str, object]]) -> Path:
        raw = json.dumps(
            {
                "schema_version": "1",
                "dataset": "AVeriTeC",
                "revision": "a" * 40,
                "source_metadata_sha256": "b" * 64,
                "seed": 20260817,
                "items": selected,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        path.write_bytes(raw)
        path.with_suffix(path.suffix + ".sha256").write_text(
            hashlib.sha256(raw).hexdigest() + "\n", encoding="ascii"
        )
        return path

    manifest_dir = paths["manifest"].parent
    return (
        write_manifest(manifest_dir / "dev.json", items),
        write_manifest(manifest_dir / "stability.json", items[:20]),
    )


def _prepare_evaluation_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cap_cny: float = 350.0
) -> dict[str, Path]:
    paths = _write_runtime_inputs(tmp_path, cap_cny=cap_cny)
    activity_dir = tmp_path / "activity"
    transport = _Transport(activity_dir)
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "fixture-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "fixture-model")
    ProductionServices(transport_factory=lambda settings: transport).calibrate_collect(
        **_collect_kwargs(paths, tmp_path, activity_dir)
    )
    gold = _write_gold_manifest(paths["manifest"], tmp_path / "gold.json")
    calibrated_config = tmp_path / "calibrated.yaml"
    calibration_report = tmp_path / "calibration-report.json"
    ProductionServices().calibrate_replay(
        runtime_manifest=paths["manifest"],
        gold_manifest=gold,
        config_path=paths["config"],
        pricing_path=paths["pricing"],
        corpus_dir=paths["corpus"],
        activity_dir=activity_dir,
        checkpoint_db=tmp_path / "checkpoints.sqlite3",
        run_store=tmp_path / "run-store.sqlite3",
        activity_id="activity",
        output_config=calibrated_config,
        output_report=calibration_report,
    )
    dev, stability = _write_evaluation_manifests(paths)
    return paths | {
        "activity": activity_dir,
        "config": calibrated_config,
        "calibration_report": calibration_report,
        "dev": dev,
        "stability": stability,
        "run_store": tmp_path / "run-store.sqlite3",
        "checkpoint": tmp_path / "checkpoints.sqlite3",
    }


def _evaluation_kwargs(paths: dict[str, Path], *, mode: str) -> dict[str, object]:
    return {
        "manifest": paths["dev"],
        "stability_manifest": paths["stability"],
        "config_path": paths["config"],
        "pricing_path": paths["pricing"],
        "corpus_dir": paths["corpus"],
        "activity_dir": paths["activity"],
        "checkpoint_db": paths["checkpoint"],
        "run_store": paths["run_store"],
        "activity_id": "activity",
        "campaign_id": "campaign",
        "calibration_report": paths["calibration_report"],
        "mode": mode,
    }


class _ZeroCallCampaignExecutor:
    def __init__(self, **kwargs: object) -> None:
        self.activity_dir = Path(kwargs["trace_dir"]).parent

    async def __call__(self, work: object) -> RunArtifact:
        assert (self.activity_dir / "plan.json").is_file()
        assert (self.activity_dir / "campaign.json").is_file()
        assert (self.activity_dir / "activity.json").is_file()
        result = VerificationResult(
            claim_id=work.claim_id,
            status=ResultStatus.FAILED,
            rationale="fixture retrieval failure",
            initial_route=None,
            failure_stage="pre_route",
            usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=True),
            errors=["PROBE_RETRIEVAL_FAILED"],
        )
        return build_run_artifact(
            activity_id="activity",
            campaign_id="campaign",
            phase=work.phase,
            run_id=work.run_id,
            claim_id=work.claim_id,
            strategy=work.strategy,
            repeat=work.repeat,
            result=result,
            summary={
                "call_ids": [],
                "usage": result.usage.model_dump(mode="json"),
                "actual_cost_micro_cny": 0,
                "known_actual_cost_micro_cny": 0,
                "committed_cost_micro_cny": 0,
                "cost_is_lower_bound": False,
                "fresh_call_count": 0,
                "cache_hit_count": 0,
                "requested_aliases": [],
                "response_model_ids_raw": [],
                "usage_sources": [],
                "billing_uncertain": False,
            },
            requested_alias="fixture-model",
            price_config_id="f" * 64,
            latency=LatencyBreakdown(
                fresh_end_to_end_ms=0,
                model_active_ms=0,
                retry_ms=0,
                queue_ms=0,
                checkpoint_downtime_ms=0,
                total_elapsed_ms=0,
                interruption_count=0,
            ),
        )


def test_evaluate_start_after_calibration_creates_campaign_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.verify_current_freeze",
        lambda expected, **kwargs: expected,
    )
    payload = ProductionServices(
        transport_factory=lambda settings: object(),
        executor_factory=_ZeroCallCampaignExecutor,
    ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))

    assert payload["status"] == CampaignStatus.COMPLETE.value
    plan = CampaignPlan.model_validate_json((paths["activity"] / "plan.json").read_text())
    assert len([item for item in plan.schedule if item.phase == "dev"]) == 240
    assert len([item for item in plan.schedule if item.phase == "stability"]) == 40
    activity = load_activity(paths["activity"] / "activity.json")
    assert activity.status is CampaignStatus.COMPLETE
    assert activity.dev_status is CampaignStatus.COMPLETE
    assert activity.stability_status is CampaignStatus.COMPLETE
    assert activity.summary is not None
    assert len(activity.summary.call_ids) == 32 * 7

    with pytest.raises(FileExistsError, match="already|contains a persisted plan"):
        ProductionServices(
            transport_factory=lambda settings: (_ for _ in ()).throw(
                AssertionError("duplicate must fail before transport")
            ),
            executor_factory=_ZeroCallCampaignExecutor,
        ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))


def test_evaluate_startup_cap_rejection_happens_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.verify_current_freeze",
        lambda expected, **kwargs: expected,
    )
    calls = 0

    original_bounds = __import__(
        "evidence_route.evaluation.production_evaluation",
        fromlist=["estimate_call_bounds"],
    ).estimate_call_bounds

    def over_cap(*args: object, **kwargs: object) -> CallBounds:
        bounds = original_bounds(*args, **kwargs)
        return bounds.model_copy(update={"startup_required_micro_cny": 10**18})

    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.estimate_call_bounds", over_cap
    )

    def transport_factory(settings: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    with pytest.raises(BudgetExceeded, match="startup"):
        ProductionServices(
            transport_factory=transport_factory,
            executor_factory=_ZeroCallCampaignExecutor,
        ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))
    assert calls == 0


def test_evaluate_resume_freeze_mismatch_happens_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.verify_current_freeze",
        lambda expected, **kwargs: expected,
    )
    ProductionServices(
        transport_factory=lambda settings: object(),
        executor_factory=_ZeroCallCampaignExecutor,
    ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))

    calls = 0

    def changed_freeze(expected: object, **kwargs: object) -> object:
        raise ValueError("config freeze mismatch")

    def transport_factory(settings: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.verify_current_freeze", changed_freeze
    )
    with pytest.raises(ValueError, match="freeze mismatch"):
        ProductionServices(
            transport_factory=transport_factory,
            executor_factory=_ZeroCallCampaignExecutor,
        ).evaluate(**_evaluation_kwargs(paths, mode="resume"))
    assert calls == 0


def test_evaluate_rejects_changed_calibration_report_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    paths["calibration_report"].write_text("{}\n", encoding="utf-8")
    calls = 0

    def transport_factory(settings: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    with pytest.raises(ValueError, match="calibration report differs"):
        ProductionServices(
            transport_factory=transport_factory,
            executor_factory=_ZeroCallCampaignExecutor,
        ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))
    assert calls == 0


def test_evaluate_rejects_empty_replacement_ledger_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    replacement = tmp_path / "empty-evaluate-store.sqlite3"
    SQLiteRunStore(
        replacement,
        activity_id="activity",
        cap_cny=350.0,
        pricing=load_price_config(paths["pricing"]),
    )
    paths["run_store"] = replacement
    calls = 0

    def transport_factory(settings: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    with pytest.raises(ValueError, match="call IDs differ|calibration ledger"):
        ProductionServices(
            transport_factory=transport_factory,
            executor_factory=_ZeroCallCampaignExecutor,
        ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))
    assert calls == 0


def test_evaluate_rejects_missing_calibration_case_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    state = load_calibration_state(paths["activity"] / "calibration-state.json")
    (paths["activity"] / state.items[0].artifact_relpath).unlink()
    calls = 0

    def transport_factory(settings: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    with pytest.raises(ValueError, match="artifact is missing"):
        ProductionServices(
            transport_factory=transport_factory,
            executor_factory=_ZeroCallCampaignExecutor,
        ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))
    assert calls == 0


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("node = 'tampered-node'", "calibration call slot identity differs"),
        (
            "actual_micro_cny = actual_micro_cny + 1",
            "calibration call cost differs from saved usage",
        ),
        (
            "response_model_id_raw = 'tampered-model'",
            "calibration call model identity differs",
        ),
        ("identity_verified = 1", "calibration call model identity differs"),
    ],
)
def test_evaluate_reaudits_calibration_call_metadata_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_error: str,
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    state = load_calibration_state(paths["activity"] / "calibration-state.json")
    case = json.loads(
        (paths["activity"] / state.items[0].artifact_relpath).read_text(encoding="utf-8")
    )
    with sqlite3.connect(paths["run_store"]) as connection:
        connection.execute(
            f"UPDATE calls SET {mutation} WHERE call_id = ?",
            (case["call_ids"][0],),
        )
    transport_calls = 0

    def transport_factory(settings: object) -> object:
        nonlocal transport_calls
        transport_calls += 1
        return object()

    with pytest.raises(ValueError, match=expected_error):
        ProductionServices(
            transport_factory=transport_factory,
            executor_factory=_ZeroCallCampaignExecutor,
        ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))

    assert transport_calls == 0


class _InterruptingCampaignExecutor:
    def __init__(self, **kwargs: object) -> None:
        pass

    async def __call__(self, work: object) -> RunArtifact:
        raise CampaignProcessInterruption("fixture interruption")


class _SentThenInterruptingCampaignExecutor:
    def __init__(self, **kwargs: object) -> None:
        self.run_store = kwargs["run_store"]

    async def __call__(self, work: object) -> RunArtifact:
        call_id = hashlib.sha256(f"{work.run_id}\0fixture-interrupted".encode()).hexdigest()
        self.run_store.reserve_call(
            call_id,
            request_sha256="e" * 64,
            run_id=work.run_id,
            node="single",
            task_id="root",
            logical_attempt=0,
            max_input_tokens=1,
            max_output_tokens=1,
        )
        self.run_store.mark_sent(call_id)
        raise CampaignProcessInterruption("fixture interruption after send")


class _BillingUncertainCampaignExecutor:
    def __init__(self, **kwargs: object) -> None:
        self.run_store = kwargs["run_store"]

    async def __call__(self, work: object) -> RunArtifact:
        call_id = hashlib.sha256(f"{work.run_id}\0fixture-uncertain".encode()).hexdigest()
        self.run_store.reserve_call(
            call_id,
            request_sha256="e" * 64,
            run_id=work.run_id,
            node="single",
            task_id="root",
            logical_attempt=0,
            max_input_tokens=1,
            max_output_tokens=1,
        )
        self.run_store.mark_sent(call_id)
        self.run_store.mark_billing_uncertain(call_id)
        raise BillingUncertain("fixture billing uncertainty")


class _FailingCampaignExecutor:
    def __init__(self, **kwargs: object) -> None:
        pass

    async def __call__(self, work: object) -> RunArtifact:
        raise RuntimeError("fixture executor failure")


class _DriftingCampaignExecutor:
    def __init__(self, **kwargs: object) -> None:
        self.run_store = kwargs["run_store"]
        self.requested_alias = kwargs["app_config"].llm.requested_alias
        self.price_config_id = kwargs["pricing"].config_id

    async def __call__(self, work: object) -> RunArtifact:
        call_id = hashlib.sha256(f"{work.run_id}\0fixture".encode()).hexdigest()
        request_sha = "d" * 64
        self.run_store.reserve_call(
            call_id,
            request_sha256=request_sha,
            run_id=work.run_id,
            node="single",
            task_id="root",
            logical_attempt=0,
            max_input_tokens=1,
            max_output_tokens=1,
        )
        self.run_store.mark_sent(call_id)
        usage = Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True)
        self.run_store.complete_call(
            call_id,
            request_sha256=request_sha,
            payload={"content": "ok"},
            usage=usage,
            usage_source="provider",
            requested_alias=self.requested_alias,
            response_model_id_raw="fixture-model-drifted",
            identity_verified=False,
        )
        result = VerificationResult(
            claim_id=work.claim_id,
            status=ResultStatus.COMPLETED,
            verdict=Verdict.SUPPORTED,
            confidence=0.9,
            rationale="fixture",
            initial_route=("multi" if work.strategy.value == "always_multi" else "single"),
            usage=usage,
        )
        return build_run_artifact(
            activity_id="activity",
            campaign_id="campaign",
            phase=work.phase,
            run_id=work.run_id,
            claim_id=work.claim_id,
            strategy=work.strategy,
            repeat=work.repeat,
            result=result,
            summary=self.run_store.summarize_run(work.run_id),
            requested_alias=self.requested_alias,
            price_config_id=self.price_config_id,
            latency=LatencyBreakdown(
                fresh_end_to_end_ms=1,
                model_active_ms=1,
                retry_ms=0,
                queue_ms=0,
                checkpoint_downtime_ms=0,
                total_elapsed_ms=1,
                interruption_count=0,
            ),
        )


@pytest.mark.parametrize(
    ("executor", "error", "status"),
    [
        (_InterruptingCampaignExecutor, CampaignProcessInterruption, CampaignStatus.INTERRUPTED),
        (_FailingCampaignExecutor, RuntimeError, CampaignStatus.FAILED),
    ],
)
def test_evaluate_synchronizes_activity_when_executor_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executor: object,
    error: type[BaseException],
    status: CampaignStatus,
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.verify_current_freeze",
        lambda expected, **kwargs: expected,
    )

    with pytest.raises(error):
        ProductionServices(
            transport_factory=lambda settings: object(), executor_factory=executor
        ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))

    activity = load_activity(paths["activity"] / "activity.json")
    assert activity.status is status
    assert activity.campaign_state_sha256 == sha256_file(paths["activity"] / "campaign.json")
    assert activity.summary is not None
    assert len(activity.summary.call_ids) == 32 * 7


def test_evaluate_billing_uncertainty_precedes_process_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.verify_current_freeze",
        lambda expected, **kwargs: expected,
    )

    with pytest.raises(CampaignProcessInterruption, match="after send"):
        ProductionServices(
            transport_factory=lambda settings: object(),
            executor_factory=_SentThenInterruptingCampaignExecutor,
        ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))

    activity = load_activity(paths["activity"] / "activity.json")
    assert activity.status is CampaignStatus.INCOMPLETE_COST_UNCERTAIN
    assert activity.stop_reason is CampaignStopReason.BILLING_UNCERTAIN
    assert activity.billing_uncertain is True


def test_direct_billing_recovery_allows_descendant_protocol_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.verify_current_freeze",
        lambda expected, **kwargs: expected,
    )
    service = ProductionServices(
        transport_factory=lambda settings: object(),
        executor_factory=_BillingUncertainCampaignExecutor,
    )
    first = service.evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))
    assert first["status"] == CampaignStatus.INCOMPLETE_COST_UNCERTAIN.value

    state = CampaignState.model_validate_json(
        (paths["activity"] / "campaign.json").read_text(encoding="utf-8")
    )
    stopped = next(item for item in state.items if item.status is WorkStatus.STOPPED)
    store = SQLiteRunStore(
        paths["run_store"],
        activity_id="activity",
        cap_cny=350.0,
        pricing=load_price_config(paths["pricing"]),
    )
    unresolved = store.unresolved_call_states(stopped.run_id)
    assert len(unresolved) == 1
    call_id = next(iter(unresolved))
    store.authorize_billing_uncertain_retry(
        call_id,
        reason="fixture recovery authorization",
        evidence="explicit test authorization",
    )

    descendant_flags: list[bool] = []

    def verify_descendant(expected: object, **kwargs: object) -> object:
        descendant_flags.append(bool(kwargs.get("allow_descendant_git")))
        return expected

    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.verify_current_freeze",
        verify_descendant,
    )
    resumed = service.evaluate(**_evaluation_kwargs(paths, mode="resume"))

    assert resumed["status"] == CampaignStatus.INCOMPLETE_COST_UNCERTAIN.value
    assert descendant_flags == [True]


def test_evaluate_resume_accepts_clean_process_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.verify_current_freeze",
        lambda expected, **kwargs: expected,
    )

    with pytest.raises(CampaignProcessInterruption, match="fixture interruption"):
        ProductionServices(
            transport_factory=lambda settings: object(),
            executor_factory=_InterruptingCampaignExecutor,
        ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))

    interrupted = load_activity(paths["activity"] / "activity.json")
    assert interrupted.stop_reason is CampaignStopReason.PROCESS_INTERRUPTION
    assert interrupted.status is CampaignStatus.INTERRUPTED

    resumed = ProductionServices(
        transport_factory=lambda settings: object(),
        executor_factory=_ZeroCallCampaignExecutor,
    ).evaluate(**_evaluation_kwargs(paths, mode="resume"))

    assert resumed["status"] == CampaignStatus.COMPLETE.value


def test_evaluate_resume_rejects_process_interruption_without_interrupted_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.verify_current_freeze",
        lambda expected, **kwargs: expected,
    )

    with pytest.raises(CampaignProcessInterruption, match="fixture interruption"):
        ProductionServices(
            transport_factory=lambda settings: object(),
            executor_factory=_InterruptingCampaignExecutor,
        ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))

    activity_path = paths["activity"] / "activity.json"
    interrupted = load_activity(activity_path)
    malformed = interrupted.model_copy(
        update={"dev_status": CampaignStatus.RUNNING},
        deep=True,
    )
    persist_activity(activity_path, ActivityRecord.model_validate(malformed.model_dump()))
    executor_calls: list[bool] = []

    def executor_factory(**kwargs: object) -> _ZeroCallCampaignExecutor:
        executor_calls.append(True)
        return _ZeroCallCampaignExecutor(**kwargs)

    with pytest.raises(ValueError, match="interrupted phase"):
        ProductionServices(
            transport_factory=lambda settings: object(),
            executor_factory=executor_factory,
        ).evaluate(**_evaluation_kwargs(paths, mode="resume"))

    assert executor_calls == []


def test_evaluate_stops_on_cross_phase_model_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_evaluation_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "evidence_route.evaluation.production_evaluation.verify_current_freeze",
        lambda expected, **kwargs: expected,
    )

    payload = ProductionServices(
        transport_factory=lambda settings: object(),
        executor_factory=_DriftingCampaignExecutor,
    ).evaluate(**_evaluation_kwargs(paths, mode="start_after_calibration"))

    assert payload["status"] == CampaignStatus.INCOMPLETE_MODEL_DRIFT.value
    activity = load_activity(paths["activity"] / "activity.json")
    assert activity.status is CampaignStatus.INCOMPLETE_MODEL_DRIFT
    assert activity.stop_reason is CampaignStopReason.MODEL_DRIFT
