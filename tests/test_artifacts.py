import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from evidence_route.artifacts import (
    BillingStateError,
    RequestFingerprintMismatch,
    SQLiteRunStore,
    TraceWriter,
    atomic_write_json,
)
from evidence_route.budget import BudgetExceeded, PriceConfig
from evidence_route.contracts import Usage


def make_store(tmp_path: Path, *, cap_cny: float = 10.0) -> SQLiteRunStore:
    pricing = PriceConfig(
        provider="fixture",
        currency="CNY",
        input_per_million=1.0,
        output_per_million=1.0,
        price_source="fixture",
        strict_evaluation=True,
    )
    return SQLiteRunStore(
        tmp_path / "run-store.sqlite3", activity_id="gate-a", cap_cny=cap_cny, pricing=pricing
    )


def reserve(store: SQLiteRunStore, call_id: str = "call-1", *, run_id: str = "run-1") -> None:
    store.reserve_call(
        call_id,
        request_sha256="a" * 64,
        run_id=run_id,
        node="router",
        task_id="root",
        logical_attempt=0,
        max_input_tokens=100,
        max_output_tokens=20,
    )


def complete(store: SQLiteRunStore, payload: dict[str, object] | None = None) -> None:
    store.mark_sent("call-1")
    usage = Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True)
    store.complete_call(
        "call-1",
        request_sha256="a" * 64,
        payload=payload or {"content": {"route": "single"}},
        usage=usage,
        usage_source="provider",
        requested_alias="alias",
        response_model_id_raw="relay-model",
        identity_verified=False,
    )


def test_run_store_keeps_first_completed_response(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    reserve(store)
    complete(store)
    usage = Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True)
    store.complete_call(
        "call-1",
        request_sha256="a" * 64,
        payload={"content": {"route": "multi"}},
        usage=usage,
        usage_source="provider",
        requested_alias="alias",
        response_model_id_raw="relay-model",
        identity_verified=False,
    )
    assert store.get_completed("call-1")["content"]["route"] == "single"
    summary = store.summarize_run("run-1")
    assert summary.usage.total_tokens == 10
    assert summary.actual_cost_micro_cny > 0
    assert summary.call_ids == ["call-1"]


def test_cache_rejects_same_call_id_with_different_request_sha(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    reserve(store)
    with pytest.raises(RequestFingerprintMismatch):
        store.reserve_call(
            "call-1",
            request_sha256="b" * 64,
            run_id="run-1",
            node="router",
            task_id="root",
            logical_attempt=0,
            max_input_tokens=100,
            max_output_tokens=20,
        )


def test_sent_without_completion_blocks_automatic_resume(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    reserve(store)
    store.mark_sent("call-1")
    reopened = make_store(tmp_path)
    with pytest.raises(BillingStateError, match="sent call has unknown billing"):
        reopened.resume_decision("call-1", request_sha256="a" * 64)


def test_explicit_billing_recovery_authorizes_one_retry_and_audits(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    reserve(store)
    store.mark_sent("call-1")
    store.mark_billing_uncertain("call-1")

    store.authorize_billing_uncertain_retry(
        "call-1",
        reason="provider dashboard shows the request was billed",
        evidence="2026-08-31 15:05:04; input=2279; output=266; cost=$0.003488",
    )

    reopened = make_store(tmp_path)
    assert reopened.resume_decision("call-1", request_sha256="a" * 64).action == "send"
    metadata = reopened.get_call_metadata("call-1")
    assert metadata is not None
    assert metadata["state"] == "reserved"
    event = reopened.get_billing_recovery_event("call-1")
    assert event is not None
    assert event["action"] == "authorized_retry"
    assert event["reason"] == "provider dashboard shows the request was billed"


def test_parallel_reservations_are_serialized_without_crossing_cap(tmp_path: Path) -> None:
    barrier = Barrier(2)

    def reserve_parallel(call_id: str) -> str:
        store = make_store(tmp_path, cap_cny=0.0001)
        barrier.wait()
        store.reserve_call(
            call_id,
            request_sha256=call_id[-1] * 64,
            run_id=call_id,
            node="worker",
            task_id="t0",
            logical_attempt=0,
            max_input_tokens=100,
            max_output_tokens=0,
        )
        return call_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reserve_parallel, call_id) for call_id in ("call-a", "call-b")]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except BudgetExceeded:
            outcomes.append("budget_exceeded")
    assert sorted(outcomes) in (["budget_exceeded", "call-a"], ["budget_exceeded", "call-b"])


def test_crash_after_cache_write_before_reconcile_counts_cost_once(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    reserve(store)
    complete(store, {"content": "first"})
    reopened = make_store(tmp_path)
    usage = Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True)
    reopened.complete_call(
        "call-1",
        request_sha256="a" * 64,
        payload={"content": "second"},
        usage=usage,
        usage_source="provider",
        requested_alias="alias",
        response_model_id_raw="relay-model",
        identity_verified=False,
    )
    summary = reopened.summarize_run("run-1")
    assert summary.actual_cost_micro_cny == reopened.pricing.estimate_micro_cny(usage)
    assert reopened.get_completed("call-1")["content"] == "first"


def test_missing_usage_response_is_terminal_and_never_reissued(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    reserve(store)
    store.mark_sent("call-1")
    store.complete_call(
        "call-1",
        request_sha256="a" * 64,
        payload={"content": "received"},
        usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False),
        usage_source="missing",
        requested_alias="alias",
        response_model_id_raw="relay-model",
        identity_verified=False,
    )
    reopened = make_store(tmp_path)
    decision = reopened.resume_decision("call-1", request_sha256="a" * 64)
    assert decision.action == "reuse_and_stop"
    assert decision.payload == {"content": "received"}
    summary = reopened.summarize_run("run-1")
    assert summary.actual_cost_micro_cny is None
    assert summary.known_actual_cost_micro_cny == 0
    assert summary.cost_is_lower_bound is True


@pytest.mark.parametrize(
    ("unresolved_state", "billing_uncertain"),
    [
        ("reserved", False),
        ("sent", True),
        ("billing_uncertain", True),
    ],
)
def test_mixed_completed_and_unresolved_calls_are_not_complete(
    tmp_path: Path,
    unresolved_state: str,
    billing_uncertain: bool,
) -> None:
    store = make_store(tmp_path)
    reserve(store)
    complete(store)
    store.reserve_call(
        "call-2",
        request_sha256="b" * 64,
        run_id="run-1",
        node="worker",
        task_id="t0",
        logical_attempt=0,
        max_input_tokens=100,
        max_output_tokens=20,
    )
    if unresolved_state in {"sent", "billing_uncertain"}:
        store.mark_sent("call-2")
    if unresolved_state == "billing_uncertain":
        store.mark_billing_uncertain("call-2")

    metadata = store.get_call_metadata("call-2")
    assert metadata is not None
    assert metadata["state"] == unresolved_state
    summary = store.summarize_run("run-1")

    assert summary.usage.complete is False
    assert summary.actual_cost_micro_cny is None
    assert summary.cost_is_lower_bound is True
    assert "missing" in summary.usage_sources
    assert summary.billing_uncertain is billing_uncertain


def test_usage_missing_reservation_stays_committed_during_parallel_reserve(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path, cap_cny=0.00015)
    reserve(store)
    store.mark_sent("call-1")
    store.complete_call(
        "call-1",
        request_sha256="a" * 64,
        payload={"content": "received"},
        usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False),
        usage_source="missing",
        requested_alias="alias",
        response_model_id_raw="relay-model",
        identity_verified=False,
    )
    with pytest.raises(BudgetExceeded):
        make_store(tmp_path, cap_cny=0.00015).reserve_call(
            "call-2",
            request_sha256="b" * 64,
            run_id="run-2",
            node="worker",
            task_id="t1",
            logical_attempt=0,
            max_input_tokens=100,
            max_output_tokens=0,
        )


def test_run_summary_propagates_usage_cost_identity_and_cache_hits(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    reserve(store)
    complete(store)
    store.reserve_call(
        "call-2",
        request_sha256="b" * 64,
        run_id="run-1",
        node="single",
        task_id="root",
        logical_attempt=0,
        max_input_tokens=100,
        max_output_tokens=20,
    )
    store.mark_sent("call-2")
    second_usage = Usage(input_tokens=3, output_tokens=1, total_tokens=4, complete=True)
    store.complete_call(
        "call-2",
        request_sha256="b" * 64,
        payload={"content": "second"},
        usage=second_usage,
        usage_source="provider",
        requested_alias="alias",
        response_model_id_raw="relay-model",
        identity_verified=False,
    )
    reopened = make_store(tmp_path)
    assert reopened.get_completed("call-1") is not None
    summary = reopened.summarize_run("run-1")
    assert summary.call_ids == ["call-1", "call-2"]
    assert summary.usage.total_tokens == 14
    assert summary.actual_cost_micro_cny == 14
    assert summary.fresh_call_count == 2
    assert summary.cache_hit_count == 1
    assert summary.requested_aliases == ["alias"]
    assert summary.response_model_ids_raw == ["relay-model"]
    assert summary.identity_verified is False


def test_trace_redacts_nested_secrets(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    TraceWriter(path).write({"api_key": "secret", "headers": {"Authorization": "Bearer secret"}})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["api_key"] == "[REDACTED]"
    assert payload["headers"]["Authorization"] == "[REDACTED]"


def test_atomic_json_replaces_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    atomic_write_json(path, {"status": "completed"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "completed"}
    assert not path.with_suffix(".json.tmp").exists()


def test_activity_summary_carries_cost_across_runs(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    reserve(store, "call-a", run_id="calibration-0")
    store.mark_sent("call-a")
    usage = Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True)
    store.complete_call(
        "call-a",
        request_sha256="a" * 64,
        payload={"content": "ok"},
        usage=usage,
        usage_source="provider",
        requested_alias="alias",
        response_model_id_raw="relay-a",
        identity_verified=False,
    )
    reopened = make_store(tmp_path)
    summary = reopened.summarize_activity()
    assert summary.actual_cost_micro_cny == 10
    assert summary.call_ids == ["call-a"]
