from pathlib import Path

import pytest

from evidence_route.contracts import ResultStatus, Strategy, Usage, Verdict, VerificationResult
from evidence_route.evaluation.activity import LatencyBreakdown, artifact_fingerprint
from evidence_route.execution import build_run_artifact, load_price_config


def test_price_loader_requires_strict_integer_billing_config(tmp_path: Path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "provider: relay\ncurrency: CNY\ninput_per_million: 1\n"
        "output_per_million: 2\nprice_source: fixture\nstrict_evaluation: true\n",
        encoding="utf-8",
    )
    pricing = load_price_config(path)
    assert pricing.strict_evaluation is True
    assert pricing.currency == "CNY"
    with pytest.raises(ValueError, match="strict"):
        load_price_config(tmp_path / "missing.yaml")


def test_price_loader_rejects_non_cny_strict_config(tmp_path: Path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "provider: relay\ncurrency: USD\ninput_per_million: 1\n"
        "output_per_million: 2\nprice_source: fixture\nstrict_evaluation: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CNY"):
        load_price_config(path)


def test_build_run_artifact_binds_graph_result_to_store_summary() -> None:
    result = VerificationResult(
        claim_id="dev-0",
        status=ResultStatus.COMPLETED,
        verdict=Verdict.SUPPORTED,
        confidence=0.9,
        rationale="fixture",
        initial_route="single",
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
    )
    summary = {
        "call_ids": ["call-0"],
        "usage": Usage(input_tokens=10, output_tokens=4, total_tokens=14, complete=True),
        "actual_cost_micro_cny": 22,
        "known_actual_cost_micro_cny": 22,
        "committed_cost_micro_cny": 22,
        "cost_is_lower_bound": False,
        "fresh_call_count": 1,
        "cache_hit_count": 2,
        "requested_aliases": ["alias"],
        "response_model_ids_raw": ["relay-id"],
        "billing_uncertain": False,
    }
    artifact = build_run_artifact(
        activity_id="activity",
        campaign_id="campaign",
        phase="dev",
        run_id="a" * 64,
        claim_id="dev-0",
        strategy=Strategy.ALWAYS_SINGLE,
        repeat=0,
        result=result,
        summary=summary,
        requested_alias="alias",
        price_config_id="b" * 64,
        latency=LatencyBreakdown(
            fresh_end_to_end_ms=10,
            model_active_ms=8,
            retry_ms=0,
            queue_ms=2,
            checkpoint_downtime_ms=0,
            total_elapsed_ms=10,
            interruption_count=0,
        ),
    )
    assert artifact.usage.total_tokens == 14
    assert artifact.result.usage.total_tokens == 14
    assert artifact.call_ids == ["call-0"]
    assert artifact.actual_cost_micro_cny == 22
    assert artifact.known_actual_cost_micro_cny == 22
    assert artifact.committed_cost_micro_cny == 22
    assert artifact.cost_is_lower_bound is False
    assert artifact.response_model_ids_raw == ["relay-id"]
    assert artifact.result.estimated_cost_micro_cny == 22
    assert artifact.result.cost_currency == "CNY"
    assert artifact.result.price_config_id == "b" * 64
    assert artifact.cache_hit_count == 2
    assert artifact.artifact_sha256 == artifact.artifact_sha256.lower()
    assert artifact.artifact_sha256 == artifact_fingerprint(artifact)


def test_build_run_artifact_marks_missing_usage_as_diagnostic() -> None:
    result = VerificationResult(
        claim_id="dev-0",
        status=ResultStatus.PARTIAL,
        verdict=Verdict.NOT_ENOUGH_EVIDENCE,
        confidence=0.4,
        rationale="partial fixture",
        initial_route="single",
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
    )
    summary = {
        "call_ids": ["call-0"],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "complete": False},
        "actual_cost_micro_cny": None,
        "known_actual_cost_micro_cny": 0,
        "committed_cost_micro_cny": 22,
        "cost_is_lower_bound": True,
        "cache_hit_count": 0,
        "requested_aliases": ["alias"],
        "response_model_ids_raw": ["relay-id"],
        "billing_uncertain": False,
        "usage_sources": ["missing"],
    }

    artifact = build_run_artifact(
        activity_id="activity",
        campaign_id="campaign",
        phase="dev",
        run_id="a" * 64,
        claim_id="dev-0",
        strategy=Strategy.ALWAYS_SINGLE,
        repeat=0,
        result=result,
        summary=summary,
        requested_alias="alias",
        price_config_id="b" * 64,
        latency=LatencyBreakdown(
            fresh_end_to_end_ms=10,
            model_active_ms=8,
            retry_ms=0,
            queue_ms=2,
            checkpoint_downtime_ms=0,
            total_elapsed_ms=10,
            interruption_count=0,
        ),
    )

    assert artifact.diagnostic_only is True
    assert artifact.usage_source == "missing"
    assert artifact.result.usage.complete is False
    assert artifact.result.price_config_id is None


def test_build_run_artifact_handles_zero_call_pre_route_failure() -> None:
    result = VerificationResult(
        claim_id="dev-0",
        status=ResultStatus.FAILED,
        rationale="probe unavailable",
        initial_route=None,
        failure_stage="pre_route",
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
        errors=["PROBE_RETRIEVAL_FAILED"],
    )
    summary = {
        "call_ids": [],
        "usage": Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=True),
        "actual_cost_micro_cny": 0,
        "known_actual_cost_micro_cny": 0,
        "committed_cost_micro_cny": 0,
        "cost_is_lower_bound": False,
        "cache_hit_count": 0,
        "requested_aliases": [],
        "response_model_ids_raw": [],
        "billing_uncertain": False,
    }

    artifact = build_run_artifact(
        activity_id="activity",
        campaign_id="campaign",
        phase="dev",
        run_id="a" * 64,
        claim_id="dev-0",
        strategy=Strategy.ALWAYS_SINGLE,
        repeat=0,
        result=result,
        summary=summary,
        requested_alias="alias",
        price_config_id="b" * 64,
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

    assert artifact.usage_source == "not_applicable"
    assert artifact.usage.total_tokens == 0
    assert artifact.diagnostic_only is False


def test_build_run_artifact_rejects_zero_call_non_pre_route_failure() -> None:
    result = VerificationResult(
        claim_id="dev-0",
        status=ResultStatus.FAILED,
        rationale="single node failed",
        initial_route="single",
        failure_stage="single",
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
    )
    summary = {
        "call_ids": [],
        "usage": Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=True),
        "actual_cost_micro_cny": 0,
        "known_actual_cost_micro_cny": 0,
        "committed_cost_micro_cny": 0,
        "cost_is_lower_bound": False,
        "cache_hit_count": 0,
        "requested_aliases": [],
        "response_model_ids_raw": [],
        "billing_uncertain": False,
    }

    with pytest.raises(ValueError, match="zero-call"):
        build_run_artifact(
            activity_id="activity",
            campaign_id="campaign",
            phase="dev",
            run_id="a" * 64,
            claim_id="dev-0",
            strategy=Strategy.ALWAYS_SINGLE,
            repeat=0,
            result=result,
            summary=summary,
            requested_alias="alias",
            price_config_id="b" * 64,
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


def test_build_run_artifact_rejects_paid_run_without_persisted_alias() -> None:
    result = VerificationResult(
        claim_id="dev-0",
        status=ResultStatus.COMPLETED,
        verdict=Verdict.SUPPORTED,
        confidence=0.9,
        rationale="fixture",
        initial_route="single",
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
    )
    summary = {
        "call_ids": ["call-0"],
        "usage": Usage(input_tokens=10, output_tokens=4, total_tokens=14, complete=True),
        "actual_cost_micro_cny": 22,
        "known_actual_cost_micro_cny": 22,
        "committed_cost_micro_cny": 22,
        "cost_is_lower_bound": False,
        "fresh_call_count": 1,
        "cache_hit_count": 0,
        "requested_aliases": [],
        "response_model_ids_raw": ["relay-id"],
        "billing_uncertain": False,
    }

    with pytest.raises(ValueError, match="alias"):
        build_run_artifact(
            activity_id="activity",
            campaign_id="campaign",
            phase="dev",
            run_id="a" * 64,
            claim_id="dev-0",
            strategy=Strategy.ALWAYS_SINGLE,
            repeat=0,
            result=result,
            summary=summary,
            requested_alias="alias",
            price_config_id="b" * 64,
            latency=LatencyBreakdown(
                fresh_end_to_end_ms=10,
                model_active_ms=8,
                retry_ms=0,
                queue_ms=2,
                checkpoint_downtime_ms=0,
                total_elapsed_ms=10,
                interruption_count=0,
            ),
        )


def test_price_loader_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_bytes(b"provider: relay\n\xff\n")

    with pytest.raises(ValueError, match="strict CNY"):
        load_price_config(path)
