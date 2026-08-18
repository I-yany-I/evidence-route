import hashlib
from pathlib import Path

import pytest

from evidence_route.contracts import ResultStatus, Usage, Verdict, VerificationResult
from evidence_route.evaluation.metrics import (
    NO_PREDICTION,
    paired_bootstrap_difference,
    score_completed_conditionally,
    score_full_manifest,
    stability_consistency,
    summarize_operations,
    wilson_interval,
)
from evidence_route.evaluation.runtime_manifest import load_runtime_manifest
from evidence_route.evaluation.scorer_manifest import (
    align_runtime_and_gold,
    load_gold_manifest,
)


def write_manifest(path: Path, payload: str) -> None:
    encoded = payload.encode("utf-8")
    path.write_bytes(encoded)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
    )


def result(
    claim_id: str, verdict: Verdict, status: ResultStatus = ResultStatus.COMPLETED
) -> VerificationResult:
    if status is ResultStatus.FAILED:
        return VerificationResult(
            claim_id=claim_id,
            status=status,
            rationale="transport failed",
            initial_route="single",
            usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False),
        )
    return VerificationResult(
        claim_id=claim_id,
        status=status,
        verdict=verdict,
        confidence=0.8,
        rationale="structured result",
        initial_route="single",
        usage=Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True),
    )


def test_runtime_loader_rejects_gold_fields(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    write_manifest(
        path,
        '{"dataset":"AVeriTeC","items":[{"claim_id":"dev-0",'
        '"original_id":0,"claim":"claim","split":"dev","label":"Refuted"}]}',
    )
    with pytest.raises(ValueError, match="runtime manifest contains forbidden field: label"):
        load_runtime_manifest(path, allowed_root=tmp_path)


def test_runtime_loader_rejects_path_outside_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "runtime"
    allowed.mkdir()
    outside = tmp_path / "scorer.json"
    write_manifest(outside, '{"dataset":"AVeriTeC","items":[]}')
    with pytest.raises(ValueError, match="outside allowed runtime root"):
        load_runtime_manifest(outside, allowed_root=allowed)


def test_runtime_loader_rejects_sidecar_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    write_manifest(path, '{"dataset":"AVeriTeC","items":[]}')
    path.write_text('{"dataset":"AVeriTeC","items":[1]}', encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        load_runtime_manifest(path, allowed_root=tmp_path)


def test_fixture_manifests_are_hash_bound_and_aligned() -> None:
    root = Path("tests/fixtures/evaluation")
    runtime = load_runtime_manifest(root / "runtime.json", allowed_root=root)
    gold = load_gold_manifest(root / "gold.json", allowed_root=root)
    aligned = align_runtime_and_gold(
        runtime,
        gold,
    )
    assert [runtime_item.claim_id for runtime_item, _gold_item in aligned] == [
        "dev-0",
        "dev-1",
    ]


def test_runtime_loader_verifies_corpus_bytes_and_records() -> None:
    root = Path("tests/fixtures/evaluation")
    runtime = load_runtime_manifest(
        root / "runtime.json", allowed_root=root, corpus_root=root
    )
    assert len(runtime.items) == 2


def test_stability_counts_incomplete_repeat_as_failure() -> None:
    runs = {
        "dev-0": [
            result("dev-0", Verdict.SUPPORTED),
            result("dev-0", Verdict.SUPPORTED),
            result("dev-0", Verdict.SUPPORTED),
        ],
        "dev-1": [
            result("dev-1", Verdict.REFUTED),
            result("dev-1", Verdict.REFUTED, ResultStatus.PARTIAL),
            result("dev-1", Verdict.REFUTED),
        ],
    }
    summary = stability_consistency(runs)
    assert summary.total == 2
    assert summary.consistent == 1
    assert summary.rate == 0.5


def test_partial_and_failed_are_no_prediction_false_negatives() -> None:
    gold = {
        "dev-0": Verdict.SUPPORTED,
        "dev-1": Verdict.REFUTED,
        "dev-2": Verdict.NOT_ENOUGH_EVIDENCE,
        "dev-3": Verdict.CONFLICTING,
    }
    results = {
        "dev-0": result("dev-0", Verdict.SUPPORTED),
        "dev-1": result("dev-1", Verdict.REFUTED, ResultStatus.PARTIAL),
        "dev-2": result("dev-2", Verdict.NOT_ENOUGH_EVIDENCE, ResultStatus.FAILED),
        "dev-3": result("dev-3", Verdict.CONFLICTING),
    }
    metrics = score_full_manifest(gold, results)
    assert metrics.sample_count == 4
    assert metrics.completed_count == 2
    assert metrics.accuracy == 0.5
    assert metrics.scored_predictions["dev-1"] == NO_PREDICTION
    assert metrics.scored_predictions["dev-2"] == NO_PREDICTION


def test_missing_result_is_not_silently_dropped() -> None:
    metrics = score_full_manifest({"dev-0": Verdict.SUPPORTED}, {})
    assert metrics.accuracy == 0.0
    assert metrics.completion_rate == 0.0
    assert metrics.missing_count == 1


def test_completed_metrics_include_completion_rate() -> None:
    metrics = score_completed_conditionally(
        {"dev-0": Verdict.SUPPORTED, "dev-1": Verdict.REFUTED},
        {"dev-0": result("dev-0", Verdict.SUPPORTED)},
    )
    assert metrics.sample_count == 1
    assert metrics.manifest_count == 2
    assert metrics.completion_rate == 0.5


def test_uncertainty_functions_are_seeded_and_bounded() -> None:
    gold = ["Supported", "Refuted", "Supported", "Refuted"]
    a = ["Supported", "Refuted", "Supported", "Refuted"]
    b = ["Supported", NO_PREDICTION, "Refuted", "Refuted"]
    first = paired_bootstrap_difference(gold, a, b, samples=10_000, seed=20260817)
    second = paired_bootstrap_difference(gold, a, b, samples=10_000, seed=20260817)
    assert first == second
    low, high = wilson_interval(successes=17, total=20)
    assert 0.0 <= low <= 0.85 <= high <= 1.0


def test_operation_summary_does_not_claim_exact_cost_with_missing_usage() -> None:
    summary = summarize_operations(
        [
            {
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 2,
                    "total_tokens": 10,
                    "complete": True,
                },
                "estimated_cost_micro_cny": 25,
                "fresh_end_to_end_ms": 20,
                "route": "single",
            },
            {
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "complete": False,
                },
                "fresh_end_to_end_ms": 30,
                "route": "multi",
            },
        ]
    )
    assert summary.total_tokens == 10
    assert summary.exact_cost_micro_cny is None
    assert summary.latency_p50_ms == 25.0
