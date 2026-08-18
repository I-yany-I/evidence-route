import json
import shutil
from pathlib import Path

import pytest

from evidence_route.contracts import Citation, ResultStatus, Usage, Verdict, VerificationResult
from evidence_route.evaluation.official import (
    build_paper_predictions,
    build_shared_task_predictions,
    run_official_evaluators,
    select_completed_triples,
    verify_evaluator_sources,
)
from evidence_route.evaluation.runtime_manifest import load_runtime_manifest
from evidence_route.evaluation.scorer_manifest import (
    align_runtime_and_gold,
    load_gold_manifest,
)

ROOT = Path("tests/fixtures/evaluation")


def _citation() -> Citation:
    return Citation(
        evidence_id="av:dev:0:0:0",
        claim_unit_ids=["u0"],
        question="What happened?",
        answer="A source says it happened.",
        quote="A source says it happened.",
        stance="supports",
        source_url="https://example.org/source",
    )


@pytest.fixture
def aligned_claims():
    runtime = load_runtime_manifest(ROOT / "runtime.json", allowed_root=ROOT)
    gold = load_gold_manifest(ROOT / "gold.json", allowed_root=ROOT)
    return [
        item
        for item in align_runtime_and_gold(runtime, gold)
    ]


@pytest.fixture
def completed_results():
    usage = Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True)
    return [
        VerificationResult(
            claim_id="dev-0",
            status=ResultStatus.COMPLETED,
            verdict=Verdict.SUPPORTED,
            confidence=0.9,
            rationale="supported by the source",
            citations=[_citation()],
            initial_route="single",
            usage=usage,
        ),
        VerificationResult(
            claim_id="dev-1",
            status=ResultStatus.COMPLETED,
            verdict=Verdict.REFUTED,
            confidence=0.8,
            rationale="refuted by the source",
            citations=[_citation()],
            initial_route="multi",
            usage=usage,
        ),
    ]


def test_pinned_evaluator_hashes_match() -> None:
    verify_evaluator_sources(Path("third_party/averitec/SOURCES.json"))


def test_shared_task_adapter_preserves_manifest_order(completed_results, aligned_claims) -> None:
    selection = select_completed_triples(aligned_claims, completed_results)
    payload = build_shared_task_predictions(selection.triples)
    assert [item["claim_id"] for item in payload] == [
        item[0].original_id for item in aligned_claims
    ]
    assert payload[0]["pred_label"] in {verdict.value for verdict in Verdict}
    assert set(payload[0]["evidence"][0]) == {
        "question", "answer", "url", "scraped_text"
    }


def test_official_selector_omits_only_non_completed(completed_results, aligned_claims) -> None:
    completed_results[0] = completed_results[0].model_copy(
        update={"status": ResultStatus.PARTIAL}
    )
    selection = select_completed_triples(aligned_claims, completed_results)
    assert selection.omitted_claim_ids == [aligned_claims[0][0].claim_id]
    assert selection.completion_rate == pytest.approx(0.5)


def test_paper_adapter_uses_only_official_fields(completed_results, aligned_claims) -> None:
    payload = build_paper_predictions(
        select_completed_triples(aligned_claims, completed_results).triples
    )
    assert payload[0]["label"] == completed_results[0].verdict.value
    assert payload[0]["questions"][0]["answers"][0]["answer"]
    assert payload[0]["justification"] == completed_results[0].rationale
    assert set(payload[0]) == {"label", "questions", "justification"}


def test_completed_empty_evidence_stays_in_official_cohort(
    completed_results, aligned_claims
) -> None:
    completed_results[0] = completed_results[0].model_copy(
        update={"citations": [], "available_evidence_ids": []}
    )
    selection = select_completed_triples(aligned_claims, completed_results)
    payload = build_shared_task_predictions(selection.triples)
    assert len(payload) == len(aligned_claims)
    assert payload[0]["evidence"] == []


def test_missing_result_is_omitted_without_reordering(completed_results, aligned_claims) -> None:
    selection = select_completed_triples(aligned_claims, completed_results[:1])
    assert [triple.runtime.claim_id for triple in selection.triples] == ["dev-0"]
    assert selection.omitted_claim_ids == ["dev-1"]
    assert selection.completion_rate == pytest.approx(0.5)


def test_out_of_order_results_are_rejected(completed_results, aligned_claims) -> None:
    with pytest.raises(ValueError, match="manifest order"):
        select_completed_triples(aligned_claims, list(reversed(completed_results)))


def test_source_verifier_requires_complete_provenance_manifest(tmp_path: Path) -> None:
    source_root = Path("third_party/averitec")
    target_root = tmp_path / "averitec"
    for relative in (
        "paper/eval.py",
        "paper/utils.py",
        "paper/leven.py",
        "shared_task/evaluate_veracity.py",
    ):
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
    spec = json.loads((source_root / "SOURCES.json").read_text(encoding="utf-8"))
    spec["sources"] = spec["sources"][:-1]
    reduced = target_root / "SOURCES.json"
    reduced.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="complete|expected"):
        verify_evaluator_sources(reduced)


def test_runner_resolves_relative_artifacts_before_changing_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed_results,
    aligned_claims,
) -> None:
    fake = tmp_path / "fake_evaluator.py"
    fake.write_text("print('Question-only score (HU-meteor): 0.5')\n", encoding="utf-8")
    source_spec = Path("third_party/averitec/SOURCES.json").resolve()
    monkeypatch.chdir(tmp_path)
    selection = select_completed_triples(aligned_claims, completed_results)
    report = run_official_evaluators(
        selection,
        output_dir=Path("relative-artifacts"),
        source_spec=source_spec,
        shared_task_script=fake,
        paper_script=fake,
    )
    artifact_dir = Path(report["artifact_dir"])
    assert artifact_dir.is_absolute()
    assert (artifact_dir / "shared_task_2024.metrics.json").is_file()
    assert (artifact_dir / "paper_2023_secondary.metrics.json").is_file()
