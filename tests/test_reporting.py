from pathlib import Path

import pytest

from evidence_route.evaluation.activity import CampaignStatus
from evidence_route.evaluation.reporting import (
    PublicationBlocked,
    build_report_bundle,
)

pytest_plugins = ["tests.fixtures.evaluation.report_factory"]


def test_report_uses_exact_subset_name(complete_report_input) -> None:
    bundle = build_report_bundle(complete_report_input)
    assert "AVeriTeC dev balanced subset (n=80)" in bundle.markdown
    assert "full benchmark" not in bundle.markdown.lower()
    assert "leaderboard" not in bundle.markdown.lower()


def test_resume_text_uses_measured_values_not_design_targets(complete_report_input) -> None:
    bundle = build_report_bundle(complete_report_input)
    adaptive = bundle.summary["strategies"]["adaptive"]
    assert f"{adaptive['token_reduction_pct']:.1f}%" in bundle.resume_snippet
    assert f"{adaptive['full_manifest_macro_f1']:.3f}" in bundle.resume_snippet
    assert "至少降低 40%" not in bundle.resume_snippet
    assert "官方 OpenAI" not in bundle.resume_snippet


@pytest.mark.parametrize(
    "status", [status for status in CampaignStatus if status is not CampaignStatus.COMPLETE]
)
def test_every_non_complete_status_is_diagnostic_only(
    status, report_input_factory, tmp_path: Path
) -> None:
    report_input = report_input_factory.with_status(status)
    bundle = build_report_bundle(report_input, publish=False)
    diagnostic = tmp_path / "reports" / "incomplete" / status.value
    bundle.write(diagnostic)
    assert bundle.summary["publishable"] is False
    assert bundle.resume_snippet is None
    assert not (diagnostic / "resume_snippet.md").exists()

    final_dir = tmp_path / "reports" / "final"
    readme = tmp_path / "README.md"
    readme.write_text("unchanged", encoding="utf-8")
    with pytest.raises(PublicationBlocked, match=status.value):
        bundle.write(final_dir, readme=readme)
    with pytest.raises(PublicationBlocked, match=status.value):
        build_report_bundle(report_input, publish=True)
    assert not final_dir.exists()
    assert readme.read_text(encoding="utf-8") == "unchanged"


def test_regeneration_reads_artifacts_without_executor(
    complete_report_input, tmp_path: Path
) -> None:
    bundle = build_report_bundle(complete_report_input)
    bundle.write(tmp_path / "generated")
    assert (tmp_path / "generated" / "summary.json").is_file()
    assert (tmp_path / "generated" / "report.md").is_file()
    assert (tmp_path / "generated" / "resume_snippet.md").is_file()
