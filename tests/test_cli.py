import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from evidence_route.cli import CliServices, ProductionServices, create_app


class FakeServices(CliServices):
    def __init__(self) -> None:
        self.verify_calls = 0
        self.smoke_calls = 0
        self.preview_calls = 0
        self.evaluate_calls = 0
        self.collect_calls = 0
        self.last_evaluate = {}
        self.last_collect = {}
        self.replay_calls = 0
        self.report_calls = 0
        self.report_kwargs = None
        self.diagnostic_calls = 0
        self.last_diagnostic = {}

    def verify(self, **kwargs):
        self.verify_calls += 1
        return {
            "run_id": kwargs["run_id"],
            "claim_id": kwargs["claim_id"],
            "status": "completed",
            "verdict": "Supported",
        }

    def provider_smoke(self, **kwargs):
        self.smoke_calls += 1
        return {
            "structured_output": True,
            "usage_complete": True,
            "requested_alias": "relay-alias",
            "response_model_id_raw": "relay-reported-id",
            "identity_verified": False,
            "nonce": kwargs["nonce"],
            "input_tokens": 8,
            "output_tokens": 2,
            "estimated_cost_micro_cny": 10_000,
            "call_ids": ["a" * 64],
            "endpoint_config_hash": "b" * 64,
        }

    def preview_campaign(self, **kwargs):
        self.preview_calls += 1
        return {
            "base_call_upper_bound": 1544,
            "repair_upper_bound": 3088,
            "fault_upper_bound": 9264,
            "startup_required_micro_cny": 1_200_000,
            "cap_micro_cny": 350_000_000,
        }

    def evaluate(self, **kwargs):
        self.evaluate_calls += 1
        self.last_evaluate = kwargs
        return {"status": "complete", "mode": kwargs["mode"]}

    def calibrate_collect(self, **kwargs):
        self.collect_calls += 1
        self.last_collect = kwargs
        return {"status": "complete"}

    def calibrate_replay(self, **kwargs):
        self.replay_calls += 1
        return {"status": "complete", "selected": "candidate-0"}

    def report(self, **kwargs):
        self.report_calls += 1
        self.report_kwargs = kwargs
        return {"publishable": kwargs["publish"], "output_dir": str(kwargs["output_dir"])}

    def diagnose_retrieval(self, **kwargs):
        self.diagnostic_calls += 1
        self.last_diagnostic = kwargs
        return {"summary": {"claim_count": 1}}


def test_diagnose_retrieval_forwards_paths_and_ablation(tmp_path: Path) -> None:
    services = FakeServices()
    result = CliRunner().invoke(
        create_app(services),
        [
            "diagnose-retrieval",
            "--runtime-manifest", str(tmp_path / "runtime.json"),
            "--gold-manifest", str(tmp_path / "gold.json"),
            "--corpus-dir", str(tmp_path / "corpora"),
            "--config", str(tmp_path / "retrieval.yaml"),
            "--progress", str(tmp_path / "progress.json"),
            "--output", str(tmp_path / "diagnostic.json"),
            "--timing-output", str(tmp_path / "timing.json"),
            "--ablation", "source-only",
        ],
    )
    assert result.exit_code == 0
    assert services.diagnostic_calls == 1
    assert services.last_diagnostic["ablation"] == "source-only"
    assert services.last_diagnostic["runtime_manifest"] == tmp_path / "runtime.json"


def test_verify_writes_machine_readable_result(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        create_app(FakeServices()),
        [
            "verify",
            "--claim-id",
            "dev-0",
            "--claim",
            "The claim",
            "--strategy",
            "adaptive",
            "--run-id",
            "run-cli",
            "--artifact-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    saved = json.loads((tmp_path / "run-cli" / "result.json").read_text(encoding="utf-8"))
    assert saved["verdict"] == "Supported"


def test_provider_smoke_requires_explicit_paid_acknowledgement() -> None:
    services = FakeServices()
    result = CliRunner().invoke(create_app(services), ["provider-smoke"])
    assert result.exit_code == 2
    assert "--accept-paid-call" in result.output
    assert services.smoke_calls == 0


def test_provider_smoke_marks_model_identity_unverified() -> None:
    result = CliRunner().invoke(
        create_app(FakeServices()), ["provider-smoke", "--accept-paid-call"]
    )
    assert result.exit_code == 0
    assert '"identity_verified": false' in result.output
    assert "api_key" not in result.output.lower()


def test_resume_requires_explicit_run_id(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        create_app(FakeServices()),
        [
            "verify",
            "--claim-id",
            "dev-0",
            "--claim",
            "Claim",
            "--resume",
            "--artifact-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "--run-id" in result.output


def test_provider_smoke_rejects_nonce_mismatch() -> None:
    class BadNonceServices(FakeServices):
        def provider_smoke(self, **kwargs):
            payload = super().provider_smoke(**kwargs)
            payload["nonce"] = "different-nonce"
            return payload

    result = CliRunner().invoke(
        create_app(BadNonceServices()), ["provider-smoke", "--accept-paid-call"]
    )
    assert result.exit_code == 1


def test_evaluate_defaults_to_budget_preview_without_network(tmp_path: Path) -> None:
    services = FakeServices()
    result = CliRunner().invoke(
        create_app(services),
        [
            "evaluate",
            "--manifest",
            str(tmp_path / "dev.json"),
            "--stability-manifest",
            str(tmp_path / "stability.json"),
            "--activity-dir",
            str(tmp_path / "activity"),
            "--calibration-report",
            str(tmp_path / "calibration-report.json"),
        ],
    )
    assert result.exit_code == 0
    assert '"base_call_upper_bound": 1544' in result.output
    assert services.preview_calls == 1
    assert services.evaluate_calls == 0


def test_paid_evaluate_forwards_batch_limit(tmp_path: Path) -> None:
    services = FakeServices()
    result = CliRunner().invoke(
        create_app(services),
        [
            "evaluate",
            "--manifest",
            str(tmp_path / "dev.json"),
            "--stability-manifest",
            str(tmp_path / "stability.json"),
            "--activity-dir",
            str(tmp_path / "activity"),
            "--calibration-report",
            str(tmp_path / "calibration-report.json"),
            "--accept-paid-campaign",
            "--start-after-calibration",
            "--max-items",
            "3",
        ],
    )
    assert result.exit_code == 0
    assert services.evaluate_calls == 1
    assert services.last_evaluate["max_items"] == 3


def test_paid_evaluate_forwards_isolated_experiment_identity(tmp_path: Path) -> None:
    services = FakeServices()
    result = CliRunner().invoke(
        create_app(services),
        [
            "evaluate",
            "--manifest",
            str(tmp_path / "dev.json"),
            "--stability-manifest",
            str(tmp_path / "stability.json"),
            "--activity-dir",
            str(tmp_path / "activity"),
            "--calibration-report",
            str(tmp_path / "calibration-report.json"),
            "--accept-paid-campaign",
            "--start-after-calibration",
            "--parent-activity",
            "gate-a-20260830",
            "--parent-report",
            str(tmp_path / "parent-report.json"),
            "--parent-config",
            str(tmp_path / "parent-config.yaml"),
            "--experiment-dir",
            str(tmp_path / "experiment"),
        ],
    )

    assert result.exit_code == 0
    assert services.last_evaluate["parent_activity"] == "gate-a-20260830"
    assert services.last_evaluate["parent_report"] == tmp_path / "parent-report.json"
    assert services.last_evaluate["parent_config"] == tmp_path / "parent-config.yaml"
    assert services.last_evaluate["experiment_dir"] == tmp_path / "experiment"


def test_calibrate_forwards_default_and_explicit_case_limits(tmp_path: Path) -> None:
    services = FakeServices()
    base = [
        "calibrate",
        "--runtime-manifest",
        str(tmp_path / "runtime.json"),
        "--activity-dir",
        str(tmp_path / "activity"),
        "--output-config",
        str(tmp_path / "calibrated.yaml"),
        "--output-report",
        str(tmp_path / "calibration.json"),
        "--collect",
        "--accept-paid-campaign",
    ]
    assert CliRunner().invoke(create_app(services), base).exit_code == 0
    assert services.last_collect["max_cases"] == 4

    explicit = CliRunner().invoke(create_app(services), [*base, "--max-cases", "2"])
    assert explicit.exit_code == 0
    assert services.last_collect["max_cases"] == 2


def test_production_preview_does_not_construct_transport() -> None:
    constructed = False

    def forbidden_transport(settings):
        nonlocal constructed
        constructed = True
        raise AssertionError("budget preview must not construct a transport")

    services = ProductionServices(transport_factory=forbidden_transport)
    payload = services.preview_campaign(
        config_path=Path("configs/default.yaml"),
        pricing_path=Path("configs/pricing.dryrun.yaml"),
    )
    assert constructed is False
    assert payload["base_call_upper_bound"] == 1544
    assert payload["repair_upper_bound"] == 3088
    assert payload["fault_upper_bound"] == 9264
    assert payload["paid_execution_started"] is False


def test_v4_preview_includes_recovery_call_allowance() -> None:
    services = ProductionServices()
    payload = services.preview_campaign(
        config_path=Path("configs/stability-v4.yaml"),
        pricing_path=Path("configs/pricing.dryrun.yaml"),
        max_items=10,
    )

    assert payload["base_call_upper_bound"] == 1664
    assert payload["batch_max_items"] == 10
    assert payload["batch_startup_required_micro_cny"] > 0
    assert payload["batch_startup_required_micro_cny"] < payload["cap_micro_cny"]


def test_paid_evaluate_requires_exactly_one_lifecycle_flag(tmp_path: Path) -> None:
    services = FakeServices()
    args = [
        "evaluate",
        "--manifest",
        str(tmp_path / "dev.json"),
        "--stability-manifest",
        str(tmp_path / "stability.json"),
        "--activity-dir",
        str(tmp_path / "activity"),
        "--calibration-report",
        str(tmp_path / "calibration-report.json"),
        "--accept-paid-campaign",
    ]
    result = CliRunner().invoke(create_app(services), args)
    assert result.exit_code == 2
    assert "exactly one" in result.output
    assert services.evaluate_calls == 0


def test_calibrate_replay_never_invokes_collect(tmp_path: Path) -> None:
    services = FakeServices()
    result = CliRunner().invoke(
        create_app(services),
        [
            "calibrate",
            "--runtime-manifest",
            str(tmp_path / "runtime.json"),
            "--activity-dir",
            str(tmp_path / "activity"),
            "--output-config",
            str(tmp_path / "calibrated.yaml"),
            "--output-report",
            str(tmp_path / "calibration.json"),
            "--replay",
            "--gold-manifest",
            str(tmp_path / "gold.json"),
        ],
    )
    assert result.exit_code == 0
    assert services.replay_calls == 1
    assert services.collect_calls == 0


def test_report_does_not_require_llm_environment(tmp_path: Path, monkeypatch) -> None:
    for name in (
        "EVIDENCE_ROUTE_API_KEY",
        "EVIDENCE_ROUTE_BASE_URL",
        "EVIDENCE_ROUTE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    services = FakeServices()
    result = CliRunner().invoke(
        create_app(services),
        [
            "report",
            "--activity-dir",
            str(tmp_path / "activity"),
            "--gold-manifest",
            str(tmp_path / "gold.json"),
            "--output-dir",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0
    assert services.report_calls == 1


def test_report_forwards_explicit_publication_evidence_paths(tmp_path: Path) -> None:
    class CapturingServices(FakeServices):
        report_kwargs: dict[str, object] | None = None

        def report(self, **kwargs):
            self.report_kwargs = kwargs
            return super().report(**kwargs)

    services = CapturingServices()
    paths = {
        "repository-root": tmp_path / "repo",
        "run-store": tmp_path / "ledger.sqlite3",
        "runtime-manifest": tmp_path / "dev-runtime.json",
        "calibration-runtime-manifest": tmp_path / "calibration-runtime.json",
        "stability-runtime-manifest": tmp_path / "stability-runtime.json",
        "calibration-report": tmp_path / "calibration-report.json",
        "calibrated-config": tmp_path / "calibrated.yaml",
        "corpus-preparation-receipt": tmp_path / "preparation-receipt.json",
        "prompt-bundle": tmp_path / "prompts.py",
        "pricing": tmp_path / "pricing.yaml",
        "requirements-lock": tmp_path / "requirements.lock",
        "nltk-data-root": tmp_path / "nltk",
    }
    args = [
        "report",
        "--activity-dir",
        str(tmp_path / "activity"),
        "--gold-manifest",
        str(tmp_path / "gold.json"),
        "--output-dir",
        str(tmp_path / "report"),
    ]
    for option, path in paths.items():
        args.extend((f"--{option}", str(path)))

    result = CliRunner().invoke(create_app(services), args)

    assert result.exit_code == 0
    assert services.report_kwargs is not None
    assert services.report_kwargs["stability_diagnostics"] is False
    for option, path in paths.items():
        assert services.report_kwargs[option.replace("-", "_")] == path


def test_report_forwards_stability_diagnostics_flag(tmp_path: Path) -> None:
    services = FakeServices()
    result = CliRunner().invoke(
        create_app(services),
        [
            "report",
            "--activity-dir",
            str(tmp_path / "activity"),
            "--gold-manifest",
            str(tmp_path / "gold.json"),
            "--output-dir",
            str(tmp_path / "report"),
            "--stability-diagnostics",
        ],
    )

    assert result.exit_code == 0
    assert services.report_calls == 1
    assert services.report_kwargs is not None
    assert services.report_kwargs["stability_diagnostics"] is True


def test_production_report_resolves_relative_paths_from_repository_root(
    tmp_path: Path, monkeypatch
) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    captured: dict[str, object] = {}

    class Bundle:
        publication_gate = SimpleNamespace(publishable=False)

        def write(self, output_dir: Path, *, readme: Path | None = None) -> None:
            captured["output_dir"] = output_dir
            captured["readme"] = readme

    def fake_build_report_bundle(report_input, *, publish=False, stability_diagnostics=False):
        captured["report_input"] = report_input
        captured["publish"] = publish
        captured["stability_diagnostics"] = stability_diagnostics
        return Bundle()

    monkeypatch.setattr("evidence_route.cli.build_report_bundle", fake_build_report_bundle)
    services = ProductionServices(transport=object())
    services.report(
        repository_root=repository_root,
        activity_dir=Path("artifacts/evaluation/gate-a"),
        gold_manifest=Path("data/scorer_manifests/dev.json"),
        output_dir=Path("reports/incomplete/gate-a"),
        run_store=Path("artifacts/gate-a.sqlite3"),
        runtime_manifest=Path("data/manifests/dev.json"),
        calibration_runtime_manifest=Path("data/manifests/calibration.json"),
        stability_runtime_manifest=Path("data/manifests/stability.json"),
        calibration_report=Path("reports/calibration/report.json"),
        calibrated_config=Path("configs/calibrated.yaml"),
        corpus_preparation_receipt=Path("data/processed/receipt.json"),
        prompt_bundle=Path("src/evidence_route/prompts.py"),
        pricing=Path("configs/pricing.local.yaml"),
        requirements_lock=Path("requirements.lock"),
        nltk_data_root=Path("data/external/nltk"),
        publish=False,
        readme=Path("README.md"),
    )

    report_input = captured["report_input"]
    assert report_input.repository_root == repository_root.resolve()
    assert report_input.activity_dir == (repository_root / "artifacts/evaluation/gate-a").resolve()
    assert (
        report_input.gold_manifest == (repository_root / "data/scorer_manifests/dev.json").resolve()
    )
    assert report_input.run_store == (repository_root / "artifacts/gate-a.sqlite3").resolve()
    assert captured["output_dir"] == (repository_root / "reports/incomplete/gate-a").resolve()
    assert captured["readme"] == (repository_root / "README.md").resolve()
    assert captured["stability_diagnostics"] is False
