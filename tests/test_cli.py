import json
from pathlib import Path

from typer.testing import CliRunner

from evidence_route.cli import CliServices, create_app


class FakeServices(CliServices):
    def __init__(self) -> None:
        self.verify_calls = 0
        self.smoke_calls = 0
        self.preview_calls = 0
        self.evaluate_calls = 0
        self.collect_calls = 0
        self.replay_calls = 0
        self.report_calls = 0

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
        return {"status": "complete", "mode": kwargs["mode"]}

    def calibrate_collect(self, **kwargs):
        self.collect_calls += 1
        return {"status": "complete"}

    def calibrate_replay(self, **kwargs):
        self.replay_calls += 1
        return {"status": "complete", "selected": "candidate-0"}

    def report(self, **kwargs):
        self.report_calls += 1
        return {"publishable": kwargs["publish"], "output_dir": str(kwargs["output_dir"])}


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
        ],
    )
    assert result.exit_code == 0
    assert '"base_call_upper_bound": 1544' in result.output
    assert services.preview_calls == 1
    assert services.evaluate_calls == 0


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
