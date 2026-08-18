import json
from pathlib import Path

from typer.testing import CliRunner

from evidence_route.cli import CliServices, create_app


class FakeServices(CliServices):
    def __init__(self) -> None:
        self.verify_calls = 0
        self.smoke_calls = 0

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
