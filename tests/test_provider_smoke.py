from __future__ import annotations

import json
from pathlib import Path

import pytest

import evidence_route.cli as cli
from evidence_route.config import stable_hash
from evidence_route.llm import RawCompletion


class FakeCapabilityTransport:
    def __init__(
        self,
        *,
        nonce: str,
        response_model_id_raw: str | None = "relay-model-2026-08",
        content: str | None = None,
    ) -> None:
        self.nonce = nonce
        self.response_model_id_raw = response_model_id_raw
        self.content = content
        self.requests: list[dict[str, object]] = []

    async def create(self, **request: object) -> RawCompletion:
        self.requests.append(request)
        return RawCompletion(
            content=self.content
            or json.dumps({"ok": True, "nonce": self.nonce}),
            response_model_id_raw=self.response_model_id_raw,
            input_tokens=11,
            output_tokens=3,
        )


@pytest.fixture
def smoke_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://relay.example/v1/")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "super-secret-provider-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "requested-relay-alias")
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        "\n".join(
            [
                "provider: fake",
                "currency: CNY",
                "input_per_million: 1.0",
                "output_per_million: 2.0",
                "price_source: test",
                "strict_evaluation: true",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "nonce": "nonce-round-trip-123",
        "config_path": Path("configs/default.yaml"),
        "pricing_path": pricing_path,
        "artifact_dir": tmp_path / "provider-smoke",
    }


def test_production_provider_smoke_records_one_audited_capability_call(
    smoke_inputs: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    nonce = str(smoke_inputs["nonce"])
    transport = FakeCapabilityTransport(nonce=nonce)
    monkeypatch.setattr(cli, "OpenAITransport", lambda settings: transport)

    payload = cli.ProductionServices().provider_smoke(**smoke_inputs)

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request["schema"] is cli.CapabilityPayload
    assert payload == {
        "structured_output": True,
        "usage_complete": True,
        "requested_alias": "requested-relay-alias",
        "response_model_id_raw": "relay-model-2026-08",
        "identity_verified": False,
        "nonce": nonce,
        "input_tokens": 11,
        "output_tokens": 3,
        "estimated_cost_micro_cny": 17,
        "call_ids": [payload["call_ids"][0]],
        "endpoint_config_hash": stable_hash(
            {"base_url": "https://relay.example/v1"}
        ),
    }
    assert len(payload["call_ids"]) == 1
    assert len(payload["call_ids"][0]) == 64
    assert (Path(smoke_inputs["artifact_dir"]) / "run-store.sqlite3").is_file()
    assert capsys.readouterr().out == ""
    assert "super-secret-provider-key" not in json.dumps(payload)
    for path in Path(smoke_inputs["artifact_dir"]).glob("*"):
        assert b"super-secret-provider-key" not in path.read_bytes()


def test_production_provider_smoke_rejects_nonce_mismatch(
    smoke_inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeCapabilityTransport(nonce="different-nonce")
    monkeypatch.setattr(cli, "OpenAITransport", lambda settings: transport)

    with pytest.raises(ValueError, match="nonce mismatch"):
        cli.ProductionServices().provider_smoke(**smoke_inputs)

    assert len(transport.requests) == 1


def test_production_provider_smoke_rejects_missing_raw_model_id(
    smoke_inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeCapabilityTransport(
        nonce=str(smoke_inputs["nonce"]), response_model_id_raw=None
    )
    monkeypatch.setattr(cli, "OpenAITransport", lambda settings: transport)

    with pytest.raises(ValueError, match="model id"):
        cli.ProductionServices().provider_smoke(**smoke_inputs)

    assert len(transport.requests) == 1


def test_production_provider_smoke_does_not_repair_failed_capability_response(
    smoke_inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeCapabilityTransport(
        nonce=str(smoke_inputs["nonce"]),
        content=json.dumps({"ok": False, "nonce": smoke_inputs["nonce"]}),
    )
    monkeypatch.setattr(cli, "OpenAITransport", lambda settings: transport)

    with pytest.raises(ValueError):
        cli.ProductionServices().provider_smoke(**smoke_inputs)

    assert len(transport.requests) == 1
