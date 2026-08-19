from __future__ import annotations

import json
from pathlib import Path

import pytest

import evidence_route.cli as cli


class FakeTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **request: object):
        from evidence_route.llm import RawCompletion

        self.calls += 1
        schema = request["schema"]
        if schema.__name__ != "VerdictDraft":
            raise AssertionError(f"unexpected schema: {schema.__name__}")
        payload = {
            "verdict": "Supported",
            "confidence": 0.95,
            "rationale": "The frozen evidence supports the claim.",
            "citations": [
                {
                    "evidence_id": "av:dev:0:0:0",
                    "claim_unit_ids": ["u0"],
                    "question": "What does the source say?",
                    "answer": "The source describes the claim.",
                    "quote": "Sean Connery never sent the letter to Steve Jobs.",
                    "stance": "supports",
                    "source_url": "https://example.org/cnet",
                }
            ],
        }
        return RawCompletion(
            content=json.dumps(payload),
            response_model_id_raw="fake-model",
            input_tokens=12,
            output_tokens=8,
        )


@pytest.fixture
def verify_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "test-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "test-model")
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "\n".join(
            [
                "provider: fake",
                "currency: CNY",
                "input_per_million: 1.0",
                "output_per_million: 1.0",
                "price_source: test",
                "strict_evaluation: true",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "run_id": "production-run",
        "claim_id": "dev-0",
        "claim": "Sean Connery sent the letter to Steve Jobs.",
        "strategy": "always_single",
        "artifact_dir": tmp_path / "artifacts",
        "config_path": Path("configs/default.yaml"),
        "pricing_path": pricing,
        "corpus_dir": Path("tests/fixtures/averitec/corpora"),
        "checkpoint_db": tmp_path / "checkpoints.sqlite3",
    }


def test_production_verify_runs_fresh_graph_with_async_checkpoint(
    verify_inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeTransport()
    monkeypatch.setattr(cli, "OpenAITransport", lambda settings: transport, raising=False)

    payload = cli.ProductionServices().verify(resume=False, **verify_inputs)

    assert payload["status"] == "completed"
    assert payload["verdict"] == "Supported"
    assert payload["run_id"] == "production-run"
    assert payload["summary"]["fresh_call_count"] == 1
    assert transport.calls == 1


def test_production_verify_resume_reuses_completed_transport_call(
    verify_inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeTransport()
    monkeypatch.setattr(cli, "OpenAITransport", lambda settings: transport, raising=False)
    services = cli.ProductionServices()

    services.verify(resume=False, **verify_inputs)
    resumed = services.verify(resume=True, **verify_inputs)

    assert resumed["status"] == "completed"
    assert len(resumed["summary"]["call_ids"]) == 1
    assert transport.calls == 1


def test_production_verify_resume_rejects_claim_mismatch(
    verify_inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "OpenAITransport", lambda settings: FakeTransport(), raising=False)
    services = cli.ProductionServices()
    services.verify(resume=False, **verify_inputs)

    with pytest.raises(ValueError, match="claim"):
        services.verify(
            resume=True,
            **{**verify_inputs, "claim": "A different claim."},
        )


def test_production_verify_resume_rejects_strategy_mismatch(
    verify_inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "OpenAITransport", lambda settings: FakeTransport(), raising=False)
    services = cli.ProductionServices()
    services.verify(resume=False, **verify_inputs)

    with pytest.raises(ValueError, match="strategy"):
        services.verify(
            resume=True,
            **{**verify_inputs, "strategy": "adaptive"},
        )
