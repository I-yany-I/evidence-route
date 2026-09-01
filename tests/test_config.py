from pathlib import Path

import pytest

from evidence_route.config import RoutingSettings, load_app_config, redact_mapping, stable_hash


def test_config_reads_secret_without_serializing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "secret-value")
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://relay.example")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "relay-alias")
    path = tmp_path / "config.yaml"
    path.write_text("routing:\n  low_confidence: 0.65\n", encoding="utf-8")
    config = load_app_config(path)
    assert config.llm.api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in config.model_dump_json()


def test_hash_is_key_order_independent() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_redaction_filters_authorization_and_keys() -> None:
    assert redact_mapping({"Authorization": "Bearer x", "api_key": "x", "name": "ok"}) == {
        "Authorization": "[REDACTED]",
        "api_key": "[REDACTED]",
        "name": "ok",
    }


def test_yaml_cannot_override_environment_identity(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  api_key: committed-secret\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must come from environment"):
        load_app_config(path)


def test_stability_v2_flags_are_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "secret-value")
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://relay.example")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "relay-alias")
    default_path = tmp_path / "default.yaml"
    default_path.write_text("routing:\n  low_confidence: 0.65\n", encoding="utf-8")
    v2_path = tmp_path / "stability-v2.yaml"
    v2_path.write_text(
        "hardening:\n"
        "  deterministic_ambiguous: true\n"
        "  deterministic_decomposition: true\n"
        "  hardened_judge: true\n"
        "  adjudication: true\n",
        encoding="utf-8",
    )

    default = load_app_config(default_path)
    v2 = load_app_config(v2_path)

    assert default.hardening.deterministic_ambiguous is False
    assert default.hardening.deterministic_decomposition is False
    assert default.hardening.hardened_judge is False
    assert default.hardening.adjudication is False
    assert v2.hardening.deterministic_ambiguous is True
    assert v2.hardening.deterministic_decomposition is True
    assert v2.hardening.hardened_judge is True
    assert v2.hardening.adjudication is True


def test_stability_v3_enables_worker_hardening_and_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "test-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "relay-model")

    v3 = load_app_config(Path("configs/stability-v3.yaml"))

    assert v3.hardening.hardened_worker is True
    assert v3.hardening.normalize_output is True


def test_stability_v4_enables_multi_single_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "test-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "relay-model")

    v4 = load_app_config(Path("configs/stability-v4.yaml"))

    assert v4.hardening.multi_single_recovery is True


def test_hardening_fields_do_not_change_historical_routing_hash() -> None:
    assert stable_hash(RoutingSettings().model_dump(mode="json")) == stable_hash(
        {
            "clear_multi_clauses": 3,
            "clear_single_min_sources": 2,
            "low_confidence": 0.65,
            "minimum_coverage": 1.0,
        }
    )
