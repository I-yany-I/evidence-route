from pathlib import Path

import pytest

from evidence_route.config import load_app_config, redact_mapping, stable_hash


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
