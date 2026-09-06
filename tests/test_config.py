from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_route.config import (
    EvidenceSettings,
    RoutingSettings,
    load_app_config,
    redact_mapping,
    stable_hash,
)


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


def test_retrieval_v2_enables_source_cap_without_hardening_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "test-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "relay-model")

    config = load_app_config(Path("configs/retrieval-v2.yaml"))

    assert config.evidence.max_per_source == 1
    assert config.evidence.retrieval_mode == "source_hybrid_v2"
    assert config.evidence.dense_model_id == "BAAI/bge-small-en-v1.5"
    assert (
        config.evidence.dense_model_receipt_sha256
        == "0a27c87394284b6505f6226c60ae242e9fdd9c29840399a2936a2b4d976ef06a"
    )
    assert config.hardening.model_dump(mode="json") == {
        "deterministic_ambiguous": False,
        "deterministic_decomposition": False,
        "hardened_judge": False,
        "hardened_worker": False,
        "adjudication": False,
        "normalize_output": False,
        "multi_single_recovery": False,
    }


def test_quality_recovery_v2_combines_retrieval_and_stability_hardening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "test-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "relay-model")

    config = load_app_config(Path("configs/evidence-quality-recovery-v2.yaml"))

    assert config.routing.clear_single_min_sources == 1
    assert config.routing.low_confidence == 0.75
    assert all(config.hardening.model_dump(mode="json").values())
    assert config.evidence.retrieval_mode == "source_hybrid_v2"
    assert config.evidence.acquisition_weight == 0.3
    assert config.evidence.source_candidate_k == 256
    assert config.evidence.passages_per_source == 1
    assert config.budget.estimated_cost_cap_cny == 750.0


def test_hardening_fields_do_not_change_historical_routing_hash() -> None:
    assert stable_hash(RoutingSettings().model_dump(mode="json")) == stable_hash(
        {
            "clear_multi_clauses": 3,
            "clear_single_min_sources": 2,
            "low_confidence": 0.65,
            "minimum_coverage": 1.0,
        }
    )


def test_evidence_source_cap_is_optional_and_configurable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "secret-value")
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://relay.example")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "relay-alias")
    path = tmp_path / "config.yaml"
    path.write_text("evidence:\n  max_per_source: 1\n", encoding="utf-8")

    config = load_app_config(path)

    assert config.evidence.max_per_source == 1


def test_default_evidence_settings_preserve_sentence_bm25_v1() -> None:
    settings = EvidenceSettings()

    assert settings.retrieval_mode == "sentence_bm25_v1"
    assert "acquisition_weight" not in settings.model_dump(mode="json")


def test_hybrid_retrieval_requires_complete_model_identity() -> None:
    with pytest.raises(ValidationError, match="dense model identity"):
        EvidenceSettings(retrieval_mode="source_hybrid_v2")


def test_hybrid_retrieval_accepts_bounded_complete_settings() -> None:
    settings = EvidenceSettings(
        retrieval_mode="source_hybrid_v2",
        source_candidate_k=64,
        passages_per_source=4,
        dense_candidate_k=256,
        final_per_source=1,
        dense_model_id="BAAI/bge-small-en-v1.5",
        dense_model_revision="52398278842ec682c6f32300af41344b1c0b0bb2",
        dense_model_receipt_sha256="a" * 64,
        lexical_weight=0.2,
        source_weight=0.1,
        dense_weight=0.4,
        acquisition_weight=0.3,
    )

    assert settings.source_candidate_k * settings.passages_per_source == 256
    assert settings.acquisition_weight == 0.3


def test_hybrid_retrieval_rejects_four_weight_sum_mismatch() -> None:
    with pytest.raises(ValidationError, match="retrieval weights"):
        EvidenceSettings(
            retrieval_mode="source_hybrid_v2",
            dense_model_id="BAAI/bge-small-en-v1.5",
            dense_model_revision="52398278842ec682c6f32300af41344b1c0b0bb2",
            dense_model_receipt_sha256="a" * 64,
            lexical_weight=0.2,
            source_weight=0.1,
            dense_weight=0.7,
            acquisition_weight=0.3,
        )


def test_hybrid_candidate_cap_cannot_exceed_source_stage_capacity() -> None:
    with pytest.raises(ValidationError, match="dense_candidate_k"):
        EvidenceSettings(
            retrieval_mode="source_hybrid_v2",
            source_candidate_k=2,
            passages_per_source=2,
            dense_candidate_k=5,
            final_per_source=1,
            dense_model_id="BAAI/bge-small-en-v1.5",
            dense_model_revision="52398278842ec682c6f32300af41344b1c0b0bb2",
            dense_model_receipt_sha256="a" * 64,
            lexical_weight=0.2,
            source_weight=0.1,
            dense_weight=0.7,
        )
