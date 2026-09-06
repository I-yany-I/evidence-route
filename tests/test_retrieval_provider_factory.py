from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

from evidence_route import cli as cli_module
from evidence_route.cli import ProductionServices
from evidence_route.config import EvidenceSettings, load_app_config
from evidence_route.evaluation.production_evaluation import ProductionCampaignService
from evidence_route.providers.averitec import AveritecFrozenProvider
from evidence_route.providers.averitec_v2 import AveritecHybridProvider
from evidence_route.providers.dense import FastEmbedEncoder, RetrievalModelError
from evidence_route.retrieval import build_evidence_provider, configured_model_receipt_path

MODEL_REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"


def _hybrid_settings(receipt_sha256: str) -> EvidenceSettings:
    return EvidenceSettings(
        retrieval_mode="source_hybrid_v2",
        source_candidate_k=2,
        passages_per_source=2,
        dense_candidate_k=4,
        final_per_source=1,
        dense_model_id="BAAI/bge-small-en-v1.5",
        dense_model_revision=MODEL_REVISION,
        dense_model_receipt_sha256=receipt_sha256,
    )


def _model_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    model_root = tmp_path / "model"
    model_root.mkdir()
    model_file = model_root / "model.onnx"
    model_file.write_bytes(b"model")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "BAAI/bge-small-en-v1.5",
                "model_revision": MODEL_REVISION,
                "files": [
                    {
                        "path": "model.onnx",
                        "sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return model_root, receipt, hashlib.sha256(receipt.read_bytes()).hexdigest()


def test_provider_factory_keeps_v1_free_of_model_environment(tmp_path: Path) -> None:
    provider = build_evidence_provider(tmp_path, EvidenceSettings())

    assert isinstance(provider, AveritecFrozenProvider)
    assert configured_model_receipt_path(EvidenceSettings()) is None


def test_provider_factory_requires_explicit_v2_model_paths(tmp_path: Path) -> None:
    with pytest.raises(RetrievalModelError, match="model root and receipt"):
        build_evidence_provider(tmp_path, _hybrid_settings("a" * 64))


def test_configured_model_receipt_path_reads_v2_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root, receipt, receipt_hash = _model_fixture(tmp_path)
    monkeypatch.setenv("EVIDENCE_ROUTE_RETRIEVAL_MODEL_ROOT", str(model_root))
    monkeypatch.setenv("EVIDENCE_ROUTE_RETRIEVAL_MODEL_RECEIPT", str(receipt))

    assert configured_model_receipt_path(_hybrid_settings(receipt_hash)) == receipt


def test_provider_factory_rejects_receipt_hash_drift(tmp_path: Path) -> None:
    model_root, receipt, _ = _model_fixture(tmp_path)

    with pytest.raises(RetrievalModelError, match="receipt hash mismatch"):
        build_evidence_provider(
            tmp_path / "corpora",
            _hybrid_settings("a" * 64),
            model_root=model_root,
            model_receipt=receipt,
            encoder_factory=lambda *_: object(),
        )


def test_provider_factory_builds_v2_only_for_matching_identity(tmp_path: Path) -> None:
    model_root, receipt, receipt_hash = _model_fixture(tmp_path)

    class Encoder:
        model_id = "BAAI/bge-small-en-v1.5"

        def score(self, query: str, passages: list[str]) -> list[float]:
            return [0.0 for _ in passages]

    provider = build_evidence_provider(
        tmp_path / "corpora",
        _hybrid_settings(receipt_hash),
        model_root=model_root,
        model_receipt=receipt,
        encoder_factory=lambda *_: Encoder(),
    )

    assert isinstance(provider, AveritecHybridProvider)
    assert provider.encoder.model_id == "BAAI/bge-small-en-v1.5"


def test_fastembed_encoder_scores_passages_in_bounded_batches() -> None:
    class Model:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def embed(self, values, *, batch_size: int):
            self.calls.append((len(values), batch_size))
            return [[float(index)] for index, _ in enumerate(values)]

    model = Model()
    encoder = FastEmbedEncoder(model, model_id="test")

    scores = encoder.score("query", [f"passage-{index}" for index in range(65)])

    assert len(scores) == 65
    assert model.calls == [(66, 1)]


def test_fastembed_encoder_serializes_shared_model_inference() -> None:
    class Model:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.guard = Lock()

        def embed(self, values, *, batch_size: int):
            assert batch_size == 1
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            with self.guard:
                self.active -= 1
            return [[float(index)] for index, _ in enumerate(values)]

    model = Model()
    encoder = FastEmbedEncoder(model, model_id="test")

    with ThreadPoolExecutor(max_workers=2) as pool:
        scores = list(pool.map(lambda _: encoder.score("query", ["passage"]), range(2)))

    assert scores == [[0.0], [0.0]]
    assert model.max_active == 1


def test_fastembed_encoder_skips_inference_for_empty_passages() -> None:
    class Model:
        def embed(self, values, *, batch_size: int):
            raise AssertionError("empty passage set must not invoke the model")

    encoder = FastEmbedEncoder(Model(), model_id="test")

    assert encoder.score("query", []) == []


def test_single_verify_uses_versioned_provider_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "test-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "relay-model")
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "provider: fixture\ncurrency: CNY\ninput_per_million: 1\n"
        "output_per_million: 1\nprice_source: fixture\nstrict_evaluation: true\n",
        encoding="utf-8",
    )

    def factory(corpus_dir: Path, settings: EvidenceSettings) -> object:
        del corpus_dir, settings
        raise RuntimeError("provider factory called")

    monkeypatch.setattr(cli_module, "build_evidence_provider", factory)
    with pytest.raises(RuntimeError, match="provider factory called"):
        ProductionServices().verify(
            run_id="run",
            claim_id="dev-0",
            claim="claim",
            strategy="always_single",
            resume=False,
            artifact_dir=tmp_path / "artifacts",
            config_path=Path("configs/default.yaml"),
            corpus_dir=tmp_path / "corpora",
            checkpoint_db=tmp_path / "checkpoints.sqlite3",
            pricing_path=pricing,
        )


def test_campaign_freeze_binds_configured_retrieval_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVIDENCE_ROUTE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EVIDENCE_ROUTE_API_KEY", "test-key")
    monkeypatch.setenv("EVIDENCE_ROUTE_MODEL", "relay-model")
    monkeypatch.setenv(
        "EVIDENCE_ROUTE_RETRIEVAL_MODEL_ROOT",
        str(Path("data/external/retrieval-models/bge-small-en-v1.5").resolve()),
    )
    receipt = Path("data/model_manifests/bge-small-en-v1.5.json").resolve()
    monkeypatch.setenv("EVIDENCE_ROUTE_RETRIEVAL_MODEL_RECEIPT", str(receipt))
    corpus = tmp_path / "processed" / "corpora"
    corpus.mkdir(parents=True)
    (corpus.parent / "preparation_receipt.json").write_text("{}\n", encoding="utf-8")
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text("currency: CNY\n", encoding="utf-8")
    config_path = Path("configs/retrieval-v2.yaml").resolve()
    config = load_app_config(config_path)
    plan = SimpleNamespace(
        runtime_manifest_sha256="a" * 64,
        seed=20260817,
        manifest_freeze_git_sha="b" * 40,
    )

    identity = ProductionCampaignService(repository_root=Path.cwd())._build_freeze(
        calibration_plan=plan,
        dev_manifest="c" * 64,
        stability_manifest="d" * 64,
        config_path=config_path,
        pricing_path=pricing,
        corpus_dir=corpus,
        app_config=config,
    )

    assert identity.retrieval_model_receipt_sha256 == hashlib.sha256(
        receipt.read_bytes()
    ).hexdigest()
