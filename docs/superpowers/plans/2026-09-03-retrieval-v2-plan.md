# EvidenceRoute Retrieval v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the opt-in retrieval path with source-diverse candidate generation and local dense reranking, proving offline top-8 gold-source recall improves from 5/27 to at least 14/27 before any paid campaign.

**Architecture:** Preserve `AveritecFrozenProvider` as the byte-compatible v1 baseline. Add a separate hybrid provider that groups sentence BM25 results by canonical source, reranks a bounded passage set through an injected local encoder, and returns the existing `Evidence` contract. Keep model assets and calibration gold outside runtime imports; bind the selected mode, model receipt, config, and dependency lock into new campaign identity.

**Tech Stack:** Python 3.11, Pydantic, rank-bm25, NumPy, FastEmbed 0.8.0, ONNX Runtime, Typer, pytest.

---

## File Map

- Modify `src/evidence_route/config.py`: strict retrieval v2 settings.
- Modify `src/evidence_route/providers/averitec.py`: expose immutable corpus record/index data without changing v1 ranking.
- Create `src/evidence_route/providers/averitec_v2.py`: source grouping, candidate generation, score fusion, final diversification.
- Create `src/evidence_route/providers/dense.py`: encoder protocol, FastEmbed adapter, local receipt verification.
- Create `src/evidence_route/evaluation/retrieval_diagnostics.py`: scorer-only recall calculation and resumable deterministic output.
- Modify `src/evidence_route/evaluation/activity.py` and `lifecycle.py`: optional retrieval-model receipt freeze field.
- Modify `src/evidence_route/cli.py`, `production.py`, `production_calibration.py`, and `production_evaluation.py`: provider factory and CLI wiring.
- Create `scripts/prepare_retrieval_model.py`: explicit model download and hash receipt generation.
- Create `configs/retrieval-v2.yaml`: opt-in settings; existing configs stay on v1.
- Modify `pyproject.toml` and regenerate `requirements.lock`: pinned retrieval dependencies.
- Add focused tests under `tests/test_config.py`, `tests/test_averitec_provider.py`, `tests/test_retrieval_v2.py`, `tests/test_retrieval_model.py`, `tests/test_retrieval_diagnostics.py`, `tests/test_lifecycle.py`, `tests/test_production_services.py`, and `tests/test_reporting.py`.

### Task 1: Strict versioned retrieval configuration

**Files:**
- Modify: `src/evidence_route/config.py`
- Create: `configs/retrieval-v2.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Add tests proving default and historical configs resolve to v1 without editing their bytes, while v2
requires internally consistent positive limits, a model ID, a pinned revision, a 64-character receipt
hash, and weights whose sum is one.

```python
def test_default_config_preserves_sentence_bm25_v1(monkeypatch) -> None:
    _set_llm_env(monkeypatch)
    config = load_app_config(Path("configs/default.yaml"))
    assert config.evidence.retrieval_mode == "sentence_bm25_v1"


def test_hybrid_retrieval_settings_require_a_complete_identity() -> None:
    with pytest.raises(ValidationError, match="receipt"):
        EvidenceSettings(retrieval_mode="source_hybrid_v2")


def test_hybrid_retrieval_limits_and_weights_are_consistent() -> None:
    settings = EvidenceSettings(
        retrieval_mode="source_hybrid_v2",
        source_candidate_k=64,
        passages_per_source=4,
        dense_candidate_k=256,
        final_per_source=2,
        dense_model_id="BAAI/bge-small-en-v1.5",
        dense_model_revision="52398278842ec682c6f32300af41344b1c0b0bb2",
        dense_model_receipt_sha256="a" * 64,
        lexical_weight=0.25,
        source_weight=0.10,
        dense_weight=0.65,
    )
    assert settings.source_candidate_k * settings.passages_per_source == 256


def test_hybrid_candidate_cap_cannot_exceed_source_stage_capacity() -> None:
    with pytest.raises(ValidationError, match="dense_candidate_k"):
        EvidenceSettings(
            retrieval_mode="source_hybrid_v2",
            source_candidate_k=2,
            passages_per_source=2,
            dense_candidate_k=5,
            final_per_source=2,
            dense_model_id="BAAI/bge-small-en-v1.5",
            dense_model_revision="52398278842ec682c6f32300af41344b1c0b0bb2",
            dense_model_receipt_sha256="a" * 64,
            lexical_weight=0.25,
            source_weight=0.10,
            dense_weight=0.65,
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_config.py -q -p no:cacheprovider
```

Expected: FAIL because the retrieval fields and cross-field validator do not exist.

- [ ] **Step 3: Add the strict settings**

Extend `EvidenceSettings` with explicit v1 defaults and a model validator. V1 must reject stray dense identity fields; v2 must require them.

```python
class EvidenceSettings(ConfigModel):
    retrieval_mode: Literal["sentence_bm25_v1", "source_hybrid_v2"] = "sentence_bm25_v1"
    source_candidate_k: int = Field(default=64, gt=0)
    passages_per_source: int = Field(default=4, gt=0)
    dense_candidate_k: int = Field(default=256, gt=0)
    final_per_source: int = Field(default=2, gt=0)
    dense_model_id: str | None = None
    dense_model_revision: str | None = None
    dense_model_receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    lexical_weight: float = Field(default=0.25, ge=0, le=1)
    source_weight: float = Field(default=0.10, ge=0, le=1)
    dense_weight: float = Field(default=0.65, ge=0, le=1)

    @model_validator(mode="after")
    def validate_retrieval(self) -> EvidenceSettings:
        if self.dense_candidate_k > self.source_candidate_k * self.passages_per_source:
            raise ValueError("dense_candidate_k cannot exceed source-stage capacity")
        if self.final_per_source > self.passages_per_source:
            raise ValueError("final_per_source cannot exceed passages_per_source")
        if not math.isclose(
            self.lexical_weight + self.source_weight + self.dense_weight,
            1.0,
        ):
            raise ValueError("retrieval weights must sum to one")
        identity = (
            self.dense_model_id,
            self.dense_model_revision,
            self.dense_model_receipt_sha256,
        )
        if self.retrieval_mode == "source_hybrid_v2" and not all(identity):
            raise ValueError("hybrid retrieval requires dense model identity and receipt")
        if self.retrieval_mode == "sentence_bm25_v1" and any(identity):
            raise ValueError("v1 retrieval cannot declare a dense model identity")
        return self
```

Keep all current evidence limits and do not edit `default.yaml` or historical experiment configs, so their
raw-byte config hashes remain unchanged. Copy the current quality-recovery behavior into
`retrieval-v2.yaml` and add the complete v2 retrieval block.

- [ ] **Step 4: Verify GREEN and compatibility**

Run:

```powershell
python -m pytest tests/test_config.py tests/test_cli.py -q -p no:cacheprovider
```

Expected: PASS; existing config hashes change only for files intentionally edited.

- [ ] **Step 5: Commit**

```powershell
git add src/evidence_route/config.py configs/retrieval-v2.yaml tests/test_config.py
git commit -m "feat: add versioned retrieval settings"
```

### Task 2: Source-diverse lexical candidates

**Files:**
- Modify: `src/evidence_route/providers/averitec.py`
- Create: `src/evidence_route/providers/averitec_v2.py`
- Test: `tests/test_averitec_provider.py`
- Create: `tests/test_retrieval_v2.py`

- [ ] **Step 1: Add failing v1 compatibility and source-diversity tests**

Use a local corpus where the highest four sentence scores come from one URL and a lower-ranked passage
from another URL contains the decisive phrase.

```python
@pytest.mark.asyncio
async def test_v1_ranked_ids_remain_unchanged() -> None:
    provider = AveritecFrozenProvider(Path("tests/fixtures/averitec/corpora"))
    result = await provider.search("dev-1", "Alpha beta claim", top_k=2, max_chars=20)
    assert [item.evidence_id for item in result] == ["ev-1", "ev-2"]


def test_source_candidates_are_diverse_and_stably_ordered(tmp_path: Path) -> None:
    index = load_frozen_index(_write_repetitive_corpus(tmp_path))
    candidates = source_candidates(
        index,
        query="claim decisive phrase",
        source_candidate_k=2,
        passages_per_source=2,
        dense_candidate_k=4,
    )
    assert {item.source_key for item in candidates} == {"https://a.test/", "https://b.test/"}
    assert candidates == source_candidates(index, query="claim decisive phrase", source_candidate_k=2, passages_per_source=2, dense_candidate_k=4)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_averitec_provider.py tests/test_retrieval_v2.py -q -p no:cacheprovider
```

Expected: v1 assertions pass; imports for `load_frozen_index` and `source_candidates` fail.

- [ ] **Step 3: Expose immutable v1 index data and implement source grouping**

Rename private `_Record` and `_Index` to `FrozenEvidenceRecord` and `FrozenClaimIndex`, keeping the exact
v1 BM25 score and tie key. Add a pure candidate function in `averitec_v2.py`:

```python
@dataclass(frozen=True)
class RetrievalCandidate:
    record: FrozenEvidenceRecord
    source_key: str
    lexical_score: float
    source_rank: int


def source_candidates(
    index: FrozenClaimIndex,
    query: str,
    *,
    source_candidate_k: int,
    passages_per_source: int,
    dense_candidate_k: int,
) -> list[RetrievalCandidate]:
    scores = index.bm25.get_scores(tokenize(query))
    by_source: dict[str, list[tuple[FrozenEvidenceRecord, float]]] = defaultdict(list)
    for record, score in zip(index.records, scores, strict=True):
        by_source[canonicalize_citation_url(record.source_url)].append((record, float(score)))

    def sentence_key(item: tuple[FrozenEvidenceRecord, float]) -> tuple[float, str, str, str]:
        record, score = item
        return (
            -score,
            record.evidence_id,
            canonicalize_citation_url(record.source_url),
            record.snapshot_sha256,
        )

    def source_key(
        item: tuple[str, list[tuple[FrozenEvidenceRecord, float]]],
    ) -> tuple[float, str, str, str]:
        canonical_url, rows = item
        best_record, best_score = min(rows, key=sentence_key)
        return (-best_score, best_record.evidence_id, canonical_url, best_record.snapshot_sha256)

    ranked_sources = sorted(
        by_source.items(),
        key=source_key,
    )[:source_candidate_k]
    candidates: list[RetrievalCandidate] = []
    for source_rank, (canonical_url, rows) in enumerate(ranked_sources):
        for record, score in sorted(rows, key=sentence_key)[:passages_per_source]:
            candidates.append(RetrievalCandidate(record, canonical_url, score, source_rank))
    return candidates[:dense_candidate_k]
```

- [ ] **Step 4: Verify v1 and source candidate tests**

```powershell
python -m pytest tests/test_averitec_provider.py tests/test_evidence_ordering.py tests/test_retrieval_v2.py -q -p no:cacheprovider
```

Expected: PASS with unchanged v1 evidence IDs, scores, truncation, and cache count.

- [ ] **Step 5: Commit**

```powershell
git add src/evidence_route/providers/averitec.py src/evidence_route/providers/averitec_v2.py tests/test_averitec_provider.py tests/test_retrieval_v2.py
git commit -m "feat: generate source-diverse evidence candidates"
```

### Task 3: Dense encoder protocol and deterministic hybrid ranking

**Files:**
- Create: `src/evidence_route/providers/dense.py`
- Modify: `src/evidence_route/providers/averitec_v2.py`
- Test: `tests/test_retrieval_v2.py`
- Create: `tests/test_retrieval_model.py`

- [ ] **Step 1: Write failing fusion and failure tests**

```python
class FakeEncoder:
    model_id = "fixture-dense"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [1.0 if "decisive" in passage else 0.0 for passage in passages]


@pytest.mark.asyncio
async def test_hybrid_reranker_promotes_semantic_passage_and_caps_each_source(tmp_path: Path) -> None:
    provider = AveritecHybridProvider(
        tmp_path,
        settings=_hybrid_settings(),
        encoder=FakeEncoder(),
    )
    result = await provider.search("claim-1", "paraphrased query", top_k=3, max_chars=80)
    assert result[0].evidence_id == "decisive-b"
    assert Counter(str(item.source_url) for item in result).most_common(1)[0][1] <= 2


@pytest.mark.asyncio
async def test_encoder_failure_is_not_silently_replaced_by_v1(tmp_path: Path) -> None:
    provider = AveritecHybridProvider(tmp_path, settings=_hybrid_settings(), encoder=FailingEncoder())
    with pytest.raises(RetrievalModelError, match="dense reranking failed"):
        await provider.search("claim-1", "query", top_k=3, max_chars=80)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_retrieval_v2.py tests/test_retrieval_model.py -q -p no:cacheprovider
```

Expected: FAIL because the encoder protocol, hybrid provider, and error type do not exist.

- [ ] **Step 3: Implement protocol, normalization, fusion, and final diversification**

```python
class DenseEncoder(Protocol):
    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        raise NotImplementedError


def _min_max(values: Sequence[float]) -> list[float]:
    low, high = min(values), max(values)
    return [0.0 for _ in values] if math.isclose(low, high) else [(value - low) / (high - low) for value in values]


def _source_signal(candidates: Sequence[RetrievalCandidate]) -> list[float]:
    return _min_max([-float(item.source_rank) for item in candidates])


def fuse_candidates(
    candidates,
    dense_scores,
    *,
    lexical_weight,
    source_weight,
    dense_weight,
    top_k,
    final_per_source,
):
    lexical = _min_max([item.lexical_score for item in candidates])
    source = _source_signal(candidates)
    ranked = sorted(
        zip(candidates, lexical, source, dense_scores, strict=True),
        key=lambda item: (
            -(lexical_weight * item[1] + source_weight * item[2] + dense_weight * item[3]),
            item[0].source_rank,
            item[0].record.evidence_id,
            item[0].source_key,
            item[0].record.snapshot_sha256,
        ),
    )
    return _take_with_source_cap(ranked, top_k=top_k, per_source=final_per_source)
```

Validate score count, finite float values, non-empty query, and all limits before ranking. Pass each
candidate to the encoder as `f"{record.title}\n\n{record.text}"`. Build `Evidence` with the original text,
URL, ID, snapshot hash, truncation offsets, and fused ranking score. Add focused tests for empty queries,
non-finite or wrong-length dense scores, deterministic ties, final per-source caps, and cache eviction.

- [ ] **Step 4: Verify GREEN**

```powershell
python -m pytest tests/test_retrieval_v2.py tests/test_retrieval_model.py -q -p no:cacheprovider
```

Expected: PASS, including deterministic ties and explicit encoder failure.

- [ ] **Step 5: Commit**

```powershell
git add src/evidence_route/providers/dense.py src/evidence_route/providers/averitec_v2.py tests/test_retrieval_v2.py tests/test_retrieval_model.py
git commit -m "feat: add deterministic hybrid evidence reranking"
```

### Task 4: Pinned local FastEmbed model assets

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.lock`
- Create: `scripts/prepare_retrieval_model.py`
- Create after verified download: `data/model_manifests/bge-small-en-v1.5.json`
- Modify: `src/evidence_route/providers/dense.py`
- Test: `tests/test_retrieval_model.py`

- [ ] **Step 1: Add failing receipt tests**

```python
def test_model_receipt_rejects_tampered_asset(tmp_path: Path) -> None:
    root, receipt = _write_model_fixture(tmp_path)
    (root / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(RetrievalModelError, match="hash mismatch"):
        verify_model_receipt(root, receipt)


def test_fastembed_adapter_requires_local_verified_files(tmp_path: Path) -> None:
    with pytest.raises(RetrievalModelError, match="receipt"):
        FastEmbedEncoder.from_local(tmp_path, tmp_path / "missing.json")
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_retrieval_model.py -q -p no:cacheprovider
```

Expected: FAIL because receipt verification and `FastEmbedEncoder` are absent.

- [ ] **Step 3: Pin dependencies and implement receipt validation**

Add `retrieval = ["fastembed==0.8.0"]` to optional dependencies and regenerate all enabled extras:

```powershell
python -m piptools compile --extra=dev --extra=eval --extra=retrieval --index-url=https://pypi.org/simple --no-emit-index-url --no-emit-trusted-host --output-file=requirements.lock pyproject.toml
python -m pip install -r requirements.lock
```

The receipt schema is strict and records `model_id`, `hf_repo_id`, `model_revision`,
`fastembed_version`, sorted relative file hashes, and a deterministic tree hash. `verify_model_receipt`
rejects path traversal, missing/extra required files, package-version drift, and hash mismatch before
constructing FastEmbed.

- [ ] **Step 4: Implement the explicit preparation script**

`prepare_retrieval_model.py` accepts `--model-root`, `--receipt`, the fixed model ID, Hugging Face source
repository `qdrant/bge-small-en-v1.5-onnx-q`, and immutable revision
`52398278842ec682c6f32300af41344b1c0b0bb2`. The preparation command calls
`huggingface_hub.snapshot_download` with that revision, then constructs FastEmbed with
`specific_model_path=<model-root>` and runs one probe embedding. It hashes all regular model files and
atomically writes the receipt. Runtime `FastEmbedEncoder.from_local` passes both
`specific_model_path=<model-root>` and `local_files_only=True`; it never invokes an implicit download.

```powershell
python scripts/prepare_retrieval_model.py --model-root data/external/retrieval-models/bge-small-en-v1.5 --receipt data/model_manifests/bge-small-en-v1.5.json --model-id BAAI/bge-small-en-v1.5 --hf-repo-id qdrant/bge-small-en-v1.5-onnx-q --model-revision 52398278842ec682c6f32300af41344b1c0b0bb2
```

Expected: receipt written, probe vector finite and non-empty, all listed files remain under model root.

- [ ] **Step 5: Run receipt tests and dependency checks**

```powershell
python -m pytest tests/test_retrieval_model.py -q -p no:cacheprovider
python -m pip check
```

Expected: PASS and `No broken requirements found.`

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml requirements.lock scripts/prepare_retrieval_model.py data/model_manifests/bge-small-en-v1.5.json src/evidence_route/providers/dense.py tests/test_retrieval_model.py
git commit -m "feat: pin local retrieval model assets"
```

### Task 5: Runtime provider factory and fail-closed wiring

**Files:**
- Create: `src/evidence_route/providers/factory.py`
- Modify: `src/evidence_route/cli.py`
- Modify: `src/evidence_route/evaluation/production.py`
- Modify: `src/evidence_route/evaluation/production_calibration.py`
- Modify: `src/evidence_route/evaluation/production_evaluation.py`
- Test: `tests/test_production_services.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing provider-selection tests**

```python
def test_provider_factory_preserves_v1_without_model_root() -> None:
    provider = build_evidence_provider(Path("corpora"), EvidenceSettings())
    assert isinstance(provider, AveritecFrozenProvider)


def test_v2_requires_explicit_local_model_root(monkeypatch) -> None:
    monkeypatch.delenv("EVIDENCE_ROUTE_RETRIEVAL_MODEL_ROOT", raising=False)
    monkeypatch.delenv("EVIDENCE_ROUTE_RETRIEVAL_MODEL_RECEIPT", raising=False)
    with pytest.raises(ValueError, match="EVIDENCE_ROUTE_RETRIEVAL_MODEL_ROOT"):
        build_evidence_provider(Path("corpora"), _hybrid_settings())


def test_v2_requires_explicit_model_receipt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVIDENCE_ROUTE_RETRIEVAL_MODEL_ROOT", str(tmp_path))
    monkeypatch.delenv("EVIDENCE_ROUTE_RETRIEVAL_MODEL_RECEIPT", raising=False)
    with pytest.raises(ValueError, match="EVIDENCE_ROUTE_RETRIEVAL_MODEL_RECEIPT"):
        build_evidence_provider(Path("corpora"), _hybrid_settings())
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_production_services.py tests/test_cli.py -q -p no:cacheprovider -k "retrieval or provider_factory"
```

Expected: FAIL because no central provider factory exists.

- [ ] **Step 3: Implement and use one provider factory**

```python
def build_evidence_provider(corpus_dir: Path, settings: EvidenceSettings) -> EvidenceProvider:
    if settings.retrieval_mode == "sentence_bm25_v1":
        return AveritecFrozenProvider(corpus_dir)
    model_root = os.environ.get("EVIDENCE_ROUTE_RETRIEVAL_MODEL_ROOT")
    if not model_root:
        raise ValueError("EVIDENCE_ROUTE_RETRIEVAL_MODEL_ROOT is required for source_hybrid_v2")
    receipt = os.environ.get("EVIDENCE_ROUTE_RETRIEVAL_MODEL_RECEIPT")
    if not receipt:
        raise ValueError("EVIDENCE_ROUTE_RETRIEVAL_MODEL_RECEIPT is required for source_hybrid_v2")
    encoder = FastEmbedEncoder.from_local(
        Path(model_root),
        Path(receipt),
        expected_receipt_sha256=settings.dense_model_receipt_sha256,
    )
    return AveritecHybridProvider(corpus_dir, settings=settings, encoder=encoder)
```

Replace every direct production construction of `AveritecFrozenProvider` with the factory. Build and
verify the provider before constructing paid transport so missing/tampered local assets cannot spend.

- [ ] **Step 4: Verify all production entry points**

```powershell
python -m pytest tests/test_cli.py tests/test_production_services.py tests/test_production_orchestration.py -q -p no:cacheprovider
```

Expected: PASS; v1 tests need no new environment variable, v2 asset failures occur before transport.

- [ ] **Step 5: Commit**

```powershell
git add src/evidence_route/providers/factory.py src/evidence_route/cli.py src/evidence_route/evaluation/production.py src/evidence_route/evaluation/production_calibration.py src/evidence_route/evaluation/production_evaluation.py tests/test_cli.py tests/test_production_services.py
git commit -m "feat: wire retrieval v2 into production services"
```

### Task 6: Scorer-only resumable retrieval diagnostics

**Files:**
- Create: `src/evidence_route/evaluation/retrieval_diagnostics.py`
- Modify: `src/evidence_route/cli.py`
- Create: `tests/test_retrieval_diagnostics.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing scorer-isolation and resume tests**

```python
def test_runtime_provider_does_not_import_scorer_manifest() -> None:
    source = Path("src/evidence_route/providers/averitec_v2.py").read_text(encoding="utf-8")
    assert "scorer_manifest" not in source


@pytest.mark.asyncio
async def test_diagnostic_resumes_completed_claims_without_retrieval(tmp_path: Path) -> None:
    retriever = RecordingRetriever()
    progress = _write_progress(tmp_path, completed_claim_id="train-1")
    report = await build_retrieval_diagnostic(
        _fixture_inputs(tmp_path),
        retriever,
        progress_path=progress,
    )
    assert "train-1" not in retriever.claim_ids
    assert report.total_claims == 2


@pytest.mark.asyncio
async def test_timing_is_not_part_of_deterministic_report_hash(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    ranked_ids = {"train-1": ["evidence-1"], "train-2": ["evidence-2"]}
    first = await build_retrieval_diagnostic(
        inputs,
        StaticRetriever(ranked_ids),
        progress_path=tmp_path / "first-progress.json",
        clock=iter([0.0, 1.0, 1.0, 2.0]).__next__,
    )
    second = await build_retrieval_diagnostic(
        inputs,
        StaticRetriever(ranked_ids),
        progress_path=tmp_path / "second-progress.json",
        clock=iter([0.0, 9.0, 9.0, 18.0]).__next__,
    )
    assert first.deterministic_payload() == second.deterministic_payload()
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_retrieval_diagnostics.py tests/test_cli.py -q -p no:cacheprovider -k retrieval
```

Expected: FAIL because diagnostic models and CLI command do not exist.

- [ ] **Step 3: Implement strict models and source identity**

Create strict `RetrievalObservation`, `RetrievalDiagnosticInputs`, `RetrievalDiagnosticItem`,
`RetrievalDiagnosticSummary`, and progress models. `RetrievalObservation` carries ordered source-stage
and final evidence IDs plus their original URLs, allowing the scorer to evaluate both gates without
giving gold data to the retriever. Define one async `DiagnosticRetriever.retrieve(claim_id, query)`
protocol used by the fake and runtime adapters, matching the awaited tests above. Source
identity canonicalizes HTTP URLs, decodes HTML entities, and treats a Wayback URL as equivalent to its
embedded original URL. Eligibility and hits use source identity only; labels and justification never
enter retrieval queries.

- [ ] **Step 4: Implement atomic claim-boundary resume and CLI**

Add `evidence-route diagnose-retrieval` with explicit runtime manifest, gold manifest, corpus directory,
config, progress, output, and timing-output paths. It verifies sidecars and config/model receipts, skips
completed claim IDs only when all input identities match, and atomically persists after each claim.

```powershell
python -m evidence_route.cli diagnose-retrieval --runtime-manifest data/manifests/averitec_calibration_runtime.json --gold-manifest data/scorer_manifests/averitec_calibration_gold.json --corpus-dir data/processed/averitec/corpora --config configs/retrieval-v2.yaml --progress artifacts/retrieval-v2-calibration/progress.json --output reports/retrieval-v2-calibration/diagnostic.json --timing-output reports/retrieval-v2-calibration/timing.json
```

- [ ] **Step 5: Verify diagnostics**

```powershell
python -m pytest tests/test_retrieval_diagnostics.py tests/test_cli.py -q -p no:cacheprovider -k retrieval
```

Expected: PASS; fake retriever performs zero work for completed matching progress and rejects stale input
hashes.

- [ ] **Step 6: Commit**

```powershell
git add src/evidence_route/evaluation/retrieval_diagnostics.py src/evidence_route/cli.py tests/test_retrieval_diagnostics.py tests/test_cli.py
git commit -m "feat: add resumable retrieval diagnostics"
```

### Task 7: Bind retrieval assets into campaign freeze and reports

**Files:**
- Modify: `src/evidence_route/evaluation/activity.py`
- Modify: `src/evidence_route/evaluation/lifecycle.py`
- Modify: `src/evidence_route/evaluation/calibration.py`
- Modify: `src/evidence_route/evaluation/production_calibration.py`
- Modify: `src/evidence_route/evaluation/production_evaluation.py`
- Modify: `src/evidence_route/evaluation/reporting.py`
- Modify: `tests/fixtures/evaluation/campaign_factory.py`
- Modify: `tests/fixtures/evaluation/report_factory.py`
- Test: `tests/test_lifecycle.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write failing freeze and report-audit tests**

```python
def test_v2_freeze_detects_retrieval_receipt_drift(tmp_path: Path) -> None:
    files = _files(tmp_path)
    receipt = _write(tmp_path / "model-receipt.json", "original\n")
    kwargs = {
        "calibration_manifest": files["calibration"],
        "dev_manifest": files["dev"],
        "stability_manifest": files["stability"],
        "corpus_receipt": files["receipt"],
        "prompt_bundle": files["prompts"],
        "config_file": files["config"],
        "pricing_file": files["pricing"],
        "requirements_lock": files["requirements"],
        "endpoint_config": {"base_url": "https://relay.example/v1"},
        "requested_alias": "relay-model",
        "seed": 20260817,
        "manifest_freeze_git_sha": "a" * 40,
        "dev_protocol_git_sha": "b" * 40,
        "retrieval_model_receipt": receipt,
    }
    frozen = build_freeze_identity(**kwargs)
    receipt.write_text("changed\n", encoding="utf-8")
    with pytest.raises(FreezeMismatch, match="retrieval_model_receipt_sha256"):
        verify_current_freeze(frozen, **kwargs)


def test_report_blocks_v2_when_retrieval_receipt_is_missing(report_input_factory) -> None:
    report_input = report_input_factory.as_retrieval_v2(receipt_sha256="a" * 64)
    assert report_input.retrieval_model_receipt is not None
    report_input.retrieval_model_receipt.unlink()
    bundle = build_report_bundle(report_input, publish=False)
    assert "retrieval_model_receipt_missing" in bundle.publication_gate.reasons
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_lifecycle.py tests/test_reporting.py -q -p no:cacheprovider -k retrieval
```

Expected: FAIL because freeze and report input have no retrieval receipt field.

- [ ] **Step 3: Extend identity without invalidating v1 artifacts**

```python
class FreezeIdentity(StrictModel):
    # existing fields remain in their current order
    retrieval_model_receipt_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
        exclude_if=lambda value: value is None,
    )
```

V1 builds `None`; `exclude_if` keeps that field out of v1 JSON so historical fingerprints and serialized
plans remain byte-compatible. V2 requires the receipt digest to equal both the config declaration and
current receipt bytes. Resume comparison, calibration plan, experiment parent checks, report
reproducibility, and current freeze audit all include the new field. Add
`ReportInput.retrieval_model_receipt: Path | None`, and add the concrete
`ReportInputFactory.as_retrieval_v2(receipt_sha256: str)` fixture helper used above; it writes a receipt,
updates the activity and plan freeze consistently, persists their sidecars, and returns a v2 report input.

- [ ] **Step 4: Verify freeze and reporting tests**

```powershell
python -m pytest tests/test_lifecycle.py tests/test_reporting.py tests/test_calibration_orchestration.py tests/test_production_services.py -q -p no:cacheprovider
```

Expected: PASS for old v1 fixtures and strict failure for v2 receipt drift.

- [ ] **Step 5: Commit**

```powershell
git add src/evidence_route/evaluation/activity.py src/evidence_route/evaluation/lifecycle.py src/evidence_route/evaluation/calibration.py src/evidence_route/evaluation/production_calibration.py src/evidence_route/evaluation/production_evaluation.py src/evidence_route/evaluation/reporting.py tests/fixtures/evaluation/campaign_factory.py tests/fixtures/evaluation/report_factory.py tests/test_lifecycle.py tests/test_reporting.py
git commit -m "feat: freeze retrieval model identity"
```

### Task 8: Run offline ablations and enforce quality gates

**Files:**
- Modify if diagnostics expose defects: only retrieval-v2 files and their focused tests
- Create generated reports: `reports/retrieval-v2-calibration/`
- Modify: `docs/RETRIEVAL_DIAGNOSTIC_20260903.md`
- Modify: `README.md`
- Modify: `docs/RESUME_PROJECT.md`

- [ ] **Step 1: Verify source-only candidate recall before dense execution**

Run the diagnostic with dense scoring disabled only through its explicit `--ablation source-only` mode.

```powershell
python -m evidence_route.cli diagnose-retrieval --runtime-manifest data/manifests/averitec_calibration_runtime.json --gold-manifest data/scorer_manifests/averitec_calibration_gold.json --corpus-dir data/processed/averitec/corpora --config configs/retrieval-v2.yaml --ablation source-only --progress artifacts/retrieval-v2-source/progress.json --output reports/retrieval-v2-source/diagnostic.json --timing-output reports/retrieval-v2-source/timing.json
```

Expected: source candidate hit count at least 22 of 27. If it is lower, add one failing fixture reproducing
the observed ranking defect, fix only stage 1, rerun focused tests, and resume this command. Do not proceed
to dense model workarounds or provider calls while source recall is below the gate.

- [ ] **Step 2: Run the hybrid diagnostic**

```powershell
$env:EVIDENCE_ROUTE_RETRIEVAL_MODEL_ROOT = (Resolve-Path data/external/retrieval-models/bge-small-en-v1.5)
$env:EVIDENCE_ROUTE_RETRIEVAL_MODEL_RECEIPT = (Resolve-Path data/model_manifests/bge-small-en-v1.5.json)
python -m evidence_route.cli diagnose-retrieval --runtime-manifest data/manifests/averitec_calibration_runtime.json --gold-manifest data/scorer_manifests/averitec_calibration_gold.json --corpus-dir data/processed/averitec/corpora --config configs/retrieval-v2.yaml --progress artifacts/retrieval-v2-hybrid/progress.json --output reports/retrieval-v2-hybrid/diagnostic.json --timing-output reports/retrieval-v2-hybrid/timing.json
```

Expected: final top-8 hit count at least 14 of 27, at least one `train-2468` gold-source passage, zero
provider calls, and no stale progress reuse.

- [ ] **Step 3: Repeat for deterministic rankings**

Run the hybrid diagnostic into a second empty progress/output directory and compare deterministic JSON:

```powershell
python -m evidence_route.cli diagnose-retrieval --runtime-manifest data/manifests/averitec_calibration_runtime.json --gold-manifest data/scorer_manifests/averitec_calibration_gold.json --corpus-dir data/processed/averitec/corpora --config configs/retrieval-v2.yaml --progress artifacts/retrieval-v2-hybrid-repeat/progress.json --output reports/retrieval-v2-hybrid-repeat/diagnostic.json --timing-output reports/retrieval-v2-hybrid-repeat/timing.json
Compare-Object (Get-Content reports/retrieval-v2-hybrid/diagnostic.json) (Get-Content reports/retrieval-v2-hybrid-repeat/diagnostic.json)
```

Expected: `Compare-Object` emits no differences.

- [ ] **Step 4: Run complete repository verification**

```powershell
python -m pytest -m "not network and not live and not paid" -q -p no:cacheprovider
python -m ruff check src tests scripts
python -m compileall -q src tests scripts
python -m pip check
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 5: Update truthful project documentation**

Record v1, source-only, and hybrid recall with exact denominators, model identity, input hashes, receipt
hash, latency scope, and the fact that no paid quality result exists yet. Do not replace Gate A resume
metrics until a later full campaign passes publication gates.

- [ ] **Step 6: Commit**

```powershell
git add docs/RETRIEVAL_DIAGNOSTIC_20260903.md README.md docs/RESUME_PROJECT.md configs/retrieval-v2.yaml data/model_manifests/bge-small-en-v1.5.json requirements.lock
git commit -m "docs: record retrieval v2 offline gates"
```

### Task 9: Paid smoke readiness checkpoint

**Files:**
- No source changes expected
- Create only after separate paid authorization: new activity, experiment, and report directories

- [ ] **Step 1: Generate a no-transport budget preview**

Use the frozen v2 config, model receipt, manifests, pricing, and a new activity ID. Verify the command does
not construct transport without `--accept-paid-campaign`.

- [ ] **Step 2: Audit readiness**

Require all Task 8 gates, a clean frozen worktree, provider smoke identity consistency, sufficient client
cap, and zero unresolved billing uncertainty. The smoke campaign must not reuse any prior activity ID.

- [ ] **Step 3: Stop before payment**

Report the exact proposed activity ID, 10-item reservation, current cap, and estimated maximum charge.
A new paid run begins only after authorization that names this v2 campaign. Generic earlier authorization
does not cover a new campaign identity.
