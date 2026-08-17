# EvidenceRoute Gate A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 `agent-collab` 原仓库重构为可复跑、可审计、可对比 single/multi/adaptive 三种策略的 EvidenceRoute Gate A 事实核查 Agent。

**Architecture:** 新包以 Pydantic contracts 为边界，LangGraph 负责有界分支、升级、并行 worker 和 checkpoint；`AveritecFrozenProvider` 在无 gold 字段的冻结语料上执行 BM25。评测侧与运行侧物理分离，完整 manifest 指标惩罚失败，官方 evaluator 只接收合法 completed 输出。

**Tech Stack:** Python 3.11、LangGraph 1.2.11、Pydantic 2.13.4、OpenAI Python SDK 3.1.0、SQLite checkpoint/transactional run store、rank-bm25、Typer、NumPy、scikit-learn、SciPy、NLTK、pytest、Ruff。

---

## Scope And File Map

本计划只实现设计规格中的 Gate A。MCP、中文案例和 Streamlit 必须在 Gate A 完成后另写计划。

**Runtime package**

- `src/evidence_route/contracts.py`: 枚举、Pydantic 输入输出和 LangGraph state。
- `src/evidence_route/config.py`: YAML/env 配置、哈希、密钥脱敏。
- `src/evidence_route/analyzer.py`: 中英文 claim units 与确定性特征。
- `src/evidence_route/providers/base.py`: `EvidenceProvider` 接口。
- `src/evidence_route/providers/averitec.py`: 冻结语料 BM25 provider。
- `src/evidence_route/budget.py`: usage、价格与整数 micro-CNY 预算算法/测试替身。
- `src/evidence_route/artifacts.py`: JSONL trace、原子 JSON、付费调用与预算共用的 SQLite run store。
- `src/evidence_route/llm.py`: OpenAI-compatible 结构化调用、repair、retry 和身份记录。
- `src/evidence_route/prompts.py`: 所有版本化 system/user prompt。
- `src/evidence_route/routing.py`: clear-multi、clear-single、LLM uncertain、fallback。
- `src/evidence_route/verification.py`: single、decomposer、worker、judge。
- `src/evidence_route/validation.py`: verdict/citation/coverage 验证与升级判断。
- `src/evidence_route/graph.py`: LangGraph 节点、`Send` 并行和 checkpoint。
- `src/evidence_route/cli.py`: `verify`、`evaluate`、`calibrate`、`report`、`provider-smoke`。

**Evaluation package**

- `src/evidence_route/evaluation/runtime_manifest.py`: claim-only runtime 清单；运行进程只导入此模块。
- `src/evidence_route/evaluation/scorer_manifest.py`: scorer-only gold 清单与运行结果对齐。
- `src/evidence_route/evaluation/metrics.py`: 全 manifest、条件指标、bootstrap、Wilson。
- `src/evidence_route/evaluation/official.py`: AVeriTeC completed-output 适配器。
- `src/evidence_route/evaluation/calibration.py`: 保存结果上的 policy replay。
- `src/evidence_route/evaluation/activity.py`: 冻结身份、严格 campaign/artifact/activity 模型。
- `src/evidence_route/evaluation/runner.py`: 交错策略执行、恢复、预算与漂移中止。
- `src/evidence_route/evaluation/reporting.py`: JSON/Markdown/简历片段。
- `scripts/prepare_averitec.py`: 固定 revision 下载、校验、抽样与语料归一化。
- `third_party/averitec/paper/{eval.py,utils.py}`: 固定字节的 2023 paper evaluator。
- `third_party/averitec/shared_task/evaluate_veracity.py`: 固定字节的 2024 shared-task evaluator。

**Configuration and outputs**

- `configs/default.yaml`: 路由、证据、节点 token 和预算上限。
- `configs/pricing.example.yaml`: 非密钥价格配置 schema 示例，禁用严格付费运行。
- `data/manifests/`: 仅提交 claim-only runtime 清单与哈希，不提交大语料。
- `data/scorer_manifests/`: 提交隔离的 gold/scorer 清单与哈希，运行图不可导入。
- `reports/final/`: 只在完整活动通过后提交报告、代表 trace 和简历片段。
- `tests/fixtures/`: 小型、无网络、无 gold 泄漏 fixture。

### Task 1: Create The New Distribution And Lock Dependencies

**Files:**
- Create: `src/evidence_route/__init__.py`
- Create: `tests/test_package_metadata.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `.env.example`

- [ ] **Step 1: Write the failing package identity test**

```python
# tests/test_package_metadata.py
from importlib.metadata import metadata

import evidence_route


def test_distribution_identity() -> None:
    package = metadata("evidence-route")
    assert package["Name"] == "evidence-route"
    assert package["Requires-Python"] == ">=3.11"
    assert evidence_route.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test and confirm the new package does not exist**

Run: `conda run -n agent-collab python -m pytest tests/test_package_metadata.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'evidence_route'`.

- [ ] **Step 3: Replace package metadata and add the package root**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=82.0"]
build-backend = "setuptools.build_meta"

[project]
name = "evidence-route"
version = "0.1.0"
description = "Cost-aware adaptive fact verification with LangGraph"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
  "langgraph==1.2.11",
  "langgraph-checkpoint-sqlite==3.1.1",
  "numpy==2.4.6",
  "openai==3.1.0",
  "pydantic==2.13.4",
  "PyYAML==6.0.3",
  "rank-bm25==0.2.2",
  "remotezip==0.12.3",
  "rich==15.0.0",
  "typer==0.27.1",
]

[project.optional-dependencies]
eval = [
  "nltk==3.10.3",
  "scikit-learn==1.9.0",
  "scipy==1.18.0",
]
dev = [
  "pip-tools==7.6.1",
  "pytest==9.1.1",
  "pytest-asyncio==1.4.0",
  "ruff==0.16.3",
]

[project.scripts]
evidence-route = "evidence_route.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

```python
# src/evidence_route/__init__.py
"""EvidenceRoute public package."""

__version__ = "0.1.0"
```

```dotenv
# .env.example
EVIDENCE_ROUTE_API_KEY=
EVIDENCE_ROUTE_BASE_URL=
EVIDENCE_ROUTE_MODEL=
EVIDENCE_ROUTE_PRICE_FILE=configs/pricing.local.yaml
```

Ensure these entries are present in `.gitignore` without duplicating the existing `artifacts/` and
`.env` rules:

```gitignore
data/external/
data/processed/
configs/pricing.local.yaml
artifacts/
*.sqlite3
```

- [ ] **Step 4: Install exact dependencies and generate the transitive lock**

Run:

```powershell
conda run -n agent-collab python -m pip install -e ".[dev,eval]"
conda run -n agent-collab python -m piptools compile pyproject.toml --extra dev --extra eval --output-file requirements.lock
conda run -n agent-collab python -m pip check
```

Expected: installation succeeds, `requirements.lock` is created, and `pip check` prints `No broken requirements found.`

- [ ] **Step 5: Run the identity test and existing regression suite**

Run: `conda run -n agent-collab python -m pytest -q`

Expected: new identity test and the existing 117 tests pass.

- [ ] **Step 6: Commit the distribution boundary**

```powershell
git add pyproject.toml requirements.lock .gitignore .env.example src/evidence_route/__init__.py tests/test_package_metadata.py
git commit -m "build: create EvidenceRoute distribution"
```

### Task 2: Define Strict Runtime Contracts

**Files:**
- Create: `src/evidence_route/contracts.py`
- Create: `tests/test_contracts.py`

- [ ] **Step 1: Write status, citation, and result contract tests**

```python
# tests/test_contracts.py
import pytest
from pydantic import ValidationError

from evidence_route.contracts import (
    Citation,
    Evidence,
    ResultStatus,
    Usage,
    VerificationResult,
    Verdict,
    WorkerResult,
)


def citation() -> Citation:
    return Citation(
        evidence_id="av:dev:0:0:4",
        claim_unit_ids=["u0"],
        question="Was the letter authentic?",
        answer="The letter was fabricated.",
        quote="Sean Connery never sent the letter.",
        stance="refutes",
        source_url="https://example.org/source",
    )


def complete_usage() -> Usage:
    return Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True)


def test_completed_result_requires_verdict() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="dev-0",
            status=ResultStatus.COMPLETED,
            confidence=0.8,
            rationale="Evidence is consistent.",
            citations=[citation()],
            initial_route="single",
            usage=complete_usage(),
        )


def test_failed_result_rejects_verdict() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="dev-0",
            status=ResultStatus.FAILED,
            verdict=Verdict.REFUTED,
            rationale="Provider unavailable.",
            initial_route="single",
            usage=complete_usage(),
        )


def test_running_is_not_a_final_result_status() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="dev-0", status="running", rationale="not final",
            initial_route="single", usage=complete_usage(),
        )


def test_result_requires_explicit_usage() -> None:
    with pytest.raises(ValidationError, match="usage"):
        VerificationResult(
            claim_id="dev-0", status=ResultStatus.FAILED,
            rationale="transport failed", initial_route="single",
        )


def test_incomplete_usage_forbids_exact_cost_fields() -> None:
    with pytest.raises(ValidationError, match="incomplete usage"):
        VerificationResult(
            claim_id="dev-0", status=ResultStatus.FAILED,
            rationale="provider omitted usage", initial_route="single",
            usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False),
            estimated_cost_micro_cny=0, cost_currency="CNY", price_config_id="a" * 64,
        )


def test_only_typed_pre_route_failure_allows_null_initial_route() -> None:
    result = VerificationResult(
        claim_id="dev-0", status=ResultStatus.FAILED,
        rationale="probe retrieval failed", initial_route=None,
        failure_stage="pre_route", usage=complete_usage(),
        errors=["PROBE_RETRIEVAL_FAILED"],
    )
    assert result.initial_route is None
    with pytest.raises(ValidationError, match="initial_route"):
        VerificationResult(
            claim_id="dev-0", status=ResultStatus.FAILED,
            rationale="single failed", initial_route=None,
            failure_stage="single", usage=complete_usage(),
        )


def test_failed_worker_rejects_verdict() -> None:
    with pytest.raises(ValidationError):
        WorkerResult(
            task_id="t0", claim_unit_ids=["u0"], status=ResultStatus.FAILED,
            verdict=Verdict.REFUTED, usage=complete_usage(),
        )


def test_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="av:dev:0:0:4",
            title="Source",
            source_url="https://example.org/source",
            text="Evidence text",
            provider="averitec_frozen",
            snapshot_sha256="a" * 64,
            ranking_score=1.0,
            gold_label="Refuted",
        )
```

- [ ] **Step 2: Run the contract tests and confirm imports fail**

Run: `conda run -n agent-collab python -m pytest tests/test_contracts.py -q`

Expected: FAIL because `evidence_route.contracts` does not exist.

- [ ] **Step 3: Implement enums and Pydantic models**

Create `src/evidence_route/contracts.py` with these public types and exact fields:

```python
from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Verdict(StrEnum):
    SUPPORTED = "Supported"
    REFUTED = "Refuted"
    NOT_ENOUGH_EVIDENCE = "Not Enough Evidence"
    CONFLICTING = "Conflicting Evidence/Cherrypicking"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class ResultStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class Strategy(StrEnum):
    ALWAYS_SINGLE = "always_single"
    ALWAYS_MULTI = "always_multi"
    ADAPTIVE = "adaptive"


class Usage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    complete: bool

    @model_validator(mode="after")
    def validate_total(self) -> "Usage":
        if self.complete and self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("complete usage total must equal input plus output")
        return self


class Evidence(StrictModel):
    evidence_id: str = Field(min_length=1)
    title: str
    source_url: HttpUrl
    text: str = Field(min_length=1)
    provider: Literal["averitec_frozen"]
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ranking_score: float
    char_start: int = Field(default=0, ge=0)
    char_end: int = Field(default=0, ge=0)


class ClaimUnit(StrictModel):
    unit_id: str
    text: str = Field(min_length=1)


class ClaimFeatures(StrictModel):
    claim_units: list[ClaimUnit] = Field(min_length=1)
    atomic_clause_count: int = Field(ge=1)
    entity_count: int = Field(ge=0)
    numeric_count: int = Field(ge=0)
    time_scope_count: int = Field(ge=0)
    has_comparison: bool
    has_causal: bool
    has_contrast: bool
    probe_source_count: int = Field(ge=0)
    probe_score_spread: float = Field(ge=0)
    probe_conflict_hint: bool


class RouteDecision(StrictModel):
    route: Literal["single", "multi"]
    source: Literal["rule", "llm", "fallback", "strategy"]
    reason_codes: list[str] = Field(min_length=1)
    explanation: str = Field(min_length=1, max_length=300)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class Citation(StrictModel):
    evidence_id: str
    claim_unit_ids: list[str] = Field(min_length=1)
    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(min_length=1, max_length=1200)
    quote: str = Field(min_length=1, max_length=600)
    stance: Literal["supports", "refutes", "conflicts", "insufficient"]
    source_url: HttpUrl


class VerificationTask(StrictModel):
    task_id: str = Field(pattern=r"^t[0-2]$")
    claim_unit_ids: list[str] = Field(min_length=1)
    query: str = Field(min_length=1)


class WorkerResult(StrictModel):
    task_id: str
    claim_unit_ids: list[str] = Field(min_length=1)
    status: ResultStatus
    verdict: Verdict | None
    confidence: float | None = Field(default=None, ge=0, le=1)
    citations: list[Citation] = Field(default_factory=list)
    available_evidence_ids: list[str] = Field(default_factory=list)
    usage: Usage
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_fields(self) -> "WorkerResult":
        if self.status == ResultStatus.FAILED:
            if self.verdict is not None or self.confidence is not None:
                raise ValueError("failed worker results cannot carry verdict or confidence")
        elif self.verdict is None or self.confidence is None:
            raise ValueError("completed and partial worker results require verdict and confidence")
        return self


class VerificationResult(StrictModel):
    claim_id: str
    status: ResultStatus
    verdict: Verdict | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str
    citations: list[Citation] = Field(default_factory=list)
    available_evidence_ids: list[str] = Field(default_factory=list)
    initial_route: Literal["single", "multi"] | None
    escalated: bool = False
    failure_stage: Literal["pre_route", "single", "decompose", "judge", "validation"] | None = None
    usage: Usage
    estimated_cost_micro_cny: int | None = Field(default=None, ge=0)
    cost_currency: Literal["CNY"] | None = None
    price_config_id: str | None = None
    latency_ms: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_fields(self) -> "VerificationResult":
        cost_fields = (
            self.estimated_cost_micro_cny,
            self.cost_currency,
            self.price_config_id,
        )
        if not self.usage.complete and any(field is not None for field in cost_fields):
            raise ValueError("incomplete usage forbids exact cost and pricing fields")
        if self.usage.complete and any(field is not None for field in cost_fields) \
                and not all(field is not None for field in cost_fields):
            raise ValueError("exact cost fields must be all present or all absent")
        if self.initial_route is None and not (
            self.status == ResultStatus.FAILED
            and self.failure_stage == "pre_route"
            and "PROBE_RETRIEVAL_FAILED" in self.errors
        ):
            raise ValueError("null initial_route requires typed pre-route failure")
        if self.failure_stage == "pre_route" and self.initial_route is not None:
            raise ValueError("pre-route failure cannot carry initial_route")
        if self.escalated and self.initial_route != "single":
            raise ValueError("only an initial single route can escalate")
        if self.status == ResultStatus.FAILED:
            if self.verdict is not None or self.confidence is not None:
                raise ValueError("failed results cannot carry verdict or confidence")
            return self
        if self.status in {ResultStatus.COMPLETED, ResultStatus.PARTIAL}:
            if self.verdict is None or self.confidence is None:
                raise ValueError("completed and partial results require verdict and confidence")
        return self


class VerificationState(TypedDict, total=False):
    run_id: str
    claim_id: str
    claim_text: str
    language: str
    strategy: Strategy
    probe_evidence: list[Evidence]
    claim_features: ClaimFeatures
    route_decision: RouteDecision
    tasks: list[VerificationTask]
    worker_results: Annotated[list[WorkerResult], operator.add]
    draft_result: VerificationResult
    final_result: VerificationResult
    escalation_count: int
    errors: Annotated[list[str], operator.add]
```

- [ ] **Step 4: Run contract tests**

Run: `conda run -n agent-collab python -m pytest tests/test_contracts.py -q`

Expected: all contract tests pass.

- [ ] **Step 5: Commit contracts**

```powershell
git add src/evidence_route/contracts.py tests/test_contracts.py
git commit -m "feat: define strict verification contracts"
```

### Task 3: Load, Hash, And Redact Configuration

**Files:**
- Create: `src/evidence_route/config.py`
- Create: `configs/default.yaml`
- Create: `configs/pricing.example.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write configuration tests**

```python
# tests/test_config.py
from pathlib import Path

import pytest

from evidence_route.config import load_app_config, redact_mapping, stable_hash


def test_config_reads_secret_without_serializing_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
```

- [ ] **Step 2: Verify tests fail before configuration code exists**

Run: `conda run -n agent-collab python -m pytest tests/test_config.py -q`

Expected: FAIL importing `evidence_route.config`.

- [ ] **Step 3: Implement strict configuration and deterministic hashing**

Implement these models and functions in `src/evidence_route/config.py`:

```python
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMSettings(ConfigModel):
    base_url: str
    api_key: SecretStr = Field(exclude=True)
    requested_alias: str
    temperature: float = Field(default=0.0, ge=0, le=2)
    timeout_s: float = Field(default=90.0, gt=0)
    transient_retries: Literal[2] = 2
    structured_mode: Literal["json_schema", "json_object"] = "json_object"


class RoutingSettings(ConfigModel):
    clear_multi_clauses: int = Field(default=3, ge=2)
    clear_single_min_sources: int = Field(default=2, ge=1)
    low_confidence: float = Field(default=0.65, ge=0, le=1)
    minimum_coverage: float = Field(default=1.0, ge=0, le=1)


class EvidenceSettings(ConfigModel):
    probe_top_k: int = Field(default=3, gt=0)
    probe_chars: int = Field(default=450, gt=0)
    single_top_k: int = Field(default=8, gt=0)
    single_chars: int = Field(default=800, gt=0)
    worker_top_k: int = Field(default=5, gt=0)
    worker_chars: int = Field(default=800, gt=0)
    judge_max_evidence: int = Field(default=12, gt=0)
    judge_chars: int = Field(default=600, gt=0)


class TokenLimit(ConfigModel):
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)


class GenerationSettings(ConfigModel):
    router: TokenLimit = TokenLimit(max_input_tokens=1800, max_output_tokens=250)
    single: TokenLimit = TokenLimit(max_input_tokens=6500, max_output_tokens=1000)
    decomposer: TokenLimit = TokenLimit(max_input_tokens=2500, max_output_tokens=500)
    worker: TokenLimit = TokenLimit(max_input_tokens=5000, max_output_tokens=800)
    judge: TokenLimit = TokenLimit(max_input_tokens=8000, max_output_tokens=1200)
    max_workers: Literal[3] = 3
    max_escalations: Literal[1] = 1
    max_transport_attempts: Literal[3] = 3
    max_repairs_per_node: Literal[1] = 1


class BudgetSettings(ConfigModel):
    estimated_cost_cap_cny: float = Field(default=350.0, gt=0)
    reserve_ratio: float = Field(default=0.2, ge=0, le=1)


class AppConfig(ConfigModel):
    llm: LLMSettings
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    evidence: EvidenceSettings = Field(default_factory=EvidenceSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def redact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    sensitive = {"authorization", "api_key", "token", "secret", "password"}
    return {
        key: "[REDACTED]" if key.lower() in sensitive else item
        for key, item in value.items()
    }


def load_app_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    llm_options = dict(raw.get("llm") or {})
    environment_only = {"base_url", "api_key", "requested_alias"}
    present = sorted(environment_only & llm_options.keys())
    if present:
        raise ValueError(f"{', '.join(present)} must come from environment")
    raw["llm"] = {
        **llm_options,
        "base_url": os.environ["EVIDENCE_ROUTE_BASE_URL"],
        "api_key": os.environ["EVIDENCE_ROUTE_API_KEY"],
        "requested_alias": os.environ["EVIDENCE_ROUTE_MODEL"],
    }
    return AppConfig.model_validate(raw)
```

Create `configs/default.yaml`:

```yaml
llm:
  temperature: 0.0
  timeout_s: 90.0
  transient_retries: 2
  structured_mode: json_object
routing:
  clear_multi_clauses: 3
  clear_single_min_sources: 2
  low_confidence: 0.65
  minimum_coverage: 1.0
evidence:
  probe_top_k: 3
  probe_chars: 450
  single_top_k: 8
  single_chars: 800
  worker_top_k: 5
  worker_chars: 800
  judge_max_evidence: 12
  judge_chars: 600
generation:
  router: {max_input_tokens: 1800, max_output_tokens: 250}
  single: {max_input_tokens: 6500, max_output_tokens: 1000}
  decomposer: {max_input_tokens: 2500, max_output_tokens: 500}
  worker: {max_input_tokens: 5000, max_output_tokens: 800}
  judge: {max_input_tokens: 8000, max_output_tokens: 1200}
  max_workers: 3
  max_escalations: 1
  max_transport_attempts: 3
  max_repairs_per_node: 1
budget:
  estimated_cost_cap_cny: 350.0
  reserve_ratio: 0.2
```

Create `configs/pricing.example.yaml`:

```yaml
provider: relay
currency: CNY
input_per_million: null
output_per_million: null
price_source: null
strict_evaluation: false
```

- [ ] **Step 4: Run configuration tests**

Run: `conda run -n agent-collab python -m pytest tests/test_config.py -q`

Expected: all tests pass without printing the secret.

- [ ] **Step 5: Commit configuration**

```powershell
git add src/evidence_route/config.py configs/default.yaml configs/pricing.example.yaml tests/test_config.py
git commit -m "feat: add hashed and redacted configuration"
```

### Task 4: Extract Deterministic Claim Features

**Files:**
- Create: `src/evidence_route/analyzer.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: Write bilingual feature tests**

```python
# tests/test_analyzer.py
from evidence_route.analyzer import analyze_claim
from evidence_route.contracts import Evidence


def test_analyzer_splits_compound_english_claim() -> None:
    features = analyze_claim(
        "Company A grew in 2024, but Company B declined in 2025.",
        probe_evidence=[],
    )
    assert features.atomic_clause_count == 2
    assert features.has_contrast is True
    assert features.time_scope_count == 2


def test_analyzer_splits_compound_chinese_claim() -> None:
    features = analyze_claim("甲公司收入增长，而且乙公司利润下降。", probe_evidence=[])
    assert [unit.unit_id for unit in features.claim_units] == ["u0", "u1"]
    assert features.atomic_clause_count == 2


def test_probe_conflict_is_only_a_hint() -> None:
    evidence = [
        Evidence(
            evidence_id="av:dev:1:0:0",
            title="A",
            source_url="https://example.org/a",
            text="The rate was 10 percent.",
            provider="averitec_frozen",
            snapshot_sha256="a" * 64,
            ranking_score=2.0,
        ),
        Evidence(
            evidence_id="av:dev:1:1:0",
            title="B",
            source_url="https://example.org/b",
            text="The rate was not 10 percent; it was 12 percent.",
            provider="averitec_frozen",
            snapshot_sha256="b" * 64,
            ranking_score=1.0,
        ),
    ]
    assert analyze_claim("The rate was 10 percent.", evidence).probe_conflict_hint is True
```

- [ ] **Step 2: Run tests and confirm analyzer import fails**

Run: `conda run -n agent-collab python -m pytest tests/test_analyzer.py -q`

Expected: FAIL importing `evidence_route.analyzer`.

- [ ] **Step 3: Implement versioned lexical analysis**

Implement `analyze_claim(claim: str, probe_evidence: list[Evidence]) -> ClaimFeatures` with these exact rules:

```python
_BOUNDARY_RE = re.compile(
    r"(?:[;；。]|\bbut\b|\band\b|\bwhile\b|\bwhereas\b|但是|但|而且|并且)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_TIME_RE = re.compile(r"\b(?:19|20)\d{2}\b|\d{4}年|今年|去年|本月|today|yesterday", re.I)
_COMPARISON_RE = re.compile(r"more than|less than|higher|lower|超过|低于|高于|相比", re.I)
_CAUSAL_RE = re.compile(r"because|caused|therefore|因为|导致|因此", re.I)
_CONTRAST_RE = re.compile(r"\bbut\b|whereas|however|但是|然而|并非", re.I)
_NEGATION_RE = re.compile(r"\bnot\b|\bnever\b|没有|并非|从未", re.I)


def analyze_claim(claim: str, probe_evidence: list[Evidence]) -> ClaimFeatures:
    parts = [part.strip(" ,，") for part in _BOUNDARY_RE.split(claim) if part.strip(" ,，")]
    units = [ClaimUnit(unit_id=f"u{index}", text=text) for index, text in enumerate(parts or [claim])]
    scores = [item.ranking_score for item in probe_evidence]
    all_numbers = [set(_NUMBER_RE.findall(item.text)) for item in probe_evidence]
    conflicting_numbers = len({number for group in all_numbers for number in group}) > 1
    negated = any(_NEGATION_RE.search(item.text) for item in probe_evidence)
    return ClaimFeatures(
        claim_units=units,
        atomic_clause_count=len(units),
        entity_count=len(re.findall(r"\b[A-Z][A-Za-z0-9-]+\b|[\u4e00-\u9fff]{2,8}(?:公司|大学|政府|协会)", claim)),
        numeric_count=len(_NUMBER_RE.findall(claim)),
        time_scope_count=len(_TIME_RE.findall(claim)),
        has_comparison=bool(_COMPARISON_RE.search(claim)),
        has_causal=bool(_CAUSAL_RE.search(claim)),
        has_contrast=bool(_CONTRAST_RE.search(claim)),
        probe_source_count=len({str(item.source_url) for item in probe_evidence}),
        probe_score_spread=(max(scores) - min(scores)) if scores else 0.0,
        probe_conflict_hint=conflicting_numbers and negated,
    )
```

- [ ] **Step 4: Run analyzer tests**

Run: `conda run -n agent-collab python -m pytest tests/test_analyzer.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit analyzer**

```powershell
git add src/evidence_route/analyzer.py tests/test_analyzer.py
git commit -m "feat: extract deterministic claim features"
```

### Task 5: Implement The Frozen AVeriTeC Provider

**Files:**
- Create: `src/evidence_route/providers/__init__.py`
- Create: `src/evidence_route/providers/base.py`
- Create: `src/evidence_route/providers/averitec.py`
- Create: `tests/fixtures/averitec/corpora/dev-0.jsonl`
- Create: `tests/test_averitec_provider.py`

- [ ] **Step 1: Add a gold-free claim corpus fixture**

```json
{"evidence_id":"av:dev:0:0:0","title":"CNET","url":"https://example.org/cnet","text":"Sean Connery never sent the letter to Steve Jobs."}
{"evidence_id":"av:dev:0:1:0","title":"Archive","url":"https://example.org/archive","text":"Scoopertino describes itself as an imaginary Apple news organization."}
{"evidence_id":"av:dev:0:2:0","title":"Other","url":"https://example.org/other","text":"The actor appeared in several advertisements."}
```

- [ ] **Step 2: Write retrieval and leakage tests**

```python
# tests/test_averitec_provider.py
from pathlib import Path

import pytest

from evidence_route.providers.averitec import AveritecFrozenProvider


@pytest.mark.asyncio
async def test_bm25_returns_stable_ids_and_truncates() -> None:
    provider = AveritecFrozenProvider(Path("tests/fixtures/averitec/corpora"))
    results = await provider.search("dev-0", "Was the Connery letter authentic?", top_k=2, max_chars=30)
    assert results[0].evidence_id == "av:dev:0:0:0"
    assert len(results[0].text) <= 30
    assert results[0].snapshot_sha256 != "0" * 64


@pytest.mark.asyncio
async def test_missing_claim_corpus_is_an_error() -> None:
    provider = AveritecFrozenProvider(Path("tests/fixtures/averitec/corpora"))
    with pytest.raises(FileNotFoundError):
        await provider.search("dev-999", "query", top_k=2, max_chars=30)


@pytest.mark.asyncio
async def test_repeated_search_reuses_one_claim_index() -> None:
    provider = AveritecFrozenProvider(
        Path("tests/fixtures/averitec/corpora"), max_cached_claims=1
    )
    await provider.search("dev-0", "letter", top_k=2, max_chars=30)
    await provider.search("dev-0", "imaginary", top_k=2, max_chars=30)
    assert provider.index_build_count == 1
```

- [ ] **Step 3: Run tests and confirm provider modules are absent**

Run: `conda run -n agent-collab python -m pytest tests/test_averitec_provider.py -q`

Expected: FAIL importing `evidence_route.providers`.

- [ ] **Step 4: Implement the provider protocol and BM25 provider**

```python
# src/evidence_route/providers/base.py
from __future__ import annotations


from evidence_route.contracts import Evidence


class EvidenceProvider(Protocol):
    async def search(self, claim_id: str, query: str, *, top_k: int, max_chars: int) -> list[Evidence]:
        raise NotImplementedError
```

`AveritecFrozenProvider.search` must read only `{claim_id}.jsonl` and require every corpus row to have
exactly `{"evidence_id", "title", "source_url", "text", "snapshot_sha256"}`; missing or extra keys
fail closed, so labels, annotated questions, justification, query/type fields, and future gold fields
cannot enter retrieval. Rank all text with `BM25Okapi`, break score ties by `evidence_id`, truncate
after ranking, and compute SHA-256 from the untruncated UTF-8 text. Keep an
LRU of exactly `max_cached_claims=1` parsed/tokenized claim index by default so probe, all three
policies, escalation, and worker searches reuse one build without retaining the whole benchmark.
Expose `index_build_count` for the offline cache test. Record retrieval configuration name
`evidence-route-regex-bm25-v1` in results/manifests; state explicitly that this project tokenizer is
not the official AVeriTeC baseline NLTK BM25 tokenizer. Use this tokenizer:

```python
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?|[\u4e00-\u9fff]", re.I)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]
```

Build each result with:

```python
Evidence(
    evidence_id=record["evidence_id"],
    title=record["title"],
    source_url=record["url"],
    text=record["text"][:max_chars],
    provider="averitec_frozen",
    snapshot_sha256=hashlib.sha256(record["text"].encode("utf-8")).hexdigest(),
    ranking_score=float(score),
    char_start=0,
    char_end=min(len(record["text"]), max_chars),
)
```

- [ ] **Step 5: Run provider tests**

Run: `conda run -n agent-collab python -m pytest tests/test_averitec_provider.py -q`

Expected: both tests pass.

- [ ] **Step 6: Commit provider**

```powershell
git add src/evidence_route/providers tests/fixtures/averitec/corpora tests/test_averitec_provider.py
git commit -m "feat: add frozen AVeriTeC BM25 provider"
```

### Task 6: Enforce Usage And Estimated-Cost Budgets

**Files:**
- Create: `src/evidence_route/budget.py`
- Create: `tests/test_budget.py`

- [ ] **Step 1: Write cost, reservation, and missing-usage tests**

```python
# tests/test_budget.py
from decimal import ROUND_CEILING, Decimal
import hashlib
import json
from threading import RLock
from typing import Protocol

import pytest

from evidence_route.budget import (
    BudgetExceeded,
    InMemoryBudgetStore,
    PriceConfig,
    UsageUnavailable,
)
from evidence_route.contracts import Usage


def pricing() -> PriceConfig:
    return PriceConfig(
        provider="relay",
        currency="CNY",
        input_per_million=10.0,
        output_per_million=20.0,
        price_source="relay dashboard captured 2026-08-17",
        strict_evaluation=True,
    )


def test_cost_uses_separate_input_and_output_rates() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=500_000, total_tokens=1_500_000, complete=True)
    assert pricing().estimate(usage) == Decimal("20.0")


def test_strict_ledger_rejects_incomplete_usage() -> None:
    ledger = InMemoryBudgetStore(cap_cny=350.0, pricing=pricing())
    with pytest.raises(UsageUnavailable):
        ledger.record_call(
            "call-1", Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False)
        )


def test_reservation_stops_before_crossing_cap() -> None:
    ledger = InMemoryBudgetStore(cap_cny=1.0, pricing=pricing())
    with pytest.raises(BudgetExceeded):
        ledger.reserve_call(
            "call-1", max_input_tokens=100_000, max_output_tokens=100_000
        )


def test_recording_same_call_is_idempotent() -> None:
    store = InMemoryBudgetStore(cap_cny=350.0, pricing=pricing())
    store.reserve_call("call-1", max_input_tokens=100, max_output_tokens=20)
    usage = Usage(input_tokens=100, output_tokens=20, total_tokens=120, complete=True)
    store.record_call("call-1", usage)
    store.record_call("call-1", usage)
    assert store.recorded_call_ids == {"call-1"}
```

- [ ] **Step 2: Run budget tests and verify import failure**

Run: `conda run -n agent-collab python -m pytest tests/test_budget.py -q`

Expected: FAIL importing `evidence_route.budget`.

- [ ] **Step 3: Implement strict pricing and a reservation ledger**

```python
# src/evidence_route/budget.py
from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidence_route.contracts import Usage


class BudgetExceeded(RuntimeError):
    pass


class UsageUnavailable(RuntimeError):
    pass


class PriceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    currency: str
    input_per_million: float | None = Field(default=None, gt=0)
    output_per_million: float | None = Field(default=None, gt=0)
    price_source: str | None = None
    strict_evaluation: bool = False

    @model_validator(mode="after")
    def validate_strict_price(self) -> "PriceConfig":
        if self.strict_evaluation and (
            self.input_per_million is None
            or self.output_per_million is None
            or not self.price_source
        ):
            raise ValueError("strict evaluation requires rates and price_source")
        return self

    @property
    def config_id(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def estimate(self, usage: Usage) -> Decimal:
        if not usage.complete:
            raise UsageUnavailable("provider usage is incomplete")
        if self.input_per_million is None or self.output_per_million is None:
            raise UsageUnavailable("pricing is not configured")
        input_cost = Decimal(usage.input_tokens) * Decimal(str(self.input_per_million)) / Decimal(1_000_000)
        output_cost = Decimal(usage.output_tokens) * Decimal(str(self.output_per_million)) / Decimal(1_000_000)
        return input_cost + output_cost

    def estimate_micro_cny(self, usage: Usage) -> int:
        return int((self.estimate(usage) * Decimal(1_000_000)).to_integral_value(
            rounding=ROUND_CEILING
        ))


class InMemoryBudgetStore:
    def __init__(self, *, cap_cny: float, pricing: PriceConfig) -> None:
        self.cap_micro_cny = int(Decimal(str(cap_cny)) * Decimal(1_000_000))
        self.pricing = pricing
        self.reservations: dict[str, int] = {}
        self.actual_by_call: dict[str, int] = {}
        self._lock = RLock()

    @property
    def recorded_call_ids(self) -> set[str]:
        return set(self.actual_by_call)

    def reserve_call(
        self, call_id: str, *, max_input_tokens: int, max_output_tokens: int
    ) -> int:
        projected = self.pricing.estimate_micro_cny(Usage(
            input_tokens=max_input_tokens,
            output_tokens=max_output_tokens,
            total_tokens=max_input_tokens + max_output_tokens,
            complete=True,
        ))
        with self._lock:
            if call_id in self.actual_by_call or call_id in self.reservations:
                return self.reservations.get(call_id, self.actual_by_call.get(call_id, 0))
            committed = sum(self.actual_by_call.values()) + sum(self.reservations.values())
            if committed + projected > self.cap_micro_cny:
                raise BudgetExceeded("next call would exceed estimated_cost_cap")
            self.reservations[call_id] = projected
            return projected

    def record_call(self, call_id: str, usage: Usage) -> int:
        actual = self.pricing.estimate_micro_cny(usage)
        with self._lock:
            if call_id in self.actual_by_call:
                return self.actual_by_call[call_id]
            self.reservations.pop(call_id, None)
            self.actual_by_call[call_id] = actual
            return actual
```

All budget comparisons and persistence use integer micro-CNY. Floats appear only in final display
fields after converting micro-CNY to decimal CNY. `InMemoryBudgetStore` is deterministic unit-test
support only; it must never be accepted by the production adapter or campaign runner. Task 7 supplies
the single production SQLite store that owns both call lifecycle and budget reservation in one
transactional database.

- [ ] **Step 4: Run budget tests**

Run: `conda run -n agent-collab python -m pytest tests/test_budget.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit budget enforcement**

```powershell
git add src/evidence_route/budget.py tests/test_budget.py
git commit -m "feat: enforce usage-aware estimated cost budget"
```

### Task 7: Persist Traces And Idempotent Paid Calls

**Files:**
- Create: `src/evidence_route/artifacts.py`
- Create: `tests/test_artifacts.py`

- [ ] **Step 1: Write cache, atomic write, and redaction tests**

```python
# tests/test_artifacts.py
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from evidence_route.artifacts import (
    BillingStateError,
    RequestFingerprintMismatch,
    SQLiteRunStore,
    TraceWriter,
    atomic_write_json,
)
from evidence_route.budget import BudgetExceeded, PriceConfig
from evidence_route.contracts import Usage


def test_run_store_keeps_first_completed_response(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.reserve_call(
        "call-1", request_sha256="a" * 64, run_id="run-1", node="router",
        task_id="root", logical_attempt=0, max_input_tokens=100, max_output_tokens=20,
    )
    store.mark_sent("call-1")
    usage = Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True)
    store.complete_call(
        "call-1", request_sha256="a" * 64,
        payload={"content": {"route": "single"}}, usage=usage,
        usage_source="provider",
        requested_alias="alias", response_model_id_raw="relay-model", identity_verified=False,
    )
    store.complete_call(
        "call-1", request_sha256="a" * 64,
        payload={"content": {"route": "multi"}}, usage=usage,
        usage_source="provider",
        requested_alias="alias", response_model_id_raw="relay-model", identity_verified=False,
    )
    assert store.get_completed("call-1")["content"]["route"] == "single"
    summary = store.summarize_run("run-1")
    assert summary.usage.total_tokens == 10
    assert summary.actual_cost_micro_cny > 0
    assert summary.call_ids == ["call-1"]


def test_cache_rejects_same_call_id_with_different_request_sha(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    kwargs = dict(
        run_id="run-1", node="router", task_id="root", logical_attempt=0,
        max_input_tokens=100, max_output_tokens=20,
    )
    store.reserve_call("call-1", request_sha256="a" * 64, **kwargs)
    with pytest.raises(RequestFingerprintMismatch):
        store.reserve_call("call-1", request_sha256="b" * 64, **kwargs)


def test_sent_without_completion_blocks_automatic_resume(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.reserve_call(
        "call-1", request_sha256="a" * 64, run_id="run-1", node="single",
        task_id="root", logical_attempt=0, max_input_tokens=100, max_output_tokens=20,
    )
    store.mark_sent("call-1")
    reopened = make_store(tmp_path)
    with pytest.raises(BillingStateError, match="sent call has unknown billing"):
        reopened.resume_decision("call-1", request_sha256="a" * 64)


def test_parallel_reservations_are_serialized_without_crossing_cap(tmp_path: Path) -> None:
    barrier = Barrier(2)

    def reserve(call_id: str) -> str:
        store = make_store(tmp_path, cap_cny=0.0001)
        barrier.wait()
        store.reserve_call(
            call_id, request_sha256=call_id[-1] * 64,
            run_id=call_id, node="worker", task_id="t0", logical_attempt=0,
            max_input_tokens=100, max_output_tokens=0,
        )
        return call_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reserve, call_id) for call_id in ("call-a", "call-b")]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except BudgetExceeded:
            outcomes.append("budget_exceeded")
    assert sorted(outcomes) in (["budget_exceeded", "call-a"], ["budget_exceeded", "call-b"])


def test_crash_after_cache_write_before_reconcile_counts_cost_once(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.reserve_call(
        "call-1", request_sha256="a" * 64, run_id="run-1",
        node="judge", task_id="root", logical_attempt=0,
        max_input_tokens=100, max_output_tokens=20,
    )
    store.mark_sent("call-1")
    usage = Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True)
    store.complete_call(
        "call-1", request_sha256="a" * 64, payload={"content": "first"}, usage=usage,
        usage_source="provider",
        requested_alias="alias", response_model_id_raw="relay-model", identity_verified=False,
    )
    reopened = make_store(tmp_path)
    reopened.complete_call(
        "call-1", request_sha256="a" * 64, payload={"content": "second"}, usage=usage,
        usage_source="provider",
        requested_alias="alias", response_model_id_raw="relay-model", identity_verified=False,
    )
    summary = reopened.summarize_run("run-1")
    assert summary.actual_cost_micro_cny == reopened.pricing.estimate_micro_cny(usage)
    assert reopened.get_completed("call-1")["content"] == "first"


def test_missing_usage_response_is_terminal_and_never_reissued(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.reserve_call(
        "call-1", request_sha256="a" * 64, run_id="run-1", node="single",
        task_id="root", logical_attempt=0, max_input_tokens=100, max_output_tokens=20,
    )
    store.mark_sent("call-1")
    store.complete_call(
        "call-1", request_sha256="a" * 64, payload={"content": "received"},
        usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False),
        usage_source="missing", requested_alias="alias",
        response_model_id_raw="relay-model", identity_verified=False,
    )
    reopened = make_store(tmp_path)
    decision = reopened.resume_decision("call-1", request_sha256="a" * 64)
    assert decision.action == "reuse_and_stop"
    assert decision.payload == {"content": "received"}
    summary = reopened.summarize_run("run-1")
    assert summary.actual_cost_micro_cny is None
    assert summary.known_actual_cost_micro_cny == 0
    assert summary.cost_is_lower_bound is True


def test_usage_missing_reservation_stays_committed_during_parallel_reserve(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path, cap_cny=0.00025)
    store.reserve_call(
        "call-missing", request_sha256="a" * 64, run_id="run-missing",
        node="worker", task_id="t0", logical_attempt=0,
        max_input_tokens=100, max_output_tokens=0,
    )
    store.mark_sent("call-missing")
    store.complete_call(
        "call-missing", request_sha256="a" * 64, payload={"content": "received"},
        usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False),
        usage_source="missing", requested_alias="alias",
        response_model_id_raw="relay-model", identity_verified=False,
    )
    barrier = Barrier(2)

    def reserve(call_id: str) -> str:
        reopened = make_store(tmp_path, cap_cny=0.00025)
        barrier.wait()
        reopened.reserve_call(
            call_id, request_sha256=call_id[-1] * 64, run_id=call_id,
            node="worker", task_id="t1", logical_attempt=0,
            max_input_tokens=100, max_output_tokens=0,
        )
        return call_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reserve, call_id) for call_id in ("call-b", "call-c")]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except BudgetExceeded:
            outcomes.append("budget_exceeded")
    assert "budget_exceeded" in outcomes
    assert len([item for item in outcomes if item.startswith("call-")]) == 1


def make_store(tmp_path: Path, *, cap_cny: float = 10.0) -> SQLiteRunStore:
    pricing = PriceConfig(
        provider="fixture", currency="CNY", input_per_million=1.0,
        output_per_million=1.0, price_source="fixture", strict_evaluation=True,
    )
    return SQLiteRunStore(
        tmp_path / "run-store.sqlite3", activity_id="gate-a", cap_cny=cap_cny,
        pricing=pricing,
    )


def test_trace_redacts_nested_secrets(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    TraceWriter(path).write({"api_key": "secret", "headers": {"Authorization": "Bearer secret"}})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["api_key"] == "[REDACTED]"
    assert payload["headers"]["Authorization"] == "[REDACTED]"


def test_atomic_json_replaces_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    atomic_write_json(path, {"status": "completed"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "completed"}
    assert not path.with_suffix(".json.tmp").exists()
```

Add `test_run_call_summary_propagates_usage_cost_identity_and_call_ids` and
`test_run_summary_counts_cache_hits_after_reopen`: complete two calls, close/reopen the database,
reuse one completed response, and assert the typed summary preserves exact call IDs, usage, alias,
raw model ID, micro-CNY cost, one cache hit, and no duplicate fresh spend.

- [ ] **Step 2: Run tests and confirm artifact module is absent**

Run: `conda run -n agent-collab python -m pytest tests/test_artifacts.py -q`

Expected: FAIL importing `evidence_route.artifacts`.

- [ ] **Step 3: Implement one transactional SQLite run store and JSONL traces**

`SQLiteRunStore` is the only production implementation used by the LLM adapter and all three paid
phases. It creates and validates this schema on construction:

```sql
CREATE TABLE IF NOT EXISTS store_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    activity_id TEXT NOT NULL,
    cap_micro_cny INTEGER NOT NULL CHECK (cap_micro_cny > 0),
    currency TEXT NOT NULL CHECK (currency = 'CNY'),
    price_config_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calls (
    call_id TEXT PRIMARY KEY,
    request_sha256 TEXT NOT NULL CHECK (length(request_sha256) = 64),
    activity_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    node TEXT NOT NULL,
    task_id TEXT NOT NULL,
    logical_attempt INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('reserved', 'sent', 'completed', 'usage_missing', 'billing_uncertain')
    ),
    reserved_micro_cny INTEGER NOT NULL CHECK (reserved_micro_cny >= 0),
    actual_micro_cny INTEGER CHECK (actual_micro_cny >= 0),
    payload_json TEXT,
    usage_json TEXT,
    usage_source TEXT CHECK (usage_source IN ('provider', 'missing')),
    requested_alias TEXT,
    response_model_id_raw TEXT,
    identity_verified INTEGER CHECK (identity_verified IN (0, 1)),
    transport_attempts INTEGER NOT NULL DEFAULT 0 CHECK (transport_attempts >= 0),
    cache_hits INTEGER NOT NULL DEFAULT 0 CHECK (cache_hits >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (run_id, node, task_id, logical_attempt)
);
CREATE INDEX IF NOT EXISTS idx_calls_run_id ON calls(run_id);
CREATE INDEX IF NOT EXISTS idx_calls_activity_id ON calls(activity_id);
```

Expose these public records so adapter and campaign code do not inspect SQLite rows directly:

```python
class CallState(StrEnum):
    RESERVED = "reserved"
    SENT = "sent"
    COMPLETED = "completed"
    USAGE_MISSING = "usage_missing"
    BILLING_UNCERTAIN = "billing_uncertain"


class ResumeDecision(StrictModel):
    action: Literal["send", "reuse", "reuse_and_stop"]
    call_id: str
    payload: dict[str, object] | None = None


class RunCallSummary(StrictModel):
    call_ids: list[str]
    usage: Usage
    actual_cost_micro_cny: int | None = Field(default=None, ge=0)
    known_actual_cost_micro_cny: int = Field(ge=0)
    committed_cost_micro_cny: int = Field(ge=0)
    cost_is_lower_bound: bool
    fresh_call_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    transport_attempts: int = Field(ge=0)
    requested_aliases: list[str]
    response_model_ids_raw: list[str]
    usage_sources: list[Literal["provider", "missing"]]
    identity_verified: bool
    billing_uncertain: bool
```

All mutating methods open `BEGIN IMMEDIATE`; they commit or roll back before returning. Construction
inserts the one metadata row or rejects a changed activity ID, cap, currency, or price hash. The
store copies its immutable activity ID into every call row. `reserve_call` first
loads an existing `call_id`: a different `request_sha256` raises `RequestFingerprintMismatch`; the
same completed call is idempotent; an existing `usage_missing` call returns `reuse_and_stop` with
the first saved payload; the same `sent` or `billing_uncertain` call raises `BillingStateError`.
For a new call it sums `actual_micro_cny` for completed rows and
`reserved_micro_cny` for `reserved`/`sent`/`usage_missing`/`billing_uncertain` rows across the whole
activity store,
rejects a cap crossing, and inserts `reserved` with the integer reservation in that same transaction.
This is what serializes the three concurrent worker reservations.

Every connection sets `PRAGMA journal_mode=WAL`, `PRAGMA synchronous=FULL`, and
`PRAGMA busy_timeout=5000`; a lock timeout fails the run rather than bypassing the cap. Timestamps are
UTC diagnostics only and never participate in call IDs, hashes, or report regeneration.

`mark_sent` changes `reserved -> sent` and increments `transport_attempts` immediately before handing
the request to the transport. Only an explicit provider response known not to be billable may call
`mark_retryable_not_billed` and change `sent -> reserved`; a timeout, connection loss after handoff,
or process restart with `sent` calls `mark_billing_uncertain`, preserves the reservation, and blocks
automatic resume. `complete_call` requires the same request hash and always persists the first
response, usage record/source, and identity fields. With complete provider usage it calculates and
writes `actual_micro_cny`; with missing usage it writes `state='usage_missing'`, leaves actual cost
null, keeps the reservation committed for cap enforcement, and returns a typed stop signal that
forces `INCOMPLETE_USAGE` without reissuing the paid response.
All fields change in one transaction. Repeating completion returns the stored first response and never charges
again. If provider usage exceeds the reserved cap, persist the already-billed response/cost first,
then surface `BudgetExceeded` so the activity stops incomplete; never discard the accounting row.
There is no separate reconciliation or production budget ledger.

`get_completed` increments `cache_hits` transactionally. A `usage_missing` payload is available only
through `resume_decision(...).action == "reuse_and_stop"` and can never be treated as a successful
cache hit. `summarize_run(run_id)` returns
`RunCallSummary`,
orders by `created_at, call_id`, counts each completed call's usage/cost once, retains uncertain
reservations in committed cost, sets exact `actual_cost_micro_cny=null`, reports only the known-cost
sum as `known_actual_cost_micro_cny` with `cost_is_lower_bound=true`, and never converts missing usage
to zero. Implement recursive
redaction with this function:

```python
_SENSITIVE_KEYS = {"authorization", "api_key", "token", "secret", "password"}


def redact_payload(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if str(key).lower() in _SENSITIVE_KEYS else redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    return value
```

`TraceWriter.write` opens the file in append mode with UTF-8, writes exactly one compact JSON object plus `\n`, flushes, and calls `os.fsync`. `atomic_write_json` writes to `path.with_suffix(path.suffix + ".tmp")` and replaces the destination with `Path.replace`.

- [ ] **Step 4: Run artifact tests**

Run: `conda run -n agent-collab python -m pytest tests/test_artifacts.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit artifact persistence**

```powershell
git add src/evidence_route/artifacts.py tests/test_artifacts.py
git commit -m "feat: persist idempotent calls and redacted traces"
```

### Task 8: Build The Structured OpenAI-Compatible Adapter

**Files:**
- Create: `src/evidence_route/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write transport-independent adapter tests**

```python
# tests/test_llm.py
import pytest
from pydantic import BaseModel, ConfigDict

from evidence_route.artifacts import BillingStateError, SQLiteRunStore
from evidence_route.budget import PriceConfig
from evidence_route.config import LLMSettings
from evidence_route.llm import BillingUncertain, RawCompletion, StructuredLLM


class RoutePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: str


class FakeTransport:
    def __init__(self, responses: list[RawCompletion]) -> None:
        self.responses = responses
        self.calls = 0

    async def create(self, **request: object) -> RawCompletion:
        response = self.responses[self.calls]
        self.calls += 1
        return response


class AmbiguousAfterSendTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **request: object) -> RawCompletion:
        self.calls += 1
        raise BillingUncertain("connection lost after request handoff")


def raw(content: str, model: str = "relay-model") -> RawCompletion:
    return RawCompletion(
        content=content,
        response_model_id_raw=model,
        input_tokens=100,
        output_tokens=20,
    )


@pytest.mark.asyncio
async def test_invalid_json_is_repaired_once(tmp_path) -> None:
    transport = FakeTransport([raw("not-json"), raw('{"route":"single"}')])
    llm = make_llm(tmp_path, transport)
    result = await llm.invoke(
        run_id="run-1", node="router", task_id="root",
        messages=[{"role": "user", "content": "route this"}],
        schema=RoutePayload, max_input_tokens=1800, max_output_tokens=250,
    )
    assert result.value.route == "single"
    assert transport.calls == 2


@pytest.mark.asyncio
async def test_completed_call_is_reused_from_cache(tmp_path) -> None:
    transport = FakeTransport([raw('{"route":"single"}')])
    llm = make_llm(tmp_path, transport)
    kwargs = dict(
        run_id="run-1", node="router", task_id="root",
        messages=[{"role": "user", "content": "route this"}],
        schema=RoutePayload, max_input_tokens=1800, max_output_tokens=250,
    )
    await llm.invoke(**kwargs)
    await llm.invoke(**kwargs)
    assert transport.calls == 1


@pytest.mark.asyncio
async def test_crash_after_transmit_before_cache_write_blocks_resume(tmp_path) -> None:
    first_transport = AmbiguousAfterSendTransport()
    kwargs = dict(
        run_id="run-uncertain", node="single", task_id="root",
        messages=[{"role": "user", "content": "verify"}],
        schema=RoutePayload, max_input_tokens=1800, max_output_tokens=250,
    )
    with pytest.raises(BillingUncertain):
        await make_llm(tmp_path, first_transport).invoke(**kwargs)
    second_transport = FakeTransport([raw('{"route":"single"}')])
    with pytest.raises(BillingStateError, match="unknown billing"):
        await make_llm(tmp_path, second_transport).invoke(**kwargs)
    assert first_transport.calls == 1
    assert second_transport.calls == 0


@pytest.mark.asyncio
async def test_received_response_without_usage_is_persisted_and_not_reissued(tmp_path) -> None:
    transport = FakeTransport([
        RawCompletion(
            content='{"route":"single"}', response_model_id_raw="relay-model",
            input_tokens=None, output_tokens=None,
        )
    ])
    kwargs = dict(
        run_id="run-missing", node="router", task_id="root",
        messages=[{"role": "user", "content": "route this"}],
        schema=RoutePayload, max_input_tokens=1800, max_output_tokens=250,
    )
    with pytest.raises(UsageUnavailable):
        await make_llm(tmp_path, transport).invoke(**kwargs)
    with pytest.raises(UsageUnavailable):
        await make_llm(tmp_path, FakeTransport([])).invoke(**kwargs)
    assert transport.calls == 1


def make_llm(tmp_path, transport) -> StructuredLLM:
    pricing = PriceConfig(
        provider="test", currency="CNY", input_per_million=1.0,
        output_per_million=1.0, price_source="test", strict_evaluation=True,
    )
    return StructuredLLM(
        settings=LLMSettings(
            base_url="https://relay.example", api_key="secret", requested_alias="alias"
        ),
        transport=transport,
        run_store=SQLiteRunStore(
            tmp_path / "run-store.sqlite3", activity_id="gate-a-test",
            cap_cny=10.0, pricing=pricing,
        ),
    )
```

- [ ] **Step 2: Run tests and confirm adapter import fails**

Run: `conda run -n agent-collab python -m pytest tests/test_llm.py -q`

Expected: FAIL importing `evidence_route.llm`.

- [ ] **Step 3: Implement response metadata and call identity**

Define these exact public records:

```python
@dataclass(frozen=True)
class RawCompletion:
    content: str
    response_model_id_raw: str | None
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    value: T
    usage: Usage
    requested_alias: str
    response_model_id_raw: str | None
    identity_verified: bool
    call_ids: tuple[str, ...]
    actual_cost_micro_cny: int | None
    cache_hit: bool
    transport_attempts: int
    usage_source: Literal["provider", "missing"]
    billing_uncertain: bool
```

Generate each internal id exactly as:

```python
def make_call_id(run_id: str, node: str, task_id: str, logical_attempt: int) -> str:
    raw = f"{run_id}\0{node}\0{task_id}\0{logical_attempt}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
```

`StructuredLLM.invoke` canonicalizes the complete non-secret request (endpoint hash, requested alias,
messages, schema, and generation settings), computes `request_sha256`, derives the deterministic call
ID, and asks `SQLiteRunStore.resume_decision` before any reservation. A completed hit returns saved
usage/cost metadata without adding spend; a `reuse_and_stop` hit proves the response was already
received with missing usage and raises the typed activity stop without transport; an unresolved
`sent`/`billing_uncertain` record stops. On a
miss it reserves the node cap in `SQLiteRunStore`, marks the call sent immediately before transport
handoff, calls the transport at most three times for explicitly non-billable 429/5xx responses,
validates JSON with `schema.model_validate_json`, and issues exactly one repair logical call when
JSON/Pydantic validation fails. Authentication/configuration errors are never retried. A timeout or
connection loss after request transmission marks the call `billing_uncertain` and raises
`BillingUncertain` unless the relay provides an idempotency/status API proving it was not billed. The
repair message is:

For the two permitted transient retries, wait
`min(4.0, 0.5 * 2**attempt) * (0.75 + deterministic_fraction(call_id, attempt) * 0.5)` seconds.
Derive the fraction from SHA-256, inject the async sleeper in tests, and never use process-global
random state. Persist retry latency and transport-attempt count.

```python
{
    "role": "user",
    "content": (
        "Return only JSON matching the supplied schema. "
        f"The previous output failed validation: {validation_error}"
    ),
}
```

If usage fields are absent, create
`Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False)`, atomically persist the raw
response and identity as `usage_missing`, and only then raise `UsageUnavailable` so strict accounting
stops the campaign. Reopening the run must recover that payload and stop without a second request.
Terminal payloads must contain `activity_id`, `run_id`, node, task ID, logical
attempt, raw content, usage, usage source, requested alias, raw model ID,
`identity_verified: false`, transport-attempt count, billing uncertainty, and integer micro-CNY cost.
Persist response and actual spend through one `SQLiteRunStore.complete_call` transaction before
returning to a graph node. The production constructor accepts `run_store` only; it has no cache or
budget-store parameters that could be pointed at different files.

- [ ] **Step 4: Implement the real async OpenAI transport**

`OpenAITransport` must create
`AsyncOpenAI(api_key=..., base_url=ensure_v1(...), timeout=..., max_retries=0)` and call
`chat.completions.create`. In `json_schema` mode send the Pydantic JSON schema; in `json_object`
mode send `{"type": "json_object"}`. Normalize `usage.prompt_tokens`,
`usage.completion_tokens`, and `response.model` into `RawCompletion`. Enforce the configured
generation limits from `GenerationSettings`; do not duplicate numeric caps in this module. Do not
log the API key or request headers.

- [ ] **Step 5: Run adapter tests**

Run: `conda run -n agent-collab python -m pytest tests/test_llm.py -q`

Expected: repair, one-repair maximum, cache-before-reserve, transient retry, missing usage, and
ambiguous-billing tests pass without a network call. `test_each_billable_retry_is_reserved_and_recorded`
asserts that a validation repair uses the next deterministic logical call ID and its own reservation;
non-billable transport retries stay on the original reservation while incrementing
`transport_attempts`.

- [ ] **Step 6: Commit the adapter**

```powershell
git add src/evidence_route/llm.py tests/test_llm.py
git commit -m "feat: add structured OpenAI-compatible adapter"
```

### Task 9: Implement The Hybrid Router

**Files:**
- Create: `src/evidence_route/routing.py`
- Create: `tests/test_routing.py`

- [ ] **Step 1: Write rule precedence and fallback tests**

```python
# tests/test_routing.py
import pytest

from evidence_route.config import GenerationSettings, RoutingSettings, stable_hash
from evidence_route.contracts import ClaimFeatures, ClaimUnit, Strategy
from evidence_route.routing import HybridRouter


def features(**overrides) -> ClaimFeatures:
    values = dict(
        claim_units=[ClaimUnit(unit_id="u0", text="Atomic claim")],
        atomic_clause_count=1,
        entity_count=1,
        numeric_count=0,
        time_scope_count=0,
        has_comparison=False,
        has_causal=False,
        has_contrast=False,
        probe_source_count=2,
        probe_score_spread=1.0,
        probe_conflict_hint=False,
    )
    values.update(overrides)
    return ClaimFeatures.model_validate(values)


@pytest.mark.asyncio
async def test_clear_multi_precedes_clear_single() -> None:
    router = HybridRouter(RoutingSettings(), GenerationSettings(), llm=None)
    decision = await router.route("run", Strategy.ADAPTIVE, features(
        atomic_clause_count=3,
        claim_units=[ClaimUnit(unit_id=f"u{i}", text=f"claim {i}") for i in range(3)],
    ))
    assert decision.route == "multi"
    assert decision.source == "rule"


@pytest.mark.asyncio
async def test_atomic_claim_uses_single_rule() -> None:
    decision = await HybridRouter(RoutingSettings(), GenerationSettings(), llm=None).route(
        "run", Strategy.ADAPTIVE, features()
    )
    assert decision.route == "single"


@pytest.mark.asyncio
async def test_fixed_strategy_never_calls_llm() -> None:
    decision = await HybridRouter(RoutingSettings(), GenerationSettings(), llm=None).route(
        "run", Strategy.ALWAYS_MULTI, features()
    )
    assert decision.route == "multi"
    assert decision.source == "strategy"
```

- [ ] **Step 2: Run routing tests and verify import failure**

Run: `conda run -n agent-collab python -m pytest tests/test_routing.py -q`

Expected: FAIL importing `evidence_route.routing`.

- [ ] **Step 3: Implement rules, uncertain LLM routing, and conservative fallback**

Use this rule order:

```python
if strategy == Strategy.ALWAYS_MULTI:
    return decision("multi", "strategy", ["fixed_multi"])
if strategy == Strategy.ALWAYS_SINGLE:
    return decision("single", "strategy", ["fixed_single"])
if clear_multi(features, settings):
    return decision("multi", "rule", ["compound_or_conflict"])
if clear_single(features, settings):
    return decision("single", "rule", ["atomic_with_sources"])
try:
    payload = await llm.invoke(
        run_id=run_id,
        node="router",
        task_id="root",
        messages=router_messages(features),
        schema=RouterPayload,
        max_input_tokens=generation.router.max_input_tokens,
        max_output_tokens=generation.router.max_output_tokens,
    )
    return decision(payload.value.route, "llm", payload.value.reason_codes)
except StructuredCallError:
    return decision("multi", "fallback", ["router_fallback"])
```

`clear_multi` is true for at least three clauses, or at least two clauses plus comparison, multiple time scopes, or conflict hint. `clear_single` is true for one clause, no comparison/causal/contrast/conflict hint, and at least `clear_single_min_sources`. The decision config hash is `stable_hash(settings.model_dump(mode="json"))`.
`HybridRouter` receives both `RoutingSettings` and `GenerationSettings`; its LLM call uses the
configured router limits shown above. Fixed strategies and clear rules never invoke the LLM.

- [ ] **Step 4: Add an uncertain-router fake LLM test**

Add a fake whose `invoke` returns `RouterPayload(route="single", reason_codes=["bounded_claim"], explanation="One source question")`; assert the router uses `source == "llm"`. Add a fake raising `StructuredCallError`; assert multi fallback and `router_fallback`.

- [ ] **Step 5: Run router tests**

Run: `conda run -n agent-collab python -m pytest tests/test_routing.py -q`

Expected: rules, uncertain routing, and fallback tests pass.

- [ ] **Step 6: Commit router**

```powershell
git add src/evidence_route/routing.py tests/test_routing.py
git commit -m "feat: route claims with rules and structured fallback"
```

### Task 10: Implement Verification Components And Structural Validation

**Files:**
- Create: `src/evidence_route/prompts.py`
- Create: `src/evidence_route/verification.py`
- Create: `src/evidence_route/validation.py`
- Modify: `src/evidence_route/contracts.py`
- Create: `tests/test_verification.py`
- Create: `tests/test_validation.py`

- [ ] **Step 1: Add LLM draft contracts**

Add strict `VerdictDraft`, `DecompositionDraft`, and `WorkerDraft` models. `VerdictDraft` has
`verdict`, `confidence`, `rationale`, and citations; `DecompositionDraft` has one to three
`VerificationTask` values and a model validator requiring unique `task_id` values; `WorkerDraft`
has verdict, confidence, citations, and errors. `ClaimDecomposer` additionally rejects any
`claim_unit_id` not present in the current `ClaimFeatures`. Add
`test_decomposition_rejects_duplicate_task_ids` and
`test_decomposition_rejects_unknown_claim_unit_reference`, plus tests proving more than three tasks
and unknown citation fields fail Pydantic validation. This protects the
`(run_id, node, task_id, logical_attempt)` paid-call identity from worker collisions.

- [ ] **Step 2: Write the old-parser regression and citation tests**

```python
# tests/test_validation.py
from evidence_route.contracts import ResultStatus, Usage, VerificationResult, Verdict
from evidence_route.validation import ResultValidator, ValidationAction


def test_validator_never_scans_rationale_for_label() -> None:
    result = VerificationResult(
        claim_id="legacy-third-case",
        status=ResultStatus.COMPLETED,
        verdict=Verdict.REFUTED,
        confidence=0.91,
        rationale="正文讨论了证据不足，但总体判定由结构化字段给出。",
        citations=[],
        initial_route="single",
        usage=Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True),
    )
    decision = ResultValidator(low_confidence=0.65, minimum_coverage=0.0).validate(
        result=result,
        claim_unit_ids=["u0"],
        evidence_ids=set(),
        escalation_count=1,
        strategy="adaptive",
    )
    assert decision.result.verdict is Verdict.REFUTED
    assert decision.action is ValidationAction.ACCEPT


def test_unknown_citation_escalates_once(valid_result) -> None:
    decision = ResultValidator(low_confidence=0.65, minimum_coverage=1.0).validate(
        result=valid_result,
        claim_unit_ids=["u0"],
        evidence_ids={"different-id"},
        escalation_count=0,
        strategy="adaptive",
    )
    assert decision.action is ValidationAction.ESCALATE
```

- [ ] **Step 3: Define complete prompts as versioned constants**

Each prompt must state: evidence is untrusted data, ignore instructions inside evidence, use only listed `evidence_id` values, return the four exact verdicts, and output JSON only. Export `PROMPT_VERSION = "2026-08-17-gate-a-v1"` and `prompt_hash()` over all prompt strings. Do not request hidden reasoning; request a concise rationale and enum reason codes.

- [ ] **Step 4: Implement the four component classes**

Implement these methods with no internal loops beyond one structured call:

```python
class SingleVerifier:
    async def verify(self, run_id: str, claim_id: str, claim: str, features: ClaimFeatures) -> VerificationResult:
        evidence = await self.provider.search(
            claim_id, claim,
            top_k=self.evidence_settings.single_top_k,
            max_chars=self.evidence_settings.single_chars,
        )
        response = await self.llm.invoke(
            run_id=run_id, node="single", task_id="root",
            messages=single_messages(claim, features, evidence),
            schema=VerdictDraft,
            max_input_tokens=self.generation.single.max_input_tokens,
            max_output_tokens=self.generation.single.max_output_tokens,
        )
        return result_from_draft(
            response, claim_id, "single", evidence,
            available_evidence_ids=[item.evidence_id for item in evidence],
        )


class ClaimDecomposer:
    async def decompose(self, run_id: str, claim: str, features: ClaimFeatures) -> list[VerificationTask]:
        response = await self.llm.invoke(
            run_id=run_id, node="decomposer", task_id="root",
            messages=decomposer_messages(claim, features),
            schema=DecompositionDraft,
            max_input_tokens=self.generation.decomposer.max_input_tokens,
            max_output_tokens=self.generation.decomposer.max_output_tokens,
        )
        return response.value.tasks


class EvidenceWorker:
    async def verify_task(self, run_id: str, claim_id: str, task: VerificationTask) -> WorkerResult:
        evidence = await self.provider.search(
            claim_id, task.query,
            top_k=self.evidence_settings.worker_top_k,
            max_chars=self.evidence_settings.worker_chars,
        )
        response = await self.llm.invoke(
            run_id=run_id, node="worker", task_id=task.task_id,
            messages=worker_messages(task, evidence),
            schema=WorkerDraft,
            max_input_tokens=self.generation.worker.max_input_tokens,
            max_output_tokens=self.generation.worker.max_output_tokens,
        )
        return worker_result_from_draft(
            response, task, evidence,
            available_evidence_ids=[item.evidence_id for item in evidence],
        )


class VerdictJudge:
    async def judge(
        self, run_id: str, claim_id: str, claim: str, workers: list[WorkerResult],
        *, initial_route: Literal["single", "multi"], escalated: bool,
    ) -> VerificationResult:
        response = await self.llm.invoke(
            run_id=run_id, node="judge", task_id="root",
            messages=judge_messages(claim, workers),
            schema=VerdictDraft,
            max_input_tokens=self.generation.judge.max_input_tokens,
            max_output_tokens=self.generation.judge.max_output_tokens,
        )
        return result_from_worker_draft(
            response, claim_id, workers,
            initial_route=initial_route, escalated=escalated,
            available_evidence_ids=sorted({
                evidence_id for worker in workers
                for evidence_id in worker.available_evidence_ids
            }),
        )
```

Deduplicate judge citations by `evidence_id`, cap them at
`self.evidence_settings.judge_max_evidence`, and mark the result partial if any worker is
partial/failed. A failed worker remains in the three-item judge input with its explicit error; an
infrastructure failure is never converted to `Not Enough Evidence`. Populate allowed evidence IDs
only from actual provider returns, never from draft citations. Aggregate worker/judge usage and
errors structurally; Task 11 replaces final run usage/cost with the complete call-cache summary so
router, decomposer, repair, and resumed calls are also counted.

Define typed component boundaries instead of allowing ordinary provider/adapter exceptions to tear
down the campaign. A worker-local retrieval, explicitly non-billable transport, or exhausted
structured-output error becomes a `WorkerResult(status=FAILED, verdict=None, confidence=None)` with
the typed error code and exact call-store usage. A critical claim-local probe, single, decomposer, or
judge failure becomes `VerificationResult(status=FAILED, verdict=None, confidence=None)`. A probe
failure before routing has `initial_route=None`, `failure_stage="pre_route"`, and typed
`PROBE_RETRIEVAL_FAILED`; it may have zero calls, `Usage(0, 0, 0, complete=True)`, and zero cost. All
later failures retain the actual initial route and failure stage.
`BudgetExceeded`, `UsageUnavailable`, `BillingUncertain`, and model-identity drift are safety-stop
exceptions and must escape these boundaries to Task 17. Add
`test_worker_retrieval_failure_becomes_failed_worker`,
`test_probe_failure_before_llm_becomes_zero_call_failed_result`, and
`test_usage_and_billing_safety_stops_are_not_converted_to_failed_predictions`.

- [ ] **Step 5: Implement validation actions**

`ResultValidator.validate` returns `ACCEPT`, `ESCALATE`, or `FAIL`. It checks verdict/status
consistency, every citation against `result.available_evidence_ids`, citation claim unit IDs,
confidence, and structural coverage. Only an adaptive draft whose `draft_origin == "single"` and
`escalation_count == 0` may escalate. `always_single` and every multi draft cannot escalate. A
second invalid result becomes failed with `verdict=None`, `confidence=None`, and explicit validation
errors. Tests must prove an unknown citation cannot make itself valid by appearing in the draft.

- [ ] **Step 6: Run component and validation tests**

Run:

```powershell
conda run -n agent-collab python -m pytest tests/test_contracts.py tests/test_verification.py tests/test_validation.py -q
```

Expected: all draft, component, citation, coverage, and regression tests pass.

- [ ] **Step 7: Commit verification components**

```powershell
git add src/evidence_route/contracts.py src/evidence_route/prompts.py src/evidence_route/verification.py src/evidence_route/validation.py tests/test_contracts.py tests/test_verification.py tests/test_validation.py
git commit -m "feat: verify claims with structured evidence contracts"
```

### Task 11: Compile The Bounded LangGraph Workflow

**Files:**
- Modify: `src/evidence_route/contracts.py`
- Create: `src/evidence_route/graph.py`
- Create: `tests/test_graph.py`
- Create: `tests/test_graph_recovery.py`

- [ ] **Step 1: Extend state with observable graph records**

Add these strict models to `src/evidence_route/contracts.py` and add `status`, `usage`,
`node_timings`, `escalated`, `draft_origin`, and the conditional-edge channel to
`VerificationState`:

```python
class NodeTiming(StrictModel):
    node: str
    started_at: str
    finished_at: str
    latency_ms: int = Field(ge=0)
    cache_hit: bool = False


class VerificationState(TypedDict, total=False):
    run_id: str
    claim_id: str
    claim_text: str
    language: str
    strategy: Strategy
    status: RunStatus
    probe_evidence: list[Evidence]
    claim_features: ClaimFeatures
    route_decision: RouteDecision
    tasks: list[VerificationTask]
    worker_results: Annotated[list[WorkerResult], operator.add]
    draft_result: VerificationResult
    draft_origin: Literal["single", "multi"]
    validation_action: Literal["accept", "escalate", "fail"]
    final_result: VerificationResult
    escalation_count: int
    escalated: bool
    usage: Usage
    node_timings: Annotated[list[NodeTiming], operator.add]
    errors: Annotated[list[str], operator.add]
```

Add contract tests asserting two parallel state updates concatenate `worker_results`,
`node_timings`, and `errors` rather than replacing prior values.

- [ ] **Step 2: Write graph-path, fan-out, and recovery tests**

Use fakes with the same async method signatures as Tasks 5, 9, and 10. The worker fake blocks on
an `asyncio.Event` until all three tasks have started, proving the graph uses LangGraph fan-out
rather than a sequential Python loop.

```python
# tests/test_graph.py
import asyncio
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from evidence_route.contracts import ResultStatus, Strategy, Verdict
from evidence_route.graph import GraphComponents, build_graph, initial_state


@pytest.mark.asyncio
async def test_single_result_finishes_without_multi(graph_fakes) -> None:
    graph = build_graph(graph_fakes.components, checkpointer=InMemorySaver())
    state = await graph.ainvoke(
        initial_state("run-single", "dev-0", "Atomic claim", Strategy.ALWAYS_SINGLE),
        config={"configurable": {"thread_id": "run-single"}},
    )
    assert state["final_result"].status is ResultStatus.COMPLETED
    assert state["final_result"].verdict is Verdict.SUPPORTED
    assert graph_fakes.decomposer.calls == 0


@pytest.mark.asyncio
async def test_adaptive_single_escalates_exactly_once(graph_fakes) -> None:
    graph_fakes.validator.actions = ["escalate", "accept"]
    graph = build_graph(graph_fakes.components, checkpointer=InMemorySaver())
    state = await graph.ainvoke(
        initial_state("run-escalate", "dev-1", "Compound claim", Strategy.ADAPTIVE),
        config={"configurable": {"thread_id": "run-escalate"}},
    )
    assert state["escalation_count"] == 1
    assert state["final_result"].escalated is True
    assert state["final_result"].initial_route == "single"
    assert graph_fakes.single.calls == 1
    assert graph_fakes.decomposer.calls == 1


@pytest.mark.asyncio
async def test_three_workers_start_before_any_finishes(graph_fakes) -> None:
    graph_fakes.router.route_name = "multi"
    graph_fakes.decomposer.task_count = 3
    graph = build_graph(graph_fakes.components, checkpointer=InMemorySaver())
    await graph.ainvoke(
        initial_state("run-parallel", "dev-2", "Three-part claim", Strategy.ALWAYS_MULTI),
        config={"configurable": {"thread_id": "run-parallel"}},
    )
    assert graph_fakes.worker.maximum_active == 3
    assert sorted(graph_fakes.worker.started_task_ids) == ["t0", "t1", "t2"]


@pytest.mark.asyncio
async def test_resume_reuses_completed_workers(graph_fakes) -> None:
    graph_fakes.router.route_name = "multi"
    graph_fakes.judge.failures_remaining = 1
    checkpointer = InMemorySaver()
    graph = build_graph(graph_fakes.components, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "run-resume"}}
    with pytest.raises(RuntimeError, match="injected judge failure"):
        await graph.ainvoke(
            initial_state("run-resume", "dev-3", "Claim", Strategy.ALWAYS_MULTI),
            config=config,
        )
    state = await graph.ainvoke(None, config=config)
    assert state["final_result"].status in {ResultStatus.COMPLETED, ResultStatus.PARTIAL}
    assert graph_fakes.worker.calls == 3
    assert graph_fakes.judge.calls == 2
```

Add three graph tests: one failed worker still gives Judge all three worker records and forces the
final status to `PARTIAL`; a validator that returns `ESCALATE` twice yields `FAILED` and calls the
decomposer only once; and a single-to-multi result preserves `initial_route="single"` plus
`escalated=true`.

In `tests/test_graph_recovery.py`, use a real `StructuredLLM`, `SQLiteRunStore`, fake transport, and
`AsyncSqliteSaver`. Inject a crash after the single response is persisted but before its node update
is checkpointed. Close/reopen both SQLite stores, resume with the same `run_id/thread_id`, and prove
the response is reused:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@pytest.mark.asyncio
async def test_async_sqlite_resume_reuses_paid_call(tmp_path, recovery_runtime) -> None:
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "run-paid-resume"}}
    recovery_runtime.crash_after_single = True
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_graph(recovery_runtime.components, checkpointer=saver)
        with pytest.raises(RuntimeError, match="after run-store completion"):
            await graph.ainvoke(
                initial_state("run-paid-resume", "dev-0", "Claim", Strategy.ALWAYS_SINGLE),
                config=config,
            )
    recovery_runtime.crash_after_single = False
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_graph(recovery_runtime.components, checkpointer=saver)
        state = await graph.ainvoke(None, config=config)
        snapshot = await graph.aget_state(config)
    assert state["final_result"].status is ResultStatus.COMPLETED
    assert snapshot.values["run_id"] == "run-paid-resume"
    assert recovery_runtime.transport.calls == 1
    assert recovery_runtime.run_store.summarize_run("run-paid-resume").usage.complete is True
```

- [ ] **Step 3: Run graph tests and confirm the graph module is absent**

Run: `conda run -n agent-collab python -m pytest tests/test_graph.py -q`

Expected: FAIL importing `evidence_route.graph`.

- [ ] **Step 4: Implement graph components and exact branch functions**

Create `src/evidence_route/graph.py` with this dependency boundary and initial state:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from evidence_route.analyzer import analyze_claim
from evidence_route.contracts import RunStatus, Strategy, Usage, VerificationState


@dataclass(frozen=True)
class GraphComponents:
    provider: Any
    router: Any
    single: Any
    decomposer: Any
    worker: Any
    judge: Any
    validator: Any
    evidence_settings: Any
    run_store: Any
    trace: Any


def initial_state(
    run_id: str, claim_id: str, claim_text: str, strategy: Strategy, language: str = "auto"
) -> VerificationState:
    return {
        "run_id": run_id,
        "claim_id": claim_id,
        "claim_text": claim_text,
        "language": language,
        "strategy": strategy,
        "status": RunStatus.RUNNING,
        "escalation_count": 0,
        "escalated": False,
        "worker_results": [],
        "usage": Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=True),
        "node_timings": [],
        "errors": [],
    }


def route_after_router(state: VerificationState) -> Literal["single", "decompose"]:
    return "single" if state["route_decision"].route == "single" else "decompose"


def fan_out_workers(state: VerificationState) -> list[Send]:
    return [
        Send(
            "worker",
            {
                "run_id": state["run_id"],
                "claim_id": state["claim_id"],
                "task": task,
            },
        )
        for task in state["tasks"][:3]
    ]


def route_after_validation(state: VerificationState) -> Literal["decompose", "end"]:
    return "decompose" if state.get("validation_action") == "escalate" else "end"
```

Define a private `WorkerInput(TypedDict)` for the payload emitted by `Send`. Implement async nodes
`analyze`, `route`, `single`, `decompose`, `worker`, `judge`, and `validate`. `analyze` first makes
the bounded probe call, then calls `analyze_claim`; `worker` returns exactly
`{"worker_results": [result]}` so the reducer merges fan-out results. `single` and `judge` set
`draft_origin` explicitly. `validate` writes `validation_action` plus an accepted or failed result;
on the first eligible `ESCALATE`, it increments `escalation_count` and sets `escalated=True`.
Do not return `worker_results=[]`: its `operator.add` reducer cannot clear prior values, and the only
legal escalation occurs before any worker fan-out. A second escalation request is converted to a
failed result. The node wrappers enforce Task 10's exception taxonomy: expected worker-local errors
return failed workers; expected claim-local errors produce a terminal failed result (including a
zero-call `initial_route=None, failure_stage="pre_route"` result when probe retrieval fails);
safety-stop exceptions propagate unchanged to the
campaign runner. No exception handler may synthesize `Not Enough Evidence` from infrastructure
failure. Before accepting the final result, read
`summary = components.run_store.summarize_run(run_id)` and replace only the result's usage,
`estimated_cost_micro_cny`, currency, and price ID, so router, repair, decomposer, resumed, and judge
calls are included exactly once. Model IDs, aliases, call IDs, cache counts, attempts, and billing
state are not fields on `VerificationResult`; the campaign/single-run executor uses the same typed
`RunCallSummary` after graph completion to construct `RunArtifact`. Wrap every node with one
timing/trace helper that records an ISO-8601 UTC start/end, integer latency, status, initial route,
escalation/reason codes, provider, requested alias, raw model ID, `identity_verified=false`, call IDs,
cache hit, transport attempts, usage/cost, evidence IDs, and redacted errors. Evidence text is
delimited as untrusted data in prompts; traces contain no hidden chain-of-thought.
When `summary.usage.complete` is false, set the result's estimated cost, currency, and price ID to
`None`; the executor persists diagnostics and stops the strict activity instead of reporting zero.
The terminal validator also maps `ResultStatus.COMPLETED/PARTIAL/FAILED` to the matching graph
`RunStatus`; no terminal graph state remains `running`.

Compile the graph exactly once through this public function:

```python
def build_graph(components: GraphComponents, *, checkpointer: Any):
    builder = StateGraph(VerificationState)
    builder.add_node("analyze", make_analyze_node(components))
    builder.add_node("route", make_route_node(components))
    builder.add_node("single", make_single_node(components))
    builder.add_node("decompose", make_decompose_node(components))
    builder.add_node("worker", make_worker_node(components))
    builder.add_node("judge", make_judge_node(components))
    builder.add_node("validate", make_validate_node(components))
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "route")
    builder.add_conditional_edges(
        "route", route_after_router, {"single": "single", "decompose": "decompose"}
    )
    builder.add_edge("single", "validate")
    builder.add_conditional_edges("decompose", fan_out_workers, ["worker"])
    builder.add_edge("worker", "judge")
    builder.add_edge("judge", "validate")
    builder.add_conditional_edges(
        "validate", route_after_validation, {"decompose": "decompose", "end": END}
    )
    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 5: Run graph and upstream tests**

Run:

```powershell
conda run -n agent-collab python -m pytest tests/test_contracts.py tests/test_graph.py tests/test_graph_recovery.py tests/test_validation.py -q
```

Expected: single, one-time escalation, true three-worker fan-out, partial worker handling, in-memory
resume, async SQLite persistence, paid-call cache recovery, and run-level usage aggregation tests
all pass.

- [ ] **Step 6: Commit the workflow**

```powershell
git add src/evidence_route/contracts.py src/evidence_route/graph.py tests/test_contracts.py tests/test_graph.py tests/test_graph_recovery.py
git commit -m "feat: orchestrate bounded verification with LangGraph"
```

### Task 12: Expose Verify And Provider Capability Commands

**Files:**
- Create: `src/evidence_route/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write offline CLI and paid-smoke guards**

```python
# tests/test_cli.py
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
    runner = CliRunner()
    app = create_app(FakeServices())
    result = runner.invoke(app, [
        "verify", "--claim-id", "dev-0", "--claim", "The claim",
        "--strategy", "adaptive", "--run-id", "run-cli",
        "--artifact-dir", str(tmp_path),
    ])
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
    result = CliRunner().invoke(create_app(FakeServices()), [
        "verify", "--claim-id", "dev-0", "--claim", "Claim", "--resume",
        "--artifact-dir", str(tmp_path),
    ])
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
```

- [ ] **Step 2: Run CLI tests and confirm the module is absent**

Run: `conda run -n agent-collab python -m pytest tests/test_cli.py -q`

Expected: FAIL importing `evidence_route.cli`.

- [ ] **Step 3: Implement dependency-injected Typer commands**

Create `CliServices` with synchronous `verify` and `provider_smoke` methods; the production
implementation may call async code only through `asyncio.run` at this boundary. `create_app` must
return a fresh `typer.Typer` so tests never mutate global command state.

```python
class CliServices(Protocol):
    def verify(
        self, *, run_id: str, resume: bool, claim_id: str, claim: str,
        strategy: str, artifact_dir: Path,
        config_path: Path, pricing_path: Path | None, corpus_dir: Path,
        checkpoint_db: Path,
    ) -> dict[str, object]: ...

    def provider_smoke(
        self, *, nonce: str, config_path: Path, pricing_path: Path, artifact_dir: Path,
    ) -> dict[str, object]: ...
```

The `verify` command exposes exact options `--run-id`, `--resume`, `--claim-id`, `--claim`,
`--strategy`, `--corpus-dir`, `--config`, `--pricing`, `--artifact-dir`, and `--checkpoint-db`.
A fresh command creates a UUIDv4 when `--run-id` is absent and rejects an existing thread ID;
`--resume` requires an existing explicit run ID and matching claim/config hashes. Production code
opens `AsyncSqliteSaver.from_conn_string` inside `async with`, invokes the graph with
`{"configurable": {"thread_id": run_id}}`, and passes `None` instead of initial state on resume.
It creates one `SQLiteRunStore` at `<artifact-dir>/<run-id>/run-store.sqlite3` with activity ID equal
to the run ID; there is no independent cache/ledger option.
It atomically writes `result.json` and prints the same JSON. A missing corpus, mismatched resume,
invalid strategy, failed final artifact write, or graph result without `final_result` returns a
non-zero exit code.

`provider-smoke` requires `--accept-paid-call`, sends one small request with this strict schema,
and fails unless structured parsing succeeds and provider usage contains both input and output
tokens:

```python
class CapabilityPayload(StrictModel):
    ok: Literal[True]
    nonce: str = Field(min_length=8, max_length=64)
```

The nonce is generated locally and must round-trip. Validate output again as a strict
`ProviderSmokeResult` containing nonce, requested alias, one non-empty raw model ID,
`identity_verified: Literal[False]`, input/output tokens, `usage_complete: Literal[True]`, call IDs,
endpoint config hash, and integer `estimated_cost_micro_cny`. Missing usage or a mismatched nonce fails
the command.
The printed record never contains environment values, headers, or raw request objects. Append
`if __name__ == "__main__": app()` only after constructing
`app = create_app(ProductionServices())`.

- [ ] **Step 4: Run CLI and help tests**

Run:

```powershell
conda run -n agent-collab python -m pytest tests/test_cli.py -q
conda run -n agent-collab evidence-route --help
conda run -n agent-collab evidence-route verify --help
```

Expected: tests pass; both help commands exit 0 without requiring API environment variables.

- [ ] **Step 5: Commit the first CLI surface**

```powershell
git add src/evidence_route/cli.py tests/test_cli.py
git commit -m "feat: expose verification and provider smoke commands"
```

### Task 13: Prepare Pinned, Gold-Isolated AVeriTeC Data

**Files:**
- Create: `data/sources/averitec.json`
- Create: `scripts/prepare_averitec.py`
- Create: `tests/fixtures/averitec/source/train.json`
- Create: `tests/fixtures/averitec/source/dev.json`
- Create: `tests/fixtures/averitec/__init__.py`
- Create: `tests/fixtures/averitec/fakes.py`
- Create: `tests/test_prepare_averitec.py`
- Create at preparation time: `data/manifests/averitec_calibration_runtime.json`
- Create at preparation time: `data/manifests/averitec_dev_runtime.json`
- Create at preparation time: `data/manifests/averitec_stability_runtime.json`
- Create at preparation time: `data/scorer_manifests/averitec_calibration_gold.json`
- Create at preparation time: `data/scorer_manifests/averitec_dev_gold.json`

- [ ] **Step 1: Commit exact upstream identities and licenses**

Create `data/sources/averitec.json` with no shortened hashes:

```json
{
  "dataset": "AVeriTeC",
  "license": "CC BY-NC 4.0",
  "huggingface_repo": "chenxwh/AVeriTeC",
  "huggingface_revision": "2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031",
  "paper_repository": "https://github.com/MichSchli/AVeriTeC",
  "paper_commit": "7c62d1ec8df3fb560d6efe2b85fa191135636f81",
  "metadata": {
    "data/train.json": {
      "sha256": "ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c",
      "size": 10184813
    },
    "data/dev.json": {
      "sha256": "499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300",
      "size": 1785475
    }
  },
  "knowledge_store": {
    "data_store/knowledge_store/dev_knowledge_store.zip": {
      "lfs_oid_sha256": "021e258cd6fb5fe6d627a4667d663e95c184c966939c15124df9206142fc2212",
      "size": 11537899362
    },
    "data_store/knowledge_store/train/train_0_999.zip": {
      "lfs_oid_sha256": "389d2284c5f25410205ec530c2101c3e6d575eae460731e1909ad19192dcb810",
      "size": 20730339734
    },
    "data_store/knowledge_store/train/train_1000_1999.zip": {
      "lfs_oid_sha256": "e2bf5c1dccc855cf3474ccd3f14678fe60f21e6c69685a34d7bc884996fec446",
      "size": 21273604733
    },
    "data_store/knowledge_store/train/train_2000_3067.zip": {
      "lfs_oid_sha256": "d3e27cc4072cd085dbac44272f1ea918dd3ff970b39d94f143e40c51bbde43d0",
      "size": 21520772573
    }
  }
}
```

- [ ] **Step 2: Write deterministic selection and leakage tests**

The tiny fixtures contain at least two examples for each of the four labels. Tests call pure
functions, never the network.

`tests/fixtures/averitec/source/train.json` is this exact eight-row array (the dev fixture uses the
same fields with claims `dev-0` through `dev-7`):

```json
[
  {"claim":"train-0","label":"Supported","questions":[],"justification":"j0","claim_types":[]},
  {"claim":"train-1","label":"Refuted","questions":[],"justification":"j1","claim_types":[]},
  {"claim":"train-2","label":"Not Enough Evidence","questions":[],"justification":"j2","claim_types":[]},
  {"claim":"train-3","label":"Conflicting Evidence/Cherrypicking","questions":[],"justification":"j3","claim_types":[]},
  {"claim":"train-4","label":"Supported","questions":[],"justification":"j4","claim_types":[]},
  {"claim":"train-5","label":"Refuted","questions":[],"justification":"j5","claim_types":[]},
  {"claim":"train-6","label":"Not Enough Evidence","questions":[],"justification":"j6","claim_types":[]},
  {"claim":"train-7","label":"Conflicting Evidence/Cherrypicking","questions":[],"justification":"j7","claim_types":[]}
]
```

`tests/fixtures/averitec/source/dev.json` is:

```json
[
  {"claim":"dev-0","label":"Supported","questions":[],"justification":"j0","claim_types":[]},
  {"claim":"dev-1","label":"Refuted","questions":[],"justification":"j1","claim_types":[]},
  {"claim":"dev-2","label":"Not Enough Evidence","questions":[],"justification":"j2","claim_types":[]},
  {"claim":"dev-3","label":"Conflicting Evidence/Cherrypicking","questions":[],"justification":"j3","claim_types":[]},
  {"claim":"dev-4","label":"Supported","questions":[],"justification":"j4","claim_types":[]},
  {"claim":"dev-5","label":"Refuted","questions":[],"justification":"j5","claim_types":[]},
  {"claim":"dev-6","label":"Not Enough Evidence","questions":[],"justification":"j6","claim_types":[]},
  {"claim":"dev-7","label":"Conflicting Evidence/Cherrypicking","questions":[],"justification":"j7","claim_types":[]}
]
```

```python
# tests/test_prepare_averitec.py
import json
from pathlib import Path

import pytest

from scripts.prepare_averitec import (
    normalize_member,
    select_balanced_ids,
    split_runtime_and_gold,
    stream_selected_member,
)
from tests.fixtures.averitec.fakes import FakeRemoteZip


LABELS = [
    "Supported", "Refuted", "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


def test_balanced_ids_are_hash_stable() -> None:
    rows = json.loads(Path("tests/fixtures/averitec/source/train.json").read_text("utf-8"))
    first = select_balanced_ids(rows, split="train", per_label=1, seed=20260817)
    second = select_balanced_ids(rows, split="train", per_label=1, seed=20260817)
    assert first == second
    assert first == [3, 4, 5, 6]
    assert {rows[index]["label"] for index in first} == set(LABELS)


def test_runtime_manifest_has_no_gold_fields() -> None:
    row = {
        "claim": "A claim", "label": "Refuted", "questions": [{"question": "why"}],
        "justification": "gold explanation",
    }
    runtime, gold = split_runtime_and_gold("dev", 7, row)
    assert set(runtime) == {"claim_id", "original_id", "claim", "split"}
    assert not ({"label", "questions", "justification", "gold", "claim_types"} & runtime.keys())
    assert gold["label"] == "Refuted"
    assert gold["claim"] == "A claim"


def test_member_normalization_discards_type_and_query() -> None:
    source = [{
        "claim_id": "7", "type": "gold", "query": "annotator question",
        "url": "https://example.org/a", "url2text": ["Sentence one.", "Sentence two."],
    }]
    records = list(normalize_member("dev", 7, source))
    text = json.dumps(records, ensure_ascii=False).lower()
    assert "annotator question" not in text
    assert '"type"' not in text
    assert set(records[0]) == {
        "evidence_id", "title", "source_url", "text", "snapshot_sha256",
    }
    assert records[0]["evidence_id"] == "av:dev:7:0:0"


def test_member_guard_rejects_oversized_or_unlisted_entries() -> None:
    fake_archive = FakeRemoteZip()
    fake_archive.add("output_dev/7.json", file_size=268_435_457)
    with pytest.raises(ValueError, match="member exceeds 268435456 bytes"):
        stream_selected_member(
            fake_archive, "output_dev/7.json",
            allowed={"output_dev/7.json"}, max_uncompressed_bytes=268_435_456,
        )
    assert fake_archive.extract_calls == 0
    assert fake_archive.extractall_calls == 0
```

- [ ] **Step 3: Run preparation tests and confirm the script is absent**

Run: `conda run -n agent-collab python -m pytest tests/test_prepare_averitec.py -q`

Expected: FAIL importing `scripts.prepare_averitec`.

- [ ] **Step 4: Implement stable manifests and streaming normalization**

Implement selection with the original JSON array index as `original_id`:

```python
def select_balanced_ids(
    rows: list[dict[str, object]], *, split: str, per_label: int, seed: int
) -> list[int]:
    by_label: dict[str, list[int]] = {label: [] for label in LABELS}
    for original_id, row in enumerate(rows):
        by_label[str(row["label"])].append(original_id)
    selected: list[int] = []
    for label in LABELS:
        ranked = sorted(
            by_label[label],
            key=lambda item: hashlib.sha256(
                f"{seed}:{split}:{label}:{item}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < per_label:
            raise ValueError(f"{split}:{label} has only {len(ranked)} rows")
        selected.extend(ranked[:per_label])
    return sorted(selected)
```

`split_runtime_and_gold` returns only `claim_id`, `original_id`, `claim`, and `split` to runtime;
the scorer object retains `claim`, `label`, `questions`, `justification`, and `claim_types` exactly
as required by both pinned evaluators. Forbidden-field checks compare exact mapping keys recursively;
they never reject an ordinary claim merely because its value contains the substring “gold”.
AVeriTeC has no explicit claim ID, so `claim_id` is exactly `f"{split}-{original_id}"`.
This is an intermediate row. After its corpus is atomically written, preparation adds
`claim_sha256`, corpus relative path/SHA/bytes/record count, schema version, source metadata hash,
revision, and seed to the final runtime manifest. The scorer manifest stores the final runtime
manifest SHA-256 so alignment is cryptographically bound rather than based only on filenames.

For each selected ID, open only its member with `remotezip.RemoteZip`; never download a complete
11-21 GB ZIP. Use these exact archive mappings:

```python
def archive_member(split: str, original_id: int) -> tuple[str, str]:
    if split == "dev":
        return "data_store/knowledge_store/dev_knowledge_store.zip", f"output_dev/{original_id}.json"
    if original_id < 1000:
        return "data_store/knowledge_store/train/train_0_999.zip", f"{original_id}.json"
    if original_id < 2000:
        return "data_store/knowledge_store/train/train_1000_1999.zip", f"{original_id}.json"
    return (
        "data_store/knowledge_store/train/train_2000_3067.zip",
        f"data_store/train/{original_id}.json",
    )
```

Although each member ends in `.json`, parse it line-by-line as JSONL. `normalize_member` reads only
`url` and `url2text`, emits one non-empty sentence per record, derives `title` from the URL hostname,
and uses `av:{split}:{original_id}:{source_index}:{sentence_index}`. It must reject local/private
URLs and never serialize `type`, `query`, label, annotated question, justification, or a gold flag.
Write each claim corpus atomically to
`data/processed/averitec/corpora/{split}-{original_id}.jsonl`, then write a SHA-256 sidecar.

- [ ] **Step 5: Implement source verification and exact preparation CLI**

Download only `train.json` and `dev.json` through the pinned `resolve/<revision>/...` URLs and verify
their full SHA-256 before parsing. For each large LFS ZIP, verify its `raw/<revision>/...` pointer
contains the expected SHA-256 and exact size (11,537,899,362; 20,730,339,734; 21,273,604,733; or
21,520,772,573 bytes). A range read cannot verify the full ZIP SHA; record the fixed LFS object ID
as upstream identity, verify the selected member CRC, and compute a SHA-256 over every extracted
member. Group selected IDs by archive and keep one `RemoteZip` context open per archive so the
central directory is not fetched once per claim. Record source revision, LFS object ID/size, member
path/CRC/SHA, normalized corpus hash, record count, and byte count in
`data/processed/averitec/preparation_receipt.json`.

Expose these exact arguments:

```text
--source-spec data/sources/averitec.json
--output-root data/processed/averitec
--runtime-manifest-root data/manifests
--scorer-manifest-root data/scorer_manifests
--seed 20260817
--calibration-per-label 8
--dev-per-label 20
--stability-per-label 5
--remote-timeout-s 60
--max-member-uncompressed-bytes 268435456
```

The stability IDs are independently ranked within the already selected dev IDs, five per label,
using exactly `sha256(f"{seed}:stability:{label}:{original_id}".encode()).digest()` as the sort key.
`tests/fixtures/averitec/fakes.py` supplies a fully in-memory `FakeRemoteZip` with `infolist` and
streaming `open`, plus counters that make any `extract`/`extractall` call fail the test. Manifest
files use canonical UTF-8 JSON (`sort_keys=True`, compact separators), and each gets
a sibling `.sha256`. Runtime and scorer files are written to their separate configured roots; the
preparation command never accepts a common parent fallback. Any incomplete corpus, hash mismatch,
absent member, unexpected label, or exact forbidden key in runtime output aborts before replacing
existing manifests.

Before `RemoteZip.open`, require one unique central-directory entry whose normalized POSIX path is
exactly in the selected-ID allowlist, contains no absolute/drive/`..` component, and has
`ZipInfo.file_size <= 268435456` (256 MiB). Stream to EOF so CRC validation executes while computing
the member SHA; never call `extract` or `extractall`. Offline fakes cover duplicate names, traversal,
unlisted members, and the byte cap. Record observed preparation totals in the receipt; the
pre-registered reference values are 2,625,947,053 compressed bytes and 8,325,415,688 uncompressed
bytes across 112 selected members, with train ID 1201 the largest at 232,050,960 uncompressed bytes.

- [ ] **Step 6: Run all offline preparation tests**

Run:

```powershell
conda run -n agent-collab python -m pytest tests/test_prepare_averitec.py tests/test_averitec_provider.py -q
```

Expected: deterministic selection, gold isolation, normalization, source rejection, and fixture
provider tests pass without a network connection.

- [ ] **Step 7: Commit the preparation implementation, not downloaded corpora**

```powershell
git add data/sources/averitec.json scripts/prepare_averitec.py tests/fixtures/averitec tests/test_prepare_averitec.py
git commit -m "feat: prepare pinned gold-isolated AVeriTeC subsets"
```

### Task 14: Score The Full Manifest And Statistical Uncertainty

**Files:**
- Create: `src/evidence_route/evaluation/__init__.py`
- Create: `src/evidence_route/evaluation/runtime_manifest.py`
- Create: `src/evidence_route/evaluation/scorer_manifest.py`
- Create: `src/evidence_route/evaluation/metrics.py`
- Create: `tests/fixtures/evaluation/runtime.json`
- Create: `tests/fixtures/evaluation/runtime.json.sha256`
- Create: `tests/fixtures/evaluation/gold.json`
- Create: `tests/fixtures/evaluation/gold.json.sha256`
- Create: `tests/fixtures/evaluation/__init__.py`
- Create: `tests/fixtures/evaluation/factories.py`
- Create: `tests/test_evaluation_metrics.py`

- [ ] **Step 1: Write manifest-boundary and failed-result tests**

```python
# tests/test_evaluation_metrics.py
import hashlib

import pytest

from evidence_route.contracts import ResultStatus, Usage, VerificationResult, Verdict
from evidence_route.evaluation.runtime_manifest import load_runtime_manifest
from evidence_route.evaluation.metrics import (
    NO_PREDICTION,
    paired_bootstrap_difference,
    score_full_manifest,
    wilson_interval,
)


def write_manifest(path, payload: str) -> None:
    encoded = payload.encode("utf-8")
    path.write_bytes(encoded)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
    )


def result(
    claim_id: str, verdict: Verdict,
    status: ResultStatus = ResultStatus.COMPLETED,
):
    return VerificationResult(
        claim_id=claim_id,
        status=status,
        verdict=verdict,
        confidence=0.8,
        rationale="structured result",
        initial_route="single",
        usage=Usage(input_tokens=8, output_tokens=2, total_tokens=10, complete=True),
    )


def test_runtime_loader_rejects_gold_fields(tmp_path) -> None:
    path = tmp_path / "runtime.json"
    write_manifest(path,
        '{"dataset":"AVeriTeC","items":[{"claim_id":"dev-0",'
        '"original_id":0,"claim":"claim","split":"dev","label":"Refuted"}]}',
    )
    try:
        load_runtime_manifest(path, allowed_root=tmp_path)
    except ValueError as error:
        assert "runtime manifest contains forbidden field: label" in str(error)
    else:
        raise AssertionError("gold-bearing runtime manifest was accepted")


def test_runtime_loader_rejects_path_outside_allowed_root(tmp_path) -> None:
    allowed = tmp_path / "runtime"
    allowed.mkdir()
    outside = tmp_path / "scorer.json"
    write_manifest(outside, '{"dataset":"AVeriTeC","items":[]}')
    with pytest.raises(ValueError, match="outside allowed runtime root"):
        load_runtime_manifest(outside, allowed_root=allowed)


def test_runtime_loader_rejects_sidecar_mismatch(tmp_path) -> None:
    path = tmp_path / "runtime.json"
    write_manifest(path, '{"dataset":"AVeriTeC","items":[]}')
    path.write_text('{"dataset":"AVeriTeC","items":[1]}', encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        load_runtime_manifest(path, allowed_root=tmp_path)


def test_partial_and_failed_are_no_prediction_false_negatives() -> None:
    gold = {
        "dev-0": Verdict.SUPPORTED,
        "dev-1": Verdict.REFUTED,
        "dev-2": Verdict.NOT_ENOUGH_EVIDENCE,
        "dev-3": Verdict.CONFLICTING,
    }
    results = {
        "dev-0": result("dev-0", Verdict.SUPPORTED),
        "dev-1": result("dev-1", Verdict.REFUTED, ResultStatus.PARTIAL),
        "dev-2": VerificationResult(
            claim_id="dev-2", status=ResultStatus.FAILED, rationale="transport failed",
            initial_route="single",
            usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False),
        ),
        "dev-3": result("dev-3", Verdict.CONFLICTING),
    }
    metrics = score_full_manifest(gold, results)
    assert metrics.sample_count == 4
    assert metrics.completed_count == 2
    assert metrics.accuracy == 0.5
    assert metrics.scored_predictions["dev-1"] == NO_PREDICTION
    assert metrics.scored_predictions["dev-2"] == NO_PREDICTION


def test_missing_result_is_not_silently_dropped() -> None:
    metrics = score_full_manifest({"dev-0": Verdict.SUPPORTED}, {})
    assert metrics.accuracy == 0.0
    assert metrics.completion_rate == 0.0


def test_uncertainty_functions_are_seeded_and_bounded() -> None:
    gold = ["Supported", "Refuted", "Supported", "Refuted"]
    a = ["Supported", "Refuted", "Supported", "Refuted"]
    b = ["Supported", NO_PREDICTION, "Refuted", "Refuted"]
    first = paired_bootstrap_difference(gold, a, b, samples=10_000, seed=20260817)
    second = paired_bootstrap_difference(gold, a, b, samples=10_000, seed=20260817)
    assert first == second
    low, high = wilson_interval(successes=17, total=20)
    assert 0.0 <= low <= 0.85 <= high <= 1.0
```

- [ ] **Step 2: Run metric tests and confirm evaluation modules are absent**

Run: `conda run -n agent-collab python -m pytest tests/test_evaluation_metrics.py -q`

Expected: FAIL importing `evidence_route.evaluation`.

- [ ] **Step 3: Implement strict runtime/scorer manifest loaders**

Define `RuntimeClaim`, `RuntimeManifest`, and `load_runtime_manifest` in `runtime_manifest.py`.
Define `GoldClaim`, `GoldManifest`, `load_gold_manifest`, and `align_runtime_and_gold` only in
`scorer_manifest.py`:

```python
class RuntimeClaim(StrictModel):
    claim_id: str
    original_id: int = Field(ge=0)
    claim: str = Field(min_length=1)
    claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["train", "dev"]
    corpus_relpath: str = Field(pattern=r"^(train|dev)-\d+\.jsonl$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_bytes: int = Field(gt=0)
    corpus_records: int = Field(gt=0)


class RuntimeManifest(StrictModel):
    schema_version: Literal["1"]
    dataset: Literal["AVeriTeC"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    items: list[RuntimeClaim] = Field(min_length=1)


class GoldClaim(StrictModel):
    claim_id: str
    original_id: int = Field(ge=0)
    claim: str
    label: Verdict
    questions: list[dict[str, object]]
    justification: str
    claim_types: list[str] = Field(default_factory=list)


class GoldManifest(StrictModel):
    schema_version: Literal["1"]
    dataset: Literal["AVeriTeC"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    items: list[GoldClaim] = Field(min_length=1)
```

`tests/fixtures/evaluation/factories.py` loads the committed runtime/gold fixture files through these
public loaders and exposes deterministic pytest fixtures `runtime_claims`, `aligned_claims`, and
`completed_results`; Task 15 imports no hidden global fixture state.

Before Pydantic parsing, each loader requires a sibling `<name>.json.sha256` containing exactly one
lowercase 64-hex digest plus newline and verifies the digest against the raw manifest bytes. It then
recursively rejects exact keys matching
`label|questions|justification|gold|claim_types` in a runtime file. Both loaders reject duplicate
claim IDs. `align_runtime_and_gold` additionally requires identical revision, seed, claim IDs,
original IDs, order, and claim text; it also compares the verified runtime file digest, not its
filename, to `GoldManifest.runtime_manifest_sha256`. The runtime graph/runner process never imports
`scorer_manifest`; calibration/report start a separate scorer stage only after inference artifacts
are atomically closed. Both loader APIs require an `allowed_root`; resolve the root and requested
file, reject symlink/`..` escape with `Path.is_relative_to`, and never infer a scorer path from a
runtime path. Runtime loading also resolves every `corpus_relpath` against the configured corpus
root and verifies its bytes, record count, and SHA before graph construction.

- [ ] **Step 4: Implement full, conditional, and operational metrics**

Use the exact label order from `Verdict`; do not infer labels from free text:

```python
LABELS = [verdict.value for verdict in Verdict]
NO_PREDICTION = "__NO_PREDICTION__"


def prediction_for_scoring(result: VerificationResult | None) -> str:
    if result is None or result.status is not ResultStatus.COMPLETED or result.verdict is None:
        return NO_PREDICTION
    return result.verdict.value
```

`score_full_manifest` builds one prediction for every gold ID, uses `accuracy_score` and
`precision_recall_fscore_support(labels=LABELS, zero_division=0)`, and returns a strict
`ManifestMetrics` containing sample/completed/partial/failed/missing counts, completion rate,
accuracy, macro-F1, per-class precision/recall/F1/support, confusion matrix with an explicit
`NO_PREDICTION` column, and the scored prediction map. Partial results keep their diagnostic label
in artifacts but are always scored as `NO_PREDICTION`.

`score_completed_conditionally` accepts only `ResultStatus.COMPLETED`, reports its sample count and
completion rate beside accuracy/macro-F1, and refuses to serialize without those two fields.
`summarize_operations` reports input/output/total token sums, per-claim token mean, usage-derived
integer micro-CNY cost
only when every usage record is complete, `numpy.quantile(values, [0.5, 0.95], method="linear")`
for fresh calls, cache-hit count separately, route distribution, escalation rate, LLM-router rate,
citation validity rate, and grouped error reasons.

- [ ] **Step 5: Implement paired bootstrap and Wilson intervals**

```python
def paired_bootstrap_difference(
    gold: Sequence[str], predictions_a: Sequence[str], predictions_b: Sequence[str],
    *, samples: int = 10_000, seed: int = 20260817,
) -> BootstrapInterval:
    if not (len(gold) == len(predictions_a) == len(predictions_b)) or not gold:
        raise ValueError("paired bootstrap requires equal non-empty arrays")
    generator = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    labels = [verdict.value for verdict in Verdict]
    for index in range(samples):
        draw = generator.integers(0, len(gold), size=len(gold))
        sampled_gold = [gold[item] for item in draw]
        sampled_a = [predictions_a[item] for item in draw]
        sampled_b = [predictions_b[item] for item in draw]
        values[index] = f1_score(
            sampled_gold, sampled_a, labels=labels, average="macro", zero_division=0
        ) - f1_score(
            sampled_gold, sampled_b, labels=labels, average="macro", zero_division=0
        )
    return BootstrapInterval(
        estimate=float(f1_score(gold, predictions_a, labels=labels, average="macro", zero_division=0)
                       - f1_score(gold, predictions_b, labels=labels, average="macro", zero_division=0)),
        low=float(np.quantile(values, 0.025, method="linear")),
        high=float(np.quantile(values, 0.975, method="linear")),
        samples=samples,
        seed=seed,
    )
```

Implement the standard score Wilson interval with `statistics.NormalDist().inv_cdf(0.975)` and
raise on `total <= 0` or successes outside `[0, total]`. Stability is the count of claim IDs whose
three completed adaptive verdicts are identical divided by 20; any incomplete repeat is a failure,
not an omitted case.

- [ ] **Step 6: Run evaluation tests**

Run:

```powershell
conda run -n agent-collab python -m pytest tests/test_evaluation_metrics.py tests/test_validation.py -q
```

Expected: all manifest, failure-penalty, latency, bootstrap, Wilson, and old-parser regression tests
pass.

- [ ] **Step 7: Commit internal scoring**

```powershell
git add src/evidence_route/evaluation tests/fixtures/evaluation tests/test_evaluation_metrics.py
git commit -m "feat: score full manifests with uncertainty intervals"
```

### Task 15: Pin And Adapt Both Official AVeriTeC Evaluators

**Files:**
- Modify: `.gitattributes`
- Create: `third_party/averitec/paper/eval.py`
- Create: `third_party/averitec/paper/utils.py`
- Create: `third_party/averitec/paper/leven.py`
- Create: `third_party/averitec/shared_task/evaluate_veracity.py`
- Create: `third_party/averitec/SOURCES.json`
- Create: `data/sources/nltk_data.json`
- Create: `scripts/prepare_nltk_data.py`
- Create: `src/evidence_route/evaluation/official.py`
- Create: `tests/test_official_evaluator.py`

- [ ] **Step 1: Fetch immutable evaluator sources and verify bytes**

Add the byte-preservation rules before fetching. The three upstream files must survive checkout
without `core.autocrlf` changing their pinned hashes; the project-owned shim remains normalized LF:

```gitattributes
third_party/averitec/paper/eval.py -text
third_party/averitec/paper/utils.py -text
third_party/averitec/shared_task/evaluate_veracity.py -text
third_party/averitec/paper/leven.py text eol=lf
```

Run these commands from the repository root:

```powershell
New-Item -ItemType Directory -Force third_party/averitec/paper,third_party/averitec/shared_task | Out-Null
curl.exe -L "https://raw.githubusercontent.com/MichSchli/AVeriTeC/7c62d1ec8df3fb560d6efe2b85fa191135636f81/eval.py" -o third_party/averitec/paper/eval.py
curl.exe -L "https://raw.githubusercontent.com/MichSchli/AVeriTeC/7c62d1ec8df3fb560d6efe2b85fa191135636f81/utils.py" -o third_party/averitec/paper/utils.py
curl.exe -L "https://huggingface.co/chenxwh/AVeriTeC/resolve/2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031/src/prediction/evaluate_veracity.py" -o third_party/averitec/shared_task/evaluate_veracity.py
Get-FileHash -Algorithm SHA256 third_party/averitec/paper/eval.py
Get-FileHash -Algorithm SHA256 third_party/averitec/paper/utils.py
Get-FileHash -Algorithm SHA256 third_party/averitec/shared_task/evaluate_veracity.py
```

Expected hashes, in the same order:

```text
01e325e5e19074037d1f1a7673b0f7999ecab82ee5ae3e81205efc86cc39b161
d4b1bcbd0ec10210892e5d41521ffd6a0ae02e50ba336b8d84830f19dc33643e
b0683b2f31f5b6ec06628e31d1c55132ffb4787f7341cf14d98411c29882ed70
```

Create `SOURCES.json` containing each URL, revision/commit, SHA-256, license
`CC BY-NC 4.0`, and the citation `Schlichtkrull et al., AVeriTeC, NeurIPS 2023 Datasets and
Benchmarks`. Do not edit the three pinned Python files.

- [ ] **Step 2: Write adapter order, failure exclusion, and hash tests**

```python
# tests/test_official_evaluator.py
from pathlib import Path

import pytest

from evidence_route.contracts import ResultStatus, VerificationResult, Verdict
from evidence_route.evaluation.official import (
    build_paper_predictions,
    build_shared_task_predictions,
    select_completed_triples,
    verify_evaluator_sources,
)


pytest_plugins = ["tests.fixtures.evaluation.factories"]


def test_pinned_evaluator_hashes_match() -> None:
    verify_evaluator_sources(Path("third_party/averitec/SOURCES.json"))


def test_shared_task_adapter_preserves_manifest_order(completed_results, aligned_claims) -> None:
    selection = select_completed_triples(aligned_claims, completed_results)
    payload = build_shared_task_predictions(selection.triples)
    assert [item["claim_id"] for item in payload] == [item.runtime.original_id for item in aligned_claims]
    assert payload[0]["pred_label"] in {verdict.value for verdict in Verdict}
    assert set(payload[0]["evidence"][0]) == {
        "question", "answer", "url", "scraped_text"
    }


def test_official_selector_omits_only_non_completed(completed_results, aligned_claims) -> None:
    completed_results[0] = completed_results[0].model_copy(
        update={"status": ResultStatus.PARTIAL}
    )
    selection = select_completed_triples(aligned_claims, completed_results)
    assert selection.omitted_claim_ids == [aligned_claims[0].runtime.claim_id]
    assert selection.completion_rate == pytest.approx(
        (len(aligned_claims) - 1) / len(aligned_claims)
    )


def test_paper_adapter_uses_only_official_fields(completed_results, aligned_claims) -> None:
    payload = build_paper_predictions(
        select_completed_triples(aligned_claims, completed_results).triples
    )
    assert payload[0]["label"] == completed_results[0].verdict.value
    assert payload[0]["questions"][0]["answers"][0]["answer"]
    assert payload[0]["justification"] == completed_results[0].rationale
    assert set(payload[0]) == {"label", "questions", "justification"}


def test_completed_empty_evidence_stays_in_official_cohort(
    completed_results, aligned_claims,
) -> None:
    completed_results[0] = completed_results[0].model_copy(
        update={"citations": [], "available_evidence_ids": []}
    )
    selection = select_completed_triples(aligned_claims, completed_results)
    payload = build_shared_task_predictions(selection.triples)
    assert len(payload) == len(aligned_claims)
    assert payload[0]["evidence"] == []
```

- [ ] **Step 3: Run adapter tests and confirm the module is absent**

Run: `conda run -n agent-collab python -m pytest tests/test_official_evaluator.py -q`

Expected: FAIL importing `evidence_route.evaluation.official`.

- [ ] **Step 4: Implement explicit cohort selection and strict low-level adapters**

First call Task 14's `align_runtime_and_gold`, which has already checked original IDs, order, claim
text, sidecars, and `runtime_manifest_sha256`. It returns `AlignedClaim(runtime, gold)`. Then
`select_completed_triples(aligned_claims, results)` requires one result with the same `claim_id` in
the same order and returns `CompletedTriple(runtime, gold, result)` values plus omitted claim IDs,
full count, completed count, and completion rate. It omits only non-completed statuses; an empty
citation list remains a valid triple so the official evidence-aware metric can assign zero instead
of silently improving the cohort. The test module defines `aligned_claims` and `completed_results`
fixtures from the committed Task 14 fixture manifests rather than hidden conftest state. The
low-level builders accept only that triple list. The shared-task format is a JSON array, not JSONL:

```python
def shared_task_item(triple: CompletedTriple) -> dict[str, object]:
    claim, result = triple.runtime, triple.result
    require_completed(result)
    return {
        "claim_id": claim.original_id,
        "claim": claim.claim,
        "pred_label": result.verdict.value,
        "evidence": [
            {
                "question": citation.question,
                "answer": citation.answer,
                "url": str(citation.source_url),
                "scraped_text": citation.quote,
            }
            for citation in result.citations[:10]
        ],
    }
```

The paper-era format uses `label`, `questions`, and `justification`:

```python
def paper_item(triple: CompletedTriple) -> dict[str, object]:
    result = triple.result
    require_completed(result)
    return {
        "label": result.verdict.value,
        "questions": [
            {
                "question": citation.question,
                "answers": [{
                    "answer": citation.answer,
                    "answer_type": "Abstractive",
                    "source_url": str(citation.source_url),
                }],
            }
            for citation in result.citations[:10]
        ],
        "justification": result.rationale,
    }
```

The paper prediction JSON contains exactly `label`, `questions`, and `justification`; claim identity
and text stay in the aligned sidecar/triples used to filter the corresponding gold array. Never
invent questions, answers, labels, or evidence for partial/failed/missing results. Never remove a
completed triple because it has no citations; both adapters emit an empty evidence/questions array
for that case.

- [ ] **Step 5: Run pinned evaluators in isolated subprocesses**

`run_official_evaluators` writes filtered completed predictions and equally filtered gold reference
arrays to a temporary artifact directory. It invokes the shared-task script with `-i ...
--label_file ...` and the paper script with `--predictions ... --references ...`, sets the paper
working directory so its pinned `utils.py` resolves, captures stdout/stderr, and returns non-zero on
hash mismatch or scorer failure. Save raw stdout and parsed JSON separately.
If the completed cohort is empty, do not invoke either script; return
`{"available": false, "reason": "no_completed_outputs", "completed_count": 0}` while the Task 14
full-manifest score remains zero. Never manufacture a row solely to make a scorer executable.

Label outputs unambiguously:

- `shared_task_2024`: question-only HU-METEOR, QA HU-METEOR, per-class F1, macro-F1, accuracy,
  and evidence-aware AVeriTeC at each threshold; `AVeriTeC@0.25` is primary.
- `paper_2023_secondary`: question-only, QA, justification METEOR, per-class/macro F1, accuracy,
  and paper-era evidence/justification-aware values.

The original `utils.py` imports the obsolete `leven` package at module load even though the official
evaluation path never calls its edit-distance helpers. Avoid an unmaintained compiled dependency by
creating this project-owned compatibility module beside `utils.py`:

```python
def levenshtein(source: str, target: str) -> int:
    if len(source) < len(target):
        return levenshtein(target, source)
    previous = list(range(len(target) + 1))
    for source_index, source_char in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_char in enumerate(target, start=1):
            current.append(min(
                current[-1] + 1,
                previous[target_index] + 1,
                previous[target_index - 1] + (source_char != target_char),
            ))
        previous = current
    return previous[-1]
```

Test `levenshtein("kitten", "sitting") == 3`. The shim does not change a metric executed by
`eval.py`; mark it `project_owned_compatibility_shim: true` with its computed SHA in `SOURCES.json`
instead of presenting it as upstream code.

- [ ] **Step 6: Pin NLTK metric assets and run evaluator integration tests**

Create `data/sources/nltk_data.json`:

```json
{
  "repository": "nltk/nltk_data",
  "commit": "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a",
  "files": {
    "packages/tokenizers/punkt_tab.zip": {
      "size": 4319076,
      "sha256": "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
    },
    "packages/corpora/wordnet.zip": {
      "size": 10775600,
      "sha256": "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"
    }
  }
}
```

`scripts/prepare_nltk_data.py` downloads from the commit-pinned raw GitHub URLs, verifies size and
SHA-256 before extraction, rejects absolute/`..` ZIP members, and atomically prepares
`data/external/nltk/tokenizers/punkt_tab` and `data/external/nltk/corpora/wordnet`. It writes a
receipt with the source commit and extracted-tree hash. Never use `python -m nltk.downloader`, whose
index can change independently of the lock file.

Run:

```powershell
conda run -n agent-collab python scripts/prepare_nltk_data.py --source-spec data/sources/nltk_data.json --output-root data/external/nltk
$env:NLTK_DATA = (Resolve-Path data/external/nltk).Path
conda run -n agent-collab python -m pytest -m network tests/test_official_evaluator.py -q
```

Expected: pinned NLTK hashes, pinned evaluator hashes, both adapters, non-completed exclusion,
empty-evidence retention, and a two-row subprocess smoke fixture pass. The result metadata reports
the completed subset size and completion rate alongside official numbers.

- [ ] **Step 7: Commit pinned evaluators and adapters**

```powershell
git add .gitattributes third_party/averitec data/sources/nltk_data.json scripts/prepare_nltk_data.py src/evidence_route/evaluation/official.py tests/test_official_evaluator.py pyproject.toml requirements.lock
git commit -m "feat: pin and adapt official AVeriTeC evaluators"
```

### Task 16: Calibrate Adaptive Routing By Replaying Saved Train Runs

**Files:**
- Create: `src/evidence_route/evaluation/calibration.py`
- Modify: `tests/fixtures/evaluation/factories.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write policy replay and deterministic selection tests**

```python
# tests/test_calibration.py
import pytest

from evidence_route.config import RoutingSettings, stable_hash
from evidence_route.evaluation.calibration import (
    CalibrationRuntimeCase,
    CalibrationScoredCase,
    CandidateOutcome,
    candidate_grid,
    choose_candidate,
    replay_candidate,
)


pytest_plugins = ["tests.fixtures.evaluation.factories"]


def test_runtime_case_rejects_gold_field(calibration_case: CalibrationScoredCase) -> None:
    payload = calibration_case.runtime.model_dump(mode="json")
    payload["gold_label"] = "Supported"
    with pytest.raises(ValueError):
        CalibrationRuntimeCase.model_validate(payload)


def test_replay_uses_saved_single_then_multi_on_escalation(
    calibration_case: CalibrationScoredCase,
) -> None:
    runtime = calibration_case.runtime.model_copy(update={
        "saved_llm_route": "single",
        "single_result": calibration_case.runtime.single_result.model_copy(
            update={"confidence": 0.50}
        ),
    })
    calibration_case = calibration_case.model_copy(update={"runtime": runtime})
    outcome = replay_candidate(calibration_case, RoutingSettings(
        clear_multi_clauses=4,
        clear_single_min_sources=3,
        low_confidence=0.65,
        minimum_coverage=1.0,
    ))
    assert outcome.final_result == calibration_case.runtime.multi_result
    assert outcome.executed_path == "single_escalated_multi"
    assert outcome.total_tokens == (
        calibration_case.runtime.router_usage.total_tokens
        + calibration_case.runtime.single_result.usage.total_tokens
        + calibration_case.runtime.multi_result.usage.total_tokens
    )


def test_candidate_prefers_token_saving_within_quality_floor() -> None:
    outcomes = [
        CandidateOutcome(config_hash="a" * 64, macro_f1=0.70, total_tokens=1000,
                         llm_router_calls=16, llm_router_rate=0.5,
                         simulated_cost_micro_cny=1_000_000, settings={}),
        CandidateOutcome(config_hash="b" * 64, macro_f1=0.68, total_tokens=500,
                         llm_router_calls=6, llm_router_rate=0.2,
                         simulated_cost_micro_cny=500_000, settings={}),
        CandidateOutcome(config_hash="c" * 64, macro_f1=0.66, total_tokens=100,
                         llm_router_calls=0, llm_router_rate=0.0,
                         simulated_cost_micro_cny=100_000, settings={}),
    ]
    selected = choose_candidate(outcomes, tolerance=0.03)
    assert selected.config_hash == "b" * 64


def test_candidate_tie_breaks_by_router_rate_then_hash() -> None:
    outcomes = [
        CandidateOutcome(config_hash="b" * 64, macro_f1=0.70, total_tokens=500,
                         llm_router_calls=6, llm_router_rate=0.2,
                         simulated_cost_micro_cny=500_000, settings={}),
        CandidateOutcome(config_hash="a" * 64, macro_f1=0.70, total_tokens=500,
                         llm_router_calls=6, llm_router_rate=0.2,
                         simulated_cost_micro_cny=500_000, settings={}),
    ]
    assert choose_candidate(outcomes, tolerance=0.03).config_hash == "a" * 64


def test_candidate_grid_has_54_unique_configs() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 54
    assert len({stable_hash(item.model_dump(mode="json")) for item in candidates}) == 54
```

- [ ] **Step 2: Run calibration tests and confirm the module is absent**

Run: `conda run -n agent-collab python -m pytest tests/test_calibration.py -q`

Expected: FAIL importing `evidence_route.evaluation.calibration`.

- [ ] **Step 3: Define the immutable calibration artifact**

Each of the 32 train cases is collected once without gold, then a separate scorer process creates
the second model:

```python
class CalibrationItemStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETE = "complete"
    STOPPED = "stopped"


class CalibrationWorkItem(StrictModel):
    order: int = Field(ge=0, le=31)
    claim_id: str
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    router_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    single_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    multi_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationPlan(StrictModel):
    schema_version: Literal["1"]
    activity_id: str
    manifest_freeze_git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_preparation_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pricing_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoint_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirements_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_alias: str = Field(min_length=1)
    seed: int
    cap_micro_cny: int = Field(gt=0)
    items: list[CalibrationWorkItem] = Field(min_length=32, max_length=32)
    plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationItemState(StrictModel):
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: CalibrationItemStatus = CalibrationItemStatus.PENDING
    artifact_relpath: str
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    errors: list[str] = Field(default_factory=list)


class CalibrationState(StrictModel):
    schema_version: Literal["1"]
    activity_id: str
    plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    items: list[CalibrationItemState] = Field(min_length=32, max_length=32)
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationRuntimeCase(StrictModel):
    claim_id: str
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    router_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    single_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    multi_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    call_ids: list[str] = Field(min_length=1)
    runtime_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    features: ClaimFeatures
    saved_llm_route: Literal["single", "multi"]
    router_usage: Usage
    router_actual_cost_micro_cny: int = Field(ge=0)
    single_result: VerificationResult
    multi_result: VerificationResult
    requested_alias: str
    response_model_ids_raw: list[str]
    identity_verified: Literal[False] = False
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationScoredCase(StrictModel):
    runtime: CalibrationRuntimeCase
    gold_label: Verdict
```

Derive IDs before any call as
`case_id=sha256(activity_id + "\\0calibration\\0" + claim_id)` and
`{router,single,multi}_run_id=sha256(case_id + "\\0" + path)`. Collection runs `always_single`,
`always_multi`, and exactly one structured router call for every
case. Saving one LLM route per case is required because some of the 54 replay candidates make a
previously clear case uncertain. The collection profile is therefore exactly router + single + one
three-worker multi path per train claim, matching Task 17's 224-call base bound. Save each case once
as atomically replaced canonical JSON at `calibration/cases/<case_id>.json`; after all 32 close,
generate the JSONL replay view in immutable plan order from those verified files. Before the first
call, atomically persist `CalibrationPlan`; its fingerprint is canonical
JSON over every field except `plan_fingerprint`; the complete ordered work items are therefore
frozen, not reconstructed from claim IDs. Atomically persist `calibration-state.json` before the
first call and before/after every case. A fresh collect rejects an existing activity, while
`--resume` recomputes each current hash/alias/cap, verifies the stored fingerprint, and reads the
stored 32 work items without reselecting them. A persisted `RUNNING` case is normalized to
`INTERRUPTED` at the resume boundary, then each of its three deterministic run IDs is reconciled
against the call store before any network request. If SQLite completed a call but the case
artifact/state update did not close, resume reuses the same call IDs and atomically writes the one
self-hashed case artifact; JSONL is regenerated from verified case files, so it never appends a
duplicate row or spends twice. Add
`test_calibration_resume_after_call_completion_before_case_close_reuses_calls` and
`test_calibration_state_has_one_artifact_hash_per_plan_item`. The same activity
`SQLiteRunStore` carries all
calibration spend and raw model IDs forward into Task 17.
Task 14's `tests/fixtures/evaluation/factories.py` owns deterministic constructors for
`runtime_claims`, `aligned_claims`, and `completed_results`; this task extends it with the explicit
`calibration_case` fixture after the calibration models exist.

Collection cannot import `CalibrationScoredCase` or expose `gold_label` to the graph/provider. After
all runtime artifacts are atomically closed, `attach_calibration_gold` takes Task 14 aligned
runtime/gold rows and rejects anything other than exactly 32 unique `train-*` IDs, exactly eight of
each `Verdict`, matching claim order, and the same verified `runtime_manifest_sha256` on every saved
case. Only then does it emit `CalibrationScoredCase` values. Report actual collection cost separately
from each candidate's simulated deployed-policy cost; all persisted costs remain integer micro-CNY.

- [ ] **Step 4: Implement a finite, explicit candidate grid and replay**

`CandidateOutcome` is strict and contains `config_hash`, the four routing settings, full-manifest
macro-F1, total tokens, `simulated_cost_micro_cny`, integer LLM-router call count, router-call rate, status
counts, and one replay decision per claim. The cost field is reported but is not a selection
tie-break.

Use this exact grid, which yields 54 candidates:

```python
def candidate_grid() -> list[RoutingSettings]:
    return [
        RoutingSettings(
            clear_multi_clauses=clauses,
            clear_single_min_sources=sources,
            low_confidence=confidence,
            minimum_coverage=coverage,
        )
        for clauses in (2, 3)
        for sources in (1, 2, 3)
        for confidence in (0.55, 0.65, 0.75)
        for coverage in (0.50, 0.75, 1.0)
    ]
```

For each candidate, rerun only deterministic rule predicates and validation over saved artifacts.
When rules remain uncertain, consume `saved_llm_route` and include its saved usage/cost; otherwise
router usage/cost is zero. A single choice uses `single_result`; rerun the pure `ResultValidator`
with the candidate's `low_confidence` and `minimum_coverage` over the saved claim units and
`available_evidence_ids`. If that candidate would escalate, append the saved multi path and use
`multi_result`. A multi choice uses only saved multi nodes. Never call an LLM or provider during
replay.

- [ ] **Step 5: Implement selection and an auditable calibration report**

Score every candidate with the Task 14 full-manifest scorer, then select exactly as pre-registered:

```python
best_candidate_macro_f1 = max(item.macro_f1 for item in outcomes)
eligible = [
    item for item in outcomes
    if item.macro_f1 >= best_candidate_macro_f1 - 0.03
]
selected = min(
    eligible,
    key=lambda item: (item.total_tokens, item.llm_router_calls, item.config_hash),
)
```

The best candidate itself always qualifies, so there is no empty/fallback branch. Report estimated
cost but do not insert it into the tie-break order. Fixed single/multi remain calibration-report
baselines; they do not define the candidate quality floor. The separate dev engineering target still
compares the frozen adaptive policy with the better fixed policy.

Write `calibration_report.json` with all 54 candidates, baselines, selected settings/hash, simulated
token/cost accounting, best-candidate quality floor, manifest hash, prompt hash, and code Git SHA. Write
`configs/calibrated.yaml` by replacing only the four routing values, then canonicalize and validate
the complete config; assert every non-routing parsed value equals `configs/default.yaml`. This
report is train-only and must never be
described as final dev performance.

- [ ] **Step 6: Run calibration tests**

Run: `conda run -n agent-collab python -m pytest tests/test_calibration.py tests/test_evaluation_metrics.py -q`

Expected: all replay paths, cost aggregation, gold isolation, candidate count, failure scoring, and
deterministic tie-break tests pass without network access.

- [ ] **Step 7: Commit calibration logic**

```powershell
git add src/evidence_route/evaluation/calibration.py tests/fixtures/evaluation/factories.py tests/test_calibration.py
git commit -m "feat: calibrate adaptive routing from saved train runs"
```

### Task 17: Run Interleaved, Resumable, Budgeted Campaigns

**Files:**
- Modify: `src/evidence_route/config.py`
- Modify: `src/evidence_route/artifacts.py`
- Create: `src/evidence_route/evaluation/activity.py`
- Create: `src/evidence_route/evaluation/runner.py`
- Modify: `tests/test_artifacts.py`
- Create: `tests/fixtures/evaluation/campaign_factory.py`
- Create: `tests/test_activity.py`
- Create: `tests/test_campaign_runner.py`

- [ ] **Step 1: Verify configured node limits are used end-to-end**

Task 3 already defines the one configured cap per LLM node. Add tests rejecting a fourth worker,
second escalation, negative token caps, or unknown generation fields. Add spy-LLM tests proving
router/single/decomposer/worker/judge each receive their corresponding `GenerationSettings` values;
there must be no duplicate production constants `1800`, `6500`, `2500`, `5000`, or `8000` outside
`config.py` and the budget-bound test.

```python
def test_generation_limits_are_literal_and_bounded() -> None:
    settings = GenerationSettings()
    assert settings.max_workers == 3
    assert settings.max_escalations == 1
    with pytest.raises(ValidationError):
        GenerationSettings(max_workers=4)
```

- [ ] **Step 2: Write schedule, upper-bound, interruption, and drift tests**

```python
# tests/test_campaign_runner.py
from pathlib import Path

import pytest

from evidence_route.artifacts import SQLiteRunStore
from evidence_route.budget import BudgetExceeded, PriceConfig
from evidence_route.config import GenerationSettings
from evidence_route.contracts import Strategy, Usage
from evidence_route.evaluation.activity import CampaignStatus, CampaignStopReason, WorkStatus
from evidence_route.evaluation.runner import (
    CampaignRunner,
    build_dev_schedule,
    compute_gate_a_call_profile,
    estimate_call_bounds,
)


pytest_plugins = [
    "tests.fixtures.evaluation.factories",
    "tests.fixtures.evaluation.campaign_factory",
]


def test_gate_a_base_profile_is_the_specified_1544_calls() -> None:
    profile = compute_gate_a_call_profile(
        calibration_claims=32, dev_claims=80, stability_claims=20,
        stability_extra_repeats=2,
    )
    assert profile.model_dump() == {
        "router": 152, "single": 232, "decomposer": 232,
        "worker": 696, "judge": 232,
    }
    assert profile.total == 1544


def test_repair_and_fault_bounds_are_explicit() -> None:
    bounds = estimate_call_bounds(
        compute_gate_a_call_profile(32, 80, 20, 2),
        GenerationSettings(),
        PriceConfig(provider="relay", currency="CNY", input_per_million=1,
                    output_per_million=2, price_source="fixture", strict_evaluation=True),
        reserve_ratio=0.2,
    )
    assert bounds.base_call_upper_bound == 1544
    assert bounds.repair_upper_bound == 3088
    assert bounds.fault_upper_bound == 9264
    assert bounds.startup_required_micro_cny == (bounds.base_cost_micro_cny * 120 + 99) // 100


def test_schedule_interleaves_all_strategies_per_claim(runtime_claims) -> None:
    schedule = build_dev_schedule(runtime_claims, seed=20260817)
    base = [item.strategy for item in schedule[:3]]
    assert set(base) == set(Strategy)
    for offset in range(0, len(schedule), 3):
        group = schedule[offset:offset + 3]
        claim_index = offset // 3
        assert len({item.claim_id for item in group}) == 1
        assert [item.strategy for item in group] == (
            base[claim_index % 3:] + base[:claim_index % 3]
        )
    assert schedule == build_dev_schedule(runtime_claims, seed=20260817)


@pytest.mark.asyncio
async def test_resume_skips_terminal_work_items(tmp_path: Path, campaign_factory) -> None:
    first_executor = campaign_factory.executor(fail_after=2)
    runner = CampaignRunner(tmp_path, first_executor)
    with pytest.raises(RuntimeError, match="injected interruption"):
        await runner.run(campaign_factory.plan())
    second_executor = campaign_factory.executor()
    resumed = await CampaignRunner(tmp_path, second_executor).resume(
        expected_identity=campaign_factory.freeze_identity()
    )
    assert resumed.status is CampaignStatus.COMPLETE
    assert not ({item.run_id for item in first_executor.completed}
                & {item.run_id for item in second_executor.completed})


@pytest.mark.asyncio
async def test_model_id_drift_stops_campaign(tmp_path: Path, campaign_factory) -> None:
    executor = campaign_factory.executor(model_ids=["relay-a", "relay-b"])
    campaign = await CampaignRunner(tmp_path, executor).run(campaign_factory.plan())
    assert campaign.status is CampaignStatus.INCOMPLETE_MODEL_DRIFT
    assert campaign.stop_reason is CampaignStopReason.MODEL_DRIFT
    assert all(item.status is WorkStatus.PENDING for item in campaign.items[2:])


@pytest.mark.asyncio
async def test_resume_reopens_persisted_running_item_after_process_crash(
    tmp_path: Path, campaign_factory,
) -> None:
    crashed = campaign_factory.persist_running_item_with_completed_call()
    executor = campaign_factory.executor()
    resumed = await CampaignRunner(tmp_path, executor).resume(
        expected_identity=campaign_factory.freeze_identity()
    )
    assert resumed.status is CampaignStatus.COMPLETE
    assert crashed.run_id not in executor.fresh_network_run_ids
    assert campaign_factory.run_store.summarize_run(crashed.run_id).cache_hit_count >= 1


@pytest.mark.asyncio
async def test_budget_stop_marks_unstarted_items(tmp_path: Path, campaign_factory) -> None:
    executor = campaign_factory.executor(exceed_budget_after=1)
    campaign = await CampaignRunner(tmp_path, executor).run(campaign_factory.plan())
    assert campaign.status is CampaignStatus.INCOMPLETE_BUDGET
    assert all(item.status is WorkStatus.NOT_RUN_BUDGET for item in campaign.items[1:])


def test_shared_run_store_carries_calibration_spend_into_dev(tmp_path: Path) -> None:
    path = tmp_path / "gate-a-run-store.sqlite3"
    pricing = PriceConfig(
        provider="fixture", currency="CNY", input_per_million=1.0,
        output_per_million=1.0, price_source="fixture", strict_evaluation=True,
    )
    first = SQLiteRunStore(path, activity_id="gate-a", cap_cny=1.0, pricing=pricing)
    first.reserve_call(
        "calibration-call", request_sha256="a" * 64, run_id="calibration-0",
        node="single", task_id="root", logical_attempt=0,
        max_input_tokens=600_000, max_output_tokens=0,
    )
    first.mark_sent("calibration-call")
    first.complete_call(
        "calibration-call", request_sha256="a" * 64, payload={"content": "ok"},
        usage=Usage(input_tokens=600_000, output_tokens=0,
                    total_tokens=600_000, complete=True),
        usage_source="provider",
        requested_alias="alias", response_model_id_raw="relay-a", identity_verified=False,
    )
    reopened = SQLiteRunStore(path, activity_id="gate-a", cap_cny=1.0, pricing=pricing)
    assert reopened.summarize_activity().actual_cost_micro_cny == 600_000
    with pytest.raises(BudgetExceeded):
        reopened.reserve_call(
            "dev-call", request_sha256="b" * 64, run_id="dev-0", node="single",
            task_id="root", logical_attempt=0,
            max_input_tokens=400_000, max_output_tokens=100_000,
        )
```

`tests/fixtures/evaluation/campaign_factory.py` defines the explicit `runtime_claims` and
`campaign_factory` fixtures used above. In `tests/test_activity.py`, add these named tests so every
recovery/freeze invariant has one direct regression:

```python
# tests/test_activity.py
import pytest

from evidence_route.evaluation.activity import CampaignStatus, FreezeMismatch


pytest_plugins = ["tests.fixtures.evaluation.campaign_factory"]


@pytest.mark.parametrize("field", [
    "manifest_freeze_git_sha", "dev_protocol_git_sha",
    "dev_runtime_manifest_sha256", "stability_runtime_manifest_sha256",
    "corpus_preparation_receipt_sha256", "prompt_bundle_sha256", "config_sha256",
    "pricing_sha256", "endpoint_config_sha256", "requested_alias",
    "requirements_lock_sha256", "seed", "schedule",
])
def test_resume_rejects_each_changed_frozen_input(field, activity_factory) -> None:
    activity_factory.persist_plan()
    with pytest.raises(FreezeMismatch, match=field):
        activity_factory.resume_with_changed(field)


def test_seeded_schedule_is_cyclic_and_resume_uses_persisted_order(
    activity_factory, monkeypatch,
) -> None:
    expected = activity_factory.persist_plan().schedule
    monkeypatch.setattr(
        "evidence_route.evaluation.runner.build_dev_schedule",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not rebuild")),
    )
    assert activity_factory.resume().plan.schedule == expected


def test_calibration_and_dev_share_budget_and_model_identity(activity_factory) -> None:
    activity = activity_factory.after_calibration(model_id="relay-a", cost_micro_cny=600_000)
    assert activity_factory.start_dev(activity, model_id="relay-a").status is CampaignStatus.RUNNING
    stopped = activity_factory.start_dev(activity, model_id="relay-b")
    assert stopped.status is CampaignStatus.INCOMPLETE_MODEL_DRIFT
    assert activity_factory.run_store.summarize_activity().actual_cost_micro_cny >= 600_000


def test_calibration_links_start_empty_and_close_in_plan_order(activity_factory) -> None:
    activity = activity_factory.before_calibration()
    expected_case_ids = [item.case_id for item in activity_factory.calibration_plan.items]
    assert [link.case_id for link in activity.calibration_artifacts] == expected_case_ids
    assert all(link.artifact_sha256 is None for link in activity.calibration_artifacts)
    with pytest.raises(ValueError, match="32 completed calibration artifacts"):
        activity_factory.mark_calibration_complete(activity)
    closed = activity_factory.close_all_calibration_cases(activity)
    assert all(link.artifact_sha256 is not None for link in closed.calibration_artifacts)
    complete = activity_factory.mark_calibration_complete(closed)
    assert complete.calibration_status is CampaignStatus.COMPLETE
```

Also retain the exact Task 7/8 recovery tests
`test_parallel_reservations_are_serialized_without_crossing_cap`,
`test_crash_after_transmit_before_cache_write_blocks_resume`,
`test_crash_after_cache_write_before_reconcile_counts_cost_once`, and add
`test_worker_resume_does_not_duplicate_usage_or_cost` plus
`test_missing_usage_is_incomplete_not_zero_cost` and
`test_model_drift_inside_one_run_stops_activity`. These tests reopen both SQLite files and compare
call IDs, usage, and integer cost before and after resume.
Add `test_stability_schedule_links_repeat_zero_to_dev_adaptive_artifact`, which checks all 20 links
against deterministic dev adaptive run IDs, then checks the 20 verified artifact SHAs recorded in
`CampaignState`; only repeats 1 and 2 become paid work items.

- [ ] **Step 3: Run campaign tests and confirm the runner is absent**

Run: `conda run -n agent-collab python -m pytest tests/test_artifacts.py tests/test_activity.py tests/test_campaign_runner.py -q`

Expected: FAIL importing `evidence_route.evaluation.activity` or `runner`.

- [ ] **Step 4: Implement exact call profiles and cost bounds**

Define `CallProfile` fields `router`, `single`, `decomposer`, `worker`, `judge` and a computed
`total`. The Gate A worst-case profile is calculated, not hard-coded:

```python
def compute_gate_a_call_profile(
    calibration_claims: int = 32,
    dev_claims: int = 80,
    stability_claims: int = 20,
    stability_extra_repeats: int = 2,
) -> CallProfile:
    # Calibration collects router + single + one three-worker multi path.
    calibration = CallProfile(
        router=calibration_claims,
        single=calibration_claims,
        decomposer=calibration_claims,
        worker=3 * calibration_claims,
        judge=calibration_claims,
    )
    # Dev includes fixed single, fixed multi, and worst-case adaptive single -> multi.
    dev = CallProfile(
        router=dev_claims,
        single=2 * dev_claims,
        decomposer=2 * dev_claims,
        worker=6 * dev_claims,
        judge=2 * dev_claims,
    )
    stability = CallProfile(
        router=stability_claims * stability_extra_repeats,
        single=stability_claims * stability_extra_repeats,
        decomposer=stability_claims * stability_extra_repeats,
        worker=3 * stability_claims * stability_extra_repeats,
        judge=stability_claims * stability_extra_repeats,
    )
    return calibration + dev + stability
```

For every node, multiply count by its configured input/output cap and `PriceConfig`. Report:

```python
reserve_basis_points = int(Decimal(str(reserve_ratio)) * Decimal(10_000))
CallBounds(
    base_call_upper_bound=profile.total,
    repair_upper_bound=profile.total * 2,
    fault_upper_bound=profile.total * 2 * 3,
    base_cost_micro_cny=base_cost_micro_cny,
    repair_cost_micro_cny=base_cost_micro_cny * 2,
    fault_cost_micro_cny=base_cost_micro_cny * 2 * 3,
    reserve_basis_points=reserve_basis_points,
    startup_required_micro_cny=(
        base_cost_micro_cny * (10_000 + reserve_basis_points) + 9_999
    ) // 10_000,
)
```

The runner refuses to start when pricing/usage is non-strict, currency is not CNY, or
`startup_required_micro_cny > cap_micro_cny`. Display-only CNY values are derived after all integer
comparisons. Retry/timeout records with unknown provider
billing set `cost_is_lower_bound=true` and stop with `INCOMPLETE_COST_UNCERTAIN`.

Use only Task 7's `SQLiteRunStore`; do not add a JSON ledger or a second budget abstraction. The same
`--run-store` file is opened for calibration, dev, and stability with the same activity ID, cap,
currency, and price hash. Its `BEGIN IMMEDIATE` reservation transaction is the sole budget authority.
An unresolved `sent`/`billing_uncertain` row keeps its reservation committed and stops for bill
review; a `usage_missing` row keeps its saved response and reservation and stops for usage review.
Code never guesses that either was free or releases it on restart.

- [ ] **Step 5: Implement immutable schedules and campaign state**

Use these enums, keeping campaign status separate from `VerificationResult.status`:

```python
class WorkStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    STOPPED = "stopped"
    NOT_RUN_BUDGET = "not_run_budget"
    CANCELLED = "cancelled"


class CampaignStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETE = "complete"
    INCOMPLETE_BUDGET = "incomplete_budget"
    INCOMPLETE_MODEL_DRIFT = "incomplete_model_drift"
    INCOMPLETE_USAGE = "incomplete_usage"
    INCOMPLETE_COST_UNCERTAIN = "incomplete_cost_uncertain"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CampaignStopReason(StrEnum):
    PROCESS_INTERRUPTION = "process_interruption"
    BUDGET = "budget"
    MODEL_DRIFT = "model_drift"
    USAGE_MISSING = "usage_missing"
    BILLING_UNCERTAIN = "billing_uncertain"
    USER_CANCELLED = "user_cancelled"
    INTERNAL_ERROR = "internal_error"
```

Define these strict models in `evaluation/activity.py`; all hash fields are lowercase 64-hex unless a
40-hex Git SHA is stated:

```python
class FreezeIdentity(StrictModel):
    manifest_freeze_git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    dev_protocol_git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    calibration_runtime_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dev_runtime_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stability_runtime_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_preparation_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pricing_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoint_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirements_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_alias: str = Field(min_length=1)
    seed: int


class LatencyBreakdown(StrictModel):
    fresh_end_to_end_ms: int = Field(ge=0)
    model_active_ms: int = Field(ge=0)
    retry_ms: int = Field(ge=0)
    queue_ms: int = Field(ge=0)
    checkpoint_downtime_ms: int = Field(ge=0)
    total_elapsed_ms: int = Field(ge=0)
    interruption_count: int = Field(ge=0)


class CampaignWorkItem(StrictModel):
    order: int = Field(ge=0)
    phase: Literal["dev", "stability"]
    claim_id: str
    strategy: Strategy
    repeat: int = Field(ge=0, le=2)
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")


class StabilityBaselineLink(StrictModel):
    claim_id: str
    dev_adaptive_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")


class CampaignItemState(StrictModel):
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: WorkStatus = WorkStatus.PENDING
    artifact_relpath: str
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stop_reason: CampaignStopReason | None = None
    interrupted_at: str | None = None
    resumed_at: str | None = None
    errors: list[str] = Field(default_factory=list)


class CampaignPlan(StrictModel):
    schema_version: Literal["1"]
    activity_id: str
    campaign_id: str
    freeze: FreezeIdentity
    cap_micro_cny: int = Field(gt=0)
    call_bounds: CallBounds
    schedule: list[CampaignWorkItem] = Field(min_length=1)
    stability_repeat_zero_links: list[StabilityBaselineLink] = Field(min_length=20, max_length=20)
    campaign_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class RunArtifact(StrictModel):
    schema_version: Literal["1"]
    activity_id: str
    campaign_id: str
    phase: Literal["calibration", "dev", "stability"]
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_id: str
    strategy: Strategy
    repeat: int = Field(ge=0, le=2)
    result: VerificationResult
    call_ids: list[str]
    usage: Usage
    usage_source: Literal["provider", "missing", "not_applicable"]
    actual_cost_micro_cny: int | None = Field(default=None, ge=0)
    known_actual_cost_micro_cny: int = Field(ge=0)
    committed_cost_micro_cny: int = Field(ge=0)
    cost_is_lower_bound: bool
    fresh_call_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    requested_alias: str = Field(min_length=1)
    response_model_ids_raw: list[str]
    identity_verified: Literal[False] = False
    billing_uncertain: bool
    diagnostic_only: bool = False
    latency: LatencyBreakdown
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RunSummary(StrictModel):
    run_ids: list[str]
    call_ids: list[str]
    usage: Usage
    usage_sources: list[Literal["provider", "missing"]]
    actual_cost_micro_cny: int | None = Field(default=None, ge=0)
    known_actual_cost_micro_cny: int = Field(ge=0)
    committed_cost_micro_cny: int = Field(ge=0)
    cost_is_lower_bound: bool
    fresh_call_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    transport_attempts: int = Field(ge=0)
    requested_aliases: list[str]
    response_model_ids_raw: list[str]
    identity_verified: Literal[False] = False
    billing_uncertain: bool


class CampaignState(StrictModel):
    schema_version: Literal["1"]
    activity_id: str
    campaign_id: str
    status: CampaignStatus
    stop_reason: CampaignStopReason | None = None
    items: list[CampaignItemState]
    stability_repeat_zero_artifact_sha256s: dict[str, str]
    observed_response_model_ids_raw: list[str]
    billing_uncertain: bool = False
    summary: RunSummary | None = None


class CalibrationArtifactLink(StrictModel):
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ActivityRecord(StrictModel):
    schema_version: Literal["1"]
    activity_id: str
    status: CampaignStatus
    calibration_status: CampaignStatus
    dev_status: CampaignStatus
    stability_status: CampaignStatus
    calibration_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_artifacts: list[CalibrationArtifactLink] = Field(min_length=32, max_length=32)
    freeze: FreezeIdentity | None = None
    campaign_plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    campaign_state_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    observed_response_model_ids_raw: list[str]
    identity_verified: Literal[False] = False
    billing_uncertain: bool = False
    stop_reason: CampaignStopReason | None = None
    summary: RunSummary | None = None
```

`RunArtifact` validators require its run-level usage/cost/call IDs to equal
`SQLiteRunStore.summarize_run`. A normal completed/partial artifact requires complete provider usage,
exact cost, at least one call, and one non-empty raw model ID. A scored `FAILED` claim may instead be
a zero-call artifact only when usage is complete zero, cost is exactly zero,
`usage_source="not_applicable"`, model IDs are empty, and the typed error proves failure occurred
before an LLM call. A usage-missing, billing-uncertain, or model-drift artifact is
`diagnostic_only=true`, may carry null exact cost/incomplete usage, and belongs to a `STOPPED` work
item rather than the scored cohort. The artifact SHA is the SHA-256 of canonical JSON with
`artifact_sha256` omitted; the file is written once with that digest inserted. `RunSummary` is always
recomputed from stored call rows/artifacts, never incrementally trusted. `CampaignState` validators
require a one-to-one, same-order run-ID mapping to the immutable plan; completed/partial/failed item
states require an existing matching artifact SHA, `STOPPED` requires a diagnostic artifact plus a
matching typed stop reason, while pending/not-run/cancelled states reject an artifact.
`ActivityRecord` is created
before calibration and links its immutable calibration plan, 32-item state journal, and exactly 32
ordered `case_id + optional artifact_sha256` links aligned one-to-one with the plan. Links begin
with null artifact hashes and are filled only
after an atomically closed case artifact validates; publication requires all 32 to be lowercase
64-hex values. After the train-only policy is
committed, it is sealed with `FreezeIdentity` and the dev/stability `CampaignPlan`. Publication later
requires all three phase statuses to be `COMPLETE`.

Build one base permutation by sorting the three strategies on
`sha256(f"{seed}\0{strategy.value}".encode()).digest()`. For dev claim index `i`, rotate that same
base left by `i % 3` and keep its three items adjacent. Stability adds adaptive repeats 1 and 2;
each immutable `StabilityBaselineLink` binds repeat 0 to the deterministic dev adaptive run ID.
After dev completes, `CampaignState.stability_repeat_zero_artifact_sha256s` records and verifies each
linked artifact SHA before stability starts. Derive every run ID as
`sha256(f"{campaign_id}\0{phase}\0{claim_id}\0{strategy}\0{repeat}")`.

Compute `campaign_fingerprint` from canonical JSON containing the complete `FreezeIdentity`, call
bounds, cap, full persisted schedule, and all repeat-zero links, plus `activity_id`/`campaign_id`, with only the fingerprint
field omitted. This binds both freeze Git SHAs, all three runtime manifests, corpus receipt, prompt,
config, pricing, endpoint hash, requested alias, `requirements.lock`, seed, and order. Atomically
persist `plan.json` before the first dev call and `activity.json`/`campaign.json` before and after
every item. A fresh start rejects an existing campaign directory; resume reads the stored schedule
only and never rebuilds/reorders it.

- [ ] **Step 6: Implement resume and all stop conditions**

`run(plan)` and `resume(expected_identity=...)` invoke an injected async executor returning a strict
`RunArtifact`. Resume loads the immutable stored plan/schedule, compares the caller's current
`FreezeIdentity`, and skips terminal work items. At the resume boundary, atomically normalize the
single persisted `RUNNING` item left by a process kill or uncaught exception to `INTERRUPTED`, record
`PROCESS_INTERRUPTION`, and reconcile its run-store/checkpoint state before allowing any transport
call. For an `INTERRUPTED` item, inspect
`await graph.aget_state(config)`: pass `None` only when a checkpoint with values exists; if the
interruption happened before the first checkpoint, rebuild the exact initial state from the saved
work item and runtime manifest. Use `AsyncSqliteSaver.from_conn_string`, never the synchronous
checkpoint saver, because the graph uses `ainvoke`. Cover both branches with
`test_resume_running_item_without_checkpoint_uses_initial_state`,
`test_resume_reopens_persisted_running_item_after_process_crash`, and the existing post-call recovery
test. Resume never requires a graceful `KeyboardInterrupt` transition to recover a real crash.

Before any resumed network call, reload current files/env, recompute every `FreezeIdentity` field,
require clean tracked Git state at `dev_protocol_git_sha`, require `manifest_freeze_git_sha` to be its
ancestor, recompute `campaign_fingerprint`, and report the first changed field through
`FreezeMismatch`. Verify the stored plan/artifact SHA sidecars and SQLite metadata as well. Resume
cannot accept an override for cap, alias, seed, schedule, or any frozen path; it uses the saved plan.

Before accepting a scored artifact that contains paid calls, require complete provider usage and
exactly one consistent non-empty raw model ID across all calls. The zero-call failed-artifact case
uses the validator defined above. The first calibration raw ID becomes the activity observation; any
different ID during calibration, dev, stability, or inside one run persists the diagnostic artifact,
marks the current item `STOPPED` with `MODEL_DRIFT`, leaves untouched items `PENDING`, and stops as
`INCOMPLETE_MODEL_DRIFT`. `BudgetExceeded` marks untouched items `NOT_RUN_BUDGET` and stops. Missing
usage persists the paid response, marks the current item `STOPPED` with `USAGE_MISSING`, and stops as
`INCOMPLETE_USAGE`; any unresolved `sent` row or ambiguous timeout billing marks the current item
`STOPPED` with `BILLING_UNCERTAIN` and stops as `INCOMPLETE_COST_UNCERTAIN`. `KeyboardInterrupt`
changes only the current `RUNNING` item to `INTERRUPTED`, leaves untouched items `PENDING`, sets the
campaign/activity to `INTERRUPTED`, and remains resumable; only an explicit user cancellation marks
items `CANCELLED`. Calibration failure prevents dev creation. None of these statuses
may be converted to a prediction label or later overwritten as complete.

Implement `result_status_to_work_status` as the exact mapping
`COMPLETED -> COMPLETED`, `PARTIAL -> PARTIAL`, `FAILED -> FAILED`.
`derive_campaign_status` returns `COMPLETE` when every scheduled item has one of those three terminal
work states, even when quality failures exist; those partial/failed results lower full-manifest
metrics and completion rate. A terminal safety stop reason is immutable once set.
`PROCESS_INTERRUPTION` is the only resumable reason: `resume` first persists `resumed_at`, completes
the freeze/store audit, then atomically clears that current reason as the item re-enters `RUNNING`;
the interruption timestamps remain as history and determine checkpoint downtime. If recovery finds corrupt
legacy state with multiple reasons, use the fixed safety precedence
`BILLING_UNCERTAIN > USAGE_MISSING > MODEL_DRIFT > BUDGET > INTERNAL_ERROR > USER_CANCELLED > PROCESS_INTERRUPTION`;
map those reasons respectively to `INCOMPLETE_COST_UNCERTAIN`, `INCOMPLETE_USAGE`,
`INCOMPLETE_MODEL_DRIFT`, `INCOMPLETE_BUDGET`, `FAILED`, `CANCELLED`, and `INTERRUPTED`. Any
`PENDING`, `RUNNING`, `INTERRUPTED`, `STOPPED`, `NOT_RUN_BUDGET`, or `CANCELLED` item prevents
`COMPLETE`. Add
`test_keyboard_interrupt_leaves_pending_items_resumable`,
`test_campaign_complete_allows_partial_and_failed_results`, and
`test_campaign_complete_rejects_budget_and_cancelled_items`, plus parameterized
`test_stop_reason_precedence_and_status_mapping`. Only an explicit cancellation command may set
`USER_CANCELLED` or `WorkStatus.CANCELLED`.

On resume, compute downtime from persisted interruption/resumed timestamps and exclude it from fresh
latency. A cached completed LLM call contributes usage/cost exactly once through `SQLiteRunStore` but
increments cache hits and contributes no fresh model latency. Define `fresh_end_to_end_ms` as wall
time from the fresh work item's scheduled start to terminal artifact, including executor queue,
network waits, retry backoff, and every fresh model attempt; exclude checkpoint downtime and
cache-only resumed items. P50/P95 use only `fresh_end_to_end_ms`. Report `model_active_ms`, retry,
queue, downtime, cache hits, and total elapsed as separate diagnostic breakdowns without subtracting
retry or queue from the published fresh latency. Add
`test_published_latency_includes_queue_and_retry_but_excludes_checkpoint_downtime_and_cache_hits`.
`ActivityRecord.status` becomes
`COMPLETE` only when calibration, dev, and stability are all complete, every artifact/store hash
matches, usage is complete, there is no uncertain billing, the activity has one requested alias and
one raw model ID, and `identity_verified` remains honestly false.

- [ ] **Step 7: Run runner, budget, and graph recovery tests**

Run:

```powershell
conda run -n agent-collab python -m pytest tests/test_budget.py tests/test_artifacts.py tests/test_activity.py tests/test_graph.py tests/test_graph_recovery.py tests/test_campaign_runner.py -q
```

Expected: all upper bounds, cyclic rotation, frozen-input rejection, fresh resume, SQLite async
checkpoint, concurrent budget, missing usage, ambiguous billing, cross-phase drift, and cancellation
tests pass.

- [ ] **Step 8: Commit the campaign runner**

```powershell
git add configs/default.yaml src/evidence_route/config.py src/evidence_route/artifacts.py src/evidence_route/evaluation/activity.py src/evidence_route/evaluation/runner.py tests/fixtures/evaluation/campaign_factory.py tests/test_artifacts.py tests/test_activity.py tests/test_campaign_runner.py tests/test_config.py tests/test_graph.py tests/test_graph_recovery.py
git commit -m "feat: run resumable cost-bounded evaluation campaigns"
```

### Task 18: Generate Reports And Complete The Evaluation CLI

**Files:**
- Create: `src/evidence_route/evaluation/reporting.py`
- Modify: `src/evidence_route/cli.py`
- Modify: `tests/test_cli.py`
- Create: `tests/fixtures/evaluation/report_factory.py`
- Create: `tests/test_reporting.py`

- [ ] **Step 1: Write publication-gate and honest-resume tests**

```python
# tests/test_reporting.py
from pathlib import Path

import pytest

from evidence_route.evaluation.activity import CampaignStatus
from evidence_route.evaluation.reporting import PublicationBlocked, build_report_bundle


pytest_plugins = ["tests.fixtures.evaluation.report_factory"]


def test_report_uses_exact_subset_name(complete_report_input) -> None:
    bundle = build_report_bundle(complete_report_input)
    assert "AVeriTeC dev balanced subset (n=80)" in bundle.markdown
    assert "full benchmark" not in bundle.markdown.lower()
    assert "leaderboard" not in bundle.markdown.lower()


def test_resume_text_uses_measured_values_not_design_targets(complete_report_input) -> None:
    bundle = build_report_bundle(complete_report_input)
    adaptive = bundle.summary["strategies"]["adaptive"]
    assert f"{adaptive['token_reduction_pct']:.1f}%" in bundle.resume_snippet
    assert f"{adaptive['full_manifest_macro_f1']:.3f}" in bundle.resume_snippet
    assert "至少降低 40%" not in bundle.resume_snippet
    assert "官方 OpenAI" not in bundle.resume_snippet


@pytest.mark.parametrize(
    "status", [status for status in CampaignStatus if status is not CampaignStatus.COMPLETE]
)
def test_every_non_complete_status_is_diagnostic_only(
    status, report_input_factory, tmp_path: Path,
) -> None:
    report_input = report_input_factory.with_status(status)
    bundle = build_report_bundle(report_input, publish=False)
    diagnostic = tmp_path / "reports" / "incomplete" / status.value
    bundle.write(diagnostic)
    assert bundle.summary["publishable"] is False
    assert bundle.resume_snippet is None
    assert not (diagnostic / "resume_snippet.md").exists()

    final_dir = tmp_path / "reports" / "final"
    readme = tmp_path / "README.md"
    readme.write_text("unchanged", encoding="utf-8")
    with pytest.raises(PublicationBlocked, match=status.value):
        bundle.write(final_dir, readme=readme)
    with pytest.raises(PublicationBlocked, match=status.value):
        build_report_bundle(report_input, publish=True)
    assert not final_dir.exists()
    assert readme.read_text(encoding="utf-8") == "unchanged"


def test_regeneration_reads_artifacts_without_executor(complete_report_input, tmp_path: Path) -> None:
    bundle = build_report_bundle(complete_report_input)
    bundle.write(tmp_path)
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "resume_snippet.md").is_file()
```

Add CLI tests proving `evaluate` defaults to a dry-run budget preview, paid execution requires
`--accept-paid-campaign`, `report` requires a scorer gold path but no API environment, and
`calibrate --replay` does not invoke the fake executor.
`tests/fixtures/evaluation/report_factory.py` builds complete and every-status Activity/campaign/run
artifact tree under `tmp_path`, including valid self-hashes; tests never rely on hidden repository
artifacts.

- [ ] **Step 2: Run reporting tests and confirm the module is absent**

Run: `conda run -n agent-collab python -m pytest tests/test_reporting.py tests/test_cli.py -q`

Expected: FAIL importing `evidence_route.evaluation.reporting`.

- [ ] **Step 3: Build one machine-readable source of truth**

Define the public reporting boundary:

```python
@dataclass(frozen=True)
class ReportInput:
    repository_root: Path
    activity_dir: Path
    gold_manifest: Path
    nltk_data_root: Path


class PublicationGateResult(StrictModel):
    publishable: bool
    reasons: list[str]


@dataclass(frozen=True)
class ReportBundle:
    summary: dict[str, object]
    markdown: str
    resume_snippet: str | None
    representative_trace_sources: tuple[Path, ...]
    publication_gate: PublicationGateResult

    def write(self, output_dir: Path, *, readme: Path | None = None) -> None: ...
```

`build_report_bundle` loads `ActivityRecord`, calibration/campaign plans, every result JSON, the
shared SQLite run-store summary, scorer-only gold manifests, and official evaluator outputs. It
recomputes every freeze/fingerprint/artifact hash before scoring. Its `summary.json`
contains:

```json
{
  "benchmark_name": "AVeriTeC dev balanced subset (n=80)",
  "publishable": true,
  "activity_status": "complete",
  "phase_statuses": {"calibration":"complete","dev":"complete","stability":"complete"},
  "strategies": {},
  "paired_bootstrap": {},
  "stability": {},
  "official": {
    "shared_task_2024": {},
    "paper_2023_secondary": {}
  },
  "reproducibility": {},
  "limitations": []
}
```

For each strategy include full-manifest and completed-conditional metrics, completion rate,
official completed subset size, Token/cost, fresh `fresh_end_to_end_ms` P50/P95 latency, cache hits,
route/escalation/LLM
router rates, citation validity, status counts, and failure reasons. Route reporting includes a
separate `route_not_reached` count/rate for typed pre-route failures and excludes those rows from the
single-versus-multi denominator. Add paired 10,000-sample
bootstrap intervals for adaptive versus each fixed policy and a Wilson interval for three-run
stability. Mark model identity exactly as
`OpenAI-compatible provider; relay-reported model ID; identity unverified`.

- [ ] **Step 4: Render report, error analysis, and resume snippet from summary JSON**

The Markdown renderer reads only `summary.json`; it may not recalculate or hand-edit metrics. Use
fixed sections: Scope, Reproducibility, Full-Manifest Results, Completed-Only Official Results,
Cost/Latency, Routing, Stability, Failure Analysis, and Limitations. State that 3 percentage points
and 85% are engineering gates, not statistical non-inferiority claims.

Select representative traces deterministically: one accepted single, one initial multi, one
single-to-multi escalation, and one failure/partial if present, choosing the lowest SHA-256 run ID
within each category. Copy only redacted trace/result files to the report bundle. Generate the
resume snippet from actual adaptive values and observed target attainment. If a target is missed,
describe the measured tradeoff without replacing the main metric with a favorable category or
run. `publish=True` is permitted only when the complete Activity has 32 closed calibration cases,
all 54 replay candidates plus the selected config/report, all 80 x 3 dev results, and 20 x 2 extra
stability results; all three phase statuses must be `COMPLETE`. It also requires all stored/current
freeze hashes and artifact SHAs to match, complete usage, no `sent`/billing-uncertain calls, one
requested alias, and one consistent raw model ID across calibration/dev/stability. Every other status
produces `publishable=false`, no `resume_snippet.md`, cannot target `reports/final`, and cannot update
README. Terminal partial/failed verification rows do not block publication when the campaign itself
completed; they remain no-prediction rows in the headline metric and lower the reported completion
rate rather than being omitted.

- [ ] **Step 5: Add exact `evaluate`, `calibrate`, and `report` commands**

Extend `CliServices` and `create_app` with:

```text
evidence-route evaluate --manifest PATH --stability-manifest PATH --config PATH
  --pricing PATH --corpus-dir PATH --activity-dir PATH --checkpoint-db PATH
  --run-store PATH --activity-id TEXT --campaign-id TEXT
  [--start-after-calibration | --resume] [--accept-paid-campaign]

evidence-route calibrate --runtime-manifest PATH --config PATH
  --pricing PATH --corpus-dir PATH --activity-dir PATH --checkpoint-db PATH
  --run-store PATH --activity-id TEXT --output-config PATH --output-report PATH
  [--resume] (--collect --accept-paid-campaign | --replay --gold-manifest PATH)

evidence-route report --activity-dir PATH --gold-manifest PATH --output-dir PATH
  [--publish] [--readme PATH]
```

Without `--accept-paid-campaign`, `evaluate` and `calibrate --collect` print base, repair, and fault
call/cost upper bounds plus the 20% startup reserve and exit before constructing a network
transport. A paid `evaluate` requires exactly one lifecycle flag. `--start-after-calibration` is the
only phase-transition path: it attaches to an existing activity whose 32-case calibration and replay
are complete, verifies the calibration state/artifact hashes and shared run-store identity, and
atomically creates a previously absent dev/stability campaign. It refuses an existing campaign.
`--resume` instead requires that exact campaign already exists, recovers interrupted/`RUNNING` work,
and passes the full frozen-input audit; it may not create a phase. `calibrate --collect` is the only
fresh activity creator and refuses an existing activity ID. Add
`test_evaluate_start_after_calibration_creates_campaign_once`,
`test_evaluate_resume_requires_existing_campaign`, and
`test_evaluate_fresh_duplicate_campaign_is_rejected`. Both paid stages must open the same
`--run-store` and `--activity-id`. `evaluate` receives only runtime manifests.
`calibrate --collect` has no gold option and imports only the runtime
loader. A later `calibrate --replay --gold-manifest ...` scorer process reads atomically closed
saved cases and makes no network call. `report` requires no LLM variables and runs
both pinned official evaluators only on aligned completed outputs. `--readme` may be used only with
`--publish`; it replaces text strictly between the two EvidenceRoute result markers from generated
`summary.json`. Reports contain no current-clock timestamp, so identical artifacts regenerate
byte-for-byte.

- [ ] **Step 6: Run reporting and complete CLI tests**

Run:

```powershell
conda run -n agent-collab python -m pytest tests/test_reporting.py tests/test_cli.py tests/test_official_evaluator.py -q
conda run -n agent-collab evidence-route evaluate --help
conda run -n agent-collab evidence-route calibrate --help
conda run -n agent-collab evidence-route report --help
```

Expected: all tests and help commands pass; help and dry runs require neither API key nor network.

- [ ] **Step 7: Commit reports and CLI orchestration**

```powershell
git add src/evidence_route/cli.py src/evidence_route/evaluation/reporting.py tests/fixtures/evaluation/report_factory.py tests/test_cli.py tests/test_reporting.py
git commit -m "feat: report measured policy quality cost and stability"
```

### Task 19: Replace Legacy Claims With Tested Documentation And Offline CI

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `LICENSE`
- Create: `THIRD_PARTY_NOTICES.md`
- Rewrite: `README.md`
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`
- Create: `tests/test_repository_contract.py`
- Delete: `src/agent_collab/`
- Delete: `config/`
- Delete: `eval/`
- Delete: `samples/`
- Delete: `requirements.txt`
- Delete: `docs/contracts.md`
- Delete: `docs/design.md`
- Delete: `docs/INTERVIEW_PREP.md`
- Delete: `docs/plan.md`
- Delete: the 16 legacy `tests/test_core_*`, `tests/test_eval_*`, `tests/test_patterns_*`,
  `tests/test_runtime_*`, and `tests/test_tools_*` files

- [ ] **Step 1: Write repository truthfulness and offline-boundary tests**

```python
# tests/test_repository_contract.py
from importlib.util import find_spec
from pathlib import Path

import tomllib


ROOT = Path(__file__).parents[1]


def test_legacy_runtime_is_removed() -> None:
    legacy_paths = [
        "src/agent_collab", "config", "eval", "samples", "requirements.txt",
        "docs/contracts.md", "docs/design.md", "docs/INTERVIEW_PREP.md", "docs/plan.md",
    ]
    assert all(not (ROOT / path).exists() for path in legacy_paths)
    assert not list((ROOT / "tests").glob("test_core_*.py"))
    assert not list((ROOT / "tests").glob("test_eval_*.py"))
    assert not list((ROOT / "tests").glob("test_patterns_*.py"))
    assert not list((ROOT / "tests").glob("test_runtime_*.py"))
    assert not list((ROOT / "tests").glob("test_tools_*.py"))
    assert find_spec("agent_collab") is None


def test_readme_does_not_claim_gate_b_or_unverified_identity() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    forbidden = [
        "A2A", "标准 MCP", "MCP 已实现", "人工审批", "Gradio",
        "官方 OpenAI 模型", "完整 AVeriTeC benchmark",
    ]
    assert all(term not in readme for term in forbidden)
    assert "OpenAI-compatible provider" in readme
    assert "identity unverified" in readme
    assert "不是官方 leaderboard 成绩" in readme


def test_lock_contains_every_direct_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8").lower()
    declared = list(project["dependencies"])
    for group in project["optional-dependencies"].values():
        declared.extend(group)
    direct = [item.split("==", 1)[0].split("[", 1)[0].lower() for item in declared]
    assert all(f"{name}==" in lock for name in direct)


def test_env_example_contains_names_not_values() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "EVIDENCE_ROUTE_API_KEY=",
        "EVIDENCE_ROUTE_BASE_URL=",
        "EVIDENCE_ROUTE_MODEL=",
        "EVIDENCE_ROUTE_PRICE_FILE=configs/pricing.local.yaml",
    ]
```

In `tests/conftest.py`, add an autouse fixture that raises on `socket.socket.connect`,
`socket.socket.connect_ex`, and `socket.create_connection` unless the test has the `network`, `live`,
or `paid` marker. Register all three markers in `pyproject.toml`; no default test may carry one of
them.

- [ ] **Step 2: Remove legacy implementation only after the new suite passes**

Run: `conda run -n agent-collab python -m pytest -q`

Expected: all new tests pass while legacy tests still pass immediately before deletion.

Then remove the old product surface through Git so its history remains recoverable:

```powershell
git rm -r src/agent_collab config eval samples
git rm requirements.txt docs/contracts.md docs/design.md docs/INTERVIEW_PREP.md docs/plan.md
git rm tests/test_core_agent.py tests/test_core_llm_client.py tests/test_core_memory.py tests/test_core_message.py
git rm tests/test_eval_dataset.py tests/test_eval_metrics.py
git rm tests/test_patterns_debate.py tests/test_patterns_factcheck.py tests/test_patterns_parallel.py tests/test_patterns_pipeline.py tests/test_patterns_supervisor.py
git rm tests/test_runtime_approval.py tests/test_runtime_audit.py
git rm tests/test_tools_builtin.py tests/test_tools_mcp_client.py tests/test_tools_registry.py
```

Expected: only `src/evidence_route` remains as the runtime package; Git records normal deletions,
not a history rewrite.

- [ ] **Step 3: Add the code license and third-party boundaries**

Create the standard OSI MIT license text with `Copyright (c) 2026 颜炎`; do not place AVeriTeC
data/evaluator files under that grant. Create
`THIRD_PARTY_NOTICES.md` stating:

- EvidenceRoute project code is MIT.
- AVeriTeC data and the three unmodified evaluator source files are CC BY-NC 4.0 and are not
  relicensed by the project.
- Dataset citation: Michael Schlichtkrull et al., “AVeriTeC: A Dataset for Real-world Claim
  Verification with Evidence from the Web,” NeurIPS 2023 Datasets and Benchmarks.
- Exact source revisions and hashes live in `data/sources/averitec.json` and
  `third_party/averitec/SOURCES.json`.
- Downloaded corpora remain under ignored `data/processed/` and are not redistributed.

- [ ] **Step 4: Rewrite README around only Gate A capabilities**

Use this fixed section order:

```markdown
# EvidenceRoute

成本感知的自适应事实核查 Agent：在同一冻结证据与模型配置下选择 single 或 multi 路径。

## What Is Implemented
## Architecture
## Quick Start
## Prepare The Frozen Benchmark Subset
## Verify One Claim
## Run Calibration And Evaluation
## Results
<!-- EVIDENCE_ROUTE_RESULTS_START -->
Final frozen run not generated yet. Do not quote design targets as measured results.
<!-- EVIDENCE_ROUTE_RESULTS_END -->
## Artifact And Metric Definitions
## Reproducibility
## Limitations
## License And Data Attribution
```

The architecture diagram must match Task 11 node names and show one allowed escalation and three
parallel workers. Quick start uses only `EVIDENCE_ROUTE_*` environment variable names and never a
real relay URL/key/model. Explain that the provider is OpenAI-compatible, the relay-reported model
ID is self-reported, and identity is unverified. Name the evaluation only as
`AVeriTeC dev balanced subset (n=80)`. Clearly mark MCP, Chinese cases, and Streamlit as unimplemented
Gate B work and do not list them in “What Is Implemented.”

- [ ] **Step 5: Add deterministic CI**

Create `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push:
  pull_request:

jobs:
  offline:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    env:
      EVIDENCE_ROUTE_API_KEY: ""
      EVIDENCE_ROUTE_BASE_URL: ""
      EVIDENCE_ROUTE_MODEL: ""
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -r requirements.lock
      - run: python -m pip install -e . --no-deps
      - run: python -m pip check
      - run: python -m ruff check src tests scripts
      - run: python -m pytest -m "not network and not live and not paid" -q
      - run: evidence-route --help
      - run: evidence-route verify --help
      - run: evidence-route evaluate --help
      - run: evidence-route calibrate --help
      - run: evidence-route report --help
```

The default suite uses only committed fixtures and temporary directories. The evaluator
subprocess test that needs NLTK corpus downloads is marked `network`; adapter, hash, manifest,
metric, and official-schema tests remain offline and run in CI.

- [ ] **Step 6: Regenerate the lock and run the clean replacement suite**

Run:

```powershell
conda run -n agent-collab python -m piptools compile pyproject.toml --extra dev --extra eval --output-file requirements.lock
conda run -n agent-collab python -m pip install -e ".[dev,eval]"
conda run -n agent-collab python -m pip check
conda run -n agent-collab python -m ruff check src tests scripts
conda run -n agent-collab python -m pytest -m "not network and not live and not paid" -q
```

Expected: dependency check, lint, repository contract, and all deterministic Gate A tests pass with
the three API variables empty.

- [ ] **Step 7: Commit the product replacement**

```powershell
git add .github/workflows/ci.yml LICENSE THIRD_PARTY_NOTICES.md README.md pyproject.toml requirements.lock tests/conftest.py tests/test_repository_contract.py
git add -u
git commit -m "refactor: replace AgentCollab prototype with EvidenceRoute"
```

### Task 20: Freeze Inputs, Run The Paid Protocol Once, And Publish Measured Results

**Files:**
- Create from Task 13: `data/manifests/*.json`
- Create from Task 13: `data/manifests/*.sha256`
- Create: `configs/calibrated.yaml`
- Create: `reports/calibration/calibration_report.json`
- Create: `reports/final/summary.json`
- Create: `reports/final/report.md`
- Create: `reports/final/resume_snippet.md`
- Create: `reports/final/traces/`
- Modify through the report renderer: `README.md`

- [ ] **Step 1: Verify a clean implementation and local secret presence without printing values**

Run:

```powershell
$ErrorActionPreference = "Stop"
function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][scriptblock]$Command
    )
    & $Command
    $nativeExitCode = $LASTEXITCODE
    if ($nativeExitCode -ne 0) {
        throw "$Label failed with native exit code $nativeExitCode"
    }
}

$gateStatus = Invoke-NativeChecked "git status" { git status --porcelain=v1 }
if ($gateStatus) { throw "worktree is not clean" }
Invoke-NativeChecked "offline tests" {
    conda run -n agent-collab python -m pytest -m "not network and not live and not paid" -q
}
Invoke-NativeChecked "ruff" {
    conda run -n agent-collab python -m ruff check src tests scripts
}
foreach ($name in "EVIDENCE_ROUTE_API_KEY","EVIDENCE_ROUTE_BASE_URL","EVIDENCE_ROUTE_MODEL") {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
        throw "$name is not set"
    }
}
$evidenceRoutePricePath = [Environment]::GetEnvironmentVariable("EVIDENCE_ROUTE_PRICE_FILE")
if ([string]::IsNullOrWhiteSpace($evidenceRoutePricePath)) { throw "EVIDENCE_ROUTE_PRICE_FILE is not set" }
if (-not (Test-Path -LiteralPath $evidenceRoutePricePath -PathType Leaf)) { throw "pricing file does not exist" }
```

Expected: clean worktree, offline tests/lint pass, and each variable is non-empty. Do not run `Get-Item
Env:*`, `echo`, or any command that prints a secret value. The local pricing file must validate with
non-null CNY input/output rates, a dated `price_source`, and `strict_evaluation: true`; it remains
ignored and is snapshotted by hash plus non-secret rate fields into campaign artifacts.
Run Steps 1-11 in this strict PowerShell session. If the shell is reopened, redefine
`Invoke-NativeChecked` before continuing. Every native `conda`, `git`, or `evidence-route` invocation
below is wrapped so a non-zero exit code throws before the next gate.

- [ ] **Step 2: Materialize only the 32 train and 80 dev frozen corpora**

Run:

```powershell
Invoke-NativeChecked "prepare AVeriTeC" {
    conda run -n agent-collab python scripts/prepare_averitec.py --source-spec data/sources/averitec.json --output-root data/processed/averitec --runtime-manifest-root data/manifests --scorer-manifest-root data/scorer_manifests --seed 20260817 --calibration-per-label 8 --dev-per-label 20 --stability-per-label 5 --remote-timeout-s 60 --max-member-uncompressed-bytes 268435456
}
Invoke-NativeChecked "prepared corpus tests" {
    conda run -n agent-collab python -m pytest tests/test_prepare_averitec.py tests/test_averitec_provider.py -q
}
```

Expected: exactly 32 train and 80 dev claim corpora, runtime/gold manifests with 32/80/20 rows,
valid sidecar hashes, and a preparation receipt. Network transfer is approximately 2.63 GB of
selected ZIP members rather than the roughly 75 GB full compressed store. `data/processed/` stays
ignored.

- [ ] **Step 3: Freeze public manifests before any calibration or dev output is viewed**

Run:

```powershell
Invoke-NativeChecked "stage frozen manifests" {
    git add data/manifests data/scorer_manifests data/sources/averitec.json src/evidence_route/prompts.py configs/default.yaml
}
Invoke-NativeChecked "validate frozen manifest diff" { git diff --cached --check }
Invoke-NativeChecked "commit frozen manifests" {
    git commit -m "data: freeze AVeriTeC calibration and dev manifests"
}
$manifestFreezeSha = Invoke-NativeChecked "read manifest freeze SHA" { git rev-parse HEAD }
```

Expected: a commit SHA that becomes `manifest_freeze_git_sha` in every later campaign. Confirm the
runtime manifests contain no label/question/justification/gold fields. The scorer manifests are
read by calibration replay/report commands only, never by the graph process.

- [ ] **Step 4: Run the one-call provider capability gate**

Run:

```powershell
Invoke-NativeChecked "provider capability smoke" {
    conda run -n agent-collab evidence-route provider-smoke --config configs/default.yaml --pricing $evidenceRoutePricePath --artifact-dir artifacts/provider-smoke --accept-paid-call
}
```

Expected: structured nonce round-trip, complete input/output usage, estimated CNY cost, requested
alias, one non-empty relay-reported model ID, and `identity_verified=false`. On missing usage,
invalid structured output, authentication failure, or unknown billing, stop here; do not start a
campaign.

- [ ] **Step 5: Preview the entire Gate A upper bound before paid calibration**

Run the command without paid acknowledgement:

```powershell
Invoke-NativeChecked "Gate A budget preview" {
    conda run -n agent-collab evidence-route evaluate --manifest data/manifests/averitec_dev_runtime.json --stability-manifest data/manifests/averitec_stability_runtime.json --config configs/default.yaml --pricing $evidenceRoutePricePath --corpus-dir data/processed/averitec/corpora --activity-dir artifacts/evaluation/averitec-gate-a-20260817 --checkpoint-db artifacts/checkpoints.sqlite3 --run-store artifacts/gate-a-run-store.sqlite3 --activity-id averitec-gate-a-20260817 --campaign-id averitec-dev-balanced-20260817
}
```

Expected: dry-run output reports `1544` base, `3088` repair, and `9264` fault transport attempts,
node-weighted cost bounds, 20% reserve, and the CNY 350 client cap. Continue only when base cost plus
reserve is at most CNY 350. Otherwise stop before paid calibration and revise the pre-registered
token/task caps or obtain explicit approval for a higher cap; never shrink the manifest after
seeing outputs.

- [ ] **Step 6: Collect and replay the 32-row train calibration once**

Run:

```powershell
Invoke-NativeChecked "collect calibration" {
    conda run -n agent-collab evidence-route calibrate --runtime-manifest data/manifests/averitec_calibration_runtime.json --config configs/default.yaml --pricing $evidenceRoutePricePath --corpus-dir data/processed/averitec/corpora --activity-dir artifacts/evaluation/averitec-gate-a-20260817 --checkpoint-db artifacts/checkpoints.sqlite3 --run-store artifacts/gate-a-run-store.sqlite3 --activity-id averitec-gate-a-20260817 --output-config configs/calibrated.yaml --output-report reports/calibration/calibration_report.json --collect --accept-paid-campaign
}
Invoke-NativeChecked "replay calibration" {
    conda run -n agent-collab evidence-route calibrate --runtime-manifest data/manifests/averitec_calibration_runtime.json --gold-manifest data/scorer_manifests/averitec_calibration_gold.json --config configs/default.yaml --pricing $evidenceRoutePricePath --corpus-dir data/processed/averitec/corpora --activity-dir artifacts/evaluation/averitec-gate-a-20260817 --checkpoint-db artifacts/checkpoints.sqlite3 --run-store artifacts/gate-a-run-store.sqlite3 --activity-id averitec-gate-a-20260817 --output-config configs/calibrated.yaml --output-report reports/calibration/calibration_report.json --replay
}
```

Expected: 32 immutable case artifacts; all 54 candidates replayed without additional model calls;
`configs/calibrated.yaml` and `reports/calibration/calibration_report.json` identify the selected
train-only policy and whether its engineering quality floor was met.

- [ ] **Step 7: Freeze calibrated policy, prompts, pricing identity, and code before dev**

Run:

```powershell
Invoke-NativeChecked "stage calibrated policy" {
    git add configs/calibrated.yaml reports/calibration/calibration_report.json
}
Invoke-NativeChecked "validate calibrated policy diff" { git diff --cached --check }
Invoke-NativeChecked "commit calibrated policy" {
    git commit -m "eval: freeze train-calibrated EvidenceRoute policy"
}
$devProtocolSha = Invoke-NativeChecked "read dev protocol SHA" { git rev-parse HEAD }
```

Expected: the new SHA becomes `dev_protocol_git_sha`. From this point until the first dev campaign
finishes, do not edit prompt strings, manifests, routing thresholds, token caps, prices, provider
alias, or scoring code. A later change requires a new campaign labeled `exploratory` and cannot
replace this frozen result.

- [ ] **Step 8: Execute the interleaved dev and stability campaign**

Run:

```powershell
Invoke-NativeChecked "start dev phase after calibration" {
    conda run -n agent-collab evidence-route evaluate --manifest data/manifests/averitec_dev_runtime.json --stability-manifest data/manifests/averitec_stability_runtime.json --config configs/calibrated.yaml --pricing $evidenceRoutePricePath --corpus-dir data/processed/averitec/corpora --activity-dir artifacts/evaluation/averitec-gate-a-20260817 --checkpoint-db artifacts/checkpoints.sqlite3 --run-store artifacts/gate-a-run-store.sqlite3 --activity-id averitec-gate-a-20260817 --campaign-id averitec-dev-balanced-20260817 --start-after-calibration --accept-paid-campaign
}
```

If the process is interrupted without a semantic stop condition, resume with the identical command
after replacing `--start-after-calibration` with `--resume`. Expected complete counts are 240 dev
strategy items plus 40 extra adaptive
stability items. If status becomes `incomplete_budget`, `incomplete_model_drift`,
`incomplete_usage`, `incomplete_cost_uncertain`, or `cancelled`, do not change a label, drop a row,
or run `report --publish`; preserve the campaign and generate only a non-publishable diagnostic
report.

- [ ] **Step 9: Score completed outputs and update README from generated data**

First prepare the pinned NLTK assets for the evaluators, then run:

```powershell
Invoke-NativeChecked "prepare pinned NLTK data" {
    conda run -n agent-collab python scripts/prepare_nltk_data.py --source-spec data/sources/nltk_data.json --output-root data/external/nltk
}
$env:NLTK_DATA = (Resolve-Path data/external/nltk).Path
Invoke-NativeChecked "publish report" {
    conda run -n agent-collab evidence-route report --activity-dir artifacts/evaluation/averitec-gate-a-20260817 --gold-manifest data/scorer_manifests/averitec_dev_gold.json --output-dir reports/final --publish --readme README.md
}
```

Expected: `summary.json`, `report.md`, `resume_snippet.md`, and redacted representative traces;
README result markers are replaced from the same summary. The headline quality number is
full-manifest macro-F1, and official completed-only scores always show completion rate/subset size.
All text uses `AVeriTeC dev balanced subset (n=80)` and identity-unverified wording.

- [ ] **Step 10: Prove deterministic regeneration and run the final repository gate**

Run:

```powershell
New-Item -ItemType Directory -Force artifacts/report-repro | Out-Null
Invoke-NativeChecked "regenerate report" {
    conda run -n agent-collab evidence-route report --activity-dir artifacts/evaluation/averitec-gate-a-20260817 --gold-manifest data/scorer_manifests/averitec_dev_gold.json --output-dir artifacts/report-repro --publish
}
if (Compare-Object (Get-Content -Raw reports/final/summary.json) (Get-Content -Raw artifacts/report-repro/summary.json)) { throw "summary regeneration differs" }
if (Compare-Object (Get-Content -Raw reports/final/report.md) (Get-Content -Raw artifacts/report-repro/report.md)) { throw "report regeneration differs" }
if (Compare-Object (Get-Content -Raw reports/final/resume_snippet.md) (Get-Content -Raw artifacts/report-repro/resume_snippet.md)) { throw "resume snippet regeneration differs" }
Invoke-NativeChecked "dependency check" { conda run -n agent-collab python -m pip check }
Invoke-NativeChecked "final ruff" { conda run -n agent-collab python -m ruff check src tests scripts }
Invoke-NativeChecked "final offline tests" {
    conda run -n agent-collab python -m pytest -m "not live and not paid" -q
}
Invoke-NativeChecked "final diff check" { git diff --check }
```

Expected: all three comparisons return no difference (otherwise they throw); dependency, lint, offline plus pinned
evaluator tests, and diff checks pass. Open `reports/final/report.md` and verify every stated number
exists in `summary.json`; do not manually type or round a different resume number.

- [ ] **Step 11: Commit only publishable measured artifacts**

```powershell
Invoke-NativeChecked "stage publishable reports" { git add README.md reports/final }
Invoke-NativeChecked "validate report diff" { git diff --cached --check }
Invoke-NativeChecked "commit measured reports" {
    git commit -m "docs: publish frozen EvidenceRoute evaluation results"
}
$finalStatus = Invoke-NativeChecked "final git status" { git status --porcelain=v1 }
if ($finalStatus) { throw "worktree is not clean after final commit" }
```

Expected: the final worktree is clean. If Step 8 was incomplete, replace `reports/final` with
`reports/incomplete/averitec-dev-balanced-20260817`, omit `resume_snippet.md`, do not update README
result markers, and commit the diagnostic report with message
`docs: record incomplete EvidenceRoute evaluation campaign` instead.
