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
    def validate_total(self) -> Usage:
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


class VerdictDraft(StrictModel):
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class DecompositionDraft(StrictModel):
    tasks: list[VerificationTask] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_unique_tasks(self) -> DecompositionDraft:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("decomposition task ids must be unique")
        return self


class WorkerDraft(StrictModel):
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    citations: list[Citation] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


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
    def validate_status_fields(self) -> WorkerResult:
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
    def validate_status_fields(self) -> VerificationResult:
        cost_fields = (
            self.estimated_cost_micro_cny,
            self.cost_currency,
            self.price_config_id,
        )
        if not self.usage.complete and any(field is not None for field in cost_fields):
            raise ValueError("incomplete usage forbids exact cost and pricing fields")
        if self.usage.complete and any(field is not None for field in cost_fields) and not all(
            field is not None for field in cost_fields
        ):
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
    node_timings: Annotated[list[NodeTiming], operator.add]
    escalated: bool
    usage: Usage
    final_result: VerificationResult
    escalation_count: int
    errors: Annotated[list[str], operator.add]
