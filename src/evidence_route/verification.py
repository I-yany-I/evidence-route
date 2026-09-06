from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any, Literal

from evidence_route.artifacts import BillingStateError
from evidence_route.budget import BudgetExceeded, UsageUnavailable
from evidence_route.config import EvidenceSettings, GenerationSettings
from evidence_route.contracts import (
    Citation,
    ClaimFeatures,
    DecompositionDraft,
    Evidence,
    ResultStatus,
    Usage,
    Verdict,
    VerdictDraft,
    VerificationResult,
    VerificationTask,
    WorkerDraft,
    WorkerResult,
)
from evidence_route.evaluation.stability import (
    canonicalize_citation_http_url,
    canonicalize_citation_url,
)
from evidence_route.llm import BillingUncertain
from evidence_route.prompts import (
    decomposer_messages,
    hardened_judge_messages,
    hardened_single_messages,
    hardened_worker_messages,
    judge_messages,
    single_messages,
    worker_messages,
)

_MAX_CITATION_CANDIDATES = 16


def _ensure_citation_candidate_limit(citations: Sequence[Citation]) -> None:
    if len(citations) > _MAX_CITATION_CANDIDATES:
        raise ValueError("citation projection accepts at most 16 candidates")


def _safety_stop(error: Exception) -> bool:
    return isinstance(
        error, (BudgetExceeded, UsageUnavailable, BillingUncertain, BillingStateError)
    )


def _usage(response: Any) -> Usage:
    value = getattr(response, "usage", None)
    if isinstance(value, Usage):
        return value
    return Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=False)


def _evidence_sort_key(item: Evidence, index: int) -> tuple[object, ...]:
    return (
        -item.ranking_score,
        item.evidence_id,
        canonicalize_citation_url(str(item.source_url)),
        item.snapshot_sha256,
        index,
    )


def _citation_provenance_key(item: Citation) -> tuple[object, ...]:
    return (
        item.evidence_id,
        canonicalize_citation_url(str(item.source_url)),
        tuple(sorted(set(item.claim_unit_ids))),
        item.question,
        item.answer,
        item.quote,
        item.stance,
    )


def _deduplicate_citations(citations: Sequence[Citation]) -> list[Citation]:
    unique: dict[tuple[object, ...], Citation] = {}
    for citation in citations:
        unique.setdefault(_citation_provenance_key(citation), citation)
    return [unique[key] for key in sorted(unique)]


def project_citations(
    citations: Sequence[Citation],
    evidence: Sequence[Evidence] | Sequence[str],
    *,
    max_citations: int,
    known_claim_unit_ids: Sequence[str] | None = None,
) -> list[Citation]:
    """Project model citations onto a deterministic, evidence-backed representation."""
    if max_citations < 1:
        raise ValueError("max_citations must be positive")
    _ensure_citation_candidate_limit(citations)

    if all(isinstance(item, Evidence) for item in evidence):
        ordered_ids = []
        known_source_urls: dict[str, set[str]] = {}
        for _, item in sorted(
            enumerate(evidence), key=lambda pair: _evidence_sort_key(pair[1], pair[0])
        ):
            if item.evidence_id not in ordered_ids:
                ordered_ids.append(item.evidence_id)
            known_source_urls.setdefault(item.evidence_id, set()).add(
                canonicalize_citation_url(str(item.source_url))
            )
    elif all(isinstance(item, str) for item in evidence):
        ordered_ids = sorted(set(evidence))
        known_source_urls = {}
    else:
        raise TypeError("evidence must contain only Evidence records or evidence IDs")
    evidence_rank = {evidence_id: index for index, evidence_id in enumerate(ordered_ids)}
    known_units = set(known_claim_unit_ids) if known_claim_unit_ids is not None else None

    candidates: list[tuple[tuple[object, ...], Citation, str]] = []
    for citation in citations:
        if citation.evidence_id not in evidence_rank:
            continue
        claim_unit_ids = set(citation.claim_unit_ids)
        if known_units is not None:
            claim_unit_ids &= known_units
        if not claim_unit_ids:
            continue
        normalized = citation.model_copy(
            update={
                "claim_unit_ids": sorted(claim_unit_ids),
                "source_url": canonicalize_citation_http_url(str(citation.source_url)),
            }
        )
        source_url = canonicalize_citation_url(str(normalized.source_url))
        if known_source_urls and source_url not in known_source_urls.get(
            normalized.evidence_id, set()
        ):
            continue
        key = (
            evidence_rank[normalized.evidence_id],
            normalized.evidence_id,
            source_url,
            tuple(sorted(set(normalized.claim_unit_ids))),
            normalized.question,
            normalized.answer,
            normalized.quote,
            normalized.stance,
        )
        candidates.append((key, normalized, source_url))
    candidates.sort(key=lambda item: item[0])
    # Select the citation subset with maximal unit coverage. Dynamic programming keeps
    # the bounded citation cap while avoiding greedy choices that can miss full coverage.
    states: dict[
        tuple[int, frozenset[str], frozenset[str], frozenset[str]], tuple[int, ...]
    ] = {(0, frozenset(), frozenset(), frozenset()): ()}
    for index, (_, citation, source_url) in enumerate(candidates):
        citation_units = frozenset(citation.claim_unit_ids)
        for (count, covered, used_urls, used_evidence_ids), path in list(states.items()):
            if (
                count >= max_citations
                or source_url in used_urls
                or citation.evidence_id in used_evidence_ids
            ):
                continue
            next_state = (
                count + 1,
                covered | citation_units,
                used_urls | {source_url},
                used_evidence_ids | {citation.evidence_id},
            )
            next_path = (*path, index)
            current_path = states.get(next_state)
            if current_path is None or next_path < current_path:
                states[next_state] = next_path

    best_state, selected_path = min(
        states.items(),
        key=lambda item: (-len(item[0][1]), item[0][0], item[1]),
    )
    del best_state
    return [candidates[index][1] for index in selected_path]


@dataclass(frozen=True)
class VerificationEnvelope:
    result: Any
    evidence_ids: frozenset[str]


def result_from_draft(
    response: Any,
    claim_id: str,
    initial_route: Literal["single", "multi"],
    evidence: list[Evidence],
    *,
    available_evidence_ids: list[str] | None = None,
    known_claim_unit_ids: list[str] | None = None,
    escalated: bool = False,
) -> VerificationResult:
    draft: VerdictDraft = response.value
    resolved_available_ids = (
        [item.evidence_id for item in evidence]
        if available_evidence_ids is None
        else list(available_evidence_ids)
    )
    supplied_ids = (
        {item.evidence_id for item in evidence}
        if available_evidence_ids is None
        else set(available_evidence_ids)
    )
    supplied_evidence = [item for item in evidence if item.evidence_id in supplied_ids]
    return VerificationResult(
        claim_id=claim_id,
        status=ResultStatus.COMPLETED,
        verdict=draft.verdict,
        confidence=draft.confidence,
        rationale=draft.rationale,
        citations=project_citations(
            draft.citations,
            supplied_evidence,
            max_citations=max(1, len(evidence)),
            known_claim_unit_ids=known_claim_unit_ids,
        ),
        available_evidence_ids=resolved_available_ids,
        initial_route=initial_route,
        escalated=escalated,
        usage=_usage(response),
    )


def worker_result_from_draft(
    response: Any,
    task: VerificationTask,
    evidence: list[Evidence],
    *,
    available_evidence_ids: list[str] | None = None,
) -> WorkerResult:
    draft: WorkerDraft = response.value
    resolved_available_ids = (
        [item.evidence_id for item in evidence]
        if available_evidence_ids is None
        else list(available_evidence_ids)
    )
    supplied_ids = (
        {item.evidence_id for item in evidence}
        if available_evidence_ids is None
        else set(available_evidence_ids)
    )
    supplied_evidence = [item for item in evidence if item.evidence_id in supplied_ids]
    status = ResultStatus.PARTIAL if draft.errors else ResultStatus.COMPLETED
    return WorkerResult(
        task_id=task.task_id,
        claim_unit_ids=task.claim_unit_ids,
        status=status,
        verdict=draft.verdict,
        confidence=draft.confidence,
        citations=project_citations(
            draft.citations,
            supplied_evidence,
            max_citations=max(1, len(evidence)),
            known_claim_unit_ids=task.claim_unit_ids,
        ),
        available_evidence_ids=resolved_available_ids,
        usage=_usage(response),
        errors=draft.errors,
    )


def failed_result(
    claim_id: str,
    *,
    initial_route: Literal["single", "multi"] | None,
    failure_stage: Literal["pre_route", "single", "decompose", "judge", "validation"],
    error_code: str,
) -> VerificationResult:
    return VerificationResult(
        claim_id=claim_id,
        status=ResultStatus.FAILED,
        rationale=error_code,
        initial_route=initial_route,
        failure_stage=failure_stage,
        usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=True),
        errors=[error_code],
    )


class SingleVerifier:
    def __init__(
        self,
        provider: Any,
        llm: Any,
        evidence_settings: EvidenceSettings,
        generation: GenerationSettings,
        *,
        hardened: bool = False,
    ) -> None:
        self.provider = provider
        self.llm = llm
        self.evidence_settings = evidence_settings
        self.generation = generation
        self.hardened = hardened

    async def verify(
        self, run_id: str, claim_id: str, claim: str, features: ClaimFeatures
    ) -> VerificationResult:
        envelope = await self.verify_with_evidence(run_id, claim_id, claim, features)
        return envelope.result

    async def verify_with_evidence(
        self, run_id: str, claim_id: str, claim: str, features: ClaimFeatures
    ) -> VerificationEnvelope:
        try:
            evidence = await self.provider.search(
                claim_id,
                claim,
                top_k=self.evidence_settings.single_top_k,
                max_chars=self.evidence_settings.single_chars,
                **(
                    {"max_per_source": self.evidence_settings.max_per_source}
                    if self.evidence_settings.max_per_source is not None
                    else {}
                ),
            )
            response = await self.llm.invoke(
                run_id=run_id,
                node="single",
                task_id="root",
                messages=(
                    hardened_single_messages(claim, features, evidence)
                    if self.hardened
                    else single_messages(claim, features, evidence)
                ),
                schema=VerdictDraft,
                max_input_tokens=self.generation.single.max_input_tokens,
                max_output_tokens=self.generation.single.max_output_tokens,
            )
            return VerificationEnvelope(
                result=result_from_draft(
                    response,
                    claim_id,
                    "single",
                    evidence,
                    known_claim_unit_ids=[item.unit_id for item in features.claim_units],
                ),
                evidence_ids=frozenset(item.evidence_id for item in evidence),
            )
        except Exception as exc:
            if _safety_stop(exc):
                raise
            return VerificationEnvelope(
                result=failed_result(
                    claim_id,
                    initial_route="single",
                    failure_stage="single",
                    error_code="SINGLE_VERIFICATION_FAILED",
                ),
                evidence_ids=frozenset(),
            )


class ClaimDecomposer:
    def __init__(
        self, llm: Any, generation: GenerationSettings, *, deterministic: bool = False
    ) -> None:
        self.llm = llm
        self.generation = generation
        self.deterministic = deterministic

    async def decompose(
        self, run_id: str, claim: str, features: ClaimFeatures
    ) -> list[VerificationTask]:
        if self.deterministic:
            units = features.claim_units
            task_count = min(3, len(units))
            chunk_size = ceil(len(units) / task_count)
            return [
                VerificationTask(
                    task_id=f"t{index}",
                    claim_unit_ids=[unit.unit_id for unit in units[start : start + chunk_size]],
                    query=" ".join(unit.text for unit in units[start : start + chunk_size]),
                )
                for index, start in enumerate(range(0, len(units), chunk_size))
            ][:3]
        response = await self.llm.invoke(
            run_id=run_id,
            node="decomposer",
            task_id="root",
            messages=decomposer_messages(claim, features),
            schema=DecompositionDraft,
            max_input_tokens=self.generation.decomposer.max_input_tokens,
            max_output_tokens=self.generation.decomposer.max_output_tokens,
        )
        tasks = response.value.tasks
        known = {item.unit_id for item in features.claim_units}
        if any(set(task.claim_unit_ids) - known for task in tasks):
            raise ValueError("decomposition references an unknown claim unit")
        return tasks


class EvidenceWorker:
    def __init__(
        self,
        provider: Any,
        llm: Any,
        evidence_settings: EvidenceSettings,
        generation: GenerationSettings,
        *,
        hardened: bool = False,
    ) -> None:
        self.provider = provider
        self.llm = llm
        self.evidence_settings = evidence_settings
        self.generation = generation
        self.hardened = hardened

    async def verify_task(self, run_id: str, claim_id: str, task: VerificationTask) -> WorkerResult:
        envelope = await self.verify_task_with_evidence(run_id, claim_id, task)
        return envelope.result

    async def verify_task_with_evidence(
        self, run_id: str, claim_id: str, task: VerificationTask
    ) -> VerificationEnvelope:
        try:
            evidence = await self.provider.search(
                claim_id,
                task.query,
                top_k=self.evidence_settings.worker_top_k,
                max_chars=self.evidence_settings.worker_chars,
                **(
                    {"max_per_source": self.evidence_settings.max_per_source}
                    if self.evidence_settings.max_per_source is not None
                    else {}
                ),
            )
            response = await self.llm.invoke(
                run_id=run_id,
                node="worker",
                task_id=task.task_id,
                messages=(
                    hardened_worker_messages(task, evidence)
                    if self.hardened
                    else worker_messages(task, evidence)
                ),
                schema=WorkerDraft,
                max_input_tokens=self.generation.worker.max_input_tokens,
                max_output_tokens=self.generation.worker.max_output_tokens,
            )
            return VerificationEnvelope(
                result=worker_result_from_draft(response, task, evidence),
                evidence_ids=frozenset(item.evidence_id for item in evidence),
            )
        except Exception as exc:
            if _safety_stop(exc):
                raise
            return VerificationEnvelope(
                result=WorkerResult(
                    task_id=task.task_id,
                    claim_unit_ids=task.claim_unit_ids,
                    status=ResultStatus.FAILED,
                    verdict=None,
                    confidence=None,
                    usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0, complete=True),
                    errors=[f"WORKER_VERIFICATION_FAILED:{type(exc).__name__}"],
                ),
                evidence_ids=frozenset(),
            )


class VerdictJudge:
    def __init__(
        self,
        llm: Any,
        evidence_settings: EvidenceSettings,
        generation: GenerationSettings,
        *,
        hardened: bool = False,
    ) -> None:
        self.llm = llm
        self.evidence_settings = evidence_settings
        self.generation = generation
        self.hardened = hardened

    async def judge(
        self,
        run_id: str,
        claim_id: str,
        claim: str,
        workers: list[WorkerResult],
        *,
        initial_route: Literal["single", "multi"],
        escalated: bool,
    ) -> VerificationResult:
        response = await self.llm.invoke(
            run_id=run_id,
            node="judge",
            task_id="root",
            messages=(
                hardened_judge_messages(claim, workers)
                if self.hardened
                else judge_messages(claim, workers)
            ),
            schema=VerdictDraft,
            max_input_tokens=self.generation.judge.max_input_tokens,
            max_output_tokens=self.generation.judge.max_output_tokens,
        )
        draft: VerdictDraft = response.value
        _ensure_citation_candidate_limit(draft.citations)
        available = sorted(
            {evidence_id for worker in workers for evidence_id in worker.available_evidence_ids}
        )
        worker_claim_unit_ids = {
            unit_id
            for worker in workers
            for unit_id in getattr(worker, "claim_unit_ids", [])
        }
        selected_provenance = {
            (citation.evidence_id, canonicalize_citation_url(str(citation.source_url)))
            for citation in draft.citations
        }
        trusted_citations = _deduplicate_citations(
            [
                citation
                for worker in workers
                for citation in worker.citations
                if citation.evidence_id in worker.available_evidence_ids
                and (
                    citation.evidence_id,
                    canonicalize_citation_url(str(citation.source_url)),
                )
                in selected_provenance
            ]
        )
        citations = project_citations(
            trusted_citations,
            available,
            max_citations=self.evidence_settings.judge_max_evidence,
            known_claim_unit_ids=sorted(worker_claim_unit_ids) or None,
        )
        usage = _usage(response)
        total_input = sum(worker.usage.input_tokens for worker in workers) + usage.input_tokens
        total_output = sum(worker.usage.output_tokens for worker in workers) + usage.output_tokens
        result_usage = Usage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            complete=all(worker.usage.complete for worker in workers) and usage.complete,
        )
        partial_workers = [
            worker for worker in workers if worker.status is ResultStatus.PARTIAL
        ]
        recoverable_insufficient = bool(partial_workers) and all(
            worker.verdict is Verdict.NOT_ENOUGH_EVIDENCE
            and worker.usage.complete
            for worker in partial_workers
        )
        partial = any(worker.status is ResultStatus.FAILED for worker in workers) or (
            bool(partial_workers)
            and not (
                draft.verdict is Verdict.NOT_ENOUGH_EVIDENCE
                and recoverable_insufficient
                and result_usage.complete
            )
        )
        return VerificationResult(
            claim_id=claim_id,
            status=ResultStatus.PARTIAL if partial else ResultStatus.COMPLETED,
            verdict=draft.verdict,
            confidence=draft.confidence,
            rationale=draft.rationale,
            citations=citations,
            available_evidence_ids=available,
            initial_route=initial_route,
            escalated=escalated,
            usage=result_usage,
        )
