"""Deterministic reporting over immutable Gate A evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from evidence_route.contracts import ResultStatus, Strategy, StrictModel, Verdict
from evidence_route.evaluation.activity import (
    ActivityRecord,
    CampaignPlan,
    CampaignState,
    CampaignStatus,
    RunArtifact,
    campaign_fingerprint,
    verify_artifact_fingerprint,
)
from evidence_route.evaluation.metrics import (
    paired_bootstrap_difference,
    score_completed_conditionally,
    score_full_manifest,
    stability_consistency,
    summarize_operations,
)
from evidence_route.evaluation.scorer_manifest import load_gold_manifest

BENCHMARK_NAME = "AVeriTeC dev balanced subset (n=80)"
IDENTITY_DESCRIPTION = (
    "OpenAI-compatible provider; relay-reported model ID; identity unverified"
)
_RESULTS_START = "<!-- EVIDENCE_ROUTE_RESULTS_START -->"
_RESULTS_END = "<!-- EVIDENCE_ROUTE_RESULTS_END -->"


class PublicationBlocked(RuntimeError):
    """Raised when diagnostic artifacts are requested for publication."""


@dataclass(frozen=True)
class ReportInput:
    repository_root: Path
    activity_dir: Path
    gold_manifest: Path
    nltk_data_root: Path


class PublicationGateResult(StrictModel):
    publishable: bool
    reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ReportBundle:
    summary: dict[str, object]
    markdown: str
    resume_snippet: str | None
    representative_trace_sources: tuple[Path, ...]
    publication_gate: PublicationGateResult

    def write(self, output_dir: Path, *, readme: Path | None = None) -> None:
        output_dir = Path(output_dir)
        if not self.publication_gate.publishable and (
            readme is not None or output_dir.name.lower() == "final"
        ):
            raise PublicationBlocked("; ".join(self.publication_gate.reasons))
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "summary.json", self.summary)
        _write_text(output_dir / "report.md", self.markdown)
        resume_path = output_dir / "resume_snippet.md"
        if self.resume_snippet is not None:
            _write_text(resume_path, self.resume_snippet)
        elif resume_path.exists():
            resume_path.unlink()
        trace_dir = output_dir / "traces"
        if self.representative_trace_sources:
            trace_dir.mkdir(parents=True, exist_ok=True)
            for source in self.representative_trace_sources:
                shutil.copyfile(source, trace_dir / source.name)
        if readme is not None:
            if not self.publication_gate.publishable:
                raise PublicationBlocked("; ".join(self.publication_gate.reasons))
            _update_readme(Path(readme), self.summary)


def _write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    _write_text(path, encoded)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, relpath: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / relpath).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"artifact path leaves activity directory: {relpath}") from exc
    if not resolved.is_file():
        raise ValueError(f"artifact file is missing: {relpath}")
    return resolved


def _load_models(
    report_input: ReportInput,
) -> tuple[ActivityRecord, CampaignPlan, CampaignState, Path, Path]:
    activity_dir = Path(report_input.activity_dir).resolve()
    activity_path = activity_dir / "activity.json"
    plan_path = activity_dir / "plan.json"
    state_path = activity_dir / "campaign.json"
    if not all(path.is_file() for path in (activity_path, plan_path, state_path)):
        raise ValueError(
            "activity directory must contain activity.json, plan.json, and campaign.json"
        )
    activity = ActivityRecord.model_validate_json(activity_path.read_text(encoding="utf-8"))
    plan = CampaignPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    state = CampaignState.model_validate_json(state_path.read_text(encoding="utf-8"))
    expected_fingerprint = campaign_fingerprint(plan)
    if plan.campaign_fingerprint != expected_fingerprint:
        raise ValueError("campaign plan fingerprint mismatch")
    if (activity.activity_id, state.activity_id, state.campaign_id) != (
        plan.activity_id,
        plan.activity_id,
        plan.campaign_id,
    ):
        raise ValueError("activity, plan, and state identities differ")
    expected_run_ids = [item.run_id for item in plan.schedule]
    actual_run_ids = [item.run_id for item in state.items]
    if actual_run_ids != expected_run_ids:
        raise ValueError("campaign state order differs from immutable plan")
    return activity, plan, state, plan_path, state_path


def _load_artifacts(
    activity_dir: Path,
    plan: CampaignPlan,
    state: CampaignState,
) -> tuple[dict[str, RunArtifact], dict[str, Path]]:
    work_by_id = {item.run_id: item for item in plan.schedule}
    artifacts: dict[str, RunArtifact] = {}
    paths: dict[str, Path] = {}
    for item_state in state.items:
        if item_state.artifact_sha256 is None:
            continue
        path = _inside(activity_dir, item_state.artifact_relpath)
        artifact = RunArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        verify_artifact_fingerprint(artifact)
        work = work_by_id[item_state.run_id]
        if (
            artifact.artifact_sha256 != item_state.artifact_sha256
            or artifact.run_id != work.run_id
            or artifact.claim_id != work.claim_id
            or artifact.strategy is not work.strategy
            or artifact.phase != work.phase
            or artifact.repeat != work.repeat
            or artifact.activity_id != plan.activity_id
            or artifact.campaign_id != plan.campaign_id
        ):
            raise ValueError(f"artifact identity mismatch: {item_state.run_id}")
        artifacts[artifact.run_id] = artifact
        paths[artifact.run_id] = path
    return artifacts, paths


def _strategy_summary(
    strategy: Strategy,
    gold: dict[str, Verdict],
    artifacts: list[RunArtifact],
) -> tuple[dict[str, object], dict[str, str]]:
    results = {
        artifact.claim_id: artifact.result
        for artifact in artifacts
        if artifact.phase == "dev"
        and artifact.strategy is strategy
        and artifact.repeat == 0
    }
    full = score_full_manifest(gold, results)
    conditional = score_completed_conditionally(gold, results)
    operations = [
        {
            "usage": artifact.usage.model_dump(mode="json"),
            "estimated_cost_micro_cny": artifact.actual_cost_micro_cny,
            "fresh_end_to_end_ms": artifact.latency.fresh_end_to_end_ms,
            "route": artifact.result.initial_route,
            "escalated": artifact.result.escalated,
            "cache_hit_count": artifact.cache_hit_count,
            "errors": artifact.result.errors,
        }
        for artifact in artifacts
        if artifact.phase == "dev"
        and artifact.strategy is strategy
        and artifact.repeat == 0
    ]
    operation_summary = summarize_operations(operations)
    route_not_reached = sum(
        artifact.result.initial_route is None
        for artifact in artifacts
        if artifact.phase == "dev"
        and artifact.strategy is strategy
        and artifact.repeat == 0
    )
    route_denominator = max(0, len(operations) - route_not_reached)
    route_counts = Counter(
        artifact.result.initial_route
        for artifact in artifacts
        if artifact.phase == "dev"
        and artifact.strategy is strategy
        and artifact.repeat == 0
        and artifact.result.initial_route is not None
    )
    status_counts = Counter(result.status.value for result in results.values())
    errors: Counter[str] = Counter()
    for result in results.values():
        errors.update(result.errors)
    payload: dict[str, object] = {
        "full_manifest_macro_f1": full.macro_f1,
        "full_manifest_accuracy": full.accuracy,
        "completed_conditional_macro_f1": conditional.macro_f1,
        "completed_conditional_accuracy": conditional.accuracy,
        "completion_rate": full.completion_rate,
        "completed_count": full.completed_count,
        "manifest_count": full.sample_count,
        "official_completed_subset_size": conditional.sample_count,
        "input_tokens": operation_summary.input_tokens,
        "output_tokens": operation_summary.output_tokens,
        "total_tokens": operation_summary.total_tokens,
        "mean_tokens_per_claim": operation_summary.mean_tokens_per_claim,
        "actual_cost_micro_cny": operation_summary.exact_cost_micro_cny,
        "fresh_latency_p50_ms": operation_summary.latency_p50_ms,
        "fresh_latency_p95_ms": operation_summary.latency_p95_ms,
        "cache_hit_count": operation_summary.cache_hit_count,
        "route_counts": dict(route_counts),
        "route_not_reached_count": route_not_reached,
        "route_not_reached_rate": route_not_reached / len(operations) if operations else 0.0,
        "single_route_rate": (
            route_counts["single"] / route_denominator if route_denominator else 0.0
        ),
        "multi_route_rate": (
            route_counts["multi"] / route_denominator if route_denominator else 0.0
        ),
        "escalation_rate": operation_summary.escalation_rate,
        "llm_router_rate": None,
        "citation_validity_rate": operation_summary.citation_validity_rate,
        "status_counts": dict(status_counts),
        "failure_reasons": dict(errors),
        "token_reduction_pct": 0.0,
        "cost_reduction_pct": 0.0,
    }
    return payload, full.scored_predictions


def _publication_gate(
    report_input: ReportInput,
    activity: ActivityRecord,
    plan: CampaignPlan,
    state: CampaignState,
    plan_path: Path,
    state_path: Path,
    artifacts: dict[str, RunArtifact],
) -> PublicationGateResult:
    reasons: list[str] = []
    if activity.status is not CampaignStatus.COMPLETE:
        reasons.append(f"activity_status={activity.status.value}")
    for phase, status in (
        ("calibration", activity.calibration_status),
        ("dev", activity.dev_status),
        ("stability", activity.stability_status),
    ):
        if status is not CampaignStatus.COMPLETE:
            reasons.append(f"{phase}_status={status.value}")
    if state.status is not CampaignStatus.COMPLETE:
        reasons.append(f"campaign_status={state.status.value}")
    if len(activity.calibration_artifacts) != 32 or any(
        link.artifact_sha256 is None for link in activity.calibration_artifacts
    ):
        reasons.append("calibration_artifacts_not_closed")
    dev_items = [item for item in plan.schedule if item.phase == "dev"]
    stability_items = [item for item in plan.schedule if item.phase == "stability"]
    if len(dev_items) != 240:
        reasons.append(f"dev_item_count={len(dev_items)}")
    if len(stability_items) != 40:
        reasons.append(f"stability_item_count={len(stability_items)}")
    if len(artifacts) != len(plan.schedule):
        reasons.append(f"artifact_count={len(artifacts)}")
    if activity.campaign_plan_sha256 != _sha256(plan_path):
        reasons.append("campaign_plan_sha256_mismatch")
    if activity.campaign_state_sha256 != _sha256(state_path):
        reasons.append("campaign_state_sha256_mismatch")
    if activity.freeze != plan.freeze:
        reasons.append("freeze_identity_mismatch")
    if activity.billing_uncertain or state.billing_uncertain:
        reasons.append("billing_uncertain")
    if activity.summary is None or not activity.summary.usage.complete:
        reasons.append("activity_usage_incomplete")
    if activity.summary is not None:
        if len(activity.summary.requested_aliases) != 1:
            reasons.append("requested_alias_count_not_one")
        if activity.summary.billing_uncertain:
            reasons.append("summary_billing_uncertain")
    model_ids = {
        model_id
        for artifact in artifacts.values()
        for model_id in artifact.response_model_ids_raw
        if model_id.strip()
    }
    if len(model_ids) != 1 or len(activity.observed_response_model_ids_raw) != 1:
        reasons.append("response_model_id_count_not_one")
    if any(
        not artifact.usage.complete
        or artifact.actual_cost_micro_cny is None
        or artifact.billing_uncertain
        or artifact.diagnostic_only
        for artifact in artifacts.values()
    ):
        reasons.append("artifact_accounting_incomplete")
    calibration_report = (
        Path(report_input.repository_root)
        / "reports"
        / "calibration"
        / "calibration_report.json"
    )
    if calibration_report.is_file():
        payload = json.loads(calibration_report.read_text(encoding="utf-8"))
        if len(payload.get("candidates", [])) != 54 or payload.get("selected") is None:
            reasons.append("calibration_replay_incomplete")
    else:
        reasons.append("calibration_report_missing")
    return PublicationGateResult(publishable=not reasons, reasons=reasons)


def _official_summary(
    activity_dir: Path,
    strategies: dict[str, dict[str, object]],
) -> dict[str, object]:
    path = activity_dir / "official" / "summary.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    adaptive = strategies[Strategy.ADAPTIVE.value]
    unavailable = {
        "available": False,
        "reason": "pinned evaluator artifacts not generated",
        "completed_count": adaptive["official_completed_subset_size"],
        "full_count": adaptive["manifest_count"],
        "completion_rate": adaptive["completion_rate"],
    }
    return {
        "shared_task_2024": unavailable,
        "paper_2023_secondary": dict(unavailable),
    }


def _representative_sources(
    artifacts: dict[str, RunArtifact], paths: dict[str, Path]
) -> tuple[Path, ...]:
    categories = {
        "single": lambda artifact: (
            artifact.phase == "dev"
            and artifact.result.status is ResultStatus.COMPLETED
            and artifact.result.initial_route == "single"
            and not artifact.result.escalated
        ),
        "multi": lambda artifact: (
            artifact.phase == "dev"
            and artifact.result.status is ResultStatus.COMPLETED
            and artifact.result.initial_route == "multi"
        ),
        "escalation": lambda artifact: artifact.phase == "dev" and artifact.result.escalated,
        "failure": lambda artifact: (
            artifact.phase == "dev"
            and artifact.result.status in {ResultStatus.PARTIAL, ResultStatus.FAILED}
        ),
    }
    selected: list[Path] = []
    seen: set[Path] = set()
    for predicate in categories.values():
        matches = sorted(
            (artifact for artifact in artifacts.values() if predicate(artifact)),
            key=lambda artifact: artifact.run_id,
        )
        if matches:
            path = paths[matches[0].run_id]
            if path not in seen:
                seen.add(path)
                selected.append(path)
    return tuple(selected)


def _render_markdown(summary: dict[str, object]) -> str:
    strategies = summary["strategies"]
    lines = [
        "# EvidenceRoute Gate A Evaluation",
        "",
        "## Scope",
        "",
        f"{summary['benchmark_name']}. This is a frozen balanced subset evaluation and must not "
        "be generalized beyond this cohort.",
        "",
        "## Reproducibility",
        "",
        f"- Model identity: {summary['reproducibility']['model_identity']}",
        f"- Activity status: {summary['activity_status']}",
        "",
        "## Full-Manifest Results",
        "",
        "| Strategy | Macro-F1 | Accuracy | Completion |",
        "|---|---:|---:|---:|",
    ]
    for name in ("always_single", "always_multi", "adaptive"):
        values = strategies[name]
        lines.append(
            f"| {name} | {values['full_manifest_macro_f1']:.3f} | "
            f"{values['full_manifest_accuracy']:.3f} | {values['completion_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Completed-Only Official Results",
            "",
            "Official evaluator numbers are completed-only and always retain the full-manifest "
            "completion denominator.",
            "",
            "## Cost/Latency",
            "",
            f"Adaptive tokens: {strategies['adaptive']['total_tokens']}; token reduction versus "
            f"always_multi: {strategies['adaptive']['token_reduction_pct']:.1f}%.",
            "",
            "## Routing",
            "",
            f"Adaptive route-not-reached: "
            f"{strategies['adaptive']['route_not_reached_count']}.",
            "",
            "## Stability",
            "",
            f"Three-run verdict consistency: {summary['stability']['rate']:.1%} "
            f"({summary['stability']['consistent']}/{summary['stability']['total']}).",
            "",
            "## Failure Analysis",
            "",
            "Adaptive status counts: "
            f"{json.dumps(strategies['adaptive']['status_counts'], sort_keys=True)}.",
            "",
            "## Limitations",
            "",
            "The 3 percentage-point quality tolerance and 85% stability threshold are engineering "
            "gates, not statistical non-inferiority claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def _resume_snippet(summary: dict[str, object]) -> str:
    adaptive = summary["strategies"]["adaptive"]
    return (
        "- EvidenceRoute（事实核查 Agent）：在同一冻结证据与模型配置下实现 single/multi/"
        "adaptive 路由；在 AVeriTeC dev balanced subset (n=80) 上，adaptive 的 full-manifest "
        f"macro-F1 为 {adaptive['full_manifest_macro_f1']:.3f}，相对 always_multi 降低 "
        f"{adaptive['token_reduction_pct']:.1f}% Token，完成率 {adaptive['completion_rate']:.1%}；"
        "使用 OpenAI-compatible provider，模型 ID 由中转服务报告且 identity unverified。\n"
    )


def _update_readme(path: Path, summary: dict[str, object]) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(_RESULTS_START) != 1 or text.count(_RESULTS_END) != 1:
        raise ValueError("README must contain exactly one EvidenceRoute result marker pair")
    start, remainder = text.split(_RESULTS_START, 1)
    _old, end = remainder.split(_RESULTS_END, 1)
    adaptive = summary["strategies"]["adaptive"]
    generated = (
        f"\n- Benchmark: {summary['benchmark_name']}\n"
        f"- Adaptive full-manifest macro-F1: {adaptive['full_manifest_macro_f1']:.3f}\n"
        f"- Adaptive completion rate: {adaptive['completion_rate']:.1%}\n"
        f"- Token reduction vs always_multi: {adaptive['token_reduction_pct']:.1f}%\n"
    )
    _write_text(path, start + _RESULTS_START + generated + _RESULTS_END + end)


def build_report_bundle(
    report_input: ReportInput,
    *,
    publish: bool = False,
) -> ReportBundle:
    activity, plan, state, plan_path, state_path = _load_models(report_input)
    artifacts, artifact_paths = _load_artifacts(Path(report_input.activity_dir), plan, state)
    gold_manifest = load_gold_manifest(
        report_input.gold_manifest,
        allowed_root=Path(report_input.gold_manifest).parent,
    )
    gold = {item.claim_id: item.label for item in gold_manifest.items}
    strategy_payloads: dict[str, dict[str, object]] = {}
    predictions: dict[str, dict[str, str]] = {}
    artifact_values = list(artifacts.values())
    for strategy in Strategy:
        payload, scored = _strategy_summary(strategy, gold, artifact_values)
        strategy_payloads[strategy.value] = payload
        predictions[strategy.value] = scored
    multi = strategy_payloads[Strategy.ALWAYS_MULTI.value]
    adaptive = strategy_payloads[Strategy.ADAPTIVE.value]
    multi_tokens = int(multi["total_tokens"])
    adaptive_tokens = int(adaptive["total_tokens"])
    adaptive["token_reduction_pct"] = (
        100.0 * (multi_tokens - adaptive_tokens) / multi_tokens if multi_tokens else 0.0
    )
    multi_cost = multi["actual_cost_micro_cny"]
    adaptive_cost = adaptive["actual_cost_micro_cny"]
    if isinstance(multi_cost, int) and isinstance(adaptive_cost, int) and multi_cost:
        adaptive["cost_reduction_pct"] = 100.0 * (multi_cost - adaptive_cost) / multi_cost

    gold_values = [gold[claim_id].value for claim_id in gold]
    paired: dict[str, object] = {}
    for fixed in (Strategy.ALWAYS_SINGLE.value, Strategy.ALWAYS_MULTI.value):
        paired[fixed] = paired_bootstrap_difference(
            gold_values,
            [predictions[Strategy.ADAPTIVE.value][claim_id] for claim_id in gold],
            [predictions[fixed][claim_id] for claim_id in gold],
            samples=10_000,
            seed=plan.freeze.seed,
        ).model_dump(mode="json")

    stability_runs: dict[str, list[Any]] = {}
    for link in plan.stability_repeat_zero_links:
        baseline = artifacts.get(link.dev_adaptive_run_id)
        repeats = sorted(
            (
                artifact
                for artifact in artifact_values
                if artifact.phase == "stability" and artifact.claim_id == link.claim_id
            ),
            key=lambda artifact: artifact.repeat,
        )
        if baseline is not None and len(repeats) == 2:
            stability_runs[link.claim_id] = [
                baseline.result,
                repeats[0].result,
                repeats[1].result,
            ]
    stability = (
        stability_consistency(stability_runs).model_dump(mode="json")
        if stability_runs
        else {"total": 0, "consistent": 0, "rate": 0.0, "wilson_low": 0.0, "wilson_high": 0.0}
    )
    gate = _publication_gate(
        report_input,
        activity,
        plan,
        state,
        plan_path,
        state_path,
        artifacts,
    )
    summary: dict[str, object] = {
        "benchmark_name": BENCHMARK_NAME,
        "publishable": gate.publishable,
        "activity_status": activity.status.value,
        "phase_statuses": {
            "calibration": activity.calibration_status.value,
            "dev": activity.dev_status.value,
            "stability": activity.stability_status.value,
        },
        "strategies": strategy_payloads,
        "paired_bootstrap": paired,
        "stability": stability,
        "official": _official_summary(Path(report_input.activity_dir), strategy_payloads),
        "reproducibility": {
            "activity_id": activity.activity_id,
            "campaign_id": plan.campaign_id,
            "campaign_fingerprint": plan.campaign_fingerprint,
            "manifest_freeze_git_sha": plan.freeze.manifest_freeze_git_sha,
            "dev_protocol_git_sha": plan.freeze.dev_protocol_git_sha,
            "requested_alias": plan.freeze.requested_alias,
            "relay_reported_model_ids_raw": activity.observed_response_model_ids_raw,
            "model_identity": IDENTITY_DESCRIPTION,
        },
        "publication_gate": gate.model_dump(mode="json"),
        "limitations": [
            "Balanced frozen subset; not the full AVeriTeC benchmark or leaderboard.",
            "Relay-reported model identity is self-reported and unverified.",
            "Engineering gates are not statistical non-inferiority claims.",
        ],
    }
    if publish and not gate.publishable:
        raise PublicationBlocked("; ".join(gate.reasons))
    markdown = _render_markdown(summary)
    resume = _resume_snippet(summary) if gate.publishable else None
    sources = _representative_sources(artifacts, artifact_paths)
    return ReportBundle(
        summary=summary,
        markdown=markdown,
        resume_snippet=resume,
        representative_trace_sources=sources,
        publication_gate=gate,
    )


__all__ = [
    "BENCHMARK_NAME",
    "PublicationBlocked",
    "PublicationGateResult",
    "ReportBundle",
    "ReportInput",
    "build_report_bundle",
]
