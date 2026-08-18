"""Full-manifest metrics and seeded uncertainty estimates."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import Field
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from evidence_route.contracts import ResultStatus, StrictModel, Usage, Verdict, VerificationResult

LABELS = [verdict.value for verdict in Verdict]
NO_PREDICTION = "__NO_PREDICTION__"


class ClassMetrics(StrictModel):
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    support: int = Field(ge=0)


class ManifestMetrics(StrictModel):
    sample_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    completion_rate: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    per_class: dict[str, ClassMetrics]
    confusion_labels: list[str]
    confusion_matrix: list[list[int]]
    scored_predictions: dict[str, str]


class ConditionalMetrics(StrictModel):
    manifest_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    completion_rate: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)


class BootstrapInterval(StrictModel):
    estimate: float
    low: float
    high: float
    samples: int = Field(gt=0)
    seed: int


class OperationSummary(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    mean_tokens_per_claim: float = Field(ge=0)
    exact_cost_micro_cny: int | None = Field(default=None, ge=0)
    latency_p50_ms: float | None = Field(default=None, ge=0)
    latency_p95_ms: float | None = Field(default=None, ge=0)
    cache_hit_count: int = Field(ge=0)
    route_counts: dict[str, int]
    escalation_rate: float = Field(ge=0, le=1)
    llm_router_rate: float = Field(ge=0, le=1)
    citation_validity_rate: float | None = Field(default=None, ge=0, le=1)
    error_reasons: dict[str, int]


class StabilitySummary(StrictModel):
    total: int = Field(gt=0)
    consistent: int = Field(ge=0)
    rate: float = Field(ge=0, le=1)
    wilson_low: float = Field(ge=0, le=1)
    wilson_high: float = Field(ge=0, le=1)


def prediction_for_scoring(result: VerificationResult | None) -> str:
    if result is None or result.status is not ResultStatus.COMPLETED or result.verdict is None:
        return NO_PREDICTION
    return result.verdict.value


def _metric_arrays(
    gold: Mapping[str, Verdict], results: Mapping[str, VerificationResult]
) -> tuple[list[str], list[str], dict[str, str]]:
    gold_values: list[str] = []
    predicted: list[str] = []
    scored: dict[str, str] = {}
    for claim_id, verdict in gold.items():
        prediction = prediction_for_scoring(results.get(claim_id))
        gold_values.append(verdict.value if isinstance(verdict, Verdict) else str(verdict))
        predicted.append(prediction)
        scored[claim_id] = prediction
    return gold_values, predicted, scored


def score_full_manifest(
    gold: Mapping[str, Verdict], results: Mapping[str, VerificationResult]
) -> ManifestMetrics:
    if not gold:
        raise ValueError("full-manifest scoring requires non-empty gold")
    gold_values, predicted, scored = _metric_arrays(gold, results)
    completed = sum(
        1
        for claim_id in gold
        if claim_id in results and results[claim_id].status is ResultStatus.COMPLETED
    )
    partial = sum(
        1
        for claim_id in gold
        if claim_id in results and results[claim_id].status is ResultStatus.PARTIAL
    )
    failed = sum(
        1
        for claim_id in gold
        if claim_id in results and results[claim_id].status is ResultStatus.FAILED
    )
    missing = len(gold) - completed - partial - failed
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_values, predicted, labels=LABELS, zero_division=0
    )
    labels_with_missing = LABELS + [NO_PREDICTION]
    matrix = confusion_matrix(gold_values, predicted, labels=labels_with_missing)
    return ManifestMetrics(
        sample_count=len(gold),
        completed_count=completed,
        partial_count=partial,
        failed_count=failed,
        missing_count=missing,
        completion_rate=completed / len(gold),
        accuracy=float(accuracy_score(gold_values, predicted)),
        macro_f1=float(
            f1_score(gold_values, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        per_class={
            label: ClassMetrics(
                precision=float(precision[index]),
                recall=float(recall[index]),
                f1=float(f1[index]),
                support=int(support[index]),
            )
            for index, label in enumerate(LABELS)
        },
        confusion_labels=labels_with_missing,
        confusion_matrix=matrix.astype(int).tolist(),
        scored_predictions=scored,
    )


def score_completed_conditionally(
    gold: Mapping[str, Verdict], results: Mapping[str, VerificationResult]
) -> ConditionalMetrics:
    completed = {
        claim_id: result
        for claim_id, result in results.items()
        if (
            claim_id in gold
            and result.status is ResultStatus.COMPLETED
            and result.verdict is not None
        )
    }
    if not completed:
        return ConditionalMetrics(
            manifest_count=len(gold),
            sample_count=0,
            completion_rate=0.0,
            accuracy=0.0,
            macro_f1=0.0,
        )
    gold_values = [gold[claim_id].value for claim_id in completed]
    predictions = [completed[claim_id].verdict.value for claim_id in completed]
    return ConditionalMetrics(
        manifest_count=len(gold),
        sample_count=len(completed),
        completion_rate=len(completed) / len(gold),
        accuracy=float(accuracy_score(gold_values, predictions)),
        macro_f1=float(
            f1_score(gold_values, predictions, labels=LABELS, average="macro", zero_division=0)
        ),
    )


def _as_usage(value: Any) -> Usage | None:
    if isinstance(value, Usage):
        return value
    if isinstance(value, Mapping):
        try:
            return Usage.model_validate(value)
        except Exception:
            return None
    return None


def summarize_operations(operations: Sequence[Mapping[str, Any]]) -> OperationSummary:
    if not operations:
        return OperationSummary(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            mean_tokens_per_claim=0.0,
            route_counts={},
            escalation_rate=0.0,
            llm_router_rate=0.0,
            cache_hit_count=0,
            error_reasons={},
        )
    usages = [_as_usage(item.get("usage")) for item in operations]
    input_tokens = sum(usage.input_tokens for usage in usages if usage is not None)
    output_tokens = sum(usage.output_tokens for usage in usages if usage is not None)
    total_tokens = sum(usage.total_tokens for usage in usages if usage is not None)
    complete = bool(usages) and all(usage is not None and usage.complete for usage in usages)
    costs = [item.get("estimated_cost_micro_cny") for item in operations]
    exact_cost = (
        sum(int(cost) for cost in costs)
        if complete and all(isinstance(cost, int) for cost in costs)
        else None
    )
    latencies = [
        float(item["fresh_end_to_end_ms"])
        for item in operations
        if item.get("fresh_end_to_end_ms") is not None
    ]
    routes = Counter(str(item["route"]) for item in operations if item.get("route") is not None)
    escalated = sum(bool(item.get("escalated", False)) for item in operations)
    llm_router = sum(item.get("route_source") == "llm" for item in operations)
    cache_hits = sum(
        int(item.get("cache_hit_count", item.get("cache_hits", 0)) or 0)
        for item in operations
    )
    errors: Counter[str] = Counter()
    for item in operations:
        raw_errors = item.get("errors", [])
        if isinstance(raw_errors, str):
            raw_errors = [raw_errors]
        if isinstance(raw_errors, Sequence):
            errors.update(str(error) for error in raw_errors)
    citation_total = citation_valid = 0
    for item in operations:
        if "citation_valid" in item:
            citation_total += 1
            citation_valid += bool(item["citation_valid"])
    return OperationSummary(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        mean_tokens_per_claim=total_tokens / len(operations),
        exact_cost_micro_cny=exact_cost,
        latency_p50_ms=float(np.quantile(latencies, 0.5, method="linear")) if latencies else None,
        latency_p95_ms=float(np.quantile(latencies, 0.95, method="linear")) if latencies else None,
        cache_hit_count=cache_hits,
        route_counts=dict(routes),
        escalation_rate=escalated / len(operations),
        llm_router_rate=llm_router / len(operations),
        citation_validity_rate=citation_valid / citation_total if citation_total else None,
        error_reasons=dict(errors),
    )


def stability_consistency(
    runs: Mapping[str, Sequence[VerificationResult]],
) -> StabilitySummary:
    if not runs:
        raise ValueError("stability scoring requires at least one claim")
    consistent = 0
    for claim_id, repetitions in runs.items():
        if len(repetitions) != 3:
            raise ValueError(f"stability claim requires exactly three runs: {claim_id}")
        verdicts = [
            result.verdict
            for result in repetitions
            if result.status is ResultStatus.COMPLETED and result.verdict is not None
        ]
        if len(verdicts) == 3 and len(set(verdicts)) == 1:
            consistent += 1
    low, high = wilson_interval(consistent, len(runs))
    return StabilitySummary(
        total=len(runs),
        consistent=consistent,
        rate=consistent / len(runs),
        wilson_low=low,
        wilson_high=high,
    )


def paired_bootstrap_difference(
    gold: Sequence[str],
    predictions_a: Sequence[str],
    predictions_b: Sequence[str],
    *,
    samples: int = 10_000,
    seed: int = 20260817,
) -> BootstrapInterval:
    if not (len(gold) == len(predictions_a) == len(predictions_b)) or not gold:
        raise ValueError("paired bootstrap requires equal non-empty arrays")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(gold), size=(samples, len(gold)))
    gold_indices = np.asarray([LABELS.index(value) for value in gold], dtype=np.int8)
    a_indices = np.asarray(
        [LABELS.index(value) if value in LABELS else -1 for value in predictions_a],
        dtype=np.int8,
    )
    b_indices = np.asarray(
        [LABELS.index(value) if value in LABELS else -1 for value in predictions_b],
        dtype=np.int8,
    )

    def sampled_macro_f1(predictions: np.ndarray) -> np.ndarray:
        sampled_gold = gold_indices[draws]
        sampled_predictions = predictions[draws]
        scores: list[np.ndarray] = []
        for label_index in range(len(LABELS)):
            true_positive = np.sum(
                (sampled_gold == label_index) & (sampled_predictions == label_index), axis=1
            )
            false_positive = np.sum(
                (sampled_gold != label_index) & (sampled_predictions == label_index), axis=1
            )
            false_negative = np.sum(
                (sampled_gold == label_index) & (sampled_predictions != label_index), axis=1
            )
            denominator = 2 * true_positive + false_positive + false_negative
            scores.append(
                np.divide(
                    2 * true_positive,
                    denominator,
                    out=np.zeros(samples, dtype=float),
                    where=denominator != 0,
                )
            )
        return np.mean(np.stack(scores, axis=1), axis=1)

    values = sampled_macro_f1(a_indices) - sampled_macro_f1(b_indices)
    estimate = f1_score(
        gold, predictions_a, labels=LABELS, average="macro", zero_division=0
    ) - f1_score(gold, predictions_b, labels=LABELS, average="macro", zero_division=0)
    return BootstrapInterval(
        estimate=float(estimate),
        low=float(np.quantile(values, 0.025, method="linear")),
        high=float(np.quantile(values, 0.975, method="linear")),
        samples=samples,
        seed=seed,
    )


def wilson_interval(successes: int, total: int, *, z: float | None = None) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires total > 0")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    z_value = statistics.NormalDist().inv_cdf(0.975) if z is None else float(z)
    proportion = successes / total
    denominator = 1 + z_value * z_value / total
    centre = (proportion + z_value * z_value / (2 * total)) / denominator
    radius = z_value * math.sqrt(
        proportion * (1 - proportion) / total + z_value * z_value / (4 * total * total)
    ) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


__all__ = [
    "BootstrapInterval",
    "ClassMetrics",
    "ConditionalMetrics",
    "LABELS",
    "ManifestMetrics",
    "NO_PREDICTION",
    "OperationSummary",
    "StabilitySummary",
    "paired_bootstrap_difference",
    "prediction_for_scoring",
    "score_completed_conditionally",
    "score_full_manifest",
    "stability_consistency",
    "summarize_operations",
    "wilson_interval",
]
