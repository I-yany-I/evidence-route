"""Adapters for the pinned official AVeriTeC evaluator formats."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evidence_route.contracts import ResultStatus, VerificationResult
from evidence_route.evaluation.runtime_manifest import RuntimeClaim
from evidence_route.evaluation.scorer_manifest import GoldClaim


@dataclass(frozen=True)
class CompletedTriple:
    """One manifest-aligned completed result eligible for official evaluation."""

    runtime: RuntimeClaim
    gold: GoldClaim
    result: VerificationResult


@dataclass(frozen=True)
class CompletedSelection:
    """The completed subset and the omitted rows from a full manifest."""

    triples: list[CompletedTriple]
    omitted_claim_ids: list[str]
    full_count: int
    completed_count: int
    completion_rate: float

    @property
    def total_count(self) -> int:
        """Alias used by report consumers that call the manifest size ``total_count``."""

        return self.full_count


class OfficialEvaluatorError(RuntimeError):
    """A pinned evaluator could not be executed or returned a non-zero status."""


def _source_entries(source_spec: object) -> Iterable[tuple[str, str]]:
    if not isinstance(source_spec, dict):
        raise ValueError("evaluator source spec must be a JSON object")
    files = source_spec.get("files", source_spec.get("sources"))
    if isinstance(files, dict):
        for relative_path, metadata in files.items():
            if not isinstance(relative_path, str) or not isinstance(metadata, dict):
                raise ValueError("evaluator source entries must map paths to metadata")
            digest = metadata.get("sha256")
            if not isinstance(digest, str):
                raise ValueError(f"evaluator source lacks SHA-256: {relative_path}")
            yield relative_path, digest
        return
    if isinstance(files, list):
        for metadata in files:
            if not isinstance(metadata, dict):
                raise ValueError("evaluator source entries must be objects")
            relative_path = metadata.get("path")
            digest = metadata.get("sha256")
            if not isinstance(relative_path, str) or not isinstance(digest, str):
                raise ValueError("evaluator source entry requires path and SHA-256")
            yield relative_path, digest
        return
    raise ValueError("evaluator source spec requires files or sources")


def verify_evaluator_sources(source_spec_path: Path | str) -> dict[str, str]:
    """Raise when a pinned evaluator file does not match its declared SHA-256."""

    spec_path = Path(source_spec_path)
    try:
        source_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evaluator source spec: {spec_path}") from exc

    if not isinstance(source_spec, dict):
        raise ValueError("evaluator source spec must be a JSON object")
    if source_spec.get("license") != "CC BY-NC 4.0":
        raise ValueError("evaluator source spec must declare the AVeriTeC license")
    if not isinstance(source_spec.get("citation"), str) or not source_spec["citation"].strip():
        raise ValueError("evaluator source spec must declare a citation")

    root = spec_path.parent.resolve()
    found = False
    verified: dict[str, str] = {}
    for relative_path, expected_digest in _source_entries(source_spec):
        is_lowercase_digest = all(char in "0123456789abcdef" for char in expected_digest)
        if len(expected_digest) != 64 or not is_lowercase_digest:
            raise ValueError(f"invalid evaluator SHA-256: {relative_path}")
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"evaluator source path escapes root: {relative_path}") from exc
        if not candidate.is_file():
            raise ValueError(f"evaluator source is missing: {relative_path}")
        actual_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(f"evaluator source SHA-256 mismatch: {relative_path}")
        verified[relative_path] = actual_digest
        found = True
    if not found:
        raise ValueError("evaluator source spec contains no files")
    return verified


def _unpack_aligned(aligned: object) -> tuple[RuntimeClaim, GoldClaim]:
    """Accept Task 14's tuple API and the named object form used by later callers."""

    if isinstance(aligned, tuple) and len(aligned) == 2:
        runtime, gold = aligned
    else:
        runtime = getattr(aligned, "runtime", None)
        gold = getattr(aligned, "gold", None)
    if not isinstance(runtime, RuntimeClaim) or not isinstance(gold, GoldClaim):
        raise ValueError("each aligned claim must contain a runtime and gold claim")
    if runtime.claim_id != gold.claim_id:
        raise ValueError("aligned runtime and gold claim IDs differ")
    return runtime, gold


def require_completed(result: VerificationResult) -> None:
    """Defend low-level official adapters from malformed or non-final rows."""

    if result.status != ResultStatus.COMPLETED:
        raise ValueError("official evaluator adapters require completed results")
    if result.verdict is None or result.confidence is None:
        raise ValueError("completed results require verdict and confidence")


def select_completed_triples(
    aligned_claims: Sequence[tuple[RuntimeClaim, GoldClaim]] | Sequence[object],
    results: Sequence[VerificationResult] | Mapping[str, VerificationResult],
) -> CompletedSelection:
    """Select completed results without changing the verified manifest ordering.

    ``results`` may be a full ordered sequence, an ordered subset (missing rows are retained in
    ``omitted_claim_ids``), or a mapping keyed by claim ID.  An ordered sequence is never silently
    re-sorted: a supplied row with an out-of-order ID is an integrity error.
    """

    aligned = [_unpack_aligned(item) for item in aligned_claims]
    manifest_ids = [runtime.claim_id for runtime, _gold in aligned]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("aligned claims contain duplicate claim IDs")
    result_by_id: dict[str, VerificationResult] = {}
    if isinstance(results, Mapping):
        for key, result in results.items():
            if not isinstance(key, str) or not isinstance(result, VerificationResult):
                raise ValueError(
                    "result mappings must contain string IDs and VerificationResult values"
                )
            if key != result.claim_id:
                raise ValueError("result mapping key does not match result claim ID")
            if key in result_by_id:
                raise ValueError("duplicate result claim ID")
            result_by_id[key] = result
    else:
        last_index = -1
        for result in results:
            if not isinstance(result, VerificationResult):
                raise ValueError("results must contain VerificationResult values")
            if result.claim_id not in manifest_ids:
                raise ValueError(f"result claim ID is absent from manifest: {result.claim_id}")
            index = manifest_ids.index(result.claim_id)
            if index <= last_index:
                raise ValueError("results must match manifest order and contain no duplicates")
            last_index = index
            result_by_id[result.claim_id] = result

    unknown = sorted(set(result_by_id) - set(manifest_ids))
    if unknown:
        raise ValueError(f"results contain unknown claim IDs: {unknown}")
    triples: list[CompletedTriple] = []
    omitted_claim_ids: list[str] = []
    for runtime, gold in aligned:
        result = result_by_id.get(runtime.claim_id)
        if result is None:
            omitted_claim_ids.append(runtime.claim_id)
        elif result.status == ResultStatus.COMPLETED:
            require_completed(result)
            triples.append(CompletedTriple(runtime=runtime, gold=gold, result=result))
        else:
            omitted_claim_ids.append(runtime.claim_id)

    full_count = len(aligned)
    completed_count = len(triples)
    return CompletedSelection(
        triples=triples,
        omitted_claim_ids=omitted_claim_ids,
        full_count=full_count,
        completed_count=completed_count,
        completion_rate=completed_count / full_count if full_count else 0.0,
    )


def _gold_item(gold: GoldClaim) -> dict[str, object]:
    """Serialize only scorer-side fields accepted by the pinned evaluators."""

    return {
        "claim": gold.claim,
        "label": gold.label.value,
        "questions": gold.questions,
        "justification": gold.justification,
        "claim_types": gold.claim_types,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _coerce_number(value: str) -> float | str:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return value


def _parse_evaluator_stdout(stdout: str) -> dict[str, object]:
    """Parse the stable ``label: value`` lines without depending on evaluator internals."""

    sections: dict[str, dict[str, object]] = {}
    current = "metrics"
    sections[current] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"=", "-"}:
            continue
        if stripped.endswith(":") and not re.search(r":\s*[-+]?\d", stripped):
            heading = stripped.rstrip(":").lower().replace(" ", "_")
            current = heading
            sections.setdefault(current, {})
            continue
        match = re.match(r"^\*?\s*(.+?):\s*(.+?)\s*$", stripped)
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        sections.setdefault(current, {})[key] = _coerce_number(match.group(2))
    return sections


def _selection_values(
    selection: CompletedSelection | Sequence[CompletedTriple],
) -> tuple[list[CompletedTriple], int, int, float]:
    if isinstance(selection, CompletedSelection):
        return (
            selection.triples,
            selection.full_count,
            selection.completed_count,
            selection.completion_rate,
        )
    triples = list(selection)
    return triples, len(triples), len(triples), 1.0 if triples else 0.0


def _run_pinned_script(
    *,
    name: str,
    script: Path,
    args: list[str],
    cwd: Path,
    output_dir: Path,
    env: dict[str, str],
    timeout_s: float,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [sys.executable, str(script), *args],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OfficialEvaluatorError(f"{name} evaluator did not complete: {exc}") from exc

    stdout_path = output_dir / f"{name}.stdout.txt"
    stderr_path = output_dir / f"{name}.stderr.txt"
    metrics_path = output_dir / f"{name}.metrics.json"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    parsed = _parse_evaluator_stdout(completed.stdout)
    _write_json(metrics_path, parsed)
    if completed.returncode != 0:
        raise OfficialEvaluatorError(
            f"{name} evaluator failed with exit code {completed.returncode}; see {stderr_path}"
        )
    return {
        "available": True,
        "returncode": completed.returncode,
        "metrics": parsed,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "metrics_path": str(metrics_path),
    }


def run_official_evaluators(
    selection: CompletedSelection | Sequence[CompletedTriple],
    output_dir: Path | str | None = None,
    *,
    source_spec: Path | str | None = None,
    shared_task_script: Path | str | None = None,
    paper_script: Path | str | None = None,
    nltk_data: Path | str | None = None,
    timeout_s: float = 300.0,
) -> dict[str, object]:
    """Run both pinned evaluators in child processes on the completed cohort only.

    The returned object is JSON-serializable.  It includes artifact paths and completion metadata;
    official scores remain explicitly completed-only and are never substituted for full-manifest
    metrics.
    """

    repo_root = Path(__file__).resolve().parents[3]
    source_spec_path = (
        Path(source_spec)
        if source_spec
        else repo_root / "third_party/averitec/SOURCES.json"
    )
    verify_evaluator_sources(source_spec_path)
    triples, full_count, completed_count, completion_rate = _selection_values(selection)
    if output_dir is None:
        output_path = Path(tempfile.mkdtemp(prefix="evidence-route-official-"))
    else:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    base = {
        "available": bool(triples),
        "completed_count": completed_count,
        "full_count": full_count,
        "completion_rate": completion_rate,
        "artifact_dir": str(output_path),
    }
    if not triples:
        unavailable = {
            "available": False,
            "reason": "no_completed_outputs",
            "completed_count": 0,
            "completion_rate": completion_rate,
        }
        return {
            **base,
            "shared_task_2024": unavailable,
            "paper_2023_secondary": unavailable.copy(),
        }

    shared_predictions = build_shared_task_predictions(triples)
    paper_predictions = build_paper_predictions(triples)
    references = [_gold_item(triple.gold) for triple in triples]
    shared_prediction_path = output_path / "shared_task_predictions.json"
    shared_reference_path = output_path / "shared_task_references.json"
    paper_prediction_path = output_path / "paper_predictions.json"
    paper_reference_path = output_path / "paper_references.json"
    _write_json(shared_prediction_path, shared_predictions)
    _write_json(shared_reference_path, references)
    _write_json(paper_prediction_path, paper_predictions)
    _write_json(paper_reference_path, references)

    shared_script = (
        Path(shared_task_script)
        if shared_task_script
        else repo_root / "third_party/averitec/shared_task/evaluate_veracity.py"
    )
    paper_eval = (
        Path(paper_script)
        if paper_script
        else repo_root / "third_party/averitec/paper/eval.py"
    )
    shared_script = shared_script.resolve()
    paper_eval = paper_eval.resolve()
    env = os.environ.copy()
    if nltk_data is not None:
        env["NLTK_DATA"] = str(Path(nltk_data).resolve())
    shared_result = _run_pinned_script(
        name="shared_task_2024",
        script=shared_script,
        args=["-i", str(shared_prediction_path), "--label_file", str(shared_reference_path)],
        cwd=shared_script.parent,
        output_dir=output_path,
        env=env,
        timeout_s=timeout_s,
    )
    paper_result = _run_pinned_script(
        name="paper_2023_secondary",
        script=paper_eval,
        args=[
            "--predictions",
            str(paper_prediction_path),
            "--references",
            str(paper_reference_path),
        ],
        cwd=paper_eval.parent,
        output_dir=output_path,
        env=env,
        timeout_s=timeout_s,
    )
    return {
        **base,
        "shared_task_2024": shared_result,
        "paper_2023_secondary": paper_result,
        "prediction_paths": {
            "shared_task": str(shared_prediction_path),
            "paper": str(paper_prediction_path),
        },
        "reference_paths": {
            "shared_task": str(shared_reference_path),
            "paper": str(paper_reference_path),
        },
    }


def build_shared_task_predictions(triples: Sequence[CompletedTriple]) -> list[dict[str, object]]:
    """Build shared-task evaluator rows with only its documented fields."""

    predictions: list[dict[str, object]] = []
    for triple in triples:
        claim, result = triple.runtime, triple.result
        require_completed(result)
        predictions.append(
            {
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
        )
    return predictions


def build_paper_predictions(triples: Sequence[CompletedTriple]) -> list[dict[str, object]]:
    """Build paper-era evaluator rows with only its documented fields."""

    predictions: list[dict[str, object]] = []
    for triple in triples:
        result = triple.result
        require_completed(result)
        predictions.append(
            {
                "label": result.verdict.value,
                "questions": [
                    {
                        "question": citation.question,
                        "answers": [
                            {
                                "answer": citation.answer,
                                "answer_type": "Abstractive",
                                "source_url": str(citation.source_url),
                            }
                        ],
                    }
                    for citation in result.citations[:10]
                ],
                "justification": result.rationale,
            }
        )
    return predictions


__all__ = [
    "CompletedSelection",
    "CompletedTriple",
    "OfficialEvaluatorError",
    "build_paper_predictions",
    "build_shared_task_predictions",
    "require_completed",
    "run_official_evaluators",
    "select_completed_triples",
    "verify_evaluator_sources",
]
