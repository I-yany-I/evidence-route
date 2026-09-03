"""Complete, deterministic report fixture built from persisted campaign artifacts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from evidence_route.artifacts import atomic_write_json
from evidence_route.budget import PriceConfig
from evidence_route.contracts import (
    ClaimFeatures,
    ClaimUnit,
    ResultStatus,
    Strategy,
    Usage,
    Verdict,
    VerificationResult,
)
from evidence_route.evaluation.activity import (
    ActivityRecord,
    CalibrationArtifactLink,
    CallBounds,
    CampaignItemState,
    CampaignPlan,
    CampaignState,
    CampaignStatus,
    FreezeIdentity,
    LatencyBreakdown,
    RunArtifact,
    RunSummary,
    WorkStatus,
    artifact_fingerprint,
    campaign_fingerprint,
    summarize_run_artifacts,
)
from evidence_route.evaluation.calibration import (
    CalibrationItemStatus,
    CalibrationReplay,
    CandidateOutcome,
    build_calibration_plan,
    build_calibration_runtime_case,
    build_calibration_state,
    state_fingerprint,
)
from evidence_route.evaluation.lifecycle import persist_activity
from evidence_route.evaluation.reporting import ReportInput
from evidence_route.evaluation.runner import build_campaign_schedule
from evidence_route.llm import make_call_id


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _price_config_id() -> str:
    return PriceConfig(
        provider="fixture",
        currency="CNY",
        input_per_million=1.0,
        output_per_million=1.0,
        price_source="fixture",
        strict_evaluation=True,
    ).config_id


def _freeze(
    dev_runtime_manifest_sha256: str = "d" * 64,
    config_sha256: str = "2" * 64,
    calibration_runtime_manifest_sha256: str = "c" * 64,
    stability_runtime_manifest_sha256: str = "e" * 64,
    corpus_preparation_receipt_sha256: str = "f" * 64,
    prompt_bundle_sha256: str = "1" * 64,
    pricing_sha256: str = "3" * 64,
    requirements_lock_sha256: str = "5" * 64,
) -> FreezeIdentity:
    return FreezeIdentity(
        manifest_freeze_git_sha="a" * 40,
        dev_protocol_git_sha="b" * 40,
        calibration_runtime_manifest_sha256=calibration_runtime_manifest_sha256,
        dev_runtime_manifest_sha256=dev_runtime_manifest_sha256,
        stability_runtime_manifest_sha256=stability_runtime_manifest_sha256,
        corpus_preparation_receipt_sha256=corpus_preparation_receipt_sha256,
        prompt_bundle_sha256=prompt_bundle_sha256,
        config_sha256=config_sha256,
        pricing_sha256=pricing_sha256,
        endpoint_config_sha256="4" * 64,
        requirements_lock_sha256=requirements_lock_sha256,
        requested_alias="fixture-alias",
        seed=20260817,
    )


def _artifact(plan: CampaignPlan, item, index: int) -> RunArtifact:
    verdict = list(Verdict)[int(item.claim_id.split("-")[1]) % len(Verdict)]
    token_total = {
        Strategy.ALWAYS_SINGLE: 100,
        Strategy.ALWAYS_MULTI: 220,
        Strategy.ADAPTIVE: 130,
    }[item.strategy]
    if item.phase == "stability":
        token_total = 132 + item.repeat
    usage = Usage(
        input_tokens=token_total - 20,
        output_tokens=20,
        total_tokens=token_total,
        complete=True,
    )
    escalated = item.strategy is Strategy.ADAPTIVE and index % 11 == 0
    result = VerificationResult(
        claim_id=item.claim_id,
        status=ResultStatus.COMPLETED,
        verdict=verdict,
        confidence=0.82,
        rationale="fixture result",
        citations=[],
        available_evidence_ids=[],
        initial_route="multi" if item.strategy is Strategy.ALWAYS_MULTI else "single",
        escalated=escalated,
        usage=usage,
        estimated_cost_micro_cny=token_total,
        cost_currency="CNY",
        price_config_id=_price_config_id(),
        latency_ms=40 + index % 7,
    )
    call_id = make_call_id(item.run_id, "router", "root", 0)
    payload: dict[str, object] = {
        "schema_version": "1",
        "activity_id": plan.activity_id,
        "campaign_id": plan.campaign_id,
        "phase": item.phase,
        "run_id": item.run_id,
        "claim_id": item.claim_id,
        "strategy": item.strategy.value,
        "repeat": item.repeat,
        "result": result.model_dump(mode="json"),
        "call_ids": [call_id],
        "usage": usage.model_dump(mode="json"),
        "usage_source": "provider",
        "actual_cost_micro_cny": token_total,
        "known_actual_cost_micro_cny": token_total,
        "committed_cost_micro_cny": token_total,
        "cost_is_lower_bound": False,
        "fresh_call_count": 1,
        "cache_hit_count": index % 3,
        "requested_alias": plan.freeze.requested_alias,
        "response_model_ids_raw": ["relay-model-a"],
        "identity_verified": False,
        "billing_uncertain": False,
        "diagnostic_only": False,
        "latency": LatencyBreakdown(
            fresh_end_to_end_ms=40 + index % 7,
            model_active_ms=30,
            retry_ms=4,
            queue_ms=6,
            checkpoint_downtime_ms=0,
            total_elapsed_ms=40 + index % 7,
            interruption_count=0,
        ).model_dump(mode="json"),
        "artifact_sha256": "0" * 64,
    }
    payload["artifact_sha256"] = artifact_fingerprint(payload)
    return RunArtifact.model_validate(payload)


class ReportInputFactory:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.activity_dir = root / "activity"
        self.artifact_dir = self.activity_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True)
        self._write_current_freeze_inputs()
        self._write_manifests()
        self.calibrated_config = self.root / "configs" / "calibrated.yaml"
        self.calibrated_config.parent.mkdir(parents=True, exist_ok=True)
        self.calibrated_config.write_text(
            "routing:\n"
            "  clear_multi_clauses: 2\n"
            "hardening:\n"
            "  deterministic_decomposition: true\n",
            encoding="utf-8",
        )
        claims = [{"claim_id": f"dev-{index}"} for index in range(80)]
        schedule, links = build_campaign_schedule(
            claims,
            stability_runtime_claims=claims[:20],
            seed=20260817,
            campaign_id="dev-campaign",
        )
        plan_payload: dict[str, object] = {
            "schema_version": "1",
            "activity_id": "gate-a",
            "campaign_id": "dev-campaign",
            "freeze": _freeze(
                _sha(self.runtime_manifest),
                _sha(self.calibrated_config),
                _sha(self.calibration_runtime_manifest),
                _sha(self.stability_runtime_manifest),
                _sha(self.corpus_receipt),
                _sha(self.prompt_bundle),
                _sha(self.pricing),
                _sha(self.requirements_lock),
            ).model_dump(mode="json"),
            "cap_micro_cny": 350_000_000,
            "call_bounds": CallBounds(
                base_call_upper_bound=1544,
                repair_upper_bound=3088,
                fault_upper_bound=9264,
                base_cost_micro_cny=1_000_000,
                repair_cost_micro_cny=2_000_000,
                fault_cost_micro_cny=6_000_000,
                reserve_basis_points=2000,
                startup_required_micro_cny=1_200_000,
            ).model_dump(mode="json"),
            "schedule": [item.model_dump(mode="json") for item in schedule],
            "stability_repeat_zero_links": links,
        }
        plan_payload["campaign_fingerprint"] = campaign_fingerprint(plan_payload)
        plan = CampaignPlan.model_validate(plan_payload)
        artifacts = [_artifact(plan, item, index) for index, item in enumerate(schedule)]
        states: list[CampaignItemState] = []
        for item, artifact in zip(schedule, artifacts, strict=True):
            relpath = f"artifacts/{artifact.run_id}.json"
            atomic_write_json(
                self.activity_dir / relpath,
                artifact.model_dump(mode="json"),
            )
            states.append(
                CampaignItemState(
                    run_id=item.run_id,
                    status=WorkStatus.COMPLETED,
                    artifact_relpath=relpath,
                    artifact_sha256=artifact.artifact_sha256,
                )
            )
        adaptive_sha = {
            item.claim_id: artifact.artifact_sha256
            for item, artifact in zip(schedule, artifacts, strict=True)
            if item.phase == "dev"
            and item.strategy is Strategy.ADAPTIVE
            and item.claim_id in {link["claim_id"] for link in links}
        }
        summary = summarize_run_artifacts(artifacts)
        state = CampaignState(
            schema_version="1",
            activity_id=plan.activity_id,
            campaign_id=plan.campaign_id,
            status=CampaignStatus.COMPLETE,
            items=states,
            stability_repeat_zero_artifact_sha256s=adaptive_sha,
            observed_response_model_ids_raw=["relay-model-a"],
            billing_uncertain=False,
            summary=summary,
        )
        atomic_write_json(self.activity_dir / "plan.json", plan.model_dump(mode="json"))
        atomic_write_json(self.activity_dir / "campaign.json", state.model_dump(mode="json"))
        calibration_plan, calibration_state, calibration_cases = self._write_calibration()
        calibration_run_ids = [
            run_id
            for case in calibration_cases
            for run_id in (case.router_run_id, case.single_run_id, case.multi_run_id)
        ]
        calibration_call_ids = [call_id for case in calibration_cases for call_id in case.call_ids]
        combined_summary = RunSummary(
            run_ids=[*calibration_run_ids, *summary.run_ids],
            call_ids=[*calibration_call_ids, *summary.call_ids],
            usage=Usage(
                input_tokens=32 * 224 + summary.usage.input_tokens,
                output_tokens=32 * 56 + summary.usage.output_tokens,
                total_tokens=32 * 280 + summary.usage.total_tokens,
                complete=True,
            ),
            usage_sources=["provider"] * (len(calibration_call_ids) + len(summary.call_ids)),
            actual_cost_micro_cny=32 * 280 + int(summary.actual_cost_micro_cny or 0),
            known_actual_cost_micro_cny=32 * 280 + summary.known_actual_cost_micro_cny,
            committed_cost_micro_cny=32 * 280 + summary.committed_cost_micro_cny,
            cost_is_lower_bound=False,
            fresh_call_count=len(calibration_call_ids) + len(summary.call_ids),
            cache_hit_count=summary.cache_hit_count,
            transport_attempts=0,
            requested_aliases=["fixture-alias"],
            response_model_ids_raw=["relay-model-a"],
            identity_verified=False,
            billing_uncertain=False,
        )
        activity = ActivityRecord(
            schema_version="1",
            activity_id=plan.activity_id,
            status=CampaignStatus.COMPLETE,
            calibration_status=CampaignStatus.COMPLETE,
            dev_status=CampaignStatus.COMPLETE,
            stability_status=CampaignStatus.COMPLETE,
            calibration_plan_sha256=_sha(self.activity_dir / "calibration-plan.json"),
            calibration_state_sha256=_sha(self.activity_dir / "calibration-state.json"),
            calibration_artifacts=[
                CalibrationArtifactLink(
                    case_id=case.case_id,
                    artifact_sha256=case.artifact_sha256,
                )
                for case in calibration_cases
            ],
            freeze=plan.freeze,
            campaign_plan_sha256=_sha(self.activity_dir / "plan.json"),
            campaign_state_sha256=_sha(self.activity_dir / "campaign.json"),
            observed_response_model_ids_raw=["relay-model-a"],
            identity_verified=False,
            billing_uncertain=False,
            summary=combined_summary,
        )
        persist_activity(self.activity_dir / "activity.json", activity, fresh=True)
        self._write_calibration_replay(calibration_plan)
        self._write_run_store(artifacts, calibration_cases)
        self._write_official_cache(artifacts)

    def _write_manifests(self) -> None:
        runtime_items = []
        for index in range(80):
            claim = f"Claim {index}"
            runtime_items.append(
                {
                    "claim_id": f"dev-{index}",
                    "original_id": index,
                    "claim": claim,
                    "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
                    "split": "dev",
                    "corpus_relpath": f"dev-{index}.jsonl",
                    "corpus_sha256": "a" * 64,
                    "corpus_bytes": 1,
                    "corpus_records": 1,
                }
            )
        runtime_payload = {
            "schema_version": "1",
            "dataset": "AVeriTeC",
            "revision": "8" * 40,
            "source_metadata_sha256": "a" * 64,
            "seed": 20260817,
            "items": runtime_items,
        }
        self.runtime_manifest = self.root / "runtime.json"
        encoded_runtime = json.dumps(
            runtime_payload, ensure_ascii=False, separators=(",", ":")
        ).encode()
        self.runtime_manifest.write_bytes(encoded_runtime)
        self.runtime_manifest.with_suffix(".json.sha256").write_text(
            hashlib.sha256(encoded_runtime).hexdigest() + "\n", encoding="ascii"
        )
        # Keep the two frozen campaign manifests valid claim-only manifests as well.  The report
        # gate deliberately reloads them with sidecar validation instead of trusting only a hash.
        for path, items in (
            (self.stability_runtime_manifest, runtime_items[:20]),
            (
                self.calibration_runtime_manifest,
                [
                    {
                        "claim_id": f"train-{index}",
                        "original_id": index,
                        "claim": f"Train claim {index}",
                        "claim_sha256": hashlib.sha256(f"Train claim {index}".encode()).hexdigest(),
                        "split": "train",
                        "corpus_relpath": f"train-{index}.jsonl",
                        "corpus_sha256": "a" * 64,
                        "corpus_bytes": 1,
                        "corpus_records": 1,
                    }
                    for index in range(32)
                ],
            ),
        ):
            payload = {
                "schema_version": "1",
                "dataset": "AVeriTeC",
                "revision": "8" * 40,
                "source_metadata_sha256": "a" * 64,
                "seed": 20260817,
                "items": items,
            }
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            path.write_bytes(encoded)
            path.with_suffix(".json.sha256").write_text(
                hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
            )
        payload = {
            "schema_version": "1",
            "dataset": "AVeriTeC",
            "revision": "8" * 40,
            "source_metadata_sha256": "a" * 64,
            "runtime_manifest_sha256": hashlib.sha256(encoded_runtime).hexdigest(),
            "seed": 20260817,
            "items": [
                {
                    "claim_id": f"dev-{index}",
                    "original_id": index,
                    "claim": f"Claim {index}",
                    "label": list(Verdict)[index % len(Verdict)].value,
                    "questions": [],
                    "justification": "fixture",
                    "claim_types": [],
                }
                for index in range(80)
            ],
        }
        path = self.root / "gold.json"
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        path.write_bytes(encoded)
        path.with_suffix(".json.sha256").write_text(
            hashlib.sha256(encoded).hexdigest() + "\n",
            encoding="ascii",
        )

    def _write_current_freeze_inputs(self) -> None:
        self.calibration_runtime_manifest = (
            self.root / "data/manifests/averitec_calibration_runtime.json"
        )
        self.stability_runtime_manifest = (
            self.root / "data/manifests/averitec_stability_runtime.json"
        )
        self.corpus_receipt = self.root / "data/processed/averitec/preparation_receipt.json"
        self.prompt_bundle = self.root / "src/evidence_route/prompts.py"
        self.pricing = self.root / "configs/pricing.local.yaml"
        self.requirements_lock = self.root / "requirements.lock"
        for path, content in (
            (self.calibration_runtime_manifest, "calibration-runtime\n"),
            (self.stability_runtime_manifest, "stability-runtime\n"),
            (self.corpus_receipt, "{}\n"),
            (self.prompt_bundle, "PROMPT = 'fixture'\n"),
            (
                self.pricing,
                "provider: fixture\ncurrency: CNY\ninput_per_million: 1.0\n"
                "output_per_million: 1.0\nprice_source: fixture\nstrict_evaluation: true\n",
            ),
            (self.requirements_lock, "fixture==1.0\n"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def _write_calibration(self):
        plan = build_calibration_plan(
            [f"train-{index}" for index in range(32)],
            activity_id="gate-a",
            manifest_freeze_git_sha="a" * 40,
            runtime_manifest_sha256=_sha(self.calibration_runtime_manifest),
            corpus_preparation_receipt_sha256=_sha(self.corpus_receipt),
            prompt_bundle_sha256=_sha(self.prompt_bundle),
            config_sha256="6" * 64,
            pricing_sha256=_sha(self.pricing),
            endpoint_config_sha256="4" * 64,
            requirements_lock_sha256=_sha(self.requirements_lock),
            requested_alias="fixture-alias",
            seed=20260817,
            cap_micro_cny=350_000_000,
        )
        state = build_calibration_state(plan)
        cases = []
        features = ClaimFeatures(
            claim_units=[ClaimUnit(unit_id="u0", text="fixture")],
            atomic_clause_count=1,
            entity_count=0,
            numeric_count=0,
            time_scope_count=0,
            has_comparison=False,
            has_causal=False,
            has_contrast=False,
            probe_source_count=1,
            probe_score_spread=0.0,
            probe_conflict_hint=False,
        )
        usage = Usage(input_tokens=80, output_tokens=20, total_tokens=100, complete=True)
        for index, work in enumerate(plan.items):
            result_single = VerificationResult(
                claim_id=work.claim_id,
                status=ResultStatus.COMPLETED,
                verdict=list(Verdict)[index % len(Verdict)],
                confidence=0.8,
                rationale="fixture calibration result",
                citations=[],
                available_evidence_ids=[],
                initial_route="single",
                escalated=False,
                usage=usage,
                estimated_cost_micro_cny=100,
                cost_currency="CNY",
                price_config_id=_price_config_id(),
                latency_ms=10,
            )
            result_multi = result_single.model_copy(
                update={
                    "initial_route": "multi",
                    "usage": Usage(
                        input_tokens=64,
                        output_tokens=16,
                        total_tokens=80,
                        complete=True,
                    ),
                    "estimated_cost_micro_cny": 80,
                },
                deep=True,
            )
            call_ids = [
                make_call_id(work.router_run_id, "router", "root", 0),
                make_call_id(work.single_run_id, "single", "root", 0),
                make_call_id(work.multi_run_id, "worker", "t0", 0),
                make_call_id(work.multi_run_id, "worker", "t1", 0),
                make_call_id(work.multi_run_id, "worker", "t2", 0),
                make_call_id(work.multi_run_id, "judge", "root", 0),
            ]
            case = build_calibration_runtime_case(
                plan=plan,
                work=work,
                call_ids=call_ids,
                request_sha256_by_call_id={call_id: "8" * 64 for call_id in call_ids},
                features=features,
                saved_llm_route="single",
                router_usage=usage,
                router_actual_cost_micro_cny=100,
                single_result=result_single,
                multi_result=result_multi,
                response_model_ids_raw=["relay-model-a"] * len(call_ids),
            )
            cases.append(case)
            state.items[index].status = CalibrationItemStatus.COMPLETE
            state.items[index].artifact_sha256 = case.artifact_sha256
            atomic_write_json(
                self.activity_dir / state.items[index].artifact_relpath,
                case.model_dump(mode="json"),
            )
        state.state_sha256 = state_fingerprint(state)
        atomic_write_json(self.activity_dir / "calibration-plan.json", plan.model_dump(mode="json"))
        atomic_write_json(
            self.activity_dir / "calibration-state.json", state.model_dump(mode="json")
        )
        return plan, state, cases

    def _write_calibration_replay(self, plan) -> None:
        candidates = [
            CandidateOutcome(
                config_hash=hashlib.sha256(f"candidate-{index}".encode()).hexdigest(),
                macro_f1=0.5,
                total_tokens=1000 + index,
                llm_router_calls=0,
                llm_router_rate=0.0,
                simulated_cost_micro_cny=1000 + index,
                settings={"clear_multi_clauses": 2},
                manifest_count=32,
                completed_count=32,
            )
            for index in range(54)
        ]
        replay = CalibrationReplay(
            activity_id=plan.activity_id,
            runtime_manifest_sha256=plan.runtime_manifest_sha256,
            prompt_bundle_sha256=plan.prompt_bundle_sha256,
            code_git_sha="b" * 40,
            quality_tolerance=0.03,
            best_candidate_macro_f1=0.5,
            quality_floor_macro_f1=0.47,
            candidates=candidates,
            baselines={"always_single": candidates[0], "always_multi": candidates[1]},
            selected=candidates[0],
            collection_accounting={},
            simulated_selected_accounting={},
            calibrated_config_sha256=_sha(self.calibrated_config),
        )
        report_path = self.root / "reports" / "calibration" / "calibration_report.json"
        atomic_write_json(report_path, replay.model_dump(mode="json"))
        atomic_write_json(
            self.activity_dir / "calibration-replay.json",
            {
                "schema_version": "1",
                "activity_id": plan.activity_id,
                "plan_fingerprint": plan.plan_fingerprint,
                "runtime_manifest_sha256": plan.runtime_manifest_sha256,
                "calibration_report_sha256": _sha(report_path),
                "calibrated_config_sha256": _sha(self.calibrated_config),
                "selected_config_hash": replay.selected.config_hash,
            },
        )

    def _write_run_store(self, artifacts, calibration_cases) -> None:
        self.run_store = self.root / "run-store.sqlite3"
        pricing = PriceConfig(
            provider="fixture",
            currency="CNY",
            input_per_million=1.0,
            output_per_million=1.0,
            price_source="fixture",
            strict_evaluation=True,
        )
        from evidence_route.artifacts import SQLiteRunStore

        SQLiteRunStore(
            self.run_store,
            activity_id="gate-a",
            cap_cny=350.0,
            pricing=pricing,
        )
        rows = []
        for case in calibration_cases:
            slots = [
                (case.router_run_id, "router", "root"),
                (case.single_run_id, "single", "root"),
                (case.multi_run_id, "worker", "t0"),
                (case.multi_run_id, "worker", "t1"),
                (case.multi_run_id, "worker", "t2"),
                (case.multi_run_id, "judge", "root"),
            ]
            for offset, (call_id, (run_id, node, task_id)) in enumerate(
                zip(case.call_ids, slots, strict=True)
            ):
                per_call_tokens = 100 if offset < 2 else 20
                per_call_output = 20 if offset < 2 else 4
                rows.append(
                    (
                        call_id,
                        case.request_sha256_by_call_id[call_id],
                        "gate-a",
                        run_id,
                        node,
                        task_id,
                        0,
                        "completed",
                        per_call_tokens,
                        per_call_tokens,
                        "{}",
                        json.dumps(
                            {
                                "input_tokens": per_call_tokens - per_call_output,
                                "output_tokens": per_call_output,
                                "total_tokens": per_call_tokens,
                                "complete": True,
                            }
                        ),
                        "provider",
                        case.requested_alias,
                        case.response_model_ids_raw[0],
                        0,
                        0,
                        0,
                        "fixture",
                        "fixture",
                    )
                )
        for artifact in artifacts:
            rows.extend(
                (
                    call_id,
                    "7" * 64,
                    "gate-a",
                    artifact.run_id,
                    "router",
                    "root",
                    0,
                    "completed",
                    artifact.actual_cost_micro_cny,
                    artifact.actual_cost_micro_cny,
                    "{}",
                    json.dumps(artifact.usage.model_dump(mode="json")),
                    "provider",
                    artifact.requested_alias,
                    artifact.response_model_ids_raw[0],
                    0,
                    0,
                    artifact.cache_hit_count,
                    "fixture",
                    "fixture",
                )
                for call_id in artifact.call_ids
            )
        connection = sqlite3.connect(self.run_store)
        connection.executemany(
            "INSERT INTO calls (call_id, request_sha256, activity_id, run_id, node, task_id, "
            "logical_attempt, state, reserved_micro_cny, actual_micro_cny, payload_json, "
            "usage_json, usage_source, requested_alias, response_model_id_raw, identity_verified, "
            "transport_attempts, cache_hits, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
        connection.close()

    def _write_official_cache(self, artifacts) -> None:
        adaptive_digest = hashlib.sha256(
            json.dumps(
                sorted(
                    (artifact.run_id, artifact.artifact_sha256)
                    for artifact in artifacts
                    if artifact.phase == "dev"
                    and artifact.strategy is Strategy.ADAPTIVE
                    and artifact.repeat == 0
                ),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        atomic_write_json(
            self.activity_dir / "official/summary.json",
            {
                "shared_task_2024": {"available": True, "metrics": {}},
                "paper_2023_secondary": {"available": True, "metrics": {}},
                "report_evidence": {
                    "activity_id": "gate-a",
                    "runtime_manifest_sha256": _sha(self.runtime_manifest),
                    "gold_manifest_sha256": _sha(self.root / "gold.json"),
                    "adaptive_artifacts_sha256": adaptive_digest,
                },
            },
        )

    def report_input(self, *, allow_official_cache: bool = False) -> ReportInput:
        return ReportInput(
            repository_root=self.root,
            activity_dir=self.activity_dir,
            gold_manifest=self.root / "gold.json",
            nltk_data_root=self.root / "nltk",
            run_store=self.run_store,
            runtime_manifest=self.runtime_manifest,
            calibrated_config=self.calibrated_config,
            calibration_runtime_manifest=self.calibration_runtime_manifest,
            stability_runtime_manifest=self.stability_runtime_manifest,
            corpus_preparation_receipt=self.corpus_receipt,
            prompt_bundle=self.prompt_bundle,
            pricing=self.pricing,
            requirements_lock=self.requirements_lock,
            require_git=False,
            allow_official_cache=allow_official_cache,
        )

    def with_status(self, status: CampaignStatus) -> ReportInput:
        path = self.activity_dir / "activity.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = status.value
        if status is not CampaignStatus.COMPLETE:
            payload["dev_status"] = status.value
        updated = ActivityRecord.model_validate(payload)
        persist_activity(path, updated)
        return self.report_input()


@pytest.fixture
def report_input_factory(tmp_path: Path) -> ReportInputFactory:
    return ReportInputFactory(tmp_path)


@pytest.fixture
def complete_report_input(report_input_factory: ReportInputFactory) -> ReportInput:
    return report_input_factory.report_input(allow_official_cache=True)
