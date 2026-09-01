from __future__ import annotations

import asyncio
import json
import os
import secrets
import uuid
from pathlib import Path
from typing import Annotated, Literal, Protocol

import typer
import yaml
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import Field

from evidence_route.artifacts import SQLiteRunStore, TraceWriter, atomic_write_json
from evidence_route.budget import PriceConfig
from evidence_route.config import (
    BudgetSettings,
    GenerationSettings,
    HardeningSettings,
    load_app_config,
    stable_hash,
)
from evidence_route.contracts import Strategy, StrictModel
from evidence_route.evaluation.production_calibration import ProductionCalibrationCollector
from evidence_route.evaluation.production_evaluation import ProductionCampaignService
from evidence_route.evaluation.reporting import ReportInput, build_report_bundle
from evidence_route.evaluation.runner import (
    compute_gate_a_call_profile,
    estimate_call_bounds,
)
from evidence_route.execution import load_price_config
from evidence_route.graph import GraphComponents, build_graph, initial_state
from evidence_route.llm import OpenAITransport, StructuredLLM, ensure_v1
from evidence_route.providers.averitec import AveritecFrozenProvider
from evidence_route.routing import HybridRouter
from evidence_route.validation import ResultValidator, adjudicate_verification_results
from evidence_route.verification import (
    ClaimDecomposer,
    EvidenceWorker,
    SingleVerifier,
    VerdictJudge,
)


class CliServices(Protocol):
    def verify(
        self,
        *,
        run_id: str,
        resume: bool,
        claim_id: str,
        claim: str,
        strategy: str,
        artifact_dir: Path,
        config_path: Path,
        pricing_path: Path | None,
        corpus_dir: Path,
        checkpoint_db: Path,
    ) -> dict[str, object]: ...

    def provider_smoke(
        self, *, nonce: str, config_path: Path, pricing_path: Path, artifact_dir: Path
    ) -> dict[str, object]: ...

    def preview_campaign(self, **kwargs: object) -> dict[str, object]: ...

    def evaluate(self, **kwargs: object) -> dict[str, object]: ...

    def calibrate_collect(self, **kwargs: object) -> dict[str, object]: ...

    def calibrate_replay(self, **kwargs: object) -> dict[str, object]: ...

    def report(self, **kwargs: object) -> dict[str, object]: ...


class CapabilityPayload(StrictModel):
    ok: Literal[True]
    nonce: str = Field(min_length=8, max_length=64)


class ProviderSmokeResult(StrictModel):
    structured_output: Literal[True]
    usage_complete: Literal[True]
    requested_alias: str = Field(min_length=1)
    response_model_id_raw: str = Field(min_length=1)
    identity_verified: Literal[False]
    nonce: str = Field(min_length=8, max_length=64)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_micro_cny: int = Field(ge=0)
    call_ids: list[Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        min_length=1, max_length=1
    )
    endpoint_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def _resolve_from(root: Path, value: object) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


class ProductionServices:
    def __init__(
        self,
        *,
        transport_factory: object | None = None,
        transport: object | None = None,
        executor_factory: object | None = None,
        repository_root: Path | None = None,
    ) -> None:
        if transport_factory is None and transport is not None:

            def transport_factory(settings: object) -> object:
                del settings
                return transport

        if transport_factory is None:

            def transport_factory(settings: object) -> object:
                return OpenAITransport(settings)  # type: ignore[arg-type]

        self._calibration_collector = ProductionCalibrationCollector(
            transport_factory=transport_factory,  # type: ignore[arg-type]
            executor_factory=executor_factory,  # type: ignore[arg-type]
            repository_root=repository_root,
        )
        self._campaign_service = ProductionCampaignService(
            transport_factory=transport_factory,  # type: ignore[arg-type]
            executor_factory=executor_factory,  # type: ignore[arg-type]
            repository_root=repository_root,
        )

    def verify(self, **kwargs: object) -> dict[str, object]:
        return asyncio.run(self._verify_async(**kwargs))

    async def _verify_async(self, **kwargs: object) -> dict[str, object]:
        run_id = str(kwargs["run_id"])
        claim_id = str(kwargs["claim_id"])
        claim = str(kwargs["claim"])
        strategy = Strategy(str(kwargs["strategy"]))
        resume = bool(kwargs["resume"])
        artifact_dir = Path(kwargs["artifact_dir"])
        config_path = Path(kwargs["config_path"])
        corpus_dir = Path(kwargs["corpus_dir"])
        checkpoint_db = Path(kwargs["checkpoint_db"])
        pricing_value = kwargs.get("pricing_path")
        if pricing_value is None:
            pricing_value = os.environ.get("EVIDENCE_ROUTE_PRICE_FILE")
        if pricing_value is None:
            raise ValueError("pricing_path or EVIDENCE_ROUTE_PRICE_FILE is required")
        pricing = load_price_config(Path(pricing_value))
        app_config = load_app_config(config_path)

        run_dir = artifact_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_store = SQLiteRunStore(
            run_dir / "run-store.sqlite3",
            activity_id=run_id,
            cap_cny=app_config.budget.estimated_cost_cap_cny,
            pricing=pricing,
        )
        provider = AveritecFrozenProvider(corpus_dir)
        transport = OpenAITransport(app_config.llm)
        llm = StructuredLLM(settings=app_config.llm, transport=transport, run_store=run_store)
        components = GraphComponents(
            provider=provider,
            router=HybridRouter(
                app_config.routing,
                app_config.generation,
                llm=llm,
                deterministic_ambiguous=app_config.hardening.deterministic_ambiguous,
            ),
            single=SingleVerifier(
                provider,
                llm,
                app_config.evidence,
                app_config.generation,
                hardened=app_config.hardening.hardened_judge,
            ),
            decomposer=ClaimDecomposer(
                llm,
                app_config.generation,
                deterministic=app_config.hardening.deterministic_decomposition,
            ),
            worker=EvidenceWorker(
                provider,
                llm,
                app_config.evidence,
                app_config.generation,
                hardened=app_config.hardening.hardened_worker,
            ),
            judge=VerdictJudge(
                llm,
                app_config.evidence,
                app_config.generation,
                hardened=app_config.hardening.hardened_judge,
            ),
            validator=ResultValidator(
                low_confidence=app_config.routing.low_confidence,
                minimum_coverage=app_config.routing.minimum_coverage,
                normalize_output=app_config.hardening.normalize_output,
            ),
            evidence_settings=app_config.evidence,
            run_store=run_store,
            trace=TraceWriter(run_dir / "trace.jsonl"),
            adjudicator=(
                adjudicate_verification_results
                if app_config.hardening.adjudication
                else None
            ),
            multi_single_recovery=app_config.hardening.multi_single_recovery,
        )
        graph_config = {"configurable": {"thread_id": run_id}}
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db)) as saver:
            # Pydantic evidence objects are part of graph state; pickle fallback keeps
            # the async SQLite checkpoint faithful across process restarts.
            saver.serde = JsonPlusSerializer(pickle_fallback=True)
            graph = build_graph(components, checkpointer=saver)
            snapshot = await graph.aget_state(graph_config)
            values = getattr(snapshot, "values", None) or {}
            next_nodes = getattr(snapshot, "next", ())
            if resume:
                if values:
                    checkpoint_claim = values.get("claim_id")
                    if checkpoint_claim != claim_id:
                        raise ValueError("resume claim_id does not match checkpoint claim")
                    checkpoint_text = values.get("claim_text")
                    if checkpoint_text != claim:
                        raise ValueError("resume claim does not match checkpoint claim")
                    checkpoint_strategy = values.get("strategy")
                    try:
                        checkpoint_strategy = Strategy(str(checkpoint_strategy))
                    except ValueError as exc:
                        raise ValueError("checkpoint strategy is invalid") from exc
                    if checkpoint_strategy is not strategy:
                        raise ValueError("resume strategy does not match checkpoint strategy")
                    input_state = None
                else:
                    input_state = initial_state(run_id, claim_id, claim, strategy)
            else:
                if values or getattr(snapshot, "next", ()):
                    raise ValueError("run_id already has a checkpoint; use --resume")
                input_state = initial_state(run_id, claim_id, claim, strategy)
            if resume and values and not next_nodes:
                state = values
            else:
                state = await graph.ainvoke(input_state, config=graph_config)

        result = state.get("final_result")
        if result is None:
            raise ValueError("graph completed without final_result")
        summary = run_store.summarize_run(run_id)
        return {
            **result.model_dump(mode="json"),
            "run_id": run_id,
            "strategy": strategy.value,
            "summary": summary.model_dump(mode="json"),
        }

    def provider_smoke(self, **kwargs: object) -> dict[str, object]:
        return asyncio.run(self._provider_smoke_async(**kwargs))

    async def _provider_smoke_async(self, **kwargs: object) -> dict[str, object]:
        nonce = str(kwargs["nonce"])
        config_path = Path(kwargs["config_path"])
        pricing_path = Path(kwargs["pricing_path"])
        artifact_dir = Path(kwargs["artifact_dir"])
        expected = CapabilityPayload(ok=True, nonce=nonce)
        app_config = load_app_config(config_path)
        pricing = load_price_config(pricing_path)
        endpoint_config_hash = stable_hash({"base_url": ensure_v1(app_config.llm.base_url)})
        run_id = stable_hash({"kind": "provider-smoke", "nonce": nonce})
        run_store = SQLiteRunStore(
            artifact_dir / "run-store.sqlite3",
            activity_id="provider-smoke",
            cap_cny=app_config.budget.estimated_cost_cap_cny,
            pricing=pricing,
        )
        llm = StructuredLLM(
            settings=app_config.llm,
            transport=OpenAITransport(app_config.llm),
            run_store=run_store,
        )
        response = await llm.invoke(
            run_id=run_id,
            node="provider_smoke",
            task_id="capability",
            messages=[
                {
                    "role": "system",
                    "content": "Return only JSON that exactly matches the supplied schema.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Echo this capability payload exactly: {expected.model_dump_json()}"
                    ),
                },
            ],
            schema=CapabilityPayload,
            max_input_tokens=app_config.generation.router.max_input_tokens,
            max_output_tokens=app_config.generation.router.max_output_tokens,
            allow_repair=False,
        )
        if response.value.nonce != nonce:
            raise ValueError("provider capability nonce mismatch")

        summary = run_store.summarize_run(run_id)
        if summary.call_ids != list(response.call_ids) or len(summary.call_ids) != 1:
            raise ValueError("provider capability must contain exactly one logical call")
        if summary.usage != response.usage or not summary.usage.complete:
            raise ValueError("provider capability usage is incomplete or inconsistent")
        if summary.requested_aliases != [app_config.llm.requested_alias]:
            raise ValueError("provider capability requested alias is inconsistent")
        if len(summary.response_model_ids_raw) != 1:
            raise ValueError("provider did not report a model id")
        if summary.actual_cost_micro_cny is None or summary.cost_is_lower_bound:
            raise ValueError("provider capability cost is not exact")
        if summary.billing_uncertain:
            raise ValueError("provider capability billing is uncertain")

        result = ProviderSmokeResult(
            structured_output=True,
            usage_complete=True,
            requested_alias=summary.requested_aliases[0],
            response_model_id_raw=summary.response_model_ids_raw[0],
            identity_verified=False,
            nonce=response.value.nonce,
            input_tokens=summary.usage.input_tokens,
            output_tokens=summary.usage.output_tokens,
            estimated_cost_micro_cny=summary.actual_cost_micro_cny,
            call_ids=summary.call_ids,
            endpoint_config_hash=endpoint_config_hash,
        )
        return result.model_dump(mode="json")

    def preview_campaign(self, **kwargs: object) -> dict[str, object]:
        config_path = Path(kwargs["config_path"])
        pricing_path = Path(kwargs["pricing_path"])
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw_config, dict):
            raise ValueError("configuration root must be a mapping")
        generation = GenerationSettings.model_validate(raw_config.get("generation", {}))
        hardening = HardeningSettings.model_validate(raw_config.get("hardening", {}))
        budget = BudgetSettings.model_validate(raw_config.get("budget", {}))
        raw_pricing = yaml.safe_load(pricing_path.read_text(encoding="utf-8")) or {}
        pricing = PriceConfig.model_validate(raw_pricing)
        bounds = estimate_call_bounds(
            compute_gate_a_call_profile(
                include_multi_recovery=hardening.multi_single_recovery
            ),
            generation,
            pricing,
            reserve_ratio=budget.reserve_ratio,
        )
        return {
            **bounds.model_dump(mode="json"),
            "cap_micro_cny": int(budget.estimated_cost_cap_cny * 1_000_000),
            "paid_execution_started": False,
        }

    def evaluate(self, **kwargs: object) -> dict[str, object]:
        return asyncio.run(self._campaign_service.evaluate(**kwargs))

    def calibrate_collect(self, **kwargs: object) -> dict[str, object]:
        return asyncio.run(self._calibrate_collect_async(**kwargs))

    async def _calibrate_collect_async(self, **kwargs: object) -> dict[str, object]:
        return await self._calibration_collector.collect(**kwargs)

    def calibrate_replay(self, **kwargs: object) -> dict[str, object]:
        return self._calibration_collector.replay(**kwargs)

    def report(self, **kwargs: object) -> dict[str, object]:
        repository_root = Path(kwargs["repository_root"]).resolve()
        activity_dir = _resolve_from(repository_root, kwargs["activity_dir"])
        output_dir = _resolve_from(repository_root, kwargs["output_dir"])
        publish = bool(kwargs["publish"])
        stability_diagnostics = bool(kwargs.get("stability_diagnostics", False))
        readme_value = kwargs.get("readme")
        bundle = build_report_bundle(
            ReportInput(
                repository_root=repository_root,
                activity_dir=activity_dir,
                gold_manifest=_resolve_from(repository_root, kwargs["gold_manifest"]),
                nltk_data_root=_resolve_from(repository_root, kwargs["nltk_data_root"]),
                run_store=_resolve_from(repository_root, kwargs["run_store"]),
                runtime_manifest=_resolve_from(repository_root, kwargs["runtime_manifest"]),
                calibration_runtime_manifest=_resolve_from(
                    repository_root, kwargs["calibration_runtime_manifest"]
                ),
                stability_runtime_manifest=_resolve_from(
                    repository_root, kwargs["stability_runtime_manifest"]
                ),
                calibration_report=_resolve_from(repository_root, kwargs["calibration_report"]),
                calibrated_config=_resolve_from(repository_root, kwargs["calibrated_config"]),
                corpus_preparation_receipt=_resolve_from(
                    repository_root, kwargs["corpus_preparation_receipt"]
                ),
                prompt_bundle=_resolve_from(repository_root, kwargs["prompt_bundle"]),
                pricing=_resolve_from(repository_root, kwargs["pricing"]),
                requirements_lock=_resolve_from(repository_root, kwargs["requirements_lock"]),
            ),
            publish=publish,
            stability_diagnostics=stability_diagnostics,
        )
        bundle.write(
            output_dir,
            readme=(
                _resolve_from(repository_root, readme_value) if readme_value is not None else None
            ),
        )
        return {
            "publishable": bundle.publication_gate.publishable,
            "output_dir": str(output_dir),
            "summary": str(output_dir / "summary.json"),
            "stability_diagnostics": (
                str(output_dir / "stability_diagnostics.json")
                if getattr(bundle, "stability_diagnostics", None) is not None
                else None
            ),
        }


def create_app(services: CliServices) -> typer.Typer:
    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.command()
    def verify(
        run_id: Annotated[str | None, typer.Option("--run-id")] = None,
        resume: Annotated[bool, typer.Option("--resume")] = False,
        claim_id: Annotated[str, typer.Option("--claim-id")] = ...,
        claim: Annotated[str, typer.Option("--claim")] = ...,
        strategy: Annotated[str, typer.Option("--strategy")] = "adaptive",
        corpus_dir: Annotated[Path, typer.Option("--corpus-dir")] = Path(
            "data/processed/averitec/corpora"
        ),
        config_path: Annotated[Path, typer.Option("--config")] = Path("configs/default.yaml"),
        pricing_path: Annotated[Path | None, typer.Option("--pricing")] = None,
        artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = Path("artifacts"),
        checkpoint_db: Annotated[Path, typer.Option("--checkpoint-db")] = Path(
            "artifacts/checkpoints.sqlite3"
        ),
    ) -> None:
        if resume and not run_id:
            typer.echo("--resume requires --run-id", err=True)
            raise typer.Exit(code=2)
        try:
            Strategy(strategy)
        except ValueError as exc:
            raise typer.BadParameter(
                "must be always_single, always_multi, or adaptive", param_hint="--strategy"
            ) from exc
        resolved_run_id = run_id or str(uuid.uuid4())
        try:
            payload = services.verify(
                run_id=resolved_run_id,
                resume=resume,
                claim_id=claim_id,
                claim=claim,
                strategy=strategy,
                artifact_dir=artifact_dir,
                config_path=config_path,
                pricing_path=pricing_path,
                corpus_dir=corpus_dir,
                checkpoint_db=checkpoint_db,
            )
            destination = artifact_dir / resolved_run_id / "result.json"
            atomic_write_json(destination, payload)
        except typer.Exit:
            raise
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    @app.command("provider-smoke")
    def provider_smoke(
        accept_paid_call: Annotated[bool, typer.Option("--accept-paid-call")] = False,
        config_path: Annotated[Path, typer.Option("--config")] = Path("configs/default.yaml"),
        pricing_path: Annotated[Path, typer.Option("--pricing")] = Path(
            "configs/pricing.local.yaml"
        ),
        artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = Path(
            "artifacts/provider-smoke"
        ),
    ) -> None:
        if not accept_paid_call:
            typer.echo("provider-smoke may incur a paid call; pass --accept-paid-call", err=True)
            raise typer.Exit(code=2)
        nonce = secrets.token_urlsafe(12)
        try:
            payload = services.provider_smoke(
                nonce=nonce,
                config_path=config_path,
                pricing_path=pricing_path,
                artifact_dir=artifact_dir,
            )
            required = {
                "structured_output": True,
                "usage_complete": True,
                "identity_verified": False,
            }
            if any(payload.get(key) != value for key, value in required.items()):
                raise ValueError("provider capability response is incomplete")
            if payload.get("nonce") != nonce:
                raise ValueError("provider capability nonce mismatch")
            if not payload.get("response_model_id_raw"):
                raise ValueError("provider did not report a model id")
            if not isinstance(payload.get("estimated_cost_micro_cny"), int):
                raise ValueError("provider did not report integer cost")
            safe_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"api_key", "headers", "request", "request_headers", "raw_request"}
            }
            atomic_write_json(artifact_dir / f"{nonce}.json", safe_payload)
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(safe_payload, ensure_ascii=False, sort_keys=True))

    @app.command()
    def evaluate(
        manifest: Annotated[Path, typer.Option("--manifest")] = ...,
        stability_manifest: Annotated[Path, typer.Option("--stability-manifest")] = ...,
        config_path: Annotated[Path, typer.Option("--config")] = Path("configs/default.yaml"),
        pricing_path: Annotated[Path, typer.Option("--pricing")] = Path(
            "configs/pricing.local.yaml"
        ),
        corpus_dir: Annotated[Path, typer.Option("--corpus-dir")] = Path(
            "data/processed/averitec/corpora"
        ),
        activity_dir: Annotated[Path, typer.Option("--activity-dir")] = ...,
        calibration_report: Annotated[Path, typer.Option("--calibration-report")] = Path(
            "reports/calibration/calibration_report.json"
        ),
        checkpoint_db: Annotated[Path, typer.Option("--checkpoint-db")] = Path(
            "artifacts/checkpoints.sqlite3"
        ),
        run_store: Annotated[Path, typer.Option("--run-store")] = Path(
            "artifacts/gate-a-run-store.sqlite3"
        ),
        activity_id: Annotated[str, typer.Option("--activity-id")] = "gate-a",
        campaign_id: Annotated[str, typer.Option("--campaign-id")] = "gate-a-dev",
        start_after_calibration: Annotated[bool, typer.Option("--start-after-calibration")] = False,
        resume: Annotated[bool, typer.Option("--resume")] = False,
        accept_paid_campaign: Annotated[bool, typer.Option("--accept-paid-campaign")] = False,
        max_items: Annotated[int, typer.Option("--max-items")] = 10,
        parent_activity: Annotated[str | None, typer.Option("--parent-activity")] = None,
        parent_report: Annotated[Path | None, typer.Option("--parent-report")] = None,
        parent_config: Annotated[Path | None, typer.Option("--parent-config")] = None,
        experiment_dir: Annotated[Path | None, typer.Option("--experiment-dir")] = None,
    ) -> None:
        common = {
            "manifest": manifest,
            "stability_manifest": stability_manifest,
            "config_path": config_path,
            "pricing_path": pricing_path,
            "corpus_dir": corpus_dir,
            "activity_dir": activity_dir,
            "calibration_report": calibration_report,
            "checkpoint_db": checkpoint_db,
            "run_store": run_store,
            "activity_id": activity_id,
            "campaign_id": campaign_id,
            "max_items": max_items,
            "parent_activity": parent_activity,
            "parent_report": parent_report,
            "parent_config": parent_config,
            "experiment_dir": experiment_dir,
        }
        try:
            if not accept_paid_campaign:
                payload = services.preview_campaign(**common)
            else:
                if start_after_calibration == resume:
                    typer.echo(
                        "paid evaluate requires exactly one of "
                        "--start-after-calibration or --resume",
                        err=True,
                    )
                    raise typer.Exit(code=2)
                payload = services.evaluate(
                    **common,
                    mode="resume" if resume else "start_after_calibration",
                )
        except typer.Exit:
            raise
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    @app.command()
    def calibrate(
        runtime_manifest: Annotated[Path, typer.Option("--runtime-manifest")] = ...,
        config_path: Annotated[Path, typer.Option("--config")] = Path("configs/default.yaml"),
        pricing_path: Annotated[Path, typer.Option("--pricing")] = Path(
            "configs/pricing.local.yaml"
        ),
        corpus_dir: Annotated[Path, typer.Option("--corpus-dir")] = Path(
            "data/processed/averitec/corpora"
        ),
        activity_dir: Annotated[Path, typer.Option("--activity-dir")] = ...,
        checkpoint_db: Annotated[Path, typer.Option("--checkpoint-db")] = Path(
            "artifacts/checkpoints.sqlite3"
        ),
        run_store: Annotated[Path, typer.Option("--run-store")] = Path(
            "artifacts/gate-a-run-store.sqlite3"
        ),
        activity_id: Annotated[str, typer.Option("--activity-id")] = "gate-a",
        output_config: Annotated[Path, typer.Option("--output-config")] = ...,
        output_report: Annotated[Path, typer.Option("--output-report")] = ...,
        resume: Annotated[bool, typer.Option("--resume")] = False,
        collect: Annotated[bool, typer.Option("--collect")] = False,
        replay: Annotated[bool, typer.Option("--replay")] = False,
        gold_manifest: Annotated[Path | None, typer.Option("--gold-manifest")] = None,
        accept_paid_campaign: Annotated[bool, typer.Option("--accept-paid-campaign")] = False,
        max_cases: Annotated[int, typer.Option("--max-cases")] = 4,
    ) -> None:
        if collect == replay:
            typer.echo("choose exactly one of --collect or --replay", err=True)
            raise typer.Exit(code=2)
        common = {
            "runtime_manifest": runtime_manifest,
            "config_path": config_path,
            "pricing_path": pricing_path,
            "corpus_dir": corpus_dir,
            "activity_dir": activity_dir,
            "checkpoint_db": checkpoint_db,
            "run_store": run_store,
            "activity_id": activity_id,
            "output_config": output_config,
            "output_report": output_report,
            "resume": resume,
            "max_cases": max_cases,
        }
        try:
            if replay:
                if gold_manifest is None:
                    typer.echo("--replay requires --gold-manifest", err=True)
                    raise typer.Exit(code=2)
                payload = services.calibrate_replay(
                    **common,
                    gold_manifest=gold_manifest,
                )
            elif not accept_paid_campaign:
                payload = services.preview_campaign(**common)
            else:
                payload = services.calibrate_collect(**common)
        except typer.Exit:
            raise
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    @app.command()
    def report(
        activity_dir: Annotated[Path, typer.Option("--activity-dir")] = ...,
        gold_manifest: Annotated[Path, typer.Option("--gold-manifest")] = ...,
        output_dir: Annotated[Path, typer.Option("--output-dir")] = ...,
        repository_root: Annotated[Path, typer.Option("--repository-root")] = Path("."),
        run_store: Annotated[Path, typer.Option("--run-store")] = Path(
            "artifacts/gate-a-run-store.sqlite3"
        ),
        runtime_manifest: Annotated[Path, typer.Option("--runtime-manifest")] = Path(
            "data/manifests/averitec_dev_runtime.json"
        ),
        calibration_runtime_manifest: Annotated[
            Path, typer.Option("--calibration-runtime-manifest")
        ] = Path("data/manifests/averitec_calibration_runtime.json"),
        stability_runtime_manifest: Annotated[
            Path, typer.Option("--stability-runtime-manifest")
        ] = Path("data/manifests/averitec_stability_runtime.json"),
        calibration_report: Annotated[Path, typer.Option("--calibration-report")] = Path(
            "reports/calibration/calibration_report.json"
        ),
        calibrated_config: Annotated[Path, typer.Option("--calibrated-config")] = Path(
            "configs/calibrated.yaml"
        ),
        corpus_preparation_receipt: Annotated[
            Path, typer.Option("--corpus-preparation-receipt")
        ] = Path("data/processed/averitec/preparation_receipt.json"),
        prompt_bundle: Annotated[Path, typer.Option("--prompt-bundle")] = Path(
            "src/evidence_route/prompts.py"
        ),
        pricing: Annotated[Path, typer.Option("--pricing")] = Path("configs/pricing.local.yaml"),
        requirements_lock: Annotated[Path, typer.Option("--requirements-lock")] = Path(
            "requirements.lock"
        ),
        nltk_data_root: Annotated[Path, typer.Option("--nltk-data-root")] = Path(
            "data/external/nltk"
        ),
        publish: Annotated[bool, typer.Option("--publish")] = False,
        stability_diagnostics: Annotated[
            bool, typer.Option("--stability-diagnostics")
        ] = False,
        readme: Annotated[Path | None, typer.Option("--readme")] = None,
    ) -> None:
        if readme is not None and not publish:
            typer.echo("--readme requires --publish", err=True)
            raise typer.Exit(code=2)
        try:
            payload = services.report(
                activity_dir=activity_dir,
                gold_manifest=gold_manifest,
                output_dir=output_dir,
                repository_root=repository_root,
                run_store=run_store,
                runtime_manifest=runtime_manifest,
                calibration_runtime_manifest=calibration_runtime_manifest,
                stability_runtime_manifest=stability_runtime_manifest,
                calibration_report=calibration_report,
                calibrated_config=calibrated_config,
                corpus_preparation_receipt=corpus_preparation_receipt,
                prompt_bundle=prompt_bundle,
                pricing=pricing,
                requirements_lock=requirements_lock,
                nltk_data_root=nltk_data_root,
                publish=publish,
                stability_diagnostics=stability_diagnostics,
                readme=readme,
            )
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    return app


app = create_app(ProductionServices())


if __name__ == "__main__":
    app()
