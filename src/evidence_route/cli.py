from __future__ import annotations

import json
import os
import secrets
import uuid
from pathlib import Path
from typing import Annotated, Protocol

import typer
import yaml

from evidence_route.artifacts import atomic_write_json
from evidence_route.budget import PriceConfig
from evidence_route.config import BudgetSettings, GenerationSettings
from evidence_route.contracts import Strategy, StrictModel
from evidence_route.evaluation.reporting import ReportInput, build_report_bundle
from evidence_route.evaluation.runner import (
    compute_gate_a_call_profile,
    estimate_call_bounds,
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
    ok: bool
    nonce: str


class ProductionServices:
    def verify(self, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("production verify wiring is provided by the evaluation runner")

    def provider_smoke(self, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("production provider wiring is not configured")

    def preview_campaign(self, **kwargs: object) -> dict[str, object]:
        config_path = Path(kwargs["config_path"])
        pricing_path = Path(kwargs["pricing_path"])
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw_config, dict):
            raise ValueError("configuration root must be a mapping")
        generation = GenerationSettings.model_validate(raw_config.get("generation", {}))
        budget = BudgetSettings.model_validate(raw_config.get("budget", {}))
        raw_pricing = yaml.safe_load(pricing_path.read_text(encoding="utf-8")) or {}
        pricing = PriceConfig.model_validate(raw_pricing)
        bounds = estimate_call_bounds(
            compute_gate_a_call_profile(),
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
        raise RuntimeError("production campaign executor is not configured")

    def calibrate_collect(self, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("production calibration executor is not configured")

    def calibrate_replay(self, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("calibration replay requires collected case artifacts")

    def report(self, **kwargs: object) -> dict[str, object]:
        activity_dir = Path(kwargs["activity_dir"])
        output_dir = Path(kwargs["output_dir"])
        publish = bool(kwargs["publish"])
        readme_value = kwargs.get("readme")
        nltk_value = os.environ.get("NLTK_DATA")
        bundle = build_report_bundle(
            ReportInput(
                repository_root=Path.cwd(),
                activity_dir=activity_dir,
                gold_manifest=Path(kwargs["gold_manifest"]),
                nltk_data_root=(
                    Path(nltk_value) if nltk_value else Path("data/external/nltk")
                ),
            ),
            publish=publish,
        )
        bundle.write(
            output_dir,
            readme=Path(readme_value) if readme_value is not None else None,
        )
        return {
            "publishable": bundle.publication_gate.publishable,
            "output_dir": str(output_dir),
            "summary": str(output_dir / "summary.json"),
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
        corpus_dir: Annotated[
            Path, typer.Option("--corpus-dir")
        ] = Path("data/processed/averitec/corpora"),
        config_path: Annotated[Path, typer.Option("--config")] = Path("configs/default.yaml"),
        pricing_path: Annotated[Path | None, typer.Option("--pricing")] = None,
        artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = Path("artifacts"),
        checkpoint_db: Annotated[
            Path, typer.Option("--checkpoint-db")
        ] = Path("artifacts/checkpoints.sqlite3"),
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
        pricing_path: Annotated[
            Path, typer.Option("--pricing")
        ] = Path("configs/pricing.local.yaml"),
        artifact_dir: Annotated[
            Path, typer.Option("--artifact-dir")
        ] = Path("artifacts/provider-smoke"),
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
                if key
                not in {"api_key", "headers", "request", "request_headers", "raw_request"}
            }
            atomic_write_json(artifact_dir / f"{nonce}.json", safe_payload)
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(safe_payload, ensure_ascii=False, sort_keys=True))

    @app.command()
    def evaluate(
        manifest: Annotated[Path, typer.Option("--manifest")] = ...,
        stability_manifest: Annotated[
            Path, typer.Option("--stability-manifest")
        ] = ...,
        config_path: Annotated[Path, typer.Option("--config")] = Path(
            "configs/default.yaml"
        ),
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
        campaign_id: Annotated[str, typer.Option("--campaign-id")] = "gate-a-dev",
        start_after_calibration: Annotated[
            bool, typer.Option("--start-after-calibration")
        ] = False,
        resume: Annotated[bool, typer.Option("--resume")] = False,
        accept_paid_campaign: Annotated[
            bool, typer.Option("--accept-paid-campaign")
        ] = False,
    ) -> None:
        common = {
            "manifest": manifest,
            "stability_manifest": stability_manifest,
            "config_path": config_path,
            "pricing_path": pricing_path,
            "corpus_dir": corpus_dir,
            "activity_dir": activity_dir,
            "checkpoint_db": checkpoint_db,
            "run_store": run_store,
            "activity_id": activity_id,
            "campaign_id": campaign_id,
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
        runtime_manifest: Annotated[
            Path, typer.Option("--runtime-manifest")
        ] = ...,
        config_path: Annotated[Path, typer.Option("--config")] = Path(
            "configs/default.yaml"
        ),
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
        accept_paid_campaign: Annotated[
            bool, typer.Option("--accept-paid-campaign")
        ] = False,
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
        publish: Annotated[bool, typer.Option("--publish")] = False,
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
                publish=publish,
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
