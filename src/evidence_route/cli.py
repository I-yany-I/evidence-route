from __future__ import annotations

import json
import secrets
import uuid
from pathlib import Path
from typing import Annotated, Protocol

import typer

from evidence_route.artifacts import atomic_write_json
from evidence_route.contracts import Strategy, StrictModel


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


class CapabilityPayload(StrictModel):
    ok: bool
    nonce: str


class ProductionServices:
    def verify(self, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("production verify wiring is provided by the evaluation runner")

    def provider_smoke(self, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("production provider wiring is not configured")


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

    return app


app = create_app(ProductionServices())


if __name__ == "__main__":
    app()
