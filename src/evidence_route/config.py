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


class HardeningSettings(ConfigModel):
    deterministic_ambiguous: bool = False
    deterministic_decomposition: bool = False
    hardened_judge: bool = False
    hardened_worker: bool = False
    adjudication: bool = False
    normalize_output: bool = False
    multi_single_recovery: bool = False


class EvidenceSettings(ConfigModel):
    probe_top_k: int = Field(default=3, gt=0)
    probe_chars: int = Field(default=450, gt=0)
    single_top_k: int = Field(default=8, gt=0)
    single_chars: int = Field(default=800, gt=0)
    worker_top_k: int = Field(default=5, gt=0)
    worker_chars: int = Field(default=800, gt=0)
    judge_max_evidence: int = Field(default=12, gt=0)
    judge_chars: int = Field(default=600, gt=0)
    max_per_source: int | None = Field(default=None, gt=0)


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
    estimated_cost_cap_cny: float = Field(default=500.0, gt=0)
    reserve_ratio: float = Field(default=0.2, ge=0, le=1)


class AppConfig(ConfigModel):
    llm: LLMSettings
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    hardening: HardeningSettings = Field(default_factory=HardeningSettings)
    evidence: EvidenceSettings = Field(default_factory=EvidenceSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 hash for JSON-compatible configuration data."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def redact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    """Redact common credential keys without mutating the input mapping."""
    sensitive = {"authorization", "api_key", "token", "secret", "password"}
    return {
        key: "[REDACTED]" if key.lower() in sensitive else item
        for key, item in value.items()
    }


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set in the environment")
    return value


def load_app_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    llm_options = dict(raw.get("llm") or {})
    environment_only = {"base_url", "api_key", "requested_alias"}
    present = sorted(environment_only & llm_options.keys())
    if present:
        raise ValueError(f"{', '.join(present)} must come from environment")
    raw["llm"] = {
        **llm_options,
        "base_url": _required_environment("EVIDENCE_ROUTE_BASE_URL"),
        "api_key": _required_environment("EVIDENCE_ROUTE_API_KEY"),
        "requested_alias": _required_environment("EVIDENCE_ROUTE_MODEL"),
    }
    return AppConfig.model_validate(raw)
