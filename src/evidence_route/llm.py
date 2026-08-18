from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from evidence_route.artifacts import SQLiteRunStore
from evidence_route.budget import UsageUnavailable
from evidence_route.config import LLMSettings
from evidence_route.contracts import Usage

T = TypeVar("T", bound=BaseModel)


class BillingUncertain(RuntimeError):
    """The transport may have handed a request to a billable provider."""


class AsyncTransport(Protocol):
    async def create(self, **request: object) -> RawCompletion:
        raise NotImplementedError


@dataclass(frozen=True)
class RawCompletion:
    content: str
    response_model_id_raw: str | None
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    value: T
    usage: Usage
    requested_alias: str
    response_model_id_raw: str | None
    identity_verified: bool
    call_ids: tuple[str, ...]
    actual_cost_micro_cny: int | None
    cache_hit: bool
    transport_attempts: int
    usage_source: Literal["provider", "missing"]
    billing_uncertain: bool


def make_call_id(run_id: str, node: str, task_id: str, logical_attempt: int) -> str:
    raw = f"{run_id}\0{node}\0{task_id}\0{logical_attempt}".encode()
    return hashlib.sha256(raw).hexdigest()


def deterministic_fraction(call_id: str, attempt: int) -> float:
    digest = hashlib.sha256(f"{call_id}\0{attempt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class StructuredLLM:
    def __init__(
        self,
        *,
        settings: LLMSettings,
        transport: AsyncTransport,
        run_store: SQLiteRunStore,
        sleeper: Any = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.run_store = run_store
        self.sleeper = sleeper

    async def invoke(
        self,
        *,
        run_id: str,
        node: str,
        task_id: str,
        messages: list[dict[str, str]],
        schema: type[T],
        max_input_tokens: int,
        max_output_tokens: int,
    ) -> StructuredResult[T]:
        return await self._invoke_attempt(
            run_id=run_id,
            node=node,
            task_id=task_id,
            messages=messages,
            schema=schema,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            logical_attempt=0,
        )

    async def _invoke_attempt(
        self,
        *,
        run_id: str,
        node: str,
        task_id: str,
        messages: list[dict[str, str]],
        schema: type[T],
        max_input_tokens: int,
        max_output_tokens: int,
        logical_attempt: int,
    ) -> StructuredResult[T]:
        call_id = make_call_id(run_id, node, task_id, logical_attempt)
        request_sha256 = self._request_hash(
            messages, schema, max_input_tokens, max_output_tokens, logical_attempt
        )
        decision = self.run_store.resume_decision(call_id, request_sha256=request_sha256)
        if decision.action == "reuse_and_stop":
            raise UsageUnavailable("cached response has missing provider usage")
        if decision.action == "reuse":
            metadata = self.run_store.get_call_metadata(call_id)
            if metadata is None or metadata["payload"] is None:
                raise RuntimeError("completed call payload is missing")
            try:
                value = schema.model_validate_json(metadata["payload"]["content"])
            except (ValidationError, KeyError, TypeError) as exc:
                if logical_attempt >= 1:
                    raise
                repair_messages = self._repair_messages(messages, str(exc))
                repaired = await self._invoke_attempt(
                    run_id=run_id,
                    node=node,
                    task_id=task_id,
                    messages=repair_messages,
                    schema=schema,
                    max_input_tokens=max_input_tokens,
                    max_output_tokens=max_output_tokens,
                    logical_attempt=1,
                )
                return StructuredResult(
                    value=repaired.value,
                    usage=repaired.usage,
                    requested_alias=repaired.requested_alias,
                    response_model_id_raw=repaired.response_model_id_raw,
                    identity_verified=repaired.identity_verified,
                    call_ids=(call_id, *repaired.call_ids),
                    actual_cost_micro_cny=repaired.actual_cost_micro_cny,
                    cache_hit=True,
                    transport_attempts=repaired.transport_attempts,
                    usage_source=repaired.usage_source,
                    billing_uncertain=repaired.billing_uncertain,
                )
            usage = metadata["usage"]
            if not isinstance(usage, Usage):
                raise RuntimeError("completed call usage is missing")
            return StructuredResult(
                value=value,
                usage=usage,
                requested_alias=metadata["requested_alias"] or self.settings.requested_alias,
                response_model_id_raw=metadata["response_model_id_raw"],
                identity_verified=bool(metadata["identity_verified"]),
                call_ids=(call_id,),
                actual_cost_micro_cny=metadata["actual_cost_micro_cny"],
                cache_hit=True,
                transport_attempts=int(metadata["transport_attempts"]),
                usage_source=metadata["usage_source"] or "provider",
                billing_uncertain=False,
            )

        self.run_store.reserve_call(
            call_id,
            request_sha256=request_sha256,
            run_id=run_id,
            node=node,
            task_id=task_id,
            logical_attempt=logical_attempt,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )
        self.run_store.mark_sent(call_id)
        try:
            raw = await self._transport_with_retries(
                call_id=call_id,
                messages=messages,
                schema=schema,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
            )
        except BillingUncertain:
            self.run_store.mark_billing_uncertain(call_id)
            raise
        except Exception as exc:
            self.run_store.mark_billing_uncertain(call_id)
            raise BillingUncertain("transport failed after request handoff") from exc

        input_tokens = raw.input_tokens
        output_tokens = raw.output_tokens
        complete_usage = input_tokens is not None and output_tokens is not None
        usage = Usage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            total_tokens=(input_tokens or 0) + (output_tokens or 0),
            complete=complete_usage,
        )
        self.run_store.complete_call(
            call_id,
            request_sha256=request_sha256,
            payload={"content": raw.content},
            usage=usage,
            usage_source="provider" if complete_usage else "missing",
            requested_alias=self.settings.requested_alias,
            response_model_id_raw=raw.response_model_id_raw or "",
            identity_verified=False,
        )
        if not complete_usage:
            raise UsageUnavailable("provider response omitted token usage")
        try:
            value = schema.model_validate_json(raw.content)
        except (ValidationError, ValueError) as exc:
            if logical_attempt >= 1:
                raise
            repaired = await self._invoke_attempt(
                run_id=run_id,
                node=node,
                task_id=task_id,
                messages=self._repair_messages(messages, str(exc)),
                schema=schema,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                logical_attempt=1,
            )
            return StructuredResult(
                value=repaired.value,
                usage=repaired.usage,
                requested_alias=repaired.requested_alias,
                response_model_id_raw=repaired.response_model_id_raw,
                identity_verified=repaired.identity_verified,
                call_ids=(call_id, *repaired.call_ids),
                actual_cost_micro_cny=None,
                cache_hit=False,
                transport_attempts=repaired.transport_attempts,
                usage_source=repaired.usage_source,
                billing_uncertain=False,
            )
        metadata = self.run_store.get_call_metadata(call_id) or {}
        return StructuredResult(
            value=value,
            usage=usage,
            requested_alias=self.settings.requested_alias,
            response_model_id_raw=raw.response_model_id_raw,
            identity_verified=False,
            call_ids=(call_id,),
            actual_cost_micro_cny=metadata.get("actual_cost_micro_cny"),
            cache_hit=False,
            transport_attempts=int(metadata.get("transport_attempts", 1)),
            usage_source="provider",
            billing_uncertain=False,
        )

    async def _transport_with_retries(
        self,
        *,
        call_id: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        max_input_tokens: int,
        max_output_tokens: int,
    ) -> RawCompletion:
        for attempt in range(self.settings.transient_retries + 1):
            try:
                return await self.transport.create(
                    messages=messages,
                    schema=schema,
                    model=self.settings.requested_alias,
                    max_input_tokens=max_input_tokens,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as exc:
                if attempt >= self.settings.transient_retries or not self._retryable(exc):
                    raise
                self.run_store.mark_transport_retry(call_id)
                delay = min(4.0, 0.5 * 2**attempt) * (
                    0.75 + deterministic_fraction(call_id, attempt) * 0.5
                )
                await self.sleeper(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _retryable(error: Exception) -> bool:
        status = getattr(error, "status_code", None)
        return status == 429 or (isinstance(status, int) and status >= 500)

    def _request_hash(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        max_input_tokens: int,
        max_output_tokens: int,
        logical_attempt: int,
    ) -> str:
        payload = {
            "base_url": self.settings.base_url.rstrip("/"),
            "requested_alias": self.settings.requested_alias,
            "messages": messages,
            "schema": schema.model_json_schema(),
            "temperature": self.settings.temperature,
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": max_output_tokens,
            "logical_attempt": logical_attempt,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _repair_messages(
        messages: list[dict[str, str]], validation_error: str
    ) -> list[dict[str, str]]:
        return [
            *messages,
            {
                "role": "user",
                "content": (
                    "Return only JSON matching the supplied schema. "
                    f"The previous output failed validation: {validation_error}"
                ),
            },
        ]


def ensure_v1(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


class OpenAITransport:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=ensure_v1(settings.base_url),
            timeout=settings.timeout_s,
            max_retries=0,
        )

    async def create(self, **request: object) -> RawCompletion:
        schema = request["schema"]
        response_format: dict[str, object]
        if self.settings.structured_mode == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(schema, "__name__", "response"),
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            }
        else:
            response_format = {"type": "json_object"}
        response = await self.client.chat.completions.create(
            model=self.settings.requested_alias,
            messages=request["messages"],
            temperature=self.settings.temperature,
            max_tokens=request["max_output_tokens"],
            response_format=response_format,
        )
        choice = response.choices[0]
        usage = response.usage
        return RawCompletion(
            content=choice.message.content or "",
            response_model_id_raw=getattr(response, "model", None),
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )
