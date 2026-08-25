import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from evidence_route.artifacts import BillingStateError, SQLiteRunStore
from evidence_route.budget import PriceConfig, UsageUnavailable
from evidence_route.config import LLMSettings
from evidence_route.llm import BillingUncertain, RawCompletion, StructuredLLM


class RoutePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: str


class FakeTransport:
    def __init__(self, responses: list[RawCompletion]) -> None:
        self.responses = responses
        self.calls = 0

    async def create(self, **request: object) -> RawCompletion:
        response = self.responses[self.calls]
        self.calls += 1
        return response


class RecordingTransport(FakeTransport):
    def __init__(self, responses: list[RawCompletion]) -> None:
        super().__init__(responses)
        self.requests: list[dict[str, object]] = []

    async def create(self, **request: object) -> RawCompletion:
        self.requests.append(request)
        return await super().create(**request)


class AmbiguousAfterSendTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **request: object) -> RawCompletion:
        self.calls += 1
        raise BillingUncertain("connection lost after request handoff")


class RetryableError(RuntimeError):
    status_code = 503


class RetryThenSuccessTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **request: object) -> RawCompletion:
        self.calls += 1
        if self.calls == 1:
            raise RetryableError("temporarily unavailable")
        return raw('{"route":"single"}')


def raw(content: str, model: str = "relay-model") -> RawCompletion:
    return RawCompletion(
        content=content,
        response_model_id_raw=model,
        input_tokens=100,
        output_tokens=20,
    )


@pytest.mark.asyncio
async def test_invalid_json_is_repaired_once(tmp_path) -> None:
    transport = FakeTransport([raw("not-json"), raw('{"route":"single"}')])
    llm = make_llm(tmp_path, transport)
    result = await llm.invoke(
        run_id="run-1",
        node="router",
        task_id="root",
        messages=[{"role": "user", "content": "route this"}],
        schema=RoutePayload,
        max_input_tokens=1800,
        max_output_tokens=250,
    )
    assert result.value.route == "single"
    assert transport.calls == 2


@pytest.mark.asyncio
async def test_json_object_mode_includes_exact_schema_in_messages(tmp_path) -> None:
    transport = RecordingTransport([raw('{"route":"single"}')])
    llm = make_llm(tmp_path, transport)

    await llm.invoke(
        run_id="run-schema",
        node="router",
        task_id="root",
        messages=[{"role": "user", "content": "route this"}],
        schema=RoutePayload,
        max_input_tokens=1800,
        max_output_tokens=250,
    )

    messages = transport.requests[0]["messages"]
    assert isinstance(messages, list)
    assert '"route"' in messages[-1]["content"]
    assert "required JSON schema" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_completed_call_is_reused_from_cache(tmp_path) -> None:
    transport = FakeTransport([raw('{"route":"single"}')])
    llm = make_llm(tmp_path, transport)
    kwargs = dict(
        run_id="run-1",
        node="router",
        task_id="root",
        messages=[{"role": "user", "content": "route this"}],
        schema=RoutePayload,
        max_input_tokens=1800,
        max_output_tokens=250,
    )
    await llm.invoke(**kwargs)
    await llm.invoke(**kwargs)
    assert transport.calls == 1


@pytest.mark.asyncio
async def test_crash_after_transmit_before_cache_write_blocks_resume(tmp_path) -> None:
    first_transport = AmbiguousAfterSendTransport()
    kwargs = dict(
        run_id="run-uncertain",
        node="single",
        task_id="root",
        messages=[{"role": "user", "content": "verify"}],
        schema=RoutePayload,
        max_input_tokens=1800,
        max_output_tokens=250,
    )
    with pytest.raises(BillingUncertain):
        await make_llm(tmp_path, first_transport).invoke(**kwargs)
    second_transport = FakeTransport([raw('{"route":"single"}')])
    with pytest.raises(BillingStateError, match="unknown billing"):
        await make_llm(tmp_path, second_transport).invoke(**kwargs)
    assert first_transport.calls == 1
    assert second_transport.calls == 0


@pytest.mark.asyncio
async def test_received_response_without_usage_is_persisted_and_not_reissued(tmp_path) -> None:
    transport = FakeTransport(
        [
            RawCompletion(
                content='{"route":"single"}',
                response_model_id_raw="relay-model",
                input_tokens=None,
                output_tokens=None,
            )
        ]
    )
    kwargs = dict(
        run_id="run-missing",
        node="router",
        task_id="root",
        messages=[{"role": "user", "content": "route this"}],
        schema=RoutePayload,
        max_input_tokens=1800,
        max_output_tokens=250,
    )
    with pytest.raises(UsageUnavailable):
        await make_llm(tmp_path, transport).invoke(**kwargs)
    with pytest.raises(UsageUnavailable):
        await make_llm(tmp_path, FakeTransport([])).invoke(**kwargs)
    assert transport.calls == 1


@pytest.mark.asyncio
async def test_non_billable_transport_retry_keeps_one_call_reservation(tmp_path) -> None:
    transport = RetryThenSuccessTransport()
    sleeps: list[float] = []

    async def capture(delay: float) -> None:
        sleeps.append(delay)

    llm = make_llm(tmp_path, transport, sleeper=capture)
    result = await llm.invoke(
        run_id="run-retry",
        node="router",
        task_id="root",
        messages=[{"role": "user", "content": "route this"}],
        schema=RoutePayload,
        max_input_tokens=1800,
        max_output_tokens=250,
    )
    assert result.value.route == "single"
    assert transport.calls == 2
    assert len(sleeps) == 1
    assert llm.run_store.summarize_run("run-retry").call_ids.__len__() == 1


def make_llm(tmp_path, transport, *, sleeper=None) -> StructuredLLM:
    pricing = PriceConfig(
        provider="test",
        currency="CNY",
        input_per_million=1.0,
        output_per_million=1.0,
        price_source="test",
        strict_evaluation=True,
    )
    return StructuredLLM(
        settings=LLMSettings(
            base_url="https://relay.example", api_key="secret", requested_alias="alias"
        ),
        transport=transport,
        sleeper=sleeper or asyncio.sleep,
        run_store=SQLiteRunStore(
            tmp_path / "run-store.sqlite3",
            activity_id="gate-a-test",
            cap_cny=10.0,
            pricing=pricing,
        ),
    )
