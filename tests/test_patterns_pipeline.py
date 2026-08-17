"""patterns/pipeline.py 的单元测试。

用 FakeLLM / FakeAudit / FakeRegistry / FakeApprover 驱动 PipelinePattern，
全程不访问网络：覆盖接力顺序、上一 agent 内容作为下一 agent 任务、审计事件与 token 统计。
"""

import asyncio

from agent_collab.core import AgentRuntime, AgentSpec, SharedMemory
from agent_collab.patterns import PipelinePattern


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeAudit:
    def __init__(self):
        self.events = []

    def record(self, kind, actor, detail):
        self.events.append((kind, actor, detail))


class FakeLLM:
    """按脚本依序返回 complete 结果，并累计 usage_total 与调用次数。"""

    def __init__(self, responses, tokens_per_response=7):
        self.responses = list(responses)
        self.calls = 0
        self.usage_total = 0
        self.tokens_per_response = tokens_per_response

    def complete(self, messages, tools=None, json_schema=None, max_tokens=None):
        self.calls += 1
        resp = self.responses.pop(0)
        usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
        self.usage_total += int(usage.get("total_tokens", self.tokens_per_response))
        return resp


class FakeRegistry:
    def openai_schemas(self, names):
        return []

    async def execute(self, name, args, approver):
        raise AssertionError("本测试不应调用工具")


class FakeApprover:
    async def request(self, tool_name, args):
        return True


def make_runtime(aid, llm, memory, audit):
    spec = AgentSpec(id=aid, role="角色", goal="目标")
    return AgentRuntime(spec, llm, FakeRegistry(), memory, audit, FakeApprover())


def make_pattern(responses, memory=None, audit=None):
    memory = memory or SharedMemory()
    audit = audit or FakeAudit()
    ids = ["researcher", "verifier", "editor"]
    team = [
        make_runtime(aid, FakeLLM([resp]), memory, audit)
        for aid, resp in zip(ids, responses)
    ]
    return PipelinePattern(team, memory, audit), team, ids


# ---------------------------------------------------------------------------
# 接力顺序与终答
# ---------------------------------------------------------------------------
def test_pipeline_relays_in_order_and_returns_last_content():
    pattern, team, _ = make_pattern([
        {"content": "第一步结论", "tool_calls": None},
        {"content": "第二步结论", "tool_calls": None},
        {"content": "第三步结论", "tool_calls": None},
    ])
    result = asyncio.run(pattern.run("原始查询"))

    assert result.answer == "第三步结论"
    # 每个 agent 恰好被调用一次
    assert all(a.llm.calls == 1 for a in team)


def test_pipeline_passes_previous_content_as_next_task():
    pattern, team, ids = make_pattern([
        {"content": "A 的产出", "tool_calls": None},
        {"content": "B 的产出", "tool_calls": None},
        {"content": "C 的产出", "tool_calls": None},
    ])
    asyncio.run(pattern.run("原始查询"))

    # 从 audit 的 message 事件验证：第一个收到原始查询，后续收到前一个 content
    messages = [e for e in pattern.audit.events if e[0] == "message"]
    assert [m[2]["task"] for m in messages] == ["原始查询", "A 的产出", "B 的产出"]
    assert [m[2]["to"] for m in messages] == ids


def test_pipeline_records_audit_events():
    audit = FakeAudit()
    pattern, _, _ = make_pattern([
        {"content": "x", "tool_calls": None},
        {"content": "y", "tool_calls": None},
    ], audit=audit)
    asyncio.run(pattern.run("q"))

    kinds = [e[0] for e in audit.events]
    assert kinds[0] == "pattern_start"
    assert kinds[-1] == "pattern_end"
    assert kinds.count("message") == 2
    assert kinds.count("pattern_start") == 1
    assert kinds.count("pattern_end") == 1


def test_pipeline_token_accounting_from_usage():
    memory = SharedMemory()
    team = [
        make_runtime("a", FakeLLM([{"content": "x", "tool_calls": None}], tokens_per_response=3), memory, FakeAudit()),
        make_runtime("b", FakeLLM([{"content": "y", "tool_calls": None}], tokens_per_response=5), memory, FakeAudit()),
    ]
    pattern = PipelinePattern(team, memory, FakeAudit())
    result = asyncio.run(pattern.run("q"))

    assert result.tokens == {"a": 3, "b": 5}
    assert result.wall_s >= 0.0
