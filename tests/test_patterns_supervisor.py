"""patterns/supervisor.py 的单元测试。

用 FakeLLM / FakeAudit / FakeRegistry / FakeApprover 驱动 SupervisorPattern，
全程不访问网络：覆盖 done 退出与终答、失败重试、重试耗尽后换人、非 JSON 决策兜底、
以及决策提示包含 facts 与未完成子任务。
"""

import asyncio

from agent_collab.core import AgentRuntime, AgentSpec, Fact, SharedMemory
from agent_collab.patterns import SupervisorPattern


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeAudit:
    def __init__(self):
        self.events = []

    def record(self, kind, actor, detail):
        self.events.append((kind, actor, detail))


class FakeLLM:
    """按脚本依序返回 complete 结果，并记录每次 user 提示与累计 usage。"""

    def __init__(self, responses, tokens_per_response=7):
        self.responses = list(responses)
        self.calls = 0
        self.prompts = []
        self.usage_total = 0
        self.tokens_per_response = tokens_per_response

    def complete(self, messages, tools=None, json_schema=None, max_tokens=None):
        self.calls += 1
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        self.prompts.append(user)
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


def make_supervisor(team_runtimes, memory=None, audit=None, config=None):
    memory = memory or SharedMemory()
    audit = audit or FakeAudit()
    return SupervisorPattern(team_runtimes, memory, audit, config)


# ---------------------------------------------------------------------------
# done 退出与终答
# ---------------------------------------------------------------------------
def test_supervisor_done_exits_and_returns_final():
    memory = SharedMemory()
    audit = FakeAudit()
    supervisor = make_runtime("supervisor", FakeLLM([
        {"content": '{"assign": {"worker_id": "worker-1", "task": "t1"}}', "tool_calls": None},
        {"content": '{"done": true, "final": "最终结论"}', "tool_calls": None},
    ]), memory, audit)
    worker = make_runtime("worker-1", FakeLLM([{"content": "核查结果", "tool_calls": None}]), memory, audit)

    pattern = make_supervisor([supervisor, worker], memory, audit)
    result = asyncio.run(pattern.run("q"))

    assert result.answer == "最终结论"
    # 两轮决策都记录为 decision 事件
    assert sum(1 for e in audit.events if e[0] == "decision") == 2


# ---------------------------------------------------------------------------
# 失败重试 ≤2 次
# ---------------------------------------------------------------------------
def test_supervisor_retries_failed_worker_then_succeeds():
    memory = SharedMemory()
    audit = FakeAudit()
    supervisor = make_runtime("supervisor", FakeLLM([
        {"content": '{"assign": {"worker_id": "worker-1", "task": "t1"}}', "tool_calls": None},
        {"content": '{"done": true, "final": "完成"}', "tool_calls": None},
    ]), memory, audit)
    worker = make_runtime("worker-1", FakeLLM([
        {"content": "", "tool_calls": None},       # 第一次失败
        {"content": "重试后成功", "tool_calls": None},  # 重试成功
    ]), memory, audit)

    pattern = make_supervisor([supervisor, worker], memory, audit)
    result = asyncio.run(pattern.run("q"))

    assert result.answer == "完成"
    retries = [e for e in audit.events if e[0] == "retry"]
    assert len(retries) == 1
    assert retries[0][2] == {"task": "t1", "attempt": 1}
    assert worker.llm.calls == 2


# ---------------------------------------------------------------------------
# 重试耗尽后换人 ≤1 次
# ---------------------------------------------------------------------------
def test_supervisor_reassigns_after_exhausted_retries():
    memory = SharedMemory()
    audit = FakeAudit()
    supervisor = make_runtime("supervisor", FakeLLM([
        {"content": '{"assign": {"worker_id": "worker-1", "task": "t1"}}', "tool_calls": None},
        {"content": '{"done": true, "final": "完成"}', "tool_calls": None},
    ]), memory, audit)
    worker1 = make_runtime("worker-1", FakeLLM([
        {"content": "", "tool_calls": None},
        {"content": "", "tool_calls": None},
        {"content": "", "tool_calls": None},
    ]), memory, audit)
    worker2 = make_runtime("worker-2", FakeLLM([
        {"content": "接手后成功", "tool_calls": None},
    ]), memory, audit)

    pattern = make_supervisor([supervisor, worker1, worker2], memory, audit)
    result = asyncio.run(pattern.run("q"))

    assert result.answer == "完成"
    reassigns = [e for e in audit.events if e[0] == "reassign"]
    assert len(reassigns) == 1
    assert reassigns[0][2] == {"task": "t1", "to": "worker-2"}
    # 原始 worker 尝试 3 次，接手 worker 1 次
    assert worker1.llm.calls == 3
    assert worker2.llm.calls == 1


# ---------------------------------------------------------------------------
# 非 JSON 决策兜底
# ---------------------------------------------------------------------------
def test_supervisor_non_json_decision_falls_back_to_final():
    memory = SharedMemory()
    audit = FakeAudit()
    supervisor = make_runtime("supervisor", FakeLLM([
        {"content": "我已核查完毕，结论为真", "tool_calls": None},
    ]), memory, audit)
    worker = make_runtime("worker-1", FakeLLM([{"content": "x", "tool_calls": None}]), memory, audit)

    pattern = make_supervisor([supervisor, worker], memory, audit)
    result = asyncio.run(pattern.run("q"))

    # 非 JSON 视为 done，原文作为终答
    assert result.answer == "我已核查完毕，结论为真"


# ---------------------------------------------------------------------------
# 决策提示包含 facts 与未完成子任务
# ---------------------------------------------------------------------------
def test_supervisor_prompt_includes_facts_and_pending():
    memory = SharedMemory()
    memory.record_fact(Fact(claim="声明 X 为真", verdict="支持", evidence="官方通报", sources=[], by="worker"))
    audit = FakeAudit()
    supervisor = make_runtime("supervisor", FakeLLM([
        {"content": '{"done": true, "final": "完成"}', "tool_calls": None},
    ]), memory, audit)
    worker = make_runtime("worker-1", FakeLLM([{"content": "x", "tool_calls": None}]), memory, audit)

    pattern = make_supervisor([supervisor, worker], memory, audit)
    asyncio.run(pattern.run("q"))

    prompt = supervisor.llm.prompts[0]
    assert "声明 X 为真" in prompt
    assert "q" in prompt  # 未完成子任务（初始为 query）
