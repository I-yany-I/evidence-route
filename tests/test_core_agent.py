"""core/agent.py 的单元测试。

用 FakeLLM / FakeRegistry / FakeAudit / FakeApprover 驱动 AgentRuntime 循环，
全程不访问网络：覆盖系统提示、纯文本终结、工具调用回填、审计、max_steps 兜底、
历史裁剪与 context 透传。
"""

import asyncio

from agent_collab.core import AgentRuntime, AgentSpec, MessageType, SharedMemory


# ---------------------------------------------------------------------------
# Fakes（鸭子类型实现契约，不 import 未实现的 tools/runtime 模块）
# ---------------------------------------------------------------------------

class FakeToolResult:
    def __init__(self, ok=True, content="", error=""):
        self.ok = ok
        self.content = content
        self.error = error


class FakeLLM:
    """按脚本依序返回 complete 结果。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools=None, json_schema=None, max_tokens=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self.responses.pop(0)

    def json_complete(self, messages, json_schema, max_tokens=None):
        return self.responses.pop(0)


class FakeRegistry:
    def __init__(self, tool_result=FakeToolResult(ok=True, content="观测结果")):
        self.tool_result = tool_result
        self.executed = []  # (name, args, approver)

    def openai_schemas(self, names):
        return [
            {"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
            for n in names
        ]

    async def execute(self, name, args, approver):
        self.executed.append((name, args, approver))
        return self.tool_result


class FakeAudit:
    def __init__(self):
        self.events = []

    def record(self, kind, actor, detail):
        self.events.append((kind, actor, detail))


class FakeApprover:
    def __init__(self, allow=True):
        self.allow = allow
        self.requests = []

    async def request(self, tool_name, args):
        self.requests.append((tool_name, args))
        return self.allow


def make_spec(**overrides):
    base = dict(id="worker", role="核查员", goal="核查声明", backstory="资深记者",
                tools=["web_search"], approval_level="read-only")
    base.update(overrides)
    return AgentSpec(**base)


def make_runtime(spec=None, llm=None, registry=None, memory=None, audit=None, approver=None,
                 max_steps=6):
    return AgentRuntime(
        spec=spec or make_spec(),
        llm=llm or FakeLLM([]),
        registry=registry or FakeRegistry(),
        memory=memory or SharedMemory(),
        audit=audit or FakeAudit(),
        approver=approver or FakeApprover(),
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# system_prompt
# ---------------------------------------------------------------------------

def test_system_prompt_contains_role_goal_backstory_tools():
    spec = make_spec(tools=["web_search", "read_file"])
    prompt = make_runtime(spec=spec).system_prompt()
    assert "核查员" in prompt
    assert "核查声明" in prompt
    assert "资深记者" in prompt
    assert "web_search" in prompt
    assert "read_file" in prompt


# ---------------------------------------------------------------------------
# run：纯文本终结
# ---------------------------------------------------------------------------

def test_run_returns_result_message_on_plain_text():
    llm = FakeLLM([{"content": "最终结论", "tool_calls": None}])
    rt = make_runtime(llm=llm)
    result = asyncio.run(rt.run("核查声明 X"))
    assert result.type is MessageType.RESULT
    assert result.sender == "worker"
    assert result.payload["content"] == "最终结论"
    assert len(llm.calls) == 1


def test_run_passes_context_into_first_user_message():
    llm = FakeLLM([{"content": "ok", "tool_calls": None}])
    rt = make_runtime(llm=llm)
    asyncio.run(rt.run("核查声明", context={"claim": "X"}))
    messages = llm.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    user_msg = messages[1]
    assert user_msg["role"] == "user"
    assert "核查声明" in user_msg["content"]
    assert "X" in user_msg["content"]


# ---------------------------------------------------------------------------
# run：工具调用
# ---------------------------------------------------------------------------

def test_run_executes_tool_and_backfills_observation():
    registry = FakeRegistry()
    llm = FakeLLM([
        {"content": None, "tool_calls": [{"name": "web_search", "arguments": {"query": "X"}}]},
        {"content": "核查完成", "tool_calls": None},
    ])
    rt = make_runtime(llm=llm, registry=registry)
    result = asyncio.run(rt.run("核查声明"))
    assert result.payload["content"] == "核查完成"
    assert len(registry.executed) == 1
    name, args, _ = registry.executed[0]
    assert name == "web_search"
    assert args == {"query": "X"}


def test_run_records_tool_call_and_result_in_audit():
    audit = FakeAudit()
    llm = FakeLLM([
        {"content": None, "tool_calls": [{"name": "web_search", "arguments": {"query": "X"}}]},
        {"content": "done", "tool_calls": None},
    ])
    rt = make_runtime(llm=llm, audit=audit)
    asyncio.run(rt.run("核查声明"))
    kinds = [e[0] for e in audit.events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    # actor 均为 agent id
    assert all(e[1] == "worker" for e in audit.events)


def test_run_backfills_tool_error_message():
    registry = FakeRegistry(FakeToolResult(ok=False, content="", error="denied"))
    llm = FakeLLM([
        {"content": None, "tool_calls": [{"name": "python_repl", "arguments": {}}]},
        {"content": "结束", "tool_calls": None},
    ])
    rt = make_runtime(llm=llm, registry=registry)
    asyncio.run(rt.run("核查声明"))
    second_messages = llm.calls[1]["messages"]
    tool_msgs = [m for m in second_messages if m["role"] == "tool"]
    assert tool_msgs, "第二轮应包含 tool 消息回填"
    assert any("denied" in m["content"] for m in tool_msgs)


def test_run_executes_multiple_tool_calls_in_one_step():
    registry = FakeRegistry()
    llm = FakeLLM([
        {"content": None, "tool_calls": [
            {"name": "a", "arguments": {"q": 1}},
            {"name": "b", "arguments": {"q": 2}},
        ]},
        {"content": "done", "tool_calls": None},
    ])
    spec = make_spec(tools=["a", "b"])
    rt = make_runtime(spec=spec, llm=llm, registry=registry)
    asyncio.run(rt.run("核查声明"))
    assert [e[0] for e in registry.executed] == ["a", "b"]


def test_run_passes_approver_to_registry():
    registry = FakeRegistry()
    approver = FakeApprover()
    llm = FakeLLM([
        {"content": None, "tool_calls": [{"name": "web_search", "arguments": {}}]},
        {"content": "done", "tool_calls": None},
    ])
    rt = make_runtime(llm=llm, registry=registry, approver=approver)
    asyncio.run(rt.run("核查声明"))
    assert registry.executed[0][2] is approver


def test_run_requests_openai_schemas_for_spec_tools():
    registry = FakeRegistry()
    llm = FakeLLM([{"content": "done", "tool_calls": None}])
    spec = make_spec(tools=["web_search", "read_file"])
    rt = make_runtime(spec=spec, llm=llm, registry=registry)
    asyncio.run(rt.run("核查声明"))
    # tools 透传给了 LLM
    tools = llm.calls[0]["tools"]
    names = [t["function"]["name"] for t in tools]
    assert names == ["web_search", "read_file"]


# ---------------------------------------------------------------------------
# run：max_steps 与历史裁剪
# ---------------------------------------------------------------------------

def test_run_forces_final_after_max_steps():
    llm = FakeLLM([{"content": None, "tool_calls": [{"name": "web_search", "arguments": {}}]}] * 10)
    rt = make_runtime(llm=llm, max_steps=2)
    result = asyncio.run(rt.run("核查声明"))
    assert result.type is MessageType.RESULT
    assert len(llm.calls) == 2


def test_run_trims_history_to_recent_20():
    registry = FakeRegistry()
    # 15 轮工具调用 + 1 轮终答 = 16 步
    responses = [{"content": None, "tool_calls": [{"name": "web_search", "arguments": {}}]}] * 15
    responses.append({"content": "done", "tool_calls": None})
    llm = FakeLLM(responses)
    rt = make_runtime(llm=llm, registry=registry, max_steps=20)
    asyncio.run(rt.run("核查声明"))
    last_messages = llm.calls[-1]["messages"]
    assert last_messages[0]["role"] == "system"
    # system + 最近 20 条历史
    assert len(last_messages) <= 21
