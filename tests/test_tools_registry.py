"""tools/registry.py 的单元测试。

覆盖：注册/重名抛错/get 未注册/execute 返回结果/sync 与 async handler 包装/
审批放行与拒绝/handler 异常转 ok=False/openai_schemas 格式。
全程不联网。
"""

import asyncio

import pytest

from agent_collab.tools import ToolRegistry, ToolResult, ToolSpec


class FakeApprover:
    """记录请求、按脚本返回是否放行。"""

    def __init__(self, allow=True):
        self.allow = allow
        self.requests = []

    async def request(self, tool_name, args):
        self.requests.append((tool_name, args))
        return self.allow


def make_spec(name="t", approval_level="read-only", handler=None, description="测试工具"):
    if handler is None:
        def handler(args):
            return ToolResult(ok=True, content="ok")

    return ToolSpec(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        approval_level=approval_level,
        handler=handler,
    )


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------

def test_register_and_get():
    reg = ToolRegistry()
    spec = make_spec(name="web_search")
    reg.register(spec)
    assert reg.get("web_search") is spec
    assert reg.get("web_search").approval_level == "read-only"


def test_register_duplicate_raises():
    reg = ToolRegistry()
    reg.register(make_spec(name="t"))
    with pytest.raises(ValueError):
        reg.register(make_spec(name="t"))


def test_get_unregistered_raises_keyerror():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------

def test_execute_returns_handler_result():
    reg = ToolRegistry()
    reg.register(make_spec(handler=lambda args: ToolResult(ok=True, content="结果")))
    result = asyncio.run(reg.execute("t", {"q": "x"}, FakeApprover()))
    assert result.ok is True
    assert result.content == "结果"


def test_execute_wraps_sync_handler_as_async():
    reg = ToolRegistry()

    def handler(args):
        return ToolResult(ok=True, content=args["q"])

    reg.register(make_spec(handler=handler))
    result = asyncio.run(reg.execute("t", {"q": "你好"}, FakeApprover()))
    assert result.content == "你好"


def test_execute_supports_async_handler():
    reg = ToolRegistry()

    async def handler(args):
        return ToolResult(ok=True, content="async")

    reg.register(make_spec(handler=handler))
    result = asyncio.run(reg.execute("t", {}, FakeApprover()))
    assert result.content == "async"


def test_execute_skips_approval_for_read_only():
    reg = ToolRegistry()
    reg.register(make_spec(approval_level="read-only"))
    approver = FakeApprover(allow=False)
    result = asyncio.run(reg.execute("t", {}, approver))
    assert result.ok is True
    assert approver.requests == []


def test_execute_requests_approval_for_write():
    reg = ToolRegistry()
    reg.register(make_spec(approval_level="write"))
    approver = FakeApprover(allow=True)
    result = asyncio.run(reg.execute("t", {"q": "x"}, approver))
    assert result.ok is True
    assert approver.requests == [("t", {"q": "x"})]


def test_execute_denied_returns_denied():
    reg = ToolRegistry()
    calls = []

    def handler(args):
        calls.append(args)
        return ToolResult(ok=True, content="")

    reg.register(make_spec(approval_level="execute", handler=handler))
    result = asyncio.run(reg.execute("t", {}, FakeApprover(allow=False)))
    assert result.ok is False
    assert result.content == ""
    assert result.error == "denied"
    assert calls == []  # 被拒绝时 handler 不应被调用


def test_execute_handler_exception_to_error():
    reg = ToolRegistry()

    def handler(args):
        raise RuntimeError("boom")

    reg.register(make_spec(handler=handler))
    result = asyncio.run(reg.execute("t", {}, FakeApprover()))
    assert result.ok is False
    assert result.content == ""
    assert result.error == "boom"


# ---------------------------------------------------------------------------
# openai_schemas
# ---------------------------------------------------------------------------

def test_openai_schemas_format():
    reg = ToolRegistry()
    reg.register(make_spec(name="web_search", description="搜索"))
    reg.register(make_spec(name="read_file", description="读文件"))
    schemas = reg.openai_schemas(["web_search", "read_file"])
    assert len(schemas) == 2
    assert [s["function"]["name"] for s in schemas] == ["web_search", "read_file"]
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert set(fn) == {"name", "description", "parameters"}
        assert fn["parameters"]["type"] == "object"
        assert fn["description"] in ("搜索", "读文件")
