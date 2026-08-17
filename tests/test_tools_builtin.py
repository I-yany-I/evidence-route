"""tools/builtin.py 的单元测试。

web_search 用 monkeypatch 换掉 DDGS（不联网）；python_repl 真跑子进程；
read_file 覆盖白名单外路径/扩展名白名单/1MB 上限。全程不联网。
"""

import asyncio

import pytest

from agent_collab.tools import ToolRegistry
from agent_collab.tools.builtin import register_builtin_tools


class Approver:
    def __init__(self, allow=True):
        self.allow = allow
        self.requests = []

    async def request(self, tool_name, args):
        self.requests.append((tool_name, args))
        return self.allow


def _execute(registry, name, args, allow=True):
    return asyncio.run(registry.execute(name, args, Approver(allow=allow)))


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

def test_web_search_formats_results(tmp_path, monkeypatch):
    class FakeDDGS:
        def text(self, query, max_results=5):
            assert query == "OpenAI"
            assert max_results == 5
            return [
                {"title": "标题A", "href": "http://a", "body": "摘要A"},
                {"title": "标题B", "href": "http://b", "body": "摘要B"},
            ]

    monkeypatch.setattr("agent_collab.tools.builtin.DDGS", FakeDDGS)
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    result = _execute(reg, "web_search", {"query": "OpenAI"})
    assert result.ok is True
    assert "标题A" in result.content
    assert "http://a" in result.content
    assert "摘要B" in result.content


def test_web_search_failure_returns_ok_false(tmp_path, monkeypatch):
    class FailingDDGS:
        def text(self, query, max_results=5):
            raise RuntimeError("网络错误")

    monkeypatch.setattr("agent_collab.tools.builtin.DDGS", FailingDDGS)
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    result = _execute(reg, "web_search", {"query": "x"})
    assert result.ok is False
    assert "网络错误" in result.error


# ---------------------------------------------------------------------------
# python_repl
# ---------------------------------------------------------------------------

def test_python_repl_runs_code(tmp_path):
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    result = _execute(reg, "python_repl", {"code": "print(1+1)"})
    assert result.ok is True
    assert "2" in result.content


def test_python_repl_truncates_long_output(tmp_path):
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    result = _execute(reg, "python_repl", {"code": "print('x' * 5000)"})
    assert result.ok is True
    assert len(result.content) == 2000


def test_python_repl_timeout(tmp_path):
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    result = _execute(reg, "python_repl", {"code": "import time; time.sleep(10)"})
    assert result.ok is False
    assert "超时" in result.error


def test_python_repl_requires_approval(tmp_path):
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    result = _execute(reg, "python_repl", {"code": "print(1)"}, allow=False)
    assert result.ok is False
    assert result.error == "denied"


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def test_read_file_within_workspace(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("你好世界", encoding="utf-8")
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    result = _execute(reg, "read_file", {"path": str(f)})
    assert result.ok is True
    assert result.content == "你好世界"


def test_read_file_relative_path_within_workspace(tmp_path):
    (tmp_path / "data.json").write_text('{"a": 1}', encoding="utf-8")
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    result = _execute(reg, "read_file", {"path": "data.json"})
    assert result.ok is True
    assert '"a": 1' in result.content


def test_read_file_outside_workspace_rejected(tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    f = outside / "secret.txt"
    f.write_text("secret", encoding="utf-8")
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    result = _execute(reg, "read_file", {"path": str(f)})
    assert result.ok is False
    assert "白名单" in result.error


def test_read_file_disallowed_extension_rejected(tmp_path):
    f = tmp_path / "malware.exe"
    f.write_bytes(b"MZ")
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    result = _execute(reg, "read_file", {"path": str(f)})
    assert result.ok is False
    assert "扩展名" in result.error


def test_read_file_too_large_rejected(tmp_path):
    f = tmp_path / "big.txt"
    f.write_bytes(b"a" * (1024 * 1024 + 1))
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    result = _execute(reg, "read_file", {"path": str(f)})
    assert result.ok is False
    assert "1MB" in result.error


# ---------------------------------------------------------------------------
# 审批等级
# ---------------------------------------------------------------------------

def test_builtin_tool_approval_levels(tmp_path):
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    assert reg.get("web_search").approval_level == "read-only"
    assert reg.get("python_repl").approval_level == "execute"
    assert reg.get("read_file").approval_level == "read-only"
