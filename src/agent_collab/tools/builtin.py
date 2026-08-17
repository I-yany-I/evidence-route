"""内置工具：web_search / python_repl / read_file。

- web_search：DuckDuckGo 文本搜索，返回前 5 条（标题/链接/摘要）；read-only。
- python_repl：subprocess 沙箱执行 python -c，5s 超时 + 输出截断 2000 字符；execute。
- read_file：workspace 白名单 + 扩展名白名单 + 1MB 上限；read-only。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from duckduckgo_search import DDGS

from .registry import ToolRegistry, ToolResult, ToolSpec

_MAX_OUTPUT_CHARS = 2000
_MAX_FILE_BYTES = 1024 * 1024  # 1MB
_ALLOWED_EXTS = {".txt", ".md", ".json", ".yaml", ".yml", ".py", ".csv", ".log"}


def _web_search(args: dict) -> ToolResult:
    """调用 DDGS().text(query, max_results=5)，把结果格式化为文本。"""
    query = str(args.get("query", "")).strip()
    if not query:
        return ToolResult(ok=False, content="", error="缺少 query 参数")
    try:
        results = DDGS().text(query, max_results=5) or []
    except Exception as exc:  # noqa: BLE001 - 搜索失败统一转 ok=False
        return ToolResult(ok=False, content="", error=str(exc))
    blocks = []
    for r in results:
        title = r.get("title", "")
        url = r.get("href") or r.get("url", "")
        snippet = r.get("body") or r.get("snippet", "")
        blocks.append(f"标题：{title}\n链接：{url}\n摘要：{snippet}")
    return ToolResult(ok=True, content="\n\n".join(blocks))


def _python_repl(args: dict) -> ToolResult:
    """在临时目录用 python -c 执行代码；5s 超时，输出截断 2000 字符。"""
    code = str(args.get("code", ""))
    if not code:
        return ToolResult(ok=False, content="", error="缺少 code 参数")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                timeout=5,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(ok=False, content="", error=f"执行超时：{exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, content="", error=str(exc))
    output = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS]
    if proc.returncode != 0:
        return ToolResult(ok=False, content=output, error=f"退出码 {proc.returncode}")
    return ToolResult(ok=True, content=output)


def register_builtin_tools(registry: ToolRegistry, workspace: Path) -> None:
    """注册三个内置工具；workspace 为 read_file 的路径白名单根。"""
    root = workspace.resolve()

    def read_file(args: dict) -> ToolResult:
        raw = str(args.get("path", "")).strip()
        if not raw:
            return ToolResult(ok=False, content="", error="缺少 path 参数")
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve()
        if not resolved.is_file():
            return ToolResult(ok=False, content="", error="文件不存在或不是普通文件")
        if not resolved.is_relative_to(root):
            return ToolResult(ok=False, content="", error="路径超出 workspace 白名单")
        if resolved.suffix.lower() not in _ALLOWED_EXTS:
            return ToolResult(ok=False, content="", error=f"不允许的扩展名：{resolved.suffix}")
        if resolved.stat().st_size > _MAX_FILE_BYTES:
            return ToolResult(ok=False, content="", error="文件超过 1MB 上限")
        try:
            return ToolResult(ok=True, content=resolved.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, content="", error=str(exc))

    registry.register(ToolSpec(
        name="web_search",
        description="用 DuckDuckGo 搜索网页，返回前 5 条结果（标题/链接/摘要）。",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词"}},
            "required": ["query"],
        },
        approval_level="read-only",
        handler=_web_search,
    ))
    registry.register(ToolSpec(
        name="python_repl",
        description="在沙箱中执行一段 Python 代码（subprocess，5 秒超时，输出截断 2000 字符）。",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "要执行的 Python 代码"}},
            "required": ["code"],
        },
        approval_level="execute",
        handler=_python_repl,
    ))
    registry.register(ToolSpec(
        name="read_file",
        description="读取 workspace 内的文本文件（白名单扩展名，≤1MB）。",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "相对或绝对路径"}},
            "required": ["path"],
        },
        approval_level="read-only",
        handler=read_file,
    ))
