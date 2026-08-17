"""stdio MCP 客户端：JSON-RPC 2.0 over stdio 的最小实现（仅工具桥接）。

只实现工具相关方法：initialize / notifications/initialized / tools/list / tools/call。
消息按 newline-delimited JSON 分帧：Content-Length: N\\r\\n\\r\\n<json>。
工具名统一映射为 mcp__<server>__<name>，全部注册为 read-only 工具。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from .registry import ToolRegistry, ToolResult, ToolSpec

_DEFAULT_PROTOCOL = "2025-03-26"


class MCPError(Exception):
    """MCP 客户端错误（连接 / 协议 / 工具调用失败）。"""


class MCPClient:
    """把一个 stdio MCP 服务器桥接为一组只读工具。"""

    def __init__(self, command: str, args: list[str], cwd: str | None = None,
                 env: dict | None = None, timeout_s: float = 180.0):
        self.command = command
        self.args = list(args)
        self.cwd = cwd
        self.env = env
        self.timeout_s = timeout_s
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._server_name = "mcp"
        self._tools: list[dict] = []

    async def connect(self) -> None:
        """启动子进程并完成 initialize + notifications/initialized + tools/list。"""
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        self._proc = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

        result = await self._initialize()
        self._server_name = (result.get("serverInfo") or {}).get("name", "mcp")
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        self._tools = await self._list_tools()

    async def _initialize(self) -> dict:
        version = _DEFAULT_PROTOCOL
        for _ in range(3):
            resp = await self._request("initialize", {
                "protocolVersion": version,
                "capabilities": {},
                "clientInfo": {"name": "agent-collab", "version": "0.1.0"},
            })
            if "error" not in resp:
                return resp.get("result") or {}
            suggested = self._suggested_version(resp["error"])
            if suggested and suggested != version:
                version = suggested
                continue
            raise MCPError(f"initialize 失败：{resp['error']}")
        raise MCPError("initialize 失败：协议版本协商未达成")

    @staticmethod
    def _suggested_version(error: dict) -> str | None:
        """从版本协商错误的 data 里取建议版本。"""
        data = error.get("data") or {}
        versions = data.get("supportedProtocolVersions") or data.get("supportedVersions")
        if versions:
            return versions[0]
        return data.get("requestedVersion") or data.get("latestVersion")

    async def _list_tools(self) -> list[dict]:
        """拉取全部工具；支持 nextCursor 分页。"""
        tools: list[dict] = []
        cursor: str | None = None
        while True:
            params: dict = {}
            if cursor:
                params["cursor"] = cursor
            resp = await self._request("tools/list", params)
            if "error" in resp:
                raise MCPError(f"tools/list 失败：{resp['error']}")
            result = resp.get("result") or {}
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def tool_names(self) -> list[str]:
        """返回桥接后的工具名 mcp__<server>__<name>。"""
        return [f"mcp__{self._server_name}__{t['name']}" for t in self._tools]

    async def call(self, name: str, arguments: dict) -> str:
        """调用工具并返回文本投影（文本块以换行连接）。name 为桥接名。"""
        prefix = f"mcp__{self._server_name}__"
        if not name.startswith(prefix):
            raise MCPError(f"未知工具：{name}")
        tool_name = name[len(prefix):]
        resp = await self._request("tools/call", {"name": tool_name, "arguments": arguments})
        if "error" in resp:
            raise MCPError(f"tools/call 失败：{resp['error']}")
        result = resp.get("result") or {}
        texts = []
        for block in result.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)

    def register_into(self, registry: ToolRegistry) -> None:
        """把全部 MCP 工具注册为 read-only 工具。"""
        for tool in self._tools:
            tname = tool["name"]
            name = f"mcp__{self._server_name}__{tname}"

            async def handler(args: dict, _name: str = name) -> ToolResult:
                try:
                    content = await self.call(_name, args)
                    return ToolResult(ok=True, content=content)
                except Exception as exc:  # noqa: BLE001
                    return ToolResult(ok=False, content="", error=str(exc))

            registry.register(ToolSpec(
                name=name,
                description=tool.get("description", ""),
                parameters=tool.get("inputSchema") or {"type": "object"},
                approval_level="read-only",
                handler=handler,
            ))

    async def close(self) -> None:
        """终止子进程并回收后台读取任务。"""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        proc = self._proc
        self._proc = None
        if proc is not None:
            if proc.returncode is None:
                try:
                    proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    await proc.wait()
                except Exception:  # noqa: BLE001
                    pass
            # 显式关闭底层 transport，规避 Windows Proactor 的 unclosed transport 告警
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                try:
                    transport.close()
                except Exception:  # noqa: BLE001
                    pass

    async def _request(self, method: str, params: dict) -> dict:
        self._next_id += 1
        req_id = self._next_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
            return await asyncio.wait_for(fut, timeout=self.timeout_s)
        finally:
            self._pending.pop(req_id, None)

    async def _send(self, msg: dict) -> None:
        data = json.dumps(msg, ensure_ascii=False)
        body = data.encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._proc.stdin.write(header + body)
        await self._proc.stdin.drain()

    async def _read_loop(self) -> None:
        reader = self._proc.stdout
        while True:
            frame = await self._read_frame(reader)
            if frame is None:
                break
            if isinstance(frame.get("id"), int):
                fut = self._pending.get(frame["id"])
                if fut is not None and not fut.done():
                    fut.set_result(frame)

    @staticmethod
    async def _read_frame(reader) -> dict | None:
        """读一帧：先读 header 行，再按 Content-Length 读 body。EOF 返回 None。"""
        length = 0
        while True:
            line = await reader.readline()
            if not line:  # EOF
                return None
            if line in (b"\r\n", b"\n"):  # 空行 = header 结束
                break
            text = line.decode("ascii", errors="replace").strip()
            if text.lower().startswith("content-length:"):
                length = int(text.split(":", 1)[1].strip())
        if length <= 0:
            return None
        body = await reader.readexactly(length)
        return json.loads(body.decode("utf-8"))
