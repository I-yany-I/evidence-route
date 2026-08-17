"""tools/mcp_client.py 的单元测试（本地 fake MCP 服务器，不联网）。

fixture 写一个极简 stdio MCP 服务器脚本（JSON-RPC 2.0 over stdio），
对 initialize 返回 serverInfo.name="fake-vision"，对 tools/list 返回 2 个工具，
对 tools/call 返回文本；用它测 connect / tool_names / call / register_into 完整链路。
"""

import asyncio
import sys

import pytest

from agent_collab.runtime import Approver
from agent_collab.tools import MCPClient, ToolRegistry

FAKE_SERVER_SRC = '''import json
import sys


def read_frame():
    stdin = sys.stdin.buffer
    headers = {}
    while True:
        line = stdin.readline()
        if not line:
            raise EOFError
        if line in (b"\\r\\n", b"\\n"):
            break
        key, _, value = line.partition(b":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers[b"content-length"])
    body = stdin.read(length)
    return json.loads(body.decode("utf-8"))


def send_frame(obj):
    data = json.dumps(obj, ensure_ascii=True).encode("utf-8")
    sys.stdout.buffer.write(
        b"Content-Length: " + str(len(data)).encode("ascii") + b"\\r\\n\\r\\n" + data
    )
    sys.stdout.buffer.flush()


def handle(msg):
    method = msg.get("method")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "serverInfo": {"name": "fake-vision", "version": "1.0.0"},
            },
        }
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {
                "tools": [
                    {
                        "name": "describe",
                        "description": "describe an image",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"image": {"type": "string"}},
                            "required": ["image"],
                        },
                    },
                    {
                        "name": "ocr",
                        "description": "extract text from image",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ],
            },
        }
    if method == "tools/call":
        params = msg.get("params") or {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}
        return {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "tool=" + tool_name + " args=" + json.dumps(arguments, ensure_ascii=True),
                    }
                ],
                "isError": False,
            },
        }
    return {"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32601, "message": "method not found"}}


while True:
    try:
        msg = read_frame()
    except (EOFError, ValueError, json.JSONDecodeError):
        break
    if msg.get("id") is not None:
        send_frame(handle(msg))
'''


@pytest.fixture
def fake_server(tmp_path):
    script = tmp_path / "fake_mcp_server.py"
    script.write_text(FAKE_SERVER_SRC, encoding="utf-8")
    return script


def _connect(server_path):
    return MCPClient(sys.executable, [str(server_path)], timeout_s=10)


def test_connect_and_tool_names(fake_server):
    async def main():
        client = _connect(fake_server)
        try:
            await client.connect()
            assert client.tool_names() == [
                "mcp__fake-vision__describe",
                "mcp__fake-vision__ocr",
            ]
        finally:
            await client.close()

    asyncio.run(main())


def test_call_returns_text(fake_server):
    async def main():
        client = _connect(fake_server)
        try:
            await client.connect()
            text = await client.call("mcp__fake-vision__describe", {"image": "cat.png"})
            assert "describe" in text
            assert "cat.png" in text
        finally:
            await client.close()

    asyncio.run(main())


def test_register_into_registers_read_only_tools(fake_server):
    async def main():
        client = _connect(fake_server)
        reg = ToolRegistry()
        try:
            await client.connect()
            client.register_into(reg)
            assert reg.get("mcp__fake-vision__describe").approval_level == "read-only"
            assert reg.get("mcp__fake-vision__ocr").approval_level == "read-only"
        finally:
            await client.close()

    asyncio.run(main())


def test_full_chain_execute_via_registry(fake_server):
    async def main():
        client = _connect(fake_server)
        reg = ToolRegistry()
        approver = Approver(auto_approve=True)
        try:
            await client.connect()
            client.register_into(reg)
            result = await reg.execute(
                "mcp__fake-vision__describe", {"image": "a.png"}, approver
            )
            assert result.ok is True
            assert "describe" in result.content
            assert "a.png" in result.content
        finally:
            await client.close()

    asyncio.run(main())
