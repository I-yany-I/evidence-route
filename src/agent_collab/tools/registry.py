"""工具注册表：ToolSpec 登记 / 查询 / OpenAI schemas 导出 / 带审批的执行。

执行契约：
- approval_level != "read-only" 时，先 await approver.request(name, args)，拒绝返回 denied；
- handler 抛异常时吞掉并转成 ToolResult(ok=False, error=str(exc))，不向外抛。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass
class ToolResult:
    """一次工具调用的结果。

    Attributes:
        ok: 是否成功。
        content: 回填给模型的文本。
        error: 失败原因（拒绝 / 异常信息）。
    """

    ok: bool
    content: str
    error: str = ""


@dataclass
class ToolSpec:
    """工具的声明：名称 / 描述 / JSON Schema 参数 / 审批等级 / 处理器。"""

    name: str
    description: str
    parameters: dict  # JSON Schema
    approval_level: str  # read-only | write | execute
    handler: Callable[[dict], Any]  # 同步或异步；register 时统一包装为 async


def _ensure_async(handler: Callable[[dict], Any]) -> Callable[[dict], Awaitable[ToolResult]]:
    """把同步/异步 handler 统一包装为 async 版本。"""
    if inspect.iscoroutinefunction(handler):
        return handler

    async def wrapper(args: dict) -> ToolResult:
        return handler(args)

    return wrapper


class ToolRegistry:
    """按名称登记 ToolSpec，并负责带审批地执行工具。"""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """登记一个工具；同名重复注册抛 ValueError。"""
        if spec.name in self._specs:
            raise ValueError(f"工具已注册：{spec.name}")
        spec.handler = _ensure_async(spec.handler)
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """按名取工具；未注册抛 KeyError。"""
        if name not in self._specs:
            raise KeyError(name)
        return self._specs[name]

    def openai_schemas(self, names: list[str]) -> list[dict]:
        """导出指定工具的 OpenAI tools 格式（type=function）。"""
        schemas = []
        for name in names:
            spec = self.get(name)
            schemas.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            })
        return schemas

    async def execute(self, name: str, args: dict, approver) -> ToolResult:
        """带审批地执行工具；拒绝/异常统一转成 ToolResult，不向外抛。"""
        spec = self.get(name)
        if spec.approval_level != "read-only":
            allowed = await approver.request(name, args)
            if not allowed:
                return ToolResult(ok=False, content="", error="denied")
        try:
            return await spec.handler(args)
        except Exception as exc:  # noqa: BLE001 - 契约要求吞异常转 ok=False
            return ToolResult(ok=False, content="", error=str(exc))
