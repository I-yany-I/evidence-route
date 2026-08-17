"""单 agent 运行时：系统提示 + LLM 决策循环 + 工具调用（经审批）。

AgentRuntime 依赖 ToolRegistry / AuditLog / Approver 的接口（由 tools/runtime 组实现），
这里仅按契约做鸭子类型使用，不 import 尚未实现的模块（TYPE_CHECKING 仅供静态分析）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .llm_client import LLMClient
from .memory import SharedMemory
from .message import Message, MessageType, new_message

if TYPE_CHECKING:
    from ..runtime.approval import Approver
    from ..runtime.audit import AuditLog
    from ..tools.registry import ToolRegistry

_HISTORY_LIMIT = 20  # 每步最多携带的最近历史条数（system 之外），节省 token


@dataclass
class AgentSpec:
    """声明式 agent 定义：身份、目标、背景、工具与审批等级。"""

    id: str
    role: str
    goal: str
    backstory: str = ""
    tools: list[str] = field(default_factory=list)
    approval_level: str = "read-only"  # read-only | write | execute


class AgentRuntime:
    """单 agent 执行循环。

    流程：系统提示 → LLM 决策 → 经 ToolRegistry 执行工具（带审批）→ 观察回填 → 循环，
    直到 LLM 返回纯文本（最终答案）或达到 max_steps（以最近文本兜底）。
    """

    def __init__(self, spec: AgentSpec, llm: LLMClient, registry: ToolRegistry,
                 memory: SharedMemory, audit: AuditLog, approver: Approver,
                 max_steps: int = 6):
        self.spec = spec
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.audit = audit
        self.approver = approver
        self.max_steps = max_steps

    def system_prompt(self) -> str:
        """组装系统提示：角色 / 目标 / 背景 / 可用工具说明。"""
        lines = [f"你是{self.spec.role}。", f"目标：{self.spec.goal}"]
        if self.spec.backstory:
            lines.append(f"背景：{self.spec.backstory}")
        if self.spec.tools:
            lines.append("可用工具：" + "、".join(self.spec.tools))
        lines.append("请基于工具返回的观察逐步推进任务，最终用一段文字给出结论。")
        return "\n".join(lines)

    async def run(self, task: str, context: dict | None = None) -> Message:
        """执行任务，返回 RESULT 消息；工具调用与结果写入 audit。"""
        tool_schemas = self.registry.openai_schemas(self.spec.tools) if self.spec.tools else None
        system = {"role": "system", "content": self.system_prompt()}
        history: list[dict] = [{"role": "user", "content": self._build_task(task, context)}]

        final_text = ""
        for step in range(self.max_steps):
            payload = [system] + history[-_HISTORY_LIMIT:]
            resp = self.llm.complete(payload, tools=tool_schemas)
            content = resp.get("content")
            tool_calls = resp.get("tool_calls")

            if content:
                final_text = content  # 记录最近非空文本，供 max_steps 兜底
            if not tool_calls:
                # 纯文本 => 视为最终答案
                final_text = content or final_text
                break

            # 有工具调用：构造 assistant 消息并逐个执行、回填观察
            assistant = self._assistant_message(step, content, tool_calls)
            history.append(assistant)
            for idx, tc in enumerate(tool_calls):
                name = tc["name"]
                args = tc.get("arguments") or {}
                self.audit.record("tool_call", self.spec.id, {"tool": name, "arguments": args})
                result = await self.registry.execute(name, args, self.approver)
                self.audit.record(
                    "tool_result", self.spec.id,
                    {"tool": name, "ok": result.ok, "content": result.content, "error": result.error},
                )
                tool_content = result.content if result.ok else f"工具执行失败：{result.error}"
                history.append({
                    "role": "tool",
                    "tool_call_id": assistant["tool_calls"][idx]["id"],
                    "content": tool_content,
                })

        # max_steps 耗尽时以最近文本兜底，仍返回 RESULT 消息
        return new_message(
            sender=self.spec.id,
            recipient="system",
            type_=MessageType.RESULT,
            payload={"content": final_text, "data": {}},
        )

    @staticmethod
    def _build_task(task: str, context: dict | None) -> str:
        """把任务与上下文拼成首条 user 消息。"""
        if not context:
            return task
        ctx = json.dumps(context, ensure_ascii=False, default=str)
        return f"{task}\n\n上下文：\n{ctx}"

    @staticmethod
    def _assistant_message(step: int, content: str | None, tool_calls: list[dict]) -> dict:
        """把契约格式的 tool_calls 转成 OpenAI assistant 消息（含调用 id）。"""
        oa_calls = []
        for idx, tc in enumerate(tool_calls):
            call_id = f"call_{step}_{idx}"
            oa_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                },
            })
        return {"role": "assistant", "content": content, "tool_calls": oa_calls}
