"""共享记忆：短期消息缓冲 + 长期事实库。

- 短期：消息缓冲（post/history），全团队可见，history 默认保留最近 50 条。
- 长期：事实库（record_fact/facts），研究员写入，编辑/裁判读取生成报告。
"""

from pydantic import BaseModel

from .message import Message


class Fact(BaseModel):
    """一条结构化事实：待核声明 + 判定 + 证据 + 来源 + 写入方。"""

    claim: str
    verdict: str  # 支持 / 反对 / 存疑 / 证据不足
    evidence: str
    sources: list[str]
    by: str


class SharedMemory:
    """团队级共享记忆（对象注入，无全局状态）。"""

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._facts: list[Fact] = []

    def post(self, msg: Message) -> None:
        """写入短期消息缓冲（全团队可见）。"""
        self._messages.append(msg)

    def history(self, limit: int = 50) -> list[Message]:
        """返回最近 limit 条消息（按写入顺序）。"""
        return self._messages[-limit:]

    def record_fact(self, fact: Fact) -> None:
        """写入长期事实库。"""
        self._facts.append(fact)

    def facts(self) -> list[Fact]:
        """返回全部事实。"""
        return list(self._facts)

    def snapshot(self) -> dict:
        """导出消息与事实的可序列化快照，供审计/回放使用。"""
        return {
            "messages": [m.model_dump() for m in self._messages],
            "facts": [f.model_dump() for f in self._facts],
        }
