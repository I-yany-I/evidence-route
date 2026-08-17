"""审计日志：记录事件、按时间序回放、导出 Markdown 报告（演示用）。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class AuditEvent:
    """一条审计事件。

    Attributes:
        ts: ISO8601 时间戳（UTC）。
        kind: 事件类型。
        actor: 发起方（agent id 或 "system"）。
        detail: 事件详情（可序列化 dict）。
    """

    ts: str
    kind: str
    actor: str
    detail: dict


class AuditLog:
    """进程内审计日志（按写入时间序追加）。"""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(self, kind: str, actor: str, detail: dict) -> None:
        """追加一条事件，ts 取当前 UTC ISO8601 时间。"""
        self._events.append(AuditEvent(
            ts=datetime.now(timezone.utc).isoformat(),
            kind=kind,
            actor=actor,
            detail=dict(detail),
        ))

    def events(self) -> list[AuditEvent]:
        """按写入序返回全部事件。"""
        return list(self._events)

    def replay(self) -> list[dict]:
        """导出可序列化的事件序列，供回放/分析。"""
        return [
            {"ts": e.ts, "kind": e.kind, "actor": e.actor, "detail": e.detail}
            for e in self._events
        ]

    def to_markdown(self) -> str:
        """生成 Markdown 审计报告（演示用）。"""
        lines = ["# 审计日志", ""]
        if not self._events:
            lines.append("（暂无事件）")
        for i, e in enumerate(self._events, 1):
            lines.append(f"## {i}. {e.kind} — {e.actor}（{e.ts}）")
            for k, v in e.detail.items():
                lines.append(f"- **{k}**：{v}")
            lines.append("")
        return "\n".join(lines)
