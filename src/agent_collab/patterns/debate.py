"""辩论模式：pro 与 con 交替发言 → judge 裁决。"""

from __future__ import annotations

import json
import time

from ..core import AgentRuntime
from .base import AgentResult, CollaborationPattern


class DebatePattern(CollaborationPattern):
    """正反方各发言 ``rounds`` 轮（默认 2），judge 依双方观点与 facts 裁决。

    每轮任务 = 对方上一轮观点 + 共享 facts 摘要；judge 输出
    ``{"verdict": "...", "reasoning": "..."}``；终答 = verdict + reasoning。
    """

    async def run(self, query: str) -> AgentResult:
        start = time.monotonic()
        self._record_start("debate", query)

        pro = self._find(self.config.get("pro_prefix", "pro"))
        con = self._find(self.config.get("con_prefix", "con"))
        judge = self._find(self.config.get("judge_prefix", "judge"))
        rounds = int(self.config.get("rounds", 2))

        facts_summary = self._facts_summary()
        pro_views: list[str] = []
        con_views: list[str] = []
        pro_last = ""
        con_last = ""

        for r in range(rounds):
            pro_task = self._turn_task(query, con_last, facts_summary, first=(r == 0))
            pro_last = await self._speak(pro, pro_task)
            pro_views.append(pro_last)
            con_task = self._turn_task(query, pro_last, facts_summary, first=False)
            con_last = await self._speak(con, con_task)
            con_views.append(con_last)

        judge_task = self._judge_task(pro_views, con_views, facts_summary)
        judge_msg = await judge.run(judge_task)
        self._record_tokens(judge)
        verdict, reasoning = _parse_verdict((judge_msg.payload.get("content") or "").strip())
        self._decision(judge.spec.id, {"verdict": verdict, "reasoning": reasoning})

        answer = verdict if not reasoning else f"{verdict}\n{reasoning}"
        return self._finish("debate", answer, start)

    async def _speak(self, agent: AgentRuntime, task: str) -> str:
        """让某方发言，返回其观点文本。"""
        self._dispatch(agent.spec.id, task)
        msg = await agent.run(task)
        self._record_tokens(agent)
        return (msg.payload.get("content") or "").strip()

    def _turn_task(self, query: str, opponent: str, facts_summary: str, first: bool) -> str:
        """组装某轮发言任务：首轮 pro 用论题，其余轮次用对方上一轮观点。"""
        parts: list[str] = []
        if first:
            parts.append(f"论题：{query}")
        else:
            parts.append(f"对方观点：{opponent}")
        if facts_summary:
            parts.append(f"已知事实：\n{facts_summary}")
        return "\n\n".join(parts)

    def _judge_task(self, pro_views: list[str], con_views: list[str],
                    facts_summary: str) -> str:
        # 证据优先、观点其次：先给共享事实，再给双方论证，裁判只认证据与论证强度
        parts = ["请根据证据与正反双方观点作出裁决（证据优先于观点）："]
        if facts_summary:
            parts.append("已知事实：\n" + facts_summary)
        parts.append("正方观点：\n" + "\n".join(f"- {v}" for v in pro_views))
        parts.append("反方观点：\n" + "\n".join(f"- {v}" for v in con_views))
        parts.append('输出 JSON：{"verdict": "...", "reasoning": "..."}，verdict 必须是 真实/虚假/部分属实/证据不足 之一。')
        return "\n\n".join(parts)


def _parse_verdict(raw: str) -> tuple[str, str]:
    """解析 judge 裁决 JSON；解析失败时兜底为 verdict=原文、reasoning=空。"""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, ""
    if isinstance(parsed, dict):
        return str(parsed.get("verdict", "")), str(parsed.get("reasoning", ""))
    return raw, ""
