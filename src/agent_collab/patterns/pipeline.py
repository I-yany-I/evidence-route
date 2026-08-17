"""流水线模式：按团队顺序接力执行。"""

from __future__ import annotations

import time

from .base import AgentResult, CollaborationPattern


class PipelinePattern(CollaborationPattern):
    """A → B → C 顺序接力。

    第一个 agent 的任务是 ``query``；后续 agent 的任务为前一个 agent 的
    RESULT.content（作为上下文）。终答 = 最后一个 agent 的 content。
    """

    async def run(self, query: str) -> AgentResult:
        start = time.monotonic()
        self._record_start("pipeline", query)

        task = query
        answer = ""
        for agent in self.team:
            self._dispatch(agent.spec.id, task)
            result = await agent.run(task)
            answer = (result.payload.get("content") or "").strip()
            self._record_tokens(agent)
            task = answer or query  # 空答案兜底：下一个 agent 仍拿到原始任务

        return self._finish("pipeline", answer, start)
