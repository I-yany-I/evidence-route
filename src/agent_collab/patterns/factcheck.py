"""组合模式：并行核查 → 辩论裁决 → 编辑撰写。

把三个独立阶段串联成事实核查的完整闭环：
1. parallel：planner 拆解 → fact_checker 并行核查 → 事实写入共享记忆（editor 汇总仅作中间产物，不作为终答）
2. debate：debater_pro / debater_con 依共享事实辩论 → judge 四选一裁决
3. editor：依 judge 裁决 + 共享事实撰写最终结构化核查报告

组合模式本身是「模式可复用」的证明：parallel / debate 作为子阶段被原样复用，
共享同一 memory 与 audit，事实库在阶段间传递。
"""

from __future__ import annotations

import time

from ..core import AgentRuntime
from .base import AgentResult, CollaborationPattern
from .debate import DebatePattern
from .parallel import ParallelPattern


class FactCheckPattern(CollaborationPattern):
    """事实核查完整闭环：核查 → 辩论 → 裁决 → 撰写。"""

    async def run(self, query: str) -> AgentResult:
        start = time.monotonic()
        self._record_start("factcheck", query)

        # 阶段 1：并行核查（fact_checker 的结果经 _record_facts 写入共享记忆）
        parallel = ParallelPattern(self.team, self.memory, self.audit, dict(self.config))
        await parallel.run(query)

        # 阶段 2：辩论 + 裁决（judge 读取共享事实，产出四选一 verdict）
        debate = DebatePattern(self.team, self.memory, self.audit, dict(self.config))
        verdict = await debate.run(query)

        # 阶段 3：编辑撰写（以 judge 裁决为判定，fact 库为证据链）
        editor = self._find(self.config.get("editor_prefix", "editor"))
        task = self._editor_task(query, verdict.answer, self._facts_summary())
        self._dispatch(editor.spec.id, task)
        editor_msg = await editor.run(task)
        answer = (editor_msg.payload.get("content") or "").strip()

        # 全团队 token 归因（子阶段各自记录到自己实例，这里按代理统一重估）
        for agent in self.team:
            self._record_tokens(agent)

        return self._finish("factcheck", answer, start)

    @staticmethod
    def _editor_task(query: str, verdict: str, facts_summary: str) -> str:
        """组装编辑撰写任务：以裁决为判定、事实库为证据链。"""
        parts: list[str] = ["请撰写最终事实核查报告。"]
        parts.append(f"待核声明：{query}")
        parts.append(f"裁判裁决（以它作为报告的最终判定）：\n{verdict}")
        if facts_summary:
            parts.append(f"核查事实：\n{facts_summary}")
        parts.append("报告需包含：总体判定、证据链（逐条）、主要来源、结论。")
        return "\n\n".join(parts)
