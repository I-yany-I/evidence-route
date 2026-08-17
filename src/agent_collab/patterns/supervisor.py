"""监督者模式：supervisor 循环调度，worker 失败重试/换人。"""

from __future__ import annotations

import json
import time

from ..core import AgentRuntime
from .base import AgentResult, CollaborationPattern


class SupervisorPattern(CollaborationPattern):
    """监督者循环：读 facts 与未完成子任务 → 决策 → 派活 → done 退出。

    supervisor 决策为 JSON：``{"assign": {"worker_id": ..., "task": ...}}`` 或
    ``{"done": true, "final": "..."}``。worker 失败（RESULT.content 为空）重试 ≤2 次、
    再失败换人 ≤1 次；done 时终答 = final。循环上限 ``config["max_rounds"]``（默认 5）。
    """

    async def run(self, query: str) -> AgentResult:
        start = time.monotonic()
        self._record_start("supervisor", query)

        supervisor_prefix = self.config.get("supervisor_prefix", "supervisor")
        supervisor = self._find(supervisor_prefix)
        workers = [a for a in self.team if a.spec.id != supervisor.spec.id]
        if not workers:
            raise ValueError("supervisor 模式至少需要一个 worker")

        max_rounds = int(self.config.get("max_rounds", 5))
        pending = [str(t) for t in (self.config.get("tasks") or [query])]
        final = ""
        last_answer = ""

        for _ in range(max_rounds):
            decision = await self._decide(supervisor, pending)
            if decision.get("done"):
                final = str(decision.get("final") or final)
                break

            assign = decision.get("assign") or {}
            worker_id = assign.get("worker_id")
            task = assign.get("task")
            if not worker_id or not task:
                # 决策无效：以决策内容兜底并结束
                final = str(decision.get("final") or json.dumps(decision, ensure_ascii=False))
                break

            worker = self._find_by_id(worker_id)
            content = await self._execute_with_retry(worker, task, workers)
            if content:
                last_answer = content
                pending = [t for t in pending if t != task]

        answer = final or last_answer
        return self._finish("supervisor", answer, start)

    async def _decide(self, supervisor: AgentRuntime, pending: list[str]) -> dict:
        """让 supervisor 依据 facts 与未完成子任务产出决策 JSON。"""
        prompt = self._build_decision_prompt(pending)
        msg = await supervisor.run(prompt)
        self._record_tokens(supervisor)
        decision = _parse_decision((msg.payload.get("content") or "").strip())
        self._decision(supervisor.spec.id, decision)
        return decision

    def _build_decision_prompt(self, pending: list[str]) -> str:
        lines = ["你是监督者。请根据当前状态决定下一步："]
        lines.append("未完成子任务：" + ("；".join(pending) if pending else "（无）"))
        facts = self._facts_summary()
        if facts:
            lines.append("已知事实：\n" + facts)
        lines.append('请输出 JSON：{"assign": {"worker_id": "...", "task": "..."}} '
                     '或 {"done": true, "final": "..."}。')
        return "\n".join(lines)

    async def _execute_with_retry(self, worker: AgentRuntime, task: str,
                                  workers: list[AgentRuntime]) -> str | None:
        """执行任务；失败（content 为空）重试 ≤2 次、再失败换人 ≤1 次。"""
        for attempt in range(3):
            self._dispatch(worker.spec.id, task)
            msg = await worker.run(task)
            self._record_tokens(worker)
            content = (msg.payload.get("content") or "").strip()
            if content:
                self._record_facts(worker.spec.id, task, content)
                return content
            if attempt < 2:
                self.audit.record("retry", worker.spec.id, {"task": task, "attempt": attempt + 1})

        for alt in workers:
            if alt.spec.id == worker.spec.id:
                continue
            self.audit.record("reassign", worker.spec.id, {"task": task, "to": alt.spec.id})
            self._dispatch(alt.spec.id, task)
            msg = await alt.run(task)
            self._record_tokens(alt)
            content = (msg.payload.get("content") or "").strip()
            if content:
                self._record_facts(alt.spec.id, task, content)
                return content
            break  # 只换 1 次
        return None


def _parse_decision(raw: str) -> dict:
    """解析 supervisor 决策 JSON；解析失败时兜底为 done + 原文终答。"""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"done": True, "final": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"done": True, "final": raw}
