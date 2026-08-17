"""并行模式：planner 拆解 → worker replicas 真并发核查 → aggregator 汇总。"""

from __future__ import annotations

import asyncio
import json
import time

from ..core import AgentRuntime
from .base import AgentResult, CollaborationPattern

_PLAN_SCHEMA = {
    "name": "plan",
    "schema": {
        "type": "object",
        "properties": {
            "subtasks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["subtasks"],
    },
}


class ParallelPattern(CollaborationPattern):
    """事实核查主模式：planner 拆解 → workers 真并发 → aggregator 汇总。

    约定 team 中按 id 前缀区分 planner / worker / aggregator（前缀由 config 提供，
    默认 ``planner``/``worker``/``aggregator``）。子任务按轮询轮流分配给 worker，
    各 worker 经 :func:`asyncio.to_thread` 放入线程真并发执行（LLM complete 为同步阻塞）。
    """

    async def run(self, query: str) -> AgentResult:
        start = time.monotonic()
        self._record_start("parallel", query)

        planner_prefix = self.config.get("planner_prefix", "planner")
        worker_prefix = self.config.get("worker_prefix", "worker")
        aggregator_prefix = self.config.get("aggregator_prefix", "aggregator")

        planner = self._find(planner_prefix)
        workers = self._find_all(worker_prefix)
        aggregator = self._find(aggregator_prefix)
        if not workers:
            raise ValueError("parallel 模式至少需要一个 worker")

        # 1. planner 拆解
        plan_msg = await planner.run(query)
        self._record_tokens(planner)
        subtasks = self._extract_subtasks(plan_msg, planner, query)
        if not subtasks:
            subtasks = [query]

        # 2. 轮流分配 + 真并发执行（每 worker 一个线程，内部顺序执行其子任务）
        buckets: list[list[str]] = [[] for _ in workers]
        for i, subtask in enumerate(subtasks):
            buckets[i % len(workers)].append(subtask)
        for worker, tasks in zip(workers, buckets):
            for t in tasks:
                self._dispatch(worker.spec.id, t)

        bucket_results = await asyncio.gather(*[
            asyncio.to_thread(self._run_bucket_sync, worker, tasks)
            for worker, tasks in zip(workers, buckets)
        ])
        for worker in workers:
            self._record_tokens(worker)
        # 把各 worker 的核查结果写入共享事实库（供 aggregator/judge/审计回放使用）
        for worker, bucket in zip(workers, bucket_results):
            for task, msg in bucket:
                content = (msg.payload.get("content") if msg else "") or ""
                self._record_facts(worker.spec.id, task, content)

        # 3. aggregator 汇总
        summary = self._summarize(bucket_results)
        agg_msg = await aggregator.run(summary)
        self._record_tokens(aggregator)
        answer = (agg_msg.payload.get("content") or "").strip()

        return self._finish("parallel", answer, start)

    @staticmethod
    def _run_bucket_sync(worker: AgentRuntime, tasks: list[str]) -> list[tuple[str, object]]:
        """在独立线程中顺序执行某 worker 被分配的子任务。

        真并发由 :func:`asyncio.to_thread` 保证：不同 worker 运行在不同线程，
        其内部的同步 LLM 调用窗口得以重叠。
        """
        results: list[tuple[str, object]] = []
        for task in tasks:
            results.append((task, asyncio.run(worker.run(task))))
        return results

    def _extract_subtasks(self, plan_msg: object, planner: AgentRuntime, query: str) -> list[str]:
        """从 planner 产出中提取子任务，多处回退，不因解析失败而崩溃。

        顺序：payload.data → content 中的 JSON → llm.json_complete（真实路径）→ 整体兜底。
        """
        subtasks = _subtasks_from(plan_msg.payload.get("data"))  # type: ignore[attr-defined]
        if subtasks:
            return subtasks
        subtasks = _subtasks_from(plan_msg.payload.get("content"))  # type: ignore[attr-defined]
        if subtasks:
            return subtasks

        llm = getattr(planner, "llm", None)
        if llm is not None and hasattr(llm, "json_complete"):
            try:
                plan = llm.json_complete(
                    [{"role": "user", "content": plan_msg.payload.get("content") or query}],  # type: ignore[attr-defined]
                    _PLAN_SCHEMA,
                )
                subtasks = _subtasks_from(plan)
                if subtasks:
                    return subtasks
            except Exception:  # noqa: BLE001 - 结构化输出失败时走兜底
                pass

        content = (plan_msg.payload.get("content") or "").strip()  # type: ignore[attr-defined]
        return [content] if content else []

    def _summarize(self, bucket_results: list[list[tuple[str, object]]]) -> str:
        """把各 worker 的子任务结果拼成给 aggregator 的汇总文本。"""
        parts: list[str] = []
        for bucket in bucket_results:
            for task, msg in bucket:
                content = (msg.payload.get("content") if msg else "") or ""  # type: ignore[attr-defined]
                parts.append(f"子任务：{task}\n结果：{content}")
        return "以下是对各子任务的核查结果：\n\n" + "\n\n".join(parts)


def _subtasks_from(value: object) -> list[str] | None:
    """从 data / content / 结构化 dict 中提取子任务列表；失败返回 None。"""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    if isinstance(value, dict):
        for key in ("subtasks", "tasks", "items", "plan"):
            if isinstance(value.get(key), list):
                return [str(x) for x in value[key] if x is not None]
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        return _subtasks_from(parsed)
    return None
