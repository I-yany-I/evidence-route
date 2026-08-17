"""协作模式公共基类与团队构造。

四种模式（pipeline/parallel/supervisor/debate）统一从 ``run(query)`` 返回
:class:`AgentResult`，内含终答、审计轨迹、合并后的事实与每个 agent 的 token 消耗。
团队构造（含 replicas 复制）与模式工厂 :func:`run_pattern` 也在此定义。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..core import AgentRuntime, AgentSpec, Fact, SharedMemory

if TYPE_CHECKING:
    from ..runtime.audit import AuditLog


@dataclass
class AgentResult:
    """一次协作模式的统一产出。

    Attributes:
        answer: 终答（各模式约定不同，见对应实现）。
        audit: 审计日志对象（鸭子类型，``.record(kind, actor, detail)``）。
        facts: 合并去重后的事实列表。
        tokens: {agent_id: 该 agent 的 LLM token 消耗}。
        wall_s: 端到端耗时（time.monotonic 差值，秒）。
    """

    answer: str
    audit: AuditLog
    facts: list[Fact]
    tokens: dict[str, int]
    wall_s: float


class CollaborationPattern(ABC):
    """协作模式抽象基类：持有团队/记忆/审计/配置，子类实现 ``run(query)``。"""

    def __init__(self, team: list[AgentRuntime], memory: SharedMemory,
                 audit: AuditLog, config: dict | None = None) -> None:
        self.team = list(team)
        self.memory = memory
        self.audit = audit
        self.config = dict(config or {})
        self._tokens: dict[str, int] = {}

    @abstractmethod
    async def run(self, query: str) -> AgentResult:
        """执行该协作模式，返回统一结果。"""

    # ------------------------------------------------------------------ 查找
    def _find(self, prefix: str) -> AgentRuntime:
        """按 id 前缀查找首个匹配的 agent。"""
        for agent in self.team:
            if agent.spec.id.startswith(prefix):
                return agent
        raise ValueError(f"未找到 id 前缀为 {prefix!r} 的 agent")

    def _find_all(self, prefix: str) -> list[AgentRuntime]:
        """按 id 前缀查找全部匹配的 agent（保持 team 顺序）。"""
        return [a for a in self.team if a.spec.id.startswith(prefix)]

    def _find_by_id(self, agent_id: str) -> AgentRuntime:
        """按精确 id 查找 agent。"""
        for agent in self.team:
            if agent.spec.id == agent_id:
                return agent
        raise ValueError(f"未找到 id 为 {agent_id!r} 的 agent")

    # ------------------------------------------------------------------ 审计
    def _record_start(self, name: str, query: str) -> None:
        self.audit.record("pattern_start", name, {"query": query})

    def _record_end(self, name: str, answer: str) -> None:
        self.audit.record("pattern_end", name, {"answer": answer})

    def _dispatch(self, agent_id: str, task: str) -> None:
        """记录一次任务派发（message 事件）。"""
        self.audit.record("message", "system", {"to": agent_id, "task": task})

    def _decision(self, actor: str, decision: dict) -> None:
        self.audit.record("decision", actor, decision)

    # ------------------------------------------------------------------ token
    def _record_tokens(self, agent: AgentRuntime) -> None:
        """记录某 agent 的 token 消耗（以 llm 累计值为准，幂等覆盖）。"""
        self._tokens[agent.spec.id] = self._estimate_tokens(agent)

    def _estimate_tokens(self, agent: AgentRuntime) -> int:
        """估算某 agent 的 token 消耗。

        优先取 llm 对象上累计的 ``usage_total``（FakeLLM 每次返回 usage 时累加）；
        否则按 ``config["tokens_per_call"]`` × complete 调用次数估算。
        """
        llm = getattr(agent, "llm", None)
        if llm is None:
            return 0
        usage = getattr(llm, "usage_total", 0) or 0
        if usage:
            return int(usage)
        calls = getattr(llm, "calls", 0)
        n = len(calls) if isinstance(calls, (list, tuple)) else int(calls or 0)
        per_call = self.config.get("tokens_per_call")
        if per_call is not None and n:
            return n * int(per_call)
        return 0

    # ------------------------------------------------------------------ 事实
    def _record_facts(self, agent_id: str, task: str, content: str) -> None:
        """把一次核查结果写入共享事实库（从文本推断判定词）。

        verdict 推断为启发式：按「证据不足→反对→支持→存疑」顺序匹配关键词，
        都未命中时默认「存疑」。evidence 截断 500 字符防事实库膨胀。
        """
        verdict = "存疑"
        for word in ("证据不足", "反对", "支持", "存疑"):
            if word in content:
                verdict = word
                break
        self.memory.record_fact(Fact(
            claim=task,
            verdict=verdict,
            evidence=(content or "").strip()[:500],
            sources=[],
            by=agent_id,
        ))

    def _facts_summary(self) -> str:
        """把共享事实库摘要成文本（供 supervisor/judge 决策使用）。"""
        facts = self.memory.facts()
        if not facts:
            return ""
        return "\n".join(f"- {f.claim}：{f.verdict}（{f.evidence}）" for f in facts)

    def _collect_facts(self) -> list[Fact]:
        """合并 self.memory 与各 team 成员 memory 的事实（按内容去重）。"""
        seen: set[tuple[str, str, str, str]] = set()
        facts: list[Fact] = []
        memories: list[object] = [self.memory] + [getattr(a, "memory", None) for a in self.team]
        for mem in memories:
            if mem is None:
                continue
            for f in mem.facts():  # type: ignore[attr-defined]
                key = (f.claim, f.verdict, f.evidence, f.by)
                if key not in seen:
                    seen.add(key)
                    facts.append(f)
        return facts

    # ------------------------------------------------------------------ 收尾
    def _finish(self, name: str, answer: str, start_s: float) -> AgentResult:
        """记录 pattern_end 并组装统一结果。"""
        self._record_end(name, answer)
        return AgentResult(
            answer=answer,
            audit=self.audit,
            facts=self._collect_facts(),
            tokens=dict(self._tokens),
            wall_s=time.monotonic() - start_s,
        )


class _CountingLLM:
    """按 agent 维度统计 LLM 调用的代理：委托真实 llm，累计调用次数与 usage。

    build_team 给每个 AgentRuntime 包一层独立代理，从而在共享同一底层 llm 时
    仍能按 agent 归因 token 消耗。
    """

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.calls = 0
        self.usage_total = 0

    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 json_schema: dict | None = None, max_tokens: int | None = None) -> dict:
        self.calls += 1
        result = self._delegate.complete(  # type: ignore[attr-defined]
            messages, tools=tools, json_schema=json_schema, max_tokens=max_tokens)
        self._accumulate(result)
        return result

    def json_complete(self, messages: list[dict], json_schema: dict,
                      max_tokens: int | None = None) -> dict:
        self.calls += 1
        return self._delegate.json_complete(  # type: ignore[attr-defined]
            messages, json_schema, max_tokens=max_tokens)

    def _accumulate(self, result: object) -> None:
        if isinstance(result, dict):
            usage = result.get("usage")
            if isinstance(usage, dict):
                self.usage_total += int(usage.get("total_tokens", 0) or 0)


def _make_runtime(spec: AgentSpec, llm: object, registry: object, memory: SharedMemory,
                  audit: object, approver: object) -> AgentRuntime:
    """构造运行时，并为每个运行时包一层计数 LLM 以做 token 归因。"""
    return AgentRuntime(spec, _CountingLLM(llm), registry, memory, audit, approver)


def build_team(specs: list[AgentSpec], llm: object, registry: object, memory: SharedMemory,
               audit: object, approver: object, replicas: dict[str, int] | None = None) -> list[AgentRuntime]:
    """按 specs 构造 AgentRuntime 列表。

    ``replicas`` 中声明的 spec（如 ``{"fact_checker": 4}``）会被复制为
    ``fact_checker-1..4`` 多个运行时实例（同 spec，id 带 ``-N`` 后缀）。
    """
    replicas = replicas or {}
    team: list[AgentRuntime] = []
    for spec in specs:
        count = int(replicas.get(spec.id, 1) or 1)
        if count <= 1:
            team.append(_make_runtime(spec, llm, registry, memory, audit, approver))
        else:
            for i in range(1, count + 1):
                child = replace(spec, id=f"{spec.id}-{i}", tools=list(spec.tools))
                team.append(_make_runtime(child, llm, registry, memory, audit, approver))
    return team


def _coerce_specs(team_specs: object) -> list[AgentSpec]:
    """把 team_specs 归一化为 AgentSpec 列表。

    支持三种形态：dict 含 ``agents`` 键（YAML team 结构）、dict 为
    ``{agent_id: AgentSpec}`` 映射、或 list[AgentSpec] / list[dict]。
    """
    if isinstance(team_specs, dict):
        agents = team_specs.get("agents")
        items = list(agents) if agents is not None else list(team_specs.values())
    else:
        items = list(team_specs)  # type: ignore[arg-type]
    specs: list[AgentSpec] = []
    for item in items:
        if isinstance(item, AgentSpec):
            specs.append(item)
        elif isinstance(item, dict):
            specs.append(AgentSpec(**item))
        else:
            raise TypeError(f"无法识别的 agent 定义：{item!r}")
    return specs


def run_pattern(name: str, team_specs: object, llm: object, registry: object,
                memory: SharedMemory, audit: object, approver: object,
                config: dict | None = None) -> CollaborationPattern:
    """按 name 构造对应协作模式实例。

    ``name`` ∈ {pipeline, parallel, supervisor, debate}；``config`` 可含
    ``replicas``（交给 build_team）及各模式专属参数（max_rounds/rounds/前缀等）。
    """
    config = dict(config or {})
    replicas = config.pop("replicas", None) or {}
    if isinstance(team_specs, dict) and isinstance(team_specs.get("replicas"), dict):
        replicas = dict(team_specs["replicas"])
    specs = _coerce_specs(team_specs)
    team = build_team(specs, llm, registry, memory, audit, approver, replicas)

    if name == "pipeline":
        from .pipeline import PipelinePattern
        return PipelinePattern(team, memory, audit, config)
    if name == "parallel":
        from .parallel import ParallelPattern
        return ParallelPattern(team, memory, audit, config)
    if name == "supervisor":
        from .supervisor import SupervisorPattern
        return SupervisorPattern(team, memory, audit, config)
    if name == "debate":
        from .debate import DebatePattern
        return DebatePattern(team, memory, audit, config)
    raise ValueError(f"未知协作模式：{name!r}（应为 pipeline/parallel/supervisor/debate）")
