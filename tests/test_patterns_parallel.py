"""patterns/parallel.py 的单元测试。

用 FakeLLM / FakeAudit / FakeRegistry / FakeApprover 驱动 ParallelPattern，
全程不访问网络：覆盖子任务轮流分配、asyncio.gather 真并发（时间窗口重叠证据）、
facts 合并、非 JSON 兜底、终答，以及 build_team 的 replicas 复制与 run_pattern 工厂。
"""

import asyncio
import json
import time

from agent_collab.core import AgentRuntime, AgentSpec, Fact, SharedMemory
from agent_collab.patterns import (
    DebatePattern,
    ParallelPattern,
    PipelinePattern,
    SupervisorPattern,
    build_team,
    run_pattern,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeAudit:
    def __init__(self):
        self.events = []

    def record(self, kind, actor, detail):
        self.events.append((kind, actor, detail))


class FakeLLM:
    """普通 Fake：按脚本返回；累计 usage_total 与调用次数。"""

    def __init__(self, responses, tokens_per_response=7):
        self.responses = list(responses)
        self.calls = 0
        self.usage_total = 0
        self.tokens_per_response = tokens_per_response

    def complete(self, messages, tools=None, json_schema=None, max_tokens=None):
        self.calls += 1
        resp = self.responses.pop(0)
        usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
        self.usage_total += int(usage.get("total_tokens", self.tokens_per_response))
        return resp


class FakeWorkerLLM:
    """记录每次 complete 收到任务与起止时间戳；可注入 delay 制造阻塞窗口。"""

    def __init__(self, content, delay=0.0):
        self.content = content
        self.delay = delay
        self.received = []
        self.windows = []
        self.calls = 0
        self.usage_total = 0

    def complete(self, messages, tools=None, json_schema=None, max_tokens=None):
        task = next((m["content"] for m in messages if m["role"] == "user"), "")
        self.received.append(task)
        start = time.monotonic()
        if self.delay:
            time.sleep(self.delay)
        end = time.monotonic()
        self.windows.append((start, end))
        self.calls += 1
        self.usage_total += 10
        return {"content": self.content, "tool_calls": None}


class FakeRegistry:
    def openai_schemas(self, names):
        return []

    async def execute(self, name, args, approver):
        raise AssertionError("本测试不应调用工具")


class FakeApprover:
    async def request(self, tool_name, args):
        return True


def make_runtime(aid, llm, memory, audit):
    spec = AgentSpec(id=aid, role="角色", goal="目标")
    return AgentRuntime(spec, llm, FakeRegistry(), memory, audit, FakeApprover())


def make_parallel(planner_content, worker_specs, aggregator_content, memory=None, audit=None):
    """构造 planner + 若干 worker + aggregator 的并行模式。"""
    memory = memory or SharedMemory()
    audit = audit or FakeAudit()
    team = [
        make_runtime("planner", FakeLLM([{"content": planner_content, "tool_calls": None}]), memory, audit),
    ]
    for wid, llm in worker_specs:
        team.append(make_runtime(wid, llm, memory, audit))
    team.append(make_runtime("aggregator", FakeLLM([{"content": aggregator_content, "tool_calls": None}]), memory, audit))
    return ParallelPattern(team, memory, audit)


# ---------------------------------------------------------------------------
# 子任务轮流分配
# ---------------------------------------------------------------------------
def test_parallel_round_robin_dispatch():
    w1 = FakeWorkerLLM("r")
    w2 = FakeWorkerLLM("r")
    pattern = make_parallel(
        planner_content=json.dumps({"subtasks": ["t0", "t1", "t2"]}, ensure_ascii=False),
        worker_specs=[("worker-1", w1), ("worker-2", w2)],
        aggregator_content="汇总",
    )
    asyncio.run(pattern.run("q"))

    # 3 个子任务轮流分配：worker-1 得 t0/t2，worker-2 得 t1（各自内部顺序确定）
    assert w1.received == ["t0", "t2"]
    assert w2.received == ["t1"]


# ---------------------------------------------------------------------------
# 真并发证据：两个 worker 的执行窗口有重叠
# ---------------------------------------------------------------------------
def test_parallel_workers_execute_concurrently():
    w1 = FakeWorkerLLM("r0", delay=0.1)
    w2 = FakeWorkerLLM("r1", delay=0.1)
    pattern = make_parallel(
        planner_content=json.dumps({"subtasks": ["t0", "t1"]}, ensure_ascii=False),
        worker_specs=[("worker-1", w1), ("worker-2", w2)],
        aggregator_content="汇总",
    )
    asyncio.run(pattern.run("q"))

    assert len(w1.windows) == 1 and len(w2.windows) == 1
    (s1, e1), (s2, e2) = w1.windows[0], w2.windows[0]
    overlap = min(e1, e2) - max(s1, s2)
    # 两 worker 的时间窗口必须重叠（overlap > 0），证明是 gather 真并发而非串行 await
    assert overlap > 0, f"执行窗口未重叠：w1={w1.windows[0]} w2={w2.windows[0]}"


# ---------------------------------------------------------------------------
# facts 合并
# ---------------------------------------------------------------------------
def test_parallel_facts_merged_from_worker_memories():
    shared = SharedMemory()
    audit = FakeAudit()

    mem1 = SharedMemory()
    mem1.record_fact(Fact(claim="事实A", verdict="支持", evidence="e1", sources=[], by="worker-1"))
    mem2 = SharedMemory()
    mem2.record_fact(Fact(claim="事实B", verdict="反对", evidence="e2", sources=[], by="worker-2"))

    team = [
        make_runtime("planner", FakeLLM([{"content": json.dumps({"subtasks": ["t0", "t1"]}), "tool_calls": None}]), shared, audit),
        make_runtime("worker-1", FakeLLM([{"content": "r0", "tool_calls": None}]), mem1, audit),
        make_runtime("worker-2", FakeLLM([{"content": "r1", "tool_calls": None}]), mem2, audit),
        make_runtime("aggregator", FakeLLM([{"content": "汇总", "tool_calls": None}]), shared, audit),
    ]
    pattern = ParallelPattern(team, shared, audit)
    result = asyncio.run(pattern.run("q"))

    claims = {f.claim for f in result.facts}
    assert claims == {"事实A", "事实B"}


# ---------------------------------------------------------------------------
# 鲁棒性：非 JSON 兜底
# ---------------------------------------------------------------------------
def test_parallel_fallback_to_single_task_on_non_json_plan():
    w1 = FakeWorkerLLM("r0")
    pattern = make_parallel(
        planner_content="就整体做一次核查（非 JSON）",
        worker_specs=[("worker-1", w1)],
        aggregator_content="汇总",
    )
    result = asyncio.run(pattern.run("q"))

    # 非 JSON 无法拆解 → 整体作为单个子任务
    assert w1.received == ["就整体做一次核查（非 JSON）"]
    assert result.answer == "汇总"


# ---------------------------------------------------------------------------
# 终答 = aggregator content
# ---------------------------------------------------------------------------
def test_parallel_aggregator_produces_final_answer():
    pattern = make_parallel(
        planner_content=json.dumps({"subtasks": ["t0", "t1"]}, ensure_ascii=False),
        worker_specs=[("worker-1", FakeWorkerLLM("r0")), ("worker-2", FakeWorkerLLM("r1"))],
        aggregator_content="核查结论：部分属实",
    )
    result = asyncio.run(pattern.run("q"))
    assert result.answer == "核查结论：部分属实"


# ---------------------------------------------------------------------------
# build_team 的 replicas 复制
# ---------------------------------------------------------------------------
def test_build_team_replicas_suffix_ids():
    specs = [
        AgentSpec(id="planner", role="规划员", goal="拆解"),
        AgentSpec(id="fact_checker", role="核查员", goal="核查"),
    ]
    team = build_team(specs, object(), FakeRegistry(), SharedMemory(), FakeAudit(), FakeApprover(),
                      replicas={"fact_checker": 4})
    assert [a.spec.id for a in team] == [
        "planner", "fact_checker-1", "fact_checker-2", "fact_checker-3", "fact_checker-4",
    ]
    # 复制的实例沿用同 spec 的 role/goal
    assert all(a.spec.role == "核查员" for a in team[1:])
    assert all(a.spec.goal == "核查" for a in team[1:])


# ---------------------------------------------------------------------------
# run_pattern 工厂
# ---------------------------------------------------------------------------
def test_run_pattern_factory_returns_correct_class():
    specs = {
        "planner": AgentSpec(id="planner", role="r", goal="g"),
        "worker": AgentSpec(id="worker", role="r", goal="g"),
        "aggregator": AgentSpec(id="aggregator", role="r", goal="g"),
        "supervisor": AgentSpec(id="supervisor", role="r", goal="g"),
        "pro": AgentSpec(id="pro", role="r", goal="g"),
        "con": AgentSpec(id="con", role="r", goal="g"),
        "judge": AgentSpec(id="judge", role="r", goal="g"),
    }
    base = dict(llm=object(), registry=FakeRegistry(), memory=SharedMemory(),
                audit=FakeAudit(), approver=FakeApprover())
    assert isinstance(run_pattern("pipeline", specs, **base), PipelinePattern)
    assert isinstance(run_pattern("parallel", specs, **base), ParallelPattern)
    assert isinstance(run_pattern("supervisor", specs, **base), SupervisorPattern)
    assert isinstance(run_pattern("debate", specs, **base), DebatePattern)
