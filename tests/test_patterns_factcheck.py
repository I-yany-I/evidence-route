"""patterns/factcheck.py 的单元测试（组合模式，全 mock 不联网）。"""

import asyncio

from agent_collab.core import AgentRuntime, AgentSpec, SharedMemory
from agent_collab.patterns import FactCheckPattern


class FakeAudit:
    def __init__(self):
        self.events = []

    def record(self, kind, actor, detail):
        self.events.append((kind, actor, detail))


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts = []
        self.usage_total = 0

    def complete(self, messages, tools=None, json_schema=None, max_tokens=None):
        self.calls += 1
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        self.prompts.append(user)
        resp = self.responses.pop(0)
        self.usage_total += 1
        return resp

    def json_complete(self, messages, json_schema, max_tokens=None):
        return self.responses.pop(0)


class FakeRegistry:
    def openai_schemas(self, names):
        return []

    async def execute(self, name, args, approver):
        raise AssertionError("本测试不应调用工具")


class FakeApprover:
    async def request(self, tool_name, args):
        return True


def make_runtime(aid, llm, memory, audit):
    return AgentRuntime(AgentSpec(id=aid, role="r", goal="g"), llm,
                        FakeRegistry(), memory, audit, FakeApprover())


def make_pattern():
    memory = SharedMemory()
    audit = FakeAudit()
    planner = make_runtime("planner", FakeLLM([
        {"content": '{"subtasks": ["事实点A"]}', "tool_calls": None},
    ]), memory, audit)
    worker = make_runtime("worker-1", FakeLLM([
        {"content": "核查结论：支持，有官方来源", "tool_calls": None},
    ]), memory, audit)
    editor = make_runtime("editor", FakeLLM([
        {"content": "中间汇总（忽略）", "tool_calls": None},
        {"content": "最终报告：判定虚假，证据如下……", "tool_calls": None},
    ]), memory, audit)
    pro = make_runtime("pro", FakeLLM([
        {"content": "正方1", "tool_calls": None}, {"content": "正方2", "tool_calls": None},
    ]), memory, audit)
    con = make_runtime("con", FakeLLM([
        {"content": "反方1", "tool_calls": None}, {"content": "反方2", "tool_calls": None},
    ]), memory, audit)
    judge = make_runtime("judge", FakeLLM([
        {"content": '{"verdict": "虚假", "reasoning": "官方已辟谣"}', "tool_calls": None},
    ]), memory, audit)
    team = [planner, worker, editor, pro, con, judge]
    config = {
        "planner_prefix": "planner", "worker_prefix": "worker",
        "aggregator_prefix": "editor", "pro_prefix": "pro",
        "con_prefix": "con", "judge_prefix": "judge", "editor_prefix": "editor",
    }
    return FactCheckPattern(team, memory, audit, config), memory, audit, editor, judge


def test_factcheck_composes_all_stages_and_returns_editor_report():
    pattern, memory, audit, editor, _judge = make_pattern()
    result = asyncio.run(pattern.run("论题"))

    # 终答 = 编辑的最终报告
    assert result.answer == "最终报告：判定虚假，证据如下……"
    # 三个阶段 + 组合本身都有 pattern_start 审计事件（actor 为模式名）
    actors = [e[1] for e in audit.events if e[0] == "pattern_start"]
    assert "factcheck" in actors
    assert "parallel" in actors
    assert "debate" in actors
    # 阶段 1 的事实已写入共享记忆
    assert any(f.claim == "事实点A" and f.verdict == "支持" for f in memory.facts())


def test_factcheck_editor_receives_judge_verdict():
    pattern, _memory, _audit, editor, judge = make_pattern()
    asyncio.run(pattern.run("论题"))

    # 编辑的第二次任务（最终撰写）应包含 judge 裁决
    final_task = editor.llm.prompts[-1]
    assert "裁判裁决" in final_task
    assert "虚假" in final_task
    assert "官方已辟谣" in final_task
    # judge 只被调用一次（裁决）
    assert judge.llm.calls == 1
