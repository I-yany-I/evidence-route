"""patterns/debate.py 的单元测试。

用 FakeLLM / FakeAudit / FakeRegistry / FakeApprover 驱动 DebatePattern，
全程不访问网络：覆盖正反方各 2 轮交替发言、pro 第 2 轮可见 con 第 1 轮观点、
judge 终答 = verdict + reasoning，以及非 JSON 裁决兜底。
"""

import asyncio

from agent_collab.core import AgentRuntime, AgentSpec, SharedMemory
from agent_collab.patterns import DebatePattern


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeAudit:
    def __init__(self):
        self.events = []

    def record(self, kind, actor, detail):
        self.events.append((kind, actor, detail))


class FakeLLM:
    """按脚本依序返回 complete 结果，并记录每次 user 提示与累计 usage。"""

    def __init__(self, responses, tokens_per_response=7):
        self.responses = list(responses)
        self.calls = 0
        self.prompts = []
        self.usage_total = 0
        self.tokens_per_response = tokens_per_response

    def complete(self, messages, tools=None, json_schema=None, max_tokens=None):
        self.calls += 1
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        self.prompts.append(user)
        resp = self.responses.pop(0)
        usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
        self.usage_total += int(usage.get("total_tokens", self.tokens_per_response))
        return resp


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


def make_debate(memory=None, audit=None):
    memory = memory or SharedMemory()
    audit = audit or FakeAudit()
    pro = make_runtime("pro", FakeLLM([
        {"content": "正方观点1", "tool_calls": None},
        {"content": "正方观点2", "tool_calls": None},
    ]), memory, audit)
    con = make_runtime("con", FakeLLM([
        {"content": "反方观点1", "tool_calls": None},
        {"content": "反方观点2", "tool_calls": None},
    ]), memory, audit)
    judge = make_runtime("judge", FakeLLM([
        {"content": '{"verdict": "部分属实", "reasoning": "证据不足"}', "tool_calls": None},
    ]), memory, audit)
    return DebatePattern([pro, con, judge], memory, audit), pro, con, judge


# ---------------------------------------------------------------------------
# 正反方各 2 轮交替
# ---------------------------------------------------------------------------
def test_debate_two_rounds_alternating():
    pattern, pro, con, _ = make_debate()
    asyncio.run(pattern.run("论题"))

    assert pro.llm.calls == 2
    assert con.llm.calls == 2
    # 派发顺序：pro → con → pro → con
    messages = [e for e in pattern.audit.events if e[0] == "message"]
    assert [m[2]["to"] for m in messages] == ["pro", "con", "pro", "con"]


# ---------------------------------------------------------------------------
# pro 第 2 轮能看到 con 第 1 轮观点
# ---------------------------------------------------------------------------
def test_debate_pro_second_round_sees_con_first_view():
    pattern, pro, con, _ = make_debate()
    asyncio.run(pattern.run("论题"))

    # pro 的两次发言：第 1 次是论题，第 2 次应包含 con 第 1 轮观点
    assert "论题" in pro.llm.prompts[0]
    assert "反方观点1" in pro.llm.prompts[1]
    # con 第 1 轮看到 pro 第 1 轮观点
    assert "正方观点1" in con.llm.prompts[0]


# ---------------------------------------------------------------------------
# judge 终答 = verdict + reasoning
# ---------------------------------------------------------------------------
def test_debate_judge_verdict_and_reasoning():
    pattern, _, _, _ = make_debate()
    result = asyncio.run(pattern.run("论题"))

    assert result.answer == "部分属实\n证据不足"
    # judge 决策记录为 decision 事件
    decisions = [e for e in pattern.audit.events if e[0] == "decision"]
    assert len(decisions) == 1
    assert decisions[0][1] == "judge"


# ---------------------------------------------------------------------------
# 非 JSON 裁决兜底
# ---------------------------------------------------------------------------
def test_debate_judge_non_json_falls_back_to_raw():
    memory = SharedMemory()
    audit = FakeAudit()
    pro = make_runtime("pro", FakeLLM([
        {"content": "p1", "tool_calls": None}, {"content": "p2", "tool_calls": None},
    ]), memory, audit)
    con = make_runtime("con", FakeLLM([
        {"content": "c1", "tool_calls": None}, {"content": "c2", "tool_calls": None},
    ]), memory, audit)
    judge = make_runtime("judge", FakeLLM([
        {"content": "直接裁决：真实", "tool_calls": None},
    ]), memory, audit)

    pattern = DebatePattern([pro, con, judge], memory, audit)
    result = asyncio.run(pattern.run("论题"))
    assert result.answer == "直接裁决：真实"
