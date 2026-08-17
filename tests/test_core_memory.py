"""core/memory.py 的单元测试。

只验证共享记忆：短期消息缓冲（post/history 及 limit）、长期事实库（record_fact/facts）
与快照（snapshot），不涉及网络。
"""

from agent_collab.core import Fact, MessageType, SharedMemory, new_message


def test_post_and_history_order():
    """history 按 post 顺序返回消息。"""
    mem = SharedMemory()
    m1 = new_message("a", "b", MessageType.TASK, {"task": "t1"})
    m2 = new_message("b", "a", MessageType.RESULT, {"content": "r1"})
    mem.post(m1)
    mem.post(m2)
    assert mem.history() == [m1, m2]


def test_history_limit_keeps_most_recent():
    """history(limit) 只保留最近 N 条。"""
    mem = SharedMemory()
    for i in range(10):
        mem.post(new_message("a", "b", MessageType.QUERY, {"i": i}))
    hist = mem.history(limit=3)
    assert len(hist) == 3
    assert [m.payload["i"] for m in hist] == [7, 8, 9]


def test_record_and_list_facts():
    """record_fact 写入事实，facts 返回全部事实。"""
    mem = SharedMemory()
    f = Fact(claim="声明 X 为真", verdict="支持", evidence="官方通报", sources=["s1"], by="worker")
    mem.record_fact(f)
    facts = mem.facts()
    assert facts == [f]
    assert facts[0].verdict == "支持"


def test_fact_model_fields():
    """Fact 模型字段齐全且 verdict 支持契约约定取值。"""
    f = Fact(claim="c", verdict="存疑", evidence="", sources=[], by="x")
    assert f.claim == "c"
    assert f.verdict in {"支持", "反对", "存疑", "证据不足"}
    assert f.sources == []


def test_snapshot_shape():
    """snapshot 返回 {'messages', 'facts'} 两个键，且内容可序列化。"""
    mem = SharedMemory()
    mem.post(new_message("a", "b", MessageType.TASK, {"task": "t"}))
    mem.record_fact(Fact(claim="c", verdict="证据不足", evidence="", sources=[], by="x"))
    snap = mem.snapshot()
    assert set(snap.keys()) == {"messages", "facts"}
    assert len(snap["messages"]) == 1
    assert len(snap["facts"]) == 1
    # 序列化为普通 dict，供审计/回放直接使用
    assert snap["messages"][0]["payload"]["task"] == "t"
    assert snap["facts"][0]["claim"] == "c"
