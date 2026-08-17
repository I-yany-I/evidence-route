"""runtime/audit.py 的单元测试。

覆盖 record / events 顺序 / replay 可序列化 / to_markdown 报告内容 / 空日志。
"""

from datetime import datetime

from agent_collab.runtime import AuditEvent, AuditLog


def test_record_appends_event_with_iso_timestamp():
    log = AuditLog()
    log.record("tool_call", "worker", {"tool": "web_search"})
    events = log.events()
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, AuditEvent)
    assert ev.kind == "tool_call"
    assert ev.actor == "worker"
    assert ev.detail == {"tool": "web_search"}
    # ts 必须是可解析的 ISO8601 时间
    datetime.fromisoformat(ev.ts)


def test_events_returns_in_order():
    log = AuditLog()
    log.record("agent_start", "a", {"task": "t"})
    log.record("agent_result", "a", {"content": "c"})
    kinds = [e.kind for e in log.events()]
    assert kinds == ["agent_start", "agent_result"]


def test_replay_returns_serializable_dicts():
    log = AuditLog()
    log.record("decision", "judge", {"verdict": "支持"})
    replay = log.replay()
    assert len(replay) == 1
    item = replay[0]
    assert set(item) == {"ts", "kind", "actor", "detail"}
    assert item["kind"] == "decision"
    assert item["actor"] == "judge"
    assert item["detail"] == {"verdict": "支持"}


def test_to_markdown_contains_event_details():
    log = AuditLog()
    log.record("tool_call", "worker", {"tool": "web_search"})
    log.record("approval", "worker", {"tool": "python_repl", "allowed": True})
    md = log.to_markdown()
    assert md.startswith("# 审计日志")
    assert "tool_call" in md
    assert "approval" in md
    assert "worker" in md
    assert "web_search" in md
    assert "python_repl" in md


def test_to_markdown_empty_log():
    log = AuditLog()
    md = log.to_markdown()
    assert md.startswith("# 审计日志")
