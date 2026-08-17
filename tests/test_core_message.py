"""core/message.py 的单元测试。

只验证消息信封协议：类型枚举、id/ts 生成、payload 结构约定，不涉及网络。
"""

from datetime import datetime

import pytest
import pydantic

from agent_collab.core import Message, MessageType, new_message


def test_message_type_enum_values():
    """四种消息类型的字符串取值与契约一致。"""
    assert MessageType.TASK == "task"
    assert MessageType.RESULT == "result"
    assert MessageType.QUERY == "query"
    assert MessageType.DECISION == "decision"


def test_new_message_generates_id_and_ts():
    """new_message 自动生成 uuid4 hex id 与 ISO8601 时间戳。"""
    msg = new_message("planner", "worker", MessageType.TASK, {"task": "x", "context": {}})
    assert isinstance(msg.id, str)
    assert len(msg.id) == 32  # uuid4 hex 为 32 位
    # ts 必须可被解析为 ISO8601 时间
    datetime.fromisoformat(msg.ts)


def test_new_message_ids_are_unique():
    """不同消息的 id 互不相同。"""
    a = new_message("a", "b", MessageType.QUERY, {})
    b = new_message("a", "b", MessageType.QUERY, {})
    assert a.id != b.id


def test_message_type_coerces_from_string():
    """pydantic 将字符串自动转成 MessageType 枚举。"""
    msg = Message(
        id="1", sender="s", recipient="r", type="task",
        payload={}, ts="2026-01-01T00:00:00+00:00",
    )
    assert msg.type is MessageType.TASK


def test_task_payload_structure():
    """TASK 消息 payload 约定为 {'task', 'context'}。"""
    msg = new_message(
        "system", "worker", MessageType.TASK,
        {"task": "核查声明", "context": {"claim": "X"}},
    )
    assert msg.payload["task"] == "核查声明"
    assert msg.payload["context"] == {"claim": "X"}


def test_result_payload_structure():
    """RESULT 消息 payload 约定为 {'content', 'data'}。"""
    msg = new_message(
        "worker", "system", MessageType.RESULT,
        {"content": "结论", "data": {"verdict": "支持"}},
    )
    assert msg.payload["content"] == "结论"
    assert msg.payload["data"]["verdict"] == "支持"


def test_message_requires_required_fields():
    """缺少必填字段时应触发校验错误。"""
    with pytest.raises(pydantic.ValidationError):
        Message(id="1", sender="s")  # 缺 recipient/type/payload/ts
