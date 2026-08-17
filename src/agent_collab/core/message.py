"""消息协议信封（A2A 风格）。

Agent 间通过 Message 交换信息：由 new_message 统一生成唯一 id 与时间戳，
type 用字符串枚举约束，payload 按消息类型约定结构，供各协作模式读写。
"""

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel


class MessageType(str, Enum):
    """消息类型：任务下发 / 结果返回 / 提问索取 / 裁决结论。"""

    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    DECISION = "decision"


class Message(BaseModel):
    """消息信封。

    Attributes:
        id: uuid4 hex，全局唯一。
        sender: 发送方 agent id（"system" 表示编排器）。
        recipient: 接收方 agent id 或 "all"（组播）。
        type: 消息类型。
        payload: 业务内容；TASK 约定 {"task", "context"}，RESULT 约定 {"content", "data"}。
        ts: ISO8601 时间戳。
    """

    id: str
    sender: str
    recipient: str
    type: MessageType
    payload: dict
    ts: str


def new_message(sender: str, recipient: str, type_: MessageType, payload: dict) -> Message:
    """构造一条带唯一 id 与当前时间戳的消息。"""
    return Message(
        id=uuid4().hex,
        sender=sender,
        recipient=recipient,
        type=type_,
        payload=payload,
        ts=datetime.now(timezone.utc).isoformat(),
    )
