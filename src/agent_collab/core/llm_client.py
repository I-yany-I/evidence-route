"""OpenAI 兼容 LLM 客户端：超时 / 重试 / 结构化输出。

将 openai SDK 的响应归一化为契约 dict：{"content": str|None, "tool_calls": [...]|None}，
对 tool_calls.arguments（可能是 JSON 字符串）做容错解析。
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import APIConnectionError, APITimeoutError, OpenAI

_RETRY_BACKOFF_S = 0.1  # 重试退避基数（秒），测试友好


def _is_transient_error(exc: Exception) -> bool:
    """仅网络/超时/服务端(≥500)/限流(429)错误值得重试；4xx 与编程错误直接失败。"""
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and (status >= 500 or status == 429)


@dataclass
class LLMConfig:
    """LLM 连接配置。api_key 为从环境变量解析后的实际值。"""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout_s: float = 120.0
    max_retries: int = 3


class LLMError(Exception):
    """LLM 调用失败（重试耗尽 / 结构化输出解析失败）。"""


class LLMClient:
    """调用 {base_url}/v1/chat/completions 的 OpenAI 兼容客户端。"""

    def __init__(self, config: LLMConfig):
        self.config = config
        # max_retries=0：重试交由本类统一管理，避免与 SDK 内部重试叠加；
        # base_url 归一化为 /v1 结尾，使最终端点形如 {base_url}/v1/chat/completions
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=_ensure_v1(config.base_url),
            timeout=config.timeout_s,
            max_retries=0,
        )

    # ------------------------------------------------------------------ 对外

    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 json_schema: dict | None = None, max_tokens: int | None = None) -> dict:
        """调用 chat/completions 并归一化为契约 dict；失败重试 max_retries 次后抛 LLMError。"""
        kwargs: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = self._chat_with_retry(kwargs)
        return self._to_result(resp)

    def json_complete(self, messages: list[dict], json_schema: dict,
                      max_tokens: int | None = None) -> dict:
        """强制结构化输出；解析失败重试一次后抛 LLMError。"""
        result = self.complete(messages, json_schema=json_schema, max_tokens=max_tokens)
        try:
            return self._parse_json(result["content"])
        except LLMError:
            # 解析失败：重试一次
            result = self.complete(messages, json_schema=json_schema, max_tokens=max_tokens)
            return self._parse_json(result["content"])

    # ------------------------------------------------------------------ 内部

    def _chat_with_retry(self, kwargs: dict):
        attempts = self.config.max_retries + 1
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - 仅瞬态错误重试
                last_exc = exc
                if not _is_transient_error(exc):
                    break
                if i < attempts - 1:
                    time.sleep(_RETRY_BACKOFF_S * (i + 1))  # 简单退避，测试友好
        raise LLMError(f"LLM 调用失败：{last_exc}")

    @staticmethod
    def _parse_json(content: str | None) -> dict:
        """解析结构化输出；为空或非法 JSON 时抛 LLMError。"""
        if content is None:
            raise LLMError("结构化输出为空")
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMError(f"结构化输出解析失败：{exc}") from exc

    @staticmethod
    def _parse_arguments(raw) -> dict:
        """容错解析 tool_calls.arguments（可能是 JSON 字符串，也可能是 dict）。"""
        if raw is None or raw == "":
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def _to_result(self, resp) -> dict:
        """把 SDK 响应归一化为契约 dict；透传 usage 供 token 计量。"""
        message = resp.choices[0].message
        tool_calls = None
        if getattr(message, "tool_calls", None):
            tool_calls = [
                {"name": tc.function.name, "arguments": self._parse_arguments(tc.function.arguments)}
                for tc in message.tool_calls
            ]
        result = {"content": getattr(message, "content", None), "tool_calls": tool_calls}
        usage = getattr(resp, "usage", None)
        total = getattr(usage, "total_tokens", None)
        if total is not None:
            result["usage"] = {"total_tokens": int(total)}
        return result


def _ensure_v1(base_url: str) -> str:
    """确保 base_url 以 /v1 结尾（openai SDK 不会自动补 /v1）。"""
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def load_llm_config(path: Path) -> LLMConfig:
    """读 config/llm.yaml；api_key 以 '$' 开头时视为环境变量名并取值。

    可选字段仅在 YAML 提供时才传入，dataclass 默认值是唯一事实源。
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    api_key = data.get("api_key", "")
    if isinstance(api_key, str) and api_key.startswith("$"):
        api_key = os.environ.get(api_key[1:], "")
    kwargs: dict = {
        "base_url": data["base_url"],
        "api_key": api_key,
        "model": data["model"],
    }
    for field in ("temperature", "timeout_s", "max_retries"):
        if field in data:
            kwargs[field] = data[field]
    return LLMConfig(**kwargs)
