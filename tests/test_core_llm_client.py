"""core/llm_client.py 的单元测试。

全部 mock openai SDK（monkeypatch _client），不访问真实网络：
覆盖 YAML 配置加载、请求组装、结果归一化、tool_calls 容错、重试与结构化输出。
"""

import json
from types import SimpleNamespace

import pytest

from agent_collab.core import LLMClient, LLMConfig, LLMError, load_llm_config


# ---------------------------------------------------------------------------
# Fake openai SDK：client.chat.completions.create(**kwargs)
# ---------------------------------------------------------------------------

def make_response(content=None, tool_calls=None, usage=None):
    """构造一个形如 openai SDK ChatCompletion 的假对象。"""
    if tool_calls:
        tcs = [
            SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))
            for name, arguments in tool_calls
        ]
        message = SimpleNamespace(content=content, tool_calls=tcs)
    else:
        message = SimpleNamespace(content=content, tool_calls=None)
    usage_obj = SimpleNamespace(total_tokens=usage) if usage is not None else None
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage_obj)


class FakeCompletions:
    """create() 依序弹出响应；元素为 Exception 时抛错（用于重试测试）。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeAPIStatusError(Exception):
    """模拟 openai SDK 的 APIStatusError：带 status_code 的瞬态/非瞬态错误。"""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class FakeOpenAI:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def make_client(responses, max_retries=3):
    """构造 LLMClient 并把其 _client 替换为 fake。"""
    cfg = LLMConfig(base_url="https://api.example.com", api_key="sk-test", model="gpt-5.5",
                    max_retries=max_retries)
    client = LLMClient(cfg)
    client._client = FakeOpenAI(responses)  # noqa: SLF001 - 测试注入 fake SDK
    return client


# ---------------------------------------------------------------------------
# load_llm_config
# ---------------------------------------------------------------------------

def test_load_llm_config_reads_yaml_and_env_key(tmp_path, monkeypatch):
    """读取 YAML；api_key 以 $ 开头时从环境变量取值。"""
    monkeypatch.setenv("VISION_API_KEY", "secret-123")
    path = tmp_path / "llm.yaml"
    path.write_text(
        "base_url: https://api.ssstoken.net\n"
        "api_key: $VISION_API_KEY\n"
        "model: gpt-5.5\n"
        "temperature: 0.1\n"
        "timeout_s: 30.0\n"
        "max_retries: 2\n",
        encoding="utf-8",
    )
    cfg = load_llm_config(path)
    assert cfg.base_url == "https://api.ssstoken.net"
    assert cfg.api_key == "secret-123"
    assert cfg.model == "gpt-5.5"
    assert cfg.temperature == 0.1
    assert cfg.timeout_s == 30.0
    assert cfg.max_retries == 2


def test_load_llm_config_plain_api_key(tmp_path):
    """api_key 不以 $ 开头时原样返回。"""
    path = tmp_path / "llm.yaml"
    path.write_text("base_url: https://x\napi_key: sk-plain\nmodel: m\n", encoding="utf-8")
    assert load_llm_config(path).api_key == "sk-plain"


def test_load_llm_config_missing_env_var(tmp_path, monkeypatch):
    """环境变量未设置时 api_key 回退为空字符串。"""
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    path = tmp_path / "llm.yaml"
    path.write_text("base_url: https://x\napi_key: $NO_SUCH_KEY\nmodel: m\n", encoding="utf-8")
    assert load_llm_config(path).api_key == ""


def test_load_llm_config_defaults(tmp_path):
    """未提供的可选字段使用 dataclass 默认值。"""
    path = tmp_path / "llm.yaml"
    path.write_text("base_url: https://x\napi_key: k\nmodel: m\n", encoding="utf-8")
    cfg = load_llm_config(path)
    assert cfg.temperature == 0.2
    assert cfg.timeout_s == 120.0
    assert cfg.max_retries == 3


def test_client_normalizes_base_url_with_v1(monkeypatch):
    """base_url 不含 /v1 时自动补上，使端点形如 {base_url}/v1/chat/completions。"""
    captured = {}

    class FakeOpenAICtor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("agent_collab.core.llm_client.OpenAI", FakeOpenAICtor)
    cfg = LLMConfig(base_url="https://api.example.com", api_key="k", model="m")
    LLMClient(cfg)
    assert captured["base_url"] == "https://api.example.com/v1"


def test_client_keeps_existing_v1_suffix(monkeypatch):
    """base_url 已含 /v1（或末尾斜杠）时保持不变。"""
    captured = {}

    class FakeOpenAICtor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("agent_collab.core.llm_client.OpenAI", FakeOpenAICtor)
    cfg = LLMConfig(base_url="https://api.example.com/v1/", api_key="k", model="m")
    LLMClient(cfg)
    assert captured["base_url"] == "https://api.example.com/v1"


# ---------------------------------------------------------------------------
# complete
# ---------------------------------------------------------------------------

def test_complete_builds_request_and_extracts_content():
    """complete 组装请求并把 SDK 响应归一化为契约 dict。"""
    client = make_client([make_response(content="你好")])
    result = client.complete([{"role": "user", "content": "hi"}])
    assert result == {"content": "你好", "tool_calls": None}
    call = client._client.chat.completions.calls[0]  # noqa: SLF001
    assert call["model"] == "gpt-5.5"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["temperature"] == 0.2


def test_complete_passes_tools_and_max_tokens():
    """tools / max_tokens 透传给 SDK。"""
    client = make_client([make_response(content="ok")])
    tools = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
    client.complete([{"role": "user", "content": "q"}], tools=tools, max_tokens=64)
    call = client._client.chat.completions.calls[0]  # noqa: SLF001
    assert call["tools"] == tools
    assert call["max_tokens"] == 64


def test_complete_passes_usage_through():
    """SDK 响应带 usage 时透传 total_tokens（供 patterns token 计量）。"""
    client = make_client([make_response(content="ok", usage=1234)])
    result = client.complete([{"role": "user", "content": "q"}])
    assert result["usage"] == {"total_tokens": 1234}


def test_complete_parses_tool_calls_arguments():
    """tool_calls 的 arguments 为 JSON 字符串时解析为 dict。"""
    client = make_client([
        make_response(content=None, tool_calls=[("web_search", '{"query": "燃油车禁令"}')]),
    ])
    result = client.complete([{"role": "user", "content": "q"}])
    assert result["content"] is None
    assert result["tool_calls"] == [{"name": "web_search", "arguments": {"query": "燃油车禁令"}}]


def test_complete_tolerates_malformed_tool_arguments():
    """arguments 非法 JSON 时容错为 {}，不抛异常。"""
    client = make_client([
        make_response(content=None, tool_calls=[("bad_tool", "not-json")]),
    ])
    result = client.complete([{"role": "user", "content": "q"}])
    assert result["tool_calls"] == [{"name": "bad_tool", "arguments": {}}]


def test_complete_retries_then_succeeds():
    """前两次瞬态失败（500）、第三次成功：重试后正常返回。"""
    client = make_client(
        [FakeAPIStatusError("boom", 500), FakeAPIStatusError("boom", 500), make_response(content="最终")],
        max_retries=3,
    )
    result = client.complete([{"role": "user", "content": "q"}])
    assert result["content"] == "最终"
    assert len(client._client.chat.completions.calls) == 3  # noqa: SLF001


def test_complete_raises_llm_error_after_max_retries():
    """重试 max_retries 次仍瞬态失败则抛 LLMError。"""
    client = make_client([FakeAPIStatusError("boom", 500)] * 4, max_retries=3)
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "q"}])
    # 尝试次数 = 1 + max_retries = 4
    assert len(client._client.chat.completions.calls) == 4  # noqa: SLF001


def test_complete_does_not_retry_non_transient_error():
    """4xx 错误（如 400 参数错误）不重试，直接抛 LLMError。"""
    client = make_client([FakeAPIStatusError("bad request", 400)] * 4, max_retries=3)
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "q"}])
    assert len(client._client.chat.completions.calls) == 1  # noqa: SLF001


# ---------------------------------------------------------------------------
# json_complete
# ---------------------------------------------------------------------------

def test_json_complete_returns_parsed_dict():
    """json_complete 强制结构化输出并解析 JSON。"""
    client = make_client([make_response(content='{"verdict": "支持"}')])
    result = client.json_complete([{"role": "user", "content": "q"}], {"name": "s", "schema": {}})
    assert result == {"verdict": "支持"}
    call = client._client.chat.completions.calls[0]  # noqa: SLF001
    assert call["response_format"]["type"] == "json_schema"


def test_json_complete_retries_once_on_parse_failure():
    """首次返回非法 JSON 时重试一次，第二次成功。"""
    client = make_client([
        make_response(content="not-json"),
        make_response(content='{"ok": true}'),
    ])
    result = client.json_complete([{"role": "user", "content": "q"}], {"name": "s", "schema": {}})
    assert result == {"ok": True}
    assert len(client._client.chat.completions.calls) == 2  # noqa: SLF001


def test_json_complete_raises_after_retry():
    """两次都解析失败则抛 LLMError。"""
    client = make_client([make_response(content="bad"), make_response(content="bad")])
    with pytest.raises(LLMError):
        client.json_complete([{"role": "user", "content": "q"}], {"name": "s", "schema": {}})
    assert len(client._client.chat.completions.calls) == 2  # noqa: SLF001
