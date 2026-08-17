# AgentCollab 接口契约（实现各组件前必读）

本文档是各组件的**唯一接口契约**。实现子 agent 必须严格按此签名实现，不得自行改签名；
跨组件只允许依赖本文档列出的接口。契约变更必须由编排者统一更新本文档。

## 0. 通用约定

- 包名 `agent_collab`，所有代码在 `src/agent_collab/` 下，`__init__.py` 只做 re-export。
- Python 3.11；类型注解 + pydantic 模型；中文 docstring（延续项目二/三风格）。
- 配置一律 YAML；路径用 `pathlib.Path`；无全局可变状态（对象注入）。
- 测试：pytest，文件命名 `tests/test_<模块>.py`；不访问真实网络（LLM/搜索全部 mock）。

## 1. core/llm_client.py

```python
@dataclass
class LLMConfig:
    base_url: str            # 如 https://api.ssstoken.net
    api_key: str             # 从环境变量读取后的实际值
    model: str               # 默认 gpt-5.5
    temperature: float = 0.2
    timeout_s: float = 120.0
    max_retries: int = 3

class LLMClient:
    def __init__(self, config: LLMConfig): ...
    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 json_schema: dict | None = None, max_tokens: int | None = None) -> dict:
        """调用 chat/completions。返回 OpenAI 风格 dict：
        {content: str | None, tool_calls: list[{name, arguments: dict}] | None}。
        失败重试 max_retries 次后抛 LLMError(msg)。"""
    def json_complete(self, messages: list[dict], json_schema: dict, max_tokens: int | None = None) -> dict:
        """强制结构化输出（response_format json_schema），解析失败重试一次后抛 LLMError。"""

def load_llm_config(path: Path) -> LLMConfig:
    """读 config/llm.yaml；api_key 字段若为环境变量名（如 VISION_API_KEY）则从 os.environ 取。"""
```

## 2. core/message.py

```python
class MessageType(str, Enum):
    TASK = "task"          # 下发任务
    RESULT = "result"      # 任务结果
    QUERY = "query"        # 提问/索取信息
    DECISION = "decision"  # 裁决/结论

class Message(BaseModel):
    id: str                # uuid4 hex
    sender: str            # agent id（"system" 表示编排器）
    recipient: str         # agent id，或 "all"（组播）
    type: MessageType
    payload: dict          # 约定：TASK 含 {"task": str, "context": dict}；RESULT 含 {"content": str, "data": dict}
    ts: str                # ISO8601 时间戳

def new_message(sender: str, recipient: str, type_: MessageType, payload: dict) -> Message: ...
```

## 3. core/memory.py

```python
class Fact(BaseModel):
    claim: str             # 待核声明或子断言
    verdict: str           # 支持/反对/存疑/证据不足
    evidence: str          # 论据摘要
    sources: list[str]     # 来源（URL 或样例文件名）
    by: str                # 写入方 agent id

class SharedMemory:
    def __init__(self): ...
    def post(self, msg: Message) -> None            # 短期缓冲（全团队可见）
    def history(self, limit: int = 50) -> list[Message]
    def record_fact(self, fact: Fact) -> None       # 长期事实库
    def facts(self) -> list[Fact]
    def snapshot(self) -> dict                      # {messages: [...], facts: [...]} 供审计/回放
```

## 4. core/agent.py

```python
@dataclass
class AgentSpec:
    id: str
    role: str
    goal: str
    backstory: str = ""
    tools: list[str] = field(default_factory=list)      # 工具名，由 ToolRegistry 解析
    approval_level: str = "read-only"                    # read-only|write|execute

class AgentRuntime:
    """单 agent 执行器：组装系统提示（角色/目标/背景+可用工具说明），
    循环：LLM 决策 → 调工具（经审批）→ 观察 → 直到产出最终内容或达 max_steps。"""
    def __init__(self, spec: AgentSpec, llm: LLMClient, registry: ToolRegistry,
                 memory: SharedMemory, audit: AuditLog, approver: Approver,
                 max_steps: int = 6): ...
    async def run(self, task: str, context: dict | None = None) -> Message:
        """返回 RESULT 消息（payload.content 为最终文本）。期间的工具调用与消息写入审计。"""
    def system_prompt(self) -> str: ...
```

## 5. tools/registry.py

```python
@dataclass
class ToolResult:
    ok: bool
    content: str          # 给模型的文本
    error: str = ""

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict      # JSON Schema
    approval_level: str   # read-only|write|execute
    handler: Callable[[dict], ToolResult]   # 同步或异步均可，注册时统一包装为 async

class ToolRegistry:
    def register(self, spec: ToolSpec) -> None
    def get(self, name: str) -> ToolSpec
    def openai_schemas(self, names: list[str]) -> list[dict]   # OpenAI tools 格式
    async def execute(self, name: str, args: dict, approver: Approver) -> ToolResult
    """执行前若 approval_level != read-only 则先 approver.request()，拒绝返回 ToolResult(ok=False, error='denied')。"""
```

## 6. tools/builtin.py

```python
def register_builtin_tools(registry: ToolRegistry, workspace: Path) -> None:
    """注册 3 个内置工具：
    web_search(query)          # duckduckgo-search，返回前 5 条 {title,url,snippet}；read-only
    python_repl(code)          # subprocess 沙箱（python -c），5s 超时 + 输出截断 2000 字符 + 临时目录；execute
    read_file(path)            # 路径白名单（workspace 内）+ 扩展名白名单 + 1MB 上限；read-only
    """
```

## 7. tools/mcp_client.py

```python
class MCPClient:
    """stdio MCP 客户端（仅工具桥接）。"""
    def __init__(self, command: str, args: list[str], cwd: str | None = None,
                 env: dict | None = None, timeout_s: float = 180.0): ...
    async def connect(self) -> None                 # 初始化 + listTools
    def tool_names(self) -> list[str]               # 形如 mcp__<server>__<name>（server 名取自初始化响应）
    async def call(self, name: str, arguments: dict) -> str   # 返回文本投影（文本块连接）
    def register_into(self, registry: ToolRegistry) -> None   # 注册为 read-only 工具
    async def close(self) -> None
```

## 8. runtime/audit.py

```python
@dataclass
class AuditEvent:
    ts: str
    kind: str        # message|tool_call|tool_result|decision|approval|pattern_start|pattern_end
    actor: str       # agent id 或 system
    detail: dict

class AuditLog:
    def record(self, kind: str, actor: str, detail: dict) -> None
    def events(self) -> list[AuditEvent]
    def replay(self) -> list[dict]     # 按时间序返回全部事件（用于回放/导出）
    def to_markdown(self) -> str       # 审计报告（演示用）
```

## 9. runtime/approval.py

```python
class Approver:
    """审批器。auto_approve=True 时直接放行（演示/评测模式）。"""
    def __init__(self, auto_approve: bool = False): ...
    async def request(self, tool_name: str, args: dict) -> bool
    # CLI 模式：input() 询问 y/n；auto_approve 直接 True。留接口供 UI 扩展。
```

## 10. patterns/base.py 与四模式

```python
@dataclass
class AgentResult:
    answer: str
    audit: AuditLog
    facts: list[Fact]
    tokens: dict[str, int]      # {agent_id: 该 agent 的 LLM token 消耗}（从 llm 返回 usage 累加）
    wall_s: float

class CollaborationPattern(ABC):
    def __init__(self, team: list[AgentRuntime], memory: SharedMemory,
                 audit: AuditLog, config: dict): ...
    @abstractmethod
    async def run(self, query: str) -> AgentResult: ...

def build_team(specs: list[AgentSpec], llm, registry, memory, audit, approver, replicas: dict[str, int]) -> list[AgentRuntime]
# replicas: {"fact_checker": 4} → 生成 fact_checker-1..4 四个运行时实例（同 spec，id 带后缀）

def run_pattern(name: str, team_specs: dict, llm, registry, memory, audit, approver,
                config: dict) -> CollaborationPattern:
    """name ∈ {pipeline, parallel, supervisor, debate}；返回对应模式实例（实现文件在 patterns/ 下同名模块）。"""
```

四模式行为约定：
- **pipeline**：按 team 顺序接力，前一个 RESULT 的 content 作为下一个的 task 上下文；终答 = 最后一个 agent 的 content。
- **parallel**：`planner` 收到 query → 产出 `{"subtasks": ["...", ...]}`（json 输出）→ `asyncio.gather` 并发执行 replicas 组（subtask 轮流分配）→ `aggregator` 汇总各 RESULT 为终答；memory.facts 合并。终答 = aggregator content。**必须真并发**（gather，不许串行 await 循环）。
- **supervisor**：supervisor 循环：读当前 facts 与未完成子任务 → 决策 `{"assign": {...} | "done": true, "final": "..."}` → 派活给 worker（worker 用同一 AgentRuntime）→ worker 失败重试 ≤2 次、换人 ≤1 次 → done 时终答 = final。
- **debate**：`pro` 与 `con` 交替发言（各 2 轮，每轮见对方上一轮观点 + facts）→ `judge` 输出 `{"verdict": "...", "reasoning": "..."}` → 终答 = verdict + reasoning。

## 11. 配置 YAML 约定

`config/agents/<team>.yaml`：
```yaml
team:
  name: <team-id>
  replicas: {fact_checker: 4}
  agents:
    - {id: planner, role: ..., goal: ..., backstory: ..., tools: [], approval_level: read-only}
    ...
```
`config/workflow.yaml`：
```yaml
workflow:
  pattern: parallel          # pipeline|parallel|supervisor|debate
  llm_config: config/llm.yaml
  team_config: config/agents/fact_check_team.yaml
  mcp: {enabled: false, command: node, args: ['E:\Agent\vision-mcp\server.js']}
  approval: {auto_approve: true}
  offline_data: samples/     # 非空则 web_search 优先从样例数据检索
```

## 12. 组件边界（不得越界）

- core 组只写 `src/agent_collab/core/*` 与 `tests/test_core_*.py`；
- tools 组只写 `src/agent_collab/tools/*`、`src/agent_collab/runtime/audit.py`、`src/agent_collab/runtime/approval.py` 与对应测试；
- patterns 组只写 `src/agent_collab/patterns/*` 与 `tests/test_patterns_*.py`（可 import core/tools 的契约，用假 LLM/工具做单测）；
- demo 组写 `src/agent_collab/demo/*`、`src/agent_collab/eval/*`、`src/agent_collab/ui/*`、`config/*`、`samples/*`；
- docs 组只写 `README.md`、`docs/INTERVIEW_PREP.md`、`requirements.txt`。
- 任何组不得修改其他组的文件；发现契约问题 → 向编排者报告，不自行改契约。
