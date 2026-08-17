# 面试模拟：项目四 — AgentCollab 多智能体协作框架

> GitHub 仓库 `agent-collab` · 4 协作模式 · A2A 消息协议 · 共享记忆 · MCP · 审计回放 · 8 题评测
> 建议：先读 `docs/design.md` 与 `docs/contracts.md` 建立全局观，再背本文的问答。

---

## 一、三句话介绍（30 秒电梯演讲）

> 我实现了一个**微型多智能体协作框架 AgentCollab**。它提供 **4 种协作模式**——流水线、并行分工、监督调度、圆桌辩论——用声明式 YAML 定义团队，用 **A2A 风格的 Agent 间消息协议**替代直接函数调用，让协作全程可审计、可回放。核心卖点是**真并发协作**：事实核查场景里 planner 拆解声明后，4 个研究员通过 `asyncio.gather` 真正并行核查，再经辩论与裁判收敛出结论。框架还自研了共享记忆、MCP 工具接入、三级人工审批，并用同一套 LLM 的**单 Agent 基线做评测对比**，用数据证明「多 Agent 协作」相对「单 Agent 多工具」的收益。

**三句话区分四个项目（面试开场用）：**

| 项目 | 核心能力 | 协作形态 | 关键差异 |
|------|----------|----------|----------|
| 项目一 kb-rag | 检索增强生成 | 顺序流水线 | BM25+向量+RRF+CE+引用/拒答 |
| 项目二 CinemaScope | 多 Agent 推荐 | 4 Agent **固定流水线** | 顺序接力，职责边界清晰 |
| 项目三 research-agent | 多工具 Agent | **单 Agent** 多工具 | LangGraph ReAct + 文本标签协议 |
| 项目四 AgentCollab | 多 Agent 协作框架 | **多 Agent 真协作** | 并行/监督/辩论 + 消息协议 + MCP |

> 进阶线一句话：**流水线协同 → 单 Agent 多工具 → 多 Agent 真协作**。

---

## 二、架构图（面试白板背这张）

```
                        ┌─────────────────────────────┐
                        │  表现层：CLI / Gradio UI      │
                        │  输入声明 → 展示消息流+回放    │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  workflow.yaml：选择协作模式  │
                        │  pipeline/parallel/supervisor/│
                        │  debate                     │
                        └──────────────┬──────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
┌───────▼────────┐        ┌────────────▼────────────┐    ┌────────────▼───────────┐
│ parallel       │        │ supervisor              │    │ debate                 │
│ planner 拆解   │        │ 循环:读facts→决策→派活   │    │ pro/con 各2轮→judge    │
│ N研究员真并发  │        │ 失败重试≤2/换人≤1        │    │ 裁判裁决               │
│ aggregator汇总 │        │                          │    │                        │
└───────┬────────┘        └────────────┬────────────┘    └────────────┬───────────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  Agent 团队 fact-check-squad │
                        │  planner / fact_checker×4 /  │
                        │  debater_pro/con / judge /   │
                        │  editor                     │
                        └──────────────┬──────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
┌───────▼────────┐        ┌────────────▼────────────┐    ┌────────────▼───────────┐
│ core 层        │        │ tools 层                │    │ runtime 层             │
│ message 信封   │        │ registry/builtin/mcp    │    │ audit 审计回放         │
│ memory 共享记忆│        │                          │    │ approval 人工审批      │
│ agent 运行时   │        │                          │    │                        │
│ llm_client     │        │                          │    │                        │
└────────────────┘        └──────────────────────────┘    └────────────────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  外部：LLM 中转 gpt-5.5 +    │
                        │  MCP Server（如 vision-mcp） │
                        └─────────────────────────────┘
```

**核心约束（背）**：Agent 之间**不直接调用**，一切交互走 `message.py` 的消息信封 → 每条消息进审计日志 → 全程可回放。

---

## 三、诚实边界（面试先立人设，避免被质疑「重复造轮子」）

> 本项目是 **LangGraph 生态实现** + 借鉴 CrewAI / A2A / MCP / codex-cli 设计思想的「**自研微型框架**」。定位是理解多 Agent 协作的**底层原理 + 工程落地**，**不是**试图替代 CrewAI / AutoGen 等成熟框架。

**LangGraph 在本项目里的真实作用**（务必诚实，这是高频追问点）：

- LangGraph 提供**状态图编排基座**：StateGraph 的节点 / 边 / 共享状态 + checkpointing，承载 4 种协作模式的**控制流骨架**（谁先谁后、何时并发、何时收敛、状态可检查可恢复）。
- LangGraph **不负责**「多 Agent 怎么协作」的业务语义——消息协议、共享记忆、审批分级、审计回放、MCP 客户端、评测对比，全部是自研。
- 单 Agent 运行时（`core/agent.py`）是**手写的 ReAct 风格循环**（延续项目三），没有用 LangGraph 的 `create_react_agent` 高级封装——为的是拿到每一轮 LLM 决策的中间态，方便审计与评测。

> 一句话话术：**「我用 LangGraph 的 StateGraph 当编排底座，但把多 Agent 协作真正有价值的部分——消息协议、共享记忆、审批、审计、MCP——自己实现了一遍，因为面试要展示的就是这些底层原理。」**

---

## 四、核心设计分节

### 4.1 4 种协作模式（各自的适用场景与实现要点）

| 模式 | 流程 | 并发 | 适用场景 | 实现要点 |
|------|------|:---:|----------|----------|
| `pipeline` | A→B→C 顺序接力 | 无 | 基线对比；职责强顺序 | 前一 RESULT 的 content 作为下一 Agent 的 task 上下文；终答 = 最后 Agent |
| `parallel` | planner 拆解 → N 研究员并发 → aggregator 汇总 | **真并发** | 事实核查主模式；子任务相互独立 | planner 结构化输出 `{"subtasks": [...]}`；`asyncio.gather` 并发；subtask 轮流分配；facts 合并 |
| `supervisor` | 监督者循环调度 | 有 | 任务动态变化、需要纠偏 | 循环读 facts → 决策 `{"assign" \| "done"}`；失败重试 ≤2、换人 ≤1 |
| `debate` | 正/反方交替 → 裁判 | 无（回合制） | 结论冲突裁决 | pro/con 各 2 轮，每轮见对方上一轮 + facts；judge 输出 verdict + reasoning |

**为什么需要 4 种，不能只用一种？** 因为它们对应协作的三种本质：**分工**（parallel：独立子任务并行）、**调度**（supervisor：动态依赖 + 失败纠偏）、**裁决**（debate：观点冲突收敛），外加 **pipeline 作为不做并发的基线**。没有一种模式能同时覆盖「并行加速」「动态纠错」「冲突裁决」三个需求，所以用 4 种各自聚焦 + 统一 `AgentResult` 出口。

### 4.2 消息协议为何优于直接函数调用

直接函数调用的三个问题：

1. **强耦合**：Agent A 调 Agent B 的方法，就得 import B、知道 B 的签名，团队规模一变大就变成网状依赖。
2. **不可观测**：函数调用不落日志，事后无法回答「谁在什么时刻给谁发了什么」。
3. **无法组播/路由**：函数调用是 1:1 同步；协作需要 1:N 组播（`recipient: "all"`）、异步收发、消息重放。

消息协议（A2A 风格）的对应收益：

- Agent 只有 **inbox/outbox**，只认 `Message` 信封（`id/sender/recipient/type/payload/ts`），不 import 彼此 → 解耦。
- 每条消息必经 `AuditLog` → **全程可回放**（OpenHands 事件流思想）。
- `type` 用字符串枚举（`task/result/query/decision`）带 pydantic 校验，`payload` 按类型约定结构，收方校验前置。

> 面试话术：**「消息协议和函数调用不是互斥，而是不同层次——消息协议解决 Agent 与 Agent 之间的解耦与可观测，函数调用解决 Agent 与 LLM/工具之间的交互。」**

### 4.3 共享记忆 vs 隔离记忆的权衡

| 维度 | 共享记忆（本项目默认） | 隔离记忆 |
|------|------------------------|----------|
| 优点 | 研究员写入的 facts 立即被编辑/裁判读取；协作收敛快 | 各 Agent 上下文干净，无信息泄露/污染 |
| 缺点 | 上下文噪声、可能互相污染 | 需要额外的显式信息交换，收敛慢 |
| 适用 | 目标一致的协作（事实核查团队同查一条声明） | 目标冲突或需要保密的场景 |

**本项目的落地**：短期「消息缓冲」（每 Agent 可见范围可配：team 级共享 / agent 私有）+ 长期「`facts.json` 结构化事实库」（研究员写 `claim→verdict→evidence→sources`，编辑读来生成报告）。**权衡**：默认 team 级共享以加速收敛，但保留「可见范围可配」的开关，代价是私有范围下需要额外消息传递才能对齐——这正是共享 vs 隔离的经典 trade-off。

### 4.4 parallel 真并发实现（asyncio.gather + to_thread）

```python
# 伪代码：planner 拆解后，N 个研究员真并发执行
async def run(self, query: str) -> AgentResult:
    subtasks = await self._plan(query)            # planner → {"subtasks": [...]}
    results = await asyncio.gather(*[            # ★ 真并发：gather，不是串行 await
        self._run_researcher(sub) for sub in subtasks
    ])
    return await self._aggregate(results)

async def _run_researcher(self, sub: str):
    # LLM 客户端与工具是同步阻塞 IO → 丢进线程池，避免阻塞事件循环
    resp = await asyncio.to_thread(self.llm.complete, messages)
    ...
```

**两个关键点**：

1. **`asyncio.gather` 编排 N 个协程并发等待**——如果写成 `for sub in subtasks: await run(sub)` 就会退化成串行，这是最常见的「假并发」坑。
2. **`asyncio.to_thread` 把同步阻塞 IO（openai SDK 调用、subprocess 工具）丢进线程池**——纯 asyncio 里直接调用阻塞函数会卡住整个事件循环，其他协程被饿死。

**并发证据**：测试用 FakeLLM 记录调用窗口重叠、输出时间戳与线程 id，断言 N 个研究员确实重叠执行（验收标准明确要求「parallel 有并发证据」）。

### 4.5 supervisor 失败重试与换人策略

supervisor 循环：

```
读当前 facts + 未完成子任务
  → LLM 决策：{"assign": {...}} 继续派活  或  {"done": true, "final": "..."} 收尾
  → 派活给 worker（同一 AgentRuntime）
  → worker 失败：重试 ≤2 次（同人再试）→ 仍失败则换人 ≤1 次（换另一个 worker）
  → 换人后仍失败：不再无限循环，标记该子任务「证据不足/存疑」，由 supervisor 在 final 中如实收敛
```

**为什么重试 ≤2、换人 ≤1 而不是无限重试？** 成本与收益的边界：LLM 失败多为瞬态（网络/超时，由 LLMClient 内部退避重试覆盖）或任务本身难；同一 worker 重试超过 2 次通常不改变结果，换人超过 1 次说明任务定义或上下文有问题。**上限是防止死循环与 token 爆炸的硬约束**，同时把失败结果显式写进 facts（存疑），保证结论诚实。

### 4.6 debate 回合设计

- `pro` 与 `con` **交替发言各 2 轮**：每轮发言时能看到「对方上一轮观点 + 当前 facts」——保证论点针锋相对，而不是自说自话。
- 2 轮是**信息增量与成本的平衡**：第 1 轮立论、第 2 轮反驳；超过 2 轮边际信息递减、token 线性上涨。
- 最后 `judge` 结构化输出 `{"verdict": "...", "reasoning": "..."}`：verdict 四选一（真实/虚假/部分属实/证据不足），reasoning 引用双方论证与 facts 证据。
- **终答 = verdict + reasoning**，不是「谁声音大谁赢」，而是裁判基于证据链裁决。

### 4.7 MCP 标准化

自研 `tools/mcp_client.py`（stdio MCP 客户端，仅**工具桥接**子集）：

```
spawn [command + args]（如 node E:\Agent\vision-mcp\server.js）
  → connect()：JSON-RPC 握手 + listTools
  → 把外部工具注册进自有 ToolRegistry，命名 mcp__<server>__<tool>
  → call()：callTool 代理，返回文本投影
```

**收益**：外部工具（如 vision MCP 的「图片证据核查」）以 **first-class 工具**身份进入框架——统一走 `ToolRegistry` 的 schema、审批、审计，与内置工具无差别。演示时把 vision MCP 挂为图片证据核查工具，展示「多 Agent + 多协议工具」的边界打通。

### 4.8 审计与回放

`AuditLog` 记录带时间戳的事件：`message / tool_call / tool_result / decision / approval / pattern_start / pattern_end`。`replay()` 按时间序返回全部事件，`to_markdown()` 导出演示报告。

**价值**：① 排障——协作出问题时能定位「哪个 Agent 哪一步错了」；② 演示——Gradio 里回放完整消息轨迹；③ 评测——token/延迟/决策链都有据可查。这与「消息协议替代函数调用」互为表里：**消息协议是数据来源，审计是数据的落地点**。

### 4.9 人工审批分级（read-only / write / execute）

工具声明 `approval_level`：

| 级别 | 含义 | 处理 |
|------|------|------|
| `read-only` | 只读（web_search / read_file） | 自动放行 |
| `write` | 写文件（如 editor 的 write_report） | 进审批队列，CLI/Gradio 确认 |
| `execute` | 执行命令（如 python_repl） | 进审批队列，CLI/Gradio 确认 |

借鉴 codex-cli：**默认最小权限**，敏感操作（写/执行）必须人工确认；演示/评测模式提供 `auto_approve` 一键全放行。审批决策本身也写审计（`approval` 事件），保证「谁批准了什么」可追溯。

---

## 五、必背数字区

> **评测数字待 Task 6 实测回填**：下表评测项一律标「待实测」，面试前以 `eval/report.md` 实际数字为准。

| 维度 | 数字 | 备注 |
|------|:----|------|
| 协作模式 | **4** | pipeline / parallel / supervisor / debate |
| 消息类型 | **4** | task / result / query / decision |
| 团队角色 | **6** | planner / fact_checker(×4) / debater_pro / debater_con / judge / editor |
| fact_checker 实例 | **4**（replicas） | 并行核查 |
| 内置工具 | **3** | web_search / python_repl / read_file |
| 审批等级 | **3** | read-only / write / execute |
| supervisor 重试上限 | **≤2 次** | 同 worker 重试 |
| supervisor 换人上限 | **≤1 次** | 换另一个 worker |
| debate 轮次 | **各 2 轮** | pro/con 交替 |
| AgentRuntime max_steps | **6** | 单 Agent 循环上限 |
| AgentRuntime 历史窗口 | **20** 条 | 每步携带最近历史（省 token） |
| 共享记忆缓冲 | **50** 条（默认） | history 上限 |
| LLM 重试 | **max_retries=3** | 仅瞬态错误（连接/超时/≥500/429） |
| LLM 超时 | **120s** | timeout_s |
| python_repl 沙箱 | **5s 超时 + 2000 字符截断** | subprocess 隔离 |
| read_file 防护 | **白名单×2 + 1MB 上限** | 路径/扩展名白名单 |
| web_search 结果 | **前 5 条** | DuckDuckGo |
| 评测任务 | **8 条** | 真实/虚假/部分属实/证据不足 各 2 条 |
| 评测指标 | **4** | 判定准确率 / 信息覆盖度 / token 成本 / 端到端延迟 |
| 验收测试 | **≥30 用例** | pytest 全绿 |

**评测结构（真实 LLM 前 3 题实测，factcheck 组合模式）：**

| 指标 | 单 Agent 基线 | 多 Agent（factcheck） |
|------|:---:|:---:|
| 判定准确率（3 题） | **100%**（3/3） | **33%**（1/3） |
| 信息覆盖度（3 题） | 100% | 100% |
| token 成本（3 题） | 42,853 | 468,040 |
| 端到端延迟（3 题） | 未单列 | 563.1s |

> **面试话术（关键，务必讲清）**：这是本项目最有价值的诚实发现——多 agent 协作在**简单可单点判定的声明**上并不更准（单 agent 直接判定更稳），它的价值在**复杂多事实声明的并行覆盖、可审计可回放、结论可解释**。优化历程要能讲：最初 parallel（核查→editor 直接汇总）缺裁决环节 → editor 保守误判「证据不足」→ 加 `factcheck` 组合模式（核查→辩论→裁决→撰写）后第 1 题修正、第 2 题收敛为「部分属实」；剩余误差来自**对称辩论为虚假声明强辩干扰裁判** → 已改为「裁判证据优先于观点」。**主动讲清 trade-off 与迭代过程，比宣称"全面碾压基线"可信得多。**

---

## 六、30+ 模拟面试问答

### 6.1 定位与诚实边界

**Q1：一句话介绍这个项目？**
A：见「一、三句话介绍」。核心是「4 种协作模式 + 消息协议 + 共享记忆 + 审计回放，用单 Agent 基线证明多 Agent 协作收益」。

**Q2：为什么不用 CrewAI / AutoGen 现成的多 Agent 框架？**
A：四个理由：① **理解原理**——CrewAI 把 agent 定义、任务编排、记忆都封装好了，用它等于跳过「多 Agent 协作到底怎么实现」这个我要展示的核心；② **白盒控制**——我需要审计回放、审批分级、token 分 agent 统计这些细粒度能力做评测，成熟框架把它们抽象掉后拿不到中间态；③ **叙事连续性**——项目三已用 LangGraph，项目四在 LangGraph 底座上自研协作层，是能力递进而非换框架；④ **诚实定位**——我不声称比 CrewAI 更好，它是一个「为面试/教学而生的微型框架」，把底层原理暴露出来。工程上如果是生产项目，我会优先评估 CrewAI/AutoGen。

**Q3：LangGraph 在你的项目里到底起了什么作用？（高频，必诚实）**
A：LangGraph 提供**状态图编排基座**——StateGraph 的节点/边/共享状态 + checkpointing，承载 4 种协作模式的控制流骨架，并让状态可检查、可恢复。但「多 Agent 怎么协作」的业务语义（消息协议、共享记忆、审批、审计、MCP、评测）全部自研；单 Agent 运行时也是手写 ReAct 循环而非 `create_react_agent`，为的是拿到每轮 LLM 决策的中间态。一句话：**用 LangGraph 当底座，展示的是底座之上的自研协作层**。

**Q4：你这个和 CrewAI 的关系/区别？**
A：设计思想借鉴（声明式 YAML 定义团队、角色/目标/背景），但实现是 LangGraph 底座 + 自研协作层。区别在于：CrewAI 是通用成熟框架，Agent 间通信、记忆、编排对用户黑盒；本项目把这些机制白盒化，并额外做了 A2A 消息信封、三级审批、审计回放、单 Agent 基线评测。**借鉴思想、独立实现、服务教学与展示。**

**Q5：会不会被认为「重复造轮子」？**
A：会，所以我主动立边界（见第三节）。轮子的价值在于**学习与展示**——简历上「用 CrewAI 搭了个 demo」和「自己实现了消息协议/共享记忆/审批/审计」的含金量完全不同。我明确不替代成熟框架，生产会选现成方案，本项目是「理解原理 + 工程落地」的证明。

**Q6：和 LangChain 的关系？**
A：底层 LLM 客户端用 openai SDK 直连中转（不依赖 LangChain 的 LLM 抽象），LangChain 体系里只取 LangGraph 的状态图。项目三已论证过：手写 ReAct 循环比黑盒 AgentExecutor 更透明可控，项目四延续这个选择——编排用 LangGraph 底座，其余自研。

### 6.2 架构与消息协议

**Q7：整体架构分几层？每层职责？**
A：五层。① 表现层（CLI/Gradio）；② 协作模式层（patterns/，4 种模式）；③ Agent 团队（声明式 YAML 定义）；④ core 层（message/memory/agent/llm_client）；⑤ 工具与运行时层（tools/ + audit/approval），对接外部 LLM 与 MCP。Agent 间通信统一走消息信封，不直接调用。

**Q8：为什么 Agent 之间用消息协议，而不是直接函数调用？**
A：见 4.2。三个核心收益：解耦（只认信封不 import 彼此）、可观测（每条消息进审计）、可路由/组播/重放（`recipient: "all"`、按时间序回放）。

**Q9：消息协议和 Function Calling 是什么关系？**
A：**正交、不同层次**。Function Calling 是「单个 Agent 与 LLM/工具之间」的调用协议，由 LLMClient 在 AgentRuntime 循环内部使用（模型返回 tool_calls → registry.execute）。消息协议是「Agent 与 Agent 之间」的通信协议，负责路由、组播、审计。两者不冲突：一个 Agent 内部可能正在 Function Calling 调工具，同时通过消息信封给别的 Agent 发 RESULT。

**Q10：消息的 type 为什么用字符串枚举？payload 怎么约定？**
A：`MessageType(str, Enum)` 限定 `task/result/query/decision` 四种，pydantic 校验前置，非法 type 直接构造失败——比裸字符串安全。payload 按 type 约定结构：TASK 含 `{task, context}`，RESULT 含 `{content, data}`，避免「自由 dict 各写各的」。`new_message()` 统一生成 uuid4 hex id 与 ISO8601 时间戳。

**Q11：消息协议借鉴了 A2A 的什么？**
A：A2A（Google 的 Agent-to-Agent 协议）的核心思想是「Agent 通过标准化信封 + 消息类型异步通信，而非直接调用」。本项目借鉴了信封结构（sender/recipient/type/payload/时间戳）与「能力解耦」思想，但做了**刻意简化**：只保留 4 种消息类型、单机 in-memory 传递，不实现 A2A 的网络发现/认证/多机传输——因为教学框架要的是「解耦 + 可审计」，不是分布式。

### 6.3 协作模式

**Q12：为什么需要 4 种协作模式？不能只用一种吗？**
A：见 4.1。协作的三种本质需求——并行分工、动态调度、冲突裁决——无法用一种模式同时覆盖；再加 pipeline 作为「无并发基线」用于评测对比。统一 `AgentResult` 出口让 4 种模式可横向对比。

**Q13：pipeline 模式的定位和实现要点？**
A：定位是**基线**——验证「不做并发、纯顺序接力」的下限。实现：按 team 顺序接力，前一 Agent 的 RESULT.content 作为下一 Agent 的 task 上下文，终答 = 最后 Agent 的 content。它的存在价值是让评测里「并行 vs 顺序」的收益可量化。

**Q14：parallel 模式怎么实现真并发？**
A：见 4.4。planner 结构化输出 `{"subtasks": [...]}` → `asyncio.gather` 并发执行 replicas 组（subtask 轮流分配）→ aggregator 汇总。关键：`gather` 而非串行 `for await`；阻塞 LLM/工具调用经 `asyncio.to_thread` 丢线程池，避免饿死事件循环。

**Q15：为什么用 asyncio.gather 而不是线程池/串行？**
A：串行 `for await` 是「假并发」——总耗时 = N 个任务之和；gather 让 N 个研究员真正重叠执行，总耗时 ≈ 最慢的那个。之所以不直接用 `ThreadPoolExecutor`：asyncio 是本项目的统一并发模型（模式层、工具执行、MCP 都 async），gather + to_thread 能把「IO 密集的 LLM 等待」和「CPU/阻塞的 SDK 调用」统一在一个事件循环里编排，便于统一管理审计时间序与取消。

**Q16：to_thread 是干什么的？为什么要它？**
A：openai SDK 的 `chat.completions.create` 是**同步阻塞**调用。在 asyncio 协程里直接调用会阻塞整个事件循环，让其他研究员（协程）无法继续。`asyncio.to_thread` 把阻塞调用丢进默认线程池执行，协程 `await` 时让出事件循环，其他研究员得以并发推进——这是「协程编排 + 线程执行阻塞 IO」的标准组合。

**Q17：planner 怎么拆解任务？拆解结果怎么传给 N 个研究员？**
A：planner 用 `json_complete` 强制结构化输出 `{"subtasks": ["...", ...]}`（response_format json_schema，解析失败重试一次）。拆解结果以 TASK 消息或直接作为 context 传给 replicas 组，**subtask 轮流分配**（round-robin 到 fact_checker-1..4）。研究员各自核查后写 RESULT 与 facts，aggregator 汇总成终答。

**Q18：parallel 和 supervisor 的区别与选型？**
A：**parallel 是「一次拆解、并发执行、汇总收尾」的静态分工**——适合子任务相互独立、拆解后无需中途调整（事实核查主场景）。**supervisor 是「动态循环调度」**——每轮读 facts 决策再派活，适合任务动态变化、需要中途纠偏、依赖关系不确定的场景。选型：子任务确定且独立 → parallel（快、省 token）；任务演化不确定、要失败恢复 → supervisor（灵活、费 token）。

**Q19：supervisor 的失败重试和换人策略？**
A：见 4.5。worker 失败 → 同 worker 重试 ≤2 次 → 仍失败换另一个 worker ≤1 次 → 再失败标记「证据不足/存疑」如实收敛进 final。每次失败写审计。

**Q20：supervisor 怎么避免死循环？**
A：三重硬上限：① 单个 worker 重试 ≤2、换人 ≤1；② supervisor 决策必须显式产出 `{"done": true, "final": "..."}` 才收尾，未完成子任务耗尽且无法推进时强制收敛为「存疑」；③ AgentRuntime 内部 max_steps=6 兜底。加上 LLMClient 只重试瞬态错误（4xx 直接失败，不重试），杜绝「对确定性错误无限重试」。

**Q21：debate 回合怎么设计？裁判怎么裁决？**
A：见 4.6。pro/con 交替各 2 轮（每轮见对方上一轮 + facts），judge 结构化输出 `{verdict, reasoning}`，verdict 四选一、reasoning 引用证据链。终答 = verdict + reasoning。

**Q22：debate 和 supervisor 什么场景用哪个？**
A：结论**冲突裁决**用 debate（两个立场对同一结论有分歧，需要正反论证 + 裁判）；**执行纠偏**用 supervisor（任务没做好，需要调度者重新分工）。一个是「观点博弈」，一个是「任务调度」，本质不同。

### 6.4 记忆

**Q23：共享记忆怎么设计？短期和长期分别是什么？**
A：`SharedMemory` 两块：**短期** = 消息缓冲（`post/history`，默认保留最近 50 条，team 级共享）；**长期** = `facts.json` 结构化事实库（`Fact{claim, verdict, evidence, sources, by}`，研究员写入、编辑/裁判读取）。`snapshot()` 导出可序列化快照供审计/回放。

**Q24：共享记忆 vs 隔离记忆的权衡？**
A：见 4.3 表格。共享加速收敛但引入噪声/污染风险；隔离干净但需显式信息交换、收敛慢。本项目默认 team 共享（事实核查团队目标一致），但「可见范围可配」保留隔离选项，并讲清 trade-off。

**Q25：共享记忆怎么做并发安全？**
A：本项目是**单进程单事件循环**模型，`record_fact`/`post` 的写操作是**短临界区**——`list.append` 是原子操作，且写操作内部**没有 `await` 让出点**，因此在事件循环内天然串行化，无需加锁。这是刻意设计：把并发阻塞点（LLM/工具 IO）放进 `to_thread`，把共享记忆写操作收口在事件循环主线程执行。**诚实边界**：这只对单机单进程成立；若未来写操作进 worker 线程或多进程部署，需加 `asyncio.Lock` 或改用队列收口，我会明确讲这个前提。

**Q26：长期事实库 facts 为什么设计成结构化而非自由文本？**
A：结构化（claim/verdict/evidence/sources/by）让「研究员写 → 编辑读 → 报告生成」变成**确定性数据流**：编辑不再靠猜研究员的意思，裁判能引用具体 evidence 与 sources 做裁决；评测的「信息覆盖度」也能按 facts 命中的关键事实点精确计算。自由文本会让下游 Agent 和评测都失去抓手。

### 6.5 工具与安全

**Q27：内置工具的安全措施？**
A：见 4.9 与项目三。`web_search`（DuckDuckGo，前 5 条，read-only）；`python_repl`（subprocess 隔离 + 5s 超时 + 2000 字符截断 + 临时目录，execute）；`read_file`（路径白名单 + 扩展名白名单 + 1MB 上限，read-only）。三层防护延续项目三的「路径遍历」防御。

**Q28：MCP 客户端为什么自己实现协议，不用现成 SDK？**
A：三个原因：① **需要 first-class 集成**——外部 MCP server 的工具要注册进自有 `ToolRegistry`，统一走审批/审计/schema，现成 SDK 的工具对象无法直接挂进我们的契约；② **stdio 协议本身简单**——JSON-RPC over stdio 的 `initialize/listTools/callTool` 子集实现可控、可测试（测试用假 stdio server fixture 验证链路）；③ **轻依赖透明**——不引入重 SDK，出错可定位到每一帧。诚实：只实现「工具桥接」子集，不做 resources/prompts/session 全量。

**Q29：MCP 工具怎么注册进框架？**
A：`connect()` 握手 + `listTools` → 用 `mcp__<server>__<tool>` 命名注册为 `ToolSpec`（read-only 默认）→ 运行时 `call()` 代理 `callTool`，返回文本投影。之后 Agent 像用内置工具一样用外部工具，无差别。

**Q30：人工审批为什么分 read-only/write/execute 三级？**
A：**最小权限原则**。只读操作（搜索/读文件）无副作用自动放行，避免打断；写文件、执行命令有副作用，必须人工确认。三级粒度让「安全」与「流畅」平衡：演示时敏感操作（editor 写报告、python_repl 执行）触发审批弹窗，展示安全设计；评测用 auto_approve 自动化。借鉴 codex-cli 的审批思路。

**Q31：python_repl 沙箱怎么做？（可能追问延续项目三）**
A：`subprocess.run(["python", tmp_file])` 独立子进程隔离（非 eval/exec，避免 LLM 生成代码破坏主进程）+ 5s 超时 + stdout 截断 2000 字符 + finally 清理临时文件。诚实：不是生产级（无 Docker/网络隔离/磁盘配额），demo/研究够用，不暴露公网。

### 6.6 审计与评测

**Q32：审计日志记录什么？回放怎么实现？**
A：`AuditEvent{ts, kind, actor, detail}`，kind 覆盖 `message/tool_call/tool_result/decision/approval/pattern_start/pattern_end`。`replay()` 按时间序返回全部事件，`to_markdown()` 导出报告。Gradio 用 replay 渲染消息流。

**Q33：如何评测多 Agent 协作收益？**
A：**对照组设计**——同一 LLM（gpt-5.5）、同一 8 条评测集，搭一个「单 Agent 基线」（单个 AgentRuntime 多工具 ReAct 循环，不经协作模式）与「多 Agent parallel」对比。指标四个：判定准确率、信息覆盖度（命中关键事实点比例）、token 成本、端到端延迟。**关键**：覆盖度是多 Agent 的核心卖点（N 个研究员并行核查能覆盖更多事实点），成本/延迟是必须诚实报告的代价——「协作收益」不等于「无代价」。

**Q34：评测集怎么设计？8 条声明怎么分布？**
A：8 条声明，**真实/虚假/部分属实/证据不足 各 2 条**——覆盖事实核查的四种判定结果，避免只测「真假」两元导致评测失真。配套 offline 样例来源数据，web_search 在 offline 模式下改为样例数据检索，保证**可复现**（不依赖网络波动与搜索随机性）。

**Q35：信息覆盖度指标怎么算？**
A：每条声明预先标注「关键事实点」集合，Agent 产出（facts + 终答）命中的关键事实点比例 = 覆盖度。多 Agent 的 N 个研究员并行核查、各写 facts，理论上能覆盖更多事实点，这是预期「多 agent 显著占优」的指标（具体数字待实测）。

**Q36：单 Agent 基线怎么搭？为什么需要它？**
A：用同一个 AgentRuntime + 同一套工具（web_search/read_file 等），让**单个 Agent 多工具循环**直接作答评测集，不经 planner/parallel/aggregator。需要它的原因：**没有对照组的「多 Agent 效果好」没有说服力**——必须证明协作带来的收益超过「单 Agent 多工具」这个项目三已达到的水平，才是本项目四的真实增量。

**Q37：评测怎么控制成本/保证可复现？**
A：① offline 样例数据模式（web_search 改查样例，不联网、不随机、可复现）；② 单次演示 ≤30 次 LLM 调用（design 风险条款）；③ 评测只跑 8 题 × 2 组（单 agent 基线 + 多 agent），控制 token；④ token 分 agent 统计（AgentResult.tokens）精确核算成本。

### 6.7 工程与反思

**Q38：声明式 YAML 定义团队有什么好处？**
A：① 团队与代码解耦——加一个 Agent 或调 replicas 只改 YAML，不改 Python；② 复用 `AgentSpec`（id/role/goal/backstory/tools/approval_level）统一校验；③ `replicas: {fact_checker: 4}` 让「同一角色多实例」声明即得，`build_team` 自动生成 fact_checker-1..4；④ 延续项目二/三「全流程 YAML 配置驱动」的风格。

**Q39：失败重试如何避免死循环？（综合）**
A：三层防线：① LLMClient 层只重试瞬态错误（连接/超时/≥500/429），4xx 与编程错误**直接失败**，退避重试 max_retries=3；② AgentRuntime 层 max_steps=6 兜底，耗尽以最近文本兜底返回；③ 协作层 supervisor 重试 ≤2 + 换人 ≤1，换人后仍失败显式收敛为「存疑」并写 facts。每层都有硬上限 + 显式失败路径，杜绝无限重试。

**Q40：如果重做，你会改进什么？**
A：① **流式输出**——当前同步返回，改 streaming 让用户看到实时进度（Gradio 体验更好）；② **记忆持久化**——当前 JSON 文件，升级为带版本/并发安全的存储或向量检索长期记忆；③ **消息协议对齐 A2A 更多特性**——多机传输、能力发现（当前单机 in-memory）；④ **评测扩展**——8 题扩到更大集 + 加人工评估维度（证据质量、可读性）；⑤ **成本优化**——动态并发数（简单声明拆 2 个、复杂声明拆 6 个）。

**Q41：这个项目和前三个项目的关系/进阶线？**
A：项目一「检索增强」（顺序流水线）→ 项目二「多 Agent 流水线」（顺序协同）→ 项目三「单 Agent 多工具」（LangGraph ReAct）→ 项目四「多 Agent 真协作」（并行/监督/辩论 + 消息协议 + MCP）。**同一种能力——理解模型/Agent 能力边界并设计工程方案——在四个场景的递进应用**，项目四是这条线的终点。

**Q42：面试官问「这不就是 LangGraph 的一个 demo 吗」怎么答？**
A：不回避，正面分层：**如果只看编排骨架，确实可以用 LangGraph 的 StateGraph 描述；但本项目的增量不在骨架，而在骨架之上的协作语义**——A2A 消息信封 + 共享记忆权衡 + 真并发（gather+to_thread）+ supervisor 失败恢复 + 三级审批 + 审计回放 + **单 Agent 基线评测**。这一整套是 LangGraph 高级 API 不直接给、需要自己实现的，而「用数据证明协作收益」是 demo 与工程的分水岭。

---

## 七、扣分陷阱

| 问 | ❌ | ✅ |
|----|----|----|
| 「你这是自研框架吗」 | 「对，比 CrewAI 更强」 | 「是 LangGraph 生态实现 + 借鉴 CrewAI/A2A/MCP 的自研微型框架，用于理解原理与展示，不替代成熟框架」 |
| 「LangGraph 起了什么作用」 | 「我用 LangGraph 实现了多 Agent」 | 「LangGraph 提供 StateGraph 编排底座，消息协议/记忆/审批/审计/MCP 是自研，单 Agent 运行时也是手写 ReAct 循环」 |
| 「为什么不用 CrewAI」 | 「CrewAI 不行」 | 「我需要白盒控制做审计/评测，且要展示底层原理；生产项目我会评估现成框架」 |
| 「评测数字是多少」 | 随口编一个 | 「评测数字待 Task 6 实测回填，评测结构是 8 题 × 4 指标 × 单/多 Agent 对照」 |
| 「并发安全怎么保证」 | 「我加了锁」 | 「单进程单事件循环 + 短临界区无 await 让出，天然串行；多进程/多线程需 asyncio.Lock，我会讲清前提」 |
| 「和 LangChain 什么关系」 | 「我用了 LangChain」 | 「只取 LangGraph 状态图当底座，LLM 客户端用 openai SDK 直连，其余自研」 |

---

## 八、自测清单

- [ ] 能背出 4 种协作模式及其并发/适用场景
- [ ] 能画出五层架构图（表现层→模式层→Agent 团队→core→工具/运行时）
- [ ] 能解释「消息协议为何优于直接函数调用」的三点理由
- [ ] 能说清消息协议与 Function Calling 的层次关系（正交、不同层）
- [ ] 能解释 parallel 真并发的实现（asyncio.gather + to_thread，为何不能串行 await）
- [ ] 能讲 supervisor 的重试 ≤2 / 换人 ≤1 与死循环防线（三层）
- [ ] 能讲 debate 的回合设计与裁判裁决
- [ ] 能说清共享记忆 vs 隔离记忆的 trade-off 与并发安全前提
- [ ] 能解释 MCP 客户端自研协议的三点理由（first-class 集成/协议简单/轻依赖）
- [ ] 能说出三级审批（read-only/write/execute）的分级逻辑
- [ ] 能解释「单 Agent 基线」在评测中的作用（对照组）
- [ ] 能说出 8 条评测任务（真/假/部分属实/证据不足 各 2）与 4 个指标
- [ ] 能诚实回答「LangGraph 的作用」和「是不是重复造轮子」
- [ ] 能背出必背数字区的结构数字（4 模式/6 角色/4 实例/3 工具/3 审批/≤2 重试/≤1 换人/各2轮/max_steps6/history20/缓冲50/重试3/超时120s/5s+2000字符/1MB/前5条/8题/4指标/≥30用例）
- [ ] 评测数字一律回答「待实测回填」，不编造

---

## 九、关键文件

| 文件 | 作用 |
|------|------|
| `src/agent_collab/core/message.py` | 消息协议信封（A2A 风格） |
| `src/agent_collab/core/memory.py` | 共享记忆（短期缓冲 + 长期事实库） |
| `src/agent_collab/core/agent.py` | 单 Agent 运行时（LLM 循环 + 工具调用） |
| `src/agent_collab/core/llm_client.py` | OpenAI 兼容客户端（重试/超时/结构化输出） |
| `src/agent_collab/patterns/base.py` | 协作模式公共接口（AgentResult + run_pattern） |
| `src/agent_collab/patterns/parallel.py` | 真并发并行分工 |
| `src/agent_collab/patterns/supervisor.py` | 监督调度 + 失败恢复 |
| `src/agent_collab/patterns/debate.py` | 圆桌辩论 + 裁判 |
| `src/agent_collab/tools/registry.py` | 工具注册表（schema/审批） |
| `src/agent_collab/tools/mcp_client.py` | stdio MCP 客户端 |
| `src/agent_collab/runtime/audit.py` | 审计日志 + 回放 |
| `src/agent_collab/runtime/approval.py` | 人工审批分级 |
| `src/agent_collab/eval/*.py` | 评测集 + 指标 |
| `src/agent_collab/demo/fact_check_demo.py` | CLI 演示入口 |
| `src/agent_collab/ui/app.py` | Gradio UI |
| `config/agents/fact_check_team.yaml` | 声明式团队定义 |
| `config/workflow.yaml` | 协作模式 + 参数配置 |
