# AgentCollab — 多智能体协作框架 设计文档

日期：2026-08-14
状态：已批准（用户选定：方案 A + 新闻事实核查演示场景 + `E:\简历\简历项目\agent-collab`）

## 1. 目标

实现一个可写入简历的微型多智能体协作框架（LangGraph），核心卖点是**真并发协作**：
4 种协作模式 + Agent 间消息协议 + 共享记忆 + MCP 工具接入 + 人工审批 + 审计回放 + 与单 agent 基线的评测对比。
演示实例：**新闻事实核查团队**。

与既有项目的简历叙事关系：
- 项目二 CinemaScope：4 agent **流水线**协同（顺序编排）
- 项目三 research-agent：**单 agent** 多工具（LangGraph ReAct）
- 项目四 AgentCollab：**多 agent 真协作**（并行/监督/辩论 + 消息协议 + MCP）——进阶线的终点

## 2. 环境与技术栈

- Python 3.11（miniconda 独立环境 `agent-collab`，规避系统 Python 3.14 的依赖兼容风险）
- `langgraph`（状态图编排）、`openai`（兼容中转 gpt-5.5 的 OpenAI SDK）、`pydantic`、`pyyaml`、`gradio`、`pytest`、`duckduckgo-search`（免费搜索，项目三已验证）
- LLM：OpenAI 兼容中转（base_url 可配），默认 gpt-5.5
- 全流程 YAML 配置驱动（延续项目二/三风格）

## 3. 目录结构

```
E:\简历\简历项目\agent-collab\
├── README.md                      # 架构图 + 演示结果 + 评测数据
├── docs\design.md                 # 本文档
├── requirements.txt
├── config\
│   ├── llm.yaml                   # 中转 base_url/model/api_key_env
│   ├── workflow.yaml              # 协作模式与参数
│   └── agents\fact_check_team.yaml# YAML 声明式定义团队（角色/目标/背景/工具/审批等级）
├── src\agent_collab\
│   ├── core\
│   │   ├── agent.py               # Agent 运行时：角色 + 系统提示 + LLM 循环 + 工具调用
│   │   ├── message.py             # 消息协议信封（A2A 风格）
│   │   ├── memory.py              # 共享记忆：短期消息缓冲 + 长期事实库(JSON)
│   │   └── llm_client.py          # OpenAI 兼容客户端（重试/超时/结构化输出）
│   ├── tools\
│   │   ├── registry.py            # 工具注册表（name→schema→handler，审批标记）
│   │   ├── builtin.py             # web_search / python_repl(沙箱,5s+截断) / read_file
│   │   └── mcp_client.py          # stdio MCP 客户端（挂外部 MCP server 为工具）
│   ├── patterns\
│   │   ├── base.py                # 协作模式公共接口（run(query)→AgentResult+审计轨迹）
│   │   ├── pipeline.py            # 流水线：顺序接力（基线模式）
│   │   ├── parallel.py            # 并行分工：规划者拆解 → N 研究员真并发 → 汇总者
│   │   ├── supervisor.py          # 监督者：动态调度、失败重试、再分工
│   │   └── debate.py              # 圆桌辩论：多立场发言 → 裁判裁决共识
│   ├── runtime\
│   │   ├── audit.py               # 审计日志：消息/工具调用/决策全记录，可回放
│   │   └── approval.py            # 人工审批：敏感工具（写文件/执行命令）拦截确认
│   ├── eval\
│   │   ├── dataset.py             # 评测集：8 条事实核查任务（真/假/混合声明）
│   │   └── metrics.py             # 任务完成率 / 信息覆盖度 / token 成本 / 延迟
│   ├── demo\
│   │   └── fact_check_demo.py     # 演示入口 + offline 样例数据模式
│   └── ui\
│       └── app.py                 # Gradio：任务输入 + 消息流可视化 + 审计回放
├── tests\                         # pytest（消息协议/4 模式/记忆/工具安全/审批）
└── samples\                       # offline 模式的样例新闻/来源数据（演示可复现）
```

## 4. 核心组件设计

### 4.1 声明式 Agent 定义（借鉴 CrewAI 长处）

```yaml
# config/agents/fact_check_team.yaml
team:
  name: fact-check-squad
  agents:
    - id: planner          # 规划者
      role: 事实核查规划员
      goal: 把待核声明拆解为可并行核查的子任务
      backstory: 资深调查编辑，擅长识别声明中的关键事实点
      tools: []
    - id: fact_checker     # 事实核查员（4 实例并行，同一角色）
      role: 事实核查员
      goal: 用来源核实指定事实点，给出 支持/反对/存疑 + 证据
      backstory: 严谨的核查记者，绝不凭记忆下结论
      tools: [web_search, read_file]
      replicas: 4
      approval_level: read-only
    - id: debater_pro       # 正方辩手
      role: 主张方辩手
      goal: 为该声明可信性给出最强论证
      ...
    - id: debater_con       # 反方辩手
    - id: judge             # 裁判
      role: 事实核查裁判
      goal: 依据双方论证与证据裁决：真实/虚假/部分属实/证据不足
    - id: editor            # 编辑
      role: 事实核查编辑
      goal: 汇总成结构化核查报告（判定+证据链+结论）
      tools: [write_report]  # approval_level: write → 需人工审批
```

### 4.2 消息协议（A2A 思想）

```json
{ "id": "msg-...", "from": "planner", "to": "fact_checker",
  "type": "task|result|query|decision",
  "payload": {"content": "...", "meta": {}},
  "ts": "2026-08-14T..." }
```

Agent 只有 inbox/outbox，不做直接调用 → 每条消息进审计日志 → 全程可回放（OpenHands 事件流思想）。`type` 采用字符串枚举并带 pydantic 校验。

### 4.3 共享记忆（memory.py）

- 短期：轮次消息缓冲（每 agent 可见范围可配：team 级共享 / agent 私有）
- 长期：`facts.json` 结构化事实库——研究员写入"claim → verdict → evidence → sources"，编辑读取生成报告 → 面试可讲共享 vs 隔离记忆的权衡

### 4.4 4 种协作模式（patterns/，LangGraph StateGraph）

| 模式 | 流程 | 并发 | 适用 |
|---|---|---|---|
| pipeline | A→B→C 接力 | 无 | 基线对比 |
| parallel | 规划者拆解 → N 研究员线程池并发 → 汇总者 | **真并发** | 事实核查主模式 |
| supervisor | 监督者循环调度（动态再分工、失败重试≤2） | 有 | 任务动态变化 |
| debate | 正/反方交替发言 2 轮 → 裁判裁决 | 无（回合制） | 结论冲突裁决 |

所有模式输出统一 `AgentResult { answer, audit_trace, facts, tokens }`。

### 4.5 MCP 客户端（tools/mcp_client.py）

标准 stdio MCP client：spawn `command + args`（如 `node E:\Agent\vision-mcp\server.js`），
`listTools` → 注册为框架工具（命名 `mcp__<server>__<tool>`），`callTool` 代理。演示：可把 vision MCP 挂为"图片证据核查"工具。

### 4.6 人工审批（runtime/approval.py，借鉴 codex-cli）

工具声明 `approval_level: read-only | write | execute`。read-only 自动放行；write/execute 进入审批队列，
CLI/Gradio 端确认（一次会话可"全放行"）。演示脚本提供自动批准模式。

### 4.7 评测（eval/，延续评测驱动风格）

- 8 条声明（真/假/部分属实/证据不足各 2 条），offline 样例数据保证可复现
- 指标：判定准确率、信息覆盖度（命中关键事实点比例）、token 成本、端到端延迟
- 基线：单 agent（同一 LLM，ReAct 多工具）跑同一评测集 → 多 agent 覆盖度应显著占优
- 输出 `eval/report.md` + 数据写入 README

## 5. 演示流程（新闻事实核查）

用户输入声明（如"某地 2026 年宣布全面禁止燃油车"）：
planner 拆解 → 4 个 fact_checker 并行核查（各自 web_search/样例数据）→ 写 facts.json →
debater_pro / debater_con 各 1 轮论证 → judge 裁决 → editor 出报告（判定 + 证据链 + 结论）→
Gradio 展示消息流 + 审计回放；敏感操作触发审批弹窗。

## 6. 里程碑（1-2 周）

1. M1 骨架：项目结构 + llm_client + message + YAML 加载 + pipeline 模式跑通（1 天）
2. M2 协作：parallel + supervisor + debate + 共享记忆（3 天）
3. M3 工具与安全：builtin 工具 + 沙箱 + MCP 客户端 + 审批（2 天）
4. M4 演示与评测：fact-check 团队 + offline 样例 + Gradio + 评测报告（2 天）
5. M5 收尾：README 架构图 + 单测补全 + 面试问答文档 INTERVIEW_PREP.md（2 天）

## 7. 风险与边界

- gpt-5.5 中转成本：offline 样例数据模式 + 单次演示 ≤ 30 次 LLM 调用
- LangGraph 版本与 Python 3.11 兼容性：conda 隔离环境，requirements 锁版本
- MCP 客户端实现复杂度：若时间紧降级为"演示挂 vision MCP"，框架只保留 stdio 通用客户端
- 不在范围：Web 部署、多用户、持久化数据库（JSON 足够）

## 8. 多 agent 协作实现方式（本项目本身的开发编排）

由 DSH 编排 4-5 个并行实现子 agent：
- A：core 组（agent/message/memory/llm_client + 测试）
- B：patterns 组（4 模式 + 测试）
- C：tools+mcp 组（registry/builtin/mcp_client/approval + 测试）
- D：demo+eval+ui 组（fact_check 团队配置/offline 样例/Gradio/评测脚本）
- E：docs 组（README/INTERVIEW_PREP）
接口契约先行（`docs/contracts.md`），我负责集成、全量测试与验收。

## 9. 验收标准

- [ ] `pytest` 全绿（≥ 30 用例）
- [ ] 4 种模式各自端到端跑通，parallel 模式有并发证据（时间戳/线程 id）
- [ ] 事实核查 demo：8 条评测任务完成，多 agent 信息覆盖度 > 单 agent 基线，报告含数据
- [ ] 审计回放可重放完整消息轨迹；审批流可演示
- [ ] Gradio 可交互演示；README 含架构图与评测表；INTERVIEW_PREP.md 可支撑面试问答
