# AgentCollab 实施计划

> **For agentic workers:** 本计划由 DSH 以 subagent-driven-development 方式执行：
> 每个任务派一个全新实现子 agent（附完整契约文本），完成后依次做「规格符合性评审 → 代码质量评审」，
> 有异议回给实现者修复再评；全部任务完成后做一次整体代码评审与集成验证。
> 接口契约见同目录 `contracts.md`（各任务的契约段随任务文本一并下发给子 agent）。

**Goal:** 在 `E:\简历\简历项目\agent-collab` 实现 AgentCollab 多智能体协作框架 MVP（4 协作模式 + 消息协议 + 共享记忆 + MCP + 审批 + 审计 + 评测 + Gradio），演示实例为新闻事实核查团队。

**Architecture:** LangGraph 状态图承载 4 种协作模式；声明式 YAML 定义团队；A2A 风格消息信封贯穿全程并全量审计；LLM 为 OpenAI 兼容中转 gpt-5.5。

**Tech Stack:** Python 3.11（conda env `agent-collab`）、langgraph、openai、pydantic、pyyaml、gradio、pytest、duckduckgo-search。

---

### Task 0: 项目骨架与仓库初始化（编排者执行）

**Files:** 创建 `agent-collab/` 骨架：`src/agent_collab/__init__.py`、`tests/conftest.py`、`tests/__init__.py`、`config/llm.yaml`（模板）、`.gitignore`、`pyproject.toml`（pytest 配置）；`git init` + 初始提交（docs/ 与骨架）。
验收：`conda run -n agent-collab pytest` 无错误（0 用例也算通过）。

### Task 1: core 组 — LLM 客户端 / 消息协议 / 共享记忆 / Agent 运行时

**Files:**
- Create: `src/agent_collab/core/__init__.py`、`llm_client.py`、`message.py`、`memory.py`、`agent.py`
- Test: `tests/test_core_message.py`、`tests/test_core_memory.py`、`tests/test_core_llm_client.py`、`tests/test_core_agent.py`

**契约:** contracts.md §1-4。
**要点:** AgentRuntime 的 LLM 循环必须可用 mock LLM 驱动（不真实联网）；LLMClient 用 `openai` SDK 直连 `base_url`；json_complete 用 response_format json_schema 并容错重试；测试全 mock（monkeypatch openai 调用）。
**验收:** `pytest tests/test_core_*.py` 全绿（≥15 用例）。

### Task 2: tools + runtime 组 — 工具注册表 / 内置工具 / MCP 客户端 / 审计 / 审批

**Files:**
- Create: `src/agent_collab/tools/__init__.py`、`registry.py`、`builtin.py`、`mcp_client.py`
- Create: `src/agent_collab/runtime/__init__.py`、`audit.py`、`approval.py`
- Test: `tests/test_tools_registry.py`、`tests/test_tools_builtin.py`、`tests/test_tools_mcp_client.py`、`tests/test_runtime_audit.py`、`tests/test_runtime_approval.py`

**契约:** contracts.md §5-9。
**要点:** builtin 的 python_repl 必须 subprocess 隔离（5s 超时/截断/临时目录），read_file 白名单三层防护（延续项目三风格）；MCP 客户端测试用一个假的 stdio 服务器脚本（fixture 生成）验证 connect/listTools/call 链路；approver 的 CLI input() 可注入（测试用 monkeypatch）。
**验收:** 对应测试全绿（≥15 用例）。

### Task 3: patterns 组 — 4 种协作模式

**Files:**
- Create: `src/agent_collab/patterns/__init__.py`、`base.py`、`pipeline.py`、`parallel.py`、`supervisor.py`、`debate.py`
- Test: `tests/test_patterns_pipeline.py`、`tests/test_patterns_parallel.py`、`tests/test_patterns_supervisor.py`、`tests/test_patterns_debate.py`

**契约:** contracts.md §10。
**要点:** parallel 用 asyncio.gather 真并发（测试断言并发证据：mock LLM 里记录并发窗口重叠）；supervisor 失败重试逻辑要测；debate 轮次与观点传递要测；所有模式用 FakeLLM/FakeRegistry 驱动，不联网。
**验收:** 对应测试全绿（≥12 用例）。

### Task 4: demo + eval + ui 组 — 事实核查团队与演示

**Files:**
- Create: `config/agents/fact_check_team.yaml`、`config/workflow.yaml`、`samples/`（8 条评测声明 + 样例来源数据）
- Create: `src/agent_collab/demo/fact_check_demo.py`、`src/agent_collab/eval/dataset.py`、`src/agent_collab/eval/metrics.py`、`src/agent_collab/eval/run_eval.py`、`src/agent_collab/ui/app.py`
- Test: `tests/test_eval_dataset.py`、`tests/test_eval_metrics.py`

**契约:** contracts.md §11；设计文档 §4.7/§5。
**要点:** 团队 YAML 按设计 6 角色（planner/fact_checker×4/debater_pro/debater_con/judge/editor）；offline 模式：web_search 在 offline_data 非空时改为样例数据检索（在 builtin 之外由 demo 组装一个样例搜索 handler 注入 registry）；Gradio 展示消息流（audit.replay 渲染）+ 终答；评测 8 题含真/假/部分属实/证据不足各 2 条。
**验收:** 配置可被 YAML 加载；单 agent 基线（直接用一个 AgentRuntime 多工具循环，不经过模式）与多 agent 各跑通 1 条任务（mock 或真实均可，真实限 1 条控制成本）；pytest 全绿。

### Task 5: docs 组 — README / 面试文档 / 依赖清单

**Files:**
- Create: `README.md`（架构图 mermaid、演示结果、评测表、快速开始）、`docs/INTERVIEW_PREP.md`（50+ 模拟问答：为什么 4 种模式、消息协议 vs 直接调用、共享记忆权衡、并发实现、MCP 标准化、审计与审批、评测结论）、`requirements.txt`（锁定主要依赖）
**要点:** 与项目二/三的面试文档风格一致（表格+必背数字）；不得虚构评测数字，占位用「待评测填入」。
**验收:** 文档齐全、无占位符残留（评测数字处明确标注待填）。

### Task 6: 集成验证与收尾（编排者执行）

- 全量 `pytest` 绿；`conda run -n agent-collab python -m agent_collab.demo.fact_check_demo --offline` 端到端跑通一条任务
- 真实评测：8 题 ×（单 agent 基线 + 多 agent parallel）跑评测集，产出 `eval/report.md` 并回填 README 数字
- 最终整体 code review（派一个 reviewer 子 agent 全仓审查）
- `git commit` 收尾；更新目标 goal 状态

## Self-Review

- Spec 覆盖：§3 目录 → Task 0-5；§4 组件 → Task 1-4；§5 演示 → Task 4；§6 里程碑 → Task 顺序；§8 多 agent 编排 → Task 1-5 的实现者分工 + Task 6 集成；§9 验收 → 各 Task 验收 + Task 6。
- 占位符：无 TBD；评测数字明确「待填」并在 Task 6 回填。
- 一致性：契约函数签名与设计文档组件描述一致（Message sender/recipient、ToolRegistry.execute(approver)、AgentResult 字段、build_team replicas）；boundaries 防止子 agent 互相改文件。
