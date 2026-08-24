# EvidenceRoute 简历与面试说明

这份说明只使用仓库中已经实现、测试或由确定性代码直接计算出的事实。真实 provider 的质量、延迟和成本结果必须等一次完整且获授权的 campaign 后再写入简历。

## 简历定位

**EvidenceRoute：成本感知的自适应事实核查 Agent**

面向 Agent / 大模型应用岗位，重点讲清楚三个工程问题：如何在 single 与 multi 路径之间做可解释路由，如何让模型调用可计费、可恢复、可审计，以及如何防止评测 gold 泄漏和失败样本被静默删除。

## 可直接使用的项目描述

- 基于 LangGraph + Pydantic 构建可审计事实核查 Agent：claim 经确定性特征分析后进入 rule-first adaptive router，在 single / multi 路径间选择；multi 最多 3 个并行 worker，single 最多升级 1 次，所有分支和失败状态均有结构化契约。
- 实现冻结 AVeriTeC claim-only 证据检索与 scorer 隔离：运行图只读取 runtime manifest，gold 与官方 evaluator 只在评测边界加载；输出强制校验 verdict、引用、覆盖度和状态一致性，避免把检索不到证据误写成确定结论。
- 设计 SQLite 幂等调用账本与整数 micro-CNY 预算预留：外层 artifact 与嵌套 result 的 usage/cost 必须一致；usage 缺失、账单不确定、模型身份漂移或预算不足时安全停止，不把 partial/failed 结果伪装成预测标签。
- 为 Gate A 建立可恢复交错评测：固定 32 条 train calibration、80 条 dev 和 20 条 stability claim 的调用上界分别为 1,544 base / 3,088 repair / 9,264 fault transport attempts，启动预算含 20% reserve，客户端上限为 CNY 500；当前离线回归套件收集 375 项并全部通过（以最终命令输出为准）。

## 数字边界

上面的 1,544 / 3,088 / 9,264 是由固定 manifest 规模、节点上限和重试规则计算出的最坏调用上界，不是实际付费调用次数。375 是当前离线测试收集数，不是模型质量分数。

以下数字在真实 provider campaign 完成前禁止写入简历或 README：macro-F1、completed-only 官方分数、P50/P95 延迟、实际 CNY 成本、稳定性比例、模型身份和任何“超过某基线”的结论。

## 90 秒面试讲法

我做的是一个成本感知的事实核查 Agent。输入 claim 先做确定性分析和冻结证据 probe，再由 rule-first router 选择 single 或 multi。简单 claim 走一次核验；复杂或低置信场景才拆成最多三个并行 worker，再由 judge 聚合，single 只允许一次升级。这样路由策略的收益可以和 always-single、always-multi 在同一 manifest、同一模型配置下比较。

我把难点放在可验证性而不是 prompt 堆叠上。运行侧只读 claim-only manifest，gold 和官方 evaluator 在 scorer 边界；每次模型调用写入 SQLite ledger，call ID、request fingerprint、usage、价格和响应模型 ID 都可回放。预算用整数 micro-CNY 预留，usage 或账单不完整就停止活动。最终报告要求完整 manifest 分母、失败惩罚和发布门禁，避免只挑成功样本报结果。

当前公开的是 Gate A 的离线实现和上界验证；provider 是 OpenAI-compatible relay，响应模型身份仍是 self-reported、identity unverified。真实质量数字需要在冻结输入、预算和授权都满足后单独跑一次，不能用本地 fixture 代替。

## 高频追问

### 为什么不是所有 claim 都走 multi？

multi 的 worker、judge 和输入 token 成本明显更高。路由器先利用 claim clause 数、实体/数字密度、probe 来源数和冲突提示做确定性判断，只有不确定时才让 LLM 参与路由；工程上优先保证上界可算和失败可停。

### 为什么要同时保留 fixed single / fixed multi？

没有固定策略对照，就无法区分“adaptive 真的节省成本”与“数据本身更容易”。三种策略共享 manifest、prompt、价格和 scorer，比较时保留完整分母。

### 如何防止评测泄漏？

runtime manifest 只保存 claim 和 corpus/evidence identity，不包含 label、question、justification 或 gold 字段；scorer manifest 独立加载。仓库测试还会审计运行进程导入边界、manifest hash 和 Git freeze identity。

### 为什么 usage/cost 要做两层一致性校验？

只校验 result 还不够：调用账本、外层 artifact 和嵌套 result 可能出现不同步。发布前要求三者的 token、成本、call ID、价格配置和 reservation 覆盖关系一致；否则只能生成 diagnostic bundle，不能写入 `reports/final`。

### 这个项目当前的限制是什么？

当前是平衡冻结子集，不代表完整 AVeriTeC leaderboard；模型身份没有供应商认证；Gate B 的 MCP、中文案例和 Streamlit 尚未实现；真实付费 campaign 未授权时不运行，也不伪造结果。

## 离线复现

在仓库根目录执行：

```powershell
conda run -n agent-collab python -m pytest -m "not network and not live and not paid" -q
conda run -n agent-collab python -m ruff check src tests scripts
conda run -n agent-collab python -m evidence_route.cli evaluate `
  --manifest tests/fixtures/evaluation/runtime.json `
  --stability-manifest tests/fixtures/evaluation/runtime.json `
  --config configs/default.yaml `
  --pricing configs/pricing.dryrun.yaml `
  --activity-dir artifacts/offline-preview
```

最后一条命令只计算调用和预算上界，不创建网络 transport，也不会产生 provider 费用。真实 `provider-smoke`、calibration、dev campaign 和 `reports/final` 发布需要单独的 provider、价格文件和明确授权。
