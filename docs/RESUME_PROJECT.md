# EvidenceRoute 简历与面试说明

这份说明只使用仓库中已经实现、测试或由确定性代码直接计算出的事实。当前数字来自一次完整且获授权的 Gate A campaign，范围严格限定为冻结的 AVeriTeC dev balanced subset (n=80)。

## 简历定位

**EvidenceRoute：成本感知的自适应事实核查 Agent**

面向 Agent / 大模型应用岗位，重点讲清楚三个工程问题：如何在 single 与 multi 路径之间做可解释路由，如何让模型调用可计费、可恢复、可审计，以及如何防止评测 gold 泄漏和失败样本被静默删除。

## 可直接使用的项目描述

- 基于 LangGraph + Pydantic 构建可审计事实核查 Agent：claim 经确定性特征分析后进入 rule-first adaptive router，在 single / multi 路径间选择；multi 最多 3 个并行 worker，single 最多升级 1 次，所有分支和失败状态均有结构化契约。
- 实现冻结 AVeriTeC claim-only 证据检索与 scorer 隔离：运行图只读取 runtime manifest，gold 与官方 evaluator 只在评测边界加载；输出强制校验 verdict、引用、覆盖度和状态一致性，避免把检索不到证据误写成确定结论。
- 设计 SQLite 幂等调用账本与整数 micro-CNY 预算预留：外层 artifact 与嵌套 result 的 usage/cost 必须一致；usage 缺失、账单不确定、模型身份漂移或预算不足时安全停止，不把 partial/failed 结果伪装成预测标签。
- 为 Gate A 建立可恢复交错评测：固定 32 条 train calibration、80 条 dev 和 20 条 stability claim 的调用上界分别为 1,544 base / 3,088 repair / 9,264 fault transport attempts，启动预算含 20% reserve，客户端上限为 CNY 500；Gate A campaign 完成 280/280 个 work item，762 个账本调用全部结账。
- 在同一冻结输入与模型配置下，adaptive 在 80 条 dev 上达到 full-manifest macro-F1 0.392、准确率 0.412、完成率 90.0%；相对 always_multi 降低 21.8% token、25.9% 实际成本，fresh latency P50/P95 为 7.13s/39.81s。completed-only 的官方 veracity macro-F1 为 0.411，分母为 72/80。

## 数字边界

上面的 1,544 / 3,088 / 9,264 是由固定 manifest 规模、节点上限和重试规则计算出的最坏调用上界，不是实际付费调用次数。离线测试数量也不是模型质量分数。

简历中的质量、成本和延迟数字必须同时附带 cohort、完成率和模型身份限制；不能把 balanced subset 结果写成完整 AVeriTeC leaderboard 成绩。当前 adaptive stability 为 13/20 (65.0%)，低于项目定义的 85% 工程阈值，因此应把稳定性写成待改进项而不是已达标指标。

## 稳定性 v2 实验结论

独立稳定性 v2 实验 `evidence-route-stability-v2-smoke-20260901` 已完成 280/280 个 work item。严格
stability 为 11/20 (55.0%)，低于 17/20 门槛；adaptive dev full-manifest macro-F1 为 0.345、
完成率为 82.5%，相对 always_multi 的 token 降幅为 6.6%，均低于已发布 Gate A baseline 的
0.392、90.0% 和 21.8%。因此 v2 只作为工程失败分析，不替换 baseline，也不作为简历主结果。

v2 的主要不稳定来源是 evidence/citation drift (12/20)，其次是 route drift (3/20)、
validation/status drift (2/20)、incomplete/failed (2/20) 和 provider variance (1/20)。
v2 run-store 的 465 条调用全部为 completed，activity 汇总成本（含继承的 calibration accounting）
为 CNY 81.811188；报告明确标记 relay model identity unverified。上述诊断用于说明下一步应
优先收紧证据选择与引用绑定，不能代替严格 stability 指标。

## 90 秒面试讲法

我做的是一个成本感知的事实核查 Agent。输入 claim 先做确定性分析和冻结证据 probe，再由 rule-first router 选择 single 或 multi。简单 claim 走一次核验；复杂或低置信场景才拆成最多三个并行 worker，再由 judge 聚合，single 只允许一次升级。这样路由策略的收益可以和 always-single、always-multi 在同一 manifest、同一模型配置下比较。

我把难点放在可验证性而不是 prompt 堆叠上。运行侧只读 claim-only manifest，gold 和官方 evaluator 在 scorer 边界；每次模型调用写入 SQLite ledger，call ID、request fingerprint、usage、价格和响应模型 ID 都可回放。预算用整数 micro-CNY 预留，usage 或账单不完整就停止活动。最终报告要求完整 manifest 分母、失败惩罚和发布门禁，避免只挑成功样本报结果。

当前公开的是 Gate A 的可审计实现和一次完整的冻结 provider 评测；provider 是 OpenAI-compatible relay，响应模型身份仍是 self-reported、identity unverified。报告同时保留 full-manifest 分数和 completed-only 官方分数，避免把 partial/failed 样本从分母中静默删除。后续 stability v2 实验没有达到严格门槛，因此简历继续使用已发布 Gate A baseline，并把新实验作为失败分析而不是成绩升级。

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

当前是平衡冻结子集，不代表完整 AVeriTeC leaderboard；模型身份没有供应商认证；Gate B 的 MCP、中文案例和 Streamlit 尚未实现。正式 Gate A activity `evidence-route-gate-a-20260830-clean1` 已完成 calibration、dev 和 stability 三个阶段，campaign 为 **280/280 item complete**，账本为 **762 completed calls / CNY 87.5052**。Gate A adaptive stability 为 **13/20 (65.0%)**，v2 为 **11/20 (55.0%)**；两者都低于 85% 工程阈值，报告没有把稳定性包装成达标结果。

## 离线复现

正式测评可以分批执行：`calibrate --collect --max-cases 4` 每次在完整 calibration case 保存后暂停，`evaluate --max-items 10` 每次在完整 work item 保存后暂停；下一次对同一 activity 使用 `--resume`。暂停是可恢复的执行状态，不是完整结果，也不会绕过 usage、预算、模型漂移或 billing uncertainty 门禁。

稳定性 v2 实验与已发布 Gate A 分离：使用新的 `activity-id`、`campaign-id` 和 `--experiment-dir`，并通过
`--parent-activity`、`--parent-report` 指向 `evidence-route-gate-a-20260830-clean1` 的父基线。实验会在任何
付费 transport 构造前核对父报告、manifest、pricing、config、prompt 和 stability manifest 的哈希；父 activity
和历史 artifact 不会被覆盖。实验结果低于 17/20 稳定 claim 时，只能作为工程失败分析，不能写成达标指标。

v2 activity 已经完成。下面的命令用于在同一 activity 上检查或复现可恢复批处理；`--max-items 10`
可重复使用，每批在完整 item 落盘后暂停：

```powershell
python -m evidence_route.cli evaluate `
  --manifest data/manifests/averitec_dev_runtime.json `
  --stability-manifest data/manifests/averitec_stability_runtime.json `
  --resume --accept-paid-campaign `
  --activity-id evidence-route-stability-v2-smoke-20260901 `
  --campaign-id evidence-route-stability-v2-smoke-20260901 `
  --activity-dir artifacts/evaluation/evidence-route-stability-v2-smoke-20260901 `
  --checkpoint-db artifacts/evaluation/evidence-route-stability-v2-smoke-20260901/checkpoints.sqlite3 `
  --run-store artifacts/evaluation/evidence-route-stability-v2-smoke-20260901/run-store.sqlite3 `
  --experiment-dir artifacts/evaluation/evidence-route-stability-v2-smoke-20260901/identity `
  --parent-activity evidence-route-gate-a-20260830-clean1 `
  --parent-report reports/evidence-route-gate-a-20260830-clean1/final/summary.json `
  --parent-config configs/calibrated.evidence-route-gate-a-20260830-clean1.yaml `
  --config configs/stability-v2.yaml `
  --pricing configs/pricing.local.yaml `
  --corpus-dir data/processed/averitec/corpora `
  --max-items 10
```

后续批次使用相同参数，将 `--start-after-calibration` 改为 `--resume`。真实 provider 调用前必须
执行离线测试和预算预览，并重新确认实验身份、model alias、价格文件和费用授权；出现
`billing_uncertain` 或 model drift 时必须暂停，不能自动重试。

其中 `--parent-config` 必须指向父 Gate A calibration 实际使用的配置，`--config` 指向当前实验配置。
两者可以不同；实验身份会分别记录并校验这两个文件的 SHA-256，避免把 v2 配置误记成历史基线。

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

最后一条命令只计算调用和预算上界，不创建网络 transport，也不会产生 provider 费用。真实 `provider-smoke`、calibration、dev campaign 和 `reports/final` 发布需要单独的 provider、价格文件和明确授权；Gate A 正式 artifact、官方 evaluator 输出和复现摘要位于 `reports/evidence-route-gate-a-20260830-clean1/final`，v2 结果位于 `reports/evidence-route-stability-v2-20260901/`。
