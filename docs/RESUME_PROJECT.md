# EvidenceRoute 简历与面试说明

这份说明只使用仓库中已经实现、测试或由确定性代码直接计算出的事实。当前数字来自一次完整且获授权的 Gate A campaign，范围严格限定为冻结的 AVeriTeC dev balanced subset (n=80)。

## 简历定位

**EvidenceRoute：成本感知的自适应事实核查 Agent**

面向 Agent / 大模型应用岗位，重点讲清楚三个工程问题：如何在 single 与 multi 路径之间做可解释路由，如何让模型调用可计费、可恢复、可审计，以及如何防止评测 gold 泄漏和失败样本被静默删除。

## 可直接使用的项目描述

- 基于 LangGraph + Pydantic 构建可审计事实核查 Agent：claim 经确定性特征分析后进入 rule-first adaptive router，在 single / multi 路径间选择；multi 最多 3 个并行 worker，single 最多升级 1 次，所有分支和失败状态均有结构化契约。
- 实现冻结 AVeriTeC claim-only 证据检索与 scorer 隔离：运行图只读取 runtime manifest，gold 与官方 evaluator 只在评测边界加载；输出强制校验 verdict、引用、覆盖度和状态一致性，避免把检索不到证据误写成确定结论。
- 设计 SQLite 幂等调用账本与整数 micro-CNY 预算预留：外层 artifact 与嵌套 result 的 usage/cost 必须一致；usage 缺失、账单不确定、模型身份漂移或预算不足时安全停止，不把 partial/failed 结果伪装成预测标签。
- 为 Gate A 建立可恢复交错评测：固定 32 条 train calibration、80 条 dev 和 20 条 stability claim 的调用上界分别为 1,544 base / 3,088 repair / 9,264 fault transport attempts，启动预算含 20% reserve，客户端上限为 CNY 500；完整 campaign 共完成 280/280 个 work item，所有 762 个账本调用均已结账。
- 在同一冻结输入与模型配置下，adaptive 在 80 条 dev 上达到 full-manifest macro-F1 0.392、准确率 0.412、完成率 90.0%；相对 always_multi 降低 21.8% token、25.9% 实际成本，fresh latency P50/P95 为 7.13s/39.81s。completed-only 的官方 veracity macro-F1 为 0.411，分母为 72/80。

## 数字边界

上面的 1,544 / 3,088 / 9,264 是由固定 manifest 规模、节点上限和重试规则计算出的最坏调用上界，不是实际付费调用次数。离线测试数量也不是模型质量分数。

简历中的质量、成本和延迟数字必须同时附带 cohort、完成率和模型身份限制；不能把 balanced subset 结果写成完整 AVeriTeC leaderboard 成绩。当前 adaptive stability 为 13/20 (65.0%)，低于项目定义的 85% 工程阈值，因此应把稳定性写成待改进项而不是已达标指标。

## 稳定性加固实验结论

独立实验 `evidence-route-hardening-20260901` 已完成 280/280 个 work item。严格 Gate A stability 从 13/20 提升到 14/20，仍低于 17/20 门槛；adaptive dev 完成率从 90.0% 降至 88.8%，full-manifest macro-F1 从 0.392 降至 0.308，因此本轮只作为工程失败分析，不替换已发布 baseline，也不作为简历主结果。

诊断口径允许有效 partial 参与比较，verdict consistency 为 16/20 -> 17/20，category-free claim 为 4/20 -> 11/20（7 条改善、0 条回退）。这两个数字用于定位 evidence/citation、route 和 status drift，不能代替严格 stability 指标。实验独立账本为 583 个 completed call / CNY 69.864480；activity 汇总包含继承的 calibration accounting，共 756 个 call / CNY 87.481944。

本轮还发现历史 `experiment.json` 将 `parent_prompt_sha256` 记录成了当前实验 prompt hash。历史文件保持不可变并在对比报告中标注；源码已修复为从父报告读取父 prompt hash，同时单独校验当前实验 prompt，避免后续实验混淆父子身份。

## 90 秒面试讲法

我做的是一个成本感知的事实核查 Agent。输入 claim 先做确定性分析和冻结证据 probe，再由 rule-first router 选择 single 或 multi。简单 claim 走一次核验；复杂或低置信场景才拆成最多三个并行 worker，再由 judge 聚合，single 只允许一次升级。这样路由策略的收益可以和 always-single、always-multi 在同一 manifest、同一模型配置下比较。

我把难点放在可验证性而不是 prompt 堆叠上。运行侧只读 claim-only manifest，gold 和官方 evaluator 在 scorer 边界；每次模型调用写入 SQLite ledger，call ID、request fingerprint、usage、价格和响应模型 ID 都可回放。预算用整数 micro-CNY 预留，usage 或账单不完整就停止活动。最终报告要求完整 manifest 分母、失败惩罚和发布门禁，避免只挑成功样本报结果。

当前公开的是 Gate A 的可审计实现和一次完整的冻结 provider 评测；provider 是 OpenAI-compatible relay，响应模型身份仍是 self-reported、identity unverified。报告同时保留 full-manifest 分数和 completed-only 官方分数，避免把 partial/failed 样本从分母中静默删除。后续 stability hardening 实验没有达到严格门槛，因此简历继续使用已发布 Gate A baseline，并把新实验作为失败分析而不是成绩升级。

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

当前是平衡冻结子集，不代表完整 AVeriTeC leaderboard；模型身份没有供应商认证；Gate B 的 MCP、中文案例和 Streamlit 尚未实现。正式 activity `evidence-route-gate-a-20260830-clean1` 已完成 calibration、dev 和 stability 三个阶段，campaign 为 **280/280 item complete**，账本为 **762 completed calls / CNY 87.5052**。adaptive 的 stability 为 **13/20 (65.0%)**，需要后续从 prompt、证据覆盖和重复运行一致性继续优化；报告没有把该指标包装成达标结果。

## 离线复现

正式测评可以分批执行：`calibrate --collect --max-cases 4` 每次在完整 calibration case 保存后暂停，`evaluate --max-items 10` 每次在完整 work item 保存后暂停；下一次对同一 activity 使用 `--resume`。暂停是可恢复的执行状态，不是完整结果，也不会绕过 usage、预算、模型漂移或 billing uncertainty 门禁。

稳定性加固实验与已发布 Gate A 分离：使用新的 `activity-id`、`campaign-id` 和 `--experiment-dir`，并通过
`--parent-activity`、`--parent-report` 指向 `evidence-route-gate-a-20260830-clean1` 的父基线。实验会在任何
付费 transport 构造前核对父报告、manifest、pricing、config、prompt 和 stability manifest 的哈希；父 activity
和历史 artifact 不会被覆盖。实验结果低于 17/20 稳定 claim 时，只能作为工程失败分析，不能写成达标指标。

新实验的命令骨架如下，`--max-items 10` 可重复使用；每批在完整 item 落盘后暂停：

```powershell
python -m evidence_route.cli evaluate `
  --start-after-calibration --accept-paid-campaign `
  --activity-id evidence-route-hardening-20260901 `
  --campaign-id evidence-route-hardening-20260901 `
  --activity-dir artifacts/evaluation/evidence-route-hardening-20260901 `
  --experiment-dir artifacts/evaluation/evidence-route-hardening-20260901/identity `
  --parent-activity evidence-route-gate-a-20260830-clean1 `
  --parent-report reports/evidence-route-gate-a-20260830-clean1/final/summary.json `
  --parent-config configs/calibrated.evidence-route-gate-a-20260830-clean1.yaml `
  --config configs/stability-v2.yaml `
  --max-items 10
```

后续批次使用相同参数，将 `--start-after-calibration` 改为 `--resume`。运行前先执行离线测试和预算预览，
确认新的实验身份、模型 alias、价格文件和预期费用后再授权 provider 调用。

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

最后一条命令只计算调用和预算上界，不创建网络 transport，也不会产生 provider 费用。真实 `provider-smoke`、calibration、dev campaign 和 `reports/final` 发布需要单独的 provider、价格文件和明确授权；本次正式 artifact、官方 evaluator 输出和复现摘要位于 `reports/evidence-route-gate-a-20260830-clean1/final`。
