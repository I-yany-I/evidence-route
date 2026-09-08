# EvidenceRoute v2g 质量恢复评测说明

本文记录 `evidence-route-quality-recovery-v2g-20260907` 的最终状态与可使用口径。它是冻结
AVeriTeC balanced subset 上的工程评测，不是完整 benchmark leaderboard 成绩。响应模型 ID 为
OpenAI-compatible relay 自报的 `gpt-5.6-sol`，`identity_verified=false`，不能写成供应商已认证。

## 活动终态

- calibration、dev、stability 三阶段均为 `complete`；活动不再继续付费调用。
- campaign 共 280 个 work item：270 completed、10 failed、0 pending、0 running。
- 账本记录 744 次 fresh call、810 次 transport attempt；总 usage 为 1,171,390 input tokens、
  216,262 output tokens，共 1,387,652 tokens。
- usage 完整、`billing_uncertain=false`，活动实际成本为 CNY 88.882632。

## 完整分母结果

| 策略 | Macro-F1 | Accuracy | Completion | Tokens | 实测成本（CNY） |
|---|---:|---:|---:|---:|---:|
| always_single | 0.328 | 0.350 | 100.0% | 231,387 | 14.045652 |
| always_multi | 0.283 | 0.338 | 91.2% | 429,786 | 28.579356 |
| adaptive | 0.317 | 0.350 | 97.5% | 310,786 | 19.440396 |

adaptive 相对 always_multi 减少 27.7% token 和 32.0%实测成本，同时 macro-F1 高 0.034；
但它仍低于 always_single 的 0.328，因此不能声称 adaptive 已取得最佳质量。adaptive 的 80 条
dev 中 78 completed、2 failed，失败均计入完整分母。

## 路由与稳定性

adaptive 的路由分布为 64 条 single、16 条 multi；LLM router 使用率为 28.75%。fresh latency
P50 为 20.10 秒，P95 为 73.32 秒。20 条 stability claim 各重复三次，严格 verdict 一致率为
16/20（80.0%，Wilson 95% 区间约为 58.4% 至 91.9%），低于项目要求的 17/20。

稳定性诊断记录了 11 条 evidence/citation drift、2 条 route drift、2 条 incomplete/failed 和
1 条 provider variance。类别可能同时描述同一 claim，因此这些计数用于定位问题，不应简单相加
为失败 claim 数。

## 离线检索门禁

source-aware 混合检索在 32 条 train calibration 上达到：

- candidate source hit：22/32，达到要求的 22/32；
- final top-8 hit：20/32，高于要求的 14/32；
- sentinel `train-2468` 在 final top-8 中命中；
- 检索门禁整体 `passed=true`。

这说明早期 sentence-level BM25 的候选召回瓶颈已有明显改善。端到端 macro-F1 没有同步达到
0.392，表明后续瓶颈已经转向证据利用、冲突证据判断、验证覆盖和评测证据完整性。

## 为什么不能发布

最终诊断报告为 `publishable=false`。直接质量门禁要求 adaptive macro-F1 至少 0.392，实际为
0.317；稳定性门禁要求至少 17/20，实际为 16/20。此外，报告审计还发现 calibration activity
identity、calibration freeze/replay、run-store call set、当前 prompt bundle、NLTK 数据和 recovery
evidence 不完整或不匹配。这些问题不改变已经落盘的策略指标，但阻止它们成为正式发布结果。

## 可用于简历的表述

可以写：在冻结 dev balanced subset（n=80）完整分母上，adaptive 完成率 97.5%、macro-F1
0.317；相对 always_multi 降低 27.7% token 和 32.0%实测成本；三次重复稳定性为 16/20，未达
17/20 发布门槛，失败样本和审计原因完整保留。

不能写：完整 AVeriTeC 达到 0.317、稳定性已经达标、adaptive 质量优于所有固定策略、模型身份
已经由供应商认证，或 v2g 是可发布的最终成绩。
