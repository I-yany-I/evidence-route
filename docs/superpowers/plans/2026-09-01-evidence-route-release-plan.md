# EvidenceRoute 秋招交付收口实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 EvidenceRoute 从“Gate A 基线已发布、稳定性 v2 smoke 已验证”收口为可复现、可审计、可写入秋招简历的重点项目。

**Architecture:** 保持已发布的 `evidence-route-gate-a-20260830-clean1` 不可变；继续使用独立的 `evidence-route-stability-v2-smoke-20260901` activity 和 SQLite ledger 分批完成 v2 评测。所有代码、配置、价格、语料和 Git freeze 先通过离线门禁，付费调用只在明确预算授权后进行；最终报告同时保留原始 baseline 和 v2 before/after 结果，不因实验未达标而覆盖 baseline。

**Tech Stack:** Python 3.11, Pydantic v2, LangGraph, SQLite, Typer, pytest, ruff, AVeriTeC frozen runtime/scorer manifests。

---

## 当前基线与验收标准

- 已发布 Gate A baseline：adaptive dev full-manifest macro-F1 `0.392`，completion `90.0%`，stability `13/20`。
- hardening 历史实验：stability `14/20`，未达到 `17/20`，不能替换 baseline。
- v2 smoke：`1/280` item 完成，`279` item pending，activity/campaign 均为 `paused`，账务已闭合。
- v2 smoke 总实际成本：`17,883,216 micro-CNY`，约 `17.8832 CNY`；该数字包含继承 calibration accounting。
- v2 的发布条件：完整 280 item、所有 ledger call completed、usage/cost 完整、无 model drift、strict stability 至少 `17/20`，且 adaptive dev 不低于 baseline 的质量/完成率门槛。
- 若 v2 未达到发布条件：保留 baseline 作为主结果，v2 只写成稳定性工程实验和失败分析。

## File Map

- Modify: `src/evidence_route/evaluation/runner.py` only when a new recovery/idempotency regression is reproduced.
- Modify: `src/evidence_route/evaluation/production_evaluation.py` only when an offline test demonstrates a state, freeze, or accounting defect.
- Modify: `src/evidence_route/evaluation/lifecycle.py` only for freeze/recovery audit defects covered by tests.
- Test: `tests/test_campaign_runner.py`, `tests/test_production_services.py`, `tests/test_lifecycle.py` for every behavior change.
- Modify: `README.md` and `docs/RESUME_PROJECT.md` only after the final report contains measured v2 values.
- Create: no new paid activity directory; continue `artifacts/evaluation/evidence-route-stability-v2-smoke-20260901`.

### Task 1: 无付费收口检查

**Files:**
- Read: `artifacts/evaluation/evidence-route-stability-v2-smoke-20260901/activity.json`
- Read: `artifacts/evaluation/evidence-route-stability-v2-smoke-20260901/campaign.json`
- Read: `artifacts/evaluation/evidence-route-stability-v2-smoke-20260901/run-store.sqlite3`
- Read: `reports/evidence-route-gate-a-20260830-clean1/final/summary.json`
- Test: `tests/test_campaign_runner.py`, `tests/test_production_services.py`, `tests/test_lifecycle.py`

- [x] **Step 1: 核对现有 v2 activity。**

确认 activity/campaign 都是 `paused`、`billing_uncertain=false`，首个 item 为 `completed`，其余 279 个 item 为 `pending`；确认原始 `evidence-route-gate-a-20260830-clean1` 的 activity、campaign、report 文件 SHA-256 未变化。

- [x] **Step 2: 运行离线验证。**

```powershell
C:\Users\ASUS\miniconda3\envs\agent-collab\python.exe -m pytest -m "not network and not live and not paid" -q
C:\Users\ASUS\miniconda3\envs\agent-collab\python.exe -m ruff check src tests scripts
C:\Users\ASUS\miniconda3\envs\agent-collab\python.exe -m compileall -q src tests scripts
C:\Users\ASUS\miniconda3\envs\agent-collab\python.exe -m pip check
```

预期：测试、lint、编译和依赖检查全部通过；pytest cache 权限 warning 可记录但不能把它当成测试失败。

- [x] **Step 3: 只做不产生网络调用的预算预览。**

使用当前 v2 参数运行不带 `--accept-paid-campaign` 的 `evaluate`，记录剩余 campaign 的 worst-case reservation、预计总成本和每批 `--max-items 5/10` 的调用上界；不构造 provider transport，不修改父 baseline。

- [x] **Step 4: 提交或记录检查结果。**

如果只发现文档或脚本问题，单独提交；如果发现 accounting/freeze 问题，先按 TDD 增加失败测试，禁止直接手改历史 JSON 或 SQLite。

### Task 2: 恢复与账务幂等性收口

**Files:**
- Modify: `src/evidence_route/evaluation/runner.py` only if required by the tests.
- Modify: `src/evidence_route/evaluation/production_evaluation.py` only if required by the tests.
- Test: `tests/test_campaign_runner.py`, `tests/test_production_services.py`, `tests/test_lifecycle.py`

- [x] **Step 1: 写恢复回归测试。**

覆盖同一个 paused activity 连续执行两次 `--resume` 的行为：第一次只执行 pending item；第二次不能重新执行已完成 item，不能新增重复 artifact，且 run-store 中已完成 call 的 usage/cost 不得重复计入。另测普通 resume 仍拒绝未经授权的 `billing_uncertain`。

- [x] **Step 2: 运行测试确认失败。**

```powershell
C:\Users\ASUS\miniconda3\envs\agent-collab\python.exe -m pytest tests/test_campaign_runner.py tests/test_production_services.py tests/test_lifecycle.py -q
```

- [x] **Step 3: 实现最小修复并保持普通冻结门禁。**

授权 recovery 只允许 ledger call 为 `reserved`、存在 `billing_recovery_events`、Git HEAD 是 frozen protocol commit 的后代且工作区 clean；普通 start/resume 不允许使用该例外。

- [x] **Step 4: 重新运行相关测试并提交。**

```powershell
C:\Users\ASUS\miniconda3\envs\agent-collab\python.exe -m ruff check src tests scripts
C:\Users\ASUS\miniconda3\envs\agent-collab\python.exe -m pytest tests/test_campaign_runner.py tests/test_production_services.py tests/test_lifecycle.py -q
git add src/evidence_route/evaluation/runner.py src/evidence_route/evaluation/production_evaluation.py src/evidence_route/evaluation/lifecycle.py tests/test_campaign_runner.py tests/test_production_services.py tests/test_lifecycle.py
git commit -m "test: close resumable campaign accounting gaps"
```

### Task 3: 分批完成稳定性 v2 评测

**Files:**
- Modify: none before the paid readiness gate passes.
- Read: `docs/superpowers/plans/2026-09-01-stability-v2-plan.md`
- Write: existing files under `artifacts/evaluation/evidence-route-stability-v2-smoke-20260901`

- [ ] **Step 1: 明确新的付费授权。**

本阶段不沿用此前“一次可能重复计费恢复重试”的授权。开始剩余 279 item 前，需要单独确认本轮预算上限；建议先按 `--max-items 5` 或 `10` 分批，每批结束检查账务后再继续。

- [ ] **Step 2: 保持电脑唤醒并运行第一批。**

沿用已验证的参数、`https://api.ssstoken.net/v1`、`configs/stability-v2.yaml`、同一 activity、同一 run-store 和同一 corpus；只把 `--resume` 和 `--accept-paid-campaign` 保留，批大小从预算预览结果确定。

- [ ] **Step 3: 每批完成后核对四类状态。**

检查 `campaign.json` 的 completed/pending 数量；检查 activity 的 `billing_uncertain`、stop reason 和 phase status；检查 SQLite 中所有新增 call 的 state、usage、actual cost；检查 artifact SHA-256 与 state link 一致。出现 billing uncertainty、model drift 或 freeze mismatch 时停止，不自动重试。

- [ ] **Step 4: 处理进程中断。**

电脑休眠、网络断开或进程中断后只对同一 activity 使用 `--resume`；不新建 activity、不删除 diagnostic artifact、不覆盖 Gate A parent 目录。若出现账单不确定，必须先保存 provider/ledger 证据并获得针对具体 call 的恢复授权。

- [ ] **Step 5: 完成 280/280 后生成报告。**

使用现有 report CLI 生成 v2 summary、stability diagnostics、before/after comparison 和 resume snippet；报告必须标注 `identity_verified=false`，并同时给出 full-manifest 与 completed-only 分母。

### Task 4: 结果决策与简历材料收口

**Files:**
- Modify: `README.md`
- Modify: `docs/RESUME_PROJECT.md`
- Write: v2 report files under `reports/evidence-route-stability-v2-20260901/` only after Task 3 completes

- [ ] **Step 1: 按发布门禁决定主结果。**

只有 v2 达到 `17/20` strict stability、质量/完成率不低于 baseline 且所有 accounting gate 通过，才允许把 v2 写成改进结果；否则 README 和简历继续使用 Gate A baseline，并将 v2 记为未达标工程实验。

- [ ] **Step 2: 更新 README 结果区块。**

只写报告中实际测得的 cohort、macro-F1、completion、token/cost、latency、stability、分母和模型身份限制；禁止把 balanced subset 写成完整 AVeriTeC leaderboard 成绩。

- [ ] **Step 3: 更新简历项目说明。**

保留“成本感知路由、证据隔离、SQLite 账本、可恢复评测”四条主线；加入一条可验证的 before/after 结果和一条诚实 limitation。简历主 bullet 不使用未达标 v2 指标作为成功表述。

- [ ] **Step 4: 做最终可复现检查。**

```powershell
C:\Users\ASUS\miniconda3\envs\agent-collab\python.exe -m pytest -q
C:\Users\ASUS\miniconda3\envs\agent-collab\python.exe -m ruff check src tests scripts
C:\Users\ASUS\miniconda3\envs\agent-collab\python.exe -m compileall -q src tests scripts
git diff --check
```

只有命令输出支持的结论才进入最终 README、简历片段和面试讲稿。

## 推荐执行顺序

先做 Task 1 和 Task 2，确认恢复/账务状态机稳定；随后做一次新的离线预算预览。预算在可接受范围内并获得新的付费授权后，按 Task 3 分批跑完 v2；最后执行 Task 4。当前最合理的批大小是 `5` 或 `10`，不建议一次执行 279 个 pending item。

## 时间估计

- Task 1：约 `30-60` 分钟。
- Task 2：约 `1-2` 小时，取决于是否发现新的状态机缺口。
- Task 3：网络稳定时约 `4-8` 小时，建议分多个批次执行；网络或账单不确定时会延长。
- Task 4：约 `1-2` 小时。

整个交付预计 `1-2` 个工作日；其中真正不可压缩的是付费评测的接口响应时间和每批完成后的账务核验。
