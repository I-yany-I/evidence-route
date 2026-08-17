# samples — 事实核查离线样例数据

offline 模式下（`config/workflow.yaml` 的 `offline_data` 非空，或 demo 传入 `--offline`），
`web_search` 工具被覆盖为「样例数据检索」：按查询关键词匹配 `sources.jsonl` 返回来源片段，
全程不访问真实网络，保证演示与评测可复现。

## claims.jsonl

每条声明一行 JSON，字段：

| 字段 | 说明 |
|---|---|
| `id` | 声明唯一标识（与 sources.jsonl 的 `claim_id` 对应） |
| `claim` | 待核声明文本 |
| `label` | 金标标签：`true` / `false` / `mixed` / `insufficient`（各 2 条） |
| `key_facts` | 核验关键事实点列表（每条 2-4 个，用于信息覆盖度评测） |

共 8 条声明，主题为 2024-2026 年可核查的科技/政策类中文声明。

## sources.jsonl

每条来源一行 JSON，字段：

| 字段 | 说明 |
|---|---|
| `claim_id` | 所属声明 id |
| `title` | 来源标题 |
| `snippet` | 摘要片段（含支持/反对/存疑证据文本） |
| `url` | 来源链接（占位） |

每条声明配 2-4 条来源；offline 检索按查询关键词与 `title`/`snippet` 的匹配度打分返回前 5 条。
