"""评测数据集：加载样例声明，定义评测项与单 agent 基线调用描述。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..core import AgentSpec

DEFAULT_DATASET = Path(__file__).resolve().parents[3] / "samples" / "claims.jsonl"


class EvalItem(BaseModel):
    """一条评测项：待核声明 + 金标标签 + 关键事实点。"""

    id: str
    claim: str
    label: str  # true | false | mixed | insufficient
    key_facts: list[str] = Field(default_factory=list)


def load_dataset(path: Path | None = None) -> list[EvalItem]:
    """读 samples/claims.jsonl，逐行解析为 EvalItem 列表（空行忽略）。"""
    path = path or DEFAULT_DATASET
    items: list[EvalItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        items.append(EvalItem(
            id=raw["id"],
            claim=raw["claim"],
            label=raw["label"],
            key_facts=list(raw.get("key_facts") or []),
        ))
    return items


def single_agent_baseline(query: str) -> dict:
    """返回单 agent 基线的调用描述（不经过任何协作模式）。

    run_eval 用它构造一个带全部工具的 AgentRuntime 直接 ``run(query)``，
    与多 agent 协作模式对比覆盖度/成本/延迟。
    """
    return {
        "kind": "single_agent",
        "query": query,
        "spec": AgentSpec(
            id="baseline",
            role="单智能体事实核查员",
            goal="独立完成声明核查：检索来源、核实关键事实点，输出判定（真实/虚假/部分属实/证据不足）与证据。",
            backstory="独立研究员，单智能体模式下自行完成拆解、检索、裁决与报告。",
            tools=["web_search", "read_file"],
            approval_level="read-only",
        ),
    }
