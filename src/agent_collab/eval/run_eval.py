"""评测脚本：单 agent 基线 vs 多 agent 协作，产出 eval/report.md。

CLI：``python -m agent_collab.eval.run_eval [--offline] [--pattern parallel] [--limit N] [--out eval/report.md]``
真实 LLM 跑全量由编排者在 Task 6 执行；此处支持 ``make_llm`` 注入假 LLM 做离线冒烟。
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Callable

from ..core import AgentRuntime, LLMClient, LLMConfig, SharedMemory, load_llm_config
from ..demo.fact_check_demo import (
    PROJECT_ROOT,
    build_registry,
    load_workflow,
    run_fact_check,
)
from ..runtime import Approver, AuditLog
from .dataset import load_dataset, single_agent_baseline
from .metrics import coverage, normalize_verdict, verdict_accuracy

DEFAULT_OUT = PROJECT_ROOT / "eval" / "report.md"


def _run_single_agent(
    query: str,
    llm: Any,
    registry: Any,
    offline: bool,
) -> dict:
    """单 agent 基线：一个带全部工具的 AgentRuntime 直接 run(query)。"""
    desc = single_agent_baseline(query)
    memory = SharedMemory()
    audit = AuditLog()
    approver = Approver(auto_approve=True)
    counted = _UsageCountingLLM(llm)
    runtime = AgentRuntime(desc["spec"], counted, registry, memory, audit, approver)
    msg = asyncio.run(runtime.run(query))
    return {
        "answer": (msg.payload.get("content") or "").strip(),
        "tokens": counted.usage_total,
    }


class _UsageCountingLLM:
    """给基线 LLM 包一层 usage 计数（LLMClient 本身不累计，只在结果里透传 usage）。"""

    def __init__(self, delegate: Any):
        self._delegate = delegate
        self.usage_total = 0

    def complete(self, messages, tools=None, json_schema=None, max_tokens=None):
        result = self._delegate.complete(messages, tools=tools, json_schema=json_schema, max_tokens=max_tokens)
        if isinstance(result, dict) and isinstance(result.get("usage"), dict):
            self.usage_total += int(result["usage"].get("total_tokens", 0) or 0)
        return result

    def json_complete(self, messages, json_schema, max_tokens=None):
        return self._delegate.json_complete(messages, json_schema, max_tokens=max_tokens)


def run_eval(
    *,
    make_llm: Callable[[LLMConfig], Any] | None = None,
    offline: bool = True,
    pattern: str = "parallel",
    limit: int | None = None,
    out: Path | None = None,
) -> dict:
    """跑评测：每题单 agent 基线 + 多 agent，写 eval/report.md，返回汇总 dict。"""
    items = load_dataset()
    if limit is not None:
        items = items[:limit]

    wf = load_workflow(PROJECT_ROOT / "config" / "workflow.yaml")
    llm_path = PROJECT_ROOT / str(wf.get("llm_config") or "config/llm.yaml")
    if not llm_path.exists():
        llm_path = PROJECT_ROOT / "config" / "llm.example.yaml"
    llm_cfg = load_llm_config(llm_path)
    samples = PROJECT_ROOT / str(wf.get("offline_data") or "samples")

    def _make_llm(_cfg: LLMConfig | None = None) -> Any:
        # run_fact_check 会以 llm_cfg 调用该钩子；此处忽略入参、用闭包里的同一配置
        return make_llm(llm_cfg) if make_llm is not None else LLMClient(llm_cfg)

    multi_preds: list[str] = []
    base_preds: list[str] = []
    labels: list[str] = []
    rows: list[dict] = []

    for item in items:
        multi = run_fact_check(
            item.claim, pattern=pattern, offline=offline, make_llm=_make_llm,
        )
        base = _run_single_agent(
            item.claim, _make_llm(), build_registry(offline=offline, samples=samples), offline,
        )

        mv = normalize_verdict(multi.answer)
        bv = normalize_verdict(base["answer"])
        multi_preds.append(mv)
        base_preds.append(bv)
        labels.append(item.label)

        rows.append({
            "id": item.id,
            "claim": item.claim,
            "label": item.label,
            "multi_verdict": mv,
            "multi_coverage": coverage(multi.answer, item.key_facts),
            "multi_tokens": sum(int(t) for t in multi.tokens.values()),
            "multi_wall_s": round(multi.wall_s, 2),
            "base_verdict": bv,
            "base_coverage": coverage(base["answer"], item.key_facts),
            "base_tokens": int(base["tokens"]),
        })

    if not rows:
        return {
            "multi_accuracy": 0.0, "base_accuracy": 0.0,
            "multi_coverage_avg": 0.0, "base_coverage_avg": 0.0,
            "multi_tokens_total": 0, "base_tokens_total": 0,
            "multi_wall_s_total": 0.0,
        }

    summary = {
        "multi_accuracy": verdict_accuracy(multi_preds, labels),
        "base_accuracy": verdict_accuracy(base_preds, labels),
        "multi_coverage_avg": sum(r["multi_coverage"] for r in rows) / len(rows),
        "base_coverage_avg": sum(r["base_coverage"] for r in rows) / len(rows),
        "multi_tokens_total": sum(r["multi_tokens"] for r in rows),
        "base_tokens_total": sum(r["base_tokens"] for r in rows),
        "multi_wall_s_total": round(sum(r["multi_wall_s"] for r in rows), 2),
    }

    out_path = out or DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_report(items, rows, summary), encoding="utf-8")
    return summary


def _render_report(items: list, rows: list[dict], summary: dict) -> str:
    lines = ["# 事实核查评测报告", ""]
    lines.append(f"- 多 agent 模式：判定准确率 {summary['multi_accuracy']:.0%}，"
                 f"平均覆盖度 {summary['multi_coverage_avg']:.0%}，"
                 f"token {summary['multi_tokens_total']}，"
                 f"耗时 {summary['multi_wall_s_total']}s")
    lines.append(f"- 单 agent 基线：判定准确率 {summary['base_accuracy']:.0%}，"
                 f"平均覆盖度 {summary['base_coverage_avg']:.0%}，"
                 f"token {summary['base_tokens_total']}")
    lines.append("")
    lines.append("| # | 声明 | 标签 | 多agent判定 | 多agent覆盖 | 基线判定 | 基线覆盖 | 多agent token | 多agent耗时 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['claim'][:24]} | {r['label']} | {r['multi_verdict']} "
            f"| {r['multi_coverage']:.0%} | {r['base_verdict']} | {r['base_coverage']:.0%} "
            f"| {r['multi_tokens']} | {r['multi_wall_s']}s |"
        )
    lines.append("")
    lines.append("（真实 LLM 跑通后由编排者回填 README 数字。）")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="事实核查评测")
    parser.add_argument("--offline", action="store_true", help="离线模式：web_search 使用样例数据")
    parser.add_argument("--pattern", default="parallel",
                        choices=["pipeline", "parallel", "supervisor", "debate"])
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（成本控制）")
    parser.add_argument("--out", default=None, help="报告输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_eval(
        offline=args.offline,
        pattern=args.pattern,
        limit=args.limit,
        out=Path(args.out) if args.out else None,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
