"""事实核查演示入口：组装 LLM / 工具 / 团队，按协作模式运行并导出审计报告。

支持：
- offline 模式：``web_search`` 被覆盖为样例数据检索（从 ``samples/sources.jsonl`` 按关键词匹配），
  全程不联网；覆盖顺序为「先注册样例检索（名 web_search），再注册其余内置工具（跳过 web_search）」。
- make_llm 注入：测试 / 编排可注入假 LLM，无需真实中转服务。
- CLI：``python -m agent_collab.demo.fact_check_demo --query "声明" [--offline] [--pattern parallel]``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from ..core import AgentSpec, LLMClient, LLMConfig, SharedMemory, load_llm_config
from ..patterns import AgentResult, run_pattern
from ..runtime import Approver, AuditLog
from ..tools import ToolRegistry, ToolResult, ToolSpec, register_builtin_tools

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_WORKFLOW = PROJECT_ROOT / "config" / "workflow.yaml"
DEFAULT_TEAM = PROJECT_ROOT / "config" / "agents" / "fact_check_team.yaml"
DEFAULT_SAMPLES = PROJECT_ROOT / "samples"
DEFAULT_LLM_CONFIG = PROJECT_ROOT / "config" / "llm.yaml"
EXAMPLE_LLM_CONFIG = PROJECT_ROOT / "config" / "llm.example.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# 各协作模式在本团队下的角色前缀映射（团队无 worker/aggregator/supervisor/pro/con 直名）
PATTERN_CONFIGS: dict[str, dict] = {
    "parallel": {
        "planner_prefix": "planner",
        "worker_prefix": "fact_checker",
        "aggregator_prefix": "editor",
    },
    "debate": {
        "pro_prefix": "debater_pro",
        "con_prefix": "debater_con",
        "judge_prefix": "judge",
    },
    "supervisor": {"supervisor_prefix": "planner"},
    "pipeline": {},
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[\-\.][a-z0-9]+)*")


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_workflow(path: Path) -> dict:
    """读 config/workflow.yaml，返回其 ``workflow`` 子字典。"""
    data = _load_yaml(path)
    wf = data.get("workflow")
    return dict(wf) if isinstance(wf, dict) else dict(data)


def load_team(path: Path) -> dict:
    """读 config/agents/<team>.yaml，返回其 ``team`` 子字典。"""
    data = _load_yaml(path)
    team = data.get("team")
    return dict(team) if isinstance(team, dict) else dict(data)


def parse_specs(team: dict) -> list[AgentSpec]:
    """把团队 YAML 的 ``agents`` 列表解析为 AgentSpec 列表。"""
    specs: list[AgentSpec] = []
    for item in team.get("agents") or []:
        specs.append(AgentSpec(
            id=item["id"],
            role=item["role"],
            goal=item["goal"],
            backstory=item.get("backstory", ""),
            tools=list(item.get("tools") or []),
            approval_level=item.get("approval_level", "read-only"),
        ))
    return specs


# ---------------------------------------------------------------------------
# 工具注册（offline 覆盖 web_search）
# ---------------------------------------------------------------------------
def build_registry(*, offline: bool, samples: Path, workspace: Path | None = None) -> ToolRegistry:
    """构造工具注册表；offline 时把 web_search 覆盖为样例数据检索。

    顺序关键：注册同名会抛 ValueError，故 offline 时先注册样例检索（名 web_search），
    再注册其余内置工具（跳过 web_search）。
    """
    workspace = workspace or samples
    registry = ToolRegistry()
    if offline:
        _register_sample_search(registry, samples)
        _register_builtin_except_web_search(registry, workspace)
    else:
        register_builtin_tools(registry, workspace)
    return registry


def _register_sample_search(registry: ToolRegistry, samples: Path) -> None:
    """注册 offline 样例数据检索，工具名 web_search（覆盖内置网络搜索）。"""
    sources = _load_sources(samples / "sources.jsonl")

    async def handler(args: dict) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, content="", error="缺少 query 参数")
        matches = _search_sources(sources, query)
        if not matches:
            return ToolResult(ok=True, content="（未在样例数据中找到与查询相关的来源。）")
        blocks = [
            f"标题：{m.get('title', '')}\n链接：{m.get('url', '')}\n摘要：{m.get('snippet', '')}"
            for m in matches
        ]
        return ToolResult(ok=True, content="\n\n".join(blocks))

    registry.register(ToolSpec(
        name="web_search",
        description="offline 样例数据检索：从 samples/sources.jsonl 按查询关键词匹配来源片段（不联网）。",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词"}},
            "required": ["query"],
        },
        approval_level="read-only",
        handler=handler,
    ))


def _register_builtin_except_web_search(registry: ToolRegistry, workspace: Path) -> None:
    """注册内置工具但跳过 web_search（offline 时已被样例检索占用同名）。

    read_file 的 handler 是 register_builtin_tools 内部的闭包，无法直接 import，
    故用临时注册表收割其 ToolSpec 后转注册到目标表（handler 已包装为 async）。
    """
    tmp = ToolRegistry()
    register_builtin_tools(tmp, workspace)
    for name in ("python_repl", "read_file"):
        spec = tmp.get(name)
        registry.register(ToolSpec(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            approval_level=spec.approval_level,
            handler=spec.handler,
        ))


def _load_sources(path: Path) -> list[dict]:
    """逐行解析 sources.jsonl，忽略空行。"""
    if not path.exists():
        return []
    sources: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            sources.append(json.loads(line))
    return sources


def _search_sources(sources: list[dict], query: str) -> list[dict]:
    """按查询关键词给来源打分，返回命中来源（得分降序，最多 5 条）。"""
    terms = _query_terms(query)
    if not terms:
        return []
    scored: list[tuple[int, dict]] = []
    for s in sources:
        text = f"{s.get('title', '')} {s.get('snippet', '')}".lower()
        score = sum(1 for t in terms if t in text)
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [s for _, s in scored[:5]]


def _query_terms(query: str) -> list[str]:
    """把查询分解为匹配词元：CJK 双字片 + 拉丁/数字词元（去重保序）。"""
    q = query.lower()
    terms: list[str] = []
    for run in _CJK_RE.findall(q):
        if len(run) >= 2:
            terms.extend(run[i:i + 2] for i in range(len(run) - 1))
    terms.extend(_TOKEN_RE.findall(q))
    seen: list[str] = []
    for t in terms:
        if t not in seen:
            seen.append(t)
    return seen


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def run_fact_check_async(
    query: str,
    *,
    pattern: str | None = None,
    offline: bool | None = None,
    make_llm: Callable[[LLMConfig], Any] | None = None,
    workflow_path: Path | None = None,
    team_path: Path | None = None,
    samples_dir: Path | None = None,
    audit: AuditLog | None = None,
) -> AgentResult:
    """组装并运行一次事实核查，返回统一结果（异步）。

    Args:
        query: 待核声明文本。
        pattern: 协作模式；None 时取 workflow 配置。
        offline: 是否离线；None 时取 workflow.offline_data 是否非空。
        make_llm: LLM 构造钩子（默认用 LLMClient；测试/编排可注入假 LLM）。
        workflow_path / team_path / samples_dir: 覆盖默认路径。
        audit: 可注入的审计日志（默认新建）。
    """
    wf = load_workflow(workflow_path or DEFAULT_WORKFLOW)
    team = load_team(team_path or DEFAULT_TEAM)

    if pattern is None:
        pattern = str(wf.get("pattern") or "parallel")
    if offline is None:
        offline = bool(wf.get("offline_data"))
    samples = samples_dir or (PROJECT_ROOT / str(wf.get("offline_data") or "samples"))

    llm_cfg = _load_llm_config(wf)
    llm = make_llm(llm_cfg) if make_llm is not None else LLMClient(llm_cfg)

    registry = build_registry(offline=offline, samples=samples)
    specs = parse_specs(team)
    memory = SharedMemory()
    audit = audit or AuditLog()
    approver = Approver(auto_approve=bool((wf.get("approval") or {}).get("auto_approve", True)))

    config = dict(PATTERN_CONFIGS.get(pattern, {}))
    config["replicas"] = dict(team.get("replicas") or {})
    pattern_obj = run_pattern(pattern, specs, llm, registry, memory, audit, approver, config)
    result = await pattern_obj.run(query)

    result.audit_path = _export_audit(result, query, pattern)  # type: ignore[attr-defined]
    return result


def run_fact_check(
    query: str,
    *,
    pattern: str | None = None,
    offline: bool | None = None,
    make_llm: Callable[[LLMConfig], Any] | None = None,
    workflow_path: Path | None = None,
    team_path: Path | None = None,
    samples_dir: Path | None = None,
    audit: AuditLog | None = None,
) -> AgentResult:
    """``run_fact_check_async`` 的同步包装（CLI / 脚本用）。"""
    return asyncio.run(run_fact_check_async(
        query,
        pattern=pattern,
        offline=offline,
        make_llm=make_llm,
        workflow_path=workflow_path,
        team_path=team_path,
        samples_dir=samples_dir,
        audit=audit,
    ))


def _load_llm_config(wf: dict) -> LLMConfig:
    """读 workflow 指定的 LLM 配置；缺失时回退到 llm.example.yaml（便于注入假 LLM）。"""
    llm_path = PROJECT_ROOT / str(wf.get("llm_config") or "config/llm.yaml")
    if not llm_path.exists():
        llm_path = EXAMPLE_LLM_CONFIG
    return load_llm_config(llm_path)


def _export_audit(result: AgentResult, query: str, pattern: str) -> Path:
    """把审计轨迹导出为 artifacts/ 下的 Markdown。"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = ARTIFACTS_DIR / f"audit_{pattern}_{ts}.md"
    header = (
        f"# 事实核查审计报告\n\n"
        f"- 声明：{query}\n"
        f"- 模式：{pattern}\n"
        f"- 耗时：{result.wall_s:.2f}s\n"
        f"- 终答：\n\n{result.answer}\n\n"
    )
    path.write_text(header + result.audit.to_markdown(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="事实核查演示入口")
    parser.add_argument("--query", required=True, help="待核声明文本")
    parser.add_argument("--offline", action="store_true", help="离线模式：web_search 使用样例数据")
    parser.add_argument(
        "--pattern", default=None,
        choices=["pipeline", "parallel", "supervisor", "debate"],
        help="协作模式（默认取 workflow 配置）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_fact_check(
        args.query,
        pattern=args.pattern,
        offline=True if args.offline else None,
    )
    print("=" * 64)
    print("核查结果：")
    print(result.answer)
    print("=" * 64)
    print(f"耗时 {result.wall_s:.2f}s，token 消耗 {result.tokens}")
    print(f"审计报告已导出到 {getattr(result, 'audit_path', 'artifacts/')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
