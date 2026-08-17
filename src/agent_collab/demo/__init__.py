"""demo 组：事实核查演示入口（offline 样例检索 + make_llm 注入）。"""

from .fact_check_demo import load_team, load_workflow, run_fact_check

__all__ = [
    "run_fact_check",
    "load_team",
    "load_workflow",
]
