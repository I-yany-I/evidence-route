import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_legacy_runtime_is_removed() -> None:
    legacy_paths = [
        "config/llm.yaml",
        "eval",
        "samples",
        "requirements.txt",
        "docs/contracts.md",
        "docs/design.md",
        "docs/INTERVIEW_PREP.md",
        "docs/plan.md",
    ]
    assert all(not (ROOT / path).exists() for path in legacy_paths)
    legacy_package = ROOT / "src/agent_collab"
    assert not any(path.is_file() for path in legacy_package.rglob("*"))
    assert not list((ROOT / "tests").glob("test_core_*.py"))
    assert not list((ROOT / "tests").glob("test_eval_*.py"))
    assert not list((ROOT / "tests").glob("test_patterns_*.py"))
    assert not list((ROOT / "tests").glob("test_runtime_*.py"))
    assert not list((ROOT / "tests").glob("test_tools_*.py"))


def test_readme_does_not_claim_gate_b_or_unverified_identity() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    forbidden = [
        "A2A",
        "标准 MCP",
        "MCP 已实现",
        "人工审批",
        "Gradio",
        "官方 OpenAI 模型",
        "完整 AVeriTeC benchmark",
    ]
    assert all(term not in readme for term in forbidden)
    assert "OpenAI-compatible provider" in readme
    assert "identity unverified" in readme
    assert "不是官方 leaderboard 成绩" in readme


def test_lock_contains_every_direct_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8").lower()
    declared = list(project["dependencies"])
    for group in project["optional-dependencies"].values():
        declared.extend(group)
    direct = [item.split("==", 1)[0].split("[", 1)[0].lower() for item in declared]
    assert all(f"{name}==" in lock for name in direct)


def test_env_example_contains_names_not_values() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "EVIDENCE_ROUTE_API_KEY=",
        "EVIDENCE_ROUTE_BASE_URL=",
        "EVIDENCE_ROUTE_MODEL=",
        "EVIDENCE_ROUTE_PRICE_FILE=configs/pricing.local.yaml",
    ]
