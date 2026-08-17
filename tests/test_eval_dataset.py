"""eval/dataset.py 的单元测试（读取本地样例数据，不联网）。"""

import json

from agent_collab.eval import EvalItem, load_dataset, single_agent_baseline

VALID_LABELS = {"true", "false", "mixed", "insufficient"}


def test_load_dataset_returns_8_eval_items():
    items = load_dataset()
    assert len(items) == 8
    assert all(isinstance(it, EvalItem) for it in items)


def test_load_dataset_label_distribution_2_each():
    items = load_dataset()
    counts: dict[str, int] = {}
    for it in items:
        counts[it.label] = counts.get(it.label, 0) + 1
    for label in VALID_LABELS:
        assert counts.get(label, 0) == 2, f"label {label!r} 应恰有 2 条"


def test_load_dataset_key_facts_between_2_and_4():
    items = load_dataset()
    for it in items:
        assert 2 <= len(it.key_facts) <= 4
        assert all(isinstance(f, str) and f for f in it.key_facts)


def test_load_dataset_ids_unique_and_claims_nonempty():
    items = load_dataset()
    ids = [it.id for it in items]
    assert len(set(ids)) == len(ids) == 8
    assert all(it.claim for it in items)


def test_load_dataset_custom_path_ignores_blank_lines(tmp_path):
    path = tmp_path / "claims.jsonl"
    path.write_text(
        json.dumps({"id": "a", "claim": "声明A", "label": "true", "key_facts": ["事实1", "事实2"]},
                   ensure_ascii=False)
        + "\n"
        + json.dumps({"id": "b", "claim": "声明B", "label": "false", "key_facts": ["事实3"]},
                     ensure_ascii=False)
        + "\n\n",  # 空行应被忽略
        encoding="utf-8",
    )
    items = load_dataset(path)
    assert len(items) == 2
    assert items[0].id == "a"
    assert items[0].label == "true"
    assert items[1].key_facts == ["事实3"]


def test_single_agent_baseline_description():
    desc = single_agent_baseline("测试声明")
    assert desc["kind"] == "single_agent"
    assert desc["query"] == "测试声明"
    spec = desc["spec"]
    assert spec.id == "baseline"
    assert "web_search" in spec.tools
    assert "read_file" in spec.tools
