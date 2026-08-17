"""eval/metrics.py 的单元测试（纯函数，不联网）。"""

from agent_collab.eval import cost_report, coverage, normalize_verdict, verdict_accuracy


# ---------------------------------------------------------------------------
# verdict_accuracy
# ---------------------------------------------------------------------------
def test_verdict_accuracy_perfect():
    labels = ["true", "false", "mixed", "insufficient"]
    preds = ["真实", "虚假", "部分属实", "证据不足"]
    assert verdict_accuracy(preds, labels) == 1.0


def test_verdict_accuracy_mixed_maps_to_partial():
    # mixed 视为「部分属实」
    assert verdict_accuracy(["部分属实"], ["mixed"]) == 1.0


def test_verdict_accuracy_normalizes_extra_text():
    preds = ["判定：部分属实（证据充分）", "结论为虚假"]
    labels = ["mixed", "false"]
    assert verdict_accuracy(preds, labels) == 1.0


def test_verdict_accuracy_empty_labels_returns_zero():
    assert verdict_accuracy([], []) == 0.0


def test_normalize_verdict_returns_raw_when_no_match():
    assert normalize_verdict(" 无判定词 ") == "无判定词"


def test_normalize_verdict_aliases_team_vocabulary():
    # 团队内部词汇（支持/反对/存疑）归一化到金标词汇
    assert normalize_verdict("核查判定：支持。") == "真实"
    assert normalize_verdict("结论为反对") == "虚假"
    assert normalize_verdict("证据不足，无法判定") == "证据不足"
    assert normalize_verdict("不支持该说法") == "虚假"


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------
def test_coverage_full_hit():
    text = "OpenAI 于 2024 年 5 月发布 GPT-4o，支持多模态推理。"
    facts = ["OpenAI 发布 GPT-4o", "2024 年 5 月"]
    assert coverage(text, facts) == 1.0


def test_coverage_miss():
    text = "这是一段与事实无关的文本。"
    facts = ["OpenAI 发布 GPT-4o", "2024 年 5 月"]
    assert coverage(text, facts) == 0.0


def test_coverage_partial_hit():
    text = "OpenAI 于 2024 年 5 月发布 GPT-4o。"
    facts = ["OpenAI 发布 GPT-4o", "欧盟人工智能法案"]
    assert coverage(text, facts) == 0.5


def test_coverage_keyword_partial_match():
    # 关键事实点未整串出现，但多数关键词命中（子串不匹配时走关键词匹配）
    text = "中国并未全面禁止燃油汽车"
    facts = ["中国全面禁止燃油汽车"]
    assert coverage(text, facts) == 1.0


def test_coverage_empty_facts_returns_zero():
    assert coverage("任意文本", []) == 0.0


# ---------------------------------------------------------------------------
# cost_report
# ---------------------------------------------------------------------------
def test_cost_report_totals():
    report = cost_report({"a": 100, "b": 50}, price_per_1k=2.0)
    assert report["total_tokens"] == 150
    assert report["per_agent"] == {"a": 100, "b": 50}
    assert report["estimated_cost"] == 0.3
