"""评测指标：判定准确率 / 信息覆盖度 / token 成本汇总。"""

from __future__ import annotations

import re

# 金标标签 → 裁判判定（mixed 视为「部分属实」）
LABEL_TO_VERDICT: dict[str, str] = {
    "true": "真实",
    "false": "虚假",
    "mixed": "部分属实",
    "insufficient": "证据不足",
}

# 判定归一化的匹配顺序：长的在前，避免「部分属实」被「真实」误命中
_VERDICT_ORDER = ["部分属实", "证据不足", "真实", "虚假"]

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RE = re.compile(r"[a-z0-9]+")


def normalize_verdict(text: str) -> str:
    """把预测文本归一化为四选一判定；未命中时返回原文（去空白）。"""
    t = (text or "").strip()
    if not t:
        return ""
    for verdict in _VERDICT_ORDER:
        if verdict in t:
            return verdict
    return t


def verdict_accuracy(predictions: list[str], labels: list[str]) -> float:
    """判定正确率：把金标标签映射为判定（mixed→部分属实），与归一化预测比较。"""
    if not labels:
        return 0.0
    correct = 0
    for pred, label in zip(predictions, labels):
        expected = LABEL_TO_VERDICT.get(label, str(label))
        if normalize_verdict(pred) == expected:
            correct += 1
    return correct / len(labels)


def coverage(predicted_text: str, key_facts: list[str]) -> float:
    """信息覆盖度 = 命中的关键事实点比例（子串/关键词匹配）。"""
    if not key_facts:
        return 0.0
    hits = sum(1 for fact in key_facts if _fact_hit(fact, predicted_text))
    return hits / len(key_facts)


def _fact_hit(key_fact: str, text: str) -> bool:
    """单个关键事实点是否命中：先整串子串匹配，再关键词（多数命中即算命中）。"""
    kf = _normalize(key_fact)
    if not kf:
        return False
    nt = _normalize(text)
    if kf in nt:
        return True
    words = _keywords(kf)
    if not words:
        return False
    hits = sum(1 for w in words if w in nt)
    return hits / len(words) >= 0.5


def _normalize(text: str) -> str:
    """去全部空白 + 小写，便于子串匹配。"""
    return re.sub(r"\s+", "", (text or "").lower())


def _keywords(text: str) -> list[str]:
    """切词：拉丁/数字词元 + 中文双字片（去重保序）。"""
    words: list[str] = _LATIN_RE.findall(text)
    for run in _CJK_RE.findall(text):
        if len(run) >= 2:
            words.extend(run[i:i + 2] for i in range(len(run) - 1))
        else:
            words.append(run)
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return seen


def cost_report(tokens: dict[str, int], price_per_1k: float = 0.0) -> dict:
    """token 成本汇总：总量 / 分 agent 明细 / 估算金额。"""
    per_agent = {str(k): int(v) for k, v in tokens.items()}
    total = sum(per_agent.values())
    return {
        "total_tokens": total,
        "per_agent": per_agent,
        "estimated_cost": total / 1000.0 * price_per_1k,
    }
