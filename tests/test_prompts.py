from evidence_route.contracts import ClaimFeatures, ClaimUnit
from evidence_route.prompts import (
    HARDENED_PROMPT_VERSION,
    HARDENED_SINGLE_PROMPT,
    HARDENED_WORKER_PROMPT,
    JUDGE_PROMPT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    hardened_judge_messages,
    hardened_prompt_hash,
    hardened_single_messages,
    hardened_worker_messages,
    judge_messages,
    prompt_hash,
)

LEGACY_PROMPT_HASH = "743c4c91d499a503b3d8333b617fc3388141a4d157cb065c4d47fcc1970336af"


def test_legacy_judge_prompt_and_hash_remain_unchanged() -> None:
    assert PROMPT_VERSION == "2026-08-17-gate-a-v1"
    assert JUDGE_PROMPT == (
        SYSTEM_PROMPT + " Judge the worker records and deduplicate their citations."
    )
    assert prompt_hash() == LEGACY_PROMPT_HASH
    assert judge_messages("claim", [])[0]["content"] == JUDGE_PROMPT


def test_hardened_judge_prompt_has_explicit_verdict_and_evidence_rules() -> None:
    prompt = hardened_judge_messages("claim", [])[0]["content"]

    assert HARDENED_PROMPT_VERSION != PROMPT_VERSION
    assert hardened_prompt_hash() != prompt_hash()

    assert "Supported" in prompt
    assert "Refuted" in prompt
    assert "Not Enough Evidence" in prompt
    assert "Conflicting Evidence/Cherrypicking" in prompt
    assert "must be Not Enough Evidence" in prompt
    assert "must be Conflicting Evidence/Cherrypicking" in prompt
    assert "do not invent citations" in prompt


def test_hardened_single_prompt_requires_support_for_the_full_literal_claim() -> None:
    features = ClaimFeatures(
        claim_units=[ClaimUnit(unit_id="u0", text="claim")],
        atomic_clause_count=1,
        entity_count=0,
        numeric_count=0,
        time_scope_count=0,
        has_comparison=False,
        has_causal=False,
        has_contrast=False,
        probe_source_count=0,
        probe_score_spread=0,
        probe_conflict_hint=False,
    )
    prompt = hardened_single_messages("claim", features, [])[0]["content"]

    assert prompt == HARDENED_SINGLE_PROMPT
    assert "full literal claim" in prompt
    assert "part of the claim" in prompt
    assert "only reports that someone made the claim" in prompt
    assert "Not Enough Evidence" in prompt


def test_hardened_worker_prompt_requires_full_task_coverage() -> None:
    task = type("Task", (), {"model_dump": lambda self, mode: {"task_id": "t0"}})()

    prompt = hardened_worker_messages(task, [])[0]["content"]

    assert prompt == HARDENED_WORKER_PROMPT
    assert "full assigned verification task" in prompt
    assert "Not Enough Evidence" in prompt
