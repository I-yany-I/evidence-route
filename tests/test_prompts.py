from evidence_route.prompts import (
    HARDENED_PROMPT_VERSION,
    JUDGE_PROMPT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    hardened_judge_messages,
    hardened_prompt_hash,
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
