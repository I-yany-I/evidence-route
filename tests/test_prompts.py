from evidence_route.prompts import judge_messages


def test_judge_prompt_requires_explicit_verdict_and_evidence_rules() -> None:
    prompt = judge_messages("claim", [])[0]["content"]

    assert "Supported" in prompt
    assert "Refuted" in prompt
    assert "Not Enough Evidence" in prompt
    assert "Conflicting Evidence/Cherrypicking" in prompt
    assert "must be Not Enough Evidence" in prompt
    assert "must be Conflicting Evidence/Cherrypicking" in prompt
    assert "do not invent citations" in prompt
