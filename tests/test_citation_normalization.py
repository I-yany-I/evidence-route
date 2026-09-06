import warnings

from pydantic import HttpUrl

from evidence_route.contracts import Citation, Evidence, Usage, Verdict, VerdictDraft
from evidence_route.evaluation.stability import citation_is_valid, citation_urls
from evidence_route.verification import result_from_draft


def _evidence(source_url: str) -> list[Evidence]:
    return [
        Evidence(
            evidence_id="e1",
            title="Source",
            source_url=source_url,
            text="The quoted text.",
            provider="averitec_frozen",
            snapshot_sha256="a" * 64,
            ranking_score=1.0,
        )
    ]


def _result(url: str):
    citation = Citation(
        evidence_id="e1",
        claim_unit_ids=["u0"],
        question="question",
        answer="answer",
        quote="The quoted text.",
        stance="supports",
        source_url=url,
    )
    draft = VerdictDraft(
        verdict=Verdict.SUPPORTED,
        confidence=0.9,
        rationale="supports",
        citations=[citation],
    )
    response = type(
        "Response",
        (),
        {
            "value": draft,
            "usage": Usage(input_tokens=1, output_tokens=1, total_tokens=2, complete=True),
        },
    )()
    return result_from_draft(
        response,
        "claim",
        "single",
        _evidence(url),
        available_evidence_ids=["e1"],
    )


def test_verification_canonicalizes_citation_url_without_changing_quote() -> None:
    result = _result("HTTPS://Example.ORG:443/fact/?b=2&a=1#quote")

    assert str(result.citations[0].source_url) == "https://example.org/fact?a=1&b=2"
    assert isinstance(result.citations[0].source_url, HttpUrl)
    assert result.citations[0].quote == "The quoted text."


def test_canonicalized_citation_serializes_without_pydantic_warning() -> None:
    result = _result("HTTPS://Example.ORG:443/fact/?b=2&a=1#quote")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result.model_dump(mode="json")

    assert not [
        warning
        for warning in caught
        if "Pydantic serializer warnings" in str(warning.message)
    ]


def test_equivalent_citation_urls_have_same_comparison_and_validity() -> None:
    first = _result("https://example.org/fact?b=2&a=1")
    second = _result("HTTPS://EXAMPLE.ORG:443/fact/?a=1&b=2#quote")

    assert citation_urls(first) == citation_urls(second)
    assert citation_is_valid(first) is True
    assert citation_is_valid(second) is True
