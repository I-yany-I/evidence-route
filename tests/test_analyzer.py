from evidence_route.analyzer import analyze_claim
from evidence_route.contracts import Evidence


def test_analyzer_splits_compound_english_claim() -> None:
    features = analyze_claim(
        "Company A grew in 2024, but Company B declined in 2025.",
        probe_evidence=[],
    )
    assert features.atomic_clause_count == 2
    assert features.has_contrast is True
    assert features.time_scope_count == 2


def test_analyzer_splits_compound_chinese_claim() -> None:
    features = analyze_claim("甲公司收入增长，而且乙公司利润下降。", probe_evidence=[])
    assert [unit.unit_id for unit in features.claim_units] == ["u0", "u1"]
    assert features.atomic_clause_count == 2


def test_probe_conflict_is_only_a_hint() -> None:
    evidence = [
        Evidence(
            evidence_id="av:dev:1:0:0",
            title="A",
            source_url="https://example.org/a",
            text="The rate was 10 percent.",
            provider="averitec_frozen",
            snapshot_sha256="a" * 64,
            ranking_score=2.0,
        ),
        Evidence(
            evidence_id="av:dev:1:1:0",
            title="B",
            source_url="https://example.org/b",
            text="The rate was not 10 percent; it was 12 percent.",
            provider="averitec_frozen",
            snapshot_sha256="b" * 64,
            ranking_score=1.0,
        ),
    ]
    assert analyze_claim("The rate was 10 percent.", evidence).probe_conflict_hint is True
