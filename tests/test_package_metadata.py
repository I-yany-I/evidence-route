from importlib.metadata import metadata

import evidence_route


def test_distribution_identity() -> None:
    package = metadata("evidence-route")
    assert package["Name"] == "evidence-route"
    assert package["Requires-Python"] == ">=3.11"
    assert evidence_route.__version__ == "0.1.0"
