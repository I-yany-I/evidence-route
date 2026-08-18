"""Evidence retrieval providers used by EvidenceRoute."""

from evidence_route.providers.averitec import AveritecFrozenProvider
from evidence_route.providers.base import EvidenceProvider

__all__ = ["AveritecFrozenProvider", "EvidenceProvider"]
