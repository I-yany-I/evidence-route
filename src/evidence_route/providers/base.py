from __future__ import annotations

from typing import Protocol

from evidence_route.contracts import Evidence


class EvidenceProvider(Protocol):
    async def search(
        self,
        claim_id: str,
        query: str,
        *,
        top_k: int,
        max_chars: int,
        max_per_source: int | None = None,
    ) -> list[Evidence]:
        raise NotImplementedError
