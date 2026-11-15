"""Guardrails deterministas: exactamente 10 IDs distintos, activos y con stock."""
from __future__ import annotations

from collections.abc import Sequence, Set as AbstractSet
from typing import Any

NUM_SLOTS = 10


def ensure_valid_submission(
    candidates: Sequence[str],
    active_ids: AbstractSet[str],
    fallback_ranked: Sequence[str],
    exclude_ids: AbstractSet[str] = frozenset(),
    k: int = NUM_SLOTS,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    def extend(source: Sequence[str], respect_excludes: bool) -> None:
        for pid in source:
            if len(result) >= k:
                return
            if not pid or pid in seen or pid not in active_ids:
                continue
            if respect_excludes and pid in exclude_ids:
                continue
            seen.add(pid)
            result.append(pid)

    extend(candidates, True)
    if len(result) < k:
        extend(fallback_ranked, True)
    if len(result) < k and exclude_ids:
        extend(candidates, False)
        extend(fallback_ranked, False)
    if len(result) < k:
        for pid in sorted(active_ids):
            if len(result) >= k:
                break
            if pid not in seen:
                seen.add(pid)
                result.append(pid)
    return result[:k]


def validate_submission_record(record: dict[str, Any], active_ids: AbstractSet[str],
                               k: int = NUM_SLOTS) -> bool:
    if not record.get("request_id"):
        return False
    items = record.get("items")
    if not isinstance(items, list) or len(items) != k or len(set(items)) != k:
        return False
    return all(isinstance(p, str) and p in active_ids for p in items)
