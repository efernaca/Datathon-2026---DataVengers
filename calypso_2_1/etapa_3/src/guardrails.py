"""Deterministic submission guardrails guaranteeing 10 distinct active product IDs."""
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
    """Guarantee exactly k distinct, active product IDs in priority order."""
    result: list[str] = []
    seen: set[str] = set()

    def _extend_from(source: Sequence[str], respect_excludes: bool) -> None:
        for pid in source:
            if len(result) >= k:
                return
            if not pid or pid in seen or pid not in active_ids:
                continue
            if respect_excludes and pid in exclude_ids:
                continue
            seen.add(pid)
            result.append(pid)

    _extend_from(candidates, respect_excludes=True)
    if len(result) < k:
        _extend_from(fallback_ranked, respect_excludes=True)
    if len(result) < k and exclude_ids:
        _extend_from(candidates, respect_excludes=False)
        _extend_from(fallback_ranked, respect_excludes=False)
    if len(result) < k:
        for pid in active_ids:
            if len(result) >= k:
                break
            if pid not in seen:
                seen.add(pid)
                result.append(pid)

    return result[:k]


def validate_submission_record(
    record: dict[str, Any],
    active_ids: AbstractSet[str],
    k: int = NUM_SLOTS,
) -> bool:
    """Return True if a submission record satisfies all RawTree validity rules."""
    if not record.get("request_id"):
        return False
    items = record.get("items")
    if not isinstance(items, list) or len(items) != k:
        return False
    if len(set(items)) != k:
        return False
    return all(isinstance(pid, str) and pid in active_ids for pid in items)
