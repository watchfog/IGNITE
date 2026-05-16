from __future__ import annotations

from typing import Any

from .datatypes import DialogueSegment
from .translation_runtime import has_kanji_overlap_from_original


def _coerce_review_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item or "").strip()
        if not s or s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def _merge_review_reasons(*values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for reason in _coerce_review_reasons(value):
            if reason in seen:
                continue
            out.append(reason)
            seen.add(reason)
    return out


def _attach_review_metadata(
    payload: dict[str, Any],
    reasons: list[str],
    force_review: bool = False,
) -> dict[str, Any]:
    if reasons or force_review:
        payload["needs_review"] = True
        payload["review_reason"] = reasons
    return payload


def _fill_short_false_gaps(flags: list[bool], max_gap: int) -> None:
    if max_gap <= 0 or len(flags) < 3:
        return
    p = 0
    while p < len(flags):
        if flags[p]:
            p += 1
            continue
        q = p
        while q < len(flags) and (not flags[q]):
            q += 1
        gap_len = q - p
        left_on = p > 0 and flags[p - 1]
        right_on = q < len(flags) and flags[q]
        if left_on and right_on and gap_len <= max_gap:
            for t in range(p, q):
                flags[t] = True
        p = q


def _first_true_run_bounds(flags: list[bool], min_run: int = 1) -> tuple[int, int] | None:
    min_len = max(1, int(min_run))
    p = 0
    while p < len(flags):
        if not flags[p]:
            p += 1
            continue
        q = p
        while q < len(flags) and flags[q]:
            q += 1
        if (q - p) >= min_len:
            return p, q - 1
        p = q
    return None


def _mark_kanji_overlap_for_review(seg: DialogueSegment) -> None:
    if has_kanji_overlap_from_original(seg.text_original, seg.translation_subtitle, min_len=5):
        seg.needs_review = True
        seg.review_reason = _merge_review_reasons(
            seg.review_reason,
            ["translation_kanji_overlap_with_original"],
        )
