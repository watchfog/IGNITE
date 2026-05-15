from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import DialogueSegment
from .review_utils import _merge_review_reasons
from .translation_runtime import has_kanji_overlap_from_original, normalize_quotes_for_subtitle


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _subtitle_cache_key(segment_id: int, time_start: float, time_end: float, dialogue_type: str) -> str:
    st = int(round(float(time_start) * 1000))
    ed = int(round(float(time_end) * 1000))
    return f"id={int(segment_id)}|{st}-{ed}|{dialogue_type}"


def _translation_cache_review_reasons(
    dialogue_type: str,
    translation_subtitle: Any,
    debug_subtitle: Any,
    original_text: Any = "",
) -> list[str]:
    dt = str(dialogue_type or "").strip().lower()
    if dt in {"blank_no_name", "blank", "title"}:
        return []
    text = normalize_quotes_for_subtitle(str(translation_subtitle or "").strip())
    original = str(original_text or "").strip()
    debug = str(debug_subtitle or "").strip()
    reasons: list[str] = []
    if not text:
        reasons.append("translation_missing")
    elif debug and text == debug:
        reasons.append("translation_fallback_debug_text")
    elif text.startswith("[DEBUG]"):
        reasons.append("translation_fallback_debug_text")
    if has_kanji_overlap_from_original(original, text, min_len=5):
        reasons.append("translation_kanji_overlap_with_original")
    return reasons


def _cache_entry_with_review_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(entry)
    out.pop("srt_start", None)
    out.pop("srt_end", None)
    reasons = _merge_review_reasons(
        out.get("review_reason"),
        out.get("review_reasons"),
        _translation_cache_review_reasons(
            str(out.get("dialogue_type", "") or ""),
            out.get("translation_subtitle", ""),
            out.get("debug_subtitle", ""),
            out.get("text_original", ""),
        ),
    )
    if bool(out.get("needs_review", False)) or reasons:
        out["needs_review"] = True
        out["review_reason"] = reasons
    return out


def _load_translation_cache(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}, {}
    try:
        raw = _load_json(path)
    except Exception:
        return {}, {}
    entries = raw.get("entries", []) if isinstance(raw, dict) else []
    by_full_key: dict[str, dict[str, Any]] = {}
    by_time_type: dict[str, dict[str, Any]] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        text = normalize_quotes_for_subtitle(str(e.get("translation_subtitle", "") or "").strip())
        original = str(e.get("text_original", "") or "").strip()
        if not text:
            continue
        sid = int(e.get("segment_id", 0) or 0)
        ts = float(e.get("time_start", 0.0) or 0.0)
        te = float(e.get("time_end", 0.0) or 0.0)
        dt = str(e.get("dialogue_type", "") or "")
        dbg = str(e.get("debug_subtitle", "") or "")
        k1 = _subtitle_cache_key(sid, ts, te, dt)
        k2 = _subtitle_cache_key(0, ts, te, dt)
        payload = {
            "speaker": str(e.get("speaker", "") or ""),
            "text_original": original,
            "translation_subtitle": text,
            "needs_review": bool(e.get("needs_review", False)),
            "review_reason": _merge_review_reasons(
                e.get("review_reason"),
                _translation_cache_review_reasons(dt, text, dbg, original),
            ),
        }
        by_full_key[k1] = payload
        by_time_type[k2] = payload
    return by_full_key, by_time_type


def _dump_translation_cache(
    path: Path,
    video_path: str,
    config_path: str,
    segments: list[DialogueSegment],
    debug_texts: list[str],
    prefix_entries: list[dict[str, Any]] | None = None,
) -> None:
    entries: list[dict[str, Any]] = []
    if prefix_entries:
        entries.extend(_cache_entry_with_review_metadata(e) for e in prefix_entries)
    for seg, dbg in zip(segments, debug_texts):
        entry = {
            "segment_id": seg.segment_id,
            "time_start": seg.time_start,
            "time_end": seg.time_end,
            "dialogue_type": seg.dialogue_type,
            "speaker": seg.speaker,
            "text_original": seg.text_original,
            "translation_subtitle": seg.translation_subtitle,
            "debug_subtitle": dbg,
        }
        reasons = _merge_review_reasons(
            seg.review_reason,
            _translation_cache_review_reasons(
                seg.dialogue_type,
                seg.translation_subtitle,
                dbg,
                seg.text_original,
            ),
        )
        if seg.needs_review or reasons:
            entry["needs_review"] = True
            entry["review_reason"] = reasons
        entries.append(entry)
    payload = {
        "version": 1,
        "video": str(video_path),
        "config_path": str(config_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "entries": entries,
    }
    _save_json(path, payload)


def _load_cache_entries(path: Path) -> list[dict[str, Any]]:
    raw = _load_json(path)
    entries = raw.get("entries", []) if isinstance(raw, dict) else []
    return [e for e in entries if isinstance(e, dict)]


def _extract_title_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        try:
            sid = int(e.get("segment_id", -1) or -1)
        except Exception:
            sid = -1
        if sid == 0:
            out.append(copy.deepcopy(e))
    return out


def _segments_from_cache_entries(entries: list[dict[str, Any]]) -> tuple[list[DialogueSegment], list[DialogueSegment]]:
    segs: list[DialogueSegment] = []
    dbg_segs: list[DialogueSegment] = []
    for i, e in enumerate(entries, start=1):
        ts = float(e.get("time_start", 0.0))
        te = float(e.get("time_end", ts))
        text = normalize_quotes_for_subtitle(str(e.get("translation_subtitle", "") or ""))
        dbg = str(e.get("debug_subtitle", "") or "")
        sid = int(e.get("segment_id", i) or i)
        dt = str(e.get("dialogue_type", "speaker_dialogue") or "speaker_dialogue")
        seg = DialogueSegment(
            segment_id=sid,
            frame_start=0,
            frame_end=0,
            time_start=ts,
            time_end=te,
            speaker=str(e.get("speaker", "") or ""),
            speaker_confidence=0.0,
            text_original=str(e.get("text_original", "") or ""),
            text_ocr_confidence=0.0,
            translation_subtitle=text,
            dialogue_type=dt,
            line_count_detected=max(1, text.count("\n") + 1) if text else 1,
        )
        dseg = copy.deepcopy(seg)
        dseg.translation_subtitle = dbg
        segs.append(seg)
        dbg_segs.append(dseg)
    return segs, dbg_segs


def _find_latest_translation_cache(base_work_dir: Path) -> Path | None:
    try:
        runs = [p for p in base_work_dir.glob("run_*") if p.is_dir()]
    except Exception:
        return None
    if not runs:
        return None
    candidates: list[Path] = []
    for r in runs:
        c = r / "translation_cache.json"
        if c.exists():
            candidates.append(c)
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except Exception:
        return candidates[-1]
