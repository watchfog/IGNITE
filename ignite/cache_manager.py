from __future__ import annotations

import copy
import json
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from .datatypes import DialogueSegment
from .review_utils import _merge_review_reasons
from .translation_runtime import has_kanji_overlap_from_original, normalize_quotes_for_subtitle


MANUAL_INSERT_RAW_ID = "manual_insert"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _subtitle_cache_key(segment_id: int, time_start: float, time_end: float, dialogue_type: str) -> str:
    st = int(round(float(time_start) * 1000))
    ed = int(round(float(time_end) * 1000))
    return f"id={int(segment_id)}|{st}-{ed}|{dialogue_type}"


def _coerce_cache_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _build_debug_subtitle(
    segment_id: int,
    raw_id: Any,
    time_start: float,
    time_end: float,
    dialogue_type: str,
    *,
    raw_seen_idx: int = 0,
    raw_count: int = 1,
) -> str:
    dt = str(dialogue_type or "").strip().lower()
    if dt == "title":
        return "[TITLE]"
    sid = _coerce_cache_int(segment_id, 0)
    raw_text = str(raw_id or "").strip()
    if raw_text == MANUAL_INSERT_RAW_ID:
        label_base = "manual"
    else:
        label_base = f"seg {_coerce_cache_int(raw_id, sid)}"
    label = label_base if int(raw_count) <= 1 else f"{label_base}-{int(raw_seen_idx)}"
    state_txt = "blank" if dt in {"blank_no_name", "blank"} else "has_name=true"
    return f"[DEBUG] {label} (id={sid}) {float(time_start):.2f}-{float(time_end):.2f}s {state_txt}"


def _debug_subtitle_from_entry(
    entry: dict[str, Any],
    *,
    default_segment_id: int = 0,
    raw_seen_idx: int = 0,
    raw_count: int = 1,
) -> str:
    legacy = str(entry.get("debug_subtitle", "") or "")
    if legacy.strip():
        return legacy
    sid = _coerce_cache_int(entry.get("segment_id"), default_segment_id)
    return _build_debug_subtitle(
        segment_id=sid,
        raw_id=entry.get("raw_id", sid),
        time_start=float(entry.get("time_start", 0.0) or 0.0),
        time_end=float(entry.get("time_end", entry.get("time_start", 0.0)) or 0.0),
        dialogue_type=str(entry.get("dialogue_type", "speaker_dialogue") or "speaker_dialogue"),
        raw_seen_idx=raw_seen_idx,
        raw_count=raw_count,
    )


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


_SPEAKER_NOISE_CHARS = set(" 　\n\r\t\u3000：:（）()「」『』【】\"'\"'｢｣“”‘’・．")


def _normalize_speaker_name(raw: str) -> str:
    text = unicodedata.normalize("NFKC", str(raw or ""))
    return "".join(ch for ch in text if ch not in _SPEAKER_NOISE_CHARS)


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[-1] + 1
            dlt = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dlt, sub))
        prev = cur
    return prev[-1]


def _copy_style(value: Any) -> dict[str, Any]:
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _resolve_speaker_subtitle_style(
    speaker_raw: str,
    subtitle_style_cfg: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    if not subtitle_style_cfg:
        return {}, False
    matching_cfg = subtitle_style_cfg.get("speaker_style_matching")
    if isinstance(matching_cfg, dict) and not matching_cfg.get("enabled", True):
        return {}, False
    raw_styles = subtitle_style_cfg.get("speaker_styles")
    if not isinstance(raw_styles, dict):
        return {}, False
    speaker_styles = {str(k).strip(): v for k, v in raw_styles.items() if str(k).strip() and isinstance(v, dict)}
    if not speaker_styles:
        return {}, False

    normalized = _normalize_speaker_name(speaker_raw)
    if not normalized:
        return {}, False
    if normalized in speaker_styles:
        return _copy_style(speaker_styles[normalized]), False

    norm_equal: list[str] = []
    contains: list[str] = []
    for key in speaker_styles:
        nk = _normalize_speaker_name(key)
        if normalized == nk:
            norm_equal.append(key)
        elif normalized in nk or nk in normalized:
            contains.append(key)

    if len(norm_equal) == 1:
        return _copy_style(speaker_styles[norm_equal[0]]), False
    if len(norm_equal) > 1:
        return {}, True
    if len(contains) == 1:
        return _copy_style(speaker_styles[contains[0]]), False
    if len(contains) > 1:
        return {}, True

    max_edit_distance = 1
    if isinstance(matching_cfg, dict):
        max_edit_distance = int(matching_cfg.get("max_edit_distance", 1))
    best_keys: list[str] = []
    best_dist = max_edit_distance + 1
    for key in speaker_styles:
        dist = _edit_distance(normalized, _normalize_speaker_name(key))
        if dist < best_dist:
            best_dist = dist
            best_keys = [key]
        elif dist == best_dist:
            best_keys.append(key)

    if best_dist <= max_edit_distance:
        if len(best_keys) == 1:
            return _copy_style(speaker_styles[best_keys[0]]), False
        return {}, True
    return {}, False


def _cache_entry_with_review_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(entry)
    out.pop("srt_start", None)
    out.pop("srt_end", None)
    out.pop("auto_review_reason", None)
    if "raw_id" not in out:
        out["raw_id"] = _coerce_cache_int(out.get("segment_id"), 0)
    debug_text = _debug_subtitle_from_entry(out)
    out.pop("debug_subtitle", None)
    reasons = _merge_review_reasons(
        out.get("review_reason"),
        out.get("review_reasons"),
        _translation_cache_review_reasons(
            str(out.get("dialogue_type", "") or ""),
            out.get("translation_subtitle", ""),
            debug_text,
            out.get("text_original", ""),
        ),
    )
    if reasons:
        out.pop("review_reason", None)
        out["needs_review"] = True
        out["review_reason"] = reasons
    else:
        out.pop("review_reason", None)
        out.pop("needs_review", None)
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
        k1 = _subtitle_cache_key(sid, ts, te, dt)
        k2 = _subtitle_cache_key(0, ts, te, dt)
        payload = {
            "speaker": str(e.get("speaker", "") or ""),
            "text_original": original,
            "translation_subtitle": text,
            "subtitle_style": _copy_style(e.get("subtitle_style")),
        }
        by_full_key[k1] = payload
        by_time_type[k2] = payload
    return by_full_key, by_time_type


def _dump_translation_cache(
    path: Path,
    video_path: str,
    config_path: str,
    segments: list[DialogueSegment],
    prefix_entries: list[dict[str, Any]] | None = None,
    source_work_cache: str | None = None,
    subtitle_style_cfg: dict[str, Any] | None = None,
) -> None:
    entries: list[dict[str, Any]] = []
    if prefix_entries:
        entries.extend(_cache_entry_with_review_metadata(e) for e in prefix_entries)
    for seg in segments:
        subtitle_style = _copy_style(seg.subtitle_style)
        style_ambiguous = False
        if not subtitle_style and str(seg.dialogue_type or "").strip().lower() not in {"blank_no_name", "blank", "title"}:
            subtitle_style, style_ambiguous = _resolve_speaker_subtitle_style(
                seg.speaker,
                subtitle_style_cfg,
            )
        debug_text = _build_debug_subtitle(
            segment_id=seg.segment_id,
            raw_id=seg.raw_id or seg.segment_id,
            time_start=seg.time_start,
            time_end=seg.time_end,
            dialogue_type=seg.dialogue_type,
        )
        entry = {
            "segment_id": seg.segment_id,
            "raw_id": seg.raw_id or seg.segment_id,
            "time_start": seg.time_start,
            "time_end": seg.time_end,
            "dialogue_type": seg.dialogue_type,
            "speaker": seg.speaker,
            "text_original": seg.text_original,
            "translation_subtitle": seg.translation_subtitle,
            "subtitle_style": subtitle_style,
        }
        auto_reason = str(getattr(seg, "auto_review_reason", "") or "").strip()
        if auto_reason:
            entry["auto_review_reason"] = auto_reason
        reasons = _merge_review_reasons(
            seg.review_reason,
            ["speaker_style_ambiguous"] if style_ambiguous else [],
            _translation_cache_review_reasons(
                seg.dialogue_type,
                seg.translation_subtitle,
                debug_text,
                seg.text_original,
            ),
        )
        if reasons:
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
    if source_work_cache:
        payload["source_work_cache"] = source_work_cache
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
    raw_ids: list[int | str] = []
    raw_counts: dict[int | str, int] = {}
    for i, e in enumerate(entries, start=1):
        sid = _coerce_cache_int(e.get("segment_id"), i)
        raw_id = e.get("raw_id", sid)
        rid = MANUAL_INSERT_RAW_ID if str(raw_id or "").strip() == MANUAL_INSERT_RAW_ID else _coerce_cache_int(raw_id, sid)
        raw_ids.append(rid)
        raw_counts[rid] = raw_counts.get(rid, 0) + 1
    raw_seen: dict[int | str, int] = {}
    for i, e in enumerate(entries, start=1):
        ts = float(e.get("time_start", 0.0))
        te = float(e.get("time_end", ts))
        text = normalize_quotes_for_subtitle(str(e.get("translation_subtitle", "") or ""))
        sid = _coerce_cache_int(e.get("segment_id"), i)
        raw_id = raw_ids[i - 1]
        raw_seen_idx = raw_seen.get(raw_id, 0)
        raw_seen[raw_id] = raw_seen_idx + 1
        dbg = _debug_subtitle_from_entry(
            e,
            default_segment_id=sid,
            raw_seen_idx=raw_seen_idx,
            raw_count=raw_counts.get(raw_id, 1),
        )
        dt = str(e.get("dialogue_type", "speaker_dialogue") or "speaker_dialogue")
        seg = DialogueSegment(
            segment_id=sid,
            raw_id=raw_id,
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
            subtitle_style=_copy_style(e.get("subtitle_style")),
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
