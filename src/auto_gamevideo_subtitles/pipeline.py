from __future__ import annotations

import argparse
import copy
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
import io
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_config
from .event_detect import (
    FrameMetric,
    MarkerTemplateMatcher,
    extract_text_features,
    frame_text_change_score,
    load_gray,
)
from .ffmpeg_utils import (
    extract_frame,
    extract_sequence,
    extract_sequence_dialogue_name_marker,
    ffprobe_video,
    extract_frame_to_memory,
)
from .models import DialogueSegment, Roi
from .ocr_engines import build_ocr_engine
from .state_machine import StateMachineConfig, segment_from_metrics
from .subtitle_export import write_ass, write_srt
from .translation_runtime import (
    BailianVlmTranslator,
    _normalize_quotes_for_subtitle,
    load_api_key,
)

_LOG_FILE_PATH: Path | None = None


def _log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE_PATH is not None:
        try:
            _LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_FILE_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def _vlm_translate_task(
    seg_id: int,
    translator: BailianVlmTranslator,
    speaker_image_path: Path,
    image_path: Path,
    speaker: str,
    history_items: list[dict[str, str]] | None = None,
) -> tuple[str, str, str, dict[str, int]]:
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    attempts = max(1, int(getattr(translator, "max_retries", 0)) + 1)
    last_err: ValueError | None = None
    for attempt in range(1, attempts + 1):
        _log(f"[VLM] segment {seg_id}: request started (attempt {attempt}/{attempts})")
        try:
            speaker_name, original_text, translated_text, usage = (
                translator.translate_image_ja_to_zh_cn_structured_with_tag(
                    image_path=image_path,
                    speaker_image_path=speaker_image_path,
                    speaker=speaker,
                    request_tag=f"segment {seg_id}",
                    history_items=history_items,
                )
            )
            pt = _safe_int(usage.get("prompt_tokens", 0))
            ct = _safe_int(usage.get("completion_tokens", 0))
            tt = _safe_int(usage.get("total_tokens", pt + ct), pt + ct)
            _log(
                f"[VLM] segment {seg_id}: request succeeded "
                f"(orig_chars={len(original_text)}, trans_chars={len(translated_text)}, "
                f"tokens_in={pt}, tokens_out={ct}, tokens_total={tt})"
            )
            return speaker_name, original_text, translated_text, usage
        except ValueError as exc:
            last_err = exc
            detail = str(exc).strip() or repr(exc)
            if attempt >= attempts:
                break
            _log(
                f"[VLM] segment {seg_id}: ValueError on attempt {attempt}/{attempts}: "
                f"{detail}; retrying"
            )
            time.sleep(max(0.0, float(getattr(translator, "retry_delay_sec", 0.0))))
    if last_err is not None:
        raise last_err
    raise RuntimeError("VLM request failed without an exception")


def _roi_from_cfg(cfg: dict[str, Any], key: str) -> Roi:
    v = cfg["roi"][key]
    return Roi(int(v[0]), int(v[1]), int(v[2]), int(v[3]))


def _collect_marker_templates(marker_cfg: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        rp = p.resolve()
        if rp in seen or not rp.exists() or not rp.is_file():
            return
        seen.add(rp)
        out.append(rp)

    tpl_paths = marker_cfg.get("template_paths")
    if isinstance(tpl_paths, list):
        for v in tpl_paths:
            p = Path(str(v))
            if not p.is_absolute():
                p = Path.cwd() / p
            _add(p)
    return out


def _resolve_active_vlm_model(cfg: dict[str, Any]) -> str:
    tr = cfg.get("translation", {}) if isinstance(cfg, dict) else {}
    raw_models = tr.get("vlm_models", [])
    models: list[str] = []
    if isinstance(raw_models, list):
        models = [str(x).strip() for x in raw_models if str(x).strip()]
    elif isinstance(raw_models, str) and raw_models.strip():
        models = [raw_models.strip()]
    raw_model = str(tr.get("model", "") or "").strip()
    if not models:
        models = ["qwen3.6-plus"]
    if raw_model and raw_model not in models:
        raise ValueError(
            "Invalid config: translation.model must be one item in translation.vlm_models."
        )
    active = raw_model if raw_model else models[0]
    return active


def _require_non_empty_translation_str(tr_cfg: dict[str, Any], key: str) -> str:
    raw = tr_cfg.get(key, None)
    value = str(raw or "").strip()
    if value:
        return value
    raise ValueError(
        f"Invalid config: translation.{key} is empty. "
        f"Please set translation.{key} in config/general_config.yaml."
    )


def _cleanup_old_work_runs(base_work_dir: Path, keep_latest: int = 3) -> None:
    keep_n = max(1, int(keep_latest))
    if not base_work_dir.exists():
        return

    base_resolved = base_work_dir.resolve()
    # Keep latest N runs per run kind to avoid frequent cache-only runs
    # evicting full/subtitle runs too aggressively.
    prefixes = ("run_full_", "run_cache_", "run_subtitle_", "run_")
    for pref in prefixes:
        run_dirs: list[Path] = []
        for p in base_work_dir.iterdir():
            if not p.is_dir():
                continue
            n = p.name
            if pref == "run_":
                # Legacy runs only.
                if n.startswith("run_") and (
                    not n.startswith("run_full_")
                    and not n.startswith("run_cache_")
                    and not n.startswith("run_subtitle_")
                ):
                    run_dirs.append(p)
            elif n.startswith(pref):
                run_dirs.append(p)
        if not run_dirs:
            continue
        run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        stale = run_dirs[keep_n:]
        for d in stale:
            try:
                resolved = d.resolve()
                if resolved.parent != base_resolved:
                    _log(f"Skip cleanup for unsafe path: {resolved}")
                    continue
                shutil.rmtree(resolved)
                _log(f"Cleanup old work dir: {resolved}")
            except Exception as exc:
                _log(f"Warning: failed to cleanup {d}: {exc}")


def _cleanup_intermediate_artifacts(work_dir: Path) -> None:
    patterns = [
        "fine_*_dialog",
        "fine_*_dialogue",
        "fine_*_name",
        "fine_*_marker",
        # Legacy name kept for backward cleanup compatibility.
        "fine_*_full",
        "fine_*_full_temp",
        "ocr_dialogue",
        "ocr_name",
        "name_by_segment",
        "marker_by_segment",
    ]
    for pat in patterns:
        for p in work_dir.glob(pat):
            if not p.exists():
                continue
            ok = False
            last_exc: Exception | None = None
            for _ in range(4):
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                    _log(f"Cleanup intermediate: {p}")
                    ok = True
                    break
                except Exception as exc:
                    last_exc = exc
                    time.sleep(0.25)
            if not ok:
                _log(f"Warning: failed to cleanup intermediate {p}: {last_exc}")


def _crop_and_save(frame_path: Path, roi: Roi, output_path: Path) -> None:
    with Image.open(frame_path) as im:
        x1 = max(0, min(int(roi.x1), im.width - 1))
        y1 = max(0, min(int(roi.y1), im.height - 1))
        x2 = max(0, min(int(roi.x2), im.width))
        y2 = max(0, min(int(roi.y2), im.height))
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"Invalid crop roi: {roi}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        im.crop((x1, y1, x2, y2)).save(output_path)


def _crop_image_to_base64(image: Image.Image, roi: Roi) -> str:
    """Crop an in-memory PIL Image and convert to base64 PNG data URL."""
    import base64
    import io
    
    x1 = max(0, min(int(roi.x1), image.width - 1))
    y1 = max(0, min(int(roi.y1), image.height - 1))
    x2 = max(0, min(int(roi.x2), image.width))
    y2 = max(0, min(int(roi.y2), image.height))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid crop roi: {roi}")
    
    cropped = image.crop((x1, y1, x2, y2))
    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _image_to_base64(image: Image.Image) -> str:
    """Convert an in-memory PIL Image to base64 PNG data URL."""
    import base64
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _timestamp_to_frame_number(timestamp: float, start_sec: float, fps: float) -> int:
    """Convert timestamp to frame number (1-indexed for ffmpeg %06d pattern)."""
    frame_index = int(round((timestamp - start_sec) * fps))
    return max(0, frame_index + 1)  # ffmpeg numbering starts at 1


def _try_load_cached_full_frame(
    timestamp: float,
    cache_dir: Path,
    start_sec: float,
    fps: float,
) -> Image.Image | None:
    """Try to load a full frame from cache. Returns None if not found."""
    frame_num = _timestamp_to_frame_number(timestamp, start_sec, fps)
    frame_path = cache_dir / f"{frame_num:06d}.png"
    if frame_path.exists():
        try:
            return Image.open(frame_path).convert("RGB")
        except Exception:
            return None
    return None


def _try_load_cached_roi_frame_with_status(
    timestamp: float,
    cache_dir: Path,
    start_sec: float,
    fps: float,
) -> tuple[Image.Image | None, str, int]:
    """Load cached ROI frame from image cache with explicit status for logging."""
    frame_num = _timestamp_to_frame_number(timestamp, start_sec, fps)
    frame_path = cache_dir / f"{frame_num:06d}.png"
    if not frame_path.exists():
        return None, "miss_not_found", frame_num
    try:
        return Image.open(frame_path).convert("RGB"), "hit", frame_num
    except Exception:
        return None, "miss_read_error", frame_num


def _background_score_marker_and_prune_dialogue_cache(
    marker_dir: Path,
    dialogue_cache_dir: Path,
    marker_matcher: MarkerTemplateMatcher,
    marker_threshold: float,
    stop_event: threading.Event,
    end_index_holder: dict[str, int | None],
    score_cache: dict[int, float],
    cache_lock: threading.Lock,
    verbose: bool = False,
) -> None:
    """Drive by dialogue-cache frame ids: wait marker counterpart, then score and prune."""
    idx = 1
    worker_tag = marker_dir.name or "marker_worker"

    def _try_delete(fid: int) -> bool:
        p = dialogue_cache_dir / f"{fid:06d}.png"
        try:
            if p.exists():
                p.unlink()
                return True
        except Exception:
            return False
        return False

    while True:
        end_idx = end_index_holder.get("end")
        if end_idx is not None and idx > end_idx:
            break

        dialogue_path = dialogue_cache_dir / f"{idx:06d}.png"
        marker_path = marker_dir / f"{idx:06d}.png"

        # Dialogue frame is the source of truth. If dialogue exists but marker not yet ready,
        # keep waiting until marker arrives.
        if dialogue_path.exists():
            if not marker_path.exists():
                if verbose:
                    _log(f"[{worker_tag}] frame={idx:06d} dialogue_ready marker_missing wait")
                time.sleep(0.02)
                continue

            sc = float(marker_matcher.score(load_gray(marker_path)))
            with cache_lock:
                score_cache[idx - 1] = sc
            if sc < marker_threshold:
                deleted_now = _try_delete(idx)
                if verbose:
                    _log(
                        f"[{worker_tag}] frame={idx:06d} score={sc:.4f} < th={marker_threshold:.4f} "
                        f"decision=DELETE deleted_now={deleted_now}"
                    )
                    if not deleted_now:
                        _log(f"[{worker_tag}] frame={idx:06d} delete failed")
            else:
                if verbose:
                    _log(
                        f"[{worker_tag}] frame={idx:06d} score={sc:.4f} >= th={marker_threshold:.4f} "
                        "decision=KEEP"
                    )
            idx += 1
            continue

        # If extraction already finished and this dialogue frame is still absent, skip it.
        if end_idx is not None and idx <= end_idx:
            if verbose:
                _log(f"[{worker_tag}] frame={idx:06d} dialogue_missing_after_end skip")
            idx += 1
            continue

        if stop_event.is_set() and end_idx is None:
            # Wait for producer to publish final end index.
            time.sleep(0.02)
            continue
        time.sleep(0.02)


def _final_prune_dialogue_cache_by_scores(
    dialogue_cache_dir: Path,
    marker_scores_cached: list[float],
    marker_threshold: float,
    log_tag: str,
    verbose: bool = False,
) -> tuple[int, int]:
    """Deterministically prune low-score dialogue cache frames after extraction completes."""
    removed = 0
    failed = 0
    for i, sc in enumerate(marker_scores_cached, start=1):
        if float(sc) >= float(marker_threshold):
            continue
        p = dialogue_cache_dir / f"{i:06d}.png"
        if not p.exists():
            continue
        try:
            p.unlink()
            removed += 1
        except Exception:
            failed += 1
            if verbose:
                _log(f"[{log_tag}] frame={i:06d} post_prune_delete_failed")
    return removed, failed


def _prune_dialogue_cache_to_anchor_frames(
    dialogue_cache_dir: Path,
    keep_frame_ids_1based: set[int],
    log_tag: str,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """Keep only anchor frames (1-based ids) in dialogue cache and delete the rest."""
    removed = 0
    failed = 0
    kept = 0
    if not dialogue_cache_dir.exists():
        return removed, failed, kept
    for p in dialogue_cache_dir.glob("*.png"):
        try:
            fid = int(p.stem)
        except Exception:
            continue
        if fid in keep_frame_ids_1based:
            kept += 1
            continue
        try:
            p.unlink()
            removed += 1
        except Exception:
            failed += 1
            if verbose:
                _log(f"[{log_tag}] frame={fid:06d} anchor_prune_delete_failed")
    return removed, failed, kept


def _compute_marker_scores(
    marker_frames: list[Path],
    marker_matcher: MarkerTemplateMatcher,
    marker_workers: int,
    marker_coarse_step: int,
    marker_refine_margin: float,
    marker_threshold_hint: float | None,
) -> tuple[list[float], dict[str, int]]:
    def _score_one(j: int) -> tuple[int, float]:
        p = marker_frames[j]
        if p and p.exists():
            return j, float(marker_matcher.score(load_gray(p)))
        return j, 0.0

    def _score_indices(indices: list[int]) -> dict[int, float]:
        out: dict[int, float] = {}
        if not indices:
            return out
        workers = max(1, int(marker_workers))
        if workers <= 1:
            for j in indices:
                jj, sc = _score_one(j)
                out[jj] = sc
            return out
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_score_one, j) for j in indices]
            for fut in as_completed(futs):
                jj, sc = fut.result()
                out[jj] = sc
        return out

    n = len(marker_frames)
    marker_scores: list[float] = [0.0] * n
    scored = [False] * n
    step = max(1, int(marker_coarse_step))
    computed = 0

    if step <= 1 or n <= step + 1:
        full = _score_indices(list(range(n)))
        for j, sc in full.items():
            marker_scores[j] = sc
            scored[j] = True
        computed = len(full)
    else:
        anchors = list(range(0, n, step))
        if anchors[-1] != (n - 1):
            anchors.append(n - 1)
        anchor_map = _score_indices(anchors)
        computed += len(anchor_map)
        for j, sc in anchor_map.items():
            marker_scores[j] = sc
            scored[j] = True

        if marker_threshold_hint is None:
            thr_hint = float(np.percentile(list(anchor_map.values()) or [0.0], 70))
        else:
            thr_hint = float(marker_threshold_hint)
        margin = max(0.01, float(marker_refine_margin))

        refine_set: set[int] = set(anchors)
        stable_intervals: list[tuple[int, int, float, float, bool, int]] = []
        for ai in range(len(anchors) - 1):
            li = anchors[ai]
            ri = anchors[ai + 1]
            ls = float(anchor_map.get(li, 0.0))
            rs = float(anchor_map.get(ri, 0.0))
            l_on = ls >= thr_hint
            r_on = rs >= thr_hint
            uncertain = (abs(ls - thr_hint) <= margin) or (abs(rs - thr_hint) <= margin)
            if (l_on != r_on) or uncertain:
                lo = max(0, li - step)
                hi = min(n - 1, ri + step)
                refine_set.update(range(lo, hi + 1))
            else:
                mid = li + ((ri - li) // 2)
                stable_intervals.append((li, ri, ls, rs, l_on, mid))

        mids = sorted(set(x[5] for x in stable_intervals if x[5] not in anchor_map))
        mid_map = _score_indices(mids) if mids else {}
        computed += len(mid_map)

        for li, ri, ls, rs, l_on, mid in stable_intervals:
            ms = float(anchor_map.get(mid, mid_map.get(mid, ls)))
            mid_uncertain = abs(ms - thr_hint) <= margin
            mid_on = ms >= thr_hint
            if mid_uncertain or (mid_on != l_on):
                lo = max(0, li - step)
                hi = min(n - 1, ri + step)
                refine_set.update(range(lo, hi + 1))
                continue
            for k in range(li + 1, ri):
                marker_scores[k] = ls if (k - li) <= (ri - k) else rs
                scored[k] = True

        refine_indices = [j for j in sorted(refine_set) if not scored[j]]
        refine_map = _score_indices(refine_indices)
        computed += len(refine_map)
        for j, sc in refine_map.items():
            marker_scores[j] = sc
            scored[j] = True

        # Fill leftovers by nearest scored frame.
        last_sc = 0.0
        for j in range(n):
            if scored[j]:
                last_sc = marker_scores[j]
            else:
                marker_scores[j] = last_sc
        last_sc = marker_scores[-1] if n else 0.0
        for j in range(n - 1, -1, -1):
            if scored[j]:
                last_sc = marker_scores[j]
            else:
                marker_scores[j] = last_sc

    return marker_scores, {
        "computed": int(computed),
        "total": int(n),
        "coarse_step": int(step),
    }


def _metrics_from_frame_lists(
    dialog_frames: list[Path],
    name_frames: list[Path] | None,
    marker_frames: list[Path] | None,
    fps: float,
    start_sec: float,
    marker_matcher: MarkerTemplateMatcher | None = None,
    marker_workers: int = 1,
    marker_coarse_step: int = 1,
    marker_refine_margin: float = 0.06,
    marker_threshold_hint: float | None = None,
    marker_stats: dict[str, int] | None = None,
    marker_scores_cached: list[float] | None = None,
) -> list[FrameMetric]:
    metrics: list[FrameMetric] = []
    prev_dialog_mask = None
    prev_name_mask = None
    prev_marker_mask = None
    prev_marker_score: float | None = None
    marker_scores: list[float] = [0.0] * len(dialog_frames)
    if marker_scores_cached is not None and len(marker_scores_cached) >= len(dialog_frames):
        marker_scores = [float(x) for x in marker_scores_cached[: len(dialog_frames)]]
        if marker_stats is not None:
            marker_stats.update(
                {
                    "computed": int(len(marker_scores)),
                    "total": int(len(marker_scores)),
                    "coarse_step": 1,
                }
            )
    elif marker_frames and marker_matcher is not None:
        marker_scores, stats = _compute_marker_scores(
            marker_frames=marker_frames,
            marker_matcher=marker_matcher,
            marker_workers=marker_workers,
            marker_coarse_step=marker_coarse_step,
            marker_refine_margin=marker_refine_margin,
            marker_threshold_hint=marker_threshold_hint,
        )
        if marker_stats is not None:
            marker_stats.update(stats)
    for i, dialog_path in enumerate(dialog_frames):
        gray_dialog = load_gray(dialog_path)
        dialog_mask, dialog_presence = extract_text_features(gray_dialog, mode="dialog")
        diff = (
            frame_text_change_score(prev_dialog_mask, dialog_mask)
            if prev_dialog_mask is not None
            else 1.0
        )
        name_diff = 0.0
        name_presence = 0.0
        marker_diff = 0.0
        marker_presence = 0.0
        name_path = name_frames[i] if name_frames and i < len(name_frames) else None
        marker_path = marker_frames[i] if marker_frames and i < len(marker_frames) else None
        if name_path and name_path.exists():
            gray_name = load_gray(name_path)
            name_mask, name_presence = extract_text_features(gray_name, mode="name")
            name_diff = (
                frame_text_change_score(prev_name_mask, name_mask)
                if prev_name_mask is not None
                else 1.0
            )
            prev_name_mask = name_mask
        if marker_matcher is not None:
            marker_presence = marker_scores[i] if i < len(marker_scores) else 0.0
            if prev_marker_score is None:
                marker_diff = 1.0
            else:
                marker_diff = abs(marker_presence - prev_marker_score)
            prev_marker_score = marker_presence
        elif marker_path and marker_path.exists():
            gray_marker = load_gray(marker_path)
            marker_mask, marker_presence = extract_text_features(gray_marker, mode="marker")
            marker_diff = (
                frame_text_change_score(prev_marker_mask, marker_mask)
                if prev_marker_mask is not None
                else 1.0
            )
            prev_marker_mask = marker_mask
        ts = start_sec + (i / fps)
        metric = FrameMetric(
            frame_index=i,
            timestamp=ts,
            diff=diff,
            presence=dialog_presence,
            name_diff=name_diff,
            name_presence=name_presence,
            marker_diff=marker_diff,
            marker_presence=marker_presence,
            dialog_path=dialog_path,
            name_path=name_path,
        )
        metrics.append(metric)
        prev_dialog_mask = dialog_mask
    return metrics


def _auto_marker_threshold(scores: list[float], fallback: float) -> float:
    vals = np.asarray([x for x in scores if x > 1e-6], dtype=np.float64)
    if vals.size < 24:
        return float(fallback)
    # Otsu on [0,1] marker scores
    hist, edges = np.histogram(vals, bins=100, range=(0.0, 1.0))
    total = vals.size
    sum_total = float(np.dot(np.arange(100), hist))
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    best_t = int(round(fallback * 99.0))
    for t in range(100):
        w_b += hist[t]
        if w_b <= 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = t
    thr = float((edges[best_t] + edges[min(best_t + 1, len(edges) - 1)]) / 2.0)
    # Blend with configured baseline to avoid over-aggressive threshold swings.
    mixed = 0.6 * float(fallback) + 0.4 * thr
    # Clamp to a sane operating region for this game marker.
    return float(max(0.25, min(0.85, mixed)))


def _group_active_ranges(
    metrics: list[FrameMetric],
    presence_threshold: float,
    diff_threshold: float,
    name_presence_threshold: float,
    name_diff_threshold: float,
    marker_presence_threshold: float,
    marker_diff_threshold: float,
    merge_gap_sec: float,
    pad_sec: float,
) -> list[tuple[float, float]]:
    if not metrics:
        return []
    ranges: list[tuple[float, float]] = []
    st: float | None = None
    ed: float | None = None
    last_split_at = -1.0
    prev_marker_on = False
    for m in metrics:
        name_active = m.name_presence >= name_presence_threshold
        marker_on = m.marker_presence >= marker_presence_threshold

        if not name_active:
            if st is not None and ed is not None and (m.timestamp - ed) > merge_gap_sec:
                ranges.append((max(0.0, st - pad_sec), ed + pad_sec))
                st = None
                ed = None
            prev_marker_on = marker_on
            continue

        if st is None:
            st = m.timestamp
            ed = m.timestamp
            prev_marker_on = marker_on
            continue

        marker_toggled = marker_on != prev_marker_on
        if marker_toggled and (m.timestamp - last_split_at) >= 0.4 and (ed - st) >= 0.35:
            ranges.append((max(0.0, st - pad_sec), ed + pad_sec))
            st = m.timestamp
            ed = m.timestamp
            last_split_at = m.timestamp
            prev_marker_on = marker_on
            continue

        if (m.timestamp - ed) <= merge_gap_sec:
            ed = m.timestamp
        else:
            ranges.append((max(0.0, st - pad_sec), ed + pad_sec))
            st = m.timestamp
            ed = m.timestamp
        prev_marker_on = marker_on

    if st is not None and ed is not None:
        ranges.append((max(0.0, st - pad_sec), ed + pad_sec))
    return ranges


def _pick_sample_indices(frame_indices: list[int], max_candidates: int) -> list[int]:
    if not frame_indices:
        return []
    unique = sorted(set(frame_indices))
    if len(unique) <= max_candidates:
        return unique
    picks = np.linspace(0, len(unique) - 1, max_candidates, dtype=int)
    return [unique[i] for i in picks]


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _backup_subtitles_to_work(output_dir: Path, work_dir: Path) -> None:
    files = [
        "subtitles.srt",
        "subtitles.ass",
        "subtitles_debug.srt",
        "subtitles_debug.ass",
    ]
    for name in files:
        src = output_dir / name
        if not src.exists():
            continue
        dst = work_dir / name
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception as exc:
            _log(f"Warning: failed to backup subtitle {name} to work dir: {exc.__class__.__name__}")


def _srt_time(sec: float) -> str:
    ms = int(round(float(sec) * 1000))
    hh = ms // 3600000
    ms -= hh * 3600000
    mm = ms // 60000
    ms -= mm * 60000
    ss = ms // 1000
    ms -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _subtitle_cache_key(segment_id: int, time_start: float, time_end: float, dialogue_type: str) -> str:
    st = int(round(float(time_start) * 1000))
    ed = int(round(float(time_end) * 1000))
    return f"id={int(segment_id)}|{st}-{ed}|{dialogue_type}"


def _load_translation_cache(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    if not path.exists():
        return {}, {}
    try:
        raw = _load_json(path)
    except Exception:
        return {}, {}
    entries = raw.get("entries", []) if isinstance(raw, dict) else []
    by_full_key: dict[str, dict[str, str]] = {}
    by_time_type: dict[str, dict[str, str]] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        text = _normalize_quotes_for_subtitle(str(e.get("translation_subtitle", "") or "").strip())
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
        entries.extend(copy.deepcopy(prefix_entries))
    for seg, dbg in zip(segments, debug_texts):
        entries.append(
            {
                "segment_id": seg.segment_id,
                "time_start": seg.time_start,
                "time_end": seg.time_end,
                "srt_start": _srt_time(seg.time_start),
                "srt_end": _srt_time(seg.time_end),
                "dialogue_type": seg.dialogue_type,
                "speaker": seg.speaker,
                "text_original": seg.text_original,
                "translation_subtitle": seg.translation_subtitle,
                "debug_subtitle": dbg,
            }
        )
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
        text = _normalize_quotes_for_subtitle(str(e.get("translation_subtitle", "") or ""))
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


def _assert_crop_size(paths: list[Path], roi: Roi, tag: str) -> None:
    if not paths:
        return
    try:
        with Image.open(paths[0]) as im:
            w, h = im.size
    except Exception:
        return
    if abs(w - roi.width) > 1 or abs(h - roi.height) > 1:
        raise RuntimeError(
            f"{tag} crop size mismatch: got {w}x{h}, expected {roi.width}x{roi.height} (tolerance=1px). "
            "Please check ROI config and ffmpeg crop filter."
        )
    if (w, h) != (roi.width, roi.height):
        _log(
            f"Warning: {tag} crop is {w}x{h}, ROI is {roi.width}x{roi.height}; "
            "within 1px tolerance."
        )


class NameOcrRunner:
    """Name ROI OCR runner with optional multi-thread parallelism."""

    def __init__(self, ocr_cfg: dict[str, Any], fallback_engine: Any, workers: int = 1) -> None:
        self._ocr_cfg = ocr_cfg
        self._fallback_engine = fallback_engine
        self._workers = max(1, int(workers))
        self._presence_mode = str(ocr_cfg.get("name_presence_mode", "fast_mask")).lower()
        self._mask_thr_on = float(ocr_cfg.get("name_presence_threshold_on", 0.018))
        self._mask_thr_off = float(ocr_cfg.get("name_presence_threshold_off", 0.012))
        self._use_ocr_fallback = bool(ocr_cfg.get("name_presence_ocr_fallback", True))
        self._pool: ThreadPoolExecutor | None = None
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self._stats = {
            "total": 0,
            "mask_only": 0,
            "ocr_fallback": 0,
            "ocr_verify": 0,
        }
        if self._workers > 1:
            self._pool = ThreadPoolExecutor(max_workers=self._workers)

    @property
    def workers(self) -> int:
        return self._workers

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _thread_engine(self) -> Any:
        eng = getattr(self._local, "engine", None)
        if eng is None:
            eng = build_ocr_engine(self._ocr_cfg)
            self._local.engine = eng
        return eng

    def _inc_stat(self, key: str) -> None:
        with self._stats_lock:
            self._stats["total"] += 1
            self._stats[key] += 1

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def _has_text_fast_mask(self, image_path: Path) -> bool | None:
        gray = load_gray(image_path)
        _, presence = extract_text_features(gray, mode="name")
        if presence >= self._mask_thr_on:
            self._inc_stat("mask_only")
            return True
        if presence <= self._mask_thr_off:
            self._inc_stat("mask_only")
            return False
        return None

    def has_text(self, image_path: Path) -> bool:
        if not image_path.exists():
            return False
        if self._presence_mode == "fast_mask":
            fast = self._has_text_fast_mask(image_path)
            if fast is not None:
                return fast
            if not self._use_ocr_fallback:
                # Uncertain band defaults to "no name" when OCR fallback is disabled.
                self._inc_stat("mask_only")
                return False
            self._inc_stat("ocr_fallback")
        else:
            self._inc_stat("ocr_fallback")
        engine = self._fallback_engine if self._pool is None else self._thread_engine()
        r = engine.recognize(image_path)
        return bool((r.text or "").strip())

    def has_text_mask(self, image_path: Path) -> bool:
        """Mask-only name presence check (no OCR fallback)."""
        if not image_path.exists():
            return False
        fast = self._has_text_fast_mask(image_path)
        if fast is None:
            # Uncertain band treats as no-name for conservative split anchoring.
            self._inc_stat("mask_only")
            return False
        return bool(fast)

    def has_text_ocr(self, image_path: Path) -> bool:
        if not image_path.exists():
            return False
        with self._stats_lock:
            self._stats["ocr_verify"] += 1
        engine = self._fallback_engine if self._pool is None else self._thread_engine()
        r = engine.recognize(image_path)
        text = (r.text or "").strip()
        # _log(f"[has_text_ocr] {image_path.name} text={text!r}")
        
        text_compact = "".join(ch for ch in text if not ch.isspace())

        noisy_chars = set("-1|lI")
        filtered = "".join(ch for ch in text_compact if ch not in noisy_chars)

        return len(filtered) >= 2

    def has_text_batch(self, image_paths: list[Path]) -> list[bool]:
        if not image_paths:
            return []
        if self._pool is None or len(image_paths) <= 1:
            return [self.has_text(p) for p in image_paths]
        out = [False] * len(image_paths)
        fut_to_idx: dict[Future[bool], int] = {}
        for i, p in enumerate(image_paths):
            fut = self._pool.submit(self.has_text, p)
            fut_to_idx[fut] = i
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            out[i] = bool(fut.result())
        return out

    def has_text_batch_ocr(self, image_paths: list[Path]) -> list[bool]:
        if not image_paths:
            return []
        if self._pool is None or len(image_paths) <= 1:
            return [self.has_text_ocr(p) for p in image_paths]
        out = [False] * len(image_paths)
        fut_to_idx: dict[Future[bool], int] = {}
        for i, p in enumerate(image_paths):
            fut = self._pool.submit(self.has_text_ocr, p)
            fut_to_idx[fut] = i
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            out[i] = bool(fut.result())
        return out


def _export_marker_frames_for_segment(
    out_root: Path,
    seg_id: int,
    marker_paths: list[Path],
    metrics: list[FrameMetric],
    frame_start: int,
    frame_end: int,
    marker_threshold: float,
) -> None:
    seg_dir = out_root / f"seg_{seg_id:04d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    hi = min(frame_end, len(marker_paths) - 1, len(metrics) - 1)
    lo = max(0, frame_start)
    if hi < lo:
        return
    for i in range(lo, hi + 1):
        src = marker_paths[i]
        if not src.exists():
            continue
        ts = float(metrics[i].timestamp)
        score = float(metrics[i].marker_presence)
        on = 1 if score >= marker_threshold else 0
        dst = seg_dir / f"f{i:06d}_t{ts:08.3f}_s{score:0.3f}_on{on}.png"
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


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


def _export_name_frames_for_segment(
    out_root: Path,
    seg_label: str,
    name_dir: Path,
    frame_start: int,
    frame_end: int,
) -> None:
    seg_key = seg_label.replace(" ", "_")
    seg_dir = out_root / seg_key
    seg_dir.mkdir(parents=True, exist_ok=True)
    if frame_end < frame_start:
        return
    for fid in range(frame_start, frame_end + 1):
        src = name_dir / f"{fid + 1:06d}.png"
        if not src.exists():
            continue
        try:
            gray = load_gray(src)
            _, mask_presence = extract_text_features(gray, mode="name")
            dst = seg_dir / f"f{fid:06d}_m{mask_presence:.4f}.png"
        except Exception:
            dst = seg_dir / f"f{fid:06d}.png"
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def _build_refined_subsegment(
    raw: dict[str, Any],
    marker_seg_id: int,
    start_sec: float,
    scan_fps: float,
    marker_thr: float,
    st: int,
    ed: int,
    has_name: bool,
) -> dict[str, Any]:
    sample = [] if (not has_name) else [st + (ed - st) // 2]
    return {
        "range_index": int(raw["range_index"]),
        "frame_start": st,
        "frame_end": ed,
        "time_start": start_sec + (st / scan_fps),
        "time_end": start_sec + (ed / scan_fps),
        "has_name": bool(has_name),
        "sample_frames": sample,
        "scan_fps": scan_fps,
        "start_sec": start_sec,
        "stable_time": (
            start_sec + (sample[0] / scan_fps)
            if sample
            else start_sec + ((st + ed) / (2.0 * scan_fps))
        ),
        "marker_presence_threshold_used": marker_thr,
        "marker_seg_id": marker_seg_id,
    }


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


def _last_true_run_start(flags: list[bool]) -> int | None:
    true_idx = [i for i, v in enumerate(flags) if v]
    if not true_idx:
        return None
    run_end = true_idx[-1]
    run_start = run_end
    while run_start > 0 and flags[run_start - 1]:
        run_start -= 1
    return run_start


def _head_probe_hits_ocr(
    name_dir: Path,
    frame_start: int,
    frame_end: int,
    scan_fps: float,
    fast_check_frames: int,
    name_ocr: NameOcrRunner,
    name_cache: dict[int, bool],
) -> int:
    probe_n = max(1, int(fast_check_frames))
    probe_offset = max(0, int(round(0.16 * scan_fps)))
    probe_start = min(frame_end, frame_start + probe_offset)
    probe_end = min(frame_end, probe_start + probe_n - 1)
    probe_fids = list(range(probe_start, probe_end + 1))
    probe_paths = [name_dir / f"{fid + 1:06d}.png" for fid in probe_fids]
    probe_flags = name_ocr.has_text_batch_ocr(probe_paths)
    for fid, flag in zip(probe_fids, probe_flags):
        name_cache[fid] = bool(flag)
    return sum(1 for x in probe_flags if x)


def _split_segment_by_name_ocr(
    raw: dict[str, Any],
    name_dir: Path,
    name_ocr: NameOcrRunner,
    fast_check_frames: int = 5,
    fast_min_hits: int = 4,
    coarse_step_frames: int = 6,
    smooth_blank_gap_frames: int = 1,
    min_blank_frames: int = 2,
    blank_verify_frames: int = 3,
    blank_verify_min_hits: int = 1,
    confirm_lookback_frames: int = 10,
) -> list[dict[str, Any]]:
    frame_start = int(raw["frame_start"])
    frame_end = int(raw["frame_end"])
    scan_fps = float(raw["scan_fps"])
    start_sec = float(raw["start_sec"])
    marker_thr = float(raw.get("marker_presence_threshold_used", 0.0))
    marker_seg_id = int(raw.get("marker_seg_id", raw.get("segment_id", 0)))
    if frame_end < frame_start:
        return [raw]
    
    log_prefix = (
        f"[split_by_name_ocr] seg_id={marker_seg_id} "
        f"frames=[{frame_start},{frame_end}]"
    )

    name_mask_cache: dict[int, bool] = {}
    name_ocr_cache: dict[int, bool] = {}
    name_cache: dict[int, bool] = {}

    def _has_name_mask(fid: int) -> bool:
        if fid in name_mask_cache:
            return name_mask_cache[fid]
        p = name_dir / f"{fid + 1:06d}.png"
        v = name_ocr.has_text_mask(p)
        name_mask_cache[fid] = bool(v)
        return bool(v)

    def _has_name_ocr(fid: int) -> bool:
        if fid in name_ocr_cache:
            return name_ocr_cache[fid]
        p = name_dir / f"{fid + 1:06d}.png"
        v = name_ocr.has_text_ocr(p)
        name_ocr_cache[fid] = bool(v)
        return bool(v)

    def _blank_confirm(start_f: int, end_f: int) -> bool:
        # Returns True only if this run is confirmed blank by OCR sampling.
        if end_f < start_f:
            return True
        k = max(1, int(blank_verify_frames))
        need = max(1, int(blank_verify_min_hits))
        step = 3

        head = [start_f + i * step for i in range(k) if start_f + i * step <= end_f]
        tail = [end_f - i * step for i in range(k) if end_f - i * step >= start_f]

        sample_frames = sorted(set(head + tail))

        verify_paths = [name_dir / f"{f + 1:06d}.png" for f in sample_frames]
        vflags = name_ocr.has_text_batch_ocr(verify_paths)

        hits = sum(1 for x in vflags if x)
        ok = hits < need
        _log(
            f"{log_prefix} blank_confirm range=[{start_f},{end_f}] "
            f"samples={sample_frames} hits={hits} need_lt={need} result={ok}"
        )
        return ok
    
    def _find_dialog_start_by_backward_ocr_linear(start_f: int, end_f: int) -> int | None:
        """Fine scan: frame-by-frame backward OCR search."""
        seen_name = False
        for fid in range(end_f, start_f - 1, -1):
            if _has_name_ocr(fid):
                seen_name = True
                continue
            if seen_name:
                return fid + 1
        if seen_name:
            return start_f
        return None

    def _find_dialog_start_by_backward_ocr(start_f: int, end_f: int) -> int | None:
        """
        Two-stage backward OCR search:
        1) coarse scan with step=coarse_step_frames to quickly locate transition range;
        2) fine scan (frame-by-frame) inside that range.
        """
        step = max(1, int(coarse_step_frames))
        if step <= 1:
            return _find_dialog_start_by_backward_ocr_linear(start_f, end_f)

        coarse_points = list(range(end_f, start_f - 1, -step))
        if not coarse_points or coarse_points[-1] != start_f:
            coarse_points.append(start_f)

        seen_name = False
        prev_fid: int | None = None
        for fid in coarse_points:
            has_name = _has_name_ocr(fid)
            if has_name:
                seen_name = True
            elif seen_name:
                # Transition is between current coarse no-name point and previous
                # coarse point where trailing name block is already observed.
                hi = prev_fid if prev_fid is not None else end_f
                lo = max(start_f, fid)
                hi = min(end_f, hi)
                if hi < lo:
                    hi, lo = lo, hi
                found = _find_dialog_start_by_backward_ocr_linear(lo, hi)
                if found is not None:
                    return found
                # Conservative fallback: split after coarse no-name point.
                return min(end_f, lo + 1)
            prev_fid = fid

        if seen_name:
            return start_f
        return None

    def _find_dialog_start_by_forward_ocr(start_f: int, end_f: int) -> int | None:
        """
        Two-stage forward OCR search:
        1) coarse scan with step=coarse_step_frames to quickly locate first name frame;
        2) fine scan (frame-by-frame) inside that coarse interval.
        """
        if end_f < start_f:
            _log(
                f"{log_prefix} forward_ocr skip_invalid_range "
                f"start_f={start_f} end_f={end_f}"
            )
            return None

        step = max(1, int(coarse_step_frames))
        reject_gap = 2
        min_true_run = 2

        def _stream_scan(
            lo: int,
            hi: int,
            candidate: int | None,
            false_run: int,
            run_start: int | None,
            run_len: int,
        ) -> tuple[int | None, int, int | None, int]:
            for ff in range(lo, hi + 1):
                v = _has_name_ocr(ff)
                if v:
                    if run_len <= 0:
                        run_start = ff
                        run_len = 1
                    else:
                        run_len += 1
                    if candidate is None:
                        if run_len >= min_true_run and run_start is not None:
                            candidate = run_start
                    elif false_run >= reject_gap and run_len >= min_true_run and run_start is not None:
                        candidate = run_start
                    false_run = 0
                else:
                    run_start = None
                    run_len = 0
                    if candidate is not None:
                        false_run += 1
            return candidate, false_run, run_start, run_len

        if step <= 1:
            candidate, false_run, _run_start, _run_len = _stream_scan(
                start_f,
                end_f,
                None,
                0,
                None,
                0,
            )
            # _log(
            #     f"{log_prefix} forward_ocr linear_scan "
            #     f"range=[{start_f},{end_f}] candidate={candidate} "
            #     f"false_run={false_run} run_start={_run_start} run_len={_run_len}"
            # )
            if candidate is None:
                return None
            if false_run >= reject_gap:
                return None
            return candidate

        coarse_points = list(range(start_f, end_f + 1, step))
        if not coarse_points or coarse_points[-1] != end_f:
            coarse_points.append(end_f)

        prev_fid = start_f
        candidate: int | None = None
        false_run = 0
        run_start: int | None = None
        run_len = 0
        activated = False

        for fid in coarse_points:
            coarse_hit = _has_name_ocr(fid)
            # _log(
            #     f"{log_prefix} forward_ocr coarse_point "
            #     f"fid={fid} has_name_ocr={coarse_hit}"
            # )

            if not activated and not coarse_hit:
                prev_fid = fid + 1
                continue

            activated = True
            lo = max(start_f, prev_fid)
            hi = min(end_f, fid)
            candidate, false_run, run_start, run_len = _stream_scan(
                lo,
                hi,
                candidate,
                false_run,
                run_start,
                run_len,
            )
            # _log(
            #     f"{log_prefix} forward_ocr stream_scan "
            #     f"range=[{lo},{hi}] result_candidate={candidate} "
            #     f"false_run={false_run} run_start={run_start} run_len={run_len}"
            # )

            prev_fid = fid + 1

        # _log(
        #     f"{log_prefix} forward_ocr final "
        #     f"candidate={candidate} false_run={false_run} reject_gap={reject_gap}"
        # )
        if candidate is None:
            return None
        if false_run >= reject_gap:
            return None
        return candidate
    
    # Step-2: quick pure-dialogue probe.
    probe_hits_need = max(1, int(fast_min_hits))
    probe_hits = _head_probe_hits_ocr(
        name_dir=name_dir,
        frame_start=frame_start,
        frame_end=frame_end,
        scan_fps=scan_fps,
        fast_check_frames=fast_check_frames,
        name_ocr=name_ocr,
        name_cache=name_cache,
    )
    for fid, flag in name_cache.items():
        name_ocr_cache[int(fid)] = bool(flag)
    if probe_hits >= probe_hits_need:
        _log(
            f"{log_prefix} abandon_blank reason=head_probe_hits "
            f"probe_hits={probe_hits} need={probe_hits_need}"
        )
        return [
            _build_refined_subsegment(
                raw=raw,
                marker_seg_id=marker_seg_id,
                start_sec=start_sec,
                scan_fps=scan_fps,
                marker_thr=marker_thr,
                st=frame_start,
                ed=frame_end,
                has_name=True,
            )
        ]

    # Step-3: blank+dialogue form -> search backward from marker-present anchor.
    anchor = frame_end
    samples = raw.get("sample_frames") or []
    if samples:
        try:
            anchor = int(samples[0])
        except Exception:
            anchor = frame_end
    anchor = max(frame_start, min(frame_end, anchor))

    fids = list(range(frame_start, anchor + 1))
    # Use mask to find candidate boundary first.
    flags = [_has_name_mask(fid) for fid in fids]
    # Fill tiny mask dropouts to keep one continuous candidate block.
    _fill_short_false_gaps(flags, max_gap=max(0, int(smooth_blank_gap_frames)))

    run_start = _last_true_run_start(flags)
    if run_start is None:
        # Mask missed everything: force backward OCR search instead of marking
        # whole marker segment as blank.
        dialog_start = _find_dialog_start_by_backward_ocr(frame_start, anchor)
        if dialog_start is None:
            _log(f"{log_prefix} abandon_blank reason=mask_miss_and_backward_ocr_failed anchor={anchor}")
            # Marker-based segment should contain dialogue text; fall back to
            # whole has-name segment rather than blank-only segment.
            return [
                _build_refined_subsegment(
                    raw=raw,
                    marker_seg_id=marker_seg_id,
                    start_sec=start_sec,
                    scan_fps=scan_fps,
                    marker_thr=marker_thr,
                    st=frame_start,
                    ed=frame_end,
                    has_name=True,
                )
            ]
        # If no leading blank remains after backward OCR, keep as pure dialogue.
        if dialog_start <= frame_start:
            _log(
                f"{log_prefix} abandon_blank reason=dialog_start_not_after_head "
                f"dialog_start={dialog_start} frame_start={frame_start}"
            )
            return [
                _build_refined_subsegment(
                    raw=raw,
                    marker_seg_id=marker_seg_id,
                    start_sec=start_sec,
                    scan_fps=scan_fps,
                    marker_thr=marker_thr,
                    st=frame_start,
                    ed=frame_end,
                    has_name=True,
                )
            ]
        min_blank = max(1, int(min_blank_frames))
        if (dialog_start - frame_start) < min_blank:
            _log(
                f"{log_prefix} abandon_blank reason=leading_blank_too_short "
                f"blank_len={dialog_start - frame_start} min_blank={min_blank}"
            )
            return [
                _build_refined_subsegment(
                    raw=raw,
                    marker_seg_id=marker_seg_id,
                    start_sec=start_sec,
                    scan_fps=scan_fps,
                    marker_thr=marker_thr,
                    st=frame_start,
                    ed=frame_end,
                    has_name=True,
                )
            ]
        if not _blank_confirm(frame_start, dialog_start - 1):
            _log(
                f"{log_prefix} abandon_blank reason=blank_confirm_failed "
                f"blank_range=[{frame_start},{dialog_start - 1}]"
            )
            return [
                _build_refined_subsegment(
                    raw=raw,
                    marker_seg_id=marker_seg_id,
                    start_sec=start_sec,
                    scan_fps=scan_fps,
                    marker_thr=marker_thr,
                    st=frame_start,
                    ed=frame_end,
                    has_name=True,
                )
            ]
        return [
            _build_refined_subsegment(
                raw=raw,
                marker_seg_id=marker_seg_id,
                start_sec=start_sec,
                scan_fps=scan_fps,
                marker_thr=marker_thr,
                st=frame_start,
                ed=dialog_start - 1,
                has_name=False,
            ),
            _build_refined_subsegment(
                raw=raw,
                marker_seg_id=marker_seg_id,
                start_sec=start_sec,
                scan_fps=scan_fps,
                marker_thr=marker_thr,
                st=dialog_start,
                ed=frame_end,
                has_name=True,
            ),
        ]

    # Candidate dialogue starts at first True of last contiguous True-run.
    candidate_start = frame_start + run_start
    # OCR confirmation: coarse+fine forward search.
    # Search from candidate_start-lookback up to anchor, and do not require mask=True.
    confirm_start = max(frame_start, candidate_start - max(0, int(confirm_lookback_frames)))
    dialog_start = _find_dialog_start_by_forward_ocr(confirm_start, anchor)
    if dialog_start is None:
        # Mask found candidate but OCR didn't confirm: force backward OCR search.
        dialog_start = _find_dialog_start_by_backward_ocr(frame_start, anchor)
        if dialog_start is None:
            _log(
                f"{log_prefix} abandon_blank reason=forward_backward_ocr_both_failed "
                f"candidate_start={candidate_start} confirm_start={confirm_start} anchor={anchor}"
            )
            return [
                _build_refined_subsegment(
                    raw=raw,
                    marker_seg_id=marker_seg_id,
                    start_sec=start_sec,
                    scan_fps=scan_fps,
                    marker_thr=marker_thr,
                    st=frame_start,
                    ed=frame_end,
                    has_name=True,
                )
            ]

    # If head blank too short, merge into dialogue to avoid jitter.
    min_blank = max(1, int(min_blank_frames))
    if (dialog_start - frame_start) < min_blank:
        _log(
            f"{log_prefix} abandon_blank reason=leading_blank_too_short "
            f"blank_len={dialog_start - frame_start} min_blank={min_blank}"
        )
        return [
            _build_refined_subsegment(
                raw=raw,
                marker_seg_id=marker_seg_id,
                start_sec=start_sec,
                scan_fps=scan_fps,
                marker_thr=marker_thr,
                st=frame_start,
                ed=frame_end,
                has_name=True,
            )
        ]

    # Blank leading part must be OCR-confirmed blank; otherwise merge.
    if not _blank_confirm(frame_start, dialog_start - 1):
        _log(
            f"{log_prefix} abandon_blank reason=blank_confirm_failed "
            f"blank_range=[{frame_start},{dialog_start - 1}]"
        )
        return [
            _build_refined_subsegment(
                raw=raw,
                marker_seg_id=marker_seg_id,
                start_sec=start_sec,
                scan_fps=scan_fps,
                marker_thr=marker_thr,
                st=frame_start,
                ed=frame_end,
                has_name=True,
            )
        ]

    return [
        _build_refined_subsegment(
            raw=raw,
            marker_seg_id=marker_seg_id,
            start_sec=start_sec,
            scan_fps=scan_fps,
            marker_thr=marker_thr,
            st=frame_start,
            ed=dialog_start - 1,
            has_name=False,
        ),
        _build_refined_subsegment(
            raw=raw,
            marker_seg_id=marker_seg_id,
            start_sec=start_sec,
            scan_fps=scan_fps,
            marker_thr=marker_thr,
            st=dialog_start,
            ed=frame_end,
            has_name=True,
        ),
    ]


def _normalize_name_subsegments_per_marker(
    segs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not segs:
        return []
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(segs):
        sid = int(segs[i].get("marker_seg_id", -1))
        grp: list[dict[str, Any]] = []
        while i < len(segs) and int(segs[i].get("marker_seg_id", -1)) == sid:
            grp.append(segs[i])
            i += 1
        if not grp:
            continue
        scan_fps = float(grp[0]["scan_fps"])
        start_sec = float(grp[0]["start_sec"])
        marker_thr = float(grp[0].get("marker_presence_threshold_used", 0.0))
        range_index = int(grp[0]["range_index"])
        has_true = any(bool(g.get("has_name", False)) for g in grp)
        fs = int(grp[0]["frame_start"])
        fe = int(grp[-1]["frame_end"])
        if not has_true:
            out.append(
                {
                    "range_index": range_index,
                    "frame_start": fs,
                    "frame_end": fe,
                    "time_start": start_sec + (fs / scan_fps),
                    "time_end": start_sec + (fe / scan_fps),
                    "has_name": False,
                    "sample_frames": [],
                    "scan_fps": scan_fps,
                    "start_sec": start_sec,
                    "stable_time": start_sec + ((fs + fe) / (2.0 * scan_fps)),
                    "marker_presence_threshold_used": marker_thr,
                    "marker_seg_id": sid,
                }
            )
            continue
        first_true_idx = next(
            j for j, g in enumerate(grp) if bool(g.get("has_name", False))
        )
        if first_true_idx > 0:
            bfs = int(grp[0]["frame_start"])
            bfe = int(grp[first_true_idx - 1]["frame_end"])
            if bfe >= bfs:
                out.append(
                    {
                        "range_index": range_index,
                        "frame_start": bfs,
                        "frame_end": bfe,
                        "time_start": start_sec + (bfs / scan_fps),
                        "time_end": start_sec + (bfe / scan_fps),
                        "has_name": False,
                        "sample_frames": [],
                        "scan_fps": scan_fps,
                        "start_sec": start_sec,
                        "stable_time": start_sec + ((bfs + bfe) / (2.0 * scan_fps)),
                        "marker_presence_threshold_used": marker_thr,
                        "marker_seg_id": sid,
                    }
                )
        dfs = int(grp[first_true_idx]["frame_start"])
        dfe = int(grp[-1]["frame_end"])
        dmid = dfs + ((dfe - dfs) // 2)
        out.append(
            {
                "range_index": range_index,
                "frame_start": dfs,
                "frame_end": dfe,
                "time_start": start_sec + (dfs / scan_fps),
                "time_end": start_sec + (dfe / scan_fps),
                "has_name": True,
                "sample_frames": [dmid],
                "scan_fps": scan_fps,
                "start_sec": start_sec,
                "stable_time": start_sec + (dmid / scan_fps),
                "marker_presence_threshold_used": marker_thr,
                "marker_seg_id": sid,
            }
        )
    return out


def _pick_marker_anchor_frame(
    raw: dict[str, Any],
    from_end_frames: int = 3,
) -> int:
    frame_start = int(raw.get("frame_start", 0))
    frame_end = int(raw.get("frame_end", frame_start))
    if frame_end < frame_start:
        return frame_start
    n = max(1, int(from_end_frames))
    return max(frame_start, frame_end - (n - 1))


def _resolve_run_prefix(args: argparse.Namespace) -> str:
    if bool(getattr(args, "subtitles_from_cache", False)):
        return "run_subtitle"
    if bool(getattr(args, "cache_only", False)):
        return "run_cache"
    return "run_full"


def run_pipeline(args: argparse.Namespace) -> int:
    global _LOG_FILE_PATH
    cfg = load_config(args.config)
    project_root = Path(__file__).resolve().parents[2]
    video_arg = str(getattr(args, "video", "") or "").strip()
    cfg_video = str(cfg.get("video_path", "") or "").strip()
    if not video_arg and not cfg_video:
        raise RuntimeError("Missing input video. Provide --video or set video_path in config.")
    video_path_raw = video_arg or cfg_video
    video_path_obj = Path(video_path_raw)
    if not video_path_obj.is_absolute():
        cand_root = (project_root / video_path_obj).resolve()
        cand_cfg = (Path(args.config).resolve().parent / video_path_obj).resolve()
        if cand_root.exists():
            video_path_obj = cand_root
        else:
            video_path_obj = cand_cfg
    if not video_path_obj.exists():
        raise RuntimeError(f"Input video does not exist: {video_path_obj}")
    # Normalize for downstream ffmpeg/ffprobe calls.
    args.video = str(video_path_obj)
    output_dir_arg = str(getattr(args, "output_dir", "") or "").strip()
    if output_dir_arg:
        output_dir = Path(output_dir_arg)
    else:
        output_dir = project_root / "outputs" / video_path_obj.stem
    base_work_dir = output_dir / "work"
    run_prefix = _resolve_run_prefix(args)
    run_work_dir = (
        base_work_dir / datetime.now().strftime(f"{run_prefix}_%Y%m%d_%H%M%S")
        if not args.resume
        else base_work_dir / "resume"
    )
    work_dir = run_work_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_work_runs(base_work_dir, keep_latest=3)
    debug_mode = bool(getattr(args, "debug", False))

    general_cfg = cfg.get("general", {})
    tr_cfg = cfg.get("translation", {})
    vlm_io_log_path = work_dir / "vlm_io.log"
    _LOG_FILE_PATH = work_dir / "run.log"
    if _LOG_FILE_PATH.exists():
        try:
            _LOG_FILE_PATH.unlink()
        except Exception:
            pass
    vlm_io_log_enabled = bool(tr_cfg.get("io_log_enabled", False))
    if vlm_io_log_enabled and vlm_io_log_path.exists():
        try:
            vlm_io_log_path.unlink()
        except Exception:
            pass
    _log(f"Log file: {_LOG_FILE_PATH}")
    if vlm_io_log_enabled:
        _log(f"VLM IO log file: {vlm_io_log_path} (cleared previous if existed)")
    _log("Pipeline started.")
    cache_dir = work_dir / "cache"
    if debug_mode:
        _log("Debug mode enabled: keep intermediate fine_* files and export segment debug folders.")
    shared_cache_path = output_dir / "translation_cache_latest.json"
    run_cache_path = work_dir / "translation_cache.json"
    if getattr(args, "translation_cache", None):
        translation_cache_lookup_path = Path(args.translation_cache)
    elif bool(getattr(args, "subtitles_from_cache", False)):
        if shared_cache_path.exists():
            translation_cache_lookup_path = shared_cache_path
        else:
            latest_cache = _find_latest_translation_cache(base_work_dir)
            translation_cache_lookup_path = latest_cache if latest_cache is not None else run_cache_path
    else:
        # Normal run: reuse previous latest cache under output root.
        translation_cache_lookup_path = shared_cache_path if shared_cache_path.exists() else run_cache_path
    ffmpeg_path = Path(cfg["tools"]["ffmpeg_path"])
    ffprobe_path = Path(cfg["tools"]["ffprobe_path"])

    if bool(getattr(args, "subtitles_from_cache", False)):
        _log(f"Loading subtitle cache: {translation_cache_lookup_path}")
        if not translation_cache_lookup_path.exists():
            raise RuntimeError(
                f"subtitle cache not found: {translation_cache_lookup_path}. "
                "Please pass --translation-cache explicitly or run cache generation first."
            )
        entries = _load_cache_entries(translation_cache_lookup_path)
        if not entries:
            raise RuntimeError(f"subtitle cache has no entries: {translation_cache_lookup_path}")
        _log(f"Subtitle cache entries: {len(entries)}")
        if bool(getattr(args, "cache_only", False)):
            _log("Cache-only mode: skip subtitle generation.")
            if not debug_mode:
                _cleanup_intermediate_artifacts(work_dir)
            return 0
        _log("Reading video metadata.")
        video_meta = ffprobe_video(ffprobe_path, args.video)
        subtitle_location = cfg["roi"].get("subtitle_location", cfg["roi"].get("subtitle_roi"))
        if not isinstance(subtitle_location, list) or len(subtitle_location) != 4:
            raise RuntimeError("Invalid config: roi.subtitle_location must be [x1,y1,x2,y2]")
        title_translation_location = cfg["roi"].get(
            "title_translation_location",
            cfg["roi"].get("title_text_roi", cfg["roi"].get("title_subtitle_roi", subtitle_location)),
        )
        title_info_location = cfg["roi"].get(
            "title_info_location",
            cfg["roi"].get("title_info_roi", cfg["roi"].get("title_speaker_roi", subtitle_location)),
        )
        segs, dbg_segs = _segments_from_cache_entries(entries)
        write_srt(segs, output_dir / "subtitles.srt")
        write_ass(
            segs,
            output_dir / "subtitles.ass",
            video_width=video_meta.width,
            video_height=video_meta.height,
            subtitle_location=subtitle_location,
            title_translation_location=title_translation_location,
            title_info_location=title_info_location,
        )
        write_srt(dbg_segs, output_dir / "subtitles_debug.srt")
        write_ass(
            dbg_segs,
            output_dir / "subtitles_debug.ass",
            video_width=video_meta.width,
            video_height=video_meta.height,
            subtitle_location=subtitle_location,
            title_translation_location=title_translation_location,
            title_info_location=title_info_location,
        )
        _backup_subtitles_to_work(output_dir, work_dir)
        _log("Subtitles generated from translation cache.")
        if not debug_mode:
            _cleanup_intermediate_artifacts(work_dir)
        return 0

    _log("Reading video metadata.")
    video_meta = ffprobe_video(ffprobe_path, args.video)
    name_roi = _roi_from_cfg(cfg, "name_roi")
    dialogue_roi = _roi_from_cfg(cfg, "dialogue_roi")
    marker_roi = _roi_from_cfg(cfg, "marker_roi")
    subtitle_location = cfg["roi"].get("subtitle_location", cfg["roi"].get("subtitle_roi"))
    if not isinstance(subtitle_location, list) or len(subtitle_location) != 4:
        raise RuntimeError("Invalid config: roi.subtitle_location must be [x1,y1,x2,y2]")
    title_translation_location = cfg["roi"].get(
        "title_translation_location",
        cfg["roi"].get("title_text_roi", cfg["roi"].get("title_subtitle_roi", subtitle_location)),
    )
    title_info_location = cfg["roi"].get(
        "title_info_location",
        cfg["roi"].get("title_info_roi", cfg["roi"].get("title_speaker_roi", subtitle_location)),
    )
    fps_override = cfg["video"].get("fps_override")
    base_fps = float(fps_override) if fps_override else video_meta.fps
    _log(
        "Name ROI: "
        f"x={name_roi.x1},y={name_roi.y1},w={name_roi.width},h={name_roi.height}"
    )
    marker_matcher = None
    marker_cfg = cfg.get("marker", {})
    marker_workers = int(marker_cfg.get("parallel_workers", 1))
    marker_coarse_step = int(marker_cfg.get("coarse_step_frames", 1))
    marker_refine_margin = float(marker_cfg.get("refine_margin", 0.06))
    try:
        marker_anchor_from_end_frames = int(marker_cfg.get("ocr_anchor_from_end_frames", 3))
    except Exception:
        marker_anchor_from_end_frames = 3
    marker_anchor_from_end_frames = max(1, marker_anchor_from_end_frames)
    _log(f"Marker OCR anchor from end frames: {marker_anchor_from_end_frames}")
    base_marker_threshold = float(cfg["threshold"]["marker_presence_threshold"])
    marker_force_threshold_raw = marker_cfg.get("force_threshold", None)
    marker_force_threshold: float | None = None
    if marker_force_threshold_raw is not None:
        try:
            marker_force_threshold = float(marker_force_threshold_raw)
        except Exception:
            marker_force_threshold = None
        if marker_force_threshold is not None:
            marker_force_threshold = max(0.0, min(1.0, marker_force_threshold))
    if bool(marker_cfg.get("use_template", True)):
        template_paths = _collect_marker_templates(marker_cfg)
        if template_paths:
            marker_matcher = MarkerTemplateMatcher(
                template_paths,
                center_width=(
                    int(marker_cfg.get("template_center_width"))
                    if marker_cfg.get("template_center_width") is not None
                    else None
                ),
                vertical_shift_px=int(marker_cfg.get("vertical_shift_px", 0)),
                vertical_shift_step=int(marker_cfg.get("vertical_shift_step", 1)),
                horizontal_shift_px=int(marker_cfg.get("horizontal_shift_px", 0)),
                horizontal_shift_step=int(marker_cfg.get("horizontal_shift_step", 1)),
                shift_mode=str(marker_cfg.get("shift_mode", "vertical")),
            )
            _log(
                f"Marker template enabled: {marker_matcher.template_count} templates "
                f"(first={template_paths[0]}) "
                f"(threshold={base_marker_threshold:.3f}, workers={max(1, marker_workers)}, "
                f"coarse_step={max(1, marker_coarse_step)})"
            )
        else:
            _log("Marker templates not found, fallback to mask mode.")

    ranges = [(0.0, video_meta.duration)]
    _log("Marker mode: scan full video once.")

    raw_seg_json = cache_dir / "segments_raw.json"
    if args.resume and raw_seg_json.exists():
        _log("Loading cached segmented ranges.")
        raw_segments = _load_json(raw_seg_json)
        for j, r in enumerate(raw_segments, start=1):
            if "marker_seg_id" not in r:
                r["marker_seg_id"] = int(r.get("segment_id", j))
            if "segment_id" not in r:
                r["segment_id"] = int(r.get("marker_seg_id", j))
    else:
        _log("Fine-scan + state-machine segmentation started.")
        raw_segments: list[dict[str, Any]] = []
        for ridx, (start_sec, end_sec) in enumerate(ranges):
            _log(f"Fine-scan window: {start_sec:.2f}s -> {end_sec:.2f}s")
            duration = max(0.01, end_sec - start_sec)
            scan_fps = 1.0 / float(cfg["video"]["sample_interval_active"])
            dialog_dir = work_dir / f"fine_{ridx:03d}_dialog"
            name_dir = work_dir / f"fine_{ridx:03d}_name"
            marker_dir = work_dir / f"fine_{ridx:03d}_marker"
            dialogue_dir = work_dir / f"fine_{ridx:03d}_dialogue"
            marker_thr_eff = base_marker_threshold
            if marker_force_threshold is not None:
                marker_thr_eff = marker_force_threshold
                _log(
                    f"Forced marker threshold: {marker_thr_eff:.3f} "
                    f"(base={base_marker_threshold:.3f})"
                )

            worker_stop_event = threading.Event()
            worker_end_holder: dict[str, int | None] = {"end": None}
            worker_score_cache: dict[int, float] = {}
            worker_cache_lock = threading.Lock()
            marker_worker: threading.Thread | None = None
            if marker_matcher is not None:
                marker_worker = threading.Thread(
                    target=_background_score_marker_and_prune_dialogue_cache,
                    kwargs={
                        "marker_dir": marker_dir,
                        "dialogue_cache_dir": dialogue_dir,
                        "marker_matcher": marker_matcher,
                        "marker_threshold": marker_thr_eff,
                        "stop_event": worker_stop_event,
                        "end_index_holder": worker_end_holder,
                        "score_cache": worker_score_cache,
                        "cache_lock": worker_cache_lock,
                        "verbose": debug_mode,
                    },
                    daemon=True,
                )
                marker_worker.start()
                _log("Marker prune worker started.")
            else:
                _log("Marker prune worker skipped (matcher unavailable).")

            dialogue_paths, name_paths, marker_paths = extract_sequence_dialogue_name_marker(
                ffmpeg_path=ffmpeg_path,
                video_path=args.video,
                dialogue_output_dir=dialogue_dir,
                name_output_dir=name_dir,
                marker_output_dir=marker_dir,
                fps=scan_fps,
                dialogue_crop_filter=dialogue_roi.as_crop_filter(),
                name_crop_filter=name_roi.as_crop_filter(),
                marker_crop_filter=marker_roi.as_crop_filter(),
                start_sec=start_sec,
                duration_sec=duration,
            )

            if marker_worker is not None:
                worker_stop_event.set()
                worker_end_holder["end"] = len(marker_paths)
                marker_worker.join()
                _log("Marker prune worker finished.")

            _assert_crop_size(name_paths, name_roi, "fine window name")
            dialog_paths = name_paths

            marker_stats: dict[str, int] = {}
            marker_scores_cached: list[float] | None = None
            if marker_matcher is not None and marker_paths:
                marker_scores_cached = [0.0] * len(marker_paths)
                with worker_cache_lock:
                    cached_items = list(worker_score_cache.items())
                for k, v in cached_items:
                    if 0 <= int(k) < len(marker_scores_cached):
                        marker_scores_cached[int(k)] = float(v)

                # If cache has holes due timing, re-read marker images to backfill scores.
                missing = [
                    i for i in range(len(marker_scores_cached))
                    if marker_scores_cached[i] <= 0.0
                ]
                if missing:
                    reread_ok = 0
                    reread_fail = 0
                    for i in missing:
                        p = marker_paths[i]
                        if not p.exists():
                            reread_fail += 1
                            continue
                        try:
                            marker_scores_cached[i] = float(marker_matcher.score(load_gray(p)))
                            reread_ok += 1
                        except Exception:
                            reread_fail += 1
                    _log(
                        "Marker score cache backfill by reread: "
                        f"missing={len(missing)} ok={reread_ok} fail={reread_fail} total={len(marker_scores_cached)}"
                    )

                pruned, prune_failed = _final_prune_dialogue_cache_by_scores(
                    dialogue_cache_dir=dialogue_dir,
                    marker_scores_cached=marker_scores_cached,
                    marker_threshold=marker_thr_eff,
                    log_tag=marker_dir.name or "marker",
                    verbose=debug_mode,
                )
                _log(
                    f"Post-prune dialogue_cache removed={pruned} failed={prune_failed}"
                )

            metrics = _metrics_from_frame_lists(
                dialog_paths,
                name_paths,
                marker_paths,
                scan_fps,
                start_sec,
                marker_matcher=marker_matcher,
                marker_workers=marker_workers,
                marker_coarse_step=marker_coarse_step,
                marker_refine_margin=marker_refine_margin,
                marker_threshold_hint=base_marker_threshold,
                marker_stats=marker_stats,
                marker_scores_cached=marker_scores_cached,
            )
            if marker_stats.get("total", 0) > 0:
                _log(
                    "Marker scoring: "
                    f"{marker_stats.get('computed', 0)}/{marker_stats.get('total', 0)} "
                    f"(step={marker_stats.get('coarse_step', max(1, marker_coarse_step))})"
                )
            if metrics:
                mvals = [float(m.marker_presence) for m in metrics]
                _log(
                    "Marker score stats: "
                    f"min={min(mvals):.3f} avg={float(np.mean(mvals)):.3f} "
                    f"p95={float(np.percentile(mvals, 95)):.3f} max={max(mvals):.3f}"
                )
            sm_cfg = StateMachineConfig(
                change_threshold=float(cfg["threshold"]["diff_dialogue_change"]),
                clear_threshold=float(cfg["threshold"]["diff_dialogue_clear"]),
                presence_threshold=float(cfg["threshold"]["presence_threshold"]),
                name_change_threshold=float(cfg["threshold"]["diff_name_change"]),
                name_presence_threshold=float(cfg["threshold"]["name_presence_threshold"]),
                split_on_name_change=bool(cfg["state_machine"].get("split_on_name_change", True)),
                use_marker_cue=bool(cfg["state_machine"].get("use_marker_cue", True)),
                marker_presence_threshold=marker_thr_eff,
                marker_min_on_frames=int(cfg["state_machine"].get("marker_min_on_frames", 5)),
                marker_min_off_frames=int(cfg["state_machine"].get("marker_min_off_frames", 4)),
                marker_smooth_window=int(cfg["state_machine"].get("marker_smooth_window", 7)),
                marker_use_debounce=bool(cfg["state_machine"].get("marker_use_debounce", True)),
                stable_frames=max(
                    1, int(float(cfg["threshold"]["stable_duration_sec"]) * scan_fps)
                ),
                clear_frames=max(1, int(float(cfg["threshold"]["clear_duration_sec"]) * scan_fps)),
                min_duration=float(cfg["threshold"]["min_dialogue_duration_sec"]),
            )
            cand = segment_from_metrics(metrics, sm_cfg)

            # After marker-based segmentation, only keep dialogue anchor frames that
            # can be used later by VLM (segment end minus configured lookback).
            keep_dialogue_frame_ids: set[int] = set()
            for seg in cand:
                if not bool(seg.has_name):
                    continue
                anchor_fid0 = max(int(seg.frame_start), int(seg.frame_end) - (marker_anchor_from_end_frames - 1))
                keep_dialogue_frame_ids.add(int(anchor_fid0) + 1)  # cache files are 1-based
            anchor_removed, anchor_failed, anchor_kept = _prune_dialogue_cache_to_anchor_frames(
                dialogue_cache_dir=dialogue_dir,
                keep_frame_ids_1based=keep_dialogue_frame_ids,
                log_tag=marker_dir.name or "marker",
                verbose=debug_mode,
            )
            _log(
                "Anchor-prune dialogue_cache "
                f"kept={anchor_kept} removed={anchor_removed} failed={anchor_failed} "
                f"anchors={len(keep_dialogue_frame_ids)}"
            )

            # Dialogue frames are written directly to fine_*_dialogue; pruning removes low-score frames.
            marker_frame_count = sum(1 for p in dialogue_paths if p.exists())
            _log(f"Dialogue frames: saved {marker_frame_count} marker frames from {len(dialogue_paths)} total")
            
            for seg in cand:
                picks = _pick_sample_indices(
                    seg.sample_frame_indices,
                    int(cfg["ocr"]["max_candidates_per_segment"]),
                )
                seg_id = len(raw_segments) + 1
                if debug_mode:
                    debug_marker_root = work_dir / "marker_by_segment"
                    _export_marker_frames_for_segment(
                        out_root=debug_marker_root,
                        seg_id=seg_id,
                        marker_paths=marker_paths,
                        metrics=metrics,
                        frame_start=int(seg.frame_start),
                        frame_end=int(seg.frame_end),
                        marker_threshold=marker_thr_eff,
                    )
                raw_segments.append(
                    {
                        "segment_id": seg_id,
                        "marker_seg_id": seg_id,
                        "range_index": ridx,
                        "frame_start": seg.frame_start,
                        "frame_end": seg.frame_end,
                        "time_start": seg.start_time,
                        "time_end": seg.end_time,
                        "has_name": seg.has_name,
                        "sample_frames": picks,
                        "scan_fps": scan_fps,
                        "start_sec": start_sec,
                        "stable_time": (
                            float(start_sec) + (float(picks[0]) / float(scan_fps))
                            if picks
                            else (float(seg.start_time) + float(seg.end_time)) / 2.0
                        ),
                        "marker_presence_threshold_used": marker_thr_eff,
                    }
                )
        _save_json(raw_seg_json, raw_segments)
        _log(f"Segmentation done. Raw segments: {len(raw_segments)}")

    _log("Initializing OCR engine (name split / OCR tasks).")
    ocr_engine = build_ocr_engine(cfg["ocr"])
    ocr_info = ocr_engine.info()
    _log(
        "OCR engine: "
        f"{ocr_info.get('engine')} / runtime={ocr_info.get('runtime')} / "
        f"target={ocr_info.get('target_lang')} / actual={ocr_info.get('actual_lang')} / "
        f"charset={ocr_info.get('charset_size')}"
    )
    if (
        str(cfg["ocr"].get("rapidocr_rec_lang", "")).lower().startswith("japan")
        and ocr_info.get("engine") == "rapidocr"
        and ocr_info.get("actual_lang") == "non_japan_like"
    ):
        raise RuntimeError(
            "OCR is configured for Japanese, but current RapidOCR model is non-Japanese. "
            "Please provide Japanese OCR model files or switch to paddleocr_cli with --lang japan."
        )
    name_ocr_workers = int(cfg["ocr"].get("name_ocr_workers", 1))
    name_ocr = NameOcrRunner(cfg["ocr"], ocr_engine, workers=name_ocr_workers)
    _log(f"Name OCR workers: {name_ocr.workers}")

    if bool(cfg["state_machine"].get("split_on_name_ocr", True)):
        _log("Refining segments by name-region OCR.")
        refined: list[dict[str, Any]] = []
        fast_frames = int(cfg["state_machine"].get("name_fast_check_frames", 5))
        fast_hits = int(cfg["state_machine"].get("name_fast_min_hits", 4))
        coarse_step = int(cfg["state_machine"].get("name_coarse_step_frames", 6))
        smooth_gap = int(cfg["state_machine"].get("name_smooth_blank_gap_frames", 1))
        general_cfg = cfg.get("general", {})
        min_blank = int(
            general_cfg.get(
                "blank_ignore_under_frames",
                cfg["state_machine"].get("name_min_blank_frames", 1),
            )
        )
        blank_verify_frames = int(cfg["state_machine"].get("name_blank_verify_frames", 3))
        blank_verify_hits = int(cfg["state_machine"].get("name_blank_verify_min_hits", 1))
        confirm_lookback_frames = int(general_cfg.get("name_confirm_lookback_frames", 10))
        _log(
            f"Name OCR fast-check: first {fast_frames} frames, hits>={fast_hits}; "
            f"coarse_step={coarse_step}; smooth_blank_gap={smooth_gap}; "
            f"blank_ignore_under={min_blank}; "
            f"blank_verify={blank_verify_frames}/{blank_verify_hits}; "
            f"confirm_lookback={confirm_lookback_frames}"
        )
        refine_total = len(raw_segments)
        refine_started_at = datetime.now()
        for idx, raw in enumerate(raw_segments, start=1):
            if idx == 1 or idx % 10 == 0 or idx == refine_total:
                elapsed = (datetime.now() - refine_started_at).total_seconds()
                _log(
                    f"Name OCR refine progress: {idx}/{refine_total} "
                    f"({(idx / max(1, refine_total)) * 100:.1f}%, elapsed={elapsed:.1f}s)"
                )
            ridx = int(raw["range_index"])
            name_dir = work_dir / f"fine_{ridx:03d}_name"
            refined.extend(
                _split_segment_by_name_ocr(
                    raw,
                    name_dir,
                    name_ocr,
                    fast_check_frames=fast_frames,
                    fast_min_hits=fast_hits,
                    coarse_step_frames=coarse_step,
                    smooth_blank_gap_frames=smooth_gap,
                    min_blank_frames=min_blank,
                    blank_verify_frames=blank_verify_frames,
                    blank_verify_min_hits=blank_verify_hits,
                    confirm_lookback_frames=confirm_lookback_frames,
                )
            )
        refined = _normalize_name_subsegments_per_marker(refined)
        st = name_ocr.stats()
        _log(
            "Name presence stats: "
            f"total={st.get('total', 0)} "
            f"mask_only={st.get('mask_only', 0)} "
            f"ocr_fallback={st.get('ocr_fallback', 0)} "
            f"ocr_verify={st.get('ocr_verify', 0)}"
        )
        _log(f"Name OCR refine: {len(raw_segments)} -> {len(refined)} segments.")
        raw_segments = refined
        _save_json(raw_seg_json, raw_segments)

    if args.skip_translation:
        _log("Skip-translation mode: dialogue OCR/VLM disabled; using debug subtitles.")
    general_cfg = cfg.get("general", {})
    tr_cfg = cfg.get("translation", {})
    enable_web_search = bool(general_cfg.get("enable_web_search", False))
    game_cfg = cfg.get("game", {})
    game_name = str(game_cfg.get("name", "")).strip()
    source_lang = str(game_cfg.get("source_language", "ja")).strip() or "ja"
    target_lang = str(
        tr_cfg.get("target_language", game_cfg.get("target_language", "zh-CN"))
    ).strip() or "zh-CN"
    llm_temperature = float(tr_cfg.get("temperature", 1.3))
    llm_enable_thinking = bool(tr_cfg.get("enable_thinking", True))
    llm_thinking_budget_raw = tr_cfg.get("thinking_budget", None)
    llm_thinking_budget: int | None = None
    if llm_thinking_budget_raw is not None and str(llm_thinking_budget_raw).strip() != "":
        try:
            llm_thinking_budget = int(llm_thinking_budget_raw)
        except Exception:
            llm_thinking_budget = None
    llm_preserve_thinking = bool(tr_cfg.get("preserve_thinking", False))
    vlm_io_log_enabled = bool(tr_cfg.get("io_log_enabled", False))
    vlm_translator: BailianVlmTranslator | None = None
    translation_mode = str(tr_cfg.get("mode", "vlm_bailian")).lower()
    translate_llm = str(tr_cfg.get("translate_llm", "")).strip().lower()
    if translate_llm in {"vlm", "vlm_bailian", "qwen_vlm", "bailian"}:
        translation_mode = "vlm_bailian"
    recog_mode = str(tr_cfg.get("recognition_mode", "")).strip().lower()
    if recog_mode in {"vlm", "vlm_translate", "image_translate"}:
        translation_mode = "vlm_bailian"
    if not args.skip_translation:
        if translation_mode != "vlm_bailian":
            _log(
                f"Translation mode '{translation_mode}' is disabled. "
                "Force switching to 'vlm_bailian' (local OCR text translation path removed)."
            )
        _log("Initializing Bailian VLM translator.")
        api_key = str(tr_cfg.get("api_key", "")).strip()
        if not api_key:
            api_key_file = _require_non_empty_translation_str(tr_cfg, "api_key_file")
            api_key = load_api_key(api_key_file)
        vlm_api = _require_non_empty_translation_str(tr_cfg, "vlm_api")
        vlm_translator = BailianVlmTranslator(
            api_key=api_key,
            model=_resolve_active_vlm_model(cfg),
            base_url=vlm_api,
            temperature=llm_temperature,
            enable_thinking=llm_enable_thinking,
            thinking_budget=llm_thinking_budget,
            preserve_thinking=llm_preserve_thinking,
            timeout_sec=int(tr_cfg.get("timeout_sec", 90)),
            timeout_backoff_sec=int(tr_cfg.get("timeout_backoff_sec", 15)),
            max_retries=int(tr_cfg.get("max_retries", 2)),
            retry_delay_sec=float(tr_cfg.get("retry_delay_sec", 1.5)),
            disable_env_proxy=bool(tr_cfg.get("disable_env_proxy", True)),
            game_name=game_name,
            source_language=source_lang,
            target_language=target_lang,
            log_fn=lambda m: _log(f"[VLM] {m}"),
            io_log_path=work_dir / "vlm_io.log",
            io_log_enabled=vlm_io_log_enabled,
            enable_web_search=enable_web_search,
        )
        _log(
            "Translation mode: VLM "
            f"({tr_cfg.get('model', 'qwen3.6-plus')}) "
            f"temp={llm_temperature:.2f} thinking={llm_enable_thinking} "
            f"thinking_budget={(llm_thinking_budget if llm_thinking_budget is not None else 'default')} "
            f"preserve_thinking={llm_preserve_thinking} "
            f"io_log={vlm_io_log_enabled} "
            f"web_search={enable_web_search}"
        )

    final_segments: list[DialogueSegment] = []
    debug_texts: list[str] = []
    _log("Running OCR fusion and translation.")
    vlm_history_enabled = bool(tr_cfg.get("history_enabled", False))
    vlm_history_n = int(tr_cfg.get("history_n", 5))
    vlm_workers = int(
        tr_cfg.get(
            "vlm_concurrent_workers",
            tr_cfg.get("concurrent_workers", 32),
        )
    )
    if vlm_history_enabled:
        vlm_workers = 1
    vlm_pool: ThreadPoolExecutor | None = None
    pending_vlm: dict[Future[tuple[str, str, str, dict[str, int]]], tuple[DialogueSegment, str]] = {}
    vlm_prompt_tokens_total = 0
    vlm_completion_tokens_total = 0
    vlm_total_tokens_total = 0
    history_records: list[dict[str, str]] = []
    cache_prefix_entries: list[dict[str, Any]] = []
    if translation_cache_lookup_path.exists():
        try:
            cache_prefix_entries = _extract_title_entries(_load_cache_entries(translation_cache_lookup_path))
            if cache_prefix_entries:
                _log(f"Loaded title entries from cache: {len(cache_prefix_entries)}")
        except Exception:
            cache_prefix_entries = []
    cache_full, cache_time_type = _load_translation_cache(translation_cache_lookup_path)
    cache_hit_count = 0
    if cache_full or cache_time_type:
        _log(
            f"Loaded translation cache: {translation_cache_lookup_path} "
            f"(entries={max(len(cache_full), len(cache_time_type))})"
        )
    if vlm_translator is not None and not args.skip_translation:
        if vlm_history_enabled:
            _log(f"VLM context mode enabled: sequential with history_n={max(0, vlm_history_n)}")
            vlm_pool = None
        else:
            vlm_pool = ThreadPoolExecutor(max_workers=max(1, vlm_workers))
            _log(f"VLM concurrent workers: {max(1, vlm_workers)}")
    marker_counts: dict[int, int] = {}
    for r in raw_segments:
        sid = int(r.get("marker_seg_id", r.get("segment_id", 0)))
        marker_counts[sid] = marker_counts.get(sid, 0) + 1
    marker_seen: dict[int, int] = {}
    name_debug_root = work_dir / "name_by_segment"
    for i, raw in enumerate(raw_segments, start=1):
        if i == 1 or i % 10 == 0 or i == len(raw_segments):
            _log(f"Processing segment {i}/{len(raw_segments)}")
        ridx = int(raw["range_index"])
        marker_seg_id = int(raw.get("marker_seg_id", raw.get("segment_id", i)))
        marker_seen_idx = marker_seen.get(marker_seg_id, 0)
        marker_seen[marker_seg_id] = marker_seen_idx + 1
        marker_count = marker_counts.get(marker_seg_id, 1)
        seg_label = (
            f"seg {marker_seg_id}"
            if marker_count <= 1
            else f"seg {marker_seg_id}-{marker_seen_idx}"
        )
        has_name = bool(raw.get("has_name", True))
        name_dir = work_dir / f"fine_{ridx:03d}_name"
        if debug_mode:
            _export_name_frames_for_segment(
                out_root=name_debug_root,
                seg_label=seg_label,
                name_dir=name_dir,
                frame_start=int(raw["frame_start"]),
                frame_end=int(raw["frame_end"]),
            )

        selected_ts_for_ocr: float | None = None
        vlm_dialog_path: str | Path | None = None
        vlm_speaker_path: str | Path | None = None
        if has_name:
            # Always use fixed frame offset from segment tail as OCR anchor.
            anchor_fid = _pick_marker_anchor_frame(
                raw=raw,
                from_end_frames=marker_anchor_from_end_frames,
            )
            selected_ts_for_ocr = float(raw["start_sec"]) + (float(anchor_fid) / float(raw["scan_fps"]))
        
        review_reasons: list[str] = []
        needs_review = False

        frame_start_abs = int(round(raw["time_start"] * base_fps))
        frame_end_abs = int(round(raw["time_end"] * base_fps))
        stable_ids = [int(round((raw["start_sec"] + (f / raw["scan_fps"])) * base_fps)) for f in raw["sample_frames"]]

        # Stable frame from marker-on selected OCR frame.
        if has_name and selected_ts_for_ocr is not None:
            stable_t = selected_ts_for_ocr
        elif has_name and raw.get("sample_frames"):
            stable_t = float(raw["start_sec"]) + (float(raw["sample_frames"][0]) / float(raw["scan_fps"]))
        elif has_name:
            anchor_fid = _pick_marker_anchor_frame(
                raw=raw,
                from_end_frames=marker_anchor_from_end_frames,
            )
            stable_t = float(raw["start_sec"]) + (float(anchor_fid) / float(raw["scan_fps"]))
            _log(
                f"Warning: segment {i} has_name=true but no selected OCR frame; "
                "fallback to tail-anchor frame for stable frame."
            )
        else:
            stable_t = float(raw.get("stable_time", (raw["time_start"] + raw["time_end"]) / 2))

        dialogue_type = "speaker_dialogue" if has_name else "blank_no_name"
        line_count = 1
        seg = DialogueSegment(
            segment_id=i,
            frame_start=frame_start_abs,
            frame_end=frame_end_abs,
            time_start=float(raw["time_start"]),
            time_end=float(raw["time_end"]),
            speaker="",
            speaker_confidence=0.0,
            text_original="",
            text_ocr_confidence=0.0,
            translation_subtitle="",
            dialogue_type=dialogue_type,
            line_count_detected=line_count,
            stable_frame_ids=stable_ids,
            keyframe_before="",
            keyframe_stable="",
            keyframe_after="",
            needs_review=needs_review,
            review_reason=review_reasons,
        )
        state_txt = "blank" if (not has_name) else "has_name=true"
        debug_text = (
            f"[DEBUG] {seg_label} (id={seg.segment_id}) "
            f"{seg.time_start:.2f}-{seg.time_end:.2f}s "
            f"{state_txt}"
        )
        debug_texts.append(debug_text)
        if args.skip_translation:
            seg.text_original = ""
            seg.speaker = ""
            seg.speaker_confidence = 0.0
            seg.text_ocr_confidence = 0.0
            seg.translation_subtitle = debug_text
        final_segments.append(seg)

        cached_item: dict[str, str] | None = None
        if has_name and vlm_translator is not None:
            k1 = _subtitle_cache_key(
                seg.segment_id,
                seg.time_start,
                seg.time_end,
                seg.dialogue_type,
            )
            k2 = _subtitle_cache_key(
                0,
                seg.time_start,
                seg.time_end,
                seg.dialogue_type,
            )
            cached_item = cache_full.get(k1) or cache_time_type.get(k2)
            if cached_item:
                cached_speaker = str(cached_item.get("speaker", "") or "")
                if cached_speaker:
                    seg.speaker = cached_speaker
                seg.text_original = str(cached_item.get("text_original", "") or "")
                seg.translation_subtitle = str(cached_item.get("translation_subtitle", "") or "")
                cache_hit_count += 1
                _log(
                    f"[VLM] segment {seg.segment_id}: cache hit, skip image load and request"
                )
                if vlm_history_enabled and seg.translation_subtitle:
                    history_records.append(
                        {
                            "time": f"{seg.time_start:.2f}-{seg.time_end:.2f}s",
                            "speaker": seg.speaker,
                            "original": seg.text_original,
                            "translation": seg.translation_subtitle,
                        }
                    )
                    if len(history_records) > max(0, vlm_history_n):
                        history_records = history_records[-max(0, vlm_history_n):]
                continue

        if has_name and vlm_translator is not None:
            ts = (
                float(selected_ts_for_ocr)
                if selected_ts_for_ocr is not None
                else float(raw.get("stable_time", (raw["time_start"] + raw["time_end"]) / 2.0))
            )
            ridx = int(raw["range_index"])
            dialogue_cache_dir = work_dir / f"fine_{ridx:03d}_dialogue"
            name_cache_dir = work_dir / f"fine_{ridx:03d}_name"

            dialogue_image, dialogue_cache_status, dialogue_cache_frame_num = _try_load_cached_roi_frame_with_status(
                ts,
                dialogue_cache_dir,
                float(raw["start_sec"]),
                float(raw["scan_fps"]),
            )
            name_image, name_cache_status, name_cache_frame_num = _try_load_cached_roi_frame_with_status(
                ts,
                name_cache_dir,
                float(raw["start_sec"]),
                float(raw["scan_fps"]),
            )

            if dialogue_image is not None:
                _log(
                    f"[IMAGE_CACHE_DIALOGUE] hit segment={i} frame={dialogue_cache_frame_num:06d} "
                    f"dir={dialogue_cache_dir}"
                )
            else:
                _log(
                    f"[IMAGE_CACHE_DIALOGUE] {dialogue_cache_status} segment={i} "
                    f"frame={dialogue_cache_frame_num:06d} dir={dialogue_cache_dir}"
                )

            if name_image is not None:
                _log(
                    f"[IMAGE_CACHE_NAME] hit segment={i} frame={name_cache_frame_num:06d} "
                    f"dir={name_cache_dir}"
                )
            else:
                _log(
                    f"[IMAGE_CACHE_NAME] {name_cache_status} segment={i} "
                    f"frame={name_cache_frame_num:06d} dir={name_cache_dir}"
                )

            fallback_image: Image.Image | None = None
            if dialogue_image is None or name_image is None:
                _log(f"[IMAGE_CACHE] fallback_ffmpeg segment={i} ts={ts:.3f}")
                frame_bytes = extract_frame_to_memory(
                    ffmpeg_path=ffmpeg_path,
                    video_path=args.video,
                    time_sec=ts,
                )
                fallback_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                if dialogue_image is None:
                    dialogue_image = fallback_image.crop(
                        (dialogue_roi.x1, dialogue_roi.y1, dialogue_roi.x2, dialogue_roi.y2)
                    )
                if name_image is None:
                    name_image = fallback_image.crop(
                        (name_roi.x1, name_roi.y1, name_roi.x2, name_roi.y2)
                    )

            if dialogue_image is None or name_image is None:
                _log(f"[IMAGE_CACHE] image_prepare_failed segment={i}")
                if fallback_image is not None:
                    fallback_image.close()
                continue

            dialog_base64_url = _image_to_base64(dialogue_image)
            speaker_base64_url = _image_to_base64(name_image)
            dialogue_image.close()
            name_image.close()
            if fallback_image is not None:
                fallback_image.close()

            if vlm_dialog_path is None:
                vlm_dialog_path = dialog_base64_url  # type: ignore
            if vlm_speaker_path is None:
                vlm_speaker_path = speaker_base64_url  # type: ignore
        
        # Check if paths are valid (either data URLs or existing file paths)
        def _path_is_valid(p: str | Path | None) -> bool:
            if p is None:
                return False
            s = str(p)
            if s.startswith("data:"):
                return len(s) > 10  # Valid data URL
            return Path(p).exists() if isinstance(p, (str, Path)) else False
        
        if (
            has_name
            and vlm_translator is not None
            and vlm_dialog_path is not None
            and vlm_speaker_path is not None
            and _path_is_valid(vlm_dialog_path)
            and _path_is_valid(vlm_speaker_path)
        ):
            history_items = (
                history_records[-max(0, vlm_history_n):] if vlm_history_enabled else None
            )
            if vlm_pool is None:
                try:
                    speaker_name, original_text, translated, usage = _vlm_translate_task(
                        i,
                        vlm_translator,
                        vlm_speaker_path,
                        vlm_dialog_path,
                        "",
                        history_items=history_items,
                    )
                    vlm_prompt_tokens_total += int(usage.get("prompt_tokens", 0))
                    vlm_completion_tokens_total += int(usage.get("completion_tokens", 0))
                    vlm_total_tokens_total += int(
                        usage.get(
                            "total_tokens",
                            int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0)),
                        )
                    )
                    if speaker_name and speaker_name.strip():
                        seg.speaker = speaker_name.strip()
                    seg.text_original = original_text
                    if translated and translated.strip():
                        seg.translation_subtitle = translated
                    else:
                        seg.translation_subtitle = debug_text
                        seg.review_reason.append("vlm_translation_empty")
                        seg.needs_review = True
                        _log(
                            f"[VLM] segment {seg.segment_id}: empty translation, fallback to debug text"
                        )
                except Exception as exc:
                    seg.translation_subtitle = debug_text
                    seg.review_reason.append(f"vlm_translation_error:{exc.__class__.__name__}")
                    seg.needs_review = True
                    detail = str(exc).strip() or repr(exc)
                    _log(
                        f"[VLM] segment {seg.segment_id}: request failed ({exc.__class__.__name__}), "
                        f"{detail}; fallback to debug text"
                    )
                if seg.translation_subtitle and seg.translation_subtitle != debug_text:
                    history_records.append(
                        {
                            "time": f"{seg.time_start:.2f}-{seg.time_end:.2f}s",
                            "speaker": seg.speaker,
                            "original": seg.text_original,
                            "translation": seg.translation_subtitle,
                        }
                    )
                    if len(history_records) > max(0, vlm_history_n):
                        history_records = history_records[-max(0, vlm_history_n):]
            else:
                fut = vlm_pool.submit(
                    _vlm_translate_task,
                    i,
                    vlm_translator,
                    vlm_speaker_path,
                    vlm_dialog_path,
                    "",
                    history_items,
                )
                pending_vlm[fut] = (seg, debug_text)

    if pending_vlm:
        _log(f"Waiting for VLM tasks: {len(pending_vlm)}")
        for fut in as_completed(list(pending_vlm.keys())):
            seg, dbg = pending_vlm[fut]
            try:
                speaker_name, original_text, translated, usage = fut.result()
                vlm_prompt_tokens_total += int(usage.get("prompt_tokens", 0))
                vlm_completion_tokens_total += int(usage.get("completion_tokens", 0))
                vlm_total_tokens_total += int(
                    usage.get(
                        "total_tokens",
                        int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0)),
                    )
                )
                if speaker_name and speaker_name.strip():
                    seg.speaker = speaker_name.strip()
                seg.text_original = original_text
                if translated and translated.strip():
                    seg.translation_subtitle = translated
                else:
                    seg.translation_subtitle = dbg
                    seg.review_reason.append("vlm_translation_empty")
                    seg.needs_review = True
                    _log(
                        f"[VLM] segment {seg.segment_id}: empty translation, fallback to debug text"
                    )
            except Exception as exc:
                seg.translation_subtitle = dbg
                seg.review_reason.append(f"vlm_translation_error:{exc.__class__.__name__}")
                seg.needs_review = True
                detail = str(exc).strip() or repr(exc)
                _log(
                    f"[VLM] segment {seg.segment_id}: request failed ({exc.__class__.__name__}), "
                    f"{detail}; fallback to debug text"
                )
        _log("All VLM tasks finished.")
    if cache_hit_count > 0:
        _log(f"Translation cache hits: {cache_hit_count}")
    if vlm_translator is not None and not args.skip_translation:
        _log(
            "VLM token usage summary: "
            f"input={vlm_prompt_tokens_total}, "
            f"output={vlm_completion_tokens_total}, "
            f"total={vlm_total_tokens_total}"
        )
    if vlm_pool is not None:
        vlm_pool.shutdown(wait=True)
    name_ocr.close()

    final_json = work_dir / "segments.json"
    _log("Writing segments.json / subtitle files")
    _save_json(final_json, [seg.to_dict() for seg in final_segments])
    _dump_translation_cache(
        run_cache_path,
        video_path=str(args.video),
        config_path=str(Path(args.config).resolve()),
        segments=final_segments,
        debug_texts=debug_texts,
        prefix_entries=cache_prefix_entries,
    )
    _log(f"Saved run cache: {run_cache_path}")
    if args.skip_translation:
        _log(
            "Skip-translation mode: keep run-local cache only; "
            "do not overwrite translation_cache_latest.json"
        )
    else:
        try:
            shutil.copy2(run_cache_path, shared_cache_path)
            _log(f"Published latest cache: {shared_cache_path}")
        except Exception as exc:
            _log(f"Warning: failed to publish latest cache to {shared_cache_path}: {exc}")
    cache_entries_for_subs = _load_cache_entries(run_cache_path)
    if bool(getattr(args, "cache_only", False)):
        _log("Cache-only mode: skip subtitle generation.")
        if not debug_mode:
            _cleanup_intermediate_artifacts(work_dir)
        return 0
    cache_sub_segments, cache_debug_segments = _segments_from_cache_entries(cache_entries_for_subs)
    _log(f"Generating subtitles from cache entries: {len(cache_entries_for_subs)}")
    if args.skip_translation:
        write_srt(cache_debug_segments, output_dir / "subtitles.srt")
        write_ass(
            cache_debug_segments,
            output_dir / "subtitles.ass",
            video_width=video_meta.width,
            video_height=video_meta.height,
            subtitle_location=subtitle_location,
            title_translation_location=title_translation_location,
            title_info_location=title_info_location,
        )
    else:
        write_srt(cache_sub_segments, output_dir / "subtitles.srt")
        write_ass(
            cache_sub_segments,
            output_dir / "subtitles.ass",
            video_width=video_meta.width,
            video_height=video_meta.height,
            subtitle_location=subtitle_location,
            title_translation_location=title_translation_location,
            title_info_location=title_info_location,
        )
        write_srt(cache_debug_segments, output_dir / "subtitles_debug.srt")
        write_ass(
            cache_debug_segments,
            output_dir / "subtitles_debug.ass",
            video_width=video_meta.width,
            video_height=video_meta.height,
            subtitle_location=subtitle_location,
            title_translation_location=title_translation_location,
            title_info_location=title_info_location,
        )
    _backup_subtitles_to_work(output_dir, work_dir)

    render_video_cfg = bool(general_cfg.get("render_video", False))
    if args.render_video or render_video_cfg:
        _log("Rendering hard-sub video with ffmpeg.")
        out_video = work_dir / "video_with_subtitles.mp4"
        cmd = [
            str(ffmpeg_path),
            "-y",
            "-i",
            str(args.video),
            "-vf",
            f"ass={str((output_dir / 'subtitles.ass')).replace('\\', '/')}",
            "-c:a",
            "copy",
            str(out_video),
        ]
        subprocess.run(cmd, check=True)

    review_json = work_dir / "review_export.json"
    review_items = []
    for seg in final_segments:
        review_items.append(
            {
                "segment_id": seg.segment_id,
                "time_start": seg.time_start,
                "time_end": seg.time_end,
                "speaker": seg.speaker,
                "text_original": seg.text_original,
                "translation_subtitle": seg.translation_subtitle,
                "confidence": seg.text_ocr_confidence,
                "keyframe_before": seg.keyframe_before,
                "keyframe_stable": seg.keyframe_stable,
                "keyframe_after": seg.keyframe_after,
                "needs_review": seg.needs_review,
                "review_reason": seg.review_reason,
            }
        )
    _save_json(review_json, review_items)
    _log("Pipeline completed.")
    if not debug_mode:
        _cleanup_intermediate_artifacts(work_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto game video subtitle pipeline (Japanese OCR -> Simplified Chinese)."
    )
    parser.add_argument(
        "--video",
        default="",
        help="Path to input video. Optional when config has video_path.",
    )
    parser.add_argument(
        "--config",
        default="config/example_profile.yaml",
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: outputs/<video filename>.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from cached JSON artifacts.")
    parser.add_argument("--skip-translation", action="store_true", help="Skip DeepSeek translation.")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only generate/update translation cache; do not generate subtitle files.",
    )
    parser.add_argument("--render-video", action="store_true", help="Render hard-sub video via ffmpeg.")
    parser.add_argument(
        "--translation-cache",
        default="",
        help="Path to translation cache json (default: <output-dir>/work/run_xxx/translation_cache.json).",
    )
    parser.add_argument(
        "--subtitles-from-cache",
        action="store_true",
        help="Only generate subtitles from translation cache and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug artifacts (marker_by_segment/name_by_segment and keep fine_* intermediates).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
