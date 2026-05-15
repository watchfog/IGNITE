from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .event_detect import MarkerTemplateMatcher, load_gray
from .log_utils import _log
from .review_utils import _attach_review_metadata, _fill_short_false_gaps, _first_true_run_bounds, _merge_review_reasons


def _score_frame(matcher: MarkerTemplateMatcher, path: Path) -> float:
    if path.exists():
        try:
            return float(matcher.score(load_gray(path)))
        except Exception:
            return 0.0
    return 0.0


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
        return j, _score_frame(marker_matcher, marker_frames[j])

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


def _build_refined_subsegment(
    raw: dict[str, Any],
    marker_seg_id: int,
    start_sec: float,
    scan_fps: float,
    marker_thr: float,
    st: int,
    ed: int,
    has_name: bool,
    review_reasons: list[str] | None = None,
) -> dict[str, Any]:
    sample = [] if (not has_name) else [st + (ed - st) // 2]
    reasons = _merge_review_reasons(raw.get("review_reason"), review_reasons)
    out = {
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
    return _attach_review_metadata(out, reasons, force_review=bool(raw.get("needs_review", False)))


def _split_segment_by_marker2(
    raw: dict[str, Any],
    marker2_dir: Path,
    matcher: MarkerTemplateMatcher,
    *,
    threshold: float,
    fast_check_frames: int = 5,
    fast_min_hits: int = 4,
    min_blank_frames: int = 1,
    blank_verify_frames: int = 3,
    blank_verify_min_hits: int = 1,
    smooth_blank_gap_frames: int = 1,
) -> list[dict[str, Any]]:
    frame_start = int(raw["frame_start"])
    frame_end = int(raw["frame_end"])
    scan_fps = float(raw["scan_fps"])
    start_sec = float(raw["start_sec"])
    marker_thr = float(raw.get("marker_presence_threshold_used", 0.0))
    marker_seg_id = int(raw.get("marker_seg_id", raw.get("segment_id", 0)))
    if frame_end < frame_start:
        return [raw]

    score_cache: dict[int, float] = {}
    flag_cache: dict[int, bool] = {}

    def _score(fid: int) -> float:
        if fid in score_cache:
            return score_cache[fid]
        sc = _score_frame(matcher, marker2_dir / f"{fid + 1:06d}.png")
        score_cache[fid] = sc
        return sc

    def _present(fid: int) -> bool:
        if fid not in flag_cache:
            flag_cache[fid] = _score(fid) >= float(threshold)
        return flag_cache[fid]

    def _review_reason(tag: str, start_f: int, end_f: int) -> list[str]:
        if end_f < start_f:
            return []
        vals = [_score(fid) for fid in range(start_f, end_f + 1)]
        if not vals:
            return []
        return [
            f"{tag}:score=[{min(vals):.4f},{max(vals):.4f}] "
            f"threshold={float(threshold):.4f}"
        ]

    def _blank_near_threshold_review_reason(start_f: int, end_f: int) -> list[str]:
        if end_f < start_f:
            return []
        vals = [_score(fid) for fid in range(start_f, end_f + 1)]
        if not vals:
            return []
        max_score = max(vals)
        low = 0.1
        high = float(threshold)
        if low <= max_score < high:
            return [
                f"marker2_blank_near_threshold:score=[{min(vals):.4f},{max_score:.4f}] "
                f"review_band=[{low:.4f},{high:.4f}) threshold={high:.4f}"
            ]
        return []

    def _blank_confirm(start_f: int, end_f: int) -> bool:
        if end_f < start_f:
            return True
        k = max(1, int(blank_verify_frames))
        need = max(1, int(blank_verify_min_hits))
        step = 3
        head = [start_f + i * step for i in range(k) if start_f + i * step <= end_f]
        tail = [end_f - i * step for i in range(k) if end_f - i * step >= start_f]
        sample_frames = sorted(set(head + tail))
        hits = sum(1 for fid in sample_frames if _present(fid))
        _log(
            f"[split_by_marker2] seg_id={marker_seg_id} blank_confirm "
            f"range=[{start_f},{end_f}] samples={sample_frames} hits={hits} "
            f"need_lt={need} threshold={float(threshold):.4f}"
        )
        return hits < need

    probe_n = max(1, int(fast_check_frames))
    probe_offset = max(0, int(round(0.16 * scan_fps)))
    probe_start = min(frame_end, frame_start + probe_offset)
    probe_end = min(frame_end, probe_start + probe_n - 1)
    probe_hits = sum(1 for fid in range(probe_start, probe_end + 1) if _present(fid))
    if probe_hits >= max(1, int(fast_min_hits)):
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

    anchor = frame_end
    samples = raw.get("sample_frames") or []
    if samples:
        try:
            anchor = int(samples[0])
        except Exception:
            anchor = frame_end
    anchor = max(frame_start, min(frame_end, anchor))
    fids = list(range(frame_start, anchor + 1))
    flags = [_present(fid) for fid in fids]
    _fill_short_false_gaps(flags, max_gap=max(0, int(smooth_blank_gap_frames)))
    run_bounds = _first_true_run_bounds(flags, min_run=min(2, max(1, len(flags))))
    if run_bounds is None:
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
                review_reasons=_review_reason("marker2_boundary_unconfirmed", frame_start, anchor),
            )
        ]

    run_start, _run_end = run_bounds
    dialog_start = frame_start + run_start
    min_blank = max(1, int(min_blank_frames))
    if (dialog_start - frame_start) < min_blank:
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
                review_reasons=_review_reason("marker2_blank_confirm_failed", frame_start, dialog_start - 1),
            )
        ]

    _log(
        f"[split_by_marker2] seg_id={marker_seg_id} split "
        f"dialog_start={dialog_start} threshold={float(threshold):.4f}"
    )
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
            review_reasons=_blank_near_threshold_review_reason(frame_start, dialog_start - 1),
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
