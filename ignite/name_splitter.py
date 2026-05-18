from __future__ import annotations

from pathlib import Path
from typing import Any

from .log_utils import _log
from .marker_ops import _build_refined_subsegment
from .name_ocr_runner import NameOcrRunner
from .review_utils import (
    _attach_review_metadata,
    _fill_short_false_gaps,
    _first_true_run_bounds,
    _merge_review_reasons,
)


def _head_probe_hits_ocr(
    name_dir: Path,
    frame_start: int,
    frame_end: int,
    scan_fps: float,
    fast_check_frames: int,
    name_ocr: NameOcrRunner,
    name_cache: dict[int, bool],
    use_ocr: bool = True,
) -> int:
    probe_n = max(1, int(fast_check_frames))
    probe_offset = max(0, int(round(0.16 * scan_fps)))
    probe_start = min(frame_end, frame_start + probe_offset)
    probe_end = min(frame_end, probe_start + probe_n - 1)
    probe_fids = list(range(probe_start, probe_end + 1))
    probe_paths = [name_dir / f"{fid + 1:06d}.png" for fid in probe_fids]
    if use_ocr:
        probe_flags = name_ocr.has_text_batch_ocr(probe_paths)
    else:
        probe_flags = [name_ocr.has_text_mask(p) for p in probe_paths]
    for fid, flag in zip(probe_fids, probe_flags):
        name_cache[fid] = bool(flag)
    return sum(1 for x in probe_flags if x)


def _split_segment_by_name_ocr(
    raw: dict[str, Any],
    name_dir: Path,
    name_ocr: NameOcrRunner,
    use_ocr: bool = True,
    fast_check_frames: int = 5,
    fast_min_hits: int = 4,
    coarse_step_frames: int = 6,
    smooth_blank_gap_frames: int = 2,
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
    name_mask_uncertain_cache: dict[int, bool] = {}
    name_mask_presence_cache: dict[int, float] = {}
    name_mask_meta_cache: dict[int, dict[str, Any]] = {}
    name_ocr_cache: dict[int, bool] = {}
    name_cache: dict[int, bool] = {}

    def _has_name_mask(fid: int) -> bool:
        if fid in name_mask_cache:
            return name_mask_cache[fid]
        p = name_dir / f"{fid + 1:06d}.png"
        v, uncertain, presence, meta = name_ocr.has_text_mask_detail_meta(p)
        name_mask_cache[fid] = bool(v)
        name_mask_uncertain_cache[fid] = bool(uncertain)
        name_mask_presence_cache[fid] = float(presence)
        name_mask_meta_cache[fid] = dict(meta)
        return bool(v)

    def _mask_uncertain_review_reasons(
        start_f: int,
        end_f: int,
        *,
        tag: str = "name_mask_uncertain",
    ) -> list[str]:
        if end_f < start_f:
            return []
        fids_check = list(range(start_f, end_f + 1))
        for fid in fids_check:
            _has_name_mask(fid)
        uncertain = [fid for fid in fids_check if name_mask_uncertain_cache.get(fid, False)]
        if not uncertain:
            return []
        scores = [name_mask_presence_cache.get(fid, 0.0) for fid in uncertain]
        reject_counts: dict[str, int] = {}
        for fid in uncertain:
            reason = str(name_mask_meta_cache.get(fid, {}).get("reject_reason", "") or "")
            if reason:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
        reject_text = ""
        if reject_counts:
            reject_text = " reject=" + ",".join(
                f"{k}:{v}" for k, v in sorted(reject_counts.items())
            )
        return [
            f"{tag}:"
            f"count={len(uncertain)} "
            f"range=[{uncertain[0]},{uncertain[-1]}] "
            f"score=[{min(scores):.4f},{max(scores):.4f}]"
            f"{reject_text}"
        ]

    def _has_name_ocr(fid: int) -> bool:
        if fid in name_ocr_cache:
            return name_ocr_cache[fid]
        p = name_dir / f"{fid + 1:06d}.png"
        v = name_ocr.has_text_ocr(p)
        name_ocr_cache[fid] = bool(v)
        return bool(v)

    def _has_name_confirm(fid: int) -> bool:
        return _has_name_ocr(fid) if use_ocr else _has_name_mask(fid)

    def _blank_confirm(start_f: int, end_f: int) -> bool:
        # Returns True only if this run is confirmed blank by OCR/mask sampling.
        if end_f < start_f:
            return True
        k = max(1, int(blank_verify_frames))
        need = max(1, int(blank_verify_min_hits))
        step = 3

        head = [start_f + i * step for i in range(k) if start_f + i * step <= end_f]
        tail = [end_f - i * step for i in range(k) if end_f - i * step >= start_f]

        sample_frames = sorted(set(head + tail))

        verify_paths = [name_dir / f"{f + 1:06d}.png" for f in sample_frames]
        if use_ocr:
            vflags = name_ocr.has_text_batch_ocr(verify_paths)
        else:
            vflags = [name_ocr.has_text_mask(p) for p in verify_paths]

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
            if _has_name_confirm(fid):
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
            has_name = _has_name_confirm(fid)
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
                v = _has_name_confirm(ff)
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
            coarse_hit = _has_name_confirm(fid)
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
    probe_n = max(1, int(fast_check_frames))
    probe_offset = max(0, int(round(0.16 * scan_fps)))
    probe_start = min(frame_end, frame_start + probe_offset)
    probe_end = min(frame_end, probe_start + probe_n - 1)
    if use_ocr:
        probe_hits = _head_probe_hits_ocr(
            name_dir=name_dir,
            frame_start=frame_start,
            frame_end=frame_end,
            scan_fps=scan_fps,
            fast_check_frames=fast_check_frames,
            name_ocr=name_ocr,
            name_cache=name_cache,
            use_ocr=True,
        )
    else:
        probe_fids = list(range(probe_start, probe_end + 1))
        probe_flags = [_has_name_mask(fid) for fid in probe_fids]
        for fid, flag in zip(probe_fids, probe_flags):
            name_cache[fid] = bool(flag)
        probe_hits = sum(1 for x in probe_flags if x)
    for fid, flag in name_cache.items():
        name_ocr_cache[int(fid)] = bool(flag)
    if probe_hits >= probe_hits_need:
        probe_review_reasons = _mask_uncertain_review_reasons(
            probe_start,
            probe_end,
            tag="name_mask_probe_uncertain",
        )
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
                review_reasons=probe_review_reasons,
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
    mask_review_reasons: list[str] = []

    # A marker segment can only be "blank -> subtitle" or "subtitle".
    # Later False/True toggles are detector noise, so use the first durable
    # name run as the single boundary candidate.
    mask_candidate_min_run = min(2, max(1, len(flags)))
    run_bounds = _first_true_run_bounds(flags, min_run=mask_candidate_min_run)
    if run_bounds is None:
        # Mask missed everything: force backward confirmation instead of marking
        # whole marker segment as blank.
        dialog_start = _find_dialog_start_by_backward_ocr(frame_start, anchor)
        if dialog_start is None:
            miss_reason = (
                "mask_miss_and_backward_ocr_failed"
                if use_ocr
                else "mask_miss_and_backward_mask_failed"
            )
            _log(f"{log_prefix} abandon_blank reason={miss_reason} anchor={anchor}")
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
                    review_reasons=(
                        _merge_review_reasons(
                            _mask_uncertain_review_reasons(
                                frame_start,
                                anchor,
                                tag="name_mask_search_uncertain",
                            ),
                            [f"name_split_boundary_unconfirmed:{miss_reason}"],
                        )
                    ),
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
                    review_reasons=(
                        _merge_review_reasons(
                            ["name_split_blank_confirm_failed"],
                        )
                    ),
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

    # Candidate dialogue starts at the first durable True-run.
    run_start, run_end = run_bounds
    candidate_start = frame_start + run_start
    candidate_end = frame_start + run_end
    mask_run_len = run_end - run_start + 1
    mask_tail_gap = max(0, anchor - candidate_end)
    left_mask_hits = sum(1 for x in flags[:run_start] if x)
    candidate_presence_scores = [
        name_mask_presence_cache.get(fid, 0.0)
        for fid in range(candidate_start, candidate_end + 1)
    ]
    candidate_presence_avg = (
        sum(candidate_presence_scores) / max(1, len(candidate_presence_scores))
        if candidate_presence_scores
        else 0.0
    )
    candidate_presence_min = min(candidate_presence_scores) if candidate_presence_scores else 0.0
    pre_window_len = max(1, min(mask_run_len, 4))
    pre_start = max(frame_start, candidate_start - pre_window_len)
    pre_scores = [
        name_mask_presence_cache.get(fid, 0.0)
        for fid in range(pre_start, candidate_start)
    ]
    pre_presence_avg = (
        sum(pre_scores) / max(1, len(pre_scores))
        if pre_scores
        else 0.0
    )
    candidate_presence_rise = candidate_presence_avg - pre_presence_avg
    boundary_margin_frames = max(1, int(smooth_blank_gap_frames) + 1)
    mask_review_reasons = _mask_uncertain_review_reasons(
        max(frame_start, candidate_start - boundary_margin_frames),
        min(anchor, candidate_end + boundary_margin_frames),
        tag="name_mask_boundary_uncertain",
    )
    mask_split_presence_threshold = float(
        name_ocr._ocr_cfg.get(
            "name_presence_split_threshold",
            float(name_ocr._mask_thr_on),
        )
    )
    mask_split_min_run = int(name_ocr._ocr_cfg.get("name_presence_split_min_run_frames", 2))
    mask_min_rise_delta = float(name_ocr._ocr_cfg.get("name_presence_min_rise_delta", 0.002))
    mask_candidate_low_reasons = []
    if (not use_ocr) and (
        mask_run_len < max(1, mask_split_min_run)
        or candidate_presence_avg < mask_split_presence_threshold
        or (
            candidate_start > frame_start
            and candidate_presence_rise < mask_min_rise_delta
        )
    ):
        mask_candidate_low_reasons = [
            "name_mask_low_confidence_candidate:"
            f"run_len={mask_run_len} "
            f"avg={candidate_presence_avg:.4f} "
            f"pre_avg={pre_presence_avg:.4f} "
            f"rise={candidate_presence_rise:.4f} "
            f"min={candidate_presence_min:.4f} "
            f"need_avg={mask_split_presence_threshold:.4f} "
            f"need_run={max(1, mask_split_min_run)} "
            f"need_rise={mask_min_rise_delta:.4f}"
        ]
        _log(
            f"{log_prefix} abandon_blank reason=mask_candidate_low_confidence "
            f"candidate_start={candidate_start} mask_run=[{candidate_start},{candidate_end}] "
            f"mask_run_len={mask_run_len} avg={candidate_presence_avg:.4f} "
            f"pre_avg={pre_presence_avg:.4f} rise={candidate_presence_rise:.4f} "
            f"min={candidate_presence_min:.4f} need_avg={mask_split_presence_threshold:.4f} "
            f"need_run={max(1, mask_split_min_run)} need_rise={mask_min_rise_delta:.4f} anchor={anchor}"
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
                review_reasons=_merge_review_reasons(
                    mask_review_reasons,
                    mask_candidate_low_reasons,
                ),
            )
        ]
    min_blank = max(1, int(min_blank_frames))
    if (candidate_start - frame_start) < min_blank:
        _log(
            f"{log_prefix} abandon_blank reason=leading_blank_too_short "
            f"blank_len={candidate_start - frame_start} min_blank={min_blank} "
            f"candidate_start={candidate_start} anchor={anchor}"
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
                review_reasons=mask_review_reasons,
            )
        ]

    boundary_review_reasons = list(mask_review_reasons)
    if not use_ocr:
        dialog_start = candidate_start
        _log(
            f"{log_prefix} mask_boundary candidate_start={candidate_start} "
            f"mask_run=[{candidate_start},{candidate_end}] "
            f"mask_run_len={mask_run_len} tail_gap={mask_tail_gap} "
            f"left_mask_hits={left_mask_hits} anchor={anchor}"
        )
    else:
        # OCR confirmation: coarse+fine forward search.
        # Search from candidate_start-lookback up to anchor, and do not require mask=True.
        confirm_start = max(frame_start, candidate_start - max(0, int(confirm_lookback_frames)))
        dialog_start = _find_dialog_start_by_forward_ocr(confirm_start, anchor)
        if dialog_start is None:
            # Mask found candidate but OCR didn't confirm: force backward OCR search.
            dialog_start = _find_dialog_start_by_backward_ocr(frame_start, anchor)
            if dialog_start is None:
                allow_mask_fallback = bool(name_ocr._ocr_cfg.get("name_ocr_allow_mask_fallback", False))
                mask_fallback_min_run = max(2, min(3, probe_hits_need))
                mask_tail_tolerance = max(0, int(smooth_blank_gap_frames))
                if (
                    allow_mask_fallback
                    and mask_run_len >= mask_fallback_min_run
                    and mask_tail_gap <= mask_tail_tolerance
                ):
                    dialog_start = candidate_start
                    boundary_review_reasons = _merge_review_reasons(
                        boundary_review_reasons,
                        ["name_ocr_boundary_fallback:mask_candidate_after_ocr_failed"],
                    )
                    _log(
                        f"{log_prefix} fallback_blank reason=mask_candidate_after_ocr_failed "
                        f"candidate_start={candidate_start} mask_run=[{candidate_start},{candidate_end}] "
                        f"mask_run_len={mask_run_len} tail_gap={mask_tail_gap} "
                        f"left_mask_hits={left_mask_hits} anchor={anchor}"
                    )
                else:
                    _log(
                        f"{log_prefix} abandon_blank reason=forward_backward_ocr_both_failed "
                        f"candidate_start={candidate_start} confirm_start={confirm_start} anchor={anchor} "
                        f"mask_run=[{candidate_start},{candidate_end}] mask_run_len={mask_run_len} "
                        f"tail_gap={mask_tail_gap} left_mask_hits={left_mask_hits}"
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
                            review_reasons=_merge_review_reasons(
                                boundary_review_reasons,
                                ["name_ocr_boundary_unconfirmed:forward_backward_ocr_both_failed"],
                            ),
                        )
                    ]

    # If head blank too short, merge into dialogue to avoid jitter.
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
                review_reasons=boundary_review_reasons,
            )
        ]

    # Blank leading part must be confirmed blank; otherwise merge.
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
                review_reasons=(
                    _merge_review_reasons(
                        boundary_review_reasons,
                        ["name_split_blank_confirm_failed"],
                    )
                ),
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
            review_reasons=boundary_review_reasons,
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
            review_reasons=boundary_review_reasons,
        ),
    ]


def _normalize_name_subsegments_per_marker(
    segs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not segs:
        return []
    out: list[dict[str, Any]] = []

    def _finalize(payload: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
        return _attach_review_metadata(
            payload,
            _merge_review_reasons(*[p.get("review_reason") for p in parts]),
        )

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
                _finalize(
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
                    },
                    grp,
                )
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
                    _finalize(
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
                        },
                        grp[:first_true_idx],
                    )
                )
        dfs = int(grp[first_true_idx]["frame_start"])
        dfe = int(grp[-1]["frame_end"])
        dmid = dfs + ((dfe - dfs) // 2)
        out.append(
            _finalize(
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
                },
                grp[first_true_idx:],
            )
        )
    return out
