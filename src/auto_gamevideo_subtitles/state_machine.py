from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .event_detect import FrameMetric


class DialogueState(str, Enum):
    NO_DIALOGUE = "NO_DIALOGUE"
    TEXT_APPEARING = "TEXT_APPEARING"
    TEXT_STABLE = "TEXT_STABLE"
    TEXT_CLEARING = "TEXT_CLEARING"


@dataclass
class SegmentCandidate:
    start_time: float
    end_time: float
    frame_start: int
    frame_end: int
    has_name: bool = True
    sample_frame_indices: list[int] = field(default_factory=list)


@dataclass
class StateMachineConfig:
    change_threshold: float
    clear_threshold: float
    presence_threshold: float
    name_change_threshold: float
    name_presence_threshold: float
    split_on_name_change: bool
    use_marker_cue: bool
    marker_presence_threshold: float
    marker_min_on_frames: int
    marker_min_off_frames: int
    marker_smooth_window: int
    marker_use_debounce: bool
    stable_frames: int
    clear_frames: int
    min_duration: float


def segment_from_metrics(
    metrics: list[FrameMetric],
    cfg: StateMachineConfig,
) -> list[SegmentCandidate]:
    if not metrics:
        return []
    if cfg.use_marker_cue:
        return _segment_from_marker(metrics, cfg)

    segments: list[SegmentCandidate] = []
    state = DialogueState.NO_DIALOGUE
    stable_count = 0
    clear_count = 0
    start_metric: FrameMetric | None = None
    last_text_metric: FrameMetric | None = None
    sample_frames: list[int] = []

    def close_segment(end_metric: FrameMetric) -> None:
        nonlocal start_metric, sample_frames
        if start_metric is None:
            return
        duration = end_metric.timestamp - start_metric.timestamp
        if duration < cfg.min_duration:
            start_metric = None
            sample_frames = []
            return
        unique_samples = sorted(set(sample_frames))
        segments.append(
            SegmentCandidate(
                start_time=start_metric.timestamp,
                end_time=end_metric.timestamp,
                frame_start=start_metric.frame_index,
                frame_end=end_metric.frame_index,
                sample_frame_indices=unique_samples,
            )
        )
        start_metric = None
        sample_frames = []

    for metric in metrics:
        has_text = metric.presence >= cfg.presence_threshold
        has_change = metric.diff >= cfg.change_threshold
        is_clear = metric.presence <= cfg.clear_threshold
        name_changed = (
            cfg.split_on_name_change
            and metric.name_presence >= cfg.name_presence_threshold
            and metric.name_diff >= cfg.name_change_threshold
        )

        if has_text:
            last_text_metric = metric
            sample_frames.append(metric.frame_index)

        if name_changed and state != DialogueState.NO_DIALOGUE:
            # Speaker/name switched: force sentence boundary.
            if last_text_metric is not None:
                close_segment(last_text_metric)
            if has_text:
                state = DialogueState.TEXT_APPEARING
                start_metric = metric
                stable_count = 0
                clear_count = 0
            else:
                state = DialogueState.NO_DIALOGUE
            continue

        if state == DialogueState.NO_DIALOGUE:
            if has_text:
                state = DialogueState.TEXT_APPEARING
                start_metric = metric
                stable_count = 0
                clear_count = 0
            continue

        if state == DialogueState.TEXT_APPEARING:
            if has_text and not has_change:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= cfg.stable_frames:
                state = DialogueState.TEXT_STABLE
            if is_clear:
                clear_count += 1
                if clear_count >= cfg.clear_frames and last_text_metric is not None:
                    close_segment(last_text_metric)
                    state = DialogueState.NO_DIALOGUE
            else:
                clear_count = 0
            continue

        if state == DialogueState.TEXT_STABLE:
            if is_clear:
                state = DialogueState.TEXT_CLEARING
                clear_count = 1
                continue
            if has_change:
                # New growth after stable generally means next sentence.
                if last_text_metric is not None:
                    close_segment(last_text_metric)
                state = DialogueState.TEXT_APPEARING
                start_metric = metric
                stable_count = 0
                clear_count = 0
            continue

        if state == DialogueState.TEXT_CLEARING:
            if has_text:
                if last_text_metric is not None:
                    close_segment(last_text_metric)
                state = DialogueState.TEXT_APPEARING
                start_metric = metric
                stable_count = 0
                clear_count = 0
                continue
            clear_count += 1
            if clear_count >= cfg.clear_frames:
                if last_text_metric is not None:
                    close_segment(last_text_metric)
                state = DialogueState.NO_DIALOGUE

    if state != DialogueState.NO_DIALOGUE and last_text_metric is not None:
        close_segment(last_text_metric)
    return segments


def _segment_from_marker(
    metrics: list[FrameMetric],
    cfg: StateMachineConfig,
) -> list[SegmentCandidate]:
    """
    Marker-driven segmentation:
    - marker disappears: typing/new sentence starts
    - marker appears: current sentence is complete and should be closed
    """
    segments: list[SegmentCandidate] = []
    scores = [float(m.marker_presence) for m in metrics]
    scores = _moving_average(scores, max(1, cfg.marker_smooth_window))
    marker_on_raw = [s >= cfg.marker_presence_threshold for s in scores]
    if cfg.marker_use_debounce:
        marker_on = _debounce_marker_state(
            marker_on_raw,
            min_on=max(1, cfg.marker_min_on_frames),
            min_off=max(1, cfg.marker_min_off_frames),
        )
        runs = _build_runs(marker_on)
        disappear_indices: list[int] = []
        appear_indices: list[int] = []
        for ridx, (is_on, rs, re) in enumerate(runs):
            if not is_on:
                continue
            on_len = re - rs + 1
            if on_len < max(1, cfg.marker_min_on_frames):
                continue
            # off -> on
            if ridx > 0 and (not runs[ridx - 1][0]):
                appear_indices.append(rs)
            # on -> off
            if ridx + 1 < len(runs) and (not runs[ridx + 1][0]):
                off_len = runs[ridx + 1][2] - runs[ridx + 1][1] + 1
                if off_len >= max(1, cfg.marker_min_off_frames):
                    disappear_indices.append(re + 1)

        min_gap_frames = max(2, cfg.marker_min_on_frames + cfg.marker_min_off_frames)
        disappear_indices = _sparse_indices(disappear_indices, min_gap_frames)
        appear_indices = _sparse_indices(appear_indices, min_gap_frames)
    else:
        marker_on = marker_on_raw
        disappear_indices = []
        appear_indices = []
        for i in range(1, len(marker_on)):
            if marker_on[i - 1] and (not marker_on[i]):
                disappear_indices.append(i)
            if (not marker_on[i - 1]) and marker_on[i]:
                appear_indices.append(i)

    if not disappear_indices:
        return []

    def emit_run(
        run_start_i: int,
        run_end_i: int,
        interval_start_i: int,
        interval_end_i: int,
    ) -> None:
        if run_end_i <= run_start_i:
            return
        start_metric = metrics[run_start_i]
        end_metric = metrics[run_end_i]

        ocr_frame_idx = None
        for ap in appear_indices:
            if run_start_i <= ap <= run_end_i:
                ocr_frame_idx = ap
                break
        if ocr_frame_idx is None:
            marker_on_frames = [
                i
                for i in range(run_start_i, run_end_i + 1)
                if marker_on[i]
            ]
            if not marker_on_frames:
                marker_on_frames = [
                    i
                    for i in range(interval_start_i, interval_end_i + 1)
                    if marker_on[i]
                ]
            if marker_on_frames:
                ocr_frame_idx = marker_on_frames[len(marker_on_frames) // 2]
        if ocr_frame_idx is None:
            for ap in appear_indices:
                if interval_start_i <= ap <= interval_end_i:
                    ocr_frame_idx = ap
                    break
        if ocr_frame_idx is None:
            ocr_frame_idx = (run_start_i + run_end_i) // 2

        segments.append(
            SegmentCandidate(
                start_time=start_metric.timestamp,
                end_time=end_metric.timestamp,
                frame_start=start_metric.frame_index,
                frame_end=end_metric.frame_index,
                has_name=True,
                sample_frame_indices=[ocr_frame_idx],
            )
        )

    # Ignore tail after last marker-disappear (e.g. credits/staff roll).
    interval_starts = [0] + disappear_indices[:-1]
    interval_ends = disappear_indices

    for idx in range(len(interval_starts)):
        start_i = interval_starts[idx]
        end_i = interval_ends[idx]
        if end_i <= start_i:
            continue
        emit_run(start_i, end_i, start_i, end_i)
    return segments


def _debounce_marker_state(raw: list[bool], min_on: int, min_off: int) -> list[bool]:
    if not raw:
        return []
    out = [raw[0]]
    state = raw[0]
    on_streak = 1 if state else 0
    off_streak = 1 if not state else 0
    for i in range(1, len(raw)):
        v = raw[i]
        if v:
            on_streak += 1
            off_streak = 0
        else:
            off_streak += 1
            on_streak = 0

        if not state and on_streak >= min_on:
            state = True
        elif state and off_streak >= min_off:
            state = False
        out.append(state)
    return out


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    out: list[float] = []
    half = window // 2
    n = len(values)
    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        out.append(sum(values[l:r]) / float(r - l))
    return out


def _build_runs(values: list[bool]) -> list[tuple[bool, int, int]]:
    if not values:
        return []
    runs: list[tuple[bool, int, int]] = []
    st = 0
    cur = values[0]
    for i in range(1, len(values)):
        if values[i] != cur:
            runs.append((cur, st, i - 1))
            st = i
            cur = values[i]
    runs.append((cur, st, len(values) - 1))
    return runs


def _sparse_indices(indices: list[int], min_gap: int) -> list[int]:
    if not indices:
        return []
    out = [indices[0]]
    for v in indices[1:]:
        if v - out[-1] >= min_gap:
            out.append(v)
    return out


def _merge_adjacent_blank_segments(
    segments: list[SegmentCandidate],
    max_gap: float,
) -> list[SegmentCandidate]:
    if not segments:
        return []
    merged: list[SegmentCandidate] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if (
            (not prev.has_name)
            and (not seg.has_name)
            and (seg.start_time - prev.end_time) <= max_gap
        ):
            prev.end_time = max(prev.end_time, seg.end_time)
            prev.frame_end = max(prev.frame_end, seg.frame_end)
            continue
        merged.append(seg)
    return merged
