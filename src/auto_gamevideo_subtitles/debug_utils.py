from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .event_detect import FrameMetric, extract_text_mask_stats, load_gray
from .log_utils import _log


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


def _name_mask_debug_label(gray: np.ndarray, ocr_cfg: dict[str, Any]) -> str:
    _mask, st = extract_text_mask_stats(gray, mode="name")
    score = float(st.presence)
    thr_on = float(ocr_cfg.get("name_presence_threshold_on", 0.018))
    thr_off = float(ocr_cfg.get("name_presence_threshold_off", 0.012))
    min_components = int(ocr_cfg.get("name_presence_min_components", 2))
    max_components = int(ocr_cfg.get("name_presence_max_components", 120))
    max_x_min = float(ocr_cfg.get("name_presence_max_x_min_ratio", 0.34))
    max_y_span = float(ocr_cfg.get("name_presence_max_y_span_ratio", 0.62))
    min_aspect = float(ocr_cfg.get("name_presence_min_aspect_ratio", 0.85))
    max_largest = float(ocr_cfg.get("name_presence_max_largest_component_ratio", 0.82))

    reject_reason = ""
    if int(st.component_count) < min_components:
        reject_reason = "few"
    elif int(st.component_count) > max_components:
        reject_reason = "many"
    elif float(st.x_min_ratio) > max_x_min:
        reject_reason = "right"
    elif float(st.y_span_ratio) > max_y_span:
        reject_reason = "tall"
    elif (float(st.x_span_ratio) / max(1e-6, float(st.y_span_ratio))) < min_aspect:
        reject_reason = "nonline"
    elif float(st.largest_component_ratio) > max_largest:
        reject_reason = "blob"

    if score <= thr_off:
        state = "off"
    elif reject_reason:
        state = f"rej-{reject_reason}"
    elif score >= thr_on:
        state = "on"
    else:
        state = "unc"

    return (
        f"m{score:.4f}_c{int(st.component_count):03d}"
        f"_xmin{float(st.x_min_ratio):.2f}_xs{float(st.x_span_ratio):.2f}"
        f"_y{float(st.y_span_ratio):.2f}"
        f"_l{float(st.largest_component_ratio):.2f}_{state}"
    )


def _export_name_frames_for_segment(
    out_root: Path,
    seg_label: str,
    name_dir: Path,
    frame_start: int,
    frame_end: int,
    ocr_cfg: dict[str, Any],
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
            mask_label = _name_mask_debug_label(gray, ocr_cfg)
            dst = seg_dir / f"f{fid:06d}_{mask_label}.png"
        except Exception:
            dst = seg_dir / f"f{fid:06d}.png"
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass
