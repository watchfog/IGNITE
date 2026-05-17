from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
import io
import shutil
import subprocess
import threading
import time
import yaml
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cache_manager import (
    _dump_translation_cache,
    _extract_title_entries,
    _find_latest_translation_cache,
    _load_cache_entries,
    _load_json,
    _load_translation_cache,
    _save_json,
    _segments_from_cache_entries,
    _subtitle_cache_key,
)
from .config import load_config
from .debug_utils import (
    _export_marker_frames_for_segment,
    _export_name_frames_for_segment,
)
from .event_detect import (
    FrameMetric,
    MarkerTemplateMatcher,
    extract_text_features,
    frame_text_change_score,
    load_gray,
)
from .ffmpeg_utils import (
    extract_sequence_dialogue_name_marker,
    ffprobe_video,
    extract_frame_to_memory,
)
from .image_utils import (
    _assert_crop_size,
    _image_to_base64,
    _try_load_cached_roi_frame_with_status,
)
from .log_utils import _log, set_log_file
from .marker_ops import (
    _background_score_marker_and_prune_dialogue_cache,
    _final_prune_dialogue_cache_by_scores,
    _pick_marker_anchor_frame,
    _prune_dialogue_cache_to_anchor_frames,
    _split_segment_by_marker2,
)
from .datatypes import DialogueSegment, Roi
from .name_ocr_runner import NameOcrRunner
from .name_splitter import (
    _normalize_name_subsegments_per_marker,
    _split_segment_by_name_ocr,
)
from .ocr_engines import build_ocr_engine
from .review_utils import (
    _coerce_review_reasons,
    _mark_kanji_overlap_for_review,
    _merge_review_reasons,
)
from .state_machine import StateMachineConfig, segment_from_metrics
from .subtitle_export import write_ass, write_srt
from .translation_runtime import (
    BailianVlmTranslator,
    load_api_key,
    translate_segment_with_retry,
)


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
        "fine_*_marker2",
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
        marker_scores, stats = marker_matcher.score_batch(
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


def _pick_sample_indices(frame_indices: list[int], max_candidates: int) -> list[int]:
    if not frame_indices:
        return []
    unique = sorted(set(frame_indices))
    if len(unique) <= max_candidates:
        return unique
    picks = np.linspace(0, len(unique) - 1, max_candidates, dtype=int)
    return [unique[i] for i in picks]


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


def _resolve_run_prefix(args: argparse.Namespace) -> str:
    if bool(getattr(args, "subtitles_from_cache", False)):
        return "run_subtitle"
    if bool(getattr(args, "cache_only", False)):
        return "run_cache"
    return "run_full"


def run_pipeline(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    project_root = Path(__file__).resolve().parents[1]
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
    log_file = work_dir / "run.log"
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass
    set_log_file(log_file)
    vlm_io_log_enabled = bool(tr_cfg.get("io_log_enabled", False))
    if vlm_io_log_enabled and vlm_io_log_path.exists():
        try:
            vlm_io_log_path.unlink()
        except Exception:
            pass
    _log(f"Log file: {log_file}")
    if vlm_io_log_enabled:
        _log(f"VLM IO log file: {vlm_io_log_path} (cleared previous if existed)")
    _log("Pipeline started.")
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

    subtitle_style: dict[str, Any] | None = None
    style_path = project_root / "config" / "subtitle_style.yaml"
    if style_path.exists():
        try:
            subtitle_style = yaml.safe_load(style_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    per_style = cfg.get("subtitle_style")
    if isinstance(per_style, dict):
        if subtitle_style is None:
            subtitle_style = {}
        subtitle_style.update(per_style)

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
        dialogue_roi = _roi_from_cfg(cfg, "dialogue_roi")
        title_ocr_roi = (
            _roi_from_cfg(cfg, "title_ocr_roi")
            if isinstance(cfg.get("roi", {}).get("title_ocr_roi"), list)
            else _roi_from_cfg(cfg, "dialogue_roi")
        )
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
            style=subtitle_style,
            dialogue_height=dialogue_roi.height,
            title_height=title_ocr_roi.height,
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
            style=subtitle_style,
            dialogue_height=dialogue_roi.height,
            title_height=title_ocr_roi.height,
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
    title_ocr_roi = (
        _roi_from_cfg(cfg, "title_ocr_roi")
        if isinstance(cfg.get("roi", {}).get("title_ocr_roi"), list)
        else _roi_from_cfg(cfg, "dialogue_roi")
    )
    marker_roi = _roi_from_cfg(cfg, "marker_roi")
    marker2_roi = (
        _roi_from_cfg(cfg, "marker_2_roi")
        if isinstance(cfg.get("roi", {}).get("marker_2_roi"), list)
        else None
    )
    marker2_match_roi = (
        _roi_from_cfg(cfg, "marker_2_match_roi")
        if isinstance(cfg.get("roi", {}).get("marker_2_match_roi"), list)
        else marker2_roi
    )
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
    marker2_matcher = None
    marker_cfg = cfg.get("marker", {})
    marker2_cfg = cfg.get("marker_2", {})
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

    marker2_threshold = float(marker2_cfg.get("force_threshold", marker2_cfg.get("presence_threshold", 0.1)))
    if bool(marker2_cfg.get("use_template", True)):
        marker2_templates = _collect_marker_templates(marker2_cfg)
        if marker2_templates:
            v_shift = int(marker2_cfg.get("vertical_shift_px", 0))
            h_shift = int(marker2_cfg.get("horizontal_shift_px", 0))
            if marker2_match_roi is not None and marker2_roi is not None:
                m2_coords = [marker2_match_roi.x1, marker2_match_roi.y1, marker2_match_roi.x2, marker2_match_roi.y2]
                roi_coords = [marker2_roi.x1, marker2_roi.y1, marker2_roi.x2, marker2_roi.y2]
                if m2_coords != roi_coords:
                    v_shift = 0
                    h_shift = 0
            marker2_matcher = MarkerTemplateMatcher(
                marker2_templates,
                center_width=(
                    int(marker2_cfg.get("template_center_width"))
                    if marker2_cfg.get("template_center_width") is not None
                    else None
                ),
                vertical_shift_px=v_shift,
                vertical_shift_step=int(marker2_cfg.get("vertical_shift_step", 1)),
                horizontal_shift_px=h_shift,
                horizontal_shift_step=int(marker2_cfg.get("horizontal_shift_step", 1)),
                shift_mode=str(marker2_cfg.get("shift_mode", "vertical")),
            )
            _log(
                f"Marker2 template enabled: {marker2_matcher.template_count} templates "
                f"(first={marker2_templates[0]}) threshold={marker2_threshold:.3f}"
            )
        else:
            _log("Marker2 templates not found; marker2 split path unavailable.")

    ranges = [(0.0, video_meta.duration)]
    _log("Marker mode: scan full video once.")

    raw_seg_json = work_dir / "segments_raw.json"
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
        marker_score_ranges: list[dict[str, Any]] = []
        for ridx, (start_sec, end_sec) in enumerate(ranges):
            _log(f"Fine-scan window: {start_sec:.2f}s -> {end_sec:.2f}s")
            duration = max(0.01, end_sec - start_sec)
            frame_stride = max(1, int(cfg["video"].get("frame_stride", 2)))
            scan_fps = base_fps / float(frame_stride)
            name_dir = work_dir / f"fine_{ridx:03d}_name"
            marker_dir = work_dir / f"fine_{ridx:03d}_marker"
            marker2_dir = work_dir / f"fine_{ridx:03d}_marker2"
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

            dialogue_paths, name_paths, marker_paths, marker2_paths = extract_sequence_dialogue_name_marker(
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
                marker2_output_dir=marker2_dir if marker2_matcher is not None else None,
                marker2_crop_filter=marker2_match_roi.as_crop_filter() if marker2_matcher is not None and marker2_match_roi else None,
            )

            if marker_worker is not None:
                worker_stop_event.set()
                worker_end_holder["end"] = len(marker_paths)
                marker_worker.join()
                _log("Marker prune worker finished.")

            _assert_crop_size(name_paths, name_roi, "fine window name")
            if marker2_paths and marker2_match_roi:
                _assert_crop_size(marker2_paths, marker2_match_roi, "fine window marker2")
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
        if marker_matcher is not None and marker_scores_cached is not None:
            marker_score_ranges.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "scan_fps": scan_fps,
                "threshold": marker_thr_eff,
                "scores": [float(x) for x in marker_scores_cached],
            })
        _save_json(raw_seg_json, raw_segments)
        _log(f"Segmentation done. Raw segments: {len(raw_segments)}")

    if marker_score_ranges:
        _save_json(work_dir / "marker_score_cache.json", {"ranges": marker_score_ranges})

    name_split_mode = str(getattr(args, "name_split_mode", "config") or "config").strip().lower()
    split_name_enabled = bool(cfg["state_machine"].get("split_on_name_ocr", True))
    if name_split_mode in {"mask", "ocr"}:
        split_name_enabled = True
    name_split_use_ocr = bool(cfg["state_machine"].get("name_split_use_ocr", True))
    if name_split_mode == "mask":
        name_split_use_ocr = False
    elif name_split_mode == "ocr":
        name_split_use_ocr = True

    name_ocr: NameOcrRunner | None = None
    if split_name_enabled and (not name_split_use_ocr) and marker2_matcher is None:
        raise RuntimeError(
            "OCR disabled, but marker_2.template_paths are not configured. "
            "Set marker_2 templates in profile editor or enable OCR."
        )
    if split_name_enabled and (not name_split_use_ocr) and marker2_matcher is not None:
        _log("Refining segments by marker2 template presence (OCR disabled).")
        refined = []
        fast_frames = int(cfg["state_machine"].get("name_fast_check_frames", 5))
        fast_hits = int(cfg["state_machine"].get("name_fast_min_hits", 4))
        smooth_gap = int(cfg["state_machine"].get("name_smooth_blank_gap_frames", 2))
        general_cfg = cfg.get("general", {})
        min_blank = int(
            general_cfg.get(
                "blank_ignore_under_frames",
                cfg["state_machine"].get("name_min_blank_frames", 2),
            )
        )
        for raw in raw_segments:
            ridx = int(raw["range_index"])
            refined.extend(
                _split_segment_by_marker2(
                    raw,
                    work_dir / f"fine_{ridx:03d}_marker2",
                    marker2_matcher,
                    threshold=marker2_threshold,
                    fast_check_frames=fast_frames,
                    fast_min_hits=fast_hits,
                    smooth_blank_gap_frames=smooth_gap,
                    min_blank_frames=min_blank,
                )
            )
        refined = _normalize_name_subsegments_per_marker(refined)
        _log(f"Marker2 refine: {len(raw_segments)} -> {len(refined)} segments.")
        raw_segments = refined
        _save_json(raw_seg_json, raw_segments)
    elif split_name_enabled:
        ocr_engine = None
        if name_split_use_ocr:
            _log("Initializing OCR engine (name split verification).")
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
        else:
            _log("Name split mode: mask-only (OCR disabled).")
        name_ocr_workers = int(cfg["ocr"].get("name_ocr_workers", 1)) if name_split_use_ocr else 1
        name_ocr = NameOcrRunner(cfg["ocr"], ocr_engine, workers=name_ocr_workers)
        _log(f"Name split workers: {name_ocr.workers}; use_ocr={name_split_use_ocr}")

        _log("Refining segments by name-region presence.")
        refined: list[dict[str, Any]] = []
        fast_frames = int(cfg["state_machine"].get("name_fast_check_frames", 5))
        fast_hits = int(cfg["state_machine"].get("name_fast_min_hits", 4))
        coarse_step = int(cfg["state_machine"].get("name_coarse_step_frames", 6))
        smooth_gap = int(cfg["state_machine"].get("name_smooth_blank_gap_frames", 2))
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
            f"Name split fast-check: first {fast_frames} frames, hits>={fast_hits}; "
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
                    f"Name presence refine progress: {idx}/{refine_total} "
                    f"({(idx / max(1, refine_total)) * 100:.1f}%, elapsed={elapsed:.1f}s)"
                )
            ridx = int(raw["range_index"])
            name_dir = work_dir / f"fine_{ridx:03d}_name"
            refined.extend(
                _split_segment_by_name_ocr(
                    raw,
                    name_dir,
                    name_ocr,
                    use_ocr=name_split_use_ocr,
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
        mask_total = int(st.get("mask_total", 0) or 0)
        mask_avg = (
            float(st.get("mask_score_sum", 0.0) or 0.0) / max(1, mask_total)
            if mask_total
            else 0.0
        )
        mask_min_score = st.get("mask_score_min")
        mask_max_score = st.get("mask_score_max")
        _log(
            "Name presence stats: "
            f"total={st.get('total', 0)} "
            f"mask_total={mask_total} "
            f"mask_only={st.get('mask_only', 0)} "
            f"mask_on={st.get('mask_on', 0)} "
            f"mask_off={st.get('mask_off', 0)} "
            f"mask_uncertain={st.get('mask_uncertain', 0)} "
            f"mask_rejected_shape={st.get('mask_rejected_shape', 0)} "
            f"mask_score=min:{float(mask_min_score or 0.0):.4f}/"
            f"avg:{mask_avg:.4f}/max:{float(mask_max_score or 0.0):.4f} "
            f"ocr_fallback={st.get('ocr_fallback', 0)} "
            f"ocr_verify={st.get('ocr_verify', 0)}"
        )
        _log(f"Name presence refine: {len(raw_segments)} -> {len(refined)} segments.")
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
    vlm_extra_requirements = str(game_cfg.get("extra_requirements", "")).strip()
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
                    ocr_cfg=cfg["ocr"],
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
        
        review_reasons = _coerce_review_reasons(raw.get("review_reason"))
        needs_review = bool(raw.get("needs_review", False)) or bool(review_reasons)

        frame_start_abs = int(round(raw["time_start"] * base_fps))
        frame_end_abs = int(round(raw["time_end"] * base_fps))
        stable_ids = [int(round((raw["start_sec"] + (f / raw["scan_fps"])) * base_fps)) for f in raw["sample_frames"]]

        if not has_name:
            selected_ts_for_ocr = None

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

        cached_item: dict[str, Any] | None = None
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
            if cached_item is None:
                cache_fuzz_frames = int(general_cfg.get("cache_fuzz_frames", 6))
                if cache_fuzz_frames > 0:
                    seg_scan_fps = float(raw.get("scan_fps", base_fps))
                    fuzz_ms = int(round(cache_fuzz_frames / max(0.1, seg_scan_fps) * 1000))
                    step_ms = max(1, fuzz_ms // 4)
                    for offset_ms in range(-fuzz_ms, fuzz_ms + 1, step_ms):
                        if offset_ms == 0:
                            continue
                        offset_s = offset_ms / 1000.0
                        fk1 = _subtitle_cache_key(
                            seg.segment_id,
                            seg.time_start + offset_s,
                            seg.time_end + offset_s,
                            seg.dialogue_type,
                        )
                        fk2 = _subtitle_cache_key(
                            0,
                            seg.time_start + offset_s,
                            seg.time_end + offset_s,
                            seg.dialogue_type,
                        )
                        cached_item = cache_full.get(fk1) or cache_time_type.get(fk2)
                        if cached_item:
                            break
            if cached_item:
                cached_speaker = str(cached_item.get("speaker", "") or "")
                if cached_speaker:
                    seg.speaker = cached_speaker
                seg.text_original = str(cached_item.get("text_original", "") or "")
                seg.translation_subtitle = str(cached_item.get("translation_subtitle", "") or "")
                cached_reasons = _coerce_review_reasons(cached_item.get("review_reason"))
                if bool(cached_item.get("needs_review", False)) or cached_reasons:
                    seg.needs_review = True
                    seg.review_reason = _merge_review_reasons(seg.review_reason, cached_reasons)
                _mark_kanji_overlap_for_review(seg)
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
                seg.translation_subtitle = debug_text
                seg.review_reason = _merge_review_reasons(
                    seg.review_reason,
                    ["image_prepare_failed"],
                )
                seg.needs_review = True
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
                    speaker_name, original_text, translated, usage = translate_segment_with_retry(
                        i,
                        vlm_translator,
                        vlm_speaker_path,
                        vlm_dialog_path,
                        "",
                        history_items=history_items,
                        extra_requirements=vlm_extra_requirements,
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
                        _mark_kanji_overlap_for_review(seg)
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
                    translate_segment_with_retry,
                    i,
                    vlm_translator,
                    vlm_speaker_path,
                    vlm_dialog_path,
                    "",
                    history_items,
                    extra_requirements=vlm_extra_requirements,
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
                    _mark_kanji_overlap_for_review(seg)
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
    if name_ocr is not None:
        name_ocr.close()

    work_relative = str(run_cache_path.resolve().relative_to(output_dir.resolve()))
    _dump_translation_cache(
        run_cache_path,
        video_path=str(args.video),
        config_path=str(Path(args.config).resolve()),
        segments=final_segments,
        debug_texts=debug_texts,
        prefix_entries=cache_prefix_entries,
        source_work_cache=work_relative,
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
            style=subtitle_style,
            dialogue_height=dialogue_roi.height,
            title_height=title_ocr_roi.height,
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
            style=subtitle_style,
            dialogue_height=dialogue_roi.height,
            title_height=title_ocr_roi.height,
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
            style=subtitle_style,
            dialogue_height=dialogue_roi.height,
            title_height=title_ocr_roi.height,
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
    parser.add_argument(
        "--name-split-mode",
        choices=["config", "mask", "ocr"],
        default="config",
        help="Name-region split verification mode. Use mask for no OCR, ocr for OCR confirmation.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
