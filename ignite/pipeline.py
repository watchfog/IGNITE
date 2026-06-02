from __future__ import annotations

import argparse
import copy
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
import io
import shutil
import subprocess
import sys
import threading
import time
import yaml
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cache_manager import (
    _build_debug_subtitle,
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
from .auto_review import _profile_chat_params, run_auto_review_entries
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
from .subtitle_export import write_ass
from .translation_runtime import (
    ChatCompletionsTextTranslator,
    TEXT_EXTRACTION_PROFILE_MODE,
    TranslationModelProfile,
    VlmImageTextExtractor,
    VlmResponsesTranslator,
    _normalize_text_extraction_backend,
    load_api_key,
    resolve_responses_base_url,
    resolve_translation_model_profile,
    translate_ocr_text_segment_with_retry,
    translate_segment_with_retry,
)


def _roi_from_cfg(cfg: dict[str, Any], key: str) -> Roi:
    v = cfg["roi"][key]
    return Roi(int(v[0]), int(v[1]), int(v[2]), int(v[3]))


def _collect_marker_templates(marker_cfg: dict[str, Any], config_dir: Path | None = None) -> list[Path]:
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
                candidates = []
                if config_dir is not None:
                    candidates.append((config_dir / p).resolve())
                candidates.append((Path.cwd() / p).resolve())
                p = next((cand for cand in candidates if cand.exists()), candidates[0])
            _add(p)
    return out


def _normalize_translation_mode(value: Any) -> str:
    mode = str(value or "vlm_responses").strip().lower()
    vlm_responses_aliases = {
        "vlm",
        "vlm_responses",
        "responses_vlm",
        "vlm_bailian",
        "qwen_vlm",
        "bailian",
    }
    ocr_chat_aliases = {
        "ocr_chat",
        "ocr_chat_completions",
        "ocr_chat_completion",
        "ocr_llm",
        "text_chat",
    }
    if mode in vlm_responses_aliases:
        return "vlm_responses"
    if mode in ocr_chat_aliases:
        return "ocr_chat_completions"
    return mode


def _load_api_key_for_pipeline_profile(
    profile: TranslationModelProfile,
    *,
    config_dir: Path,
    project_root: Path,
) -> str:
    if profile.api_key:
        return profile.api_key
    if not profile.api_key_file:
        return ""
    p = Path(profile.api_key_file)
    if not p.is_absolute():
        candidates = [(config_dir / p).resolve(), (project_root / p).resolve()]
        p = next((cand for cand in candidates if cand.exists()), candidates[0])
    return load_api_key(p)


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
        "subtitles.ass",
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


def _cleanup_obsolete_srt_outputs(output_dir: Path) -> None:
    for name in ("subtitles.srt", "subtitles_debug.srt"):
        path = output_dir / name
        if not path.exists():
            continue
        try:
            path.unlink()
            _log(f"Removed obsolete SRT output: {path}")
        except Exception as exc:
            _log(f"Warning: failed to remove obsolete SRT {path}: {exc.__class__.__name__}")


def _resolve_run_prefix(args: argparse.Namespace) -> str:
    if bool(getattr(args, "subtitles_from_cache", False)):
        return "run_subtitle"
    if bool(getattr(args, "cache_only", False)):
        return "run_cache"
    return "run_full"


def run_pipeline(args: argparse.Namespace) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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
        _cleanup_obsolete_srt_outputs(output_dir)
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
        write_ass(
            segs,
            output_dir / "subtitles_debug.ass",
            video_width=video_meta.width,
            video_height=video_meta.height,
            subtitle_location=subtitle_location,
            title_translation_location=title_translation_location,
            title_info_location=title_info_location,
            style=subtitle_style,
            dialogue_height=dialogue_roi.height,
            title_height=title_ocr_roi.height,
            debug_overlay_segments=dbg_segs,
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
        template_paths = _collect_marker_templates(marker_cfg, Path(args.config).resolve().parent)
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
        marker2_templates = _collect_marker_templates(marker2_cfg, Path(args.config).resolve().parent)
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

    dialogue_presence_mode = str(getattr(args, "dialogue_presence_mode", "") or "").strip().lower()
    if dialogue_presence_mode not in {"marker2", "ocr"}:
        raise RuntimeError(
            "Missing --dialogue-presence-mode. Use 'marker2' or 'ocr' for normal pipeline runs."
        )
    name_ocr: NameOcrRunner | None = None
    ocr_engine: Any | None = None
    if dialogue_presence_mode == "marker2":
        if marker2_matcher is None:
            raise RuntimeError(
                "--dialogue-presence-mode marker2 requires marker_2.template_paths. "
                "Set Marker2 templates in Profile Editor or use --dialogue-presence-mode ocr."
            )
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
    elif dialogue_presence_mode == "ocr":
        _log("Initializing OCR engine (dialogue presence verification).")
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
        name_split_use_ocr = True
        name_ocr_workers = int(cfg["ocr"].get("name_ocr_workers", 1))
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
    vlm_translator: VlmResponsesTranslator | None = None
    text_translator: ChatCompletionsTextTranslator | None = None
    image_text_extractor: VlmImageTextExtractor | None = None
    translation_mode = _normalize_translation_mode(
        getattr(args, "translation_mode", "") or tr_cfg.get("mode", "vlm_responses")
    )
    text_extraction_backend = _normalize_text_extraction_backend(
        getattr(args, "text_extraction_backend", "")
        or tr_cfg.get("text_extraction_backend", tr_cfg.get("text_extraction_mode", "ocr"))
    )
    text_extraction_model_arg = str(
        getattr(args, "text_extraction_model", "")
        or tr_cfg.get("text_extraction_model_profile", "")
        or ""
    ).strip()
    translation_model_arg = str(getattr(args, "translation_model", "") or "").strip()
    translate_llm = str(tr_cfg.get("translate_llm", "")).strip().lower()
    if translate_llm:
        translation_mode = _normalize_translation_mode(translate_llm)
    recog_mode = str(tr_cfg.get("recognition_mode", "")).strip().lower()
    if recog_mode in {"vlm", "vlm_translate", "image_translate"}:
        translation_mode = "vlm_responses"
    if not args.skip_translation:
        profile = resolve_translation_model_profile(tr_cfg, translation_mode, translation_model_arg)
        api_key = _load_api_key_for_pipeline_profile(
            profile,
            config_dir=Path(args.config).resolve().parent,
            project_root=project_root,
        )
        if translation_mode == "vlm_responses":
            _log("Initializing VLM Responses translator.")
            vlm_translator = VlmResponsesTranslator(
                api_key=api_key,
                model=profile.model,
                responses_base_url=profile.base_url or resolve_responses_base_url(tr_cfg),
                temperature=llm_temperature,
                enable_thinking=llm_enable_thinking,
                thinking_budget=llm_thinking_budget,
                preserve_thinking=llm_preserve_thinking,
                timeout_sec=int(tr_cfg.get("timeout_sec", 90)),
                timeout_backoff_sec=int(tr_cfg.get("timeout_backoff_sec", 15)),
                max_retries=int(tr_cfg.get("max_retries", 2)),
                retry_delay_sec=float(tr_cfg.get("retry_delay_sec", 1.5)),
                empty_max_attempts=int(tr_cfg.get("empty_max_attempts", 3)),
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
                "Translation mode: VLM Responses "
                f"profile={profile.name} model={profile.model} "
                f"responses_base_url={profile.base_url} "
                f"temp={llm_temperature:.2f} thinking={llm_enable_thinking} "
                f"thinking_budget={(llm_thinking_budget if llm_thinking_budget is not None else 'default')} "
                f"preserve_thinking={llm_preserve_thinking} "
                f"io_log={vlm_io_log_enabled} "
                f"web_search={enable_web_search}"
            )
        elif translation_mode == "ocr_chat_completions":
            _log("Initializing OCR + Chat Completions translator.")
            profile_temp, profile_top_p, profile_top_k = _profile_chat_params(tr_cfg, profile.name)
            chat_temp = profile_temp if profile_temp is not None else llm_temperature
            ctx_window = int(tr_cfg.get("chat_context_window", 8))
            text_translator = ChatCompletionsTextTranslator(
                api_key=api_key,
                model=profile.model,
                base_url=profile.base_url,
                temperature=chat_temp,
                top_p=profile_top_p,
                top_k=profile_top_k,
                context_window=ctx_window,
                timeout_sec=int(tr_cfg.get("timeout_sec", 90)),
                timeout_backoff_sec=int(tr_cfg.get("timeout_backoff_sec", 15)),
                max_retries=int(tr_cfg.get("max_retries", 2)),
                retry_delay_sec=float(tr_cfg.get("retry_delay_sec", 1.5)),
                empty_max_attempts=int(tr_cfg.get("empty_max_attempts", 3)),
                disable_env_proxy=bool(tr_cfg.get("disable_env_proxy", True)),
                game_name=game_name,
                source_language=source_lang,
                target_language=target_lang,
                log_fn=lambda m: _log(f"[LLM] {m}"),
                io_log_path=work_dir / "chat_io.log",
                io_log_enabled=vlm_io_log_enabled,
                enable_web_search=enable_web_search,
            )
            _log(
                "Translation mode: OCR + Chat Completions "
                f"profile={profile.name} model={profile.model} base_url={profile.base_url} "
                f"temp={chat_temp:.2f} top_p={profile_top_p} top_k={profile_top_k} "
                f"io_log={vlm_io_log_enabled} "
                f"web_search={enable_web_search} auth={'yes' if api_key else 'no'}"
            )
            if text_extraction_backend != "ocr":
                extraction_profile = resolve_translation_model_profile(
                    tr_cfg,
                    TEXT_EXTRACTION_PROFILE_MODE,
                    text_extraction_model_arg,
                )
                extraction_api_key = _load_api_key_for_pipeline_profile(
                    extraction_profile,
                    config_dir=Path(args.config).resolve().parent,
                    project_root=project_root,
                )
                ext_temp, ext_top_p, ext_top_k = _profile_chat_params(tr_cfg, extraction_profile.name)
                extraction_temp = ext_temp if ext_temp is not None else float(tr_cfg.get("text_extraction_temperature", 0.0))
                extraction_max_tokens = int(tr_cfg.get("text_extraction_max_tokens", 512) or 512)
                extraction_thinking = bool(tr_cfg.get("text_extraction_enable_thinking", False))
                extraction_budget_raw = tr_cfg.get("text_extraction_thinking_budget", None)
                extraction_budget: int | None = None
                if extraction_budget_raw is not None and str(extraction_budget_raw).strip() != "":
                    try:
                        extraction_budget = int(extraction_budget_raw)
                    except Exception:
                        extraction_budget = None
                image_text_extractor = VlmImageTextExtractor(
                    api_key=extraction_api_key,
                    model=extraction_profile.model,
                    base_url=extraction_profile.base_url,
                    backend=text_extraction_backend,
                    temperature=extraction_temp,
                    top_p=ext_top_p,
                    top_k=ext_top_k,
                    max_tokens=extraction_max_tokens,
                    enable_thinking=extraction_thinking,
                    thinking_budget=extraction_budget,
                    preserve_thinking=bool(tr_cfg.get("text_extraction_preserve_thinking", False)),
                    timeout_sec=int(tr_cfg.get("text_extraction_timeout_sec", tr_cfg.get("timeout_sec", 90))),
                    timeout_backoff_sec=int(tr_cfg.get("timeout_backoff_sec", 15)),
                    max_retries=int(tr_cfg.get("max_retries", 2)),
                    retry_delay_sec=float(tr_cfg.get("retry_delay_sec", 1.5)),
                    empty_max_attempts=int(tr_cfg.get("empty_max_attempts", 3)),
                    disable_env_proxy=bool(tr_cfg.get("disable_env_proxy", True)),
                    game_name=game_name,
                    source_language=source_lang,
                    log_fn=lambda m: _log(f"[VLM_OCR] {m}"),
                    io_log_path=work_dir / "vlm_text_extraction_io.log",
                    io_log_enabled=vlm_io_log_enabled,
                )
                _log(
                    "Text extraction mode: VLM image text extraction "
                    f"backend={text_extraction_backend} profile={extraction_profile.name} "
                    f"model={extraction_profile.model} base_url={extraction_profile.base_url} "
                    f"temp={extraction_temp:.2f} top_p={ext_top_p} top_k={ext_top_k} "
                    f"max_tokens={extraction_max_tokens} thinking={extraction_thinking} "
                    f"auth={'yes' if extraction_api_key else 'no'}"
                )
            else:
                _log("Text extraction mode: RapidOCR")
        else:
            raise RuntimeError(f"Unsupported translation mode: {translation_mode}")

    if text_translator is not None and text_extraction_backend == "ocr" and ocr_engine is None:
        _log("Initializing OCR engine (OCR + Chat Completions text extraction).")
        ocr_engine = build_ocr_engine(cfg["ocr"])
        ocr_info = ocr_engine.info()
        _log(
            "OCR engine: "
            f"{ocr_info.get('engine')} / runtime={ocr_info.get('runtime')} / "
            f"target={ocr_info.get('target_lang')} / actual={ocr_info.get('actual_lang')} / "
            f"charset={ocr_info.get('charset_size')}"
        )

    final_segments: list[DialogueSegment] = []
    debug_text_by_segment_id: dict[int, str] = {}
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
            raw_id=marker_seg_id,
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
        debug_text = _build_debug_subtitle(
            segment_id=seg.segment_id,
            raw_id=seg.raw_id,
            time_start=seg.time_start,
            time_end=seg.time_end,
            dialogue_type=seg.dialogue_type,
            raw_seen_idx=marker_seen_idx,
            raw_count=marker_count,
        )
        debug_text_by_segment_id[seg.segment_id] = debug_text
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
                cached_style = cached_item.get("subtitle_style")
                seg.subtitle_style = copy.deepcopy(cached_style) if isinstance(cached_style, dict) else {}
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
            # VLM Responses history is intentionally used only by Review retranslation.
            history_items = None
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

    work_relative = str(run_cache_path.resolve().relative_to(output_dir.resolve()))

    if text_translator is not None and not args.skip_translation:
        if text_extraction_backend == "ocr" and ocr_engine is None:
            raise RuntimeError("OCR engine is required for ocr_chat_completions mode")
        if text_extraction_backend != "ocr" and image_text_extractor is None:
            raise RuntimeError("VLM image text extractor is required for selected text extraction backend")
        _log(
            "OCR + Chat Completions: sliding text-extraction lookahead "
            f"with background prefetch (backend={text_extraction_backend})."
        )
        ctx_window = max(0, int(getattr(text_translator, "context_window", 0)))
        prefetch_setting = str(
            tr_cfg.get("text_extraction_prefetch_during_translation", "auto") or "auto"
        ).strip().lower()
        if prefetch_setting in {"1", "true", "yes", "on"}:
            prefetch_during_translation = True
        elif prefetch_setting in {"0", "false", "no", "off"}:
            prefetch_during_translation = False
        else:
            prefetch_during_translation = text_extraction_backend == "ocr"
        _log(
            "Text extraction prefetch during translation: "
            f"{prefetch_during_translation} (setting={prefetch_setting})"
        )
        ocr_done_indices: set[int] = set()
        ocr_lock = threading.Lock()
        ocr_pool = ThreadPoolExecutor(max_workers=1)
        ocr_futures: dict[int, Future[None]] = {}

        def _is_chat_dialogue(seg: DialogueSegment) -> bool:
            return str(seg.dialogue_type or "").strip().lower() not in {"blank_no_name", "blank", "title"}

        def _run_ocr_for_index(seg_idx: int) -> None:
            with ocr_lock:
                if seg_idx in ocr_done_indices:
                    return
                ocr_done_indices.add(seg_idx)
            seg = final_segments[seg_idx]
            if not _is_chat_dialogue(seg):
                return
            raw = raw_segments[seg_idx]
            anchor_fid = _pick_marker_anchor_frame(raw=raw, from_end_frames=marker_anchor_from_end_frames)
            ts = float(raw["start_sec"]) + (float(anchor_fid) / float(raw["scan_fps"]))
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
            fallback_image: Image.Image | None = None
            if dialogue_image is None or name_image is None:
                _log(
                    f"[OCR_CHAT_IMAGE_CACHE] fallback_ffmpeg segment={seg.segment_id} "
                    f"dialogue={dialogue_cache_status}:{dialogue_cache_frame_num:06d} "
                    f"name={name_cache_status}:{name_cache_frame_num:06d} ts={ts:.3f}"
                )
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

            debug_text = debug_text_by_segment_id.get(seg.segment_id, "")
            if dialogue_image is None or name_image is None:
                seg.translation_subtitle = debug_text
                seg.review_reason = _merge_review_reasons(seg.review_reason, ["ocr_image_prepare_failed"])
                seg.needs_review = True
                if fallback_image is not None:
                    fallback_image.close()
                return

            try:
                if text_extraction_backend == "ocr":
                    if ocr_engine is None:
                        raise RuntimeError("OCR engine is not initialized")
                    name_arr = np.asarray(name_image, dtype=np.uint8)
                    dialogue_arr = np.asarray(dialogue_image, dtype=np.uint8)
                    speaker_ocr = ocr_engine.recognize_array(name_arr)
                    dialogue_ocr = ocr_engine.recognize_array(dialogue_arr)
                    seg.speaker = str(speaker_ocr.text or "").strip()
                    seg.speaker_confidence = float(speaker_ocr.confidence or 0.0)
                    seg.text_original = str(dialogue_ocr.text or "").strip()
                    seg.text_ocr_confidence = float(dialogue_ocr.confidence or 0.0)
                else:
                    if image_text_extractor is None:
                        raise RuntimeError("VLM image text extractor is not initialized")
                    speaker_url = _image_to_base64(name_image)
                    dialogue_url = _image_to_base64(dialogue_image)
                    speaker_text, original_text, usage = image_text_extractor.extract_text_from_images(
                        speaker_image=speaker_url,
                        dialogue_image=dialogue_url,
                        request_tag=f"segment {seg.segment_id}",
                    )
                    seg.speaker = str(speaker_text or "").strip()
                    seg.speaker_confidence = 0.0
                    seg.text_original = str(original_text or "").strip()
                    seg.text_ocr_confidence = 0.0
                    _log(
                        f"[VLM_OCR] segment {seg.segment_id}: "
                        f"tokens_in={int(usage.get('prompt_tokens', 0))} "
                        f"tokens_out={int(usage.get('completion_tokens', 0))}"
                    )
                seg.line_count_detected = max(1, seg.text_original.count("\n") + 1) if seg.text_original else 1
                if not seg.text_original:
                    seg.translation_subtitle = debug_text
                    seg.review_reason = _merge_review_reasons(seg.review_reason, ["text_extraction_missing"])
                    seg.needs_review = True
                _log(
                    f"[TEXT_EXTRACT] segment {seg.segment_id}: "
                    f"speaker_chars={len(seg.speaker)} text_chars={len(seg.text_original)} "
                    f"speaker_conf={seg.speaker_confidence:.3f} text_conf={seg.text_ocr_confidence:.3f}"
                )
            except Exception as exc:
                seg.translation_subtitle = debug_text
                seg.review_reason = _merge_review_reasons(
                    seg.review_reason,
                    [f"text_extraction_error:{exc.__class__.__name__}"],
                )
                seg.needs_review = True
                _log(f"[TEXT_EXTRACT] segment {seg.segment_id}: failed ({exc.__class__.__name__}): {exc}")
            finally:
                name_image.close()
                dialogue_image.close()
                if fallback_image is not None:
                    fallback_image.close()

        def _submit_ocr(idx: int) -> None:
            if idx < 0 or idx >= len(final_segments):
                return
            with ocr_lock:
                if idx in ocr_futures:
                    return
                ocr_futures[idx] = ocr_pool.submit(_run_ocr_for_index, idx)

        def _wait_ocr(idx: int) -> None:
            with ocr_lock:
                fut = ocr_futures.get(idx)
            if fut is not None:
                fut.result()
                return
            with ocr_lock:
                if idx in ocr_done_indices:
                    return
            _run_ocr_for_index(idx)

        for i in range(min(len(final_segments), 1 + ctx_window)):
            _submit_ocr(i)

        _log("Chat Completions strict sequential mode: concurrent_workers=1")

        def _context_item(seg_idx: int, include_translation: bool) -> dict[str, str] | None:
            seg = final_segments[seg_idx]
            if not _is_chat_dialogue(seg):
                return None
            original = str(seg.text_original or "").strip()
            if not original:
                return None
            item = {"speaker": str(seg.speaker or "").strip(), "original": original}
            if include_translation:
                translation = str(seg.translation_subtitle or "").strip()
                debug_text = debug_text_by_segment_id.get(seg.segment_id, "")
                if translation and translation != debug_text:
                    item["translation"] = translation
            return item

        def _context_before_with_translations(seg_idx: int) -> list[dict[str, str]] | None:
            if ctx_window <= 0:
                return None
            items: list[dict[str, str]] = []
            scan_idx = seg_idx - 1
            while scan_idx >= 0 and len(items) < ctx_window:
                item = _context_item(scan_idx, include_translation=True)
                if item is not None:
                    items.insert(0, item)
                scan_idx -= 1
            return items or None

        def _context_after_with_lookahead(seg_idx: int) -> list[dict[str, str]] | None:
            if ctx_window <= 0:
                return None
            items: list[dict[str, str]] = []
            scan_idx = seg_idx + 1
            while scan_idx < len(final_segments) and len(items) < ctx_window:
                _wait_ocr(scan_idx)
                item = _context_item(scan_idx, include_translation=False)
                if item is not None:
                    items.append(item)
                scan_idx += 1
            return items or None

        for seg_idx, seg in enumerate(final_segments):
            _wait_ocr(seg_idx)
            if prefetch_during_translation:
                _submit_ocr(seg_idx + ctx_window + 1)
            if str(seg.dialogue_type or "").strip().lower() in {"blank_no_name", "blank", "title"}:
                continue
            debug_text = debug_text_by_segment_id.get(seg.segment_id, "")
            if not str(seg.text_original or "").strip():
                if not seg.translation_subtitle:
                    seg.translation_subtitle = debug_text
                continue
            cached_item = None
            k1 = _subtitle_cache_key(seg.segment_id, seg.time_start, seg.time_end, seg.dialogue_type)
            k2 = _subtitle_cache_key(0, seg.time_start, seg.time_end, seg.dialogue_type)
            candidate = cache_full.get(k1) or cache_time_type.get(k2)
            if candidate and str(candidate.get("text_original", "") or "").strip() == str(seg.text_original or "").strip():
                cached_item = candidate
            if cached_item and str(cached_item.get("translation_subtitle", "") or "").strip():
                cached_speaker = str(cached_item.get("speaker", "") or "")
                if cached_speaker:
                    seg.speaker = cached_speaker
                seg.translation_subtitle = str(cached_item.get("translation_subtitle", "") or "")
                cached_style = cached_item.get("subtitle_style")
                seg.subtitle_style = copy.deepcopy(cached_style) if isinstance(cached_style, dict) else {}
                _mark_kanji_overlap_for_review(seg)
                cache_hit_count += 1
                _log(f"[LLM] segment {seg.segment_id}: cache hit after OCR, skip request")
                continue

            history_items = None
            ctx_before = _context_before_with_translations(seg_idx)
            ctx_after = _context_after_with_lookahead(seg_idx)
            try:
                translated, usage = translate_ocr_text_segment_with_retry(
                    seg.segment_id,
                    text_translator,
                    seg.text_original,
                    seg.speaker,
                    history_items=history_items,
                    extra_requirements=vlm_extra_requirements,
                    context_before=ctx_before,
                    context_after=ctx_after,
                )
                vlm_prompt_tokens_total += int(usage.get("prompt_tokens", 0))
                vlm_completion_tokens_total += int(usage.get("completion_tokens", 0))
                vlm_total_tokens_total += int(
                    usage.get(
                        "total_tokens",
                        int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0)),
                    )
                )
                if translated and translated.strip():
                    seg.translation_subtitle = translated
                    _mark_kanji_overlap_for_review(seg)
                else:
                    seg.translation_subtitle = debug_text
                    seg.review_reason.append("llm_translation_empty")
                    seg.needs_review = True
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
            except Exception as exc:
                seg.translation_subtitle = debug_text
                seg.review_reason.append(f"llm_translation_error:{exc.__class__.__name__}")
                seg.needs_review = True
                _log(f"[LLM] segment {seg.segment_id}: request failed ({exc.__class__.__name__}): {exc}")

        ocr_pool.shutdown(wait=True)

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
    if (vlm_translator is not None or text_translator is not None) and not args.skip_translation:
        _log(
            "Translation token usage summary: "
            f"input={vlm_prompt_tokens_total}, "
            f"output={vlm_completion_tokens_total}, "
            f"total={vlm_total_tokens_total}"
        )
    if vlm_pool is not None:
        vlm_pool.shutdown(wait=True)
    if name_ocr is not None:
        name_ocr.close()

    auto_review_enabled = bool(getattr(args, "auto_review", False)) or bool(tr_cfg.get("auto_review_enabled", False))
    if auto_review_enabled and not args.skip_translation:
        auto_review_entries = []
        for seg in final_segments:
            if str(seg.dialogue_type or "").strip().lower() in {"blank_no_name", "blank", "title"}:
                continue
            original = str(seg.text_original or "").strip()
            translation = str(seg.translation_subtitle or "").strip()
            if not original or not translation:
                continue
            auto_review_entries.append(
                {
                    "id": int(seg.segment_id),
                    "speaker": str(seg.speaker or "").strip(),
                    "original": original,
                    "translation": translation,
                }
            )
        if auto_review_entries:
            auto_review_model = str(
                getattr(args, "auto_review_model", "")
                or tr_cfg.get("auto_review_model_profile", "")
                or ""
            ).strip()
            auto_review_chunk_size = int(
                getattr(args, "auto_review_chunk_size", 0)
                or tr_cfg.get("auto_review_chunk_size", 80)
                or 80
            )
            _log(
                "自动review 开始: "
                f"entries={len(auto_review_entries)} chunk_size={auto_review_chunk_size} "
                f"model={auto_review_model or '(default)'}"
            )
            updates, report = run_auto_review_entries(
                entries=auto_review_entries,
                glossary=vlm_extra_requirements,
                tr_cfg=tr_cfg,
                model_profile=auto_review_model,
                chunk_size=auto_review_chunk_size,
                timeout_sec=int(tr_cfg.get("auto_review_timeout_sec", tr_cfg.get("timeout_sec", 120))),
                temperature=None,
                max_tokens=int(tr_cfg.get("auto_review_max_tokens", 0) or 0),
                parse_retries=int(tr_cfg.get("auto_review_parse_retries", 1)),
                repair_max_tokens=int(tr_cfg.get("auto_review_repair_max_tokens", 0) or 0),
                review_mode=str(tr_cfg.get("auto_review_mode", "thorough") or "thorough"),
                stream=False,
                log_fn=None,
            )
            update_map: dict[int, str] = {}
            reason_map: dict[int, str] = {}
            for item in updates:
                if item.get("changed"):
                    update_map[int(item["id"])] = str(item.get("new_translation", "") or "").strip()
                    reason_map[int(item["id"])] = str(item.get("reason", "") or "").strip()
            changed_count = 0
            for seg in final_segments:
                new_text = update_map.get(int(seg.segment_id))
                if not new_text:
                    continue
                old_text = str(seg.translation_subtitle or "").strip()
                if old_text == new_text:
                    continue
                seg.translation_subtitle = new_text
                reason = reason_map.get(int(seg.segment_id), "")
                if reason:
                    seg.auto_review_reason = reason
                changed_count += 1
                _log(
                    f"[自动review] seg {seg.segment_id} {seg.speaker}: "
                    f"{old_text} → {new_text}"
                )
                _mark_kanji_overlap_for_review(seg)
            _log(
                "自动review 结束: "
                f"model_updates={report.get('model_update_count', 0)} "
                f"changed={changed_count} parse_errors={report.get('parse_error_count', 0)}"
            )
        else:
            _log("自动review 已启用但没有可用的对话条目。")

    _dump_translation_cache(
        run_cache_path,
        video_path=str(args.video),
        config_path=str(Path(args.config).resolve()),
        segments=final_segments,
        prefix_entries=cache_prefix_entries,
        source_work_cache=work_relative,
        subtitle_style_cfg=subtitle_style,
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
    _cleanup_obsolete_srt_outputs(output_dir)
    if args.skip_translation:
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
            debug_overlay_segments=cache_debug_segments,
        )
    else:
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
        write_ass(
            cache_sub_segments,
            output_dir / "subtitles_debug.ass",
            video_width=video_meta.width,
            video_height=video_meta.height,
            subtitle_location=subtitle_location,
            title_translation_location=title_translation_location,
            title_info_location=title_info_location,
            style=subtitle_style,
            dialogue_height=dialogue_roi.height,
            title_height=title_ocr_roi.height,
            debug_overlay_segments=cache_debug_segments,
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
                "raw_id": seg.raw_id,
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
        "--dialogue-presence-mode",
        choices=["marker2", "ocr"],
        default="",
        help="How to suppress blank marker segments: marker2 template or name OCR. Required for normal runs.",
    )
    parser.add_argument(
        "--translation-mode",
        choices=["vlm_responses", "ocr_chat_completions"],
        default="",
        help="Translation backend. Overrides translation.mode in config.",
    )
    parser.add_argument(
        "--translation-model",
        default="",
        help="translation.model_profiles key to use. Overrides translation.mode_models/config model.",
    )
    parser.add_argument(
        "--text-extraction-backend",
        choices=["ocr", "vlm_responses", "vlm_chat_completions"],
        default="",
        help="Text extraction backend for ocr_chat_completions mode. Overrides translation.text_extraction_backend.",
    )
    parser.add_argument(
        "--text-extraction-model",
        default="",
        help="translation.model_profiles key for VLM image text extraction.",
    )
    parser.add_argument(
        "--auto-review",
        action="store_true",
        help="翻译后运行 LLM 自动review，将修改后的译文写入 cache。",
    )
    parser.add_argument(
        "--auto-review-model",
        default="",
        help="自动review 使用的 translation.model_profiles key。默认读取 translation.auto_review_model_profile/qwen。",
    )
    parser.add_argument(
        "--auto-review-chunk-size",
        type=int,
        default=0,
        help="自动review chunk size。默认读取 translation.auto_review_chunk_size 或 80。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
