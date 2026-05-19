from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Any

import yaml

from .config import load_config


ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


@dataclass
class ArchiveResult:
    archive_dir: Path
    cache_path: Path
    copied: dict[str, Path] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    manifest_path: Path | None = None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_name(value: str, fallback: str = "archive") -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", str(value or "")).strip(" ._\t\r\n")
    return cleaned or fallback


def _unique_archive_dir(dest_root: Path, name: str, overwrite: bool) -> Path:
    base = dest_root / _safe_name(name)
    if overwrite or not base.exists():
        return base
    i = 2
    while True:
        cand = dest_root / f"{base.name}_{i}"
        if not cand.exists():
            return cand
        i += 1


def _resolve_ref(raw: Any, *, cache_path: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    for base in (cache_path.parent, ROOT):
        cand = (base / path).resolve()
        if cand.exists():
            return cand
    return (cache_path.parent / path).resolve()


def _resolve_existing_path(raw: str | Path | None, *, cache_path: Path, label: str) -> Path:
    if raw is None:
        raise RuntimeError(f"无法定位{label}")
    path = raw if isinstance(raw, Path) else Path(str(raw))
    if not path.is_absolute():
        path = _resolve_ref(path, cache_path=cache_path) or path
    path = path.resolve()
    if not path.exists():
        raise RuntimeError(f"{label}不存在: {path}")
    return path


def _resolve_output_dir_from_cache(cache_path: Path) -> Path:
    cur = cache_path.resolve().parent
    while cur != cur.parent:
        if cur.name == "work":
            return cur.parent
        cur = cur.parent
    return cache_path.resolve().parent


def _copy_file(src: Path, dst: Path, copied: dict[str, Path], key: str, *, overwrite: bool) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if src == dst:
        copied[key] = dst
        return
    if dst.exists() and not overwrite:
        raise RuntimeError(f"目标文件已存在: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied[key] = dst


def _maybe_copy(src: Path, dst: Path, copied: dict[str, Path], missing: list[str], key: str, *, overwrite: bool) -> None:
    if src.exists():
        _copy_file(src, dst, copied, key, overwrite=overwrite)
    else:
        missing.append(str(src))


def _normalize_path_setting(cfg: dict[str, Any], section: str, key: str, *, config_path: Path) -> None:
    sec = cfg.get(section)
    if not isinstance(sec, dict):
        return
    raw = str(sec.get(key, "") or "").strip()
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute():
        for base in (config_path.parent, ROOT):
            cand = (base / path).resolve()
            if cand.exists():
                sec[key] = str(cand)
                return
        sec[key] = str((config_path.parent / path).resolve())


def _unique_file_path(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while True:
        cand = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not cand.exists():
            return cand
        i += 1


def _resolve_config_asset(raw: Any, *, config_path: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    for base in (config_path.parent, ROOT):
        cand = (base / path).resolve()
        if cand.exists():
            return cand
    return (config_path.parent / path).resolve()


def _archive_marker_templates(
    cfg: dict[str, Any],
    *,
    config_path: Path,
    archive_dir: Path,
    copied: dict[str, Path],
    missing: list[str],
    overwrite: bool,
) -> None:
    target_dir = archive_dir / "marker_templates"
    for section_name, manifest_prefix in (("marker", "marker_template"), ("marker_2", "marker2_template")):
        section = cfg.get(section_name)
        if not isinstance(section, dict):
            continue
        raw_paths = section.get("template_paths")
        if not isinstance(raw_paths, list):
            continue
        new_paths: list[str] = []
        for idx, raw in enumerate(raw_paths, start=1):
            src = _resolve_config_asset(raw, config_path=config_path)
            if src is None or not src.exists():
                missing.append(f"{section_name}.template_paths: {raw}")
                new_paths.append(str(raw))
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            dst = target_dir / _safe_name(src.stem, manifest_prefix)
            dst = dst.with_suffix(src.suffix.lower() or ".png")
            if dst.exists() and not overwrite:
                dst = _unique_file_path(dst)
            _copy_file(src, dst, copied, f"{manifest_prefix}_{idx}", overwrite=overwrite)
            new_paths.append(dst.relative_to(archive_dir).as_posix())
        section["template_paths"] = new_paths


def _write_merged_config(
    src_config: Path,
    dst_config: Path,
    *,
    video_filename: str,
    archive_dir: Path,
    copied: dict[str, Path],
    missing: list[str],
    overwrite: bool,
) -> None:
    cfg = load_config(src_config)
    cfg.pop("extends", None)
    cfg["video_path"] = video_filename
    _normalize_path_setting(cfg, "translation", "api_key_file", config_path=src_config)
    _normalize_path_setting(cfg, "tools", "ffmpeg_path", config_path=src_config)
    _normalize_path_setting(cfg, "tools", "ffprobe_path", config_path=src_config)
    _archive_marker_templates(
        cfg,
        config_path=src_config,
        archive_dir=archive_dir,
        copied=copied,
        missing=missing,
        overwrite=overwrite,
    )
    dst_config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _candidate_hard_subtitle_videos(cache_path: Path, video_path: Path | None = None) -> list[Path]:
    output_dir = _resolve_output_dir_from_cache(cache_path)
    candidates: list[Path] = []
    if video_path is not None:
        candidates.append(output_dir / f"{video_path.stem}_subtitled.mp4")
    candidates.extend(sorted(output_dir.glob("*subtitled*.mp4"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
    raw = _load_json(cache_path) if cache_path.exists() else {}
    source_work = str(raw.get("source_work_cache", "") or "") if isinstance(raw, dict) else ""
    if source_work:
        candidates.append((output_dir / source_work).resolve().parent / "video_with_subtitles.mp4")
    work_dir = output_dir / "work"
    if work_dir.exists():
        candidates.extend(sorted(work_dir.glob("run_*/video_with_subtitles.mp4"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def find_hard_subtitle_video(cache_path: Path, video_path: Path | None = None) -> Path | None:
    for cand in _candidate_hard_subtitle_videos(cache_path, video_path):
        if cand.exists() and cand.is_file():
            return cand
    return None


def _archive_name(cache_payload: dict[str, Any], cache_path: Path, video_path: Path | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if video_path is not None:
        return video_path.stem
    raw_video = str(cache_payload.get("video", "") or "").strip()
    if raw_video:
        return Path(raw_video).stem
    return cache_path.parent.name


def archive_project(
    *,
    cache_path: Path,
    dest_root: Path,
    video_path: Path | None = None,
    config_path: Path | None = None,
    hard_sub_video: Path | None = None,
    name: str | None = None,
    overwrite: bool = False,
) -> ArchiveResult:
    cache_path = cache_path.resolve()
    if not cache_path.exists():
        raise RuntimeError(f"cache不存在: {cache_path}")
    payload = _load_json(cache_path)
    if not isinstance(payload, dict):
        raise RuntimeError("仅支持对象格式 translation_cache.json")

    resolved_video = _resolve_existing_path(video_path or _resolve_ref(payload.get("video"), cache_path=cache_path), cache_path=cache_path, label="视频")
    resolved_config = _resolve_existing_path(config_path or _resolve_ref(payload.get("config_path"), cache_path=cache_path), cache_path=cache_path, label="配置")
    dest_root = dest_root.resolve()
    archive_dir = _unique_archive_dir(dest_root, _archive_name(payload, cache_path, resolved_video, name), overwrite)
    archive_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, Path] = {}
    missing: list[str] = []
    video_dst = archive_dir / (_safe_name(resolved_video.stem, "video") + (resolved_video.suffix.lower() or ".mp4"))
    config_dst = archive_dir / "config.yaml"
    cache_dst = archive_dir / "translation_cache_latest.json"

    _copy_file(resolved_video, video_dst, copied, "video", overwrite=overwrite)
    _write_merged_config(
        resolved_config,
        config_dst,
        video_filename=video_dst.name,
        archive_dir=archive_dir,
        copied=copied,
        missing=missing,
        overwrite=overwrite,
    )
    copied["config"] = config_dst

    archived_payload = dict(payload)
    archived_payload["video"] = video_dst.name
    archived_payload["config_path"] = config_dst.name
    archived_payload.pop("source_work_cache", None)
    _save_json(cache_dst, archived_payload)
    copied["cache"] = cache_dst

    output_dir = _resolve_output_dir_from_cache(cache_path)
    for filename, key in (("subtitles.ass", "subtitles"), ("subtitles_debug.ass", "subtitles_debug")):
        _maybe_copy(output_dir / filename, archive_dir / filename, copied, missing, key, overwrite=overwrite)

    hard_src = hard_sub_video.resolve() if hard_sub_video else find_hard_subtitle_video(cache_path, resolved_video)
    if hard_src is not None:
        if not hard_src.exists():
            raise RuntimeError(f"硬字幕视频不存在: {hard_src}")
        hard_name = _safe_name(hard_src.stem, "video_subtitled") + hard_src.suffix.lower()
        _copy_file(hard_src, archive_dir / hard_name, copied, "hard_sub_video", overwrite=overwrite)
    else:
        missing.append("hard_sub_video")

    manifest = {
        "version": 1,
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "cache": str(cache_path),
            "video": str(resolved_video),
            "config": str(resolved_config),
            "output_dir": str(output_dir),
            "hard_sub_video": str(hard_src) if hard_src is not None else "",
        },
        "archive": {k: v.name for k, v in copied.items()},
        "missing": missing,
    }
    manifest_path = archive_dir / "archive_manifest.json"
    _save_json(manifest_path, manifest)
    return ArchiveResult(archive_dir=archive_dir, cache_path=cache_dst, copied=copied, missing=missing, manifest_path=manifest_path)
