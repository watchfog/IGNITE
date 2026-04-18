from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .models import VideoMeta


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def extract_frame_to_memory(
    ffmpeg_path: str | Path,
    video_path: str | Path,
    time_sec: float,
) -> bytes:
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{time_sec:.6f}",
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        "1",
        "-f",
        "image2",
        "-",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    return result.stdout


def ffprobe_video(ffprobe_path: str | Path, video_path: str | Path) -> VideoMeta:
    cmd = [
        str(ffprobe_path),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(video_path),
    ]
    out = _run(cmd).stdout
    data = json.loads(out)["streams"][0]
    fps_num, fps_den = data["r_frame_rate"].split("/")
    fps = float(fps_num) / float(fps_den)
    return VideoMeta(
        width=int(data["width"]),
        height=int(data["height"]),
        fps=fps,
        duration=float(data["duration"]),
    )


def extract_frame(
    ffmpeg_path: str | Path,
    video_path: str | Path,
    time_sec: float,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{time_sec:.6f}",
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        "1",
        str(output_path),
    ]
    _run(cmd)


def extract_frame_with_filter(
    ffmpeg_path: str | Path,
    video_path: str | Path,
    time_sec: float,
    output_path: str | Path,
    vf_filter: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{time_sec:.6f}",
        "-vf",
        vf_filter,
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        "1",
        str(output_path),
    ]
    _run(cmd)


def extract_sequence(
    ffmpeg_path: str | Path,
    video_path: str | Path,
    output_dir: str | Path,
    fps: float,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    vf_filters: list[str] | None = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "%06d.png"
    filter_chain = [f"fps={fps:.6f}"]
    if vf_filters:
        filter_chain.extend(vf_filters)
    vf = ",".join(filter_chain)

    cmd = [str(ffmpeg_path), "-y"]
    if start_sec is not None:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    cmd.extend(["-i", str(video_path)])
    if duration_sec is not None:
        cmd.extend(["-t", f"{duration_sec:.3f}"])
    cmd.extend(["-vf", vf, str(pattern)])
    _run(cmd)
    return sorted(output_dir.glob("*.png"))


def extract_sequence_dialogue_name_marker(
    ffmpeg_path: str | Path,
    video_path: str | Path,
    dialogue_output_dir: str | Path,
    name_output_dir: str | Path,
    marker_output_dir: str | Path,
    fps: float,
    dialogue_crop_filter: str,
    name_crop_filter: str,
    marker_crop_filter: str,
    start_sec: float | None = None,
    duration_sec: float | None = None,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Extract dialogue/name/marker frame sequences in one ffmpeg run using split+crop."""
    dialogue_output_dir = Path(dialogue_output_dir)
    name_output_dir = Path(name_output_dir)
    marker_output_dir = Path(marker_output_dir)
    dialogue_output_dir.mkdir(parents=True, exist_ok=True)
    name_output_dir.mkdir(parents=True, exist_ok=True)
    marker_output_dir.mkdir(parents=True, exist_ok=True)

    dialogue_pattern = dialogue_output_dir / "%06d.png"
    name_pattern = name_output_dir / "%06d.png"
    marker_pattern = marker_output_dir / "%06d.png"

    filter_complex = (
        f"[0:v]fps={fps:.6f},split=3[vfull][vname][vmarker];"
        f"[vfull]{dialogue_crop_filter}[fullout];"
        f"[vname]{name_crop_filter}[nameout];"
        f"[vmarker]{marker_crop_filter}[markerout]"
    )

    cmd = [str(ffmpeg_path), "-y"]
    if start_sec is not None:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    cmd.extend(["-i", str(video_path)])
    if duration_sec is not None:
        cmd.extend(["-t", f"{duration_sec:.3f}"])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[fullout]",
            str(dialogue_pattern),
            "-map",
            "[nameout]",
            str(name_pattern),
            "-map",
            "[markerout]",
            str(marker_pattern),
        ]
    )
    _run(cmd)

    return (
        sorted(dialogue_output_dir.glob("*.png")),
        sorted(name_output_dir.glob("*.png")),
        sorted(marker_output_dir.glob("*.png")),
    )
