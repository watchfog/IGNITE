from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from .models import Roi


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
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _timestamp_to_frame_number(timestamp: float, start_sec: float, fps: float) -> int:
    frame_index = int(round((timestamp - start_sec) * fps))
    return max(0, frame_index + 1)


def _try_load_cached_full_frame(
    timestamp: float,
    cache_dir: Path,
    start_sec: float,
    fps: float,
) -> Image.Image | None:
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
    frame_num = _timestamp_to_frame_number(timestamp, start_sec, fps)
    frame_path = cache_dir / f"{frame_num:06d}.png"
    if not frame_path.exists():
        return None, "miss_not_found", frame_num
    try:
        return Image.open(frame_path).convert("RGB"), "hit", frame_num
    except Exception:
        return None, "miss_read_error", frame_num


def _assert_crop_size(paths: list[Path], roi: Roi, tag: str) -> None:
    if not paths:
        return
    try:
        im = Image.open(paths[0])
        w, h = im.size
    except Exception:
        return
    if abs(w - roi.width) > 1 or abs(h - roi.height) > 1:
        raise RuntimeError(
            f"{tag} crop size mismatch: got {w}x{h}, expected {roi.width}x{roi.height} (tolerance=1px). "
            "Please check ROI config and ffmpeg crop filter."
        )
