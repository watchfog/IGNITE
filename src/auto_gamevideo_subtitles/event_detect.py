from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def load_gray(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def _to_text_mask(gray: np.ndarray, mode: str = "text", scale: int = 2) -> np.ndarray:
    """Extract text-like foreground mask to reduce background interference."""
    if cv2 is not None:
        up = cv2.resize(
            gray,
            (gray.shape[1] * scale, gray.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        blur = cv2.bilateralFilter(up, 5, 40, 40)
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10,
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean = np.zeros_like(binary, dtype=np.uint8)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if mode == "marker":
                if area < 25 or area > 6000:
                    continue
                if w < 4 or h < 4:
                    continue
            else:
                if area < 8 or area > 2500:
                    continue
                if w < 2 or h < 4:
                    continue
                ratio = w / max(h, 1)
                if ratio < 0.1 or ratio > 10:
                    continue
            clean[labels == i] = 1
        return clean

    pil = Image.fromarray(gray, mode="L").resize(
        (gray.shape[1] * scale, gray.shape[0] * scale),
        Image.Resampling.BICUBIC,
    )
    denoise = pil.filter(ImageFilter.MedianFilter(size=3))
    blur = denoise.filter(ImageFilter.GaussianBlur(radius=1.2))
    arr = np.asarray(denoise, dtype=np.int16)
    arr_blur = np.asarray(blur, dtype=np.int16)
    highpass = np.abs(arr - arr_blur)
    hp_thr = float(np.percentile(highpass, 82))
    contrast_mask = highpass >= hp_thr

    # Text tends to have very bright fill and/or dark outline.
    lum = np.asarray(denoise, dtype=np.uint8)
    lum_mask = (lum >= 170) | (lum <= 95)
    mask = contrast_mask & lum_mask

    # Tiny 3x3 close operation via neighborhood count.
    m = mask.astype(np.uint8)
    padded = np.pad(m, 1, mode="constant", constant_values=0)
    neigh = (
        padded[0:-2, 0:-2]
        + padded[0:-2, 1:-1]
        + padded[0:-2, 2:]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, 0:-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    cleaned = neigh >= 3
    return cleaned.astype(np.uint8)


def mask_presence_score(mask: np.ndarray) -> float:
    return float(mask.mean())


def frame_text_change_score(prev_mask: np.ndarray, curr_mask: np.ndarray) -> float:
    xor = np.logical_xor(prev_mask > 0, curr_mask > 0).sum()
    union = np.logical_or(prev_mask > 0, curr_mask > 0).sum()
    if union == 0:
        return 0.0
    return float(xor / union)


def extract_text_features(gray: np.ndarray, mode: str = "text") -> tuple[np.ndarray, float]:
    mask = _to_text_mask(gray, mode=mode)
    return mask, mask_presence_score(mask)


@dataclass
class FrameMetric:
    frame_index: int
    timestamp: float
    diff: float
    presence: float
    name_diff: float
    name_presence: float
    marker_diff: float
    marker_presence: float
    dialog_path: Path
    name_path: Path | None = None


class MarkerTemplateMatcher:
    def __init__(
        self,
        template_paths: str | Path | Iterable[str | Path],
        center_width: int | None = None,
        vertical_shift_px: int = 0,
        vertical_shift_step: int = 1,
        horizontal_shift_px: int = 0,
        horizontal_shift_step: int = 1,
        shift_mode: str = "vertical",
    ) -> None:
        if isinstance(template_paths, (str, Path)):
            paths = [Path(template_paths)]
        else:
            paths = [Path(p) for p in template_paths]
        if not paths:
            raise ValueError("marker template list is empty")

        self.center_width = center_width
        self.vertical_shift_px = max(0, int(vertical_shift_px))
        self.vertical_shift_step = max(1, int(vertical_shift_step))
        self.horizontal_shift_px = max(0, int(horizontal_shift_px))
        self.horizontal_shift_step = max(1, int(horizontal_shift_step))
        mode = str(shift_mode).strip().lower()
        self.shift_mode = "horizontal" if mode == "horizontal" else "vertical"
        self.template_paths: list[Path] = []
        self.template_edges: list[np.ndarray] = []
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"marker template not found: {p}")
            tpl = Image.open(p).convert("L")
            tpl_gray = np.asarray(tpl, dtype=np.uint8)
            tpl_gray = self._crop_center_width(tpl_gray)
            if cv2 is not None:
                tpl_edge = cv2.Canny(tpl_gray, 60, 160)
            else:
                tpl_edge = tpl_gray
            self.template_paths.append(p)
            self.template_edges.append(tpl_edge)
        if not self.template_edges:
            raise ValueError("no valid marker templates loaded")

    def _crop_center_width(self, img: np.ndarray) -> np.ndarray:
        if self.center_width is None or self.center_width <= 0:
            return img
        h, w = img.shape[:2]
        cw = min(self.center_width, w)
        x1 = (w - cw) // 2
        x2 = x1 + cw
        return img[:, x1:x2]

    def score(self, roi_gray: np.ndarray) -> float:
        img_base = self._crop_center_width(roi_gray)
        best_score = 0.0
        if self.shift_mode == "horizontal":
            h_shifts = [0]
            if self.horizontal_shift_px > 0:
                for dx in range(
                    self.horizontal_shift_step, self.horizontal_shift_px + 1, self.horizontal_shift_step
                ):
                    h_shifts.append(-dx)
                    h_shifts.append(dx)
            for dx in h_shifts:
                img = self._shift_horizontal(img_base, dx)
                for tpl in self.template_edges:
                    score = self._score_single(img, tpl)
                    if score > best_score:
                        best_score = score
            return best_score

        v_shifts = [0]
        if self.vertical_shift_px > 0:
            for dy in range(self.vertical_shift_step, self.vertical_shift_px + 1, self.vertical_shift_step):
                v_shifts.append(-dy)
                v_shifts.append(dy)
        for dy in v_shifts:
            img = self._shift_vertical(img_base, dy)
            for tpl in self.template_edges:
                score = self._score_single(img, tpl)
                if score > best_score:
                    best_score = score
        return best_score

    @property
    def template_count(self) -> int:
        return len(self.template_edges)

    def _score_single(self, img: np.ndarray, tpl: np.ndarray) -> float:
        if cv2 is None:
            # Fallback: normalized overlap by resized absolute diff.
            h, w = img.shape[:2]
            th, tw = tpl.shape[:2]
            if th > h or tw > w:
                ratio = min(h / max(th, 1), w / max(tw, 1))
                nh = max(1, int(th * ratio))
                nw = max(1, int(tw * ratio))
                tpl = np.asarray(
                    Image.fromarray(tpl).resize((nw, nh), Image.Resampling.BICUBIC),
                    dtype=np.uint8,
                )
                th, tw = tpl.shape[:2]
            crop = img[:th, :tw].astype(np.int16)
            diff = np.abs(crop - tpl.astype(np.int16)).mean() / 255.0
            return float(max(0.0, 1.0 - diff))

        edge = cv2.Canny(img, 60, 160)
        ih, iw = edge.shape[:2]
        th, tw = tpl.shape[:2]
        if th > ih or tw > iw:
            ratio = min(ih / max(th, 1), iw / max(tw, 1))
            nh = max(4, int(th * ratio))
            nw = max(4, int(tw * ratio))
            tpl = cv2.resize(tpl, (nw, nh), interpolation=cv2.INTER_AREA)
            th, tw = tpl.shape[:2]

        if th > ih or tw > iw:
            return 0.0
        res = cv2.matchTemplate(edge, tpl, cv2.TM_CCOEFF_NORMED)
        score = float(np.max(res)) if res.size else 0.0
        return max(0.0, min(1.0, score))

    def _shift_vertical(self, img: np.ndarray, dy: int) -> np.ndarray:
        if dy == 0:
            return img
        h, w = img.shape[:2]
        out = np.empty_like(img)
        if dy > 0:
            out[:dy, :] = img[0:1, :]
            out[dy:, :] = img[: h - dy, :]
            return out
        k = -dy
        out[h - k :, :] = img[h - 1 : h, :]
        out[: h - k, :] = img[k:, :]
        return out

    def _shift_horizontal(self, img: np.ndarray, dx: int) -> np.ndarray:
        if dx == 0:
            return img
        h, w = img.shape[:2]
        out = np.empty_like(img)
        if dx > 0:
            out[:, :dx] = img[:, 0:1]
            out[:, dx:] = img[:, : w - dx]
            return out
        k = -dx
        out[:, w - k :] = img[:, w - 1 : w]
        out[:, : w - k] = img[:, k:]
        return out
