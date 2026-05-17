from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .event_detect import extract_text_mask_stats, load_gray
from .ocr_engines import build_ocr_engine


class NameOcrRunner:
    """Name ROI OCR runner with optional multi-thread parallelism."""

    def __init__(self, ocr_cfg: dict[str, Any], fallback_engine: Any | None, workers: int = 1) -> None:
        self._ocr_cfg = ocr_cfg
        self._fallback_engine = fallback_engine
        self._workers = max(1, int(workers))
        self._presence_mode = str(ocr_cfg.get("name_presence_mode", "fast_mask")).lower()
        self._mask_thr_on = float(ocr_cfg.get("name_presence_threshold_on", 0.018))
        self._mask_thr_off = float(ocr_cfg.get("name_presence_threshold_off", 0.012))
        self._mask_min_components = int(ocr_cfg.get("name_presence_min_components", 2))
        self._mask_max_components = int(ocr_cfg.get("name_presence_max_components", 120))
        self._mask_max_x_min_ratio = float(ocr_cfg.get("name_presence_max_x_min_ratio", 0.34))
        self._mask_max_y_span_ratio = float(ocr_cfg.get("name_presence_max_y_span_ratio", 0.62))
        self._mask_min_aspect_ratio = float(ocr_cfg.get("name_presence_min_aspect_ratio", 0.85))
        self._mask_max_largest_ratio = float(ocr_cfg.get("name_presence_max_largest_component_ratio", 0.82))
        self._use_ocr_fallback = bool(ocr_cfg.get("name_presence_ocr_fallback", True))
        self._pool: ThreadPoolExecutor | None = None
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self._stats = {
            "total": 0,
            "mask_total": 0,
            "mask_only": 0,
            "mask_on": 0,
            "mask_off": 0,
            "mask_uncertain": 0,
            "mask_rejected_shape": 0,
            "mask_score_sum": 0.0,
            "mask_score_min": None,
            "mask_score_max": None,
            "ocr_fallback": 0,
            "ocr_verify": 0,
        }
        if self._workers > 1:
            self._pool = ThreadPoolExecutor(max_workers=self._workers)

    @property
    def workers(self) -> int:
        return self._workers

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _thread_engine(self) -> Any:
        eng = getattr(self._local, "engine", None)
        if eng is None:
            eng = build_ocr_engine(self._ocr_cfg)
            self._local.engine = eng
        return eng

    def _inc_stat(self, key: str) -> None:
        with self._stats_lock:
            self._stats["total"] += 1
            self._stats[key] += 1

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return dict(self._stats)

    def _inc_mask_decision(self, decision: str, score: float) -> None:
        with self._stats_lock:
            self._stats["mask_total"] += 1
            self._stats["mask_only"] += 1
            self._stats[decision] = self._stats.get(decision, 0) + 1
            self._stats["mask_score_sum"] = float(self._stats.get("mask_score_sum", 0.0) or 0.0) + float(score)
            cur_min = self._stats.get("mask_score_min")
            cur_max = self._stats.get("mask_score_max")
            self._stats["mask_score_min"] = float(score) if cur_min is None else min(float(cur_min), float(score))
            self._stats["mask_score_max"] = float(score) if cur_max is None else max(float(cur_max), float(score))

    def _mask_shape_reject_reason(
        self,
        component_count: int,
        x_min: float,
        x_span: float,
        y_span: float,
        largest_ratio: float,
    ) -> str:
        if component_count < self._mask_min_components:
            return "few_components"
        if component_count > self._mask_max_components:
            return "too_many_components"
        if x_min > self._mask_max_x_min_ratio:
            return "right_side_only"
        if y_span > self._mask_max_y_span_ratio:
            return "wide_y_span"
        if (x_span / max(1e-6, y_span)) < self._mask_min_aspect_ratio:
            return "not_horizontal_line"
        if largest_ratio > self._mask_max_largest_ratio:
            return "large_single_component"
        return ""

    def _mask_detail_from_gray(self, gray: np.ndarray) -> tuple[bool, bool, float, dict[str, Any]]:
        _mask, st = extract_text_mask_stats(gray, mode="name")
        score = float(st.presence)
        meta: dict[str, Any] = {
            "presence": score,
            "components": int(st.component_count),
            "x_min": float(st.x_min_ratio),
            "x_max": float(st.x_max_ratio),
            "x_span": float(st.x_span_ratio),
            "y_span": float(st.y_span_ratio),
            "largest": float(st.largest_component_ratio),
            "reject_reason": "",
        }
        reject_reason = self._mask_shape_reject_reason(
            int(st.component_count),
            float(st.x_min_ratio),
            float(st.x_span_ratio),
            float(st.y_span_ratio),
            float(st.largest_component_ratio),
        )
        if score <= self._mask_thr_off:
            self._inc_mask_decision("mask_off", score)
            return False, False, score, meta
        if reject_reason:
            meta["reject_reason"] = reject_reason
            self._inc_mask_decision("mask_rejected_shape", score)
            return False, True, score, meta
        if score >= self._mask_thr_on:
            self._inc_mask_decision("mask_on", score)
            return True, False, score, meta
        self._inc_mask_decision("mask_uncertain", score)
        return False, True, score, meta

    def _has_text_fast_mask(self, image_path: Path) -> bool | None:
        gray = load_gray(image_path)
        present, uncertain, _score, _meta = self._mask_detail_from_gray(gray)
        if present:
            return True
        if not uncertain:
            return False
        return None

    def has_text(self, image_path: Path) -> bool:
        if not image_path.exists():
            return False
        if self._presence_mode == "fast_mask":
            fast = self._has_text_fast_mask(image_path)
            if fast is not None:
                return fast
            if not self._use_ocr_fallback:
                # Uncertain band defaults to "no name" when OCR fallback is disabled.
                return False
            self._inc_stat("ocr_fallback")
        else:
            self._inc_stat("ocr_fallback")
        engine = self._fallback_engine if self._pool is None else self._thread_engine()
        if engine is None:
            return False
        r = engine.recognize(image_path)
        return bool((r.text or "").strip())

    def has_text_mask(self, image_path: Path) -> bool:
        """Mask-only name presence check (no OCR fallback)."""
        present, _uncertain, _presence = self.has_text_mask_detail(image_path)
        return present

    def has_text_mask_detail(self, image_path: Path) -> tuple[bool, bool, float]:
        """Mask-only name presence with uncertainty and raw presence score."""
        present, uncertain, score, _meta = self.has_text_mask_detail_meta(image_path)
        return present, uncertain, score

    def has_text_mask_detail_meta(self, image_path: Path) -> tuple[bool, bool, float, dict[str, Any]]:
        """Mask-only name presence with diagnostics used by debug/stat output."""
        if not image_path.exists():
            return False, False, 0.0, {
                "presence": 0.0,
                "components": 0,
                "x_min": 0.0,
                "x_max": 0.0,
                "x_span": 0.0,
                "y_span": 0.0,
                "largest": 0.0,
                "reject_reason": "missing",
            }
        gray = load_gray(image_path)
        return self._mask_detail_from_gray(gray)

    def has_text_ocr(self, image_path: Path) -> bool:
        if not image_path.exists():
            return False
        with self._stats_lock:
            self._stats["ocr_verify"] += 1
        engine = self._fallback_engine if self._pool is None else self._thread_engine()
        if engine is None:
            return False
        r = engine.recognize(image_path)
        text = (r.text or "").strip()
        # _log(f"[has_text_ocr] {image_path.name} text={text!r}")

        text_compact = "".join(ch for ch in text if not ch.isspace())

        noisy_chars = set("-1|lI")
        filtered = "".join(ch for ch in text_compact if ch not in noisy_chars)

        return len(filtered) >= 2

    def has_text_batch(self, image_paths: list[Path]) -> list[bool]:
        if not image_paths:
            return []
        if self._pool is None or len(image_paths) <= 1:
            return [self.has_text(p) for p in image_paths]
        out = [False] * len(image_paths)
        fut_to_idx: dict[Future[bool], int] = {}
        for i, p in enumerate(image_paths):
            fut = self._pool.submit(self.has_text, p)
            fut_to_idx[fut] = i
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            out[i] = bool(fut.result())
        return out

    def has_text_batch_ocr(self, image_paths: list[Path]) -> list[bool]:
        if not image_paths:
            return []
        if self._pool is None or len(image_paths) <= 1:
            return [self.has_text_ocr(p) for p in image_paths]
        out = [False] * len(image_paths)
        fut_to_idx: dict[Future[bool], int] = {}
        for i, p in enumerate(image_paths):
            fut = self._pool.submit(self.has_text_ocr, p)
            fut_to_idx[fut] = i
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            out[i] = bool(fut.result())
        return out
