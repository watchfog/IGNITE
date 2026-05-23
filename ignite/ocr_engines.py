from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .datatypes import OcrResult


_RAPIDOCR_LOG_FILTER_INSTALLED = False


class _RapidOcrEmptyDetWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # This warning is expected for blank frames and is too noisy for this pipeline.
        if "text detection result is empty" in msg.lower():
            return False
        return True


def _install_rapidocr_log_filter() -> None:
    global _RAPIDOCR_LOG_FILTER_INSTALLED
    if _RAPIDOCR_LOG_FILTER_INSTALLED:
        return
    try:
        logger = logging.getLogger("RapidOCR")
        logger.addFilter(_RapidOcrEmptyDetWarningFilter())
        _RAPIDOCR_LOG_FILTER_INSTALLED = True
    except Exception:
        # Never block OCR initialization due to log filter setup.
        pass


class OcrEngine(ABC):
    @abstractmethod
    def recognize(self, image_path: str | Path) -> OcrResult:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> dict[str, Any]:
        raise NotImplementedError


def _load_for_ocr(image_path: str | Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _add_black_border_for_ocr(img: np.ndarray, border_ratio: float, min_border_px: int) -> np.ndarray:
    h, w = img.shape[:2]
    ratio = max(0.0, float(border_ratio))
    min_px = max(0, int(min_border_px))

    border_x = max(min_px, int(round(w * ratio / 2.0)))
    border_y = max(min_px, int(round(h * ratio / 2.0)))
    if border_x <= 0 and border_y <= 0:
        return img

    return np.pad(
        img,
        ((border_y, border_y), (border_x, border_x), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def _normalize_box_pts(box: Any) -> list[tuple[float, float]]:
    if hasattr(box, "tolist"):
        try:
            box = box.tolist()
        except Exception:
            pass
    if not isinstance(box, (list, tuple)):
        return []
    pts: list[tuple[float, float]] = []
    for pt in box:
        if hasattr(pt, "tolist"):
            try:
                pt = pt.tolist()
            except Exception:
                return []
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return []
        try:
            pts.append((float(pt[0]), float(pt[1])))
        except (ValueError, TypeError):
            return []
    return pts


def _sort_ocr_texts_ltr_topdown(
    txts: list[str],
    scores: list[float],
    boxes: Any,
) -> tuple[list[str], list[float], list[int]]:
    raw_boxes: list[Any] = []
    if hasattr(boxes, "tolist"):
        raw_boxes = list(boxes.tolist())
    elif isinstance(boxes, (list, tuple)):
        raw_boxes = list(boxes)
    else:
        return txts, scores, [len(txts)]

    if len(raw_boxes) <= 1 or len(raw_boxes) != len(txts):
        return txts, scores, [len(txts)]

    items: list[tuple[float, float, float, int]] = []
    for i, box in enumerate(raw_boxes):
        pts = _normalize_box_pts(box)
        if len(pts) < 2:
            return txts, scores, [len(txts)]
        ys = [p[1] for p in pts]
        xs = [p[0] for p in pts]
        items.append((min(ys), max(ys), sum(xs) / len(xs), i))

    items.sort(key=lambda it: it[0])

    rows: list[list[tuple[float, float, float, int]]] = []
    current_row = [items[0]]
    current_bottom = items[0][1]
    for it in items[1:]:
        min_y, max_y, _, _ = it
        if min_y < current_bottom:
            current_row.append(it)
            if max_y > current_bottom:
                current_bottom = max_y
        else:
            rows.append(current_row)
            current_row = [it]
            current_bottom = max_y
    rows.append(current_row)

    ordered: list[int] = []
    row_lengths: list[int] = []
    for row in rows:
        row.sort(key=lambda it: it[2])
        ordered.extend(idx for _, _, _, idx in row)
        row_lengths.append(len(row))

    new_txts = [txts[i] for i in ordered]
    new_scores = [scores[i] for i in ordered]
    return new_txts, new_scores, row_lengths


class RapidOcrEngine(OcrEngine):
    def __init__(
        self,
        rec_lang: str = "japan",
        backend: str = "cpu",
        disable_env_proxy: bool = True,
        model_root_dir: str | None = None,
        input_border_ratio: float = 0.5,
        input_border_min_px: int = 40,
        rec_only: bool = False,
        model_type: str = "mobile",
        provider: str = "cpu",
    ) -> None:
        _install_rapidocr_log_filter()
        self._provider = str(provider or "cpu").strip().lower()
        self._requested_lang = rec_lang
        self._target_lang = "ch"
        self._input_border_ratio = max(0.0, float(input_border_ratio))
        self._input_border_min_px = max(0, int(input_border_min_px))
        self._rec_only = bool(rec_only)
        self._model_type = "server" if str(model_type or "mobile").strip().lower() == "server" else "mobile"
        if disable_env_proxy:
            for k in [
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            ]:
                os.environ.pop(k, None)
        errors: list[str] = []
        try:
            from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR  # type: ignore

            params: dict[str, Any] = {
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.CH,
                "Det.model_type": ModelType.SERVER if self._model_type == "server" else ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.CH,
                "Rec.model_type": ModelType.SERVER if self._model_type == "server" else ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Cls.engine_type": EngineType.ONNXRUNTIME,
                "Cls.lang_type": LangDet.CH,
                "Cls.model_type": ModelType.SERVER if self._model_type == "server" else ModelType.MOBILE,
                "Cls.ocr_version": OCRVersion.PPOCRV5,
            }

            custom_model_root = str(model_root_dir or "").strip()
            if custom_model_root:
                params["Global.model_root_dir"] = str(Path(custom_model_root).resolve())

            if self._provider == "dml":
                params["EngineConfig.onnxruntime.use_dml"] = True
            elif self._provider == "cuda":
                params["EngineConfig.onnxruntime.use_cuda"] = True

            self._engine = RapidOCR(params=params)
            self._runtime = "rapidocr"
            return
        except Exception as exc:
            errors.append(f"rapidocr:{exc.__class__.__name__}")

        if True:  # pragma: no cover
            raise RuntimeError(
                "No usable RapidOCR runtime found. Install rapidocr, "
                f"and ensure local OCR model files are available. errors={errors}"
            )

    def recognize(self, image_path: str | Path) -> OcrResult:
        return self.recognize_array(_load_for_ocr(image_path))

    def recognize_array(self, img: np.ndarray) -> OcrResult:
        img = _add_black_border_for_ocr(
            img,
            border_ratio=self._input_border_ratio,
            min_border_px=self._input_border_min_px,
        )
        output = self._engine(
            img,
            use_det=not self._rec_only,
            use_cls=not self._rec_only,
        )
        return self._parse_ocr_output(output)

    def _parse_ocr_output(self, output: Any) -> OcrResult:
        if hasattr(output, "txts") and hasattr(output, "scores"):
            txts = list(getattr(output, "txts", []) or [])
            scores = [float(x) for x in (getattr(output, "scores", []) or [])]
            if not txts:
                return OcrResult(text="", confidence=0.0)
            boxes = getattr(output, "boxes", None)
            if boxes is not None:
                txts, scores, row_lengths = _sort_ocr_texts_ltr_topdown(txts, scores, boxes)
                conf = float(np.mean(scores)) if scores else 0.0
                if row_lengths and len(row_lengths) > 1:
                    parts: list[str] = []
                    idx = 0
                    for count in row_lengths:
                        parts.append("".join(txts[idx : idx + count]))
                        idx += count
                    text = "\n".join(p for p in parts if p).strip()
                else:
                    text = "".join(txts).strip()
                return OcrResult(text=text, confidence=conf)
            conf = float(np.mean(scores)) if scores else 0.0
            return OcrResult(text="".join(txts).strip(), confidence=conf)

        result = output[0] if isinstance(output, (tuple, list)) else output
        if not result:
            return OcrResult(text="", confidence=0.0)
        texts: list[str] = []
        confs: list[float] = []
        for item in result:
            texts.append(item[1])
            confs.append(float(item[2]))
        return OcrResult(text="".join(texts).strip(), confidence=float(np.mean(confs)))

    def info(self) -> dict[str, Any]:
        return {
            "engine": "rapidocr",
            "runtime": self._runtime,
            "target_lang": self._target_lang,
            "requested_lang": self._requested_lang,
            "input_border_ratio": self._input_border_ratio,
            "input_border_min_px": self._input_border_min_px,
            "rec_only": self._rec_only,
            "model_type": self._model_type,
            "provider": self._provider,
        }


def build_ocr_engine(cfg: dict[str, Any]) -> OcrEngine:
    engine = str(cfg.get("engine", "rapidocr")).lower()
    if engine == "rapidocr":
        rec_lang = str(cfg.get("rapidocr_rec_lang", "japan"))
        backend = str(cfg.get("backend", "cpu")).lower()
        return RapidOcrEngine(
            rec_lang=rec_lang,
            backend=backend,
            disable_env_proxy=bool(cfg.get("disable_env_proxy", True)),
            model_root_dir=cfg.get("rapidocr_model_root_dir"),
            input_border_ratio=float(cfg.get("input_border_ratio", 0.5)),
            input_border_min_px=int(cfg.get("input_border_min_px", 40)),
            rec_only=bool(cfg.get("rec_only", False)),
            model_type=str(cfg.get("rapidocr_model_type", "mobile") or "mobile").strip(),
            provider=str(cfg.get("rapidocr_provider", "cpu") or "cpu").strip(),
        )
    raise ValueError(f"Unsupported OCR engine: {engine}")
