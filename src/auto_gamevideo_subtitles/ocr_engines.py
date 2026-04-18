from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .models import OcrResult


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


class RapidOcrEngine(OcrEngine):
    def __init__(
        self,
        rec_lang: str = "japan",
        backend: str = "cpu",
        disable_env_proxy: bool = True,
        model_root_dir: str | None = None,
        input_border_ratio: float = 0.5,
        input_border_min_px: int = 40,
    ) -> None:
        _install_rapidocr_log_filter()
        self._runtime = ""
        self._requested_lang = rec_lang
        self._target_lang = "ch"
        self._input_border_ratio = max(0.0, float(input_border_ratio))
        self._input_border_min_px = max(0, int(input_border_min_px))
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
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.CH,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Cls.engine_type": EngineType.ONNXRUNTIME,
                "Cls.lang_type": LangDet.CH,
                "Cls.model_type": ModelType.MOBILE,
                "Cls.ocr_version": OCRVersion.PPOCRV5,
            }

            custom_model_root = str(model_root_dir or "").strip()
            if custom_model_root:
                params["Global.model_root_dir"] = str(Path(custom_model_root).resolve())

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
        img = _load_for_ocr(image_path)
        img = _add_black_border_for_ocr(
            img,
            border_ratio=self._input_border_ratio,
            min_border_px=self._input_border_min_px,
        )
        output = self._engine(img)
        # rapidocr returns RapidOCROutput(txts=..., scores=...)
        if hasattr(output, "txts") and hasattr(output, "scores"):
            txts = list(getattr(output, "txts", []) or [])
            scores = [float(x) for x in (getattr(output, "scores", []) or [])]
            if not txts:
                return OcrResult(text="", confidence=0.0)
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
        )
    raise ValueError(f"Unsupported OCR engine: {engine}")
