from __future__ import annotations

from .models import OcrResult


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def fuse_ocr_candidates(candidates: list[OcrResult]) -> OcrResult:
    normalized = [
        OcrResult(text=_normalize_text(item.text), confidence=item.confidence)
        for item in candidates
        if _normalize_text(item.text)
    ]
    if not normalized:
        return OcrResult(text="", confidence=0.0)

    normalized.sort(key=lambda x: (len(x.text), x.confidence), reverse=True)
    top = normalized[0]

    # Keep longest stable string but reject obvious noise when confidence is very low.
    if top.confidence < 0.25 and len(top.text) <= 2:
        return OcrResult(text="", confidence=top.confidence)
    return top

