from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    duration: float


@dataclass
class Roi:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_crop_filter(self) -> str:
        # exact=1 avoids chroma-subsampling alignment rounding (e.g. 43x81 -> 42x80).
        return f"crop=w={self.width}:h={self.height}:x={self.x1}:y={self.y1}:exact=1"


@dataclass
class OcrResult:
    text: str
    confidence: float


@dataclass
class DialogueSegment:
    segment_id: int
    frame_start: int
    frame_end: int
    time_start: float
    time_end: float
    speaker: str
    speaker_confidence: float
    text_original: str
    text_ocr_confidence: float
    translation_subtitle: str
    dialogue_type: str
    line_count_detected: int
    stable_frame_ids: list[int] = field(default_factory=list)
    keyframe_before: str = ""
    keyframe_stable: str = ""
    keyframe_after: str = ""
    needs_review: bool = False
    review_reason: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
