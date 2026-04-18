from __future__ import annotations

from pathlib import Path

from .models import DialogueSegment


def _srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    hh = ms // 3600000
    ms -= hh * 3600000
    mm = ms // 60000
    ms -= mm * 60000
    ss = ms // 1000
    ms -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _ass_time(sec: float) -> str:
    cs = int(round(sec * 100))
    hh = cs // 360000
    cs -= hh * 360000
    mm = cs // 6000
    cs -= mm * 6000
    ss = cs // 100
    cs -= ss * 100
    return f"{hh:d}:{mm:02d}:{ss:02d}.{cs:02d}"


def write_srt(segments: list[DialogueSegment], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    idx = 1
    for seg in segments:
        text = seg.translation_subtitle or seg.text_original
        if str(seg.dialogue_type or "").strip().lower() == "title":
            speaker_text = str(seg.speaker or "").strip()
            lines_out: list[str] = []
            if text.strip():
                lines_out.append(text)
            if speaker_text:
                lines_out.append(speaker_text)
            if not lines_out:
                continue
            text = "\n".join(lines_out)
        if not text.strip():
            continue
        lines.extend(
            [
                str(idx),
                f"{_srt_time(seg.time_start)} --> {_srt_time(seg.time_end)}",
                text,
                "",
            ]
        )
        idx += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ass(
    segments: list[DialogueSegment],
    path: str | Path,
    video_width: int,
    video_height: int,
    subtitle_location: list[int],
    title_translation_location: list[int] | None = None,
    title_info_location: list[int] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = subtitle_location
    anchor_x = x1
    anchor_y = y2
    if title_translation_location is None:
        title_translation_location = subtitle_location
    if title_info_location is None:
        title_info_location = subtitle_location
    tx1, ty1, tx2, ty2 = title_translation_location
    sx1, sy1, sx2, sy2 = title_info_location
    title_anchor_x = int(round((tx1 + tx2) / 2.0))
    title_anchor_y = int(round((ty1 + ty2) / 2.0))
    speaker_anchor_x = int(round((sx1 + sx2) / 2.0))
    speaker_anchor_y = int(round((sy1 + sy2) / 2.0))
    font_size = max(26, int(video_height * 0.045))
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Default,Microsoft YaHei,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "0,0,0,0,100,100,0,0,1,2.2,0.8,2,40,40,40,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    events: list[str] = []
    for seg in segments:
        text = (seg.translation_subtitle or seg.text_original).replace("\n", "\\N")
        is_title = str(seg.dialogue_type or "").strip().lower() == "title"
        if is_title:
            speaker_text = str(seg.speaker or "").replace("\n", "\\N").strip()
            if text.strip():
                events.append(
                    "Dialogue: 0,"
                    f"{_ass_time(seg.time_start)},{_ass_time(seg.time_end)},Default,,0,0,0,,"
                    f"{{\\an5\\pos({title_anchor_x},{title_anchor_y})}}{text}"
                )
            if speaker_text:
                events.append(
                    "Dialogue: 0,"
                    f"{_ass_time(seg.time_start)},{_ass_time(seg.time_end)},Default,,0,0,0,,"
                    f"{{\\an5\\pos({speaker_anchor_x},{speaker_anchor_y})}}{speaker_text}"
                )
            continue
        if text.strip():
            events.append(
                "Dialogue: 0,"
                f"{_ass_time(seg.time_start)},{_ass_time(seg.time_end)},Default,,0,0,0,,"
                f"{{\\an1\\pos({anchor_x},{anchor_y})}}{text}"
            )
    path.write_text("\n".join(header + events), encoding="utf-8")
