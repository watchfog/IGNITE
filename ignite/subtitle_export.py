from __future__ import annotations

from pathlib import Path
from typing import Any

from .datatypes import DialogueSegment


_DEFAULT_STYLE: dict[str, Any] = {
    "font_name": "Microsoft YaHei",
    "font_size_ratio": 0.05,
    "min_font_size": 28,
    "primary_colour": "&H00FFFFFF",
    "secondary_colour": "&H000000FF",
    "outline_colour": "&H00000000",
    "back_colour": "&H64000000",
    "border_style": 1,
    "outline": 2.4,
    "shadow": 0.8,
    "margin_l": 40,
    "margin_r": 40,
    "margin_v": 40,
    "bold": 0,
    "italic": 0,
    "underline": 0,
    "strike_out": 0,
    "scale_x": 100,
    "scale_y": 100,
    "spacing": 0,
    "angle": 0,
}


def _merge_style(style: dict[str, Any] | None) -> dict[str, Any]:
    if not style:
        return dict(_DEFAULT_STYLE)
    out = dict(_DEFAULT_STYLE)
    out.update(style)
    return out


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


def _format_multiline_bracket_indent(text: str, *, ass_mode: bool = False) -> str:
    if "\n" not in text:
        return text
    lines = text.split("\n")
    if len(lines) < 2 or not lines[0]:
        return text
    fw = 0
    hw = 0
    for ch in lines[0]:
        if ch in "（【〔｟":
            fw += 1
        elif ch in "([":
            hw += 1
        else:
            break
    if fw == 0 and hw == 0:
        return text
    if ass_mode:
        indent = "\\h" * hw + "\u3000" * fw
    else:
        indent = " " * hw + "\u3000" * fw
    return "\n".join([lines[0]] + [indent + ln for ln in lines[1:]])


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
        text = _format_multiline_bracket_indent(text)
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
    style: dict[str, Any] | None = None,
) -> None:
    s = _merge_style(style)
    font_name = str(s["font_name"])
    font_size = max(int(s["min_font_size"]), int(video_height * float(s["font_size_ratio"])))
    primary = str(s["primary_colour"])
    secondary = str(s["secondary_colour"])
    outline_c = str(s["outline_colour"])
    back_c = str(s["back_colour"])
    border = int(s["border_style"])
    outline_w = float(s["outline"])
    shadow_w = float(s["shadow"])
    ml = int(s["margin_l"])
    mr = int(s["margin_r"])
    mv = int(s["margin_v"])
    b = int(s["bold"])
    i = int(s["italic"])
    u = int(s["underline"])
    st = int(s["strike_out"])
    sx = int(s["scale_x"])
    sy = int(s["scale_y"])
    sp = int(s["spacing"])
    a = int(s["angle"])

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
        f"Style: Default,{font_name},{font_size},{primary},{secondary},{outline_c},{back_c},"
        f"{b},{i},{u},{st},{sx},{sy},{sp},{a},{border},{outline_w},{shadow_w},2,{ml},{mr},{mv},1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    events: list[str] = []
    for seg in segments:
        raw = seg.translation_subtitle or seg.text_original
        raw = _format_multiline_bracket_indent(raw, ass_mode=True)
        text = raw.replace("\n", "\\N")
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
