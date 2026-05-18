from __future__ import annotations

from pathlib import Path
from typing import Any

from .datatypes import DialogueSegment


_DEFAULT_STYLE: dict[str, Any] = {
    "font_name": "Microsoft YaHei",
    "font_size_scale": 0.8,
    "min_font_size": 20,
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


def write_ass(
    segments: list[DialogueSegment],
    path: str | Path,
    video_width: int,
    video_height: int,
    subtitle_location: list[int],
    title_translation_location: list[int] | None = None,
    title_info_location: list[int] | None = None,
    style: dict[str, Any] | None = None,
    *,
    dialogue_height: int = 0,
    title_height: int = 0,
    debug_overlay_segments: list[DialogueSegment] | None = None,
) -> None:
    s = _merge_style(style)
    font_name = str(s["font_name"])
    scale = float(s.get("font_size_scale", 0.8))
    if dialogue_height > 0:
        font_size = max(int(s.get("min_font_size", 20)), int(dialogue_height / 2 * scale))
    else:
        font_size = max(int(s.get("min_font_size", 20)), int(video_height * float(s.get("font_size_ratio", 0.05))))
    title_font_size = int(title_height * scale) if title_height > 0 else font_size

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
    debug_cfg = s.get("debug_overlay", {})
    if not isinstance(debug_cfg, dict):
        debug_cfg = {}
    debug_overlay_enabled = bool(debug_cfg.get("enabled", True))
    debug_font_name = str(debug_cfg.get("font_name", font_name))
    debug_font_size = max(12, int(font_size * float(debug_cfg.get("font_size_scale", 1.0))))
    debug_primary = str(debug_cfg.get("primary_colour", "&H0000FFFF"))
    debug_outline_c = str(debug_cfg.get("outline_colour", "&H00000000"))
    debug_back_c = str(debug_cfg.get("back_colour", back_c))
    debug_outline_w = float(debug_cfg.get("outline", 2.0))
    debug_shadow_w = float(debug_cfg.get("shadow", 0.5))
    debug_top_margin = int(debug_cfg.get("top_margin", 40))

    style_override_keys = {
        "font_name",
        "font_size",
        "font_size_scale",
        "primary_colour",
        "secondary_colour",
        "outline_colour",
        "back_colour",
        "outline",
        "shadow",
        "bold",
        "italic",
        "underline",
        "strike_out",
        "scale_x",
        "scale_y",
        "spacing",
        "angle",
        "border_style",
        "margin_l",
        "margin_r",
        "margin_v",
    }

    def _entry_style(seg: DialogueSegment) -> dict[str, Any]:
        return seg.subtitle_style if isinstance(seg.subtitle_style, dict) else {}

    def _get(entry_style: dict[str, Any], key: str, default: Any) -> Any:
        value = entry_style.get(key)
        return default if value is None or value == "" else value

    def _entry_font_size(entry_style: dict[str, Any]) -> int:
        raw_font_size = entry_style.get("font_size")
        if raw_font_size is not None and str(raw_font_size).strip() != "":
            return max(1, int(float(raw_font_size)))
        raw_scale = entry_style.get("font_size_scale")
        if raw_scale is not None and str(raw_scale).strip() != "":
            return max(1, int(font_size * float(raw_scale)))
        return font_size

    def _entry_style_line(style_name: str, entry_style: dict[str, Any]) -> str:
        entry_font = _entry_font_size(entry_style)
        return (
            f"Style: {style_name},{_get(entry_style, 'font_name', font_name)},{entry_font},"
            f"{_get(entry_style, 'primary_colour', primary)},{_get(entry_style, 'secondary_colour', secondary)},"
            f"{_get(entry_style, 'outline_colour', outline_c)},{_get(entry_style, 'back_colour', back_c)},"
            f"{int(_get(entry_style, 'bold', b))},{int(_get(entry_style, 'italic', i))},"
            f"{int(_get(entry_style, 'underline', u))},{int(_get(entry_style, 'strike_out', st))},"
            f"{int(_get(entry_style, 'scale_x', sx))},{int(_get(entry_style, 'scale_y', sy))},"
            f"{int(_get(entry_style, 'spacing', sp))},{int(_get(entry_style, 'angle', a))},"
            f"{int(_get(entry_style, 'border_style', border))},{float(_get(entry_style, 'outline', outline_w))},"
            f"{float(_get(entry_style, 'shadow', shadow_w))},2,"
            f"{int(_get(entry_style, 'margin_l', ml))},{int(_get(entry_style, 'margin_r', mr))},"
            f"{int(_get(entry_style, 'margin_v', mv))},1"
        )

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
    ]
    if debug_overlay_segments and debug_overlay_enabled:
        header.append(
            f"Style: DebugOverlay,{debug_font_name},{debug_font_size},{debug_primary},{secondary},"
            f"{debug_outline_c},{debug_back_c},0,0,0,0,100,100,0,0,1,"
            f"{debug_outline_w},{debug_shadow_w},8,{ml},{mr},{mv},1"
        )
    segment_style_names: dict[int, str] = {}
    for idx, seg in enumerate(segments):
        entry_style = _entry_style(seg)
        if not any(k in entry_style for k in style_override_keys):
            continue
        style_name = f"Entry_{idx + 1}"
        segment_style_names[idx] = style_name
        header.append(_entry_style_line(style_name, entry_style))
    header.extend(["", "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"])
    events: list[str] = []

    for idx, seg in enumerate(segments):
        entry_style = _entry_style(seg)
        pos = entry_style.get("position")
        event_anchor_x = anchor_x
        event_anchor_y = anchor_y
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            event_anchor_x = int(pos[0])
            event_anchor_y = int(pos[1])
        alignment = int(_get(entry_style, "alignment", 1))
        layer = int(_get(entry_style, "layer", 0))
        style_name = segment_style_names.get(idx, "Default")
        raw = seg.translation_subtitle or seg.text_original
        raw = _format_multiline_bracket_indent(raw, ass_mode=True)
        text = raw.replace("\n", "\\N")
        is_title = str(seg.dialogue_type or "").strip().lower() == "title"
        if is_title:
            title_override = f"{{\\an5\\pos({title_anchor_x},{title_anchor_y})\\fs{title_font_size}}}"
            speaker_text = str(seg.speaker or "").replace("\n", "\\N").strip()
            if text.strip():
                events.append(
                    f"Dialogue: {layer},"
                    f"{_ass_time(seg.time_start)},{_ass_time(seg.time_end)},{style_name},,0,0,0,,"
                    f"{title_override}{text}"
                )
            if speaker_text:
                speaker_override = f"{{\\an5\\pos({speaker_anchor_x},{speaker_anchor_y})\\fs{title_font_size}}}"
                events.append(
                    f"Dialogue: {layer},"
                    f"{_ass_time(seg.time_start)},{_ass_time(seg.time_end)},{style_name},,0,0,0,,"
                    f"{speaker_override}{speaker_text}"
                )
            continue
        if text.strip():
            events.append(
                f"Dialogue: {layer},"
                f"{_ass_time(seg.time_start)},{_ass_time(seg.time_end)},{style_name},,0,0,0,,"
                f"{{\\an{alignment}\\pos({event_anchor_x},{event_anchor_y})}}{text}"
            )
    if debug_overlay_segments and debug_overlay_enabled:
        debug_anchor_x = int(round(video_width / 2.0))
        debug_anchor_y = max(0, debug_top_margin)
        for seg in debug_overlay_segments:
            raw = str(seg.translation_subtitle or seg.text_original or "").strip()
            if not raw:
                continue
            text = raw.replace("\n", "\\N")
            events.append(
                "Dialogue: 1,"
                f"{_ass_time(seg.time_start)},{_ass_time(seg.time_end)},DebugOverlay,,0,0,0,,"
                f"{{\\an8\\pos({debug_anchor_x},{debug_anchor_y})}}{text}"
            )
    path.write_text("\n".join(header + events), encoding="utf-8")
