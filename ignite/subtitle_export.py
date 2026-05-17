from __future__ import annotations

import copy
import unicodedata
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


_NOISE_CHARS = set(" 　\n\r\t\u3000：:（）()「」『』【】\"'\"'｢｣“”‘’・．")


def _normalize_speaker_name(raw: str) -> str:
    text = unicodedata.normalize("NFKC", str(raw or ""))
    chars: list[str] = []
    for ch in text:
        if ch in _NOISE_CHARS:
            continue
        chars.append(ch)
    return "".join(chars)


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[-1] + 1
            dlt = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dlt, sub))
        prev = cur
    return prev[-1]


def _match_speaker_style(
    speaker_raw: str,
    speaker_styles: dict[str, dict[str, Any]],
    max_edit_distance: int,
) -> tuple[str | None, bool]:
    normalized = _normalize_speaker_name(speaker_raw)
    if not normalized:
        return None, False

    if normalized in speaker_styles:
        return normalized, False

    norm_equal: list[str] = []
    contains: list[str] = []
    for key in speaker_styles:
        nk = _normalize_speaker_name(key)
        if normalized == nk:
            norm_equal.append(key)
        elif normalized in nk or nk in normalized:
            contains.append(key)

    if len(norm_equal) == 1:
        return norm_equal[0], False
    if len(norm_equal) > 1:
        return None, True

    if len(contains) == 1:
        return contains[0], False
    if len(contains) > 1:
        return None, True

    best_dists: list[str] = []
    best_dist = max_edit_distance + 1
    for key in speaker_styles:
        nk = _normalize_speaker_name(key)
        dist = _edit_distance(normalized, nk)
        if dist < best_dist:
            best_dist = dist
            best_dists = [key]
        elif dist == best_dist:
            best_dists.append(key)

    if best_dist <= max_edit_distance:
        if len(best_dists) == 1:
            return best_dists[0], False
        return None, True

    return None, False


def _build_speaker_style_map(
    style: dict[str, Any],
    default_style_clone: dict[str, Any],
) -> tuple[dict[str, str], list[str]]:
    speaker_styles_raw: Any = style.get("speaker_styles", None)
    if not isinstance(speaker_styles_raw, dict):
        return {}, []

    matching_cfg: Any = style.get("speaker_style_matching")
    if isinstance(matching_cfg, dict) and not matching_cfg.get("enabled", True):
        return {}, []

    speaker_styles: dict[str, dict[str, Any]] = {}
    for k, v in speaker_styles_raw.items():
        if isinstance(v, dict) and k.strip():
            speaker_styles[k.strip()] = v

    style_lines: list[str] = []
    speaker_name_to_style_name: dict[str, str] = {}

    fmt = (
        "{font_name},{font_size},{primary},{secondary},{outline},{back},"
        "{bold},{italic},{underline},{strike_out},{scale_x},{scale_y},"
        "{spacing},{angle},{border},{outline_w},{shadow_w},2,{ml},{mr},{mv},1"
    )

    for speaker_name, overrides in speaker_styles.items():
        sp = copy.deepcopy(default_style_clone)
        for attr in (
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
            "margin_l",
            "margin_r",
            "margin_v",
        ):
            val = overrides.get(attr)
            if val is not None:
                sp[attr] = val

        style_name = f"Speaker_{speaker_name}"
        safe_name = style_name.replace(",", "").replace(";", "")
        speaker_name_to_style_name[speaker_name] = safe_name

        style_lines.append(
            f"Style: {safe_name},"
            + fmt.format(
                font_name=sp["font_name"],
                font_size=int(sp.get("font_size", default_style_clone.get("font_size", 20))),
                primary=sp["primary_colour"],
                secondary=sp["secondary_colour"],
                outline=sp["outline_colour"],
                back=sp["back_colour"],
                bold=int(sp.get("bold", 0)),
                italic=int(sp.get("italic", 0)),
                underline=int(sp.get("underline", 0)),
                strike_out=int(sp.get("strike_out", 0)),
                scale_x=int(sp.get("scale_x", 100)),
                scale_y=int(sp.get("scale_y", 100)),
                spacing=int(sp.get("spacing", 0)),
                angle=int(sp.get("angle", 0)),
                border=int(sp.get("border_style", 1)),
                outline_w=float(sp.get("outline", 2.4)),
                shadow_w=float(sp.get("shadow", 0.8)),
                ml=int(sp.get("margin_l", 40)),
                mr=int(sp.get("margin_r", 40)),
                mv=int(sp.get("margin_v", 40)),
            )
        )

    return speaker_name_to_style_name, style_lines


def mark_ambiguous_speaker_segments(
    segments: list[DialogueSegment],
    style: dict[str, Any] | None = None,
) -> None:
    if not style:
        return
    raw_styles = style.get("speaker_styles")
    if not isinstance(raw_styles, dict):
        return
    matching_cfg = style.get("speaker_style_matching")
    if isinstance(matching_cfg, dict) and not matching_cfg.get("enabled", True):
        return
    max_edit_distance = 1
    if isinstance(matching_cfg, dict):
        max_edit_distance = int(matching_cfg.get("max_edit_distance", 1))

    speaker_styles: dict[str, dict[str, Any]] = {}
    for k, v in raw_styles.items():
        if isinstance(v, dict) and k.strip():
            speaker_styles[k.strip()] = v
    if not speaker_styles:
        return

    for seg in segments:
        if not seg.speaker:
            continue
        dt = str(seg.dialogue_type or "").strip().lower()
        if dt == "title":
            continue
        matched, ambiguous = _match_speaker_style(seg.speaker, speaker_styles, max_edit_distance)
        if ambiguous:
            seg.needs_review = True
            if "speaker_ambiguous" not in seg.review_reason:
                seg.review_reason.append("speaker_ambiguous")


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
    *,
    dialogue_height: int = 0,
    title_height: int = 0,
) -> None:
    s = _merge_style(style)
    font_name = str(s["font_name"])
    scale = float(s.get("font_size_scale", 0.8))
    if dialogue_height > 0:
        font_size = max(int(s.get("min_font_size", 20)), int(dialogue_height / 2 * scale))
    else:
        font_size = max(int(s.get("min_font_size", 20)), int(video_height * float(s.get("font_size_ratio", 0.05))))
    title_font_size = int(title_height * scale) if title_height > 0 else font_size

    default_style_clone = dict(s)
    default_style_clone["font_size"] = font_size
    speaker_name_to_style_name, speaker_style_lines = _build_speaker_style_map(s, default_style_clone)
    max_edit_distance = 1
    matching_cfg = s.get("speaker_style_matching", {})
    if isinstance(matching_cfg, dict):
        max_edit_distance = int(matching_cfg.get("max_edit_distance", 1))

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
    ]
    header.extend(speaker_style_lines)
    header.extend(["", "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"])
    events: list[str] = []

    speaker_styles: dict[str, dict[str, Any]] = {}
    raw_styles = s.get("speaker_styles", {})
    if isinstance(raw_styles, dict):
        speaker_styles = {k.strip(): v for k, v in raw_styles.items() if isinstance(v, dict) and k.strip()}

    for seg in segments:
        raw = seg.translation_subtitle or seg.text_original
        raw = _format_multiline_bracket_indent(raw, ass_mode=True)
        text = raw.replace("\n", "\\N")
        is_title = str(seg.dialogue_type or "").strip().lower() == "title"
        if is_title:
            title_override = f"{{\\an5\\pos({title_anchor_x},{title_anchor_y})\\fs{title_font_size}}}"
            speaker_text = str(seg.speaker or "").replace("\n", "\\N").strip()
            if text.strip():
                events.append(
                    "Dialogue: 0,"
                    f"{_ass_time(seg.time_start)},{_ass_time(seg.time_end)},Default,,0,0,0,,"
                    f"{title_override}{text}"
                )
            if speaker_text:
                speaker_override = f"{{\\an5\\pos({speaker_anchor_x},{speaker_anchor_y})\\fs{title_font_size}}}"
                events.append(
                    "Dialogue: 0,"
                    f"{_ass_time(seg.time_start)},{_ass_time(seg.time_end)},Default,,0,0,0,,"
                    f"{speaker_override}{speaker_text}"
                )
            continue
        style_name = "Default"
        if speaker_name_to_style_name and seg.speaker:
            matched, ambiguous = _match_speaker_style(seg.speaker, speaker_styles, max_edit_distance)
            if matched and not ambiguous:
                style_name = speaker_name_to_style_name[matched]
        if text.strip():
            events.append(
                "Dialogue: 0,"
                f"{_ass_time(seg.time_start)},{_ass_time(seg.time_end)},{style_name},,0,0,0,,"
                f"{{\\an1\\pos({anchor_x},{anchor_y})}}{text}"
            )
    path.write_text("\n".join(header + events), encoding="utf-8")
