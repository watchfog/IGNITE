from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib import error as urlerror

from .log_utils import _log
from urllib import request


DEFAULT_RESPONSES_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_CHAT_COMPLETIONS_BASE_URL = DEFAULT_RESPONSES_BASE_URL
TEXT_EXTRACTION_PROFILE_MODE = "vlm_text_extraction"


@dataclass(frozen=True)
class TranslationModelProfile:
    name: str
    model: str
    base_url: str
    api_key: str = ""
    api_key_file: str = ""


def resolve_responses_base_url(cfg: dict[str, Any]) -> str:
    for key in ("responses_base_url", "vlm_api", "base_url"):
        value = str(cfg.get(key, "") or "").strip()
        if value:
            return value
    return DEFAULT_RESPONSES_BASE_URL


def resolve_chat_completions_base_url(cfg: dict[str, Any]) -> str:
    for key in ("chat_completions_base_url", "responses_base_url", "vlm_api", "base_url"):
        value = str(cfg.get(key, "") or "").strip()
        if value:
            return value
    return DEFAULT_CHAT_COMPLETIONS_BASE_URL


def _profile_base_url(cfg: dict[str, Any], mode: str) -> str:
    if str(mode or "").strip().lower() == "ocr_chat_completions":
        return resolve_chat_completions_base_url(cfg)
    return resolve_responses_base_url(cfg)


def _list_model_names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _profile_supports_mode(raw_profile: Any, mode: str) -> bool:
    if not isinstance(raw_profile, dict):
        return True
    allowed = raw_profile.get("modes")
    if allowed is None:
        return True
    if not isinstance(allowed, list):
        allowed = [allowed]
    if not allowed:
        return True
    clean_mode = str(mode or "").strip().lower()
    return clean_mode in [str(m or "").strip().lower() for m in allowed]


def available_translation_model_profiles(cfg: dict[str, Any], mode: str = "") -> list[str]:
    raw_profiles = cfg.get("model_profiles")
    if isinstance(raw_profiles, dict) and raw_profiles:
        clean_mode = str(mode or "").strip().lower()
        return [
            str(k).strip()
            for k, v in raw_profiles.items()
            if str(k).strip() and _profile_supports_mode(v, clean_mode)
        ]
    names = _list_model_names(cfg.get("text_models" if mode == "ocr_chat_completions" else "vlm_models"))
    if not names:
        names = _list_model_names(cfg.get("vlm_models")) or _list_model_names(cfg.get("text_models"))
    raw_model = str(cfg.get("model", "") or "").strip()
    if raw_model and raw_model not in names:
        names.insert(0, raw_model)
    return names or ["qwen3.6-plus"]


def resolve_translation_model_profile(
    cfg: dict[str, Any],
    mode: str,
    requested_profile: str = "",
) -> TranslationModelProfile:
    clean_mode = str(mode or "vlm_responses").strip().lower()
    requested = str(requested_profile or "").strip()
    mode_models = cfg.get("mode_models")
    mode_default = ""
    if isinstance(mode_models, dict):
        mode_default = str(mode_models.get(clean_mode, "") or "").strip()

    raw_profiles = cfg.get("model_profiles")
    if isinstance(raw_profiles, dict) and raw_profiles:
        profile_names = [str(k).strip() for k in raw_profiles.keys() if str(k).strip()]
        if not profile_names:
            raise ValueError("translation.model_profiles is empty")
        selected = requested or mode_default or str(cfg.get("model", "") or "").strip() or profile_names[0]
        if selected not in raw_profiles:
            matched = ""
            for name, raw_profile in raw_profiles.items():
                if isinstance(raw_profile, dict) and str(raw_profile.get("model", "") or "").strip() == selected:
                    matched = str(name)
                    break
            selected = matched or selected
        if selected not in raw_profiles:
            raise ValueError(f"translation model profile not found: {selected}")
        raw = raw_profiles[selected]
        if not _profile_supports_mode(raw, clean_mode):
            raise ValueError(
                f"model profile '{selected}' does not support translation mode '{clean_mode}'"
            )
        profile = raw if isinstance(raw, dict) else {}
        profile_has_auth_key = "api_key" in profile
        profile_has_auth_file = "api_key_file" in profile
        api_key = str(profile.get("api_key", "") or "").strip()
        api_key_file = str(profile.get("api_key_file", "") or "").strip()
        if not api_key_file and not profile_has_auth_key and not profile_has_auth_file:
            api_key_file = str(cfg.get("api_key_file", "") or "").strip()
        return TranslationModelProfile(
            name=selected,
            model=str(profile.get("model", "") or selected).strip() or selected,
            base_url=str(profile.get("base_url", "") or _profile_base_url(cfg, clean_mode)).strip(),
            api_key=api_key,
            api_key_file=api_key_file,
        )

    names = available_translation_model_profiles(cfg, clean_mode)
    selected = requested or mode_default or str(cfg.get("model", "") or "").strip() or names[0]
    return TranslationModelProfile(
        name=selected,
        model=selected,
        base_url=_profile_base_url(cfg, clean_mode),
        api_key=str(cfg.get("api_key", "") or "").strip(),
        api_key_file=str(cfg.get("api_key_file", "") or "").strip(),
    )


def load_api_key_for_profile(profile: TranslationModelProfile) -> str:
    if profile.api_key:
        return profile.api_key
    if profile.api_key_file:
        return load_api_key(profile.api_key_file)
    return ""


def _auth_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    clean_key = str(api_key or "").strip()
    if clean_key:
        headers["Authorization"] = f"Bearer {clean_key}"
    return headers


def load_api_key(path: str | Path) -> str:
    path_str = str(path or "").strip()
    if not path_str:
        raise ValueError(
            "Invalid config: translation.api_key_file is empty. "
            "Please set translation.api_key_file in config/general_config.yaml."
        )
    key_path = Path(path_str)
    if not key_path.exists() or not key_path.is_file():
        raise ValueError(
            f"api key file does not exist: {key_path}. "
            "Please check translation.api_key_file in config/general_config.yaml."
        )
    text = key_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"api key file is empty: {key_path}")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip().lower() in {"key", "api_key", "apikey"} and v.strip():
                return v.strip()
        else:
            return line
    raise ValueError(f"cannot parse api key from file: {key_path}")


def normalize_quotes_for_subtitle(text: str) -> str:
    if not text:
        return ""
    s = text.strip()

    paired_patterns = [
        r'^\u201c([\s\S]*)\u201d$',
        r'^"([\s\S]*)"$',
        r"^\u300e([\s\S]*)\u300f$",
        r"^\u300c([\s\S]*)\u300d$",
    ]
    for p in paired_patterns:
        m = re.match(p, s)
        if m:
            s = f"「{m.group(1).strip()}」"
            break

    # Boundary-only mismatch check:
    # only care whether first/last quote symbol is paired at boundaries.
    first = s[0] if s else ""
    last = s[-1] if s else ""
    openers = {'"', '『', '「', '"'}
    closers = {'"', '』', '」', '"'}
    pair_map = {'"': '"', '『': '』', '「': '」', '"': '"'}  # noqa: F601
    rev_pair_map = {'"': '"', '』': '『', '」': '「', '"': '"'}  # noqa: F601

    starts_with_opener = first in openers
    ends_with_closer = last in closers
    boundary_mismatch = False
    if starts_with_opener and not ends_with_closer:
        boundary_mismatch = True
    elif ends_with_closer and not starts_with_opener:
        boundary_mismatch = True
    elif starts_with_opener and ends_with_closer:
        if pair_map.get(first) != last and rev_pair_map.get(last) != first:
            boundary_mismatch = True

    if boundary_mismatch:
        cleaned = s.strip('"' + '"' + '"『』「」').strip()
        s = f"「{cleaned}」" if cleaned else "「」"

    # Normalize paired internal single quotes to Japanese corner quotes.
    chars = list(s)
    single_quote_indices = []
    for i, ch in enumerate(chars):
        if ch != "'" or not (0 < i < len(chars) - 1):
            continue
        prev_is_ascii_alnum = chars[i - 1].isascii() and chars[i - 1].isalnum()
        next_is_ascii_alnum = chars[i + 1].isascii() and chars[i + 1].isalnum()
        if prev_is_ascii_alnum and next_is_ascii_alnum:
            continue
        single_quote_indices.append(i)
    if len(single_quote_indices) >= 2:
        for opener, closer in zip(single_quote_indices[0::2], single_quote_indices[1::2]):
            chars[opener] = "『"
            chars[closer] = "』"
        s = "".join(chars)
    s = "".join(
        "『" if ch == "'" and 0 < i < len(s) - 1 else
        "』" if ch == "'" and 0 < i < len(s) - 1 else
        ch
        for i, ch in enumerate(s)
    )

    # Final output rule: remove paired outer 「」 if present.
    if len(s) >= 2 and s.startswith("「") and s.endswith("」"):
        s = s[1:-1].strip()

    # Normalize half-width brackets to full-width
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        s = "（" + s[1:-1] + "）"

    return s


def has_kana_leak_from_original(original_text: str, translated_text: str, min_len: int = 3) -> bool:
    if not original_text or not translated_text:
        return False
    n = max(1, int(min_len))
    kana_seq = re.compile(r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fd-\u30ff\u31f0-\u31ff\uff66-\uff9d\uff70\u30fc]+")
    for m in kana_seq.finditer(translated_text):
        seq = m.group(0)
        if _has_original_subsequence(original_text, seq, n):
            return True
    return False


def has_kanji_overlap_from_original(original_text: str, translated_text: str, min_len: int = 3) -> bool:
    if not original_text or not translated_text:
        return False
    n = max(1, int(min_len))
    kanji_seq = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
    for m in kanji_seq.finditer(translated_text):
        seq = m.group(0)
        if _has_original_subsequence(original_text, seq, n):
            return True
    return False


def _has_original_subsequence(original_text: str, seq: str, min_len: int) -> bool:
    if len(seq) < min_len:
        return False
    for i in range(0, len(seq) - min_len + 1):
        if seq[i : i + min_len] in original_text:
            return True
    return False


def _language_label_zh(lang: str) -> str:
    v = (lang or "").strip().lower().replace("_", "-")
    mapping = {
        "ja": "日语",
        "jpn": "日语",
        "jp": "日语",
        "japanese": "日语",
        "zh": "简体中文",
        "zh-cn": "简体中文",
        "zh-hans": "简体中文",
        "zh-sg": "简体中文",
        "chinese": "简体中文",
        "simplified chinese": "简体中文",
    }
    return mapping.get(v, lang)


def _normalize_text_extraction_backend(value: Any) -> str:
    v = str(value or "ocr").strip().lower().replace("-", "_")
    if v in {"rapidocr", "ocr", "default"}:
        return "ocr"
    if v in {"vlm_responses", "responses", "response"}:
        return "vlm_responses"
    if v in {"vlm_chat", "vlm_chat_completions", "chat", "chat_completions", "openai_chat"}:
        return "vlm_chat_completions"
    raise ValueError(f"Unsupported text extraction backend: {value}")


def _format_history_reference(history_items: list[dict[str, str]] | None, limit: int = 8) -> str:
    if not history_items:
        return ""
    lines: list[str] = []
    for item in history_items:
        if not isinstance(item, dict):
            continue
        time_text = str(item.get("time", "") or "").strip()
        speaker = str(item.get("speaker", "") or "").strip()
        original = str(item.get("original", "") or "").strip()
        translation = str(item.get("translation", "") or "").strip()
        if not original and not translation:
            continue
        prefix = f"{time_text} " if time_text else ""
        speaker_prefix = f"{speaker}: " if speaker else ""
        if translation:
            lines.append(f"{prefix}{speaker_prefix}{original} -> {translation}".strip())
        else:
            lines.append(f"{prefix}{speaker_prefix}{original}".strip())
    if not lines:
        return ""
    window = max(1, int(limit))
    return "### 前文参考\n" + "\n".join(lines[-window:]) + "\n\n"


def translate_segment_with_retry(
    seg_id: int,
    translator: VlmResponsesTranslator,
    speaker_image_path: Path,
    image_path: Path,
    speaker: str,
    history_items: list[dict[str, str]] | None = None,
    extra_requirements: str = "",
) -> tuple[str, str, str, dict[str, int]]:
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    attempts = max(1, int(getattr(translator, "max_retries", 0)) + 1)
    last_err: ValueError | None = None
    for attempt in range(1, attempts + 1):
        _log(f"[VLM] segment {seg_id}: request started (attempt {attempt}/{attempts})")
        try:
            speaker_name, original_text, translated_text, usage = (
                translator.translate_image_ja_to_zh_cn_structured_with_tag(
                    image_path=image_path,
                    speaker_image_path=speaker_image_path,
                    speaker=speaker,
                    request_tag=f"segment {seg_id}",
                    history_items=history_items,
                    extra_requirements=extra_requirements,
                )
            )
            pt = _safe_int(usage.get("prompt_tokens", 0))
            ct = _safe_int(usage.get("completion_tokens", 0))
            tt = _safe_int(usage.get("total_tokens", pt + ct), pt + ct)
            _log(
                f"[VLM] segment {seg_id}: request succeeded "
                f"(orig_chars={len(original_text)}, trans_chars={len(translated_text)}, "
                f"tokens_in={pt}, tokens_out={ct}, tokens_total={tt})"
            )
            return speaker_name, original_text, translated_text, usage
        except ValueError as exc:
            last_err = exc
            detail = str(exc).strip() or repr(exc)
            if attempt >= attempts:
                break
            _log(
                f"[VLM] segment {seg_id}: ValueError on attempt {attempt}/{attempts}: "
                f"{detail}; retrying"
            )
            time.sleep(max(0.0, float(getattr(translator, "retry_delay_sec", 0.0))))
    if last_err is not None:
        raise last_err
    raise RuntimeError("VLM request failed without an exception")


def translate_ocr_text_segment_with_retry(
    seg_id: int,
    translator: ChatCompletionsTextTranslator,
    original_text: str,
    speaker: str = "",
    history_items: list[dict[str, str]] | None = None,
    extra_requirements: str = "",
    context_before: list[str | dict[str, str]] | None = None,
    context_after: list[str | dict[str, str]] | None = None,
) -> tuple[str, dict[str, int]]:
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    attempts = max(1, int(getattr(translator, "max_retries", 0)) + 1)
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        _log(f"[LLM] segment {seg_id}: request started (attempt {attempt}/{attempts})")
        try:
            translated_text, usage = translator.translate_text_with_prompt(
                original_text=original_text,
                speaker=speaker,
                request_tag=f"segment {seg_id}",
                history_items=history_items,
                extra_requirements=extra_requirements,
                context_before=context_before,
                context_after=context_after,
            )
            pt = _safe_int(usage.get("prompt_tokens", 0))
            ct = _safe_int(usage.get("completion_tokens", 0))
            tt = _safe_int(usage.get("total_tokens", pt + ct), pt + ct)
            _log(
                f"[LLM] segment {seg_id}: request succeeded "
                f"(orig_chars={len(original_text)}, trans_chars={len(translated_text)}, "
                f"tokens_in={pt}, tokens_out={ct}, tokens_total={tt})"
            )
            return translated_text, usage
        except Exception as exc:
            last_err = exc
            detail = str(exc).strip() or repr(exc)
            if attempt >= attempts:
                break
            _log(
                f"[LLM] segment {seg_id}: {exc.__class__.__name__} on attempt "
                f"{attempt}/{attempts}: {detail}; retrying"
            )
            time.sleep(max(0.0, float(getattr(translator, "retry_delay_sec", 0.0))))
    if last_err is not None:
        raise last_err
    raise RuntimeError("LLM request failed without an exception")


class VlmResponsesTranslator:
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3.6-plus",
        responses_base_url: str = DEFAULT_RESPONSES_BASE_URL,
        temperature: float = 1.3,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        preserve_thinking: bool = False,
        empty_max_attempts: int = 3,
        timeout_sec: int = 30,
        timeout_backoff_sec: int = 15,
        max_retries: int = 2,
        retry_delay_sec: float = 1.5,
        disable_env_proxy: bool = True,
        game_name: str = "",
        source_language: str = "ja",
        target_language: str = "zh-CN",
        log_fn: Callable[[str], None] | None = None,
        io_log_path: str | Path | None = None,
        io_log_enabled: bool = False,
        enable_web_search: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.responses_base_url = responses_base_url.rstrip("/")
        self.temperature = float(temperature)
        self.enable_thinking = bool(enable_thinking)
        self.thinking_budget = int(thinking_budget) if thinking_budget is not None else None
        self.preserve_thinking = bool(preserve_thinking)
        self.empty_max_attempts = max(1, int(empty_max_attempts))
        self.timeout_sec = timeout_sec
        self.timeout_backoff_sec = max(0, int(timeout_backoff_sec))
        self.max_retries = max(0, int(max_retries))
        self.retry_delay_sec = max(0.0, float(retry_delay_sec))
        self.game_name = game_name.strip()
        self.source_language = _language_label_zh(source_language.strip() or "ja")
        self.target_language = _language_label_zh(target_language.strip() or "zh-CN")
        self.log_fn = log_fn
        self.io_log_path = Path(io_log_path) if io_log_path else None
        self.io_log_enabled = bool(io_log_enabled) and (self.io_log_path is not None)
        self.enable_web_search = bool(enable_web_search)
        self._io_lock = threading.Lock()
        self._io_pending: dict[str, dict[str, Any]] = {}
        self._opener = (
            request.build_opener(request.ProxyHandler({}))
            if disable_env_proxy
            else request.build_opener()
        )
        self._search_hint_cn = "如果遇到不确定含义的词语，请联网搜索辅助翻译。"

    def _messages_to_responses_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            role = str(m.get("role", "user") or "user")
            content = m.get("content", "")
            if isinstance(content, str):
                if role == "user":
                    out.append(
                        {
                            "role": role,
                            "content": [{"type": "input_text", "text": content}],
                        }
                    )
                else:
                    out.append({"role": role, "content": content})
                continue
            parts: list[dict[str, Any]] = []
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    typ = str(item.get("type", "")).strip().lower()
                    if typ == "text":
                        txt = str(item.get("text", "") or "")
                        if txt:
                            parts.append({"type": "input_text", "text": txt})
                        continue
                    if typ == "image_url":
                        iu = item.get("image_url")
                        url = ""
                        if isinstance(iu, dict):
                            url = str(iu.get("url", "") or "")
                        elif isinstance(iu, str):
                            url = iu
                        if url:
                            parts.append({"type": "input_image", "image_url": url})
                        continue
            out.append({"role": role, "content": parts if parts else ""})
        return out

    def _attach_web_search_tool(self, payload: dict[str, Any]) -> None:
        if not self.enable_web_search:
            return
        tools = payload.get("tools")
        if not isinstance(tools, list):
            tools = []
            payload["tools"] = tools
        if not any(isinstance(t, dict) and str(t.get("type", "")).strip() == "web_search" for t in tools):
            tools.append({"type": "web_search"})

    def _exception_message(self, exc: Exception) -> str:
        msg = str(exc)
        if isinstance(exc, urlerror.HTTPError):
            try:
                body = exc.read().decode("utf-8", errors="ignore").strip()
                if body:
                    return f"{msg} | body={body[:2000]}"
            except Exception:
                pass
        return msg

    def _image_input_log_value(self, image_input: str | Path | None) -> str:
        if image_input is None:
            return ""
        s = str(image_input)
        if s.startswith("data:"):
            mime = s[5:].split(";", 1)[0] or "unknown"
            return f"<data-url mime={mime} chars={len(s)}>"
        return str(Path(s).resolve())

    def translate_image_ja_to_zh_cn_structured_with_tag(
        self,
        image_path: str | Path,
        speaker: str,
        speaker_image_path: str | Path | None = None,
        request_tag: str = "",
        history_items: list[dict[str, str]] | None = None,
        custom_prompt: str = "",
        extra_requirements: str = "",
    ) -> tuple[str, str, str, dict[str, int]]:
        system_prompt = (
            f"### 角色\n"
            f"**你是{self.game_name}的游戏文本翻译助手，负责识别日文文本并翻译为中文。**\n"
            f"### 输入\n"
            "**你会收到两张图像：**\n"
            "- **图片1**：当前说话人的日文姓名\n"
            "- **图片2**：日文对话文本\n"
            f"### 任务\n"
            f"**识别图片1中的日文说话人名，结合你对{self.game_name}的了解理解说话人的背景和性格辅助翻译。**\n"
            "**不要把说话人信息输出到译文中。**\n"
            "**识别图片2中的日文对话文本，结合说话人信息翻译为中文译文。**\n"
            "**图像文字识别可能有误，对无法辨认的字形请结合上下文自行判断，不要强行翻译乱码内容。**\n"
            f"### 翻译规则\n"
            "- **语义优先**：在保证语义通顺的前提下，尽可能保留原文的换行与符号。\n"
            "- **禁止添加**：绝对禁止添加与原文无关的内容。\n"
            "- **保留符号**：保留原文使用的『』，禁止替换为\"\"或''。\n"
            "- **禁止自创标点**：禁止自行添加原文没有的标点。\n"
            "- **禁止残留**：禁止在句尾保留「ッ」等日文助词。\n"
            "- **英文处理**：原文若为纯英文，不需要翻译，直接输出原文。\n"
            "### 额外要求\n"
            "**{placeholder}**\n"
            "### 输出JSON\n"
            "**必须返回符合以下JSON Schema的JSON对象：**\n"
            "```json\n"
            "{\n"
            '  "speaker_name": "日文说话人姓名（保持原文，不翻译）",\n'
            '  "original_text": "图片2中识别的日文原文",\n'
            '  "translated_text": "中文译文"\n'
            "}\n"
            "```"
        )
        extra_requirements_text = str(extra_requirements or "").strip()
        if extra_requirements_text:
            system_prompt = system_prompt.replace("**{placeholder}**", f"**{extra_requirements_text}**")
        else:
            system_prompt = system_prompt.replace(
                "### 额外要求\n**{placeholder}**\n", ""
            )
        if self.enable_web_search:
            system_prompt += self._search_hint_cn
        history_text = _format_history_reference(history_items)
        user_text = (
            f"{history_text}"
            "### 图片1：说话人姓名\n"
            "### 图片2：对话文本\n"
            "**请识别并翻译：**"
        )
        custom = str(custom_prompt or "").strip()
        if custom:
            user_text += f"\n### 附加要求\n**{custom}**"
        
        # Support both file paths and base64 data URLs
        def _get_data_url(image_input: str | Path | None) -> str:
            if image_input is None:
                return ""
            s = str(image_input)
            if s.startswith("data:"):
                return s  # Already a data URL
            return self._to_data_url(Path(s))
        
        speaker_image_input = speaker_image_path if speaker_image_path is not None else image_path
        speaker_image_url = _get_data_url(speaker_image_input)
        dialogue_image_url = _get_data_url(image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": speaker_image_url}},
                    {"type": "image_url", "image_url": {"url": dialogue_image_url}},
                ],
            },
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "input": self._messages_to_responses_input(messages),
        }
        if self.io_log_enabled:
            self._append_io_log(
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": "request",
                    "request_tag": request_tag,
                    "speaker_image_path": self._image_input_log_value(speaker_image_input),
                    "dialogue_image_path": self._image_input_log_value(image_path),
                    "image_path": self._image_input_log_value(image_path),
                    "speaker": speaker,
                    "history_count": len(history_items or []),
                    "prompt_preview": {
                        "model": payload.get("model"),
                        "temperature": payload.get("temperature"),
                        "enable_thinking": payload.get("enable_thinking", self.enable_thinking),
                        "enable_search": self.enable_web_search,
                        "thinking_budget": payload.get("thinking_budget", self.thinking_budget),
                        "preserve_thinking": payload.get("preserve_thinking", self.preserve_thinking),
                        "response_format": payload.get("response_format"),
                        "system_prompt": system_prompt,
                        "user_prompt": user_text,
                    },
                }
            )
        payload["enable_thinking"] = self.enable_thinking
        self._attach_web_search_tool(payload)
        if self.thinking_budget is not None and self.thinking_budget > 0:
            payload["thinking_budget"] = int(self.thinking_budget)
        if self.preserve_thinking:
            payload["preserve_thinking"] = True
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{self.responses_base_url}/responses",
            data=body,
            headers=_auth_headers(self.api_key),
            method="POST",
        )
        last_err: Exception | None = None
        empty_limit = self.empty_max_attempts
        attempts = max(self.max_retries + 1, empty_limit)
        empty_count = 0
        err_count = 0
        current_timeout = int(self.timeout_sec)
        tag_prefix = f"{request_tag}: " if request_tag else ""
        for i in range(attempts):
            try:
                with self._opener.open(req, timeout=current_timeout) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                usage = self._extract_usage_from_data(data)
                speaker_name, original_text, translated_text = self._extract_structured_texts_from_data(data)
                speaker_name = self._strip_thinking_content(speaker_name).strip()
                original_text = self._strip_thinking_content(original_text).strip()
                translated_text = self._strip_thinking_content(translated_text).strip()
                translated_text = normalize_quotes_for_subtitle(translated_text)
                if translated_text and self._has_japanese_leak(original_text, translated_text, min_len=3):
                    if self.log_fn is not None:
                        self.log_fn(
                            f"{tag_prefix}warning: translated text leaks >=3 continuous kana chars from original; retrying"
                        )
                    translated_text = ""
                if self.io_log_enabled:
                    self._append_io_log(
                        {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event": "response",
                            "request_tag": request_tag,
                            "attempt": i + 1,
                            "usage": usage,
                            "raw_response_text": self._extract_text(data),
                            "parsed_speaker_name": speaker_name,
                            "parsed_original_text": original_text,
                            "parsed_translated_text": translated_text,
                        }
                    )
                if translated_text:
                    return speaker_name, original_text, translated_text, usage
                empty_count += 1
                if self.log_fn is not None:
                    self.log_fn(
                        f"{tag_prefix}warning: empty response (chars=0) on empty-attempt {empty_count + 1}/{empty_limit}"
                    )
                if empty_count >= empty_limit:
                    if self.log_fn is not None:
                        self.log_fn(
                            f"{tag_prefix}warning: empty response reached fixed max empty attempts ({empty_limit}); giving up this segment"
                        )
                    return "", "", "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                if i + 1 >= attempts:
                    break
                time.sleep(self.retry_delay_sec)
                continue
            except Exception as exc:
                last_err = exc
                err_count += 1
                err_detail = self._exception_message(exc)
                is_timeout_like = (
                    isinstance(exc, TimeoutError)
                    or "timeout" in exc.__class__.__name__.lower()
                    or "timed out" in str(exc).lower()
                )
                if self.log_fn is not None:
                    self.log_fn(
                        f"{tag_prefix}request attempt {i + 1}/{attempts} failed: {exc.__class__.__name__} | {err_detail}"
                    )
                    if is_timeout_like and self.timeout_backoff_sec > 0:
                        self.log_fn(
                            f"{tag_prefix}timeout detected: next request timeout +{self.timeout_backoff_sec}s"
                        )
                if is_timeout_like and self.timeout_backoff_sec > 0:
                    current_timeout += self.timeout_backoff_sec
                if self.io_log_enabled:
                    self._append_io_log(
                        {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event": "error",
                            "request_tag": request_tag,
                            "attempt": i + 1,
                            "error_type": exc.__class__.__name__,
                            "error_message": err_detail,
                            "next_timeout_sec": current_timeout,
                        }
                    )
                if err_count > self.max_retries or i + 1 >= attempts:
                    break
                time.sleep(self.retry_delay_sec)
        if last_err is not None:
            raise last_err
        return "", "", "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def translate_text_with_prompt(
        self,
        original_text: str,
        speaker: str = "",
        request_tag: str = "",
        custom_prompt: str = "",
        history_items: list[dict[str, str]] | None = None,
        extra_requirements: str = "",
    ) -> tuple[str, dict[str, int]]:
        src = str(original_text or "").strip()
        if not src:
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        game_hint = f"文本内容与{self.game_name}相关，请结合你的相关知识判断。" if self.game_name else ""
        custom = str(custom_prompt or "").strip()
        system_prompt = (
            f"### 角色\n"
            f"**你是{self.game_name}的游戏文本翻译助手，负责将{self.source_language}文本翻译为{self.target_language}。**\n"
            f"### 输入\n"
            f"**你会收到当前说话人姓名（仅供参考，不要输出到译文中）和已识别出的{self.source_language}原文文本。**\n"
            "**文本识别可能有误，对无法辨认的词请结合字形和上下文自行判断，不要强行翻译乱码内容。**\n"
            f"### 任务\n"
            f"**结合你对{self.game_name}的了解，将原文翻译为{self.target_language}译文。**{game_hint}\n"
            f"### 翻译规则\n"
            "- **语义优先**：在保证语义通顺的前提下，尽可能保留原文的换行与符号。\n"
            "- **禁止添加**：绝对禁止添加与原文无关的内容。\n"
            "- **保留符号**：保留原文使用的『』，禁止替换为\"\"或''。\n"
            "- **禁止自创标点**：禁止自行添加原文没有的标点。\n"
            "- **英文处理**：原文若为纯英文，不需要翻译，直接输出原文。\n"
            "### 额外要求\n"
            "**{placeholder}**\n"
            "### 输出JSON\n"
            "**必须返回符合以下JSON Schema的JSON对象：**\n"
            "```json\n"
            "{\n"
            f'  "translated_text": "{self.target_language}译文"\n'
            "}\n"
            "```"
        )
        extra_requirements_text = str(extra_requirements or "").strip()
        if extra_requirements_text:
            system_prompt = system_prompt.replace("**{placeholder}**", f"**{extra_requirements_text}**")
        else:
            system_prompt = system_prompt.replace(
                "### 额外要求\n**{placeholder}**\n", ""
            )
        if self.enable_web_search:
            system_prompt += self._search_hint_cn
        history_text = _format_history_reference(history_items)
        user_text = (
            f"{history_text}"
            f"### 当前说话人（仅供参考）\n"
            f"**{speaker}**\n"
            f"### 原文\n"
            f"{src}\n"
            "**请返回JSON：**"
        )
        if custom:
            user_text += f"\n### 附加要求\n**{custom}**"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "input": self._messages_to_responses_input(messages),
        }
        payload["enable_thinking"] = self.enable_thinking
        self._attach_web_search_tool(payload)
        if self.thinking_budget is not None and self.thinking_budget > 0:
            payload["thinking_budget"] = int(self.thinking_budget)
        if self.preserve_thinking:
            payload["preserve_thinking"] = True
        req = request.Request(
            f"{self.responses_base_url}/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=_auth_headers(self.api_key),
            method="POST",
        )
        empty_limit = self.empty_max_attempts
        attempts = max(self.max_retries + 1, empty_limit)
        empty_count = 0
        err_count = 0
        current_timeout = int(self.timeout_sec)
        last_err: Exception | None = None
        tag_prefix = f"{request_tag}: " if request_tag else ""
        for i in range(attempts):
            try:
                with self._opener.open(req, timeout=current_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                usage = self._extract_usage_from_data(data)
                raw = self._extract_text(data).strip()
                parsed: dict[str, Any] = {}
                if raw:
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        m = re.search(r"\{[\s\S]*\}", raw)
                        if m:
                            try:
                                parsed = json.loads(m.group(0))
                            except Exception:
                                parsed = {}
                translated = str((parsed or {}).get("translated_text", "")).strip()
                translated = self._strip_thinking_content(translated)
                translated = normalize_quotes_for_subtitle(translated)
                if translated and self._has_japanese_leak(src, translated, min_len=3):
                    translated = ""
                if translated:
                    return translated, usage
                empty_count += 1
                if self.log_fn is not None:
                    self.log_fn(
                        f"{tag_prefix}warning: empty translated_text on empty-attempt {empty_count + 1}/{empty_limit}"
                    )
                if empty_count >= empty_limit:
                    return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                time.sleep(self.retry_delay_sec)
                continue
            except Exception as exc:
                last_err = exc
                err_count += 1
                err_detail = self._exception_message(exc)
                is_timeout_like = (
                    isinstance(exc, TimeoutError)
                    or "timeout" in exc.__class__.__name__.lower()
                    or "timed out" in str(exc).lower()
                )
                if self.log_fn is not None:
                    self.log_fn(
                        f"{tag_prefix}request attempt {i + 1}/{attempts} failed: {exc.__class__.__name__} | {err_detail}"
                    )
                if is_timeout_like and self.timeout_backoff_sec > 0:
                    current_timeout += self.timeout_backoff_sec
                if err_count > self.max_retries or i + 1 >= attempts:
                    break
                time.sleep(self.retry_delay_sec)
        if last_err is not None:
            raise last_err
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def translate_single_image_ja_to_zh_cn_structured_with_tag(
        self,
        image_path: str | Path,
        request_tag: str = "",
        history_items: list[dict[str, str]] | None = None,
        custom_prompt: str = "",
        extra_requirements: str = "",
    ) -> tuple[str, str, dict[str, int]]:
        game_hint = f"图像中的文本内容与{self.game_name}相关，请结合你的相关知识判断。" if self.game_name else ""
        system_prompt = (
            f"### 角色\n"
            f"**你是{self.game_name}的游戏文本翻译助手，负责识别{self.source_language}文本并翻译为{self.target_language}。**\n"
            f"### 输入\n"
            "**你会收到一张图像，其中包含标题或文本区域（不含说话人姓名）。**\n"
            f"### 任务\n"
            f"**识别图像中的{self.source_language}文本，结合你对{self.game_name}的了解翻译为{self.target_language}译文。**{game_hint}\n"
            "**图像文字识别可能有误，对无法辨认的字形请结合上下文自行判断，不要强行翻译乱码内容。**\n"
            f"### 翻译规则\n"
            "- **语义优先**：在保证语义通顺的前提下，尽可能保留原文的换行与符号。\n"
            "- **禁止添加**：绝对禁止添加与原文无关的内容。\n"
            "- **保留符号**：保留原文使用的『』，禁止替换为\"\"或''。\n"
            "- **禁止自创标点**：禁止自行添加原文没有的标点。\n"
            "- **英文处理**：原文若为纯英文，不需要翻译，直接输出原文。\n"
            "### 额外要求\n"
            "**{placeholder}**\n"
            "### 输出JSON\n"
            "**必须返回符合以下JSON Schema的JSON对象：**\n"
            "```json\n"
            "{\n"
            f'  "original_text": "图像中识别的{self.source_language}原文",\n'
            f'  "translated_text": "{self.target_language}译文"\n'
            "}\n"
            "```"
        )
        extra_requirements_text = str(extra_requirements or "").strip()
        if extra_requirements_text:
            system_prompt = system_prompt.replace("**{placeholder}**", f"**{extra_requirements_text}**")
        else:
            system_prompt = system_prompt.replace(
                "### 额外要求\n**{placeholder}**\n", ""
            )
        if self.enable_web_search:
            system_prompt += self._search_hint_cn
        history_text = _format_history_reference(history_items)
        user_text = (
            f"{history_text}"
            "### 图像：标题/文本区域\n"
            "**请识别并翻译（前文仅作风格参考，不一定与当前图像直接相关）：**"
        )
        custom = str(custom_prompt or "").strip()
        if custom:
            user_text += f"\n### 附加要求\n**{custom}**"
        image_url = self._to_data_url(image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "input": self._messages_to_responses_input(messages),
        }
        if self.io_log_enabled:
            self._append_io_log(
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": "request",
                    "request_tag": request_tag,
                    "image_path": self._image_input_log_value(image_path),
                    "history_count": len(history_items or []),
                    "prompt_preview": {
                        "model": payload.get("model"),
                        "temperature": payload.get("temperature"),
                        "enable_thinking": payload.get("enable_thinking", self.enable_thinking),
                        "thinking_budget": payload.get("thinking_budget", self.thinking_budget),
                        "preserve_thinking": payload.get("preserve_thinking", self.preserve_thinking),
                        "response_format": payload.get("response_format"),
                        "system_prompt": system_prompt,
                        "user_prompt": user_text,
                    },
                }
            )
        payload["enable_thinking"] = self.enable_thinking
        self._attach_web_search_tool(payload)
        if self.thinking_budget is not None and self.thinking_budget > 0:
            payload["thinking_budget"] = int(self.thinking_budget)
        if self.preserve_thinking:
            payload["preserve_thinking"] = True
        req = request.Request(
            f"{self.responses_base_url}/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=_auth_headers(self.api_key),
            method="POST",
        )
        empty_limit = self.empty_max_attempts
        attempts = max(self.max_retries + 1, empty_limit)
        empty_count = 0
        err_count = 0
        current_timeout = int(self.timeout_sec)
        last_err: Exception | None = None
        tag_prefix = f"{request_tag}: " if request_tag else ""
        for i in range(attempts):
            try:
                with self._opener.open(req, timeout=current_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                usage = self._extract_usage_from_data(data)
                raw_text = self._extract_text(data).strip()
                obj: dict[str, Any] = {}
                if raw_text:
                    try:
                        obj = json.loads(raw_text)
                    except Exception:
                        m = re.search(r"\{[\s\S]*\}", raw_text)
                        if m:
                            try:
                                obj = json.loads(m.group(0))
                            except Exception:
                                obj = {}
                original_text = str((obj or {}).get("original_text", "")).strip()
                translated_text = str((obj or {}).get("translated_text", "")).strip()
                original_text = self._strip_thinking_content(original_text)
                translated_text = self._strip_thinking_content(translated_text)
                translated_text = normalize_quotes_for_subtitle(translated_text)
                if translated_text and self._has_japanese_leak(original_text, translated_text, min_len=3):
                    translated_text = ""
                if self.io_log_enabled:
                    self._append_io_log(
                        {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event": "response",
                            "request_tag": request_tag,
                            "attempt": i + 1,
                            "usage": usage,
                            "raw_response_text": self._extract_text(data),
                            "parsed_original_text": original_text,
                            "parsed_translated_text": translated_text,
                        }
                    )
                if translated_text:
                    return original_text, translated_text, usage
                empty_count += 1
                if self.log_fn is not None:
                    self.log_fn(
                        f"{tag_prefix}warning: empty response (chars=0) on empty-attempt {empty_count + 1}/{empty_limit}"
                    )
                if empty_count >= empty_limit:
                    return "", "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                time.sleep(self.retry_delay_sec)
                continue
            except Exception as exc:
                last_err = exc
                err_count += 1
                err_detail = self._exception_message(exc)
                is_timeout_like = (
                    isinstance(exc, TimeoutError)
                    or "timeout" in exc.__class__.__name__.lower()
                    or "timed out" in str(exc).lower()
                )
                if self.log_fn is not None:
                    self.log_fn(
                        f"{tag_prefix}request attempt {i + 1}/{attempts} failed: {exc.__class__.__name__} | {err_detail}"
                    )
                if is_timeout_like and self.timeout_backoff_sec > 0:
                    current_timeout += self.timeout_backoff_sec
                if err_count > self.max_retries or i + 1 >= attempts:
                    break
                time.sleep(self.retry_delay_sec)
        if last_err is not None:
            raise last_err
        return "", "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _extract_structured_texts_from_data(self, data: dict[str, Any]) -> tuple[str, str, str]:
        raw_text = self._extract_text(data)
        obj: dict[str, Any] = {}
        if raw_text:
            try:
                obj = json.loads(raw_text)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", raw_text)
                if m:
                    try:
                        obj = json.loads(m.group(0))
                    except Exception:
                        obj = {}
        if not isinstance(obj, dict):
            obj = {}
        speaker_name = str(
            obj.get("speaker_name")
            or obj.get("speaker")
            or obj.get("character_name")
            or ""
        ).strip()
        original = str(
            obj.get("original_text")
            or obj.get("ocr_text")
            or obj.get("source_text")
            or ""
        ).strip()
        translated = str(
            obj.get("translated_text")
            or obj.get("translation")
            or obj.get("target_text")
            or ""
        ).strip()
        return speaker_name, original, translated

    def _extract_usage_from_data(self, data: dict[str, Any]) -> dict[str, int]:
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        prompt = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        )
        completion = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        total = int(usage.get("total_tokens") or (prompt + completion))
        return {
            "prompt_tokens": max(0, prompt),
            "completion_tokens": max(0, completion),
            "total_tokens": max(0, total),
        }

    def _has_japanese_leak(self, original_text: str, translated_text: str, min_len: int = 3) -> bool:
        return has_kana_leak_from_original(original_text, translated_text, min_len=min_len)

    def _append_io_log(self, record: dict[str, Any]) -> None:
        if not self.io_log_enabled or self.io_log_path is None:
            return
        try:
            self.io_log_path.parent.mkdir(parents=True, exist_ok=True)
            event = record.get("event", "")
            tag = record.get("request_tag", "")
            with self._io_lock:
                if event == "request":
                    self._io_pending[tag] = record
                    return
                request_record = self._io_pending.pop(tag, None)
                pair: dict[str, Any] = {"response": record}
                if request_record is not None:
                    pair["request"] = request_record
                line = json.dumps(pair, ensure_ascii=False)
                with self.io_log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass

    def _to_data_url(self, image_path: str | Path) -> str:
        if str(image_path).startswith("data:"):
            return str(image_path)
        p = Path(image_path)
        raw = p.read_bytes()
        suffix = p.suffix.lower()
        mime = "image/png"
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _extract_text(self, data: dict[str, Any]) -> str:
        out_text = data.get("output_text")
        if isinstance(out_text, str) and out_text.strip():
            return out_text
        output = data.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type", "")).strip().lower() != "message":
                    continue
                content = item.get("content")
                if isinstance(content, str):
                    if content.strip():
                        parts.append(content)
                    continue
                if isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        txt = c.get("text")
                        if txt is None:
                            txt = c.get("content")
                        if txt is None:
                            continue
                        s = str(txt)
                        if s.strip():
                            parts.append(s)
            if parts:
                return "\n".join(parts)
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        parts.append(str(item.get("text")))
                    elif item.get("content"):
                        parts.append(str(item.get("content")))
            return "\n".join([x for x in parts if x.strip()])
        return ""

    def _strip_thinking_content(self, text: str) -> str:
        if not text:
            return ""
        out = text
        # Remove common "thinking" blocks used by reasoning models.
        out = re.sub(r"<think>[\s\S]*?</think>", "", out, flags=re.IGNORECASE)
        out = re.sub(r"<thinking>[\s\S]*?</thinking>", "", out, flags=re.IGNORECASE)
        out = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", out, flags=re.IGNORECASE)

        # If model outputs "final answer" style wrapper, keep only final part.
        lowered = out.lower()
        marker_pos = -1
        for marker in ["final answer:", "answer:"]:
            p = lowered.rfind(marker)
            if p > marker_pos:
                marker_pos = p
        if marker_pos >= 0:
            out = out[marker_pos:].split(":", 1)[-1]

        return out.strip()


class VlmImageTextExtractor:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        backend: str = "vlm_chat_completions",
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        preserve_thinking: bool = False,
        empty_max_attempts: int = 3,
        timeout_sec: int = 30,
        timeout_backoff_sec: int = 15,
        max_retries: int = 2,
        retry_delay_sec: float = 1.5,
        disable_env_proxy: bool = True,
        game_name: str = "",
        source_language: str = "ja",
        log_fn: Callable[[str], None] | None = None,
        io_log_path: str | Path | None = None,
        io_log_enabled: bool = False,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.base_url = str(base_url or DEFAULT_CHAT_COMPLETIONS_BASE_URL).rstrip("/")
        self.backend = _normalize_text_extraction_backend(backend)
        self.temperature = float(temperature)
        self.top_p = float(top_p) if top_p is not None else None
        self.top_k = int(top_k) if top_k is not None else None
        self.max_tokens = max(0, int(max_tokens))
        self.enable_thinking = bool(enable_thinking)
        self.thinking_budget = int(thinking_budget) if thinking_budget is not None else None
        self.preserve_thinking = bool(preserve_thinking)
        self.empty_max_attempts = max(1, int(empty_max_attempts))
        self.timeout_sec = int(timeout_sec)
        self.timeout_backoff_sec = max(0, int(timeout_backoff_sec))
        self.max_retries = max(0, int(max_retries))
        self.retry_delay_sec = max(0.0, float(retry_delay_sec))
        self.game_name = str(game_name or "").strip()
        self.source_language = _language_label_zh(str(source_language or "ja").strip() or "ja")
        self.log_fn = log_fn
        self.io_log_path = Path(io_log_path) if io_log_path else None
        self.io_log_enabled = bool(io_log_enabled) and (self.io_log_path is not None)
        self._io_lock = threading.Lock()
        self._io_pending: dict[str, dict[str, Any]] = {}
        self._opener = (
            request.build_opener(request.ProxyHandler({}))
            if disable_env_proxy
            else request.build_opener()
        )

    def extract_text_from_images(
        self,
        speaker_image: str | Path,
        dialogue_image: str | Path,
        request_tag: str = "",
    ) -> tuple[str, str, dict[str, int]]:
        system_prompt = (
            "### 角色\n"
            f"**你是{self.game_name}的游戏字幕图像文字识别助手。**\n"
            "### 输入\n"
            "**你会收到两张图像：**\n"
            "- **图片1**：当前说话人的姓名区域\n"
            "- **图片2**：对白文本区域\n"
            "### 任务\n"
            f"**识别图片1中的{self.source_language}说话人姓名。**\n"
            f"**识别图片2中的{self.source_language}对白原文。只做文字识别和格式化，不要翻译。**\n"
            "### 识别规则\n"
            "- **保留原文**：保留原文语言、换行、符号、括号和语气符号。\n"
            "- **多行换行**：图片2若为多行对白，`original_text` 必须使用 `\\n` 表示换行。\n"
            "- **空白处理**：某个区域没有可见文字时，对应字段输出空字符串。\n"
            "- **禁止编造**：文字模糊时可结合字形判断，但不要编造看不见的内容。\n"
            "- **禁止翻译**：不要把日文、英文或其他原文翻译成中文。\n"
            "### 输出JSON\n"
            "**只输出 JSON，不要 Markdown，不要解释。必须符合以下格式：**\n"
            "```json\n"
            "{\n"
            '  "speaker_name": "原文说话人姓名",\n'
            '  "original_text": "原文对白文本"\n'
            "}\n"
            "```"
        )
        user_text = "### 图片1：说话人姓名\n### 图片2：对白文本\n**请识别并输出 JSON：**"
        speaker_url = self._to_data_url(speaker_image)
        dialogue_url = self._to_data_url(dialogue_image)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": speaker_url}},
                    {"type": "image_url", "image_url": {"url": dialogue_url}},
                ],
            },
        ]
        payload = self._build_payload(messages)
        if self.io_log_enabled:
            self._append_io_log(
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": "request",
                    "request_tag": request_tag,
                    "backend": self.backend,
                    "speaker_image_path": self._image_input_log_value(speaker_image),
                    "dialogue_image_path": self._image_input_log_value(dialogue_image),
                    "prompt_preview": {
                        "model": payload.get("model"),
                        "temperature": payload.get("temperature"),
                        "top_p": payload.get("top_p"),
                        "top_k": payload.get("top_k"),
                        "max_tokens": payload.get("max_tokens"),
                        "chat_template_kwargs": payload.get("chat_template_kwargs"),
                        "system_prompt": system_prompt,
                        "user_prompt": user_text,
                    },
                }
            )
        return self._request_with_retries(payload, request_tag)

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "stream": False,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if self.thinking_budget is not None:
            payload["thinking_budget"] = int(self.thinking_budget)
        if self.preserve_thinking:
            payload["preserve_thinking"] = True
        if self.backend == "vlm_responses":
            payload["input"] = VlmResponsesTranslator._messages_to_responses_input(self, messages)
        else:
            payload["messages"] = messages
        return payload

    def _request_with_retries(self, payload: dict[str, Any], request_tag: str) -> tuple[str, str, dict[str, int]]:
        path = "/responses" if self.backend == "vlm_responses" else "/chat/completions"
        req = request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=_auth_headers(self.api_key),
            method="POST",
        )
        empty_limit = self.empty_max_attempts
        attempts = max(self.max_retries + 1, empty_limit)
        empty_count = 0
        err_count = 0
        current_timeout = int(self.timeout_sec)
        last_err: Exception | None = None
        tag_prefix = f"{request_tag}: " if request_tag else ""
        for i in range(attempts):
            try:
                with self._opener.open(req, timeout=current_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                usage = self._extract_usage_from_data(data)
                raw = self._extract_text(data).strip()
                speaker, original = self._parse_json_text(raw)
                if self.io_log_enabled:
                    self._append_io_log(
                        {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event": "response",
                            "request_tag": request_tag,
                            "attempt": i + 1,
                            "usage": usage,
                            "raw_response_text": raw,
                            "parsed_speaker_name": speaker,
                            "parsed_original_text": original,
                        }
                    )
                if speaker or original:
                    return speaker, original, usage
                empty_count += 1
                if self.log_fn is not None:
                    self.log_fn(
                        f"{tag_prefix}warning: empty extraction on empty-attempt {empty_count + 1}/{empty_limit}"
                    )
                if empty_count >= empty_limit:
                    return "", "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                time.sleep(self.retry_delay_sec)
                continue
            except Exception as exc:
                last_err = exc
                err_count += 1
                err_detail = self._exception_message(exc)
                is_timeout_like = (
                    isinstance(exc, TimeoutError)
                    or "timeout" in exc.__class__.__name__.lower()
                    or "timed out" in str(exc).lower()
                )
                if self.log_fn is not None:
                    self.log_fn(
                        f"{tag_prefix}request attempt {i + 1}/{attempts} failed: {exc.__class__.__name__} | {err_detail}"
                    )
                if is_timeout_like and self.timeout_backoff_sec > 0:
                    current_timeout += self.timeout_backoff_sec
                if err_count > self.max_retries or i + 1 >= attempts:
                    break
                time.sleep(self.retry_delay_sec)
        if last_err is not None:
            raise last_err
        return "", "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _parse_json_text(self, raw_text: str) -> tuple[str, str]:
        clean = self._strip_thinking_content(raw_text).strip()
        obj: dict[str, Any] = {}
        if clean:
            try:
                obj = json.loads(clean)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", clean)
                if m:
                    try:
                        obj = json.loads(m.group(0))
                    except Exception:
                        obj = {}
        if not isinstance(obj, dict):
            obj = {}
        speaker = str(
            obj.get("speaker_name")
            or obj.get("speaker")
            or obj.get("character_name")
            or ""
        ).strip()
        original = str(
            obj.get("original_text")
            or obj.get("ocr_text")
            or obj.get("source_text")
            or ""
        ).strip()
        return speaker, original

    def _extract_usage_from_data(self, data: dict[str, Any]) -> dict[str, int]:
        return VlmResponsesTranslator._extract_usage_from_data(self, data)

    def _extract_text(self, data: dict[str, Any]) -> str:
        if self.backend == "vlm_responses":
            return VlmResponsesTranslator._extract_text(self, data)
        return ChatCompletionsTextTranslator._extract_text(self, data)

    def _strip_thinking_content(self, text: str) -> str:
        return VlmResponsesTranslator._strip_thinking_content(self, text)

    def _exception_message(self, exc: Exception) -> str:
        return VlmResponsesTranslator._exception_message(self, exc)

    def _image_input_log_value(self, image_input: str | Path | None) -> str:
        return VlmResponsesTranslator._image_input_log_value(self, image_input)

    def _to_data_url(self, image_input: str | Path) -> str:
        return VlmResponsesTranslator._to_data_url(self, image_input)

    def _append_io_log(self, record: dict[str, Any]) -> None:
        return VlmResponsesTranslator._append_io_log(self, record)


class ChatCompletionsTextTranslator:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_CHAT_COMPLETIONS_BASE_URL,
        temperature: float = 1.3,
        top_p: float | None = None,
        top_k: int | None = None,
        empty_max_attempts: int = 3,
        timeout_sec: int = 30,
        timeout_backoff_sec: int = 15,
        max_retries: int = 2,
        retry_delay_sec: float = 1.5,
        disable_env_proxy: bool = True,
        game_name: str = "",
        source_language: str = "ja",
        target_language: str = "zh-CN",
        log_fn: Callable[[str], None] | None = None,
        io_log_path: str | Path | None = None,
        io_log_enabled: bool = False,
        enable_web_search: bool = False,
        context_window: int = 8,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.base_url = str(base_url or DEFAULT_CHAT_COMPLETIONS_BASE_URL).rstrip("/")
        self.temperature = float(temperature)
        self.top_p = float(top_p) if top_p is not None else None
        self.top_k = int(top_k) if top_k is not None else None
        self.empty_max_attempts = max(1, int(empty_max_attempts))
        self.timeout_sec = int(timeout_sec)
        self.timeout_backoff_sec = max(0, int(timeout_backoff_sec))
        self.max_retries = max(0, int(max_retries))
        self.retry_delay_sec = max(0.0, float(retry_delay_sec))
        self.game_name = game_name.strip()
        self.source_language = _language_label_zh(source_language.strip() or "ja")
        self.target_language = _language_label_zh(target_language.strip() or "zh-CN")
        self.log_fn = log_fn
        self.io_log_path = Path(io_log_path) if io_log_path else None
        self.io_log_enabled = bool(io_log_enabled) and (self.io_log_path is not None)
        self.enable_web_search = bool(enable_web_search)
        self.context_window = max(0, int(context_window))
        self._io_lock = threading.Lock()
        self._io_pending: dict[str, dict[str, Any]] = {}
        self._opener = (
            request.build_opener(request.ProxyHandler({}))
            if disable_env_proxy
            else request.build_opener()
        )
        self._search_hint_cn = "如果遇到不确定含义的词语，请联网搜索辅助翻译。"

    def translate_text_with_prompt(
        self,
        original_text: str,
        speaker: str = "",
        request_tag: str = "",
        custom_prompt: str = "",
        history_items: list[dict[str, str]] | None = None,
        extra_requirements: str = "",
        context_before: list[str | dict[str, str]] | None = None,
        context_after: list[str | dict[str, str]] | None = None,
    ) -> tuple[str, dict[str, int]]:
        src = str(original_text or "").strip()
        if not src:
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        game_hint = f"文本内容与{self.game_name}相关，请结合你的相关知识判断。" if self.game_name else ""
        system_prompt = (
            f"### 角色\n"
            f"**你是{self.game_name}的游戏文本翻译助手，负责将{self.source_language}文本翻译为{self.target_language}。**\n"
            f"### 输入\n"
            f"**你会收到当前说话人姓名（仅供参考，不要输出到译文中）和已通过OCR识别的{self.source_language}原文文本。**\n"
            "**OCR识别可能有误，对无法辨认的词请结合字形和上下文自行判断，不要强行翻译乱码内容。**\n"
            f"### 任务\n"
            f"**结合你对{self.game_name}的了解，将原文翻译为{self.target_language}译文。**{game_hint}\n"
            f"### 翻译规则\n"
            "- **语义优先**：在保证语义通顺的前提下，尽可能保留原文的换行与符号。\n"
            "- **禁止添加**：绝对禁止添加与原文无关的内容。\n"
            "- **保留符号**：保留原文使用的『』，禁止替换为\"\"或''。\n"
            "- **禁止自创标点**：禁止自行添加原文没有的标点。\n"
            "- **禁止残留**：禁止在句尾保留「ッ」等日文助词。\n"
            "- **英文处理**：原文若为纯英文，不需要翻译，直接输出原文。\n"
            f"### 额外要求\n"
            "**{placeholder}**\n"
            "### 输出要求\n"
            "**请直接输出译文，不要添加任何额外说明、前缀或标记。**"
        )
        extra_requirements_text = str(extra_requirements or "").strip()
        if extra_requirements_text:
            system_prompt = system_prompt.replace("**{placeholder}**", f"**{extra_requirements_text}**")
        else:
            system_prompt = system_prompt.replace(
                "### 额外要求\n**{placeholder}**\n", ""
            )
        if self.enable_web_search:
            system_prompt += self._search_hint_cn
        context_block = ""
        if context_before or context_after:
            context_block = "### 上下文参考（仅供理解语境，不需要翻译本节内容）\n"
            if context_before:
                trimmed: list[str] = []
                for item in context_before:
                    if isinstance(item, dict):
                        speaker_name = str(item.get("speaker", "") or "").strip() or "未知"
                        original = str(item.get("original", "") or "").strip()
                        translation = str(item.get("translation", "") or "").strip()
                        if original and translation:
                            trimmed.append(f"【说话人：{speaker_name}】原文：{original} => 译文：{translation}")
                        elif original:
                            trimmed.append(f"【说话人：{speaker_name}】原文：{original}")
                    else:
                        text = str(item or "").strip()
                        if text:
                            trimmed.append(text)
                if trimmed:
                    window = max(1, self.context_window) if self.context_window else len(trimmed)
                    context_block += "上文（格式：【说话人】原文 => 已采用译文）：" + " | ".join(trimmed[-window:]) + "\n"
            if context_after:
                trimmed = []
                for item in context_after:
                    if isinstance(item, dict):
                        speaker_name = str(item.get("speaker", "") or "").strip() or "未知"
                        original = str(item.get("original", "") or "").strip()
                        if original:
                            trimmed.append(f"【说话人：{speaker_name}】原文：{original}")
                    else:
                        text = str(item or "").strip()
                        if text:
                            trimmed.append(text)
                if trimmed:
                    window = max(1, self.context_window) if self.context_window else len(trimmed)
                    context_block += "下文（格式：【说话人】原文；不含译文）：" + " | ".join(trimmed[:window]) + "\n"
            context_block += "\n"
        if context_block:
            system_prompt += context_block
        history_text = ""
        if history_items:
            lines = []
            for item in history_items:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"{item.get('time', '')} {item.get('speaker', '')}: "
                    f"{item.get('original', '')} -> {item.get('translation', '')}"
                )
                if lines:
                    history_text = "### 前文参考\n" + "\n".join(lines[-8:]) + "\n\n"
        user_text = (
            f"{history_text}"
            f"### 当前说话人（仅供参考）\n"
            f"**{speaker}**\n"
            f"### 原文\n"
            f"{src}\n"
            f"**请直接输出译文：**"
        )
        custom = str(custom_prompt or "").strip()
        if custom:
            user_text += f"\n附加要求：{custom}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.io_log_enabled:
            self._append_io_log(
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": "request",
                    "request_tag": request_tag,
                    "history_count": len(history_items or []),
                    "prompt_preview": {
                        "model": payload.get("model"),
                        "temperature": payload.get("temperature"),
                        "response_format": payload.get("response_format"),
                        "system_prompt": system_prompt,
                        "user_prompt": user_text,
                    },
                }
            )
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=_auth_headers(self.api_key),
            method="POST",
        )
        empty_limit = self.empty_max_attempts
        attempts = max(self.max_retries + 1, empty_limit)
        empty_count = 0
        err_count = 0
        current_timeout = int(self.timeout_sec)
        last_err: Exception | None = None
        tag_prefix = f"{request_tag}: " if request_tag else ""
        for i in range(attempts):
            try:
                with self._opener.open(req, timeout=current_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                usage = self._extract_usage_from_data(data)
                raw = self._extract_text(data).strip()
                translated = self._strip_thinking_content(raw)
                translated = normalize_quotes_for_subtitle(translated)
                if translated and has_kana_leak_from_original(src, translated, min_len=3):
                    translated = ""
                if self.io_log_enabled:
                    self._append_io_log(
                        {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event": "response",
                            "request_tag": request_tag,
                            "attempt": i + 1,
                            "usage": usage,
                            "raw_response_text": raw,
                            "parsed_translated_text": translated,
                        }
                    )
                if translated:
                    return translated, usage
                empty_count += 1
                if self.log_fn is not None:
                    self.log_fn(
                        f"{tag_prefix}warning: empty translated_text on empty-attempt {empty_count + 1}/{empty_limit}"
                    )
                if empty_count >= empty_limit:
                    return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                time.sleep(self.retry_delay_sec)
                continue
            except Exception as exc:
                last_err = exc
                err_count += 1
                err_detail = self._exception_message(exc)
                is_timeout_like = (
                    isinstance(exc, TimeoutError)
                    or "timeout" in exc.__class__.__name__.lower()
                    or "timed out" in str(exc).lower()
                )
                if self.log_fn is not None:
                    self.log_fn(
                        f"{tag_prefix}request attempt {i + 1}/{attempts} failed: {exc.__class__.__name__} | {err_detail}"
                    )
                if is_timeout_like and self.timeout_backoff_sec > 0:
                    current_timeout += self.timeout_backoff_sec
                if err_count > self.max_retries or i + 1 >= attempts:
                    break
                time.sleep(self.retry_delay_sec)
        if last_err is not None:
            raise last_err
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _exception_message(self, exc: Exception) -> str:
        msg = str(exc)
        if isinstance(exc, urlerror.HTTPError):
            try:
                body = exc.read().decode("utf-8", errors="ignore").strip()
                if body:
                    return f"{msg} | body={body[:2000]}"
            except Exception:
                pass
        return msg

    def _extract_usage_from_data(self, data: dict[str, Any]) -> dict[str, int]:
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        return {
            "prompt_tokens": max(0, prompt),
            "completion_tokens": max(0, completion),
            "total_tokens": max(0, total),
        }

    def _extract_text(self, data: dict[str, Any]) -> str:
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        parts.append(str(item.get("text")))
                    elif item.get("content"):
                        parts.append(str(item.get("content")))
            return "\n".join([x for x in parts if x.strip()])
        return ""

    def _strip_thinking_content(self, text: str) -> str:
        return VlmResponsesTranslator._strip_thinking_content(self, text)

    def _append_io_log(self, record: dict[str, Any]) -> None:
        if not self.io_log_enabled or self.io_log_path is None:
            return
        try:
            self.io_log_path.parent.mkdir(parents=True, exist_ok=True)
            event = record.get("event", "")
            tag = record.get("request_tag", "")
            with self._io_lock:
                if event == "request":
                    self._io_pending[tag] = record
                    return
                request_record = self._io_pending.pop(tag, None)
                pair: dict[str, Any] = {"response": record}
                if request_record is not None:
                    pair["request"] = request_record
                line = json.dumps(pair, ensure_ascii=False)
                with self.io_log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass
