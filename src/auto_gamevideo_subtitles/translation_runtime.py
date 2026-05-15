from __future__ import annotations

import base64
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib import error as urlerror
from urllib import request


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


def _normalize_quotes_for_subtitle(text: str) -> str:
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
    pair_map = {'"': '"', '『': '』', '「': '」', '"': '"'}
    rev_pair_map = {'"': '"', '』': '『', '」': '「', '"': '"'}

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


def _has_kana_leak_from_original(original_text: str, translated_text: str, min_len: int = 3) -> bool:
    if not original_text or not translated_text:
        return False
    n = max(1, int(min_len))
    kana_seq = re.compile(r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fd-\u30ff\u31f0-\u31ff\uff66-\uff9d\uff70\u30fc]+")
    for m in kana_seq.finditer(translated_text):
        seq = m.group(0)
        if _has_original_subsequence(original_text, seq, n):
            return True
    return False


def _has_kanji_overlap_from_original(original_text: str, translated_text: str, min_len: int = 3) -> bool:
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


# class DeepSeekTranslator:
#     def __init__(
#         self,
#         api_url: str,
#         api_key: str,
#         model: str = "deepseek-chat",
#         temperature: float = 1.3,
#         game_name: str = "",
#         source_language: str = "ja",
#         target_language: str = "zh-CN",
#     ) -> None:
#         self.api_url = api_url
#         self.api_key = api_key
#         self.model = model
#         self.temperature = float(temperature)
#         self.game_name = game_name.strip()
#         self.source_language = _language_label_zh(source_language.strip() or "ja")
#         self.target_language = _language_label_zh(target_language.strip() or "zh-CN")

#     def translate_ja_to_zh_cn(self, text: str) -> str:
#         if not text.strip():
#             return ""
#         game_hint = f"，内容与{self.game_name}相关" if self.game_name else ""
#         prompt = (
#             f"将我给的{self.source_language}原文翻译为{self.target_language}{game_hint}。"
#             "注意保留换行符和格式，绝对禁止添加和原文无关的翻译。"
#             "不要解释翻译方式，直接给出译文。"
#             ""
#         )
#         payload: dict[str, Any] = {
#             "model": self.model,
#             "temperature": self.temperature,
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": f"你是一个游戏文本翻译助手，专注于将{self.game_name}相关的{self.source_language}文本翻译为{self.target_language}。\n译文需要保留原文的换行符和格式，禁止添加和原文无关的内容。",
#                 },
#                 {
#                     "role": "user",
#                     "content": f"{prompt}\n{text}",
#                 },
#             ],
#         }
#         body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
#         req = request.Request(
#             self.api_url,
#             data=body,
#             headers={
#                 "Content-Type": "application/json",
#                 "Authorization": f"Bearer {self.api_key}",
#             },
#             method="POST",
#         )
#         with request.urlopen(req, timeout=60) as resp:
#             raw = resp.read().decode("utf-8")
#         data = json.loads(raw)
#         out = (
#             data.get("choices", [{}])[0]
#             .get("message", {})
#             .get("content", "")
#             .strip()
#         )
#         return _normalize_quotes_for_subtitle(out)


class BailianVlmTranslator:
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3.6-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 1.3,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        preserve_thinking: bool = False,
        empty_max_attempts: int = 5,
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
        self.base_url = base_url.rstrip("/")
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

    def translate_image_ja_to_zh_cn(self, image_path: str | Path, speaker: str) -> str:
        speaker_name, original_text, translated, usage = self.translate_image_ja_to_zh_cn_structured_with_tag(
            image_path=image_path,
            speaker_image_path=image_path,
            speaker=speaker,
            request_tag="",
            history_items=None,
        )
        _ = speaker_name
        _ = original_text
        _ = usage
        return translated

    def translate_image_ja_to_zh_cn_with_tag(
        self,
        image_path: str | Path,
        speaker: str,
        request_tag: str = "",
    ) -> str:
        game_hint = f"，内容与{self.game_name}相关" if self.game_name else ""
        prompt = (
            f"识别图像中的{self.source_language}文本，翻译为{self.target_language}{game_hint}。"
            f"说话人为{speaker}。说话人信息仅作语气和背景参考，不需要输出到译文中。"
            "注意保留换行和格式(如果图片中有两行,则你也需要返回两行)，语气与原文和说话角色一致,绝对禁止添加和原文无关的翻译，输出且仅输出译文。"
        )
        if self.enable_web_search:
            prompt += self._search_hint_cn
        image_url = self._to_data_url(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "input": self._messages_to_responses_input(messages),
        }
        # DashScope/OpenAI-compatible optional thinking/reasoning controls.
        payload["enable_thinking"] = self.enable_thinking
        self._attach_web_search_tool(payload)
        if self.thinking_budget is not None and self.thinking_budget > 0:
            payload["thinking_budget"] = int(self.thinking_budget)
        if self.preserve_thinking:
            payload["preserve_thinking"] = True
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/responses",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_err: Exception | None = None
        empty_limit = 3
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
                text = self._extract_text(data).strip()
                text = self._strip_thinking_content(text)
                if text:
                    return _normalize_quotes_for_subtitle(text)
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
                    return ""
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
                if err_count > self.max_retries or i + 1 >= attempts:
                    break
                time.sleep(self.retry_delay_sec)
        if last_err is not None:
            raise last_err
        return ""

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
        game_hint = f"图像中的文本内容与{self.game_name}相关，请结合你的相关知识判断。" if self.game_name else ""
        system_prompt = (
            f"你是一个游戏文本翻译助手，专注于将{self.game_name}相关的日文文本翻译为中文。"
            f"接下来会给你两张图像，图片1是当前说话人日文姓名，图片2是日文对话文本。"
            f"请识别图像1中的日文说话人名，结合你对{self.game_name}的了解用于理解说话人的背景和性格辅助翻译。不要把说话人信息输出到译文中。"
            f"之后识别图像2中的日文对话文本，并结合你对{self.game_name}的了解和说话人信息翻译为中文译文。当出现英文时不需要翻译，输出原文作为译文。"
            "翻译必须在优先保证语义通顺的情况下保留原文的换行与符号，绝对禁止添加与原文无关的内容，保留原文使用的『』，不需要替换为‘’或''，禁止自行添加标点，禁止在句尾保留“ッ”等日文助词。"
            "输出格式要求为：必须且仅能是符合Schema的JSON，包括speaker_name、original_text、translated_text三个字段。"
            f"speaker_name字段为说话人姓名，保持日文原文，不需要翻译。original_text字段为图像2中识别的日文原文文本。translated_text字段为中文译文。"
        )
        # Only include extra requirements sentence when provided
        extra_requirements_text = str(extra_requirements or "").strip()
        if extra_requirements_text:
            system_prompt = system_prompt.replace(
                "输出格式要求为：", f"翻译需要满足：{extra_requirements_text}。输出格式要求为："
            )
        if self.enable_web_search:
            system_prompt += self._search_hint_cn
        refs: list[str] = []
        if history_items:
            for idx, item in enumerate(history_items, start=1):
                refs.append(
                    f"{idx}. time={item.get('time','')}; speaker={item.get('speaker','')}; "
                    f"original={item.get('original','')}; translation={item.get('translation','')}"
                )
        history_text = "\n".join(refs) if refs else "(none)"
        user_text = (
            "下面会提供两张图片用于识别和翻译，"
            "图片1：当前说话人。图片2：当前对话文本。"
        )
        custom = str(custom_prompt or "").strip()
        if custom:
            user_text += f"\n附加要求：{custom}"
        
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
            f"{self.base_url}/responses",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_err: Exception | None = None
        empty_limit = 3
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
                translated_text = _normalize_quotes_for_subtitle(translated_text)
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
    ) -> tuple[str, dict[str, int]]:
        src = str(original_text or "").strip()
        if not src:
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        game_hint = f"文本内容与{self.game_name}相关，请结合你的相关知识判断。" if self.game_name else ""
        custom = str(custom_prompt or "").strip()
        system_prompt = (
            f"你是一个游戏文本翻译助手，专注于将{self.game_name}相关的{self.source_language}文本翻译为{self.target_language}。"
            f"接下来会给你当前说话人信息和已经识别出的{self.source_language}原文文本。"
            f"请结合你对{self.game_name}的了解和给出的说话人信息，将原文翻译为{self.target_language}译文。说话人信息仅用于参考，不要把说话人信息输出到译文中。"
            f"{game_hint}原文也可能为纯英文，此时不需要翻译，直接输出原文作为译文即可。"
            "翻译必须在语义通顺的情况下尽可能保留原文的换行与符号，绝对禁止添加与原文无关的内容，保留原文使用的『』，不需要替换为‘’或''。"
            "输出格式要求为：必须且仅能是符合Schema的JSON，包括translated_text一个字段。"
            f"translated_text 字段为将给定原文翻译成{self.target_language}的译文。"
        )
        if self.enable_web_search:
            system_prompt += self._search_hint_cn
        user_text = (
            f"当前说话人（仅作语气参考，不需要输出）：{speaker}\n"
            f"当前{self.source_language}原文：\n{src}\n"
            "请返回JSON，字段为：translated_text。"
        )
        if custom:
            user_text += f"\n附加要求：{custom}"
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
            f"{self.base_url}/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        empty_limit = 3
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
                translated = _normalize_quotes_for_subtitle(translated)
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
    ) -> tuple[str, str, dict[str, int]]:
        game_hint = f"图像中的文本内容与{self.game_name}相关，请结合你的相关知识判断。" if self.game_name else ""
        system_prompt = (
            f"你是一个游戏文本翻译助手，专注于将{self.game_name}相关的{self.source_language}文本翻译为{self.target_language}。"
            f"接下来会给你一张图像，图像中可能是标题或文本区域，不包含独立的说话人姓名图像。"
            f"请识别图像中的{self.source_language}文本，并结合你对{self.game_name}的了解翻译为{self.target_language}译文。"
            f"{game_hint}原文也可能为纯英文，此时不需要翻译，直接输出原文作为译文即可。"
            "翻译必须在语义通顺的情况下尽可能保留原文的换行与符号，绝对禁止添加与原文无关的内容，保留原文使用的『』，不需要替换为‘’或''。"
            "输出格式要求为：必须且仅能是符合Schema的JSON，包括original_text、translated_text两个字段。"
            f"original_text 字段为图像中识别的{self.source_language}原文文本。translated_text 字段为将original_text翻译成{self.target_language}的译文。"
        )
        if self.enable_web_search:
            system_prompt += self._search_hint_cn
        refs: list[str] = []
        if history_items:
            for idx, item in enumerate(history_items, start=1):
                refs.append(
                    f"{idx}. time={item.get('time','')}; speaker={item.get('speaker','')}; "
                    f"original={item.get('original','')}; translation={item.get('translation','')}"
                )
        history_text = "\n".join(refs) if refs else "(none)"
        user_text = (
            "下面提供一张图像（标题或文本区域），请执行OCR并翻译。\n"
            "给出的前文仅作风格和术语参考，不一定与当前图像直接相关。\n"
            # f"前文参考：\n{history_text}\n"
            "请返回JSON，字段为：original_text、translated_text。"
        )
        custom = str(custom_prompt or "").strip()
        if custom:
            user_text += f"\n附加要求：{custom}"
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
            f"{self.base_url}/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        empty_limit = 3
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
                translated_text = _normalize_quotes_for_subtitle(translated_text)
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
        return _has_kana_leak_from_original(original_text, translated_text, min_len=min_len)

    def _has_reviewable_kanji_overlap(
        self,
        original_text: str,
        translated_text: str,
        min_len: int = 3,
    ) -> bool:
        return _has_kanji_overlap_from_original(original_text, translated_text, min_len=min_len)

    def _append_io_log(self, record: dict[str, Any]) -> None:
        if not self.io_log_enabled or self.io_log_path is None:
            return
        try:
            self.io_log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with self._io_lock:
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

    def _to_data_url_from_base64(self, b64_str: str, mime: str = "image/png") -> str:
        """Convert a base64-encoded image string to data URL."""
        return f"data:{mime};base64,{b64_str}"

    def _to_data_url_from_bytes(self, image_bytes: bytes, mime: str = "image/png") -> str:
        """Convert raw image bytes to data URL."""
        b64 = base64.b64encode(image_bytes).decode("ascii")
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
