from __future__ import annotations

import json
import re
from typing import Any, Callable
from urllib import error as urlerror
from urllib import request

from .translation_runtime import (
    available_translation_model_profiles,
    load_api_key_for_profile,
    resolve_translation_model_profile,
)


SKIP_DIALOGUE_TYPES = {"blank_no_name", "blank", "title"}


def default_auto_review_profile(tr_cfg: dict[str, Any], requested: str = "") -> str:
    requested = str(requested or "").strip()
    if requested:
        return requested
    profiles = available_translation_model_profiles(tr_cfg, "ocr_chat_completions")
    for name in profiles:
        if name == "qwen3.6-plus":
            return name
    for name in profiles:
        if "qwen" in name.lower():
            return name
    return profiles[0] if profiles else "qwen3.6-plus"


def dialogue_review_entries_from_cache_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        dialogue_type = str(entry.get("dialogue_type", "") or "").strip().lower()
        if dialogue_type in SKIP_DIALOGUE_TYPES:
            continue
        original = str(entry.get("text_original", "") or "").strip()
        translation = str(entry.get("translation_subtitle", "") or "").strip()
        if not original or not translation:
            continue
        try:
            segment_id = int(entry.get("segment_id", 0) or 0)
        except Exception:
            continue
        if segment_id <= 0:
            continue
        out.append(
            {
                "id": segment_id,
                "speaker": str(entry.get("speaker", "") or "").strip(),
                "original": original,
                "translation": translation,
            }
        )
    return out


def chunks(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    if chunk_size <= 0:
        return [items] if items else []
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _auth_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = str(api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def build_auto_review_messages(glossary: str, entries: list[dict[str, Any]], review_mode: str = "thorough") -> list[dict[str, str]]:
    mode_prompt = (
        "### 检查方式\n"
        "**请按 entries 数组顺序逐条检查，不要只做整体判断。**\n"
        "**每一条都必须对照 original、speaker、translation 和术语表。**\n"
        "**发现术语冲突、漏译、误译、多译、明显不符合角色语气时，输出该条。**\n"
        if review_mode == "thorough"
        else ""
    )
    system_prompt = (
        "### 角色\n"
        "**你是游戏字幕翻译校对助手，负责检查现有译文是否需要根据原文、说话人和术语表修正。**\n"
        "### 输入\n"
        "**你会收到术语表/额外要求，以及若干字幕条目。**\n"
        "**每个字幕条目包含 id、speaker、original、translation。**\n"
        "**id 只是定位编号，请勿翻译、改写或解释 id。**\n"
        "### 任务\n"
        "**检查每条 translation 是否准确表达 original，并保持术语表、角色语气和前后用语一致。**\n"
        "**只输出确实需要修改的条目。无需修改的条目不要输出。**\n"
        "**如果修改后的译文与当前 translation 完全相同，不要输出该条。**\n"
        "### 校对规则\n"
        "- **术语表优先**：若 translation 与术语表冲突，需要修改。\n"
        "- **语义优先**：修正误译、漏译、过度发挥、不自然表达。\n"
        "- **通顺润色**：如果 translation 明显拗口、生硬、病句或不符合自然中文表达，可以在不改变原意的前提下小幅润色。\n"
        "- **润色边界**：润色必须保留 original 的信息、语气、称呼、术语和情绪强度；不得新增原文没有的信息，也不得删除原文已有信息。\n"
        "- **禁止重写正确译文**：如果 translation 准确、通顺且符合术语表，不要因为个人风格偏好改写。\n"
        "- **禁止添加**：不要添加原文没有的信息。\n"
        "- **保留符号**：尽量保留原文使用的『』等符号语气。\n"
        "- **不确定则不修改**：如果无法判断，不要输出该 id。\n"
        "- **必须真实修改**：输出的 translation 必须与输入的 translation 不同。\n"
        f"{mode_prompt}"
        "### 输出要求\n"
        "**只输出 JSON，不要 Markdown，不要在 JSON 外解释。**\n"
        "**必须严格使用以下 JSON 格式，字段名必须完全一致。**\n"
        "**字段名只允许使用 id、translation、reason。**\n"
        '{"updates":[{"id":数字,"translation":"修改后的译文","reason":"修改原因"}]}\n'
        "**如果没有需要修改的条目，输出：**\n"
        '{"updates":[]}\n'
    )
    user_payload = {"glossary": glossary, "entries": entries}
    user_prompt = (
        "### 待校对数据\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n"
        "**请只返回符合指定格式的 JSON：**"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _profile_chat_params(tr_cfg: dict[str, Any], profile_name: str) -> tuple[float | None, float | None, int | None]:
    raw_profiles = tr_cfg.get("model_profiles")
    if not isinstance(raw_profiles, dict):
        return None, None, None
    raw = raw_profiles.get(profile_name)
    if not isinstance(raw, dict):
        return None, None, None
    temperature = raw.get("temperature")
    if temperature is not None:
        temperature = float(temperature)
    top_p = raw.get("top_p")
    if top_p is not None:
        top_p = float(top_p)
    top_k = raw.get("top_k")
    if top_k is not None:
        top_k = int(top_k)
    return temperature, top_p, top_k


def call_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int,
    timeout_sec: int,
    stream: bool = False,
    stream_callback: Callable[[str], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens

    req = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_auth_headers(api_key),
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            if stream:
                content_parts: list[str] = []
                finish_reason: Any = None
                usage: dict[str, Any] = {}
                stream_errors: list[str] = []
                reasoning_ticks = 0
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception as exc:
                        stream_errors.append(f"{exc.__class__.__name__}: {line[:200]}")
                        continue
                    if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        delta = {}
                    content = str(delta.get("content") or "")
                    reasoning = str(delta.get("reasoning_content") or delta.get("reasoning") or "")
                    if content:
                        if stream_callback is not None:
                            stream_callback(content)
                        content_parts.append(content)
                    elif reasoning:
                        reasoning_ticks += 1
                        if stream_callback is not None and (reasoning_ticks == 1 or reasoning_ticks % 16 == 0):
                            stream_callback("·")
                content = "".join(content_parts).strip()
                if not content:
                    raise RuntimeError("stream response content is empty")
                return content, {
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "stream": True,
                    "stream_errors": stream_errors,
                }
            data = json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc

    choices = data.get("choices", []) if isinstance(data, dict) else []
    if not choices:
        raise RuntimeError(f"response has no choices: {json.dumps(data, ensure_ascii=False)[:1000]}")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = str(message.get("content", "") or "").strip()
    if not content:
        raise RuntimeError(f"response content is empty: {json.dumps(data, ensure_ascii=False)[:1000]}")
    return content, {
        "finish_reason": choices[0].get("finish_reason") if isinstance(choices[0], dict) else None,
        "usage": data.get("usage", {}) if isinstance(data, dict) else {},
        "stream": False,
    }


def _escape_control_chars_in_strings(raw: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    for ch in raw:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = not in_string
            continue
        if in_string and ch in {"\n", "\r", "\t"}:
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
        else:
            out.append(ch)
    return "".join(out)


def _try_parse_json_dict(raw: str) -> dict[str, Any] | None:
    candidates = [raw, _escape_control_chars_in_strings(raw)]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed, _ = decoder.raw_decode(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    try:
        import dirtyjson  # type: ignore

        parsed = dirtyjson.loads(raw)
        normalized = json.loads(json.dumps(parsed, ensure_ascii=False))
        if isinstance(normalized, dict):
            return normalized
    except Exception:
        return None
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", raw, flags=re.IGNORECASE)
        if fenced:
            raw = fenced.group(1).strip()
    parsed = _try_parse_json_dict(raw)
    if parsed is not None:
        return parsed
    start = raw.find("{")
    if start < 0:
        raise ValueError("LLM 输出中没有 JSON object")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = _try_parse_json_dict(raw[start : idx + 1])
                if parsed is not None:
                    return parsed
                break
    raise ValueError("无法从 LLM 输出解析 JSON object")


def _build_repair_messages(raw_text: str) -> list[dict[str, str]]:
    system_prompt = (
        "### 角色\n"
        "**你是 JSON 修复器。**\n"
        "### 任务\n"
        "**把用户提供的文本修复为合法 JSON。**\n"
        "**不要新增、删除或改写 id、translation、reason 的语义内容。**\n"
        "### 输出要求\n"
        "**只输出 JSON，不要 Markdown，不要解释。**\n"
        "**必须使用以下格式：**\n"
        '{"updates":[{"id":数字,"translation":"修改后的译文","reason":"修改原因"}]}\n'
        "**如果原文表示没有修改，输出：**\n"
        '{"updates":[]}\n'
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "### 待修复文本\n" + raw_text.strip()},
    ]


def parse_with_repair(
    raw_text: str,
    *,
    parse_retries: int,
    base_url: str,
    api_key: str,
    model: str,
    timeout_sec: int,
    repair_max_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repair_attempts: list[dict[str, Any]] = []
    try:
        return extract_json_object(raw_text), repair_attempts
    except Exception as first_exc:
        last_exc: Exception = first_exc
    for attempt in range(1, max(0, parse_retries) + 1):
        repaired_text, meta = call_chat_completions(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=_build_repair_messages(raw_text),
            temperature=0.0,
            top_p=None,
            top_k=None,
            max_tokens=repair_max_tokens,
            timeout_sec=timeout_sec,
            stream=False,
        )
        record: dict[str, Any] = {"attempt": attempt, "response_meta": meta, "raw_response": repaired_text}
        try:
            parsed = extract_json_object(repaired_text)
            record["success"] = True
            repair_attempts.append(record)
            return parsed, repair_attempts
        except Exception as exc:
            last_exc = exc
            record["success"] = False
            record["error"] = f"{exc.__class__.__name__}: {exc}"
            repair_attempts.append(record)
    raise last_exc


def validate_updates(parsed: dict[str, Any], id_map: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    updates = parsed.get("updates", [])
    if not isinstance(updates, list):
        raise ValueError("JSON 字段 updates 必须是数组")
    out: list[dict[str, Any]] = []
    for item in updates:
        if not isinstance(item, dict):
            continue
        try:
            segment_id = int(item.get("id", 0) or 0)
        except Exception:
            continue
        new_text = str(
            item.get("translation")
            or item.get("translation_subtitle")
            or item.get("translationSubtitle")
            or item.get("new_translation")
            or item.get("newTranslation")
            or ""
        ).strip()
        if segment_id not in id_map or not new_text:
            continue
        old = id_map[segment_id]
        old_text = str(old.get("translation", "") or "").strip()
        out.append(
            {
                "id": segment_id,
                "speaker": old.get("speaker", ""),
                "original": old.get("original", ""),
                "old_translation": old_text,
                "new_translation": new_text,
                "reason": str(item.get("reason") or item.get("理由") or "").strip(),
                "changed": new_text != old_text,
            }
        )
    return out


def run_auto_review_entries(
    *,
    entries: list[dict[str, Any]],
    glossary: str,
    tr_cfg: dict[str, Any],
    model_profile: str = "",
    chunk_size: int = 80,
    timeout_sec: int = 120,
    temperature: float | None = None,
    max_tokens: int = 0,
    parse_retries: int = 1,
    repair_max_tokens: int = 0,
    review_mode: str = "thorough",
    stream: bool = False,
    log_fn: Callable[[str], None] | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_profile = default_auto_review_profile(tr_cfg, model_profile)
    profile = resolve_translation_model_profile(tr_cfg, "ocr_chat_completions", selected_profile)
    api_key = load_api_key_for_profile(profile)
    profile_temp, profile_top_p, profile_top_k = _profile_chat_params(tr_cfg, selected_profile)
    temp = temperature if temperature is not None else profile_temp
    top_p = profile_top_p
    top_k = profile_top_k
    all_updates: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    request_chunks = chunks(entries, int(chunk_size))
    if log_fn:
        log_fn(
            f"自动review: profile={profile.name} model={profile.model} "
            f"entries={len(entries)} chunks={len(request_chunks)} chunk_size={chunk_size}"
        )
    for idx, chunk in enumerate(request_chunks, start=1):
        if not chunk:
            continue
        id_map = {int(e["id"]): e for e in chunk}
        messages = build_auto_review_messages(glossary, chunk, review_mode)
        if log_fn:
            log_fn(f"自动review chunk {idx}/{len(request_chunks)} ids={chunk[0]['id']}..{chunk[-1]['id']}")
        raw_text, meta = call_chat_completions(
            base_url=profile.base_url,
            api_key=api_key,
            model=profile.model,
            messages=messages,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            stream=stream,
            stream_callback=stream_callback,
        )
        try:
            parsed, repair_attempts = parse_with_repair(
                raw_text,
                parse_retries=parse_retries,
                base_url=profile.base_url,
                api_key=api_key,
                model=profile.model,
                timeout_sec=timeout_sec,
                repair_max_tokens=repair_max_tokens,
            )
            updates = validate_updates(parsed, id_map)
        except Exception as exc:
            updates = []
            repair_attempts = []
            parse_errors.append(
                {
                    "chunk": idx,
                    "ids": f"{chunk[0]['id']}..{chunk[-1]['id']}",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "raw_response": raw_text,
                }
            )
        all_updates.extend(updates)
        if log_fn:
            changed = sum(1 for item in updates if item.get("changed"))
            log_fn(
                f"自动review chunk {idx}/{len(request_chunks)} done: "
                f"updates={len(updates)} changed={changed} meta={json.dumps(meta, ensure_ascii=False)}"
            )
            if repair_attempts:
                log_fn(f"自动review chunk {idx}: repair_attempts={len(repair_attempts)}")
    changed_updates = [item for item in all_updates if item.get("changed")]
    report = {
        "profile": profile.name,
        "model": profile.model,
        "base_url": profile.base_url,
        "model_updates": all_updates,
        "changed_updates": changed_updates,
        "model_update_count": len(all_updates),
        "changed_count": len(changed_updates),
        "parse_errors": parse_errors,
        "parse_error_count": len(parse_errors),
    }
    return all_updates, report


def apply_updates_to_cache_entries(entries: list[dict[str, Any]], updates: list[dict[str, Any]]) -> int:
    by_id = {int(item["id"]): str(item.get("new_translation", "") or "").strip() for item in updates if item.get("changed")}
    count = 0
    for entry in entries:
        try:
            segment_id = int(entry.get("segment_id", 0) or 0)
        except Exception:
            continue
        new_text = by_id.get(segment_id)
        if not new_text:
            continue
        if str(entry.get("translation_subtitle", "") or "").strip() == new_text:
            continue
        entry["translation_subtitle"] = new_text
        count += 1
    return count
