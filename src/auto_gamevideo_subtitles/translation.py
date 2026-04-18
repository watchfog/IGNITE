from __future__ import annotations

import json
import base64
import time
from pathlib import Path
from typing import Any
from urllib import request


def load_deepseek_api(path: str | Path) -> tuple[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    api = ""
    key = ""
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k.strip().lower() == "api":
            api = v.strip()
        if k.strip().lower() == "key":
            key = v.strip()
    if not api or not key:
        raise ValueError("deepseek api/key missing in config file")
    return api, key


class DeepSeekTranslator:
    def __init__(self, api_url: str, api_key: str, model: str = "deepseek-chat") -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def translate_ja_to_zh_cn(self, text: str) -> str:
        if not text.strip():
            return ""
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个严格的日文到简体中文字幕翻译器。",
                },
                {
                    "role": "user",
                    "content": (
                        "将我给的原文翻译为为简体中文, 待翻译文本内容为战姬绝唱XD "
                        "Unlimited相关的日文内容 注意保留换行符和格式,绝对禁止添加和原文无关的翻译, "
                        "不要解释翻译方式，直接给出译文\n"
                        f"{text}"
                    ),
                },
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )


def load_api_key(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"api key file is empty: {path}")
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
    raise ValueError(f"cannot parse api key from file: {path}")


class BailianVlmTranslator:
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3.6-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_sec: int = 30,
        max_retries: int = 2,
        retry_delay_sec: float = 1.5,
        disable_env_proxy: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max(0, int(max_retries))
        self.retry_delay_sec = max(0.0, float(retry_delay_sec))
        self._opener = (
            request.build_opener(request.ProxyHandler({}))
            if disable_env_proxy
            else request.build_opener()
        )

    def translate_image_ja_to_zh_cn(self, image_path: str | Path, speaker: str) -> str:
        prompt = (
            "识别图像中的日文，翻译为简体中文。日文内容为战姬绝唱XD Unlimited的相关内容，"
            f"说话人为{speaker}。说话人信息仅作语气和背景参考，不需要输出到译文中。"
            "注意保留换行符和格式，绝对禁止添加和原文无关的翻译，"
            "输出且仅输出译文。"
        )
        image_url = self._to_data_url(image_path)
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_err: Exception | None = None
        attempts = self.max_retries + 1
        for i in range(attempts):
            try:
                with self._opener.open(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                return self._extract_text(data).strip()
            except Exception as exc:
                last_err = exc
                if i + 1 >= attempts:
                    break
                time.sleep(self.retry_delay_sec)
        if last_err is not None:
            raise last_err
        return ""

    def _to_data_url(self, image_path: str | Path) -> str:
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
