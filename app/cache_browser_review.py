from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_gamevideo_subtitles.config import load_config  # noqa: E402
from auto_gamevideo_subtitles.translation_runtime import (  # noqa: E402
    BailianVlmTranslator,
    _normalize_quotes_for_subtitle,
    load_api_key,
)


ROI_KEYS = ["name_roi", "dialogue_roi", "title_ocr_roi"]
ROI_COLORS = {
    "name_roi": "#ffbf00",
    "dialogue_roi": "#0896e0",
    "title_ocr_roi": "#c0c0c0",
}


class CacheReviewApp:
    def __init__(self, cache_path: Path, video_path: Path | None, config_path: Path) -> None:
        self.cache_path = cache_path.resolve()
        self.video_path = video_path.resolve() if video_path else None
        self.config_path = config_path.resolve()

        self.root = tk.Tk()
        self.root.title("字幕校对工具 - IGNITE")
        self.root.geometry("1700x1020")

        self.status_var = tk.StringVar(value="就绪")
        self.seg_info_var = tk.StringVar(value="-")
        self.review_meta_var = tk.StringVar(value="")
        self.goto_var = tk.StringVar(value="1")
        self.time_var = tk.StringVar(value="0.00")
        self.roi_key_var = tk.StringVar(value="dialogue_roi")
        self.title_start_var = tk.StringVar(value="0.00")
        self.title_end_var = tk.StringVar(value="2.00")
        self.title_info_var = tk.StringVar(value="")
        self.review_web_search_var = tk.BooleanVar(value=False)

        self.entries: list[dict[str, Any]] = []
        self.cache_payload: dict[str, Any] = {}
        self.current_index = 0

        self.cap: cv2.VideoCapture | None = None
        self.video_w = 0
        self.video_h = 0
        self.duration_sec = 0.0
        self.fps = 25.0
        self.current_sec = 0.0
        self.ffmpeg_path = (ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe").resolve()
        self.ffmpeg_hwaccel = "auto"
        self.ffmpeg_two_stage_seek = True
        self.ffmpeg_two_stage_margin_sec = 2.0
        self._seek_after_id: str | None = None
        self._pending_seek_sec: float | None = None
        self._last_fast_seek_frame_idx: int | None = None
        self._suppress_scale_callback = False
        self._precise_req_id = 0
        self._decode_overlay_visible = False
        self._decode_spinner_phase = 0
        self._decode_spinner_after_id: str | None = None
        self._decode_spinner_chars = ["|", "/", "-", "\\"]
        self.current_frame_rgb: Any | None = None
        self._img: ImageTk.PhotoImage | None = None
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._render_w = 1
        self._render_h = 1
        self._offset_x = 0
        self._offset_y = 0
        self._is_seeking = False
        self._frame_cache: OrderedDict[int, Any] = OrderedDict()
        self._frame_cache_max = 32
        self._cache_lock = threading.Lock()
        self._prefetch_segment_radius = 4
        self._prefetch_max_workers = 3
        self._prefetch_request_seq = 0
        self._prefetch_job_seq = 0
        self._prefetch_jobs_lock = threading.Lock()
        self._prefetch_jobs: dict[int, dict[str, Any]] = {}
        self._prefetch_targets: list[int] = []
        self._last_prefetch_center_idx: int | None = None

        self.default_rois: dict[str, list[int]] = {}
        self.review_rois: dict[str, list[int]] = {}
        self.drag_start: tuple[int, int] | None = None
        self.drag_now: tuple[int, int] | None = None

        self.cfg: dict[str, Any] = {}
        self.translator: BailianVlmTranslator | None = None
        self.last_review_result: dict[str, Any] | None = None
        self._busy = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._bind_keys()
        self._load_config()
        self._load_cache()
        self._resolve_video_from_cache()
        if self.video_path and self.video_path.exists():
            self._open_video(self.video_path)
        self._show_segment(0, request_prefetch=False)
        if self.cap is not None and self.entries:
            self._request_prefetch(0)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        self.cache_var = tk.StringVar(value=str(self.cache_path))
        self.video_var = tk.StringVar(value=str(self.video_path) if self.video_path else "")
        self.config_var = tk.StringVar(value=str(self.config_path))

        ttk.Label(top, text="缓存").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.cache_var, width=108).grid(row=0, column=1, columnspan=8, sticky="ew", padx=(6, 4))
        ttk.Button(top, text="重载缓存", command=self._load_cache).grid(row=0, column=9, padx=2)

        ttk.Label(top, text="视频").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.video_var, width=108).grid(row=1, column=1, columnspan=8, sticky="ew", padx=(6, 4))
        ttk.Button(top, text="加载视频", command=self._reload_video).grid(row=1, column=9, padx=2)

        ttk.Label(top, text="配置").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.config_var, width=108).grid(row=2, column=1, columnspan=8, sticky="ew", padx=(6, 4))
        ttk.Button(top, text="重载配置", command=self._load_config).grid(row=2, column=9, padx=2)

        nav = ttk.Frame(top)
        nav.grid(row=3, column=0, columnspan=10, sticky="ew", pady=(6, 0))
        ttk.Button(nav, text="上一段 (←)", command=self._prev_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="下一段 (→)", command=self._next_segment).pack(side=tk.LEFT, padx=2)
        ttk.Label(nav, text="跳转段号").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(nav, textvariable=self.goto_var, width=10).pack(side=tk.LEFT)
        ttk.Button(nav, text="跳转", command=self._jump_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="保存当前段", command=self._save_current_entry).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(nav, text="保存全部", command=self._save_cache_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            nav,
            text="生成字幕（当前Cache）",
            command=self._generate_subtitles_from_cache,
        ).pack(side=tk.RIGHT, padx=(12, 2))

        title_bar = ttk.Frame(top)
        title_bar.grid(row=4, column=0, columnspan=10, sticky="ew", pady=(6, 0))
        ttk.Label(title_bar, text="标题开始(秒)").pack(side=tk.LEFT)
        ttk.Entry(title_bar, textvariable=self.title_start_var, width=14).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(title_bar, text="标题结束(秒)").pack(side=tk.LEFT)
        ttk.Entry(title_bar, textvariable=self.title_end_var, width=14).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(title_bar, text="翻译信息").pack(side=tk.LEFT)
        ttk.Entry(title_bar, textvariable=self.title_info_var, width=48).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(
            title_bar,
            text="插入/更新Title(title_ocr_roi+VLM)",
            command=self._insert_title_segment_from_roi,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(top, textvariable=self.seg_info_var, foreground="#1f5f99").grid(row=5, column=0, columnspan=10, sticky="w", pady=(6, 0))
        ttk.Label(top, textvariable=self.status_var, foreground="#2f6f3e").grid(row=6, column=0, columnspan=10, sticky="w")
        top.columnconfigure(1, weight=1)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.canvas = tk.Canvas(left, bg="#111111", highlightthickness=1, highlightbackground="#444")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        bar = ttk.Frame(left, padding=(0, 6, 0, 0))
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="时间(秒)").pack(side=tk.LEFT)
        self.time_scale = ttk.Scale(bar, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=self._on_scale_change)
        self.time_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self.time_scale.bind("<ButtonRelease-1>", self._on_scale_release)
        ttk.Entry(bar, textvariable=self.time_var, width=10).pack(side=tk.LEFT)
        ttk.Button(bar, text="跳转时间", command=self._jump_time).pack(side=tk.LEFT, padx=(6, 0))

        frame_bar = ttk.Frame(left, padding=(0, 4, 0, 0))
        frame_bar.pack(fill=tk.X)
        ttk.Button(frame_bar, text="上一帧", command=self._prev_frame).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(frame_bar, text="下一帧", command=self._next_frame).pack(side=tk.LEFT)
        ttk.Button(frame_bar, text="前10帧", command=self._prev_10_frames).pack(side=tk.LEFT, padx=(12, 6))
        ttk.Button(frame_bar, text="后10帧", command=self._next_10_frames).pack(side=tk.LEFT)

        roi_bar = ttk.Frame(left, padding=(0, 6, 0, 0))
        roi_bar.pack(fill=tk.X)
        ttk.Label(roi_bar, text="拖拽编辑ROI").pack(side=tk.LEFT)
        ttk.OptionMenu(roi_bar, self.roi_key_var, self.roi_key_var.get(), *ROI_KEYS).pack(side=tk.LEFT, padx=6)
        ttk.Button(roi_bar, text="恢复默认ROI", command=self._reset_rois).pack(side=tk.LEFT, padx=2)

        # Right panel: vertical stack (JSON on top, review on bottom)
        stack = ttk.Panedwindow(right, orient=tk.VERTICAL)
        stack.pack(fill=tk.BOTH, expand=True)
        json_panel = ttk.Frame(stack)
        review_panel = ttk.Frame(stack)
        stack.add(json_panel, weight=3)
        stack.add(review_panel, weight=4)

        ttk.Label(json_panel, text="当前段 JSON（可编辑）").pack(anchor="w")
        self.json_text = tk.Text(json_panel, height=16, wrap=tk.NONE, undo=True, font=("Consolas", 10))
        self.json_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        ttk.Label(review_panel, text="Review 复译").pack(anchor="w")
        prompt_box = ttk.LabelFrame(review_panel, text="自定义 Prompt（可选，会追加到复译请求）", padding=6)
        prompt_box.pack(fill=tk.X, pady=(6, 0))
        self.custom_prompt_text = tk.Text(prompt_box, height=4, wrap=tk.WORD, font=("Consolas", 10))
        self.custom_prompt_text.pack(fill=tk.X)

        btns = ttk.Frame(review_panel, padding=(0, 6, 0, 0))
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="1) 重新截图并复译", command=self._review_by_new_crops).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="2) 用原文重译", command=self._review_by_text_only).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="一键替换(人名+原文+译文)", command=self._apply_last_result_all).pack(side=tk.LEFT, padx=(16, 2))
        ttk.Button(btns, text="仅替换译文", command=self._apply_last_result_translation).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(btns, text="复译联网", variable=self.review_web_search_var).pack(side=tk.RIGHT, padx=(8, 2))

        review_head = ttk.Frame(review_panel)
        review_head.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(review_head, text="复译结果").pack(side=tk.LEFT, anchor="w")
        ttk.Label(review_head, textvariable=self.review_meta_var, foreground="#1f5f99").pack(side=tk.RIGHT, anchor="e")
        self.review_text = tk.Text(review_panel, height=14, wrap=tk.WORD, font=("Consolas", 10))
        self.review_text.pack(fill=tk.BOTH, expand=True)

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", self._on_prev_segment_shortcut)
        self.root.bind("<Right>", self._on_next_segment_shortcut)
        self.root.bind("<Control-s>", lambda _e: self._save_current_entry())
        self.root.bind("<Control-S>", lambda _e: self._save_current_entry())

    def _is_focus_on_input_widget(self) -> bool:
        widget = self.root.focus_get()
        if widget is None:
            return False
        input_types = (tk.Entry, tk.Text, tk.Spinbox, ttk.Entry, ttk.Combobox, ttk.Spinbox)
        if isinstance(widget, input_types):
            return True
        # Fallback for widget wrappers where isinstance may not match reliably.
        widget_class = str(widget.winfo_class()).lower()
        return widget_class in {"entry", "text", "spinbox", "tentry", "tcombobox", "tspinbox"}

    def _on_prev_segment_shortcut(self, _e: tk.Event[Any]) -> str | None:
        if self._is_focus_on_input_widget():
            return None
        self._prev_segment()
        return "break"

    def _on_next_segment_shortcut(self, _e: tk.Event[Any]) -> str | None:
        if self._is_focus_on_input_widget():
            return None
        self._next_segment()
        return "break"

    def _get_custom_prompt(self) -> str:
        return self.custom_prompt_text.get("1.0", tk.END).strip()

    def _set_status_threadsafe(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def _set_review_meta_threadsafe(self, text: str) -> None:
        self.root.after(0, lambda: self.review_meta_var.set(text))

    def _usage_to_meta(self, usage: dict[str, Any] | None, running: bool = False) -> str:
        if running:
            return "正在进行中..."
        if not isinstance(usage, dict):
            return ""
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        tt = int(usage.get("total_tokens", pt + ct) or (pt + ct))
        return f"tokens: in={pt} out={ct} total={tt}"

    def _build_review_entry(
        self,
        base_entry: dict[str, Any],
        speaker: str,
        original: str,
        translated: str,
    ) -> dict[str, Any]:
        normalized = _normalize_quotes_for_subtitle(str(translated or "").strip())
        out: dict[str, Any] = {
            "segment_id": base_entry.get("segment_id"),
            "time_start": base_entry.get("time_start"),
            "time_end": base_entry.get("time_end"),
            "srt_start": base_entry.get("srt_start"),
            "srt_end": base_entry.get("srt_end"),
            "dialogue_type": base_entry.get("dialogue_type", "speaker_dialogue"),
            "speaker": str(speaker or base_entry.get("speaker", "")),
            "text_original": str(original or ""),
            "translation_subtitle": normalized,
            "debug_subtitle": base_entry.get("debug_subtitle", ""),
        }
        return out

    def _run_bg(self, title: str, fn: Callable[[], None]) -> None:
        if self._busy:
            self.status_var.set("任务执行中，请稍候")
            return
        self._busy = True
        self.review_meta_var.set(self._usage_to_meta(None, running=True))

        def _runner() -> None:
            try:
                fn()
            except Exception as exc:
                self._set_status_threadsafe(f"{title} 失败: {exc.__class__.__name__}: {exc}")
            finally:
                self._busy = False

        threading.Thread(target=_runner, daemon=True).start()

    def _load_config(self) -> None:
        p = Path(self.config_var.get().strip() or self.config_path).resolve()
        self.config_path = p
        self.cfg = load_config(p)
        self.translator = None
        tools_cfg = self.cfg.get("tools", {})
        ffmpeg_cfg = str(tools_cfg.get("ffmpeg_path", "") or "").strip()
        if ffmpeg_cfg:
            fp = Path(ffmpeg_cfg)
            if not fp.is_absolute():
                fp = (ROOT / fp).resolve()
            self.ffmpeg_path = fp
        self.ffmpeg_hwaccel = str(tools_cfg.get("ffmpeg_hwaccel", "auto") or "auto").strip() or "auto"
        self.ffmpeg_two_stage_seek = bool(tools_cfg.get("ffmpeg_two_stage_seek", True))
        try:
            self.ffmpeg_two_stage_margin_sec = max(
                0.0,
                float(tools_cfg.get("ffmpeg_two_stage_margin_sec", 2.0) or 2.0),
            )
        except Exception:
            self.ffmpeg_two_stage_margin_sec = 2.0
        roi_cfg = self.cfg.get("roi", {})
        loaded: dict[str, list[int]] = {}
        for k in ROI_KEYS:
            v = roi_cfg.get(k)
            if v is None and k == "title_ocr_roi":
                v = roi_cfg.get("title_roi", roi_cfg.get("dialogue_roi"))
            if isinstance(v, list) and len(v) == 4:
                loaded[k] = [int(v[0]), int(v[1]), int(v[2]), int(v[3])]
        if loaded:
            self.default_rois = {k: list(v) for k, v in loaded.items()}
            if not self.review_rois:
                self.review_rois = {k: list(v) for k, v in loaded.items()}
        self.status_var.set(f"已加载配置: {p}")
        self._refresh_canvas()

    def _build_translator(self) -> BailianVlmTranslator:
        want_web_search = bool(self.review_web_search_var.get())
        if (
            self.translator is not None
            and bool(getattr(self.translator, "enable_web_search", False)) == want_web_search
        ):
            return self.translator
        self.translator = None
        tcfg = self.cfg.get("translation", {})
        game_cfg = self.cfg.get("game", {})
        raw_models = tcfg.get("vlm_models", [])
        models: list[str] = []
        if isinstance(raw_models, list):
            models = [str(x).strip() for x in raw_models if str(x).strip()]
        elif isinstance(raw_models, str) and raw_models.strip():
            models = [raw_models.strip()]
        selected_model = str(tcfg.get("model", "") or "").strip()
        if not models:
            models = ["qwen3.6-plus"]
        if selected_model and selected_model not in models:
            raise RuntimeError("Invalid config: translation.model must be in translation.vlm_models")
        if not selected_model:
            selected_model = models[0]
        api_key_file = str(tcfg.get("api_key_file", "")).strip()
        if not api_key_file:
            raise RuntimeError("translation.api_key_file 未配置")
        api_key_path = Path(api_key_file)
        if not api_key_path.is_absolute():
            api_key_path = (ROOT / api_key_path).resolve()
        api_key = load_api_key(api_key_path)
        self.translator = BailianVlmTranslator(
            api_key=api_key,
            model=selected_model,
            base_url=str(tcfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            temperature=float(tcfg.get("temperature", 1.3)),
            enable_thinking=bool(tcfg.get("enable_thinking", True)),
            thinking_budget=tcfg.get("thinking_budget", None),
            preserve_thinking=bool(tcfg.get("preserve_thinking", False)),
            timeout_sec=int(tcfg.get("timeout_sec", 45)),
            timeout_backoff_sec=int(tcfg.get("timeout_backoff_sec", 15)),
            max_retries=int(tcfg.get("max_retries", 2)),
            retry_delay_sec=float(tcfg.get("retry_delay_sec", 1.5)),
            disable_env_proxy=bool(tcfg.get("disable_env_proxy", True)),
            game_name=str(game_cfg.get("name", "")),
            source_language=str(game_cfg.get("source_language", "ja")),
            target_language=str(game_cfg.get("target_language", "zh-CN")),
            io_log_enabled=True,
            io_log_path=self.cache_path.parent / "review_vlm_io.log",
            log_fn=lambda s: self._set_status_threadsafe(f"[VLM] {s}"),
            enable_web_search=want_web_search,
        )
        return self.translator

    def _load_cache(self) -> None:
        p = Path(self.cache_var.get().strip() or self.cache_path).resolve()
        self.cache_path = p
        raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            self.cache_payload = raw
            self.entries = [x for x in raw.get("entries", []) if isinstance(x, dict)]
        elif isinstance(raw, list):
            self.cache_payload = {"version": 1, "entries": [x for x in raw if isinstance(x, dict)]}
            self.entries = self.cache_payload["entries"]
        else:
            raise RuntimeError("cache json 格式不支持")
        self.current_index = min(self.current_index, max(0, len(self.entries) - 1))
        self._init_title_time_defaults()
        self.status_var.set(f"已加载缓存: {self.cache_path}，共 {len(self.entries)} 段")
        self._show_segment(self.current_index)

    def _seg_numeric_id(self, entry: dict[str, Any]) -> int | None:
        raw = entry.get("segment_id")
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        m = re.match(r"^(\d+)", s)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _init_title_time_defaults(self) -> None:
        if not self.entries:
            self.title_start_var.set("0.00")
            self.title_end_var.set("2.00")
            return
        seg1_candidates: list[tuple[float, dict[str, Any]]] = []
        non_title: list[tuple[float, dict[str, Any]]] = []
        for e in self.entries:
            if str(e.get("dialogue_type", "")).strip().lower() == "title":
                continue
            try:
                st = float(e.get("time_start", 0.0) or 0.0)
            except Exception:
                st = 0.0
            non_title.append((st, e))
            if self._seg_numeric_id(e) == 1:
                seg1_candidates.append((st, e))
        if seg1_candidates:
            base_start = min(seg1_candidates, key=lambda x: x[0])[0]
        elif non_title:
            base_start = min(non_title, key=lambda x: x[0])[0]
        else:
            base_start = 0.0
        base_end = max(base_start + 2.0, base_start)
        self.title_start_var.set(f"{base_start:.2f}")
        self.title_end_var.set(f"{base_end:.2f}")

    def _resolve_video_from_cache(self) -> None:
        if self.video_path and self.video_path.exists():
            self.video_var.set(str(self.video_path))
            return
        video = self.cache_payload.get("video")
        if not video:
            return
        v = Path(str(video))
        if not v.is_absolute():
            # cache 里保存的是相对项目根目录的路径（与主程序一致）
            v = (ROOT / v).resolve()
        self.video_path = v
        self.video_var.set(str(v))

    def _reload_video(self) -> None:
        raw = self.video_var.get().strip()
        if not raw:
            return
        self.video_path = Path(raw).resolve()
        self._open_video(self.video_path)
        self._seek_to_segment(self.current_index, request_prefetch=False)
        if self.entries:
            self._request_prefetch(self.current_index)

    def _open_video(self, path: Path) -> None:
        self._last_fast_seek_frame_idx = None
        with self._cache_lock:
            self._frame_cache.clear()
        self._cancel_all_prefetch_jobs()
        self._prefetch_targets = []
        self._last_prefetch_center_idx = None
        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"打开视频失败: {path}")
        self.cap = cap
        self.video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        self.duration_sec = (frame_count / self.fps) if self.fps > 0 and frame_count > 0 else 0.0
        self.time_scale.configure(to=max(0.0, self.duration_sec))
        self._seek(0.0, request_prefetch=False)

    def _on_close(self) -> None:
        try:
            if self._decode_spinner_after_id is not None:
                try:
                    self.root.after_cancel(self._decode_spinner_after_id)
                except Exception:
                    pass
                self._decode_spinner_after_id = None
            self._cancel_all_prefetch_jobs()
            self._prefetch_targets = []
            self._last_prefetch_center_idx = None
            with self._cache_lock:
                self._frame_cache.clear()
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()

    def _save_current_entry(self) -> None:
        if not self.entries:
            return
        text = self.json_text.get("1.0", tk.END).strip()
        parsed = json.loads(text) if text else {}
        if not isinstance(parsed, dict):
            raise RuntimeError("当前JSON必须是对象")
        old_seg_id = self.entries[self.current_index].get("segment_id")
        new_seg_id = parsed.get("segment_id", old_seg_id)
        if old_seg_id is not None and new_seg_id != old_seg_id:
            parsed["segment_id"] = old_seg_id
            self.status_var.set(f"提示：segment_id 不应修改，已恢复为 {old_seg_id}")
        self.entries[self.current_index] = parsed
        self._update_seg_info(self.current_index)
        if self.status_var.get().startswith("提示：segment_id"):
            return
        self.status_var.set(f"已更新第 {self.current_index + 1} 段（内存）")

    def _save_cache_file(self) -> None:
        self.cache_payload["entries"] = self.entries
        self.cache_path.write_text(json.dumps(self.cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        sync_msg = self._sync_latest_cache_file()
        self.status_var.set(f"已保存: {self.cache_path} {sync_msg}".strip())

    def _sync_latest_cache_file(self) -> str:
        try:
            out_dir = self._resolve_output_dir_from_cache(self.cache_path)
            latest = out_dir / "translation_cache_latest.json"
            shutil.copy2(self.cache_path, latest)
            return f"(已同步: {latest})"
        except Exception as exc:
            return f"(同步 latest 失败: {exc})"

    def _resolve_output_dir_from_cache(self, cache_path: Path) -> Path:
        cur = cache_path.parent
        while cur != cur.parent:
            if cur.name == "work":
                return cur.parent
            cur = cur.parent
        return cache_path.parent

    def _build_subtitle_command(self, cache_path: Path, video_path: Path, config_path: Path, output_dir: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "src.auto_gamevideo_subtitles.pipeline",
            "--video",
            str(video_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--subtitles-from-cache",
            "--translation-cache",
            str(cache_path),
        ]

    def _generate_subtitles_from_cache(self) -> None:
        def _job() -> None:
            self._save_current_entry()
            self._save_cache_file()

            cache_path = self.cache_path.resolve()
            if not cache_path.exists():
                raise RuntimeError(f"cache不存在: {cache_path}")
            output_dir = self._resolve_output_dir_from_cache(cache_path)

            video_path = Path(self.video_var.get().strip()).resolve()
            if not video_path.exists():
                raise RuntimeError(f"视频不存在: {video_path}")
            config_path = Path(self.config_var.get().strip()).resolve()
            if not config_path.exists():
                raise RuntimeError(f"配置不存在: {config_path}")

            cmd = self._build_subtitle_command(
                cache_path=cache_path,
                video_path=video_path,
                config_path=config_path,
                output_dir=output_dir,
            )
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if proc.returncode != 0:
                tail = "\n".join((proc.stdout or "").splitlines()[-30:])
                raise RuntimeError(f"生成字幕失败(退出码={proc.returncode})\n{tail}")
            self._set_status_threadsafe(f"字幕已生成: {output_dir}")

        self._run_bg("生成字幕", _job)

    def _parse_time_input(self, text: str) -> float:
        s = str(text or "").strip()
        if not s:
            raise RuntimeError("时间不能为空")
        if re.fullmatch(r"\d+(\.\d+)?", s):
            return max(0.0, float(s))
        s = s.replace("，", ",").replace("：", ":")
        m = re.fullmatch(r"(\d+):(\d{1,2}):(\d{1,2})(?:[,.](\d{1,3}))?", s)
        if not m:
            raise RuntimeError(f"无效时间格式: {text}")
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3))
        ms = int((m.group(4) or "0").ljust(3, "0")[:3])
        return max(0.0, hh * 3600 + mm * 60 + ss + ms / 1000.0)

    def _to_srt_time(self, sec: float) -> str:
        x = max(0.0, float(sec))
        total_ms = int(round(x * 1000.0))
        hh = total_ms // 3_600_000
        rem = total_ms % 3_600_000
        mm = rem // 60_000
        rem = rem % 60_000
        ss = rem // 1_000
        ms = rem % 1_000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    def _upsert_title_entry(self, entry: dict[str, Any]) -> int:
        def _safe_int(v: Any, default: int = 1) -> int:
            try:
                return int(v)
            except Exception:
                return default

        def _safe_float(v: Any, default: float = 0.0) -> float:
            try:
                return float(v)
            except Exception:
                return default

        existing_idx = -1
        for i, e in enumerate(self.entries):
            try:
                if int(e.get("segment_id", -1)) == 0:
                    existing_idx = i
                    break
            except Exception:
                continue
        if existing_idx >= 0:
            self.entries[existing_idx] = entry
        else:
            self.entries.append(entry)
        self.entries.sort(
            key=lambda e: (
                _safe_float(e.get("time_start", 0.0), 0.0),
                0 if _safe_int(e.get("segment_id", 1), 1) == 0 else 1,
                _safe_int(e.get("segment_id", 1), 1),
            )
        )
        for i, e in enumerate(self.entries):
            try:
                if int(e.get("segment_id", -1)) == 0:
                    return i
            except Exception:
                pass
        return 0

    def _insert_title_segment_from_roi(self) -> None:
        def _job() -> None:
            self._save_current_entry()
            tr = self._build_translator()
            st = self._parse_time_input(self.title_start_var.get())
            ed = self._parse_time_input(self.title_end_var.get())
            if ed < st:
                raise RuntimeError("Title End 必须大于等于 Title Start")

            roi_key = "title_ocr_roi"
            roi = self.review_rois.get(roi_key)
            if not roi:
                raise RuntimeError(f"未找到ROI: {roi_key}")
            crop = self._crop(roi)
            out_dir = self.cache_path.parent / "review_tmp"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            img_path = out_dir / f"title_roi_{ts}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            original_text, translated_text, usage = tr.translate_single_image_ja_to_zh_cn_structured_with_tag(
                image_path=img_path,
                request_tag="title-segment",
                history_items=None,
                custom_prompt=self._get_custom_prompt(),
            )
            translated_text = _normalize_quotes_for_subtitle(str(translated_text or "").strip())
            title_entry: dict[str, Any] = {
                "segment_id": 0,
                "time_start": float(st),
                "time_end": float(ed),
                "srt_start": self._to_srt_time(st),
                "srt_end": self._to_srt_time(ed),
                "dialogue_type": "title",
                "speaker": str(self.title_info_var.get() or "").strip(),
                "text_original": str(original_text or "").strip(),
                "translation_subtitle": str(translated_text or "").strip(),
                "debug_subtitle": "[TITLE]",
            }
            idx = self._upsert_title_entry(title_entry)
            self.last_review_result = dict(title_entry)
            self.root.after(
                0,
                lambda: (
                    self._show_segment(idx),
                    self._show_review_result(title_entry, usage, "Title 段已插入/更新（内存）"),
                ),
            )

        self._run_bg("Title段插入", _job)

    def _entry_times(self, idx: int) -> tuple[float, float]:
        e = self.entries[idx]
        st = float(e.get("time_start", 0.0) or 0.0)
        ed = float(e.get("time_end", st) or st)
        return st, ed

    def _update_seg_info(self, idx: int) -> None:
        if not self.entries:
            self.seg_info_var.set("无分段")
            return
        e = self.entries[idx]
        st, ed = self._entry_times(idx)
        self.seg_info_var.set(f"段 {idx + 1}/{len(self.entries)} | segment_id={e.get('segment_id', idx + 1)} | {st:.2f}-{ed:.2f}s")

    def _show_segment(self, idx: int, request_prefetch: bool = True) -> None:
        if not self.entries:
            self.json_text.delete("1.0", tk.END)
            self.json_text.insert("1.0", "{}")
            return
        self._set_current_segment_ui(idx, request_prefetch=False)
        self._seek_to_segment(idx, request_prefetch=request_prefetch)

    def _set_current_segment_ui(self, idx: int, request_prefetch: bool = True) -> None:
        if not self.entries:
            self.json_text.delete("1.0", tk.END)
            self.json_text.insert("1.0", "{}")
            return
        idx = max(0, min(idx, len(self.entries) - 1))
        self.current_index = idx
        seg_id = self.entries[idx].get("segment_id")
        self.goto_var.set(str(seg_id if seg_id is not None else (idx + 1)))
        self._update_seg_info(idx)
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert("1.0", json.dumps(self.entries[idx], ensure_ascii=False, indent=2))
        if request_prefetch:
            self._request_prefetch(idx)

    def _sync_segment_ui_by_time(self, sec: float) -> None:
        idx = self._segment_index_for_time(float(sec))
        if idx is None or idx == self.current_index:
            return
        # Preserve current edits before switching the segment JSON panel.
        try:
            self._save_current_entry()
        except Exception:
            return
        self._set_current_segment_ui(idx, request_prefetch=True)

    def _seek_to_segment(self, idx: int, request_prefetch: bool = True) -> None:
        if self.cap is None or not self.entries:
            return
        self._seek(
            self._segment_anchor_sec(idx),
            request_prefetch=request_prefetch,
            prefetch_center_idx=idx,
        )

    def _segment_anchor_sec(self, idx: int) -> float:
        if self.cap is None or not self.entries:
            return 0.0
        idx = max(0, min(int(idx), len(self.entries) - 1))
        st, ed = self._entry_times(idx)
        end_sec = max(st, ed)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        marker_cfg = self.cfg.get("marker", {}) if isinstance(self.cfg, dict) else {}
        n_from_end = int(marker_cfg.get("ocr_anchor_from_end_frames", 3) or 3)
        n_from_end = max(0, n_from_end)
        anchor_sec = end_sec - (n_from_end / max(1.0, fps))
        target = max(st, anchor_sec)
        fps_safe = max(1.0, float(fps))
        # Align to previous frame boundary (never forward).
        frame_idx = int(np.floor(max(0.0, target) * fps_safe + 1e-9))
        target_aligned = frame_idx / fps_safe
        return max(st, target_aligned)

    def _prev_segment(self) -> None:
        self._save_current_entry()
        self._show_segment(self.current_index - 1, request_prefetch=True)

    def _next_segment(self) -> None:
        self._save_current_entry()
        self._show_segment(self.current_index + 1, request_prefetch=True)

    def _jump_segment(self) -> None:
        self._save_current_entry()
        raw = self.goto_var.get().strip()
        if not raw:
            self.status_var.set("跳转值不能为空")
            return
        try:
            target = int(raw)
        except Exception:
            self.status_var.set(f"无效段号: {raw}")
            return

        idx: int | None = None
        for i, entry in enumerate(self.entries):
            seg_num = self._seg_numeric_id(entry)
            if seg_num is not None and seg_num == target:
                idx = i
                break

        if idx is None:
            self.status_var.set(f"未找到 segment_id={target}")
            return

        self._show_segment(idx, request_prefetch=True)

    def _cancel_pending_seek(self) -> None:
        self._pending_seek_sec = None
        if self._seek_after_id is not None:
            try:
                self.root.after_cancel(self._seek_after_id)
            except Exception:
                pass
            self._seek_after_id = None

    def _on_scale_change(self, value: str) -> None:
        if self._suppress_scale_callback:
            return
        sec = float(value)
        self._seek(sec, update_scale=False, prefer_fast=True)
        self._pending_seek_sec = sec
        if self._seek_after_id is not None:
            try:
                self.root.after_cancel(self._seek_after_id)
            except Exception:
                pass
        self._seek_after_id = self.root.after(120, self._flush_precise_seek)

    def _on_scale_release(self, _e: tk.Event[Any]) -> None:
        self._flush_precise_seek()

    def _flush_precise_seek(self) -> None:
        self._seek_after_id = None
        sec = self._pending_seek_sec
        if sec is None:
            return
        self._pending_seek_sec = None
        self._seek(
            float(sec),
            update_scale=False,
            prefer_fast=False,
            request_prefetch=False,
            force_ffmpeg=True,
        )

    def _jump_time(self) -> None:
        self._cancel_pending_seek()
        self._seek(float(self.time_var.get().strip()), request_prefetch=False, force_ffmpeg=True)

    def _segment_index_for_time(self, sec: float) -> int | None:
        if not self.entries:
            return None
        x = float(sec)
        for i, e in enumerate(self.entries):
            st = float(e.get("time_start", 0.0) or 0.0)
            ed = float(e.get("time_end", st) or st)
            if st <= x <= ed:
                return i
        return None

    def _step_frames(self, delta_frames: int) -> None:
        self._cancel_pending_seek()
        fps = max(1.0, float(self.fps or 25.0))
        self._seek(self.current_sec + (float(delta_frames) / fps), request_prefetch=False)

    def _next_frame(self) -> None:
        self._step_frames(1)

    def _prev_frame(self) -> None:
        self._step_frames(-1)

    def _next_10_frames(self) -> None:
        self._step_frames(10)

    def _prev_10_frames(self) -> None:
        self._step_frames(-10)

    def _canvas_to_src(self, x: int, y: int) -> tuple[int, int]:
        rx = x - self._offset_x
        ry = y - self._offset_y
        sx = int(round(rx / max(self._scale_x, 1e-6)))
        sy = int(round(ry / max(self._scale_y, 1e-6)))
        sx = max(0, min(sx, max(0, self.video_w - 1)))
        sy = max(0, min(sy, max(0, self.video_h - 1)))
        return sx, sy

    def _src_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        return (
            int(round(x * self._scale_x)) + self._offset_x,
            int(round(y * self._scale_y)) + self._offset_y,
        )

    def _on_canvas_configure(self, _e: tk.Event[Any]) -> None:
        # Re-render on canvas resize so preview always adapts to window size.
        if self.current_frame_rgb is not None:
            self._refresh_canvas()

    def _resolve_video_for_decode(self) -> Path | None:
        video_for_decode = self.video_path if self.video_path and self.video_path.exists() else None
        if video_for_decode is None:
            raw_video = self.video_var.get().strip()
            if raw_video:
                cand = Path(raw_video).resolve()
                if cand.exists():
                    video_for_decode = cand
        return video_for_decode

    def _read_frame_safe(self, sec: float, prefer_fast: bool = False) -> tuple[bool, Any | None]:
        if self.cap is None:
            return False, None
        if not prefer_fast:
            ok, frame = self._read_frame_ffmpeg_once(sec)
            if ok and frame is not None:
                return True, frame
        fps = float(self.fps or self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        fps = max(1.0, fps)
        target_frame = int(round(max(0.0, sec) * fps))
        preroll_frames = max(8, min(120, int(round(fps * 1.5))))
        start_frame = max(0, target_frame - preroll_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        frame = None
        ok = False
        steps = max(1, target_frame - start_frame + 1)
        for _ in range(steps):
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
        if ok and frame is not None:
            return True, frame
        self.cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        return self.cap.read()

    def _cache_get(self, frame_idx: int) -> Any | None:
        with self._cache_lock:
            frame = self._frame_cache.get(int(frame_idx))
            if frame is None:
                return None
            self._frame_cache.move_to_end(int(frame_idx))
            return frame.copy()

    def _cache_put(self, frame_idx: int, frame_bgr: Any) -> None:
        with self._cache_lock:
            self._frame_cache[int(frame_idx)] = frame_bgr.copy()
            self._frame_cache.move_to_end(int(frame_idx))
            while len(self._frame_cache) > self._frame_cache_max:
                self._frame_cache.popitem(last=False)

    def _build_prefetch_targets(self, center_segment_idx: int) -> list[int]:
        radius = max(0, int(self._prefetch_segment_radius))
        center = int(center_segment_idx)
        if not self.entries or radius <= 0:
            self._last_prefetch_center_idx = center
            return []

        prev = self._last_prefetch_center_idx
        targets: list[int] = []

        def _push(idx: int) -> None:
            if 0 <= idx < len(self.entries) and idx != center and idx not in targets:
                targets.append(idx)

        if prev is not None and center == prev + 1:
            for d in range(1, radius + 1):
                _push(center + d)
            for d in range(1, radius + 1):
                _push(center - d)
        elif prev is not None and center == prev - 1:
            for d in range(1, radius + 1):
                _push(center - d)
            for d in range(1, radius + 1):
                _push(center + d)
        else:
            for d in range(1, radius + 1):
                _push(center + d)
                _push(center - d)

        self._last_prefetch_center_idx = center
        return targets

    def _cancel_prefetch_job(self, job: dict[str, Any]) -> None:
        proc = job.get("proc")
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=0.3)
                except Exception:
                    proc.kill()
        except Exception:
            pass

    def _cancel_all_prefetch_jobs(self) -> None:
        with self._prefetch_jobs_lock:
            jobs = list(self._prefetch_jobs.values())
            self._prefetch_jobs.clear()
            self._prefetch_request_seq += 1
        for job in jobs:
            self._cancel_prefetch_job(job)

    def _spawn_ffmpeg_frame_job(self, sec: float) -> subprocess.Popen[bytes] | None:
        video_for_decode = self._resolve_video_for_decode()
        if video_for_decode is None or not self.ffmpeg_path.exists():
            return None
        if self.video_w <= 0 or self.video_h <= 0:
            return None
        cmd = self._build_ffmpeg_single_frame_cmd(
            video_for_decode=video_for_decode,
            sec=float(sec),
            two_stage_seek=bool(self.ffmpeg_two_stage_seek),
        )
        try:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                cwd=str(ROOT),
            )
        except Exception:
            return None

    def _collect_ffmpeg_frame_from_proc(self, proc: subprocess.Popen[bytes]) -> tuple[bool, Any | None]:
        try:
            expected = int(self.video_w) * int(self.video_h) * 3
            if expected <= 0 or proc.stdout is None:
                return False, None
            buf = bytearray()
            while len(buf) < expected:
                chunk = proc.stdout.read(expected - len(buf))
                if not chunk:
                    break
                buf.extend(chunk)
            rc = proc.wait(timeout=2)
            if rc != 0 or len(buf) < expected:
                return False, None
            arr = np.frombuffer(bytes(buf[:expected]), dtype=np.uint8)
            frame = arr.reshape((int(self.video_h), int(self.video_w), 3))
            return True, frame
        except Exception:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
            return False, None

    def _schedule_prefetch_locked(self) -> None:
        if len(self._prefetch_jobs) > self._prefetch_max_workers:
            overflow = len(self._prefetch_jobs) - self._prefetch_max_workers
            oldest_ids = sorted(
                self._prefetch_jobs,
                key=lambda jid: float(self._prefetch_jobs[jid].get("created_at", 0.0)),
            )[:overflow]
            for jid in oldest_ids:
                job = self._prefetch_jobs.pop(jid, None)
                if job is not None:
                    self._cancel_prefetch_job(job)

        active_seg_idxs = {int(job["seg_idx"]) for job in self._prefetch_jobs.values()}
        fps = max(1.0, float(self.fps or 25.0))
        req_seq = int(self._prefetch_request_seq)

        for seg_idx in list(self._prefetch_targets):
            if len(self._prefetch_jobs) >= self._prefetch_max_workers:
                break
            if int(seg_idx) in active_seg_idxs:
                continue
            sec = self._segment_anchor_sec(seg_idx)
            fid = int(np.floor(max(0.0, sec) * fps + 1e-9))
            if self._cache_get(fid) is not None:
                continue
            proc = self._spawn_ffmpeg_frame_job(sec)
            if proc is None:
                continue
            self._prefetch_job_seq += 1
            job_id = int(self._prefetch_job_seq)
            job = {
                "job_id": job_id,
                "seg_idx": int(seg_idx),
                "frame_idx": int(fid),
                "sec": float(sec),
                "proc": proc,
                "request_seq": req_seq,
                "created_at": time.time(),
            }
            self._prefetch_jobs[job_id] = job
            active_seg_idxs.add(int(seg_idx))
            threading.Thread(target=self._prefetch_job_runner, args=(job_id,), daemon=True).start()

    def _prefetch_job_runner(self, job_id: int) -> None:
        with self._prefetch_jobs_lock:
            job = self._prefetch_jobs.get(int(job_id))
        if job is None:
            return

        proc = job.get("proc")
        if proc is None:
            return

        ok, frame = self._collect_ffmpeg_frame_from_proc(proc)
        cache_item: tuple[int, Any] | None = None
        with self._prefetch_jobs_lock:
            cur = self._prefetch_jobs.pop(int(job_id), None)
            if cur is not None and ok and frame is not None:
                current_targets = set(int(x) for x in self._prefetch_targets)
                if int(cur["seg_idx"]) in current_targets:
                    cache_item = (int(cur["frame_idx"]), frame)
            self._schedule_prefetch_locked()
        if cache_item is not None:
            self._cache_put(cache_item[0], cache_item[1])

    def _request_prefetch(self, center_segment_idx: int) -> None:
        targets = self._build_prefetch_targets(center_segment_idx)
        with self._prefetch_jobs_lock:
            self._prefetch_request_seq += 1
            self._prefetch_targets = list(targets)
            needed = set(int(x) for x in targets)
            stale_ids: list[int] = []
            for job_id, job in self._prefetch_jobs.items():
                if int(job.get("seg_idx", -1)) not in needed:
                    stale_ids.append(int(job_id))
            for job_id in stale_ids:
                job = self._prefetch_jobs.pop(int(job_id), None)
                if job is not None:
                    self._cancel_prefetch_job(job)
            self._schedule_prefetch_locked()

    def _read_frame_ffmpeg_once(self, sec: float) -> tuple[bool, Any | None]:
        video_for_decode = self._resolve_video_for_decode()
        if video_for_decode is None or not self.ffmpeg_path.exists():
            return False, None
        if self.video_w <= 0 or self.video_h <= 0:
            return False, None
        # Preferred path: two-stage seek (fast pre-seek + short precise seek).
        # Fallback path: original precise-only seek.
        cmd_primary = self._build_ffmpeg_single_frame_cmd(
            video_for_decode=video_for_decode,
            sec=float(sec),
            two_stage_seek=bool(self.ffmpeg_two_stage_seek),
        )
        if self.ffmpeg_two_stage_seek:
            cmd_fallback = self._build_ffmpeg_single_frame_cmd(
                video_for_decode=video_for_decode,
                sec=float(sec),
                two_stage_seek=False,
            )
        else:
            cmd_fallback = None

        ok, frame = self._run_ffmpeg_single_frame_cmd(cmd_primary)
        if ok and frame is not None:
            return True, frame
        if cmd_fallback is not None:
            return self._run_ffmpeg_single_frame_cmd(cmd_fallback)
        return False, None

    def _build_ffmpeg_single_frame_cmd(
        self,
        video_for_decode: Path,
        sec: float,
        two_stage_seek: bool,
    ) -> list[str]:
        target_sec = max(0.0, float(sec))
        cmd: list[str] = [str(self.ffmpeg_path), "-v", "error"]
        hw = str(self.ffmpeg_hwaccel or "").strip().lower()
        if hw and hw not in {"none", "off", "disable", "disabled"}:
            cmd.extend(["-hwaccel", self.ffmpeg_hwaccel])
        if two_stage_seek:
            margin = max(0.0, float(self.ffmpeg_two_stage_margin_sec))
            pre_seek_sec = max(0.0, target_sec - margin)
            post_seek_sec = max(0.0, target_sec - pre_seek_sec)
            cmd.extend(
                [
                    "-ss",
                    f"{pre_seek_sec:.6f}",
                    "-i",
                    str(video_for_decode),
                    "-ss",
                    f"{post_seek_sec:.6f}",
                ]
            )
        else:
            # Original precise seek path (kept for compatibility/fallback).
            cmd.extend(
                [
                    "-i",
                    str(video_for_decode),
                    "-ss",
                    f"{target_sec:.6f}",
                ]
            )
        cmd.extend(
            [
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ]
        )
        return cmd

    def _run_ffmpeg_single_frame_cmd(self, cmd: list[str]) -> tuple[bool, Any | None]:
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
            if proc.returncode != 0 or not proc.stdout:
                return False, None
            expected = int(self.video_w) * int(self.video_h) * 3
            if expected <= 0 or len(proc.stdout) < expected:
                return False, None
            arr = np.frombuffer(proc.stdout[:expected], dtype=np.uint8)
            frame = arr.reshape((int(self.video_h), int(self.video_w), 3))
            return True, frame
        except Exception:
            return False, None

    def _set_decode_overlay(self, visible: bool) -> None:
        show = bool(visible)
        if show == self._decode_overlay_visible:
            return
        self._decode_overlay_visible = show
        if show:
            self._decode_spinner_phase = 0
            self._tick_decode_overlay()
        else:
            if self._decode_spinner_after_id is not None:
                try:
                    self.root.after_cancel(self._decode_spinner_after_id)
                except Exception:
                    pass
                self._decode_spinner_after_id = None
            self._refresh_canvas()

    def _tick_decode_overlay(self) -> None:
        if not self._decode_overlay_visible:
            self._decode_spinner_after_id = None
            return
        self._decode_spinner_phase = (self._decode_spinner_phase + 1) % len(self._decode_spinner_chars)
        self._refresh_canvas()
        self._decode_spinner_after_id = self.root.after(120, self._tick_decode_overlay)

    def _seek_force_ffmpeg_async(
        self,
        sec: float,
        update_scale: bool,
        request_prefetch: bool,
        prefetch_center_idx: int | None,
    ) -> None:
        self._precise_req_id += 1
        req_id = int(self._precise_req_id)
        target_sec = float(sec)
        self._set_decode_overlay(True)

        def _worker() -> None:
            ok, frame = self._read_frame_ffmpeg_once(target_sec)

            def _apply() -> None:
                if req_id != int(self._precise_req_id):
                    return
                self._set_decode_overlay(False)
                if not ok or frame is None:
                    return
                fps = max(1.0, float(self.fps or 25.0))
                frame_idx = int(round(target_sec * fps))
                self.current_sec = target_sec
                self._cache_put(frame_idx, frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame_rgb = rgb.copy()
                self._last_fast_seek_frame_idx = frame_idx
                self._refresh_canvas()
                self.time_var.set(f"{target_sec:.2f}")
                if update_scale:
                    self._suppress_scale_callback = True
                    try:
                        self.time_scale.set(target_sec)
                    finally:
                        self._suppress_scale_callback = False
                self._sync_segment_ui_by_time(target_sec)
                if request_prefetch:
                    center = self.current_index if prefetch_center_idx is None else int(prefetch_center_idx)
                    self._request_prefetch(center)

            self.root.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    def _seek(
        self,
        sec: float,
        update_scale: bool = True,
        prefer_fast: bool = False,
        request_prefetch: bool = False,
        prefetch_center_idx: int | None = None,
        force_ffmpeg: bool = False,
    ) -> None:
        sec = max(0.0, min(float(sec), max(0.0, self.duration_sec)))
        fps = max(1.0, float(self.fps or 25.0))
        frame_idx = int(round(sec * fps))
        sec = frame_idx / fps

        if force_ffmpeg:
            self._seek_force_ffmpeg_async(
                sec=sec,
                update_scale=bool(update_scale),
                request_prefetch=bool(request_prefetch),
                prefetch_center_idx=prefetch_center_idx,
            )
            return

        if self._is_seeking or self.cap is None:
            return
        self._is_seeking = True
        try:
            if (
                prefer_fast
                and self._last_fast_seek_frame_idx == frame_idx
                and self.current_frame_rgb is not None
            ):
                self.current_sec = sec
                self.time_var.set(f"{sec:.2f}")
                return
            self.current_sec = sec
            cached = self._cache_get(frame_idx)
            if cached is not None:
                ok, frame = True, cached
            else:
                ok, frame = self._read_frame_safe(sec, prefer_fast=prefer_fast)
            if not ok or frame is None:
                return
            self._cache_put(frame_idx, frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame_rgb = frame.copy()
            if prefer_fast:
                self._last_fast_seek_frame_idx = frame_idx
            self._refresh_canvas()
            self.time_var.set(f"{sec:.2f}")
            if update_scale:
                self._suppress_scale_callback = True
                try:
                    self.time_scale.set(sec)
                finally:
                    self._suppress_scale_callback = False
            self._sync_segment_ui_by_time(sec)
            if (not prefer_fast) and request_prefetch:
                center = self.current_index if prefetch_center_idx is None else int(prefetch_center_idx)
                self._request_prefetch(center)
        finally:
            self._is_seeking = False

    def _refresh_canvas(self) -> None:
        self.canvas.delete("all")
        if self.current_frame_rgb is None:
            return
        frame = self.current_frame_rgb
        h, w = frame.shape[:2]
        canvas_w = max(1, int(self.canvas.winfo_width()))
        canvas_h = max(1, int(self.canvas.winfo_height()))
        # Keep aspect ratio, prefer fitting width to canvas.
        scale = canvas_w / float(max(1, w))
        self._render_w = canvas_w
        self._render_h = max(1, int(round(h * scale)))
        # If height overflows too much in small windows, clamp by height as fallback.
        if self._render_h > canvas_h and canvas_h > 1:
            scale = canvas_h / float(max(1, h))
            self._render_h = canvas_h
            self._render_w = max(1, int(round(w * scale)))
        resized = cv2.resize(frame, (self._render_w, self._render_h), interpolation=cv2.INTER_LINEAR)
        self._img = ImageTk.PhotoImage(Image.fromarray(resized))
        self._scale_x = self._render_w / float(max(1, w))
        self._scale_y = self._scale_x
        self._offset_x = max(0, (canvas_w - self._render_w) // 2)
        self._offset_y = max(0, (canvas_h - self._render_h) // 2)
        self.canvas.create_image(self._offset_x, self._offset_y, anchor="nw", image=self._img)
        self.canvas.configure(scrollregion=(0, 0, canvas_w, canvas_h))
        for key in ROI_KEYS:
            rect = self.review_rois.get(key)
            if not rect:
                continue
            x0, y0 = self._src_to_canvas(rect[0], rect[1])
            x1, y1 = self._src_to_canvas(rect[2], rect[3])
            color = ROI_COLORS.get(key, "#fff")
            width = 3 if key == self.roi_key_var.get() else 2
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=width)
            self.canvas.create_text(x0 + 4, y0 + 4, text=key, anchor="nw", fill=color, font=("Segoe UI", 11, "bold"))
        if self.drag_start and self.drag_now:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_now
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#fff", width=2, dash=(6, 3))
        if self._decode_overlay_visible:
            spinner = self._decode_spinner_chars[self._decode_spinner_phase]
            tip = f"{spinner} Decoding..."
            pad_x = 14
            pad_y = 9
            box_w = 190
            box_h = 34
            cx = self._offset_x + (self._render_w // 2)
            cy = self._offset_y + (self._render_h // 2)
            x1 = cx - (box_w // 2)
            y1 = cy - (box_h // 2)
            x2 = x1 + box_w
            y2 = y1 + box_h
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#000000", outline="#4c9aff", width=1)
            self.canvas.create_text(x1 + pad_x, y1 + pad_y, text=tip, anchor="nw", fill="#ffffff", font=("Segoe UI", 10, "bold"))

    def _on_mouse_down(self, e: tk.Event[Any]) -> None:
        if self.current_frame_rgb is None:
            return
        self.drag_start = (int(e.x), int(e.y))
        self.drag_now = (int(e.x), int(e.y))
        self._refresh_canvas()

    def _on_mouse_drag(self, e: tk.Event[Any]) -> None:
        if self.drag_start is None:
            return
        self.drag_now = (int(e.x), int(e.y))
        self._refresh_canvas()

    def _on_mouse_up(self, e: tk.Event[Any]) -> None:
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = int(e.x), int(e.y)
        self.drag_start = None
        self.drag_now = None
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            self._refresh_canvas()
            return
        cx0, cx1 = sorted([x0, x1])
        cy0, cy1 = sorted([y0, y1])
        sx0, sy0 = self._canvas_to_src(cx0, cy0)
        sx1, sy1 = self._canvas_to_src(cx1, cy1)
        key = self.roi_key_var.get().strip()
        if key in ROI_KEYS:
            self.review_rois[key] = [sx0, sy0, sx1, sy1]
        self._refresh_canvas()

    def _reset_rois(self) -> None:
        self.review_rois = {k: list(v) for k, v in self.default_rois.items()}
        self._refresh_canvas()

    def _crop(self, roi: list[int]) -> Any:
        if self.current_frame_rgb is None:
            raise RuntimeError("当前无帧")
        x0, y0, x1, y1 = [int(v) for v in roi]
        h, w = self.current_frame_rgb.shape[:2]
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(f"ROI无效: {roi}")
        return self.current_frame_rgb[y0:y1, x0:x1].copy()

    def _review_by_new_crops(self) -> None:
        def _job() -> None:
            self._save_current_entry()
            if not self.entries:
                raise RuntimeError("无分段")
            tr = self._build_translator()
            name_crop = self._crop(self.review_rois["name_roi"])
            dia_crop = self._crop(self.review_rois["dialogue_roi"])
            out_dir = self.cache_path.parent / "review_tmp"
            out_dir.mkdir(parents=True, exist_ok=True)
            seg_no = self.current_index + 1
            ts = int(time.time())
            name_img = out_dir / f"seg_{seg_no:04d}_name_{ts}.png"
            dia_img = out_dir / f"seg_{seg_no:04d}_dialogue_{ts}.png"
            cv2.imwrite(str(name_img), cv2.cvtColor(name_crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(dia_img), cv2.cvtColor(dia_crop, cv2.COLOR_RGB2BGR))
            speaker_hint = str(self.entries[self.current_index].get("speaker", "") or "")
            custom = self._get_custom_prompt()
            spk, orig, trans, usage = tr.translate_image_ja_to_zh_cn_structured_with_tag(
                image_path=dia_img,
                speaker_image_path=name_img,
                speaker=speaker_hint,
                request_tag=f"review-seg-{seg_no}",
                history_items=None,
                custom_prompt=custom,
            )
            base_entry = self.entries[self.current_index]
            result = self._build_review_entry(
                base_entry=base_entry,
                speaker=spk,
                original=orig,
                translated=trans,
            )
            self.last_review_result = result
            self.root.after(0, lambda: self._show_review_result(result, usage, "截图复译完成"))

        self._run_bg("截图复译", _job)

    def _review_by_text_only(self) -> None:
        def _job() -> None:
            parsed = self._read_current_json_from_editor()
            tr = self._build_translator()
            original_text = str(parsed.get("text_original", "") or "").strip()
            if not original_text:
                raise RuntimeError("当前段 text_original 为空")
            speaker = str(parsed.get("speaker", "") or "").strip()
            translated, usage = tr.translate_text_with_prompt(
                original_text=original_text,
                speaker=speaker,
                request_tag=f"review-text-{self.current_index + 1}",
                custom_prompt=self._get_custom_prompt(),
            )
            base_entry = self.entries[self.current_index]
            result = self._build_review_entry(
                base_entry=base_entry,
                speaker=speaker,
                original=original_text,
                translated=translated,
            )
            self.last_review_result = result
            self.root.after(0, lambda: self._show_review_result(result, usage, "原文重译完成"))

        self._run_bg("原文重译", _job)

    def _read_current_json_from_editor(self) -> dict[str, Any]:
        content = self.json_text.get("1.0", tk.END).strip()
        if not content:
            raise RuntimeError("当前JSON为空")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise RuntimeError("当前JSON必须是对象")
        return parsed

    def _show_review_result(self, result: dict[str, Any], usage: dict[str, Any] | None, status: str) -> None:
        self.review_text.delete("1.0", tk.END)
        self.review_text.insert("1.0", json.dumps(result, ensure_ascii=False, indent=2))
        self.review_meta_var.set(self._usage_to_meta(usage))
        self.status_var.set(status)

    def _apply_last_result_all(self) -> None:
        if not self.last_review_result or not self.entries:
            self.status_var.set("没有可应用的复译结果")
            return
        e = self.entries[self.current_index]
        e["speaker"] = str(self.last_review_result.get("speaker", "") or e.get("speaker", ""))
        e["text_original"] = str(self.last_review_result.get("text_original", "") or e.get("text_original", ""))
        e["translation_subtitle"] = str(self.last_review_result.get("translation_subtitle", "") or e.get("translation_subtitle", ""))
        e["debug_subtitle"] = str(self.last_review_result.get("debug_subtitle", "") or e.get("debug_subtitle", ""))
        self._show_segment(self.current_index)
        self.status_var.set("已一键替换当前段（内存）")

    def _apply_last_result_translation(self) -> None:
        if not self.last_review_result or not self.entries:
            self.status_var.set("没有可应用的复译结果")
            return
        e = self.entries[self.current_index]
        e["translation_subtitle"] = str(self.last_review_result.get("translation_subtitle", "") or e.get("translation_subtitle", ""))
        self._show_segment(self.current_index)
        self.status_var.set("已替换当前段译文（内存）")

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="cache 图形化浏览、编辑与Review工具")
    parser.add_argument("--cache", required=True, help="translation_cache.json 路径")
    parser.add_argument("--video", default="", help="视频路径（可选）")
    parser.add_argument("--config", default="", help="游戏配置路径（可选；为空时优先从cache中的config_path读取）")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    video_path = Path(args.video) if str(args.video).strip() else None
    config_path: Path | None = Path(args.config).resolve() if str(args.config).strip() else None
    if config_path is None:
        if not cache_path.exists():
            raise RuntimeError(
                "未提供 --config，且 cache 文件不存在，无法从 cache 读取 config_path。"
            )
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        cache_cfg = ""
        if isinstance(raw, dict):
            cache_cfg = str(raw.get("config_path", "") or "").strip()
        if not cache_cfg:
            raise RuntimeError(
                "未提供 --config，且 cache 中缺少 config_path。"
                "请在命令行传入 --config，或先使用新版本 pipeline 重新生成 cache。"
            )
        config_path = Path(cache_cfg)
        if not config_path.is_absolute():
            # Relative path in cache is resolved against project root.
            config_path = (ROOT / config_path).resolve()
    assert config_path is not None
    if not config_path.exists():
        raise RuntimeError(f"配置不存在: {config_path}")
    app = CacheReviewApp(cache_path=cache_path, video_path=video_path, config_path=config_path)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
