from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
import yaml

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from ignite.archive_manager import archive_project, find_hard_subtitle_video
from ignite.auto_review import (
    _profile_chat_params,
    default_auto_review_profile,
    dialogue_review_entries_from_cache_entries,
    run_auto_review_entries,
)
from ignite.cache_manager import (
    MANUAL_INSERT_RAW_ID,
    _debug_subtitle_from_entry,
    _resolve_speaker_subtitle_style,
)
from ignite.config import load_config
from ignite.event_detect import MarkerTemplateMatcher
from ignite.gui.local_state import (
    load_dialog_dirs,
    load_window_state,
    related_dialog_dir,
    remember_dialog_dir,
    remember_window_state,
)
from ignite.ocr_engines import build_ocr_engine
from ignite.review_utils import _merge_review_reasons
from ignite.translation_runtime import (
    available_translation_model_profiles,
    ChatCompletionsTextTranslator,
    TEXT_EXTRACTION_PROFILE_MODE,
    VlmImageTextExtractor,
    TranslationModelProfile,
    VlmResponsesTranslator,
    _normalize_text_extraction_backend,
    has_kanji_overlap_from_original,
    normalize_quotes_for_subtitle,
    load_api_key,
    resolve_translation_model_profile,
)


ROOT = Path(__file__).resolve().parents[2]


ROI_KEYS = ["name_roi", "dialogue_roi", "title_ocr_roi"]
ROI_COLORS = {
    "name_roi": "#ff8000",
    "dialogue_roi": "#0168b7",
    "title_ocr_roi": "#b0b9c0",
}
TITLE_RECOGNITION_MODE_LABELS = {
    "auto": "自动",
    "direct_vlm": "翻译API直识别",
    "local_vlm": "本地VLM直识别",
}
TITLE_RECOGNITION_MODE_BY_LABEL = {v: k for k, v in TITLE_RECOGNITION_MODE_LABELS.items()}


class CacheReviewApp:
    def __init__(self, cache_path: Path, video_path: Path | None, config_path: Path) -> None:
        self.cache_path = cache_path.resolve()
        self.video_path = video_path.resolve() if video_path else None
        self.config_path = config_path.resolve()

        self.root = tk.Tk()
        self.root.title("字幕校对工具 - IGNITE")
        self.root.geometry("1900x1020")
        self._restore_window_geometry()

        self.status_var = tk.StringVar(value="就绪")
        self.seg_info_var = tk.StringVar(value="-")
        self.marker_score_var = tk.StringVar(value="Marker: N/A")
        self.review_meta_var = tk.StringVar(value="")
        self.goto_var = tk.StringVar(value="1")
        self.time_var = tk.StringVar(value="0.00")
        self.roi_key_var = tk.StringVar(value="dialogue_roi")
        self.title_start_var = tk.StringVar(value="0.00")
        self.title_end_var = tk.StringVar(value="2.00")
        self.title_capture_var = tk.StringVar(value="1.00")
        self.title_info_var = tk.StringVar(value="")
        self.title_recognition_mode_var = tk.StringVar(value=TITLE_RECOGNITION_MODE_LABELS["auto"])
        self.review_web_search_var = tk.BooleanVar(value=False)
        self.embed_ffmpeg_path_var = tk.StringVar(value="")
        self.embed_output_path_var = tk.StringVar(value="")
        self.embed_vcodec_var = tk.StringVar(value="libx264")
        self.embed_crf_var = tk.StringVar(value="18")
        self.embed_preset_var = tk.StringVar(value="medium")
        self.embed_acodec_var = tk.StringVar(value="copy")
        self.embed_extra_input_args_var = tk.StringVar(value="")
        self.embed_extra_output_args_var = tk.StringVar(value="")

        self.entries: list[dict[str, Any]] = []
        self.cache_payload: dict[str, Any] = {}
        self.current_index = 0
        self.suspect_indices: list[int] = []
        self.suspect_info_var = tk.StringVar(value="存疑: 0")
        self.neighbor_preview_texts: dict[str, tk.Text] = {}

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
        self.translator: VlmResponsesTranslator | None = None
        self.text_translator: ChatCompletionsTextTranslator | None = None
        self.image_text_extractor: VlmImageTextExtractor | None = None
        self.auto_review_model_var = tk.StringVar(value="")
        self.last_review_result: dict[str, Any] | None = None
        self._busy = False
        self.marker_score_cache: dict[str, Any] | None = None
        self._marker_matcher: MarkerTemplateMatcher | None = None
        self._marker2_matcher: MarkerTemplateMatcher | None = None
        self._marker_roi: list[int] | None = None
        self._marker2_roi: list[int] | None = None
        self._undo_stack: list[dict[str, Any]] = []
        self._redo_stack: list[dict[str, Any]] = []
        self._insert_dialog: dict[str, Any] | None = None
        self.subtitle_style_cfg: dict[str, Any] | None = None
        self._dialog_dirs: dict[str, Path] = load_dialog_dirs()
        self.show_all_rois = tk.BooleanVar(value=False)
        self._body_paned: ttk.Panedwindow | None = None
        self._stack_paned: ttk.Panedwindow | None = None

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
        self.root.after(100, self._restore_window_layout)

    def _dialog_initial_dir(self, key: str, fallback: str | Path | None = None) -> str:
        remembered = self._dialog_dirs.get(key)
        if remembered is not None and remembered.exists():
            return str(remembered)
        related = related_dialog_dir(self._dialog_dirs, key)
        if related is not None:
            return str(related)
        base = Path(fallback).expanduser() if fallback else ROOT
        try:
            base = base.resolve()
        except Exception:
            base = ROOT
        if base.is_file():
            base = base.parent
        if not base.exists():
            base = ROOT
        return str(base)

    def _remember_dialog_dir(self, key: str, selected_path: str | Path) -> None:
        remembered = remember_dialog_dir(key, selected_path, self._dialog_dirs)
        if remembered is not None:
            self._dialog_dirs[key] = remembered

    def _restore_window_geometry(self) -> None:
        state = load_window_state("review")
        geometry = str(state.get("geometry", "") or "").strip()
        if geometry:
            try:
                self.root.geometry(geometry)
            except Exception:
                pass
        if str(state.get("state", "") or "").strip() == "zoomed":
            self.root.after_idle(lambda: self._set_root_state("zoomed"))

    def _set_root_state(self, state: str) -> None:
        try:
            self.root.state(state)
        except Exception:
            pass

    def _restore_window_layout(self) -> None:
        state = load_window_state("review")
        layout = state.get("layout")
        if not isinstance(layout, dict):
            return
        self.root.update_idletasks()
        for key, paned in (
            ("body_sash", self._body_paned),
            ("stack_sash", self._stack_paned),
        ):
            if paned is None or key not in layout:
                continue
            try:
                paned.sashpos(0, int(layout[key]))
            except Exception:
                pass

    def _save_window_state(self) -> None:
        try:
            self.root.update_idletasks()
            layout: dict[str, int] = {}
            if self._body_paned is not None:
                layout["body_sash"] = int(self._body_paned.sashpos(0))
            if self._stack_paned is not None:
                layout["stack_sash"] = int(self._stack_paned.sashpos(0))
            remember_window_state(
                "review",
                geometry=str(self.root.geometry()),
                window_state=str(self.root.state() or ""),
                layout=layout,
            )
        except Exception:
            pass

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        self.cache_var = tk.StringVar(value=str(self.cache_path))
        self.video_var = tk.StringVar(value=str(self.video_path) if self.video_path else "")
        self.config_var = tk.StringVar(value=str(self.config_path))

        ttk.Label(top, text="缓存").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.cache_var, width=108).grid(row=0, column=1, columnspan=7, sticky="ew", padx=(6, 4))
        ttk.Button(top, text="选择缓存", command=self._pick_cache).grid(row=0, column=8, padx=2)
        ttk.Button(top, text="重载缓存", command=self._load_cache).grid(row=0, column=9, padx=2)

        ttk.Label(top, text="视频").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.video_var, width=108).grid(row=1, column=1, columnspan=7, sticky="ew", padx=(6, 4))
        ttk.Button(top, text="选择视频", command=self._pick_video).grid(row=1, column=8, padx=2)
        ttk.Button(top, text="重载视频", command=self._reload_video).grid(row=1, column=9, padx=2)

        ttk.Label(top, text="配置").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.config_var, width=108).grid(row=2, column=1, columnspan=7, sticky="ew", padx=(6, 4))
        ttk.Button(top, text="选择配置", command=self._pick_config).grid(row=2, column=8, padx=2)
        ttk.Button(top, text="重载配置", command=self._load_config).grid(row=2, column=9, padx=2)

        nav = ttk.Frame(top)
        nav.grid(row=3, column=0, columnspan=10, sticky="ew", pady=(6, 0))
        ttk.Button(nav, text="上一段 (←)", command=self._prev_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="下一段 (→)", command=self._next_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="上一个存疑", command=self._prev_suspect).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Button(nav, text="下一个存疑", command=self._next_suspect).pack(side=tk.LEFT, padx=2)
        ttk.Label(nav, textvariable=self.suspect_info_var, foreground="#9a4b00").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Label(nav, text="跳转段号").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(nav, textvariable=self.goto_var, width=10).pack(side=tk.LEFT)
        ttk.Button(nav, text="跳转", command=self._jump_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="保存缓存", command=self._save_cache_file).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(nav, text="归档项目", command=self._open_archive_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Label(nav, text="自动review模型").pack(side=tk.LEFT, padx=(12, 2))
        self._auto_review_model_combo = ttk.Combobox(
            nav,
            textvariable=self.auto_review_model_var,
            state="readonly",
            width=18,
        )
        self._auto_review_model_combo.pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="再次自动review", command=self._auto_review_current_cache).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            nav,
            text="生成字幕（当前Cache）",
            command=self._generate_subtitles_from_cache,
        ).pack(side=tk.RIGHT, padx=(12, 2))
        ttk.Button(
            nav,
            text="生成内嵌字幕视频",
            command=self._open_embed_subtitles_dialog,
        ).pack(side=tk.RIGHT, padx=2)

        title_bar = ttk.Frame(top)
        title_bar.grid(row=4, column=0, columnspan=10, sticky="ew", pady=(6, 0))
        ttk.Label(title_bar, text="标题开始(秒)").pack(side=tk.LEFT)
        ttk.Entry(title_bar, textvariable=self.title_start_var, width=14).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(title_bar, text="标题结束(秒)").pack(side=tk.LEFT)
        ttk.Entry(title_bar, textvariable=self.title_end_var, width=14).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(title_bar, text="截图时间(秒)").pack(side=tk.LEFT)
        ttk.Entry(title_bar, textvariable=self.title_capture_var, width=14).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(title_bar, text="翻译信息").pack(side=tk.LEFT)
        ttk.Entry(title_bar, textvariable=self.title_info_var, width=36).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(title_bar, text="识别方式").pack(side=tk.LEFT)
        ttk.Combobox(
            title_bar,
            textvariable=self.title_recognition_mode_var,
            values=list(TITLE_RECOGNITION_MODE_LABELS.values()),
            state="readonly",
            width=16,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(
            title_bar,
            text="识别并翻译Title",
            command=self._insert_title_segment_from_roi,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            title_bar,
            text="直接插入Title空白段",
            command=self._insert_blank_title_segment,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            title_bar,
            text="从Cache读取Title",
            command=lambda: self._load_title_fields_from_cache(show_status=True),
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(top, textvariable=self.seg_info_var, foreground="#1f5f99").grid(row=5, column=0, columnspan=10, sticky="w", pady=(6, 0))
        ttk.Label(top, textvariable=self.status_var, foreground="#2f6f3e").grid(row=6, column=0, columnspan=10, sticky="w")
        top.columnconfigure(1, weight=1)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        self._body_paned = body
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=4)
        body.add(right, weight=1)

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
        ttk.Button(frame_bar, text="设为段开始", command=self._set_segment_start_to_current_time).pack(side=tk.LEFT, padx=(12,6))
        ttk.Button(frame_bar, text="设为段结束", command=self._set_segment_end_to_current_time).pack(side=tk.LEFT)
        ttk.Button(frame_bar, text="设为标题开始", command=self._set_title_start_to_current_time).pack(side=tk.LEFT, padx=(12,6))
        ttk.Button(frame_bar, text="设为标题结束", command=self._set_title_end_to_current_time).pack(side=tk.LEFT)

        roi_bar = ttk.Frame(left, padding=(0, 6, 0, 0))
        roi_bar.pack(fill=tk.X)
        ttk.Label(roi_bar, text="拖拽编辑ROI").pack(side=tk.LEFT)
        ttk.OptionMenu(roi_bar, self.roi_key_var, self.roi_key_var.get(), *ROI_KEYS).pack(side=tk.LEFT, padx=6)
        ttk.Button(roi_bar, text="恢复默认ROI", command=self._reset_rois).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(roi_bar, text="显示ROI", variable=self.show_all_rois, command=self._refresh_canvas).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(roi_bar, textvariable=self.marker_score_var, foreground="#9a4b00").pack(side=tk.LEFT, padx=(16, 0))

        # Right panel: vertical stack (JSON on top, review on bottom)
        stack = ttk.Panedwindow(right, orient=tk.VERTICAL)
        self._stack_paned = stack
        stack.pack(fill=tk.BOTH, expand=True)
        json_panel = ttk.Frame(stack)
        review_panel = ttk.Frame(stack)
        stack.add(json_panel, weight=3)
        stack.add(review_panel, weight=4)

        ttk.Label(json_panel, text="前后段 Cache 预览").pack(anchor="w")
        merge_bar = ttk.Frame(json_panel)
        merge_bar.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(merge_bar, text="合并上一段", command=self._merge_with_prev).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(merge_bar, text="合并下一段", command=self._merge_with_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(merge_bar, text="前插入段", command=self._insert_before_segment).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Button(merge_bar, text="后插入段", command=self._insert_after_segment).pack(side=tk.LEFT, padx=2)
        ttk.Button(merge_bar, text="删除当前段", command=self._delete_current_segment).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Button(merge_bar, text="撤回操作", command=self._undo_merge).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Button(merge_bar, text="恢复操作", command=self._redo_operation).pack(side=tk.LEFT, padx=2)

        self.auto_mark_manual_review = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            merge_bar,
            text="插入/合并额外review标记",
            variable=self.auto_mark_manual_review,
        ).pack(side=tk.LEFT, padx=(14, 2))
        neighbor_preview = ttk.Frame(json_panel)
        neighbor_preview.pack(fill=tk.X, pady=(4, 6))
        for key, label in (("prev", "上一段"), ("next", "下一段")):
            box = ttk.LabelFrame(neighbor_preview, text=label, padding=4)
            box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0 if key == "prev" else 4, 0))
            txt = tk.Text(box, height=6, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
            txt.pack(fill=tk.BOTH, expand=True)
            self.neighbor_preview_texts[key] = txt

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
        self.root.bind("<Control-s>", lambda _e: self._save_cache_file())
        self.root.bind("<Control-S>", lambda _e: self._save_cache_file())
        self.title_start_var.trace_add("write", lambda *_: self._update_title_capture_default())
        self.title_end_var.trace_add("write", lambda *_: self._update_title_capture_default())

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
        normalized = normalize_quotes_for_subtitle(str(translated or "").strip())
        out: dict[str, Any] = {
            "segment_id": base_entry.get("segment_id"),
            "time_start": base_entry.get("time_start"),
            "time_end": base_entry.get("time_end"),
            "dialogue_type": base_entry.get("dialogue_type", "speaker_dialogue"),
            "speaker": str(speaker or base_entry.get("speaker", "")),
            "text_original": str(original or ""),
            "translation_subtitle": normalized,
        }
        if "raw_id" in base_entry:
            out["raw_id"] = base_entry.get("raw_id")
        reasons = self._entry_review_reasons(out)
        if reasons:
            out["needs_review"] = True
            out["review_reason"] = reasons
        return out

    def _run_bg(self, title: str, fn: Callable[[], None], *, show_review_progress: bool = False) -> None:
        if self._busy:
            self.status_var.set("任务执行中，请稍候")
            return
        self._busy = True
        if show_review_progress:
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
        self.subtitle_style_cfg = self._load_subtitle_style_cfg()
        self.translator = None
        self.text_translator = None
        self.image_text_extractor = None
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
        self._marker_roi = roi_cfg.get("marker_roi")
        if not isinstance(self._marker_roi, list) or len(self._marker_roi) != 4:
            self._marker_roi = None
        self._marker2_roi = roi_cfg.get("marker_2_roi")
        if not isinstance(self._marker2_roi, list) or len(self._marker2_roi) != 4:
            self._marker2_roi = None
        self._init_marker_matchers()
        self._load_marker_score_cache()
        self._refresh_auto_review_model_choices()
        self.status_var.set(f"已加载配置: {p}")
        self._refresh_canvas()

    def _reload_config_for_action(self, action: str) -> bool:
        try:
            self._load_config()
            return True
        except Exception as exc:
            messagebox.showerror(action, f"重新读取配置失败：{exc}", parent=self.root)
            self.status_var.set(f"{action}失败：重新读取配置失败: {exc}")
            return False

    def _read_editable_config_payload(self, path: Path) -> tuple[dict[str, Any], bool]:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        stripped = text.strip()
        is_json = path.suffix.lower() == ".json" or stripped.startswith("{")
        if not stripped:
            return {}, is_json
        data = json.loads(stripped) if is_json else yaml.safe_load(text)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise RuntimeError("配置文件根节点必须是对象")
        return data, is_json

    def _write_editable_config_payload(self, path: Path, data: dict[str, Any], *, is_json: bool) -> None:
        if is_json:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def _save_auto_review_game_config(self, game_name: str, extra_requirements: str) -> None:
        path = Path(self.config_var.get().strip() or self.config_path).resolve()
        if not path.exists():
            raise RuntimeError(f"配置不存在: {path}")
        data, is_json = self._read_editable_config_payload(path)
        game = data.get("game")
        if not isinstance(game, dict):
            game = {}
            data["game"] = game
        game["name"] = str(game_name or "").strip()
        game["extra_requirements"] = str(extra_requirements or "").strip()
        self._write_editable_config_payload(path, data, is_json=is_json)

    def _edit_auto_review_game_config_dialog(self) -> bool:
        game_cfg = self.cfg.get("game", {}) if isinstance(self.cfg.get("game"), dict) else {}
        name_var = tk.StringVar(value=str(game_cfg.get("name", "") or ""))
        result = {"ok": False}

        win = tk.Toplevel(self.root)
        win.title("自动review配置")
        win.transient(self.root)
        win.geometry("760x480")
        win.resizable(True, True)

        body = ttk.Frame(win, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(2, weight=1)

        ttk.Label(body, text="自动review前请确认游戏名和额外要求；保存后会写入当前配置文件并重新读取配置。", foreground="#1f5f99").grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 10),
        )
        ttk.Label(body, text="游戏名").grid(row=1, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(body, textvariable=name_var).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(0, 6))
        ttk.Label(body, text="额外要求 / 术语表").grid(row=2, column=0, sticky="nw")

        text_frame = ttk.Frame(body)
        text_frame.grid(row=2, column=1, sticky="nsew", padx=(8, 0))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        extra_text = tk.Text(text_frame, height=14, wrap=tk.WORD, font=("Consolas", 10))
        scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=extra_text.yview)
        extra_text.configure(yscrollcommand=scroll.set)
        extra_text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        extra_text.insert("1.0", str(game_cfg.get("extra_requirements", "") or ""))

        btns = ttk.Frame(body)
        btns.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def close(ok: bool) -> None:
            result["ok"] = ok
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()

        def save_and_start() -> None:
            try:
                self._save_auto_review_game_config(
                    name_var.get(),
                    extra_text.get("1.0", tk.END).strip(),
                )
                self._load_config()
            except Exception as exc:
                messagebox.showerror("自动review配置", f"保存配置失败：{exc}", parent=win)
                self.status_var.set(f"自动review配置保存失败: {exc}")
                return
            close(True)

        ttk.Button(btns, text="取消", command=lambda: close(False)).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btns, text="保存并开始自动review", command=save_and_start).pack(side=tk.RIGHT)
        win.protocol("WM_DELETE_WINDOW", lambda: close(False))
        win.bind("<Escape>", lambda _e: close(False))
        try:
            win.grab_set()
            win.focus_set()
        except Exception:
            pass
        self.root.wait_window(win)
        return bool(result["ok"])

    def _refresh_auto_review_model_choices(self) -> None:
        tcfg = self.cfg.get("translation", {}) if isinstance(self.cfg, dict) else {}
        if not isinstance(tcfg, dict):
            tcfg = {}
        model_list = available_translation_model_profiles(tcfg, "ocr_chat_completions")
        try:
            self._auto_review_model_combo["values"] = model_list
        except Exception:
            pass
        cur = str(self.auto_review_model_var.get() or "").strip()
        if model_list and cur not in model_list:
            self.auto_review_model_var.set(default_auto_review_profile(tcfg, str(tcfg.get("auto_review_model_profile", "") or "")))

    def _load_subtitle_style_cfg(self) -> dict[str, Any] | None:
        subtitle_style: dict[str, Any] | None = None
        style_path = ROOT / "config" / "subtitle_style.yaml"
        if style_path.exists():
            try:
                loaded = yaml.safe_load(style_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    subtitle_style = loaded
            except Exception:
                subtitle_style = None
        per_style = self.cfg.get("subtitle_style") if isinstance(self.cfg, dict) else None
        if isinstance(per_style, dict):
            if subtitle_style is None:
                subtitle_style = {}
            subtitle_style.update(per_style)
        return subtitle_style

    def _init_marker_matchers(self) -> None:
        self._marker_matcher = None
        self._marker2_matcher = None
        marker_cfg = self.cfg.get("marker", {})
        marker_tpl_paths = marker_cfg.get("template_paths")
        if isinstance(marker_tpl_paths, list) and marker_tpl_paths:
            valid = [p for p in self._resolve_config_asset_paths(marker_tpl_paths) if p.exists()]
            if valid:
                try:
                    self._marker_matcher = MarkerTemplateMatcher(
                        template_paths=valid,
                        center_width=marker_cfg.get("template_center_width"),
                        vertical_shift_px=int(marker_cfg.get("vertical_shift_px", 6)),
                        vertical_shift_step=int(marker_cfg.get("vertical_shift_step", 1)),
                        horizontal_shift_px=int(marker_cfg.get("horizontal_shift_px", 0)),
                        horizontal_shift_step=int(marker_cfg.get("horizontal_shift_step", 1)),
                        shift_mode=str(marker_cfg.get("shift_mode", "vertical")),
                    )
                except Exception:
                    self._marker_matcher = None
        marker2_cfg = self.cfg.get("marker_2", {})
        marker2_tpl_paths = marker2_cfg.get("template_paths")
        if isinstance(marker2_tpl_paths, list) and marker2_tpl_paths:
            valid = [p for p in self._resolve_config_asset_paths(marker2_tpl_paths) if p.exists()]
            if valid:
                try:
                    self._marker2_matcher = MarkerTemplateMatcher(
                        template_paths=valid,
                        center_width=marker2_cfg.get("template_center_width"),
                        vertical_shift_px=int(marker2_cfg.get("vertical_shift_px", 6)),
                        vertical_shift_step=int(marker2_cfg.get("vertical_shift_step", 1)),
                        horizontal_shift_px=int(marker2_cfg.get("horizontal_shift_px", 0)),
                        horizontal_shift_step=int(marker2_cfg.get("horizontal_shift_step", 1)),
                        shift_mode=str(marker2_cfg.get("shift_mode", "vertical")),
                    )
                except Exception:
                    self._marker2_matcher = None

    def _resolve_config_asset_paths(self, paths: list[Any]) -> list[Path]:
        out: list[Path] = []
        for raw in paths:
            p = Path(str(raw))
            if p.is_absolute():
                out.append(p.resolve())
                continue
            for base in (self.config_path.parent, ROOT):
                cand = (base / p).resolve()
                if cand.exists():
                    out.append(cand)
                    break
            else:
                out.append((self.config_path.parent / p).resolve())
        return out

    def _load_marker_score_cache(self) -> None:
        self.marker_score_cache = None
        cache_parent = self.cache_path.parent
        while cache_parent != cache_parent.parent:
            if cache_parent.name == "work":
                work_dir = cache_parent
                break
            cache_parent = cache_parent.parent
        else:
            try:
                latest = self.cache_path.parent / "marker_score_cache.json"
                if latest.exists():
                    self.marker_score_cache = json.loads(latest.read_text(encoding="utf-8"))
            except Exception:
                pass
            return
        for p in sorted(work_dir.glob("run_*"), reverse=True):
            score_file = p / "marker_score_cache.json"
            if score_file.exists():
                try:
                    self.marker_score_cache = json.loads(score_file.read_text(encoding="utf-8"))
                    return
                except Exception:
                    continue
        # Fallback: try alongside current cache
        score_file = self.cache_path.parent / "marker_score_cache.json"
        if score_file.exists():
            try:
                self.marker_score_cache = json.loads(score_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _build_translator(self) -> VlmResponsesTranslator:
        want_web_search = bool(self.review_web_search_var.get())
        if (
            self.translator is not None
            and bool(getattr(self.translator, "enable_web_search", False)) == want_web_search
        ):
            return self.translator
        self.translator = None
        tcfg = self.cfg.get("translation", {})
        game_cfg = self.cfg.get("game", {})
        profile = resolve_translation_model_profile(tcfg, "vlm_responses")
        api_key = self._api_key_for_profile(profile)
        self.translator = VlmResponsesTranslator(
            api_key=api_key,
            model=profile.model,
            responses_base_url=profile.base_url,
            temperature=float(tcfg.get("temperature", 1.3)),
            enable_thinking=bool(tcfg.get("enable_thinking", True)),
            thinking_budget=tcfg.get("thinking_budget", None),
            preserve_thinking=bool(tcfg.get("preserve_thinking", False)),
            timeout_sec=int(tcfg.get("timeout_sec", 45)),
            timeout_backoff_sec=int(tcfg.get("timeout_backoff_sec", 15)),
            max_retries=int(tcfg.get("max_retries", 2)),
            retry_delay_sec=float(tcfg.get("retry_delay_sec", 1.5)),
            empty_max_attempts=int(tcfg.get("empty_max_attempts", 3)),
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

    def _translation_mode(self) -> str:
        mode = str(self.cfg.get("translation", {}).get("mode", "vlm_responses") or "vlm_responses").strip().lower()
        return "ocr_chat_completions" if mode in {"ocr_chat", "ocr_chat_completions", "ocr_llm"} else "vlm_responses"

    def _api_key_for_profile(self, profile: TranslationModelProfile) -> str:
        if profile.api_key:
            return profile.api_key
        if not profile.api_key_file:
            return ""
        p = Path(profile.api_key_file)
        if not p.is_absolute():
            cfg_dir = self.config_path.resolve().parent if self.config_path else ROOT
            candidates = [(cfg_dir / p).resolve(), (ROOT / p).resolve()]
            p = next((cand for cand in candidates if cand.exists()), candidates[0])
        return load_api_key(p)

    def _build_text_translator(self) -> ChatCompletionsTextTranslator:
        want_web_search = bool(self.review_web_search_var.get())
        if (
            self.text_translator is not None
            and bool(getattr(self.text_translator, "enable_web_search", False)) == want_web_search
        ):
            return self.text_translator
        self.text_translator = None
        tcfg = self.cfg.get("translation", {})
        game_cfg = self.cfg.get("game", {})
        profile = resolve_translation_model_profile(tcfg, "ocr_chat_completions")
        api_key = self._api_key_for_profile(profile)
        profile_temp, profile_top_p, profile_top_k = _profile_chat_params(tcfg, profile.name)
        self.text_translator = ChatCompletionsTextTranslator(
            api_key=api_key,
            model=profile.model,
            base_url=profile.base_url,
            temperature=profile_temp if profile_temp is not None else 1.3,
            top_p=profile_top_p,
            top_k=profile_top_k,
            timeout_sec=int(tcfg.get("timeout_sec", 45)),
            timeout_backoff_sec=int(tcfg.get("timeout_backoff_sec", 15)),
            max_retries=int(tcfg.get("max_retries", 2)),
            retry_delay_sec=float(tcfg.get("retry_delay_sec", 1.5)),
            empty_max_attempts=int(tcfg.get("empty_max_attempts", 3)),
            disable_env_proxy=bool(tcfg.get("disable_env_proxy", True)),
            game_name=str(game_cfg.get("name", "")),
            source_language=str(game_cfg.get("source_language", "ja")),
            target_language=str(game_cfg.get("target_language", "zh-CN")),
            io_log_enabled=True,
            io_log_path=self.cache_path.parent / "review_chat_io.log",
            log_fn=lambda s: self._set_status_threadsafe(f"[LLM] {s}"),
            enable_web_search=want_web_search,
            context_window=int(tcfg.get("chat_context_window", 8)),
        )
        return self.text_translator

    def _text_extraction_backend(self) -> str:
        tcfg = self.cfg.get("translation", {}) if isinstance(self.cfg, dict) else {}
        if not isinstance(tcfg, dict):
            tcfg = {}
        return _normalize_text_extraction_backend(
            tcfg.get("text_extraction_backend", tcfg.get("text_extraction_mode", "ocr"))
        )

    def _build_image_text_extractor(self) -> VlmImageTextExtractor:
        if self.image_text_extractor is not None:
            return self.image_text_extractor
        tcfg = self.cfg.get("translation", {}) if isinstance(self.cfg, dict) else {}
        if not isinstance(tcfg, dict):
            tcfg = {}
        game_cfg = self.cfg.get("game", {}) if isinstance(self.cfg.get("game"), dict) else {}
        backend = self._text_extraction_backend()
        requested = str(tcfg.get("text_extraction_model_profile", "") or "").strip()
        profile = resolve_translation_model_profile(tcfg, TEXT_EXTRACTION_PROFILE_MODE, requested)
        api_key = self._api_key_for_profile(profile)
        profile_temp, profile_top_p, profile_top_k = _profile_chat_params(tcfg, profile.name)
        temp = profile_temp if profile_temp is not None else float(tcfg.get("text_extraction_temperature", 0.0))
        budget_raw = tcfg.get("text_extraction_thinking_budget", None)
        budget: int | None = None
        if budget_raw is not None and str(budget_raw).strip() != "":
            try:
                budget = int(budget_raw)
            except Exception:
                budget = None
        self.image_text_extractor = VlmImageTextExtractor(
            api_key=api_key,
            model=profile.model,
            base_url=profile.base_url,
            backend=backend,
            temperature=temp,
            top_p=profile_top_p,
            top_k=profile_top_k,
            max_tokens=int(tcfg.get("text_extraction_max_tokens", 512) or 512),
            enable_thinking=bool(tcfg.get("text_extraction_enable_thinking", False)),
            thinking_budget=budget,
            preserve_thinking=bool(tcfg.get("text_extraction_preserve_thinking", False)),
            timeout_sec=int(tcfg.get("text_extraction_timeout_sec", tcfg.get("timeout_sec", 90))),
            timeout_backoff_sec=int(tcfg.get("timeout_backoff_sec", 15)),
            max_retries=int(tcfg.get("max_retries", 2)),
            retry_delay_sec=float(tcfg.get("retry_delay_sec", 1.5)),
            empty_max_attempts=int(tcfg.get("empty_max_attempts", 3)),
            disable_env_proxy=bool(tcfg.get("disable_env_proxy", True)),
            game_name=str(game_cfg.get("name", "") or ""),
            source_language=str(game_cfg.get("source_language", "ja") or "ja"),
            io_log_enabled=True,
            io_log_path=self.cache_path.parent / "review_vlm_text_extraction_io.log",
            log_fn=lambda s: self._set_status_threadsafe(f"[VLM_OCR] {s}"),
        )
        return self.image_text_extractor

    def _translation_extra_requirements(self) -> str:
        game_cfg = self.cfg.get("game", {}) if isinstance(self.cfg, dict) else {}
        if not isinstance(game_cfg, dict):
            return ""
        return str(game_cfg.get("extra_requirements", "") or "").strip()

    def _get_history_items_before_index(self, index: int) -> list[dict[str, str]] | None:
        if index <= 0 or not self.entries:
            return None
        tcfg = self.cfg.get("translation", {}) if isinstance(self.cfg, dict) else {}
        if not isinstance(tcfg, dict):
            tcfg = {}
        limit = max(0, int(tcfg.get("history_n", 5) or 5))
        if limit <= 0:
            return None
        items: list[dict[str, str]] = []
        for e in reversed(self.entries[:index]):
            if not isinstance(e, dict):
                continue
            dtype = str(e.get("dialogue_type", "") or "").strip().lower()
            if dtype in {"blank_no_name", "blank", "title"}:
                continue
            original = str(e.get("text_original", "") or "").strip()
            translation = str(e.get("translation_subtitle", "") or "").strip()
            if not original or not translation:
                continue
            try:
                start = float(e.get("time_start", 0.0) or 0.0)
                end = float(e.get("time_end", 0.0) or 0.0)
                time_text = f"{start:.2f}-{end:.2f}s"
            except Exception:
                time_text = ""
            items.insert(
                0,
                {
                    "time": time_text,
                    "speaker": str(e.get("speaker", "") or ""),
                    "original": original,
                    "translation": translation,
                },
            )
            if len(items) >= limit:
                break
        return items or None

    def _get_context_around_index(self, index: int, before: bool) -> list[str | dict[str, str]] | None:
        ctx_window = max(0, int(getattr(self.text_translator, "context_window", 0)))
        if ctx_window <= 0 or index < 0 or index >= len(self.entries):
            return None
        entries = self.entries
        result: list[str | dict[str, str]] = []
        if before:
            start = max(0, index - ctx_window)
            for i in range(index - 1, start - 1, -1):
                t = str(entries[i].get("text_original", "") or "").strip()
                if t:
                    speaker = str(entries[i].get("speaker", "") or "").strip()
                    translation = str(entries[i].get("translation_subtitle", "") or "").strip()
                    if translation:
                        result.insert(0, {"speaker": speaker, "original": t, "translation": translation})
                    else:
                        result.insert(0, {"speaker": speaker, "original": t})
        else:
            end = min(len(entries), index + 1 + ctx_window)
            for i in range(index + 1, end):
                t = str(entries[i].get("text_original", "") or "").strip()
                if t:
                    speaker = str(entries[i].get("speaker", "") or "").strip()
                    result.append({"speaker": speaker, "original": t})
        return result or None

    def _get_cached_marker_score(self, sec: float) -> tuple[float | None, float | None]:
        """Return (marker_score, marker2_score) from cache, or (None, None)."""
        if not self.marker_score_cache:
            return None, None
        ranges = self.marker_score_cache.get("ranges")
        if not isinstance(ranges, list):
            return None, None
        sec = max(0.0, float(sec))
        for r in ranges:
            start_sec = float(r.get("start_sec", 0.0))
            end_sec = float(r.get("end_sec", start_sec))
            if sec < start_sec or sec > end_sec:
                continue
            scan_fps = float(r.get("scan_fps", 10.0))
            scores = r.get("scores")
            if not isinstance(scores, list) or not scores:
                continue
            idx = int(round((sec - start_sec) * scan_fps))
            if 0 <= idx < len(scores):
                return float(scores[idx]), None
        return None, None

    def _compute_marker_scores_on_the_fly(self) -> tuple[float | None, float | None]:
        """Compute marker/marker2 scores from the currently displayed frame."""
        if self.current_frame_rgb is None:
            return None, None
        try:
            img = Image.fromarray(self.current_frame_rgb)
        except Exception:
            return None, None
        ms = None
        m2s = None
        if self._marker_matcher is not None and self._marker_roi is not None:
            try:
                x1, y1, x2, y2 = self._marker_roi
                crop = img.crop((x1, y1, x2, y2))
                gray = np.asarray(crop.convert("L"), dtype=np.uint8)
                ms = float(self._marker_matcher.score(gray))
            except Exception:
                ms = None
        if self._marker2_matcher is not None and self._marker2_roi is not None:
            try:
                x1, y1, x2, y2 = self._marker2_roi
                crop = img.crop((x1, y1, x2, y2))
                gray = np.asarray(crop.convert("L"), dtype=np.uint8)
                m2s = float(self._marker2_matcher.score(gray))
            except Exception:
                m2s = None
        return ms, m2s

    def _update_marker_score_display(self, sec: float) -> None:
        ms_cache, m2s_cache = self._get_cached_marker_score(sec)
        ms = ms_cache
        m2s = m2s_cache
        if ms is None or m2s is None:
            ms_fly, m2s_fly = self._compute_marker_scores_on_the_fly()
            if ms is None:
                ms = ms_fly
            if m2s is None:
                m2s = m2s_fly
        parts: list[str] = []
        if ms is not None:
            parts.append(f"Marker: {ms:.3f}")
        if m2s is not None:
            parts.append(f"Marker2: {m2s:.3f}")
        self.marker_score_var.set(" | ".join(parts) if parts else "Marker: N/A")

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
        for e in self.entries:
            e.pop("srt_start", None)
            e.pop("srt_end", None)
        self.current_index = min(self.current_index, max(0, len(self.entries) - 1))
        self._rebuild_suspect_indices()
        self._init_title_time_defaults()
        self.status_var.set(f"已加载缓存: {self.cache_path}，共 {len(self.entries)} 段")
        self._show_segment(self.current_index)

    def _entry_translation_review_reasons(self, entry: dict[str, Any]) -> list[str]:
        dialogue_type = str(entry.get("dialogue_type", "") or "").strip().lower()
        if dialogue_type in {"blank_no_name", "blank", "title"}:
            return []
        translated = str(entry.get("translation_subtitle", "") or "").strip()
        original = str(entry.get("text_original", "") or "").strip()
        debug_text = _debug_subtitle_from_entry(entry).strip()
        reasons: list[str] = []
        if not translated:
            reasons.append("translation_missing")
        elif debug_text and translated == debug_text:
            reasons.append("translation_fallback_debug_text")
        elif translated.startswith("[DEBUG]"):
            reasons.append("translation_fallback_debug_text")
        if has_kanji_overlap_from_original(original, translated, min_len=5):
            reasons.append("translation_kanji_overlap_with_original")
        return reasons

    def _entry_review_reasons(self, entry: dict[str, Any]) -> list[str]:
        items: list[Any] = []
        for raw in (entry.get("review_reason"), entry.get("review_reasons")):
            if isinstance(raw, list):
                items.extend(raw)
            elif raw is not None:
                items.append(raw)
        items.extend(self._entry_translation_review_reasons(entry))
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            s = str(item or "").strip()
            if not s or s in seen:
                continue
            out.append(s)
            seen.add(s)
        return out

    def _entry_is_suspect(self, entry: dict[str, Any]) -> bool:
        if bool(entry.get("needs_review", False)) or bool(entry.get("suspect", False)):
            return True
        return bool(self._entry_review_reasons(entry))

    def _entry_with_review_metadata(self, entry: dict[str, Any]) -> dict[str, Any]:
        reasons = self._entry_review_reasons(entry)
        if reasons:
            entry.pop("review_reason", None)
            entry["needs_review"] = True
            entry["review_reason"] = reasons
        else:
            entry.pop("review_reason", None)
            if not self._entry_is_suspect(entry):
                entry.pop("needs_review", None)
        return entry

    def _rebuild_suspect_indices(self) -> None:
        self.suspect_indices = [
            i for i, e in enumerate(self.entries)
            if isinstance(e, dict) and self._entry_is_suspect(e)
        ]
        self._update_suspect_info()

    def _update_suspect_info(self) -> None:
        total = len(self.suspect_indices)
        if not total:
            self.suspect_info_var.set("存疑: 0")
            return
        pos = 0
        for j, idx in enumerate(self.suspect_indices, start=1):
            if idx == self.current_index:
                pos = j
                break
        if pos:
            self.suspect_info_var.set(f"存疑: {pos}/{total}")
        else:
            self.suspect_info_var.set(f"存疑: {total}")

    def _clip_preview_text(self, value: Any, limit: int = 150) -> str:
        text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
        if len(text) <= limit:
            return text
        return f"{text[: max(0, limit - 3)]}..."

    def _entry_preview_summary(self, idx: int | None) -> str:
        if not self.entries:
            return "(无 cache)"
        if idx is None or idx < 0 or idx >= len(self.entries):
            return "(无)"
        entry = self.entries[idx]
        try:
            st, ed = self._entry_times(idx)
            time_text = f"{st:.2f}-{ed:.2f}s"
        except Exception:
            time_text = "-"
        reasons = self._entry_review_reasons(entry)
        lines = [
            f"idx: {idx + 1}/{len(self.entries)}",
            f"segment_id: {entry.get('segment_id', idx + 1)}",
            f"time: {time_text}",
            f"speaker: {self._clip_preview_text(entry.get('speaker', ''), 80)}",
            f"original: {self._clip_preview_text(entry.get('text_original', ''), 180)}",
            f"translation: {self._clip_preview_text(entry.get('translation_subtitle', ''), 180)}",
        ]
        if self._entry_is_suspect(entry):
            lines.append("needs_review: true")
            if reasons:
                lines.append(f"reason: {self._clip_preview_text(', '.join(reasons), 220)}")
        return "\n".join(lines)

    def _update_neighbor_preview(self) -> None:
        if not self.neighbor_preview_texts:
            return
        indices = {
            "prev": self._neighbor_index_by_segment_id(-1),
            "next": self._neighbor_index_by_segment_id(1),
        }
        for key, idx in indices.items():
            txt = self.neighbor_preview_texts.get(key)
            if txt is None:
                continue
            txt.configure(state=tk.NORMAL)
            txt.delete("1.0", tk.END)
            txt.insert("1.0", self._entry_preview_summary(idx))
            txt.configure(state=tk.DISABLED)

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

    def _find_title_entry(self) -> dict[str, Any] | None:
        for e in self.entries:
            if self._seg_numeric_id(e) == 0:
                return e
        for e in self.entries:
            if str(e.get("dialogue_type", "")).strip().lower() == "title":
                return e
        return None

    def _load_title_fields_from_cache(self, *, show_status: bool = False) -> bool:
        title = self._find_title_entry()
        if title is None:
            if show_status:
                self.status_var.set("未找到 segment_id=0 的标题段")
            return False
        try:
            st = float(title.get("time_start", 0.0) or 0.0)
        except Exception:
            st = 0.0
        try:
            ed = float(title.get("time_end", st) or st)
        except Exception:
            ed = st
        self.title_start_var.set(f"{max(0.0, st):.2f}")
        self.title_end_var.set(f"{max(0.0, ed):.2f}")
        self._update_title_capture_default()
        self.title_info_var.set(str(title.get("speaker", "") or ""))
        if show_status:
            self.status_var.set("已从 cache 的标题段读取标题时间和翻译信息")
        return True

    def _update_title_capture_default(self) -> None:
        try:
            st = self._parse_time_input(self.title_start_var.get())
            ed = self._parse_time_input(self.title_end_var.get())
        except Exception:
            return
        self.title_capture_var.set(f"{((float(st) + float(ed)) / 2.0):.3f}")

    def _init_title_time_defaults(self) -> None:
        if not self.entries:
            self.title_start_var.set("0.00")
            self.title_end_var.set("2.00")
            self._update_title_capture_default()
            self.title_info_var.set("")
            return
        if self._load_title_fields_from_cache(show_status=False):
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
        self._update_title_capture_default()

    def _resolve_video_from_cache(self) -> None:
        if self.video_path and self.video_path.exists():
            self.video_var.set(str(self.video_path))
            return
        video = self.cache_payload.get("video")
        if not video:
            return
        v = self._resolve_cache_ref_path(video)
        self.video_path = v
        self.video_var.set(str(v))
        if not v.exists():
            self.status_var.set(f"视频不存在，可继续编辑 JSON；如需预览/重译请重新选择视频: {v}")

    def _resolve_cache_ref_path(self, value: Any) -> Path:
        p = Path(str(value or ""))
        if p.is_absolute():
            return p.resolve()
        for base in (self.cache_path.resolve().parent, ROOT):
            cand = (base / p).resolve()
            if cand.exists():
                return cand
        return (self.cache_path.resolve().parent / p).resolve()

    def _path_for_cache(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.cache_path.resolve().parent).as_posix()
        except Exception:
            pass
        try:
            return resolved.relative_to(ROOT).as_posix()
        except Exception:
            return str(resolved)

    def _sync_video_path_to_cache(self) -> None:
        raw = self.video_var.get().strip()
        if not raw:
            return
        path = Path(raw).resolve()
        if path.exists():
            self.cache_payload["video"] = self._path_for_cache(path)

    def _sync_config_path_to_cache(self) -> None:
        raw = self.config_var.get().strip()
        if not raw:
            return
        path = Path(raw).resolve()
        if path.exists():
            self.cache_payload["config_path"] = self._path_for_cache(path)

    def _pick_video(self) -> None:
        p = filedialog.askopenfilename(
            title="选择视频",
            initialdir=self._dialog_initial_dir("review.video", self.video_var.get().strip() or ROOT),
            filetypes=[("视频文件", "*.mp4 *.mkv *.mov *.avi *.webm"), ("所有文件", "*.*")],
        )
        if not p:
            return
        self._remember_dialog_dir("review.video", p)
        self.video_var.set(str(Path(p).resolve()))
        self._reload_video()

    def _pick_cache(self) -> None:
        p = filedialog.askopenfilename(
            title="选择缓存",
            initialdir=self._dialog_initial_dir("review.cache", self.cache_var.get().strip() or ROOT),
            filetypes=[("JSON", "*.json"), ("所有文件", "*.*")],
        )
        if not p:
            return
        self._remember_dialog_dir("review.cache", p)
        self.cache_var.set(str(Path(p).resolve()))
        self._load_cache()

    def _pick_config(self) -> None:
        p = filedialog.askopenfilename(
            title="选择配置",
            initialdir=self._dialog_initial_dir("review.config", self.config_var.get().strip() or (ROOT / "config")),
            filetypes=[("YAML", "*.yaml *.yml"), ("JSON", "*.json"), ("所有文件", "*.*")],
        )
        if not p:
            return
        self._remember_dialog_dir("review.config", p)
        self.config_var.set(str(Path(p).resolve()))
        self._load_config()

    def _reload_video(self) -> None:
        raw = self.video_var.get().strip()
        if not raw:
            return
        self.video_path = Path(raw).resolve()
        if not self.video_path.exists():
            self.status_var.set(f"视频不存在，可继续编辑 JSON；如需预览/重译请重新选择视频: {self.video_path}")
            return
        try:
            self._open_video(self.video_path)
        except Exception as exc:
            self.status_var.set(f"打开视频失败，可继续编辑 JSON: {exc}")
            return
        self._sync_video_path_to_cache()
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
            self._save_window_state()
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
        parsed = self._entry_with_review_metadata(parsed)
        self.entries[self.current_index] = parsed
        self._rebuild_suspect_indices()
        self._update_seg_info(self.current_index)
        self._update_neighbor_preview()
        if self.status_var.get().startswith("提示：segment_id"):
            return
        self.status_var.set(f"已更新第 {self.current_index + 1} 段（内存）")

    def _merge_with_prev(self) -> None:
        if self._block_if_insert_dialog_active():
            return
        if self.current_index <= 0 or not self.entries:
            self.status_var.set("已是第一段，无法向前合并")
            return
        self._merge_segments(self.current_index - 1, self.current_index, self.current_index, "上一段")

    def _merge_with_next(self) -> None:
        if self._block_if_insert_dialog_active():
            return
        if self.current_index >= len(self.entries) - 1 or not self.entries:
            self.status_var.set("已是最后一段，无法向后合并")
            return
        self._merge_segments(self.current_index, self.current_index + 1, self.current_index, "下一段")

    def _block_if_insert_dialog_active(self) -> bool:
        active = self._active_dialog_state()
        if active is None:
            return False
        win = active.get("window")
        try:
            if win is not None:
                win.lift()
                win.focus_set()
        except Exception:
            pass
        self.status_var.set("请先确认或取消插入段窗口")
        return True

    def _save_undo_snapshot(self, kind: str) -> None:
        self._undo_stack.append(
            {
                "kind": kind,
                "entries": copy.deepcopy(self.entries),
                "current_idx": self.current_index,
            }
        )
        self._redo_stack.clear()

    def _current_snapshot(self, kind: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "entries": copy.deepcopy(self.entries),
            "current_idx": self.current_index,
        }

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.entries = copy.deepcopy(snapshot.get("entries", []))
        self._rebuild_suspect_indices()
        self.current_index = max(0, min(int(snapshot.get("current_idx", 0)), len(self.entries) - 1))
        self._show_segment(self.current_index)

    def _undo_label(self, kind: str) -> str:
        return (
            "合并"
            if kind == "merge"
            else "插入"
            if kind == "insert"
            else "删除"
            if kind == "delete"
            else "操作"
        )

    def _insert_before_segment(self) -> None:
        self._insert_segment(-1)

    def _insert_after_segment(self) -> None:
        self._insert_segment(1)

    def _delete_current_segment(self) -> None:
        if self._block_if_insert_dialog_active():
            return
        if not self.entries:
            self.status_var.set("无分段，无法删除")
            return
        try:
            self._save_current_entry()
            idx = self.current_index
            entry = self.entries[idx]
            st, ed = self._entry_times(idx)
            sid = entry.get("segment_id", idx + 1)
            if not messagebox.askyesno(
                "删除当前段",
                f"确认删除 segment_id={sid} ({st:.2f}s-{ed:.2f}s)？\n可用“撤回操作”恢复。",
                parent=self.root,
            ):
                return
            deleted_sid = self._positive_segment_id(entry)
            self._save_undo_snapshot("delete")
            deleted = dict(entry)
            self.entries = self.entries[:idx] + self.entries[idx + 1:]
            if deleted_sid is not None:
                self._shift_segment_ids_from_index(idx, delta=-1)
            self._rebuild_suspect_indices()
            self.current_index = max(0, min(idx, len(self.entries) - 1))
            self._show_segment(self.current_index)
            self.status_var.set(f"已删除 seg={deleted.get('segment_id')}（可撤回）")
        except Exception as exc:
            self.status_var.set(f"删除当前段失败: {exc}")

    def _default_insert_times(self, direction: int) -> tuple[float, float]:
        cur_st, cur_ed = self._entry_times(self.current_index)
        if direction < 0:
            end = max(0.0, cur_st)
            prev_idx = self._neighbor_index_by_segment_id(-1)
            if prev_idx is not None:
                _, prev_ed = self._entry_times(prev_idx)
                start = max(0.0, min(prev_ed, end))
            else:
                start = max(0.0, end - 1.0)
            if end <= start:
                start = max(0.0, end - 1.0)
            return start, max(start, end)

        start = max(0.0, cur_ed)
        next_idx = self._neighbor_index_by_segment_id(1)
        if next_idx is not None:
            next_st, _ = self._entry_times(next_idx)
            end = max(start, next_st)
        else:
            end = start + 1.0
        if end <= start:
            end = start + 1.0
        return start, end

    def _insert_segment(self, direction: int) -> None:
        if not self.entries:
            self.status_var.set("无分段，无法插入")
            return
        try:
            active = self._active_dialog_state()
            if active is not None:
                win = active.get("window")
                if win is not None:
                    win.lift()
                    win.focus_set()
                self.status_var.set("已有插入段窗口，请先确认或取消")
                return
            self._save_current_entry()
            anchor_idx = self.current_index
            st, ed = self._default_insert_times(direction)
            self._open_insert_dialog(
                anchor_idx=anchor_idx,
                st=st,
                ed=ed,
                direction=direction,
            )
        except Exception as exc:
            self.status_var.set(f"插入段失败: {exc}")

    def _active_dialog_state(self) -> dict[str, Any] | None:
        if hasattr(self, "_active_dialog") and self._active_dialog:
            win = self._active_dialog.get("window")
            try:
                if win is not None and win.winfo_exists():
                    return self._active_dialog
            except Exception:
                pass
            self._active_dialog = None
            
        state = self._insert_dialog
        if not state:
            return None
        win = state.get("window")
        try:
            if win is not None and win.winfo_exists():
                return state
        except Exception:
            pass
        self._insert_dialog = None
        return None

    def _set_active_dialog_time(self, key: str, value: float) -> bool:
        state = self._active_dialog_state()
        if state is None:
            return False
            
        # check merge dialog format
        cb = state.get("on_time_capture")
        if cb:
            cb(key, value)
            self.status_var.set(f"已将时间提取至弹窗 ({key}={float(value):.3f}s)")
            return True
            
        # check insert dialog format
        var = state.get("time_start_var" if key == "start" else "time_end_var")
        if isinstance(var, tk.StringVar):
            var.set(f"{float(value):.3f}")
            self.status_var.set(f"已将弹窗{'开始' if key == 'start' else '结束'}时间设置为 {float(value):.3f}s")
            return True
            
        return False

    def _positive_segment_id(self, entry: dict[str, Any]) -> int | None:
        try:
            sid = int(entry.get("segment_id"))
        except Exception:
            return None
        return sid if sid > 0 else None

    def _insert_target_segment_id(self, anchor_idx: int, direction: int) -> int:
        base_sid = self._positive_segment_id(self.entries[anchor_idx])
        if base_sid is not None:
            return base_sid if direction < 0 else base_sid + 1
        positive_ids = [sid for e in self.entries if (sid := self._positive_segment_id(e)) is not None]
        return min(positive_ids) if positive_ids else 1

    def _shift_segment_ids_from_index(self, start_idx: int, *, delta: int = 1) -> None:
        for entry in self.entries[max(0, int(start_idx)):]:
            sid = self._positive_segment_id(entry)
            if sid is not None:
                new_sid = sid + int(delta)
                if new_sid > 0:
                    entry["segment_id"] = new_sid

    def _speaker_choices(self) -> list[str]:
        choices: list[str] = []
        seen: set[str] = set()
        for entry in self.entries:
            speaker = str(entry.get("speaker", "") or "").strip()
            if not speaker or speaker in seen:
                continue
            choices.append(speaker)
            seen.add(speaker)
        return choices

    def _subtitle_style_for_speaker(self, speaker: str, dialogue_type: str) -> tuple[dict[str, Any], list[str]]:
        dt = str(dialogue_type or "").strip().lower()
        if dt in {"blank_no_name", "blank", "title"}:
            return {}, []
        style, ambiguous = _resolve_speaker_subtitle_style(speaker, self.subtitle_style_cfg)
        return style, ["speaker_style_ambiguous"] if ambiguous else []

    def _make_title_entry(self, st: float, ed: float, original: str, translated: str) -> dict[str, Any]:
        return self._entry_with_review_metadata(
            {
                "segment_id": 0,
                "raw_id": 0,
                "time_start": float(st),
                "time_end": float(ed),
                "dialogue_type": "title",
                "speaker": str(self.title_info_var.get() or "").strip(),
                "text_original": str(original or "").strip(),
                "translation_subtitle": normalize_quotes_for_subtitle(str(translated or "").strip()),
            }
        )

    def _title_recognition_mode_key(self) -> str:
        raw = str(self.title_recognition_mode_var.get() or "").strip()
        mode = TITLE_RECOGNITION_MODE_BY_LABEL.get(raw, raw)
        return mode if mode in TITLE_RECOGNITION_MODE_LABELS else "auto"

    def _resolve_title_recognition_mode(self) -> str:
        mode = self._title_recognition_mode_key()
        if mode != "auto":
            return mode
        return "local_vlm" if self._text_extraction_backend() != "ocr" else "direct_vlm"

    def _translate_title_direct_vlm(self, img_path: Path) -> tuple[str, str, dict[str, int]]:
        tr = self._build_translator()
        return tr.translate_single_image_ja_to_zh_cn_structured_with_tag(
            image_path=img_path,
            request_tag="title-segment",
            history_items=None,
            custom_prompt=self._get_custom_prompt(),
            extra_requirements=self._translation_extra_requirements(),
        )

    def _translate_title_local_vlm(self, img_path: Path) -> tuple[str, str, dict[str, int]]:
        extractor = self._build_image_text_extractor()
        tcfg = self.cfg.get("translation", {}) if isinstance(self.cfg, dict) else {}
        if not isinstance(tcfg, dict):
            tcfg = {}
        game_cfg = self.cfg.get("game", {}) if isinstance(self.cfg.get("game"), dict) else {}
        target_language = str(tcfg.get("target_language", game_cfg.get("target_language", "zh-CN")) or "zh-CN")
        return extractor.translate_single_image_text_with_tag(
            image_path=img_path,
            request_tag="title-local-vlm",
            custom_prompt=self._get_custom_prompt(),
            extra_requirements=self._translation_extra_requirements(),
            target_language=target_language,
        )

    def _open_insert_dialog(
        self,
        *,
        anchor_idx: int,
        st: float,
        ed: float,
        direction: int,
    ) -> None:
        win = tk.Toplevel(self.root)
        win.title("插入段")
        win.geometry("720x520")
        win.transient(self.root)
        win.resizable(True, True)

        target_seg_id = self._insert_target_segment_id(anchor_idx, direction)
        time_start_var = tk.StringVar(value=f"{float(st):.3f}")
        time_end_var = tk.StringVar(value=f"{float(ed):.3f}")
        type_var = tk.StringVar(value="speaker_dialogue")
        base_speaker = str(self.entries[anchor_idx].get("speaker", "") or "").strip()
        speaker_var = tk.StringVar(value=base_speaker)
        speaker_choices = self._speaker_choices()

        body = ttk.Frame(win, padding=8)
        body.pack(fill=tk.BOTH, expand=True)
        label = "当前段前" if direction < 0 else "当前段后"
        ttk.Label(
            body,
            text=(
                f"插入位置：{label}；新段 segment_id={target_seg_id}，"
                "确认后后续非 Title 段会自动 +1；raw_id=manual_insert。"
            ),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        ttk.Label(body, text="开始(秒)").grid(row=1, column=0, sticky="w")
        ttk.Entry(body, textvariable=time_start_var, width=14).grid(row=1, column=1, sticky="w", padx=(6, 12))
        ttk.Label(body, text="结束(秒)").grid(row=1, column=2, sticky="w")
        ttk.Entry(body, textvariable=time_end_var, width=14).grid(row=1, column=3, sticky="w", padx=(6, 0))

        ttk.Label(body, text="类型").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            body,
            textvariable=type_var,
            values=["speaker_dialogue", "blank_no_name"],
            state="readonly",
            width=18,
        ).grid(row=2, column=1, sticky="w", padx=(6, 12), pady=(8, 0))
        ttk.Label(body, text="speaker").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Combobox(
            body,
            textvariable=speaker_var,
            values=speaker_choices,
            width=14,
        ).grid(row=2, column=3, sticky="ew", padx=(6, 0), pady=(8, 0))

        ttk.Label(body, text="原文").grid(row=3, column=0, sticky="nw", pady=(8, 0))
        original_text = tk.Text(body, height=5, wrap=tk.WORD, font=("Consolas", 10))
        original_text.grid(row=3, column=1, columnspan=3, sticky="nsew", padx=(6, 0), pady=(8, 0))

        ttk.Label(body, text="译文").grid(row=4, column=0, sticky="nw", pady=(8, 0))
        translation_text = tk.Text(body, height=5, wrap=tk.WORD, font=("Consolas", 10))
        translation_text.grid(row=4, column=1, columnspan=3, sticky="nsew", padx=(6, 0), pady=(8, 0))

        ttk.Label(body, text="预览").grid(row=5, column=0, sticky="nw", pady=(8, 0))
        preview_text = tk.Text(body, height=8, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED, bg="#f5f5f5")
        preview_text.grid(row=5, column=1, columnspan=3, sticky="nsew", padx=(6, 0), pady=(8, 0))
        body.columnconfigure(3, weight=1)
        body.rowconfigure(3, weight=1)
        body.rowconfigure(4, weight=1)
        body.rowconfigure(5, weight=1)

        def _build_entry() -> dict[str, Any]:
            start = self._parse_time_input(time_start_var.get())
            end = self._parse_time_input(time_end_var.get())
            if end < start:
                raise RuntimeError("结束时间必须大于等于开始时间")
            dialogue_type = str(type_var.get() or "speaker_dialogue").strip()
            speaker = str(speaker_var.get() or "").strip()
            original = original_text.get("1.0", tk.END).strip()
            translation = normalize_quotes_for_subtitle(translation_text.get("1.0", tk.END).strip())
            if dialogue_type == "blank_no_name":
                speaker = ""
                original = ""
                translation = ""
            subtitle_style, style_reasons = self._subtitle_style_for_speaker(speaker, dialogue_type)
            review_reasons: list[str] = []
            if self.auto_mark_manual_review.get():
                review_reasons.append("manual_insert")
            review_reasons.extend(style_reasons)
            entry = {
                "segment_id": target_seg_id,
                "raw_id": MANUAL_INSERT_RAW_ID,
                "time_start": float(start),
                "time_end": float(end),
                "dialogue_type": dialogue_type,
                "speaker": speaker,
                "text_original": original,
                "translation_subtitle": translation,
                "review_reason": review_reasons,
                "subtitle_style": subtitle_style,
            }
            return self._entry_with_review_metadata(entry)

        def _refresh_preview(*_: object) -> None:
            try:
                payload = _build_entry()
                text = json.dumps(payload, ensure_ascii=False, indent=2)
            except Exception as exc:
                text = f"预览失败: {exc}"
            preview_text.configure(state=tk.NORMAL)
            preview_text.delete("1.0", tk.END)
            preview_text.insert("1.0", text)
            preview_text.configure(state=tk.DISABLED)

        def _on_ok() -> None:
            try:
                inserted = _build_entry()
                self._save_current_entry()
            except Exception as exc:
                self.status_var.set(f"插入段参数无效: {exc}")
                return
            self._save_undo_snapshot("insert")
            insert_idx = anchor_idx if direction < 0 else anchor_idx + 1
            self._shift_segment_ids_from_index(insert_idx, delta=1)
            self.entries = self.entries[:insert_idx] + [inserted] + self.entries[insert_idx:]
            self.current_index = insert_idx
            self._rebuild_suspect_indices()
            label_text = "前" if direction < 0 else "后"
            self._insert_dialog = None
            try:
                win.destroy()
            except Exception:
                pass
            self._show_segment(insert_idx)
            self.status_var.set(
                f"已在原第 {anchor_idx + 1} 段{label_text}插入 seg={inserted.get('segment_id')} "
                f"({inserted.get('time_start'):.2f}s-{inserted.get('time_end'):.2f}s)（可撤回）"
            )

        def _on_cancel() -> None:
            self._insert_dialog = None
            try:
                win.destroy()
            except Exception:
                pass

        for var in (time_start_var, time_end_var, type_var, speaker_var):
            var.trace_add("write", _refresh_preview)
        original_text.bind("<KeyRelease>", _refresh_preview)
        translation_text.bind("<KeyRelease>", _refresh_preview)

        btns = ttk.Frame(body)
        btns.grid(row=6, column=0, columnspan=4, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="确认插入", command=_on_ok).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btns, text="取消", command=_on_cancel).pack(side=tk.RIGHT, padx=2)
        win.protocol("WM_DELETE_WINDOW", _on_cancel)
        win.bind("<Escape>", lambda _e: _on_cancel())
        _refresh_preview()
        self.root.update_idletasks()
        win.update_idletasks()
        rx = self.root.winfo_rootx()
        ry = self.root.winfo_rooty()
        rw = self.root.winfo_width()
        rh = self.root.winfo_height()
        ww = win.winfo_width()
        wh = win.winfo_height()
        px = rx + max(0, (rw - ww) // 2)
        py = ry + max(0, (rh - wh) // 2)
        win.geometry(f"{ww}x{wh}+{px}+{py}")
        self._insert_dialog = {
            "window": win,
            "time_start_var": time_start_var,
            "time_end_var": time_end_var,
        }
        win.focus_set()

    def _merge_segments(self, idx_a: int, idx_b: int, current_idx: int, label: str) -> None:
        self._save_current_entry()
        entry_a = dict(self.entries[idx_a])
        entry_b = dict(self.entries[idx_b])
        ts_a = float(entry_a.get("time_start", 0.0))
        te_a = float(entry_a.get("time_end", 0.0))
        ts_b = float(entry_b.get("time_start", 0.0))
        te_b = float(entry_b.get("time_end", 0.0))
        a_label = f"seg={entry_a.get('segment_id', idx_a + 1)} ({ts_a:.2f}s-{te_a:.2f}s)"
        b_label = f"seg={entry_b.get('segment_id', idx_b + 1)} ({ts_b:.2f}s-{te_b:.2f}s)"
        dt_a = str(entry_a.get("dialogue_type", "speaker_dialogue") or "speaker_dialogue").strip()
        dt_b = str(entry_b.get("dialogue_type", "speaker_dialogue") or "speaker_dialogue").strip()
        non_blank = [dt for dt in (dt_a, dt_b) if dt != "blank_no_name"]
        default_dt = non_blank[0] if non_blank else dt_a
        dt_choices: list[str] | None = None
        if dt_a != dt_b:
            seen: set[str] = set()
            dt_choices = []
            for dt in (dt_a, dt_b):
                if dt not in seen:
                    seen.add(dt)
                    dt_choices.append(dt)
            if dt_choices[0] != default_dt and len(dt_choices) > 1:
                dt_choices.sort(key=lambda x: (x == "blank_no_name", x))
        current_raw_id = self.entries[current_idx].get(
            "raw_id",
            self.entries[current_idx].get("segment_id", current_idx + 1),
        )

        def _on_merge_confirm(choices: dict[str, Any]) -> None:
            speaker = choices["speaker"]
            original = choices["original"]
            translation = choices["translation"]
            dialogue_type = choices.get("dialogue_type", default_dt)
            if dialogue_type == "blank_no_name":
                speaker = ""
                original = ""
                translation = ""
            
            st = choices.get("time_start", min(ts_a, ts_b))
            ed = choices.get("time_end", max(te_a, te_b))
            
            reason_source = choices.get("reason_source", "合并")
            style_source = choices.get("style_source", "重新生成")

            new_speaker_style, speaker_style_reasons = self._subtitle_style_for_speaker(speaker, dialogue_type)
            
            # 1. Handle reasons
            final_reasons: list[Any] = []
            if self.auto_mark_manual_review.get():
                final_reasons.append("manual_merge")
            if reason_source == "合并":
                final_reasons.extend(entry_a.get("review_reason") or [])
                final_reasons.extend(entry_b.get("review_reason") or [])
            elif reason_source == "保留当前段":
                final_reasons.extend((entry_a if current_idx == idx_a else entry_b).get("review_reason") or [])
            elif reason_source.startswith("保留"):
                final_reasons.extend((entry_b if current_idx == idx_a else entry_a).get("review_reason") or [])
            
            # 2. Handle subtitle style (except colors)
            base_style: dict[str, Any] = {}
            if style_source == "保留当前段":
                base_style = dict((entry_a if current_idx == idx_a else entry_b).get("subtitle_style") or {})
            elif style_source.startswith("保留"):
                base_style = dict((entry_b if current_idx == idx_a else entry_a).get("subtitle_style") or {})
            elif style_source == "合并":
                base_style = dict(entry_a.get("subtitle_style") or {})
                base_style.update(entry_b.get("subtitle_style") or {})
                
            final_style = dict(base_style)
            if style_source == "重新生成":
                final_style = dict(new_speaker_style)
                final_reasons.extend(speaker_style_reasons)
            else:
                color_keys = ["primary_colour", "secondary_colour", "outline_colour", "back_colour"]
                for k in color_keys:
                    if k in base_style:
                        final_style.pop(k, None)
                    if k in new_speaker_style:
                        final_style[k] = new_speaker_style[k]
                        
            merged_reasons_list = _merge_review_reasons(*final_reasons)
            
            merged: dict[str, Any] = dict(entry_a)
            merged.update({
                "segment_id": self.entries[current_idx].get("segment_id", current_idx + 1),
                "raw_id": current_raw_id,
                "time_start": float(st),
                "time_end": float(ed),
                "dialogue_type": dialogue_type,
                "speaker": speaker,
                "text_original": original,
                "translation_subtitle": translation,
                "subtitle_style": final_style,
                "review_reason": merged_reasons_list,
            })
            merged = self._entry_with_review_metadata(merged)
            self._save_undo_snapshot("merge")
            new_entries = self.entries[:idx_a] + [merged] + self.entries[idx_b + 1:]
            self.entries = new_entries
            self._rebuild_suspect_indices()
            self.current_index = max(0, min(idx_a, len(self.entries) - 1))
            self._show_segment(self.current_index)
            self.status_var.set(f"已合并 {label} → seg={merged.get('segment_id')} ({st:.2f}s-{ed:.2f}s)（可撤回）")

        self._ask_merge_options(
            entry_a, entry_b, a_label, b_label,
            current_idx_is_a=(current_idx == idx_a),
            merge_dir_label=label,
            seg_id=self.entries[current_idx].get("segment_id", current_idx + 1),
            raw_id=current_raw_id,
            st=min(ts_a, ts_b), ed=max(te_a, te_b),
            dialogue_type=default_dt,
            dt_choices=dt_choices,
            on_ok_callback=_on_merge_confirm,
        )

    def _ask_merge_options(
        self,
        entry_a: dict[str, Any],
        entry_b: dict[str, Any],
        a_label: str,
        b_label: str,
        *,
        current_idx_is_a: bool,
        merge_dir_label: str,
        seg_id: int | str,
        raw_id: Any,
        st: float,
        ed: float,
        dialogue_type: str,
        dt_choices: list[str] | None = None,
        on_ok_callback: Callable[[dict[str, Any]], None],
    ) -> None:
        result: dict[str, Any] = {"speaker": None, "original": None, "translation": None, "dialogue_type": None}
        win = tk.Toplevel(self.root)
        win.title("合并选项")
        win.geometry("780x600")
        win.transient(self.root)
        win.resizable(True, True)

        spk_a = str(entry_a.get("speaker", "") or "").strip()
        spk_b = str(entry_b.get("speaker", "") or "").strip()
        orig_a = str(entry_a.get("text_original", "") or "").strip()
        orig_b = str(entry_b.get("text_original", "") or "").strip()
        trans_a = str(entry_a.get("translation_subtitle", "") or "").strip()
        trans_b = str(entry_b.get("translation_subtitle", "") or "").strip()

        def _join_non_empty(a_val: str, b_val: str) -> str:
            if a_val and b_val:
                return a_val + "\n" + b_val
            return a_val or b_val

        body = ttk.Frame(win, padding=8)
        body.pack(fill=tk.BOTH, expand=True)

        if current_idx_is_a:
            name_a = "当前段"
            name_b = merge_dir_label # "下一段"
            opt_other = f"保留{merge_dir_label}"
        else:
            name_a = merge_dir_label # "上一段"
            name_b = "当前段"
            opt_other = f"保留{merge_dir_label}"
            
        ttk.Label(body, text=f"{name_a} (A): {a_label}").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 2))
        ttk.Label(body, text=f"{name_b} (B): {b_label}").grid(row=1, column=0, columnspan=4, sticky="w")

        all_dt_choices: list[str] = []
        for item in [dialogue_type, *(dt_choices or []), "speaker_dialogue", "blank_no_name"]:
            if item and item not in all_dt_choices:
                all_dt_choices.append(item)
        type_var = tk.StringVar(value=dialogue_type)
        speaker_var = tk.StringVar(value=(spk_a if current_idx_is_a else spk_b) or spk_a or spk_b)
        speaker_choices = self._speaker_choices()
        for speaker in (spk_a, spk_b):
            if speaker and speaker not in speaker_choices:
                speaker_choices.append(speaker)
        speaker_source_var = tk.StringVar(value="保留当前段")
        original_source_var = tk.StringVar(value="保留当前段")
        translation_source_var = tk.StringVar(value="保留当前段")
        reason_source_var = tk.StringVar(value="清除")
        style_source_var = tk.StringVar(value="保留当前段")

        source_options = ["合并", "保留当前段", opt_other]
        reason_options = ["合并", "保留当前段", opt_other, "清除"]
        style_options = ["合并", "保留当前段", opt_other, "重新生成"]

        time_frame = ttk.Frame(body)
        time_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(time_frame, text="开始(秒)").pack(side=tk.LEFT)
        start_var = tk.StringVar(value=f"{st:.2f}")
        ttk.Entry(time_frame, textvariable=start_var, width=10).pack(side=tk.LEFT, padx=(4, 16))
        ttk.Label(time_frame, text="结束(秒)").pack(side=tk.LEFT)
        end_var = tk.StringVar(value=f"{ed:.2f}")
        ttk.Entry(time_frame, textvariable=end_var, width=10).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(body, text="类型").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            body,
            textvariable=type_var,
            values=all_dt_choices,
            state="readonly",
            width=18,
        ).grid(row=3, column=1, sticky="w", padx=(6, 12), pady=(8, 0))
        ttk.Label(body, text="speaker").grid(row=3, column=2, sticky="w", pady=(8, 0))
        ttk.Combobox(
            body,
            textvariable=speaker_var,
            values=speaker_choices,
            width=14,
        ).grid(row=3, column=3, sticky="ew", padx=(6, 0), pady=(8, 0))

        source_frame = ttk.Frame(body)
        source_frame.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))
        for label_text, var, opts in (
            ("speaker", speaker_source_var, source_options),
            ("原文", original_source_var, source_options),
            ("译文", translation_source_var, source_options),
            ("needs_review", reason_source_var, reason_options),
            ("style", style_source_var, style_options),
        ):
            ttk.Label(source_frame, text=label_text).pack(side=tk.LEFT, padx=(0, 2))
            ttk.Combobox(
                source_frame,
                textvariable=var,
                values=opts,
                state="readonly",
                width=10,
            ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Label(body, text="原文").grid(row=5, column=0, sticky="nw", pady=(8, 0))
        original_text = tk.Text(body, height=5, wrap=tk.WORD, font=("Consolas", 10))
        original_text.grid(row=5, column=1, columnspan=3, sticky="nsew", padx=(6, 0), pady=(8, 0))
        original_text.insert("1.0", orig_a if current_idx_is_a else orig_b)

        ttk.Label(body, text="译文").grid(row=6, column=0, sticky="nw", pady=(8, 0))
        translation_text = tk.Text(body, height=5, wrap=tk.WORD, font=("Consolas", 10))
        translation_text.grid(row=6, column=1, columnspan=3, sticky="nsew", padx=(6, 0), pady=(8, 0))
        translation_text.insert("1.0", trans_a if current_idx_is_a else trans_b)

        ttk.Label(body, text="预览").grid(row=7, column=0, sticky="nw", pady=(8, 0))
        preview_text = tk.Text(body, height=8, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED, bg="#f5f5f5")
        preview_text.grid(row=7, column=1, columnspan=3, sticky="nsew", padx=(6, 0), pady=(8, 0))
        body.columnconfigure(3, weight=1)
        body.rowconfigure(5, weight=1)
        body.rowconfigure(6, weight=1)
        body.rowconfigure(7, weight=1)

        def _build_merge_entry() -> dict[str, Any]:
            try:
                curr_st = float(start_var.get() or "0")
            except ValueError:
                curr_st = st
            try:
                curr_ed = float(end_var.get() or "0")
            except ValueError:
                curr_ed = ed

            dt = type_var.get()
            if dt == "blank_no_name":
                s = ""
                o = ""
                t = ""
            else:
                s = str(speaker_var.get() or "").strip()
                o = original_text.get("1.0", tk.END).strip()
                t = normalize_quotes_for_subtitle(translation_text.get("1.0", tk.END).strip())

            style, _ = self._subtitle_style_for_speaker(s, dt)
            entry: dict[str, Any] = {
                "segment_id": seg_id,
                "raw_id": raw_id,
                "time_start": curr_st,
                "time_end": curr_ed,
                "dialogue_type": dt,
                "speaker": s,
                "text_original": o,
                "translation_subtitle": t,
            }
            if style:
                entry["subtitle_style"] = style
            return entry

        def _refresh_preview(*_: object) -> None:
            try:
                payload = _build_merge_entry()
                text = json.dumps(payload, ensure_ascii=False, indent=2)
            except Exception as exc:
                text = f"预览失败: {exc}"
            preview_text.configure(state=tk.NORMAL)
            preview_text.delete("1.0", tk.END)
            preview_text.insert("1.0", text)
            preview_text.configure(state=tk.DISABLED)

        def _source_value(source: str, a_val: str, b_val: str, current_val: str = "") -> str:
            if source == "保留当前段":
                return current_val
            if source == opt_other:
                return b_val if current_idx_is_a else a_val
            return _join_non_empty(a_val, b_val)

        def _apply_source_fields(*_: object) -> None:
            c_spk = spk_a if current_idx_is_a else spk_b
            c_orig = orig_a if current_idx_is_a else orig_b
            c_trans = trans_a if current_idx_is_a else trans_b
            
            speaker_var.set(_source_value(speaker_source_var.get(), spk_a, spk_b, c_spk))
            original_text.delete("1.0", tk.END)
            original_text.insert("1.0", _source_value(original_source_var.get(), orig_a, orig_b, c_orig))
            translation_text.delete("1.0", tk.END)
            translation_text.insert("1.0", _source_value(translation_source_var.get(), trans_a, trans_b, c_trans))
            _refresh_preview()

        start_var.trace_add("write", _refresh_preview)
        end_var.trace_add("write", _refresh_preview)
        type_var.trace_add("write", _refresh_preview)
        speaker_var.trace_add("write", _refresh_preview)
        speaker_source_var.trace_add("write", _apply_source_fields)
        original_source_var.trace_add("write", _apply_source_fields)
        translation_source_var.trace_add("write", _apply_source_fields)
        original_text.bind("<KeyRelease>", _refresh_preview)
        translation_text.bind("<KeyRelease>", _refresh_preview)

        btns = ttk.Frame(body)
        btns.grid(row=8, column=0, columnspan=4, sticky="e", pady=(8, 0))

        def _on_time_capture_event(event_type: str, t: float) -> None:
            if event_type == "start":
                start_var.set(f"{t:.2f}")
            elif event_type == "end":
                end_var.set(f"{t:.2f}")

        self._active_dialog = {
            "window": win,
            "on_time_capture": _on_time_capture_event,
        }

        def _on_ok() -> None:
            dt = type_var.get()
            if dt == "blank_no_name":
                result["speaker"] = ""
                result["original"] = ""
                result["translation"] = ""
            else:
                result["speaker"] = str(speaker_var.get() or "").strip()
                result["original"] = original_text.get("1.0", tk.END).strip()
                result["translation"] = normalize_quotes_for_subtitle(translation_text.get("1.0", tk.END).strip())
            result["dialogue_type"] = dt
            result["reason_source"] = reason_source_var.get()
            result["style_source"] = style_source_var.get()
            try:
                result["time_start"] = float(start_var.get() or "0")
            except ValueError:
                result["time_start"] = st
            try:
                result["time_end"] = float(end_var.get() or "0")
            except ValueError:
                result["time_end"] = ed
            try:
                win.destroy()
            except Exception:
                pass
            if hasattr(self, "_active_dialog") and self._active_dialog and self._active_dialog.get("window") == win:
                self._active_dialog = None
                
            dt = result.get("dialogue_type") or ""
            if dt == "blank_no_name":
                if result.get("original") is None:
                    return
            elif result.get("speaker") is None:
                return
            
            payload = {
                "speaker": result.get("speaker") or "",
                "original": result.get("original") or "",
                "translation": result.get("translation") or "",
                "dialogue_type": dt,
                "reason_source": result.get("reason_source", "合并"),
                "style_source": result.get("style_source", "重新生成"),
                "time_start": result.get("time_start", st),
                "time_end": result.get("time_end", ed),
            }
            on_ok_callback(payload)

        def _on_cancel() -> None:
            result["speaker"] = None
            try:
                win.destroy()
            except Exception:
                pass
            if self._active_dialog and self._active_dialog.get("window") == win:
                self._active_dialog = None

        win.protocol("WM_DELETE_WINDOW", _on_cancel)
        win.bind("<Escape>", lambda _e: _on_cancel())
        
        ttk.Button(btns, text="确认合并", command=_on_ok).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btns, text="取消", command=_on_cancel).pack(side=tk.RIGHT, padx=2)
        
        _refresh_preview()
        self.root.update_idletasks()
        win.update_idletasks()
        rx = self.root.winfo_rootx()
        ry = self.root.winfo_rooty()
        rw = self.root.winfo_width()
        rh = self.root.winfo_height()
        ww = win.winfo_width()
        wh = win.winfo_height()
        px = rx + max(0, (rw - ww) // 2)
        py = ry + max(0, (rh - wh) // 2)
        win.geometry(f"{ww}x{wh}+{px}+{py}")
        
        # Don't use grab_set() to allow non-modal behavior
        # win.focus_set()
        # No wait_window because it's non-modal
        # self.root.wait_window(win)
        pass

    def _undo_merge(self) -> None:
        if not self._undo_stack:
            self.status_var.set("没有可撤回的操作")
            return
        saved = self._undo_stack.pop()
        kind = str(saved.get("kind", "操作") or "操作")
        self._redo_stack.append(self._current_snapshot(kind))
        self._restore_snapshot(saved)
        self.status_var.set(f"已撤回{self._undo_label(kind)}")

    def _redo_operation(self) -> None:
        if not self._redo_stack:
            self.status_var.set("没有可恢复的操作")
            return
        saved = self._redo_stack.pop()
        kind = str(saved.get("kind", "操作") or "操作")
        self._undo_stack.append(self._current_snapshot(kind))
        self._restore_snapshot(saved)
        self.status_var.set(f"已恢复{self._undo_label(kind)}")

    def _save_cache_file(self, *, apply_current: bool = True) -> None:
        if apply_current:
            self._save_current_entry()
        self.entries = [self._entry_with_review_metadata(e) for e in self.entries]
        self._rebuild_suspect_indices()
        self._sync_video_path_to_cache()
        self._sync_config_path_to_cache()
        self.cache_payload["entries"] = self.entries
        self.cache_path.write_text(json.dumps(self.cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        sync_msg = self._sync_all_cache_files()
        self.status_var.set(f"已保存: {self.cache_path} {sync_msg}".strip())

    def _sync_all_cache_files(self) -> str:
        out_dir = self._resolve_output_dir_from_cache(self.cache_path)
        latest = out_dir / "translation_cache_latest.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        src = self.cache_path.resolve()
        dst = latest.resolve()
        msgs: list[str] = []
        if src != dst:
            shutil.copy2(src, dst)
            msgs.append(f"(已同步: {dst})")
        else:
            msgs.append("(latest已是当前文件)")
        source_work = self.cache_payload.get("source_work_cache")
        if source_work and (out_dir / "work").is_dir():
            work_path = (out_dir / source_work).resolve()
            if work_path != src:
                try:
                    work_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, work_path)
                    msgs.append(f"(已回写至: {work_path})")
                except Exception as exc:
                    msgs.append(f"(回写失败: {exc})")
        return " ".join(msgs)

    def _autosave_cache_for_action(self, action: str) -> bool:
        try:
            self._save_cache_file()
            return True
        except Exception as exc:
            self.status_var.set(f"{action}前自动保存失败: {exc}")
            return False

    def _entry_index_by_segment_id(self, segment_id: int) -> int | None:
        for idx, entry in enumerate(self.entries):
            try:
                if int(entry.get("segment_id", 0) or 0) == int(segment_id):
                    return idx
            except Exception:
                continue
        return None

    def _ask_auto_review_update(
        self,
        *,
        pos: int,
        total: int,
        segment_id: int,
        speaker: str,
        original: str,
        current_text: str,
        new_text: str,
        reason: str,
    ) -> bool | None:
        result: dict[str, bool | None] = {"value": None}
        win = tk.Toplevel(self.root)
        win.title("确认自动review修改")
        win.transient(self.root)
        win.resizable(False, False)
        width = 900
        height = 700
        try:
            self.root.update_idletasks()
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            rw = max(1, self.root.winfo_width())
            rh = max(1, self.root.winfo_height())
            px = rx + max(0, (rw - width) // 2)
            py = ry + max(0, (rh - height) // 2)
            win.geometry(f"{width}x{height}+{px}+{py}")
        except Exception:
            win.geometry(f"{width}x{height}")

        root = ttk.Frame(win, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        header = (
            f"第 {pos}/{total} 条自动review建议    "
            f"段号：{segment_id}    说话人：{speaker or '（空）'}"
        )
        ttk.Label(root, text=header, foreground="#1f5f99").pack(anchor="w")
        ttk.Label(root, text="请选择是否应用此修改。窗口大小固定，长文本可滚动查看。").pack(anchor="w", pady=(2, 8))

        def add_box(title: str, text: str, height_lines: int) -> None:
            box = ttk.LabelFrame(root, text=title, padding=4)
            box.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
            frame = ttk.Frame(box)
            frame.pack(fill=tk.BOTH, expand=True)
            scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL)
            txt = tk.Text(
                frame,
                height=height_lines,
                wrap=tk.WORD,
                font=("Consolas", 10),
                yscrollcommand=scroll.set,
            )
            scroll.configure(command=txt.yview)
            txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)
            txt.insert("1.0", text)
            txt.configure(state=tk.DISABLED)

        add_box("原文", original, 5)
        add_box("当前译文", current_text, 5)
        add_box("建议译文", new_text, 5)
        add_box("原因", reason or "（无）", 4)

        btns = ttk.Frame(root)
        btns.pack(fill=tk.X, pady=(6, 0))

        def finish(value: bool | None) -> None:
            result["value"] = value
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()

        ttk.Button(btns, text="应用", command=lambda: finish(True)).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btns, text="跳过", command=lambda: finish(False)).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btns, text="停止后续确认", command=lambda: finish(None)).pack(side=tk.RIGHT)
        win.protocol("WM_DELETE_WINDOW", lambda: finish(None))
        win.bind("<Escape>", lambda _e: finish(None))
        win.bind("<Return>", lambda _e: finish(True))
        try:
            win.grab_set()
            win.focus_set()
        except Exception:
            pass
        self.root.wait_window(win)
        return result["value"]

    def _confirm_auto_review_updates(self, updates: list[dict[str, Any]], report: dict[str, Any]) -> None:
        changed_updates = [item for item in updates if item.get("changed")]
        report_out = copy.deepcopy(report)
        report_out["confirmation_required"] = True
        report_out["confirmed_applied_count"] = 0
        report_out["confirmed_skipped_count"] = 0
        report_out["missing_entry_count"] = 0
        report_out["confirmation_cancelled"] = False
        if not changed_updates:
            self.review_text.delete("1.0", tk.END)
            self.review_text.insert(tk.END, json.dumps(report_out, ensure_ascii=False, indent=2))
            self.review_meta_var.set(f"自动review: 待修改=0 updates={len(updates)}")
            self.status_var.set("自动review 完成：没有需要确认的修改。")
            return

        applied = 0
        skipped = 0
        missing = 0
        cancelled = False
        last_idx: int | None = None
        total = len(changed_updates)
        for pos, update in enumerate(changed_updates, start=1):
            try:
                segment_id = int(update.get("id", 0) or 0)
            except Exception:
                missing += 1
                continue
            idx = self._entry_index_by_segment_id(segment_id)
            if idx is None:
                missing += 1
                continue
            new_text = str(update.get("new_translation", "") or "").strip()
            if not new_text:
                skipped += 1
                continue
            entry = self.entries[idx]
            current_text = str(entry.get("translation_subtitle", "") or "").strip()
            if current_text == new_text:
                skipped += 1
                continue
            speaker = str(update.get("speaker", entry.get("speaker", "")) or "").strip()
            original = str(update.get("original", entry.get("text_original", "")) or "").strip()
            reason = str(update.get("reason", "") or "").strip()
            preview = {
                "segment_id": segment_id,
                "speaker": speaker,
                "original": original,
                "current_translation": current_text,
                "suggested_translation": new_text,
                "reason": reason,
            }
            self._show_segment(idx)
            self.review_text.delete("1.0", tk.END)
            self.review_text.insert(tk.END, json.dumps(preview, ensure_ascii=False, indent=2))
            self.review_meta_var.set(f"自动review确认: {pos}/{total}")
            answer = self._ask_auto_review_update(
                pos=pos,
                total=total,
                segment_id=segment_id,
                speaker=speaker,
                original=original,
                current_text=current_text,
                new_text=new_text,
                reason=reason,
            )
            if answer is None:
                cancelled = True
                break
            if not answer:
                skipped += 1
                continue
            entry["translation_subtitle"] = new_text
            applied += 1
            last_idx = idx

        report_out["confirmed_applied_count"] = applied
        report_out["confirmed_skipped_count"] = skipped
        report_out["missing_entry_count"] = missing
        report_out["confirmation_cancelled"] = cancelled
        if applied:
            self._rebuild_suspect_indices()
            self._show_segment(last_idx if last_idx is not None else self.current_index)
            self._save_cache_file(apply_current=False)
        else:
            self._rebuild_suspect_indices()
            self._show_segment(self.current_index)
        self.review_text.delete("1.0", tk.END)
        self.review_text.insert(tk.END, json.dumps(report_out, ensure_ascii=False, indent=2))
        self.review_meta_var.set(f"自动review: 已应用={applied} 跳过={skipped} 缺失={missing}")
        tail = "，用户已停止后续确认" if cancelled else ""
        save_note = "，已保存缓存" if applied else ""
        self.status_var.set(f"自动review 确认完成：应用 {applied} 条，跳过 {skipped} 条{tail}{save_note}。")

    def _auto_review_current_cache(self) -> None:
        try:
            self._save_current_entry()
        except Exception as exc:
            messagebox.showerror("自动review", f"当前段 JSON 保存失败：{exc}", parent=self.root)
            return
        if not self._reload_config_for_action("自动review"):
            return
        if not self._edit_auto_review_game_config_dialog():
            self.status_var.set("自动review 已取消。")
            return
        entries_snapshot = copy.deepcopy(self.entries)
        tcfg = copy.deepcopy(self.cfg.get("translation", {}) if isinstance(self.cfg, dict) else {})
        if not isinstance(tcfg, dict):
            tcfg = {}
        game_cfg = self.cfg.get("game", {}) if isinstance(self.cfg.get("game"), dict) else {}
        game_name = str(game_cfg.get("name", "") or "").strip()
        extra_requirements = str(game_cfg.get("extra_requirements", "") or "").strip()
        glossary_parts = []
        if game_name:
            glossary_parts.append(f"游戏名：{game_name}")
        if extra_requirements:
            glossary_parts.append(extra_requirements)
        glossary = "\n\n".join(glossary_parts).strip()
        model_profile = str(self.auto_review_model_var.get() or "").strip()
        review_entries = dialogue_review_entries_from_cache_entries(entries_snapshot)
        if not review_entries:
            messagebox.showinfo("自动review", "没有可 review 的 dialogue 条目。", parent=self.root)
            return

        def _job() -> None:
            updates, report = run_auto_review_entries(
                entries=review_entries,
                glossary=glossary,
                tr_cfg=tcfg,
                model_profile=model_profile,
                chunk_size=int(tcfg.get("auto_review_chunk_size", 80) or 80),
                timeout_sec=int(tcfg.get("auto_review_timeout_sec", tcfg.get("timeout_sec", 120))),
                max_tokens=int(tcfg.get("auto_review_max_tokens", 0) or 0),
                parse_retries=int(tcfg.get("auto_review_parse_retries", 1)),
                repair_max_tokens=int(tcfg.get("auto_review_repair_max_tokens", 0) or 0),
                review_mode=str(tcfg.get("auto_review_mode", "thorough") or "thorough"),
                stream=False,
                log_fn=lambda s: self._set_status_threadsafe(f"[自动review] {s}"),
            )

            self.root.after(0, lambda: self._confirm_auto_review_updates(updates, report))

        self._run_bg("自动review", _job, show_review_progress=True)

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
            "ignite.pipeline",
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

    def _generate_subtitles_for_current_cache(self) -> Path:
        self._save_current_entry()
        self._save_cache_file(apply_current=False)

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
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-30:])
            raise RuntimeError(f"生成字幕失败(退出码={proc.returncode})\n{tail}")
        return output_dir

    def _generate_subtitles_from_cache(self) -> None:
        if not self._autosave_cache_for_action("生成字幕"):
            return

        def _job() -> None:
            output_dir = self._generate_subtitles_for_current_cache()
            self._set_status_threadsafe(f"字幕已生成: {output_dir}")

        self._run_bg("生成字幕", _job)

    def _default_embed_output_path(self) -> Path:
        raw_video = self.video_var.get().strip()
        stem = Path(raw_video).stem if raw_video else "video"
        safe_stem = re.sub(r'[<>:"/\\\\|?*]+', "_", stem).strip("._ ") or "video"
        return (ROOT / "outputs" / safe_stem / f"{safe_stem}_subtitled.mp4").resolve()

    def _refresh_embed_defaults(self) -> None:
        if not self.embed_ffmpeg_path_var.get().strip():
            self.embed_ffmpeg_path_var.set(str(self.ffmpeg_path))
        self.embed_output_path_var.set(str(self._default_embed_output_path()))

    def _current_existing_path(self, raw: str) -> Path | None:
        text = str(raw or "").strip()
        if not text:
            return None
        path = Path(text).resolve()
        return path if path.exists() else None

    def _open_archive_dialog(self) -> None:
        if not self._autosave_cache_for_action("归档项目"):
            return
        video_path = self._current_existing_path(self.video_var.get())
        config_path = self._current_existing_path(self.config_var.get())
        found_hard = find_hard_subtitle_video(self.cache_path.resolve(), video_path)

        win = tk.Toplevel(self.root)
        win.title("归档当前项目")
        win.geometry("820x230")
        win.transient(self.root)
        body = ttk.Frame(win, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)

        dest_root_var = tk.StringVar(value=self._dialog_initial_dir("review.archive_root", ROOT.parent))
        hard_video_var = tk.StringVar(value=str(found_hard) if found_hard else "")

        def _browse_dest_root() -> None:
            p = filedialog.askdirectory(
                title="选择归档根目录",
                initialdir=self._dialog_initial_dir("review.archive_root", dest_root_var.get() or ROOT.parent),
                parent=win,
            )
            if p:
                self._remember_dialog_dir("review.archive_root", p)
                dest_root_var.set(str(Path(p).resolve()))

        def _browse_hard_video() -> None:
            p = filedialog.askopenfilename(
                title="选择硬字幕视频（可选）",
                initialdir=self._dialog_initial_dir(
                    "review.hard_sub_video",
                    hard_video_var.get() or self.video_var.get() or ROOT,
                ),
                filetypes=[("视频文件", "*.mp4 *.mkv *.mov *.avi *.webm"), ("所有文件", "*.*")],
                parent=win,
            )
            if p:
                self._remember_dialog_dir("review.hard_sub_video", p)
                hard_video_var.set(str(Path(p).resolve()))

        def _open_generate_hard_video() -> None:
            self._refresh_embed_defaults()
            hard_video_var.set(self.embed_output_path_var.get().strip())
            self._open_embed_subtitles_dialog()

        def _run_archive() -> None:
            dest_root = Path(dest_root_var.get().strip()).resolve()
            hard_video = Path(hard_video_var.get().strip()).resolve() if hard_video_var.get().strip() else None
            try:
                win.destroy()
            except Exception:
                pass

            def _job() -> None:
                result = archive_project(
                    cache_path=self.cache_path.resolve(),
                    dest_root=dest_root,
                    video_path=video_path,
                    config_path=config_path,
                    hard_sub_video=hard_video,
                )
                missing = "；缺失可选文件: " + ", ".join(result.missing) if result.missing else ""

                def _done() -> None:
                    self.status_var.set(f"已归档到: {result.archive_dir}{missing}")
                    messagebox.showinfo(
                        "归档完成",
                        f"已归档到:\n{result.archive_dir}\n\ncache:\n{result.cache_path}{missing}",
                        parent=self.root,
                    )

                self.root.after(0, _done)

            self.status_var.set(f"开始归档到: {dest_root}")
            self._run_bg("归档项目", _job)

        ttk.Label(body, text="归档根目录").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(body, textvariable=dest_root_var).grid(row=0, column=1, sticky="ew", padx=(8, 4), pady=4)
        ttk.Button(body, text="浏览", command=_browse_dest_root).grid(row=0, column=2, sticky="ew", pady=4)
        ttk.Label(body, text="硬字幕视频").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(body, textvariable=hard_video_var).grid(row=1, column=1, sticky="ew", padx=(8, 4), pady=4)
        ttk.Button(body, text="选择", command=_browse_hard_video).grid(row=1, column=2, sticky="ew", pady=4)
        ttk.Label(body, text="不填写硬字幕视频时会自动查找；找不到则跳过。config 会归档为合并后的独立 config.yaml。", foreground="#555").grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 4))
        btns = ttk.Frame(body)
        btns.grid(row=3, column=0, columnspan=3, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="生成硬字幕视频...", command=_open_generate_hard_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="取消", command=win.destroy).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="开始归档", command=_run_archive).pack(side=tk.LEFT, padx=4)

    def _open_embed_subtitles_dialog(self) -> None:
        if not self._autosave_cache_for_action("生成内嵌字幕视频"):
            return
        self._refresh_embed_defaults()
        win = tk.Toplevel(self.root)
        win.title("生成内嵌字幕视频")
        win.geometry("880x390")
        win.transient(self.root)

        body = ttk.Frame(win, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)

        def _row(idx: int, label: str, var: tk.StringVar, browse: Callable[[], None] | None = None) -> None:
            ttk.Label(body, text=label).grid(row=idx, column=0, sticky="w", pady=4)
            ttk.Entry(body, textvariable=var).grid(row=idx, column=1, sticky="ew", padx=(8, 4), pady=4)
            if browse is not None:
                ttk.Button(body, text="浏览", command=browse).grid(row=idx, column=2, sticky="ew", pady=4)

        def _browse_ffmpeg() -> None:
            p = filedialog.askopenfilename(
                title="选择 ffmpeg",
                initialdir=self._dialog_initial_dir("review.ffmpeg", self.ffmpeg_path.parent if self.ffmpeg_path else ROOT),
                filetypes=[("ffmpeg", "*.exe"), ("所有文件", "*.*")],
                parent=win,
            )
            if p:
                self._remember_dialog_dir("review.ffmpeg", p)
                self.embed_ffmpeg_path_var.set(str(Path(p).resolve()))

        def _browse_output() -> None:
            cur = Path(self.embed_output_path_var.get().strip() or self._default_embed_output_path())
            p = filedialog.asksaveasfilename(
                title="保存内嵌字幕视频",
                initialdir=self._dialog_initial_dir("review.embed_output", cur.parent),
                initialfile=cur.name,
                defaultextension=".mp4",
                filetypes=[("MP4", "*.mp4"), ("所有文件", "*.*")],
                parent=win,
            )
            if p:
                self._remember_dialog_dir("review.embed_output", p)
                self.embed_output_path_var.set(str(Path(p).resolve()))

        def _set_cpu_x264() -> None:
            self.embed_vcodec_var.set("libx264")
            self.embed_crf_var.set("18")
            self.embed_preset_var.set("medium")

        def _set_nvenc() -> None:
            self.embed_vcodec_var.set("h264_nvenc")
            self.embed_crf_var.set("20")
            self.embed_preset_var.set("p5")

        def _set_qsv() -> None:
            self.embed_vcodec_var.set("h264_qsv")
            self.embed_crf_var.set("20")
            self.embed_preset_var.set("medium")

        def _set_amf() -> None:
            self.embed_vcodec_var.set("h264_amf")
            self.embed_crf_var.set("20")
            self.embed_preset_var.set("balanced")

        _row(0, "ffmpeg", self.embed_ffmpeg_path_var, _browse_ffmpeg)
        _row(1, "输出视频", self.embed_output_path_var, _browse_output)
        _row(2, "视频编码(-c:v)", self.embed_vcodec_var)
        _row(3, "质量(CRF/CQ/QP)", self.embed_crf_var)
        _row(4, "Preset", self.embed_preset_var)
        _row(5, "音频编码(-c:a)", self.embed_acodec_var)
        _row(6, "额外输入参数", self.embed_extra_input_args_var)
        _row(7, "额外输出参数", self.embed_extra_output_args_var)

        preset_bar = ttk.Frame(body)
        preset_bar.grid(row=8, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(preset_bar, text="编码预设").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(preset_bar, text="CPU x264", command=_set_cpu_x264).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_bar, text="NVIDIA NVENC", command=_set_nvenc).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_bar, text="Intel QSV", command=_set_qsv).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_bar, text="AMD AMF", command=_set_amf).pack(side=tk.LEFT, padx=2)

        btns = ttk.Frame(body)
        btns.grid(row=9, column=0, columnspan=3, sticky="e", pady=(14, 0))
        ttk.Button(btns, text="重置默认值", command=self._reset_embed_defaults).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="取消", command=win.destroy).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="生成", command=lambda: self._start_embed_subtitled_video(win)).pack(side=tk.LEFT, padx=4)

    def _reset_embed_defaults(self) -> None:
        self.embed_ffmpeg_path_var.set(str(self.ffmpeg_path))
        self.embed_output_path_var.set(str(self._default_embed_output_path()))
        self.embed_vcodec_var.set("libx264")
        self.embed_crf_var.set("18")
        self.embed_preset_var.set("medium")
        self.embed_acodec_var.set("copy")
        self.embed_extra_input_args_var.set("")
        self.embed_extra_output_args_var.set("")

    def _start_embed_subtitled_video(self, win: tk.Toplevel) -> None:
        try:
            win.destroy()
        except Exception:
            pass

        def _job() -> None:
            output_dir = self._generate_subtitles_for_current_cache()
            ass_path = output_dir / "subtitles.ass"
            if not ass_path.exists():
                raise RuntimeError(f"字幕文件不存在: {ass_path}")
            video_path = Path(self.video_var.get().strip()).resolve()
            if not video_path.exists():
                raise RuntimeError(f"视频不存在: {video_path}")
            out_path = Path(self.embed_output_path_var.get().strip() or self._default_embed_output_path()).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = self._build_embed_subtitled_video_cmd(
                video_path=video_path,
                ass_path=ass_path.resolve(),
                output_path=out_path,
            )
            self._set_status_threadsafe(f"开始生成内嵌字幕视频: {out_path}")
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
                tail = "\n".join((proc.stdout or "").splitlines()[-40:])
                raise RuntimeError(f"ffmpeg生成失败(退出码={proc.returncode})\n{tail}")
            self._set_status_threadsafe(f"内嵌字幕视频已生成: {out_path}")

        self._run_bg("生成内嵌字幕视频", _job)

    def _split_custom_ffmpeg_args(self, text: str) -> list[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        return shlex.split(raw, posix=False)

    def _ass_filter_path(self, ass_path: Path) -> str:
        try:
            rel = ass_path.resolve().relative_to(ROOT)
            p = f"./{rel.as_posix()}"
        except Exception:
            p = ass_path.resolve().as_posix()
        p = p.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        return f"ass='{p}'"

    def _build_embed_subtitled_video_cmd(self, video_path: Path, ass_path: Path, output_path: Path) -> list[str]:
        ffmpeg = Path(self.embed_ffmpeg_path_var.get().strip() or self.ffmpeg_path).resolve()
        if not ffmpeg.exists():
            raise RuntimeError(f"ffmpeg不存在: {ffmpeg}")
        vcodec = self.embed_vcodec_var.get().strip() or "libx264"
        quality = self.embed_crf_var.get().strip() or "18"
        preset = self.embed_preset_var.get().strip() or "medium"
        acodec = self.embed_acodec_var.get().strip() or "copy"
        cmd = [str(ffmpeg), "-y"]
        cmd.extend(self._split_custom_ffmpeg_args(self.embed_extra_input_args_var.get()))
        cmd.extend(["-i", str(video_path), "-vf", self._ass_filter_path(ass_path)])
        cmd.extend(["-c:v", vcodec])
        codec_l = vcodec.lower()
        if codec_l in {"libx264", "libx265"}:
            if quality:
                cmd.extend(["-crf", quality])
            if preset:
                cmd.extend(["-preset", preset])
        elif codec_l.endswith("_nvenc"):
            if preset:
                cmd.extend(["-preset", preset])
            if quality:
                cmd.extend(["-rc", "vbr", "-cq", quality])
        elif codec_l.endswith("_qsv"):
            if preset:
                cmd.extend(["-preset", preset])
            if quality:
                cmd.extend(["-global_quality", quality])
        elif codec_l.endswith("_amf"):
            if preset:
                cmd.extend(["-quality", preset])
            if quality:
                cmd.extend(["-qp_i", quality, "-qp_p", quality])
        else:
            if preset:
                cmd.extend(["-preset", preset])
        cmd.extend(["-c:a", acodec])
        cmd.extend(self._split_custom_ffmpeg_args(self.embed_extra_output_args_var.get()))
        cmd.append(str(output_path))
        return cmd

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
            st = self._parse_time_input(self.title_start_var.get())
            ed = self._parse_time_input(self.title_end_var.get())
            if ed < st:
                raise RuntimeError("Title End 必须大于等于 Title Start")
            capture_sec = self._parse_time_input(self.title_capture_var.get())
            mode = self._resolve_title_recognition_mode()

            roi_key = "title_ocr_roi"
            roi = self.review_rois.get(roi_key)
            if not roi:
                raise RuntimeError(f"未找到ROI: {roi_key}")
            ok, frame_bgr = self._read_frame_ffmpeg_once(capture_sec)
            if not ok or frame_bgr is None:
                ok, frame_bgr = self._read_frame_safe(capture_sec, prefer_fast=True)
            if not ok or frame_bgr is None:
                raise RuntimeError(f"无法读取标题中间时间帧: {capture_sec:.3f}s")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            crop = self._crop_from_frame(frame_rgb, roi)
            out_dir = self.cache_path.parent / "review_tmp"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            img_path = out_dir / f"title_roi_{capture_sec:.3f}_{ts}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if mode == "local_vlm":
                original_text, translated_text, usage = self._translate_title_local_vlm(img_path)
                mode_label = "本地VLM直识别"
            else:
                original_text, translated_text, usage = self._translate_title_direct_vlm(img_path)
                mode_label = "翻译API直识别"
            title_entry = self._make_title_entry(st, ed, str(original_text or ""), str(translated_text or ""))
            idx = self._upsert_title_entry(title_entry)
            self.last_review_result = dict(title_entry)
            self.root.after(
                0,
                lambda: (
                    self._show_segment(idx),
                    self._show_review_result(
                        title_entry,
                        usage,
                        f"Title 段已插入/更新（{mode_label}，截图时间 {capture_sec:.3f}s，内存）",
                    ),
                ),
            )

        self._run_bg("Title段插入", _job, show_review_progress=True)

    def _insert_blank_title_segment(self) -> None:
        try:
            self._save_current_entry()
            st = self._parse_time_input(self.title_start_var.get())
            ed = self._parse_time_input(self.title_end_var.get())
            if ed < st:
                raise RuntimeError("Title End 必须大于等于 Title Start")
            title_entry = self._make_title_entry(st, ed, "", "")
            idx = self._upsert_title_entry(title_entry)
            self.last_review_result = dict(title_entry)
            self._show_segment(idx)
            self._show_review_result(title_entry, {}, "Title 空白段已插入/更新（未截图，未调用VLM，内存）")
        except Exception as exc:
            self.status_var.set(f"直接插入Title空白段失败: {exc}")

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
        info = f"段 {idx + 1}/{len(self.entries)} | segment_id={e.get('segment_id', idx + 1)} | {st:.2f}-{ed:.2f}s"
        reasons = self._entry_review_reasons(e)
        if self._entry_is_suspect(e):
            info += " | 存疑"
            if reasons:
                info += f": {', '.join(reasons[:3])}"
                if len(reasons) > 3:
                    info += "..."
        self.seg_info_var.set(info)

    def _show_segment(self, idx: int, request_prefetch: bool = True) -> None:
        if not self.entries:
            self.json_text.delete("1.0", tk.END)
            self.json_text.insert("1.0", "{}")
            self._update_neighbor_preview()
            return
        self._set_current_segment_ui(idx, request_prefetch=False)
        self._seek_to_segment(idx, request_prefetch=request_prefetch)

    def _set_current_segment_ui(self, idx: int, request_prefetch: bool = True) -> None:
        if not self.entries:
            self.json_text.delete("1.0", tk.END)
            self.json_text.insert("1.0", "{}")
            self._update_neighbor_preview()
            return
        idx = max(0, min(idx, len(self.entries) - 1))
        self.current_index = idx
        seg_id = self.entries[idx].get("segment_id")
        self.goto_var.set(str(seg_id if seg_id is not None else (idx + 1)))
        self._update_seg_info(idx)
        self._update_suspect_info()
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert("1.0", json.dumps(self.entries[idx], ensure_ascii=False, indent=2))
        self._update_neighbor_preview()
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
            sync_segment=False,
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

    def _nav_sort_key(self, idx: int) -> tuple[int, int, int]:
        entry = self.entries[idx]
        seg_num = self._seg_numeric_id(entry)
        if seg_num is None:
            return (1, idx, idx)
        return (0, seg_num, idx)

    def _indices_by_segment_id(self, indices: list[int] | None = None) -> list[int]:
        if indices is None:
            indices = list(range(len(self.entries)))
        return sorted(indices, key=self._nav_sort_key)

    def _neighbor_index_by_segment_id(self, direction: int) -> int | None:
        if not self.entries:
            return None
        order = self._indices_by_segment_id()
        try:
            pos = order.index(self.current_index)
        except ValueError:
            pos = max(0, min(self.current_index, len(order) - 1))
        target_pos = pos + int(direction)
        if target_pos < 0 or target_pos >= len(order):
            return None
        return order[target_pos]

    def _adjacent_index_by_segment_id(self, direction: int) -> int:
        neighbor = self._neighbor_index_by_segment_id(direction)
        return self.current_index if neighbor is None else neighbor

    def _prev_segment(self) -> None:
        self._save_current_entry()
        self._show_segment(self._adjacent_index_by_segment_id(-1), request_prefetch=True)

    def _next_segment(self) -> None:
        self._save_current_entry()
        self._show_segment(self._adjacent_index_by_segment_id(1), request_prefetch=True)

    def _jump_suspect(self, direction: int) -> None:
        self._save_current_entry()
        self._rebuild_suspect_indices()
        if not self.suspect_indices:
            self.status_var.set("没有存疑段")
            return
        ordered = self._indices_by_segment_id(self.suspect_indices)
        cur_key = self._nav_sort_key(int(self.current_index))
        if direction >= 0:
            candidates = [idx for idx in ordered if self._nav_sort_key(idx) > cur_key]
            target = candidates[0] if candidates else ordered[0]
        else:
            candidates = [idx for idx in ordered if self._nav_sort_key(idx) < cur_key]
            target = candidates[-1] if candidates else ordered[-1]
        self._show_segment(target, request_prefetch=True)

    def _prev_suspect(self) -> None:
        self._jump_suspect(-1)

    def _next_suspect(self) -> None:
        self._jump_suspect(1)

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

    def _set_segment_start_to_current_time(self) -> None:
        try:
            val = float(self.current_sec or 0.0)
        except Exception:
            return
        if self._set_active_dialog_time("start", val):
            return
        if not self.entries:
            return
        idx = self.current_index
        entry = dict(self.entries[idx])
        entry["time_start"] = float(val)
        # preserve other metadata
        self.entries[idx] = self._entry_with_review_metadata(entry)
        # update UI
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert("1.0", json.dumps(self.entries[idx], ensure_ascii=False, indent=2))
        self._update_seg_info(idx)
        self._update_neighbor_preview()
        self.status_var.set(f"已将第 {idx + 1} 段开始时间更新为 {val:.3f}s（内存）")

    def _set_segment_end_to_current_time(self) -> None:
        try:
            val = float(self.current_sec or 0.0)
        except Exception:
            return
        if self._set_active_dialog_time("end", val):
            return
        if not self.entries:
            return
        idx = self.current_index
        entry = dict(self.entries[idx])
        entry["time_end"] = float(val)
        self.entries[idx] = self._entry_with_review_metadata(entry)
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert("1.0", json.dumps(self.entries[idx], ensure_ascii=False, indent=2))
        self._update_seg_info(idx)
        self._update_neighbor_preview()
        self.status_var.set(f"已将第 {idx + 1} 段结束时间更新为 {val:.3f}s（内存）")

    def _set_title_start_to_current_time(self) -> None:
        try:
            val = float(self.current_sec or 0.0)
        except Exception:
            return
        self.title_start_var.set(f"{val:.3f}")
        self.status_var.set(f"已将标题开始时间设置为 {val:.3f}s")

    def _set_title_end_to_current_time(self) -> None:
        try:
            val = float(self.current_sec or 0.0)
        except Exception:
            return
        self.title_end_var.set(f"{val:.3f}")
        self.status_var.set(f"已将标题结束时间设置为 {val:.3f}s")

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
        sync_segment: bool,
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
                if sync_segment:
                    self._sync_segment_ui_by_time(target_sec)
                self._update_marker_score_display(target_sec)
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
        sync_segment: bool = True,
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
                sync_segment=bool(sync_segment),
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
                self._update_marker_score_display(sec)
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
            if sync_segment:
                self._sync_segment_ui_by_time(sec)
            self._update_marker_score_display(sec)
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
        if self.show_all_rois.get():
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

    def _crop_from_frame(self, frame_rgb: Any, roi: list[int]) -> Any:
        x0, y0, x1, y1 = [int(v) for v in roi]
        h, w = frame_rgb.shape[:2]
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(f"ROI无效: {roi}")
        return frame_rgb[y0:y1, x0:x1].copy()

    def _crop(self, roi: list[int]) -> Any:
        if self.current_frame_rgb is None:
            raise RuntimeError("当前无帧")
        return self._crop_from_frame(self.current_frame_rgb, roi)

    def _review_by_new_crops(self) -> None:
        if not self._reload_config_for_action("截图复译"):
            return

        def _job() -> None:
            self._save_current_entry()
            if not self.entries:
                raise RuntimeError("无分段")
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
            extra_requirements = self._translation_extra_requirements()
            history_items = self._get_history_items_before_index(self.current_index)
            if self._translation_mode() == "ocr_chat_completions":
                if self._text_extraction_backend() == "ocr":
                    ocr = build_ocr_engine(self.cfg.get("ocr", {}))
                    spk_result = ocr.recognize(name_img)
                    orig_result = ocr.recognize(dia_img)
                    spk = str(spk_result.text or "").strip() or speaker_hint
                    orig = str(orig_result.text or "").strip()
                else:
                    extractor = self._build_image_text_extractor()
                    spk, orig, _extract_usage = extractor.extract_text_from_images(
                        speaker_image=name_img,
                        dialogue_image=dia_img,
                        request_tag=f"review-extract-{seg_no}",
                    )
                    spk = str(spk or "").strip() or speaker_hint
                    orig = str(orig or "").strip()
                if not orig:
                    raise RuntimeError("文本抽取未识别到对话原文")
                tr_text = self._build_text_translator()
                trans, usage = tr_text.translate_text_with_prompt(
                    original_text=orig,
                    speaker=spk,
                    request_tag=f"review-ocr-chat-{seg_no}",
                    custom_prompt=custom,
                    extra_requirements=extra_requirements,
                    context_before=self._get_context_around_index(self.current_index, before=True),
                    context_after=self._get_context_around_index(self.current_index, before=False),
                )
            else:
                tr = self._build_translator()
                spk, orig, trans, usage = tr.translate_image_ja_to_zh_cn_structured_with_tag(
                    image_path=dia_img,
                    speaker_image_path=name_img,
                    speaker=speaker_hint,
                    request_tag=f"review-seg-{seg_no}",
                    history_items=history_items,
                    custom_prompt=custom,
                    extra_requirements=extra_requirements,
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

        self._run_bg("截图复译", _job, show_review_progress=True)

    def _review_by_text_only(self) -> None:
        if not self._reload_config_for_action("原文重译"):
            return

        def _job() -> None:
            parsed = self._read_current_json_from_editor()
            original_text = str(parsed.get("text_original", "") or "").strip()
            if not original_text:
                raise RuntimeError("当前段 text_original 为空")
            speaker = str(parsed.get("speaker", "") or "").strip()
            custom = self._get_custom_prompt()
            extra_requirements = self._translation_extra_requirements()
            history_items = self._get_history_items_before_index(self.current_index)
            if self._translation_mode() == "ocr_chat_completions":
                tr_text = self._build_text_translator()
                translated, usage = tr_text.translate_text_with_prompt(
                    original_text=original_text,
                    speaker=speaker,
                    request_tag=f"review-text-{self.current_index + 1}",
                    custom_prompt=custom,
                    extra_requirements=extra_requirements,
                    context_before=self._get_context_around_index(self.current_index, before=True),
                    context_after=self._get_context_around_index(self.current_index, before=False),
                )
            else:
                tr = self._build_translator()
                translated, usage = tr.translate_text_with_prompt(
                    original_text=original_text,
                    speaker=speaker,
                    request_tag=f"review-text-{self.current_index + 1}",
                    custom_prompt=custom,
                    history_items=history_items,
                    extra_requirements=extra_requirements,
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

        self._run_bg("原文重译", _job, show_review_progress=True)

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
        if bool(self.last_review_result.get("needs_review", False)):
            e["needs_review"] = True
        if self.last_review_result.get("review_reason"):
            e["review_reason"] = self.last_review_result.get("review_reason")
        self.entries[self.current_index] = self._entry_with_review_metadata(e)
        self._show_segment(self.current_index)
        self.status_var.set("已一键替换当前段（内存）")

    def _apply_last_result_translation(self) -> None:
        if not self.last_review_result or not self.entries:
            self.status_var.set("没有可应用的复译结果")
            return
        e = self.entries[self.current_index]
        e["translation_subtitle"] = str(self.last_review_result.get("translation_subtitle", "") or e.get("translation_subtitle", ""))
        if bool(self.last_review_result.get("needs_review", False)):
            e["needs_review"] = True
        if self.last_review_result.get("review_reason"):
            e["review_reason"] = self.last_review_result.get("review_reason")
        self.entries[self.current_index] = self._entry_with_review_metadata(e)
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
            for base in (cache_path.resolve().parent, ROOT):
                cand = (base / config_path).resolve()
                if cand.exists():
                    config_path = cand
                    break
            else:
                config_path = (cache_path.resolve().parent / config_path).resolve()
    assert config_path is not None
    if not config_path.exists():
        raise RuntimeError(f"配置不存在: {config_path}")
    app = CacheReviewApp(cache_path=cache_path, video_path=video_path, config_path=config_path)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
