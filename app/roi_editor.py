from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_gamevideo_subtitles.config import load_config  # noqa: E402
from auto_gamevideo_subtitles.event_detect import MarkerTemplateMatcher  # noqa: E402


ROI_KEYS = [
    "name_roi",
    "dialogue_roi",
    "marker_roi",
    "subtitle_location",
    "title_ocr_roi",
    "title_translation_location",
    "title_info_location",
]
ROI_COLORS = {
    "name_roi": "#ffbf00",
    "dialogue_roi": "#0896e0",
    "marker_roi": "#e32636",
    "subtitle_location": "#bf00ff",
    "title_ocr_roi": "#c0c0c0",
    "title_translation_location": "#36bf36",
    "title_info_location": "#ff0da6",
}


def _load_raw_cfg(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    text_strip = text.strip()
    if not text_strip:
        return {}
    if text_strip.startswith("{"):
        return json.loads(text_strip)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_raw_cfg(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        path.write_text(text, encoding="utf-8")
        return
    except Exception:
        pass
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class RoiEditorApp:
    def __init__(self, video_path: str, config_path: str, output_name: str = "") -> None:
        self.root = tk.Tk()
        self.root.title("Profile编辑器 - IGNITE")
        self.root.geometry("1460x980")

        self.video_path_var = tk.StringVar(value=video_path)
        self.config_path_var = tk.StringVar(value=config_path)
        self.roi_key_var = tk.StringVar(value="name_roi")
        self.time_var = tk.StringVar(value="0.00")
        self.status_var = tk.StringVar(value="就绪")
        self.template_paths_var = tk.StringVar(value="")
        self.template_selected_var = tk.StringVar(value="")
        self.shift_mode_var = tk.StringVar(value="vertical")
        self.vshift_px_var = tk.StringVar(value="6")
        self.vshift_step_var = tk.StringVar(value="1")
        self.hshift_px_var = tk.StringVar(value="0")
        self.hshift_step_var = tk.StringVar(value="1")
        self.force_thd_var = tk.StringVar(value="")
        self.anchor_from_end_var = tk.StringVar(value="3")
        self.blank_ignore_var = tk.StringVar(value="1")
        self.source_lang_var = tk.StringVar(value="ja")
        self.target_lang_var = tk.StringVar(value="zh-CN")
        self.vlm_models_var = tk.StringVar(value="qwen3.6-plus")
        self.vlm_model_var = tk.StringVar(value="qwen3.6-plus")
        self.match_score_var = tk.StringVar(value="匹配分数: N/A")
        self.output_name_var = tk.StringVar(value=output_name.strip())
        self._last_auto_output_name = ""
        self._suppress_output_name_trace = False
        self.post_action_make_subtitle_var = tk.BooleanVar(value=False)
        self.skip_translation_var = tk.BooleanVar(value=False)
        self.debug_mode_var = tk.BooleanVar(value=False)
        self.template_center_width: int | None = None
        self.ffmpeg_path = (ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe").resolve()
        self.ffmpeg_hwaccel = "auto"
        self.ffmpeg_two_stage_seek = True
        self.ffmpeg_two_stage_margin_sec = 2.0
        self._seek_after_id: str | None = None
        self._pending_seek_sec: float | None = None
        self._last_fast_seek_frame_idx: int | None = None
        self._suppress_scale_callback = False
        self._decode_overlay_visible = False
        self._decode_spinner_phase = 0
        self._decode_spinner_after_id: str | None = None
        self._decode_spinner_chars = ["|", "/", "-", "\\"]
        self._run_window: tk.Toplevel | None = None
        self._run_log_text: tk.Text | None = None
        self._run_video_var = tk.StringVar(value="")
        self._run_config_var = tk.StringVar(value="")
        self._run_output_var = tk.StringVar(value="")
        self._run_elapsed_var = tk.StringVar(value="00:00")
        self._run_proc: subprocess.Popen[str] | None = None
        self._run_started_ts = 0.0
        self._run_active = False
        self._run_elapsed_after_id: str | None = None
        self._run_last_heartbeat_sec = -1
        self._run_cancel_requested = False

        self.cap: cv2.VideoCapture | None = None
        self.video_w = 0
        self.video_h = 0
        self.fps = 25.0
        self.duration_sec = 0.0
        self.current_sec = 0.0

        self.scale = 1.0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.display_w = 0
        self.display_h = 0
        self.offset_x = 0
        self.offset_y = 0
        self.tk_img: ImageTk.PhotoImage | None = None
        self.frame_rgb_full: Any | None = None

        self.cfg_raw: dict[str, Any] = {}
        self.cfg_merged: dict[str, Any] = {}
        self.rois: dict[str, list[int]] = {}

        self.drag_start: tuple[int, int] | None = None
        self.drag_now: tuple[int, int] | None = None
        self._is_seeking = False

        self.undo_stack: list[dict[str, Any]] = []
        self.redo_stack: list[dict[str, Any]] = []
        self._applying_history = False
        
        self._preview_seek_after_id: str | None = None
        self._last_preview_seek_ts = 0.0
        self._preview_interval_ms = 80
        
        self._precise_req_id = 0
        self._precise_running = False
        self._precise_pending_sec: float | None = None
        self._precise_result_after_id: str | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._bind_live_updates()
        if str(config_path).strip():
            self._load_config_only()
        else:
            self.status_var.set("未加载配置，请先选择或新建 profile。")
        if str(video_path).strip() and Path(video_path).exists():
            self._open_video(video_path)
        self._refresh_canvas()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="视频").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.video_path_var, width=86).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(6, 4)
        )
        ttk.Button(top, text="选择视频", command=self._pick_video).grid(row=0, column=4, padx=2)
        ttk.Button(top, text="加载视频", command=self._reload_video).grid(row=0, column=5, padx=2)
        ttk.Button(top, text="修复视频(ffmpeg)", command=self._repair_video_with_ffmpeg).grid(row=0, column=6, padx=2)

        ttk.Label(top, text="配置").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.config_path_var, width=86).grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=(6, 4)
        )
        ttk.Button(top, text="选择配置", command=self._pick_config).grid(row=1, column=4, padx=2)
        ttk.Button(top, text="加载配置", command=self._load_config_only).grid(row=1, column=5, padx=2)
        ttk.Button(top, text="新建配置", command=self._create_new_profile).grid(row=1, column=6, padx=2)
        ttk.Button(top, text="导入现有配置新建", command=self._create_profile_from_existing).grid(row=1, column=7, padx=2)

        ttk.Label(top, text="ROI 组件").grid(row=2, column=0, sticky="w")
        ttk.OptionMenu(top, self.roi_key_var, self.roi_key_var.get(), *ROI_KEYS).grid(
            row=2, column=1, sticky="w", padx=(6, 4)
        )
        ttk.Button(top, text="保存配置", command=self._save_config).grid(row=2, column=2, padx=2)
        ttk.Button(top, text="撤回", command=self._undo).grid(row=2, column=3, padx=2)
        ttk.Button(top, text="恢复", command=self._redo).grid(row=2, column=4, padx=2)
        ttk.Label(top, text="输出文件夹名").grid(row=2, column=5, sticky="e")
        ttk.Entry(top, textvariable=self.output_name_var, width=24).grid(
            row=2, column=6, columnspan=2, sticky="w", padx=(6, 0)
        )

        ttk.Label(top, text="时间(秒)").grid(row=3, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.time_var, width=12).grid(row=3, column=1, sticky="w", padx=(6, 4))
        ttk.Button(top, text="跳转", command=self._jump_time).grid(row=3, column=2, padx=2)
        ttk.Button(top, text="上一帧", command=lambda: self._step_frame(-1)).grid(row=3, column=3, padx=2)
        ttk.Button(top, text="下一帧", command=lambda: self._step_frame(1)).grid(row=3, column=4, padx=2)
        ttk.Checkbutton(
            top,
            text="直接生成字幕",
            variable=self.post_action_make_subtitle_var,
        ).grid(row=3, column=5, sticky="w")
        ttk.Checkbutton(top, text="Debug", variable=self.debug_mode_var).grid(row=3, column=6, sticky="w")
        ttk.Checkbutton(top, text="跳过翻译", variable=self.skip_translation_var).grid(row=3, column=7, sticky="w")
        ttk.Button(top, text="运行流程", command=self._run_pipeline_from_gui).grid(row=3, column=8, padx=2)

        self.time_scale = ttk.Scale(
            top,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            command=self._on_scale_change,
        )
        self.time_scale.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(6, 0))
        self.time_scale.bind("<ButtonRelease-1>", self._on_scale_release)

        ttk.Label(top, text="模板路径").grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(top, textvariable=self.template_paths_var, width=86).grid(
            row=5, column=1, columnspan=3, sticky="ew", padx=(6, 4), pady=(6, 0)
        )
        ttk.Button(top, text="选择模板", command=self._pick_templates).grid(row=5, column=4, padx=2, pady=(6, 0))
        ttk.Button(top, text="截取 Marker", command=self._capture_marker_template).grid(
            row=5, column=5, padx=2, pady=(6, 0)
        )

        ttk.Label(top, text="当前模板").grid(row=6, column=0, sticky="w", pady=(4, 0))
        self.template_select_combo = ttk.Combobox(
            top,
            textvariable=self.template_selected_var,
            state="readonly",
            width=72,
        )
        self.template_select_combo.grid(row=6, column=1, columnspan=3, sticky="ew", padx=(6, 4), pady=(4, 0))
        ttk.Label(top, textvariable=self.match_score_var, foreground="#1f5f99").grid(
            row=6, column=4, columnspan=2, sticky="w", pady=(4, 0)
        )

        ttk.Label(top, text="位移方向").grid(row=7, column=0, sticky="w", pady=(4, 0))
        ttk.OptionMenu(top, self.shift_mode_var, self.shift_mode_var.get(), "vertical", "horizontal").grid(
            row=7, column=1, sticky="w", padx=(6, 4), pady=(4, 0)
        )
        ttk.Label(top, text="上下位移(px/步长)").grid(row=7, column=2, sticky="e", pady=(4, 0))
        vs = ttk.Frame(top)
        vs.grid(row=7, column=3, sticky="w", pady=(4, 0))
        ttk.Entry(vs, textvariable=self.vshift_px_var, width=8).pack(side=tk.LEFT)
        ttk.Entry(vs, textvariable=self.vshift_step_var, width=8).pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="左右位移(px/步长)").grid(row=8, column=0, sticky="w", pady=(4, 0))
        hs = ttk.Frame(top)
        hs.grid(row=8, column=1, sticky="w", padx=(6, 4), pady=(4, 0))
        ttk.Entry(hs, textvariable=self.hshift_px_var, width=8).pack(side=tk.LEFT)
        ttk.Entry(hs, textvariable=self.hshift_step_var, width=8).pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="强制阈值").grid(row=8, column=2, sticky="e", pady=(4, 0))
        ttk.Entry(top, textvariable=self.force_thd_var, width=12).grid(row=8, column=3, sticky="w", pady=(4, 0))

        ttk.Label(top, text="OCR锚点(距尾帧数)").grid(row=9, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(top, textvariable=self.anchor_from_end_var, width=12).grid(
            row=9, column=1, sticky="w", padx=(6, 4), pady=(4, 0)
        )
        ttk.Label(top, text="blank最小帧数").grid(row=9, column=2, sticky="e", pady=(4, 0))
        ttk.Entry(top, textvariable=self.blank_ignore_var, width=12).grid(row=9, column=3, sticky="w", pady=(4, 0))

        ttk.Label(top, text="源语言").grid(row=10, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(top, textvariable=self.source_lang_var, width=12).grid(
            row=10, column=1, sticky="w", padx=(6, 4), pady=(4, 0)
        )
        ttk.Label(top, text="目标语言").grid(row=10, column=2, sticky="e", pady=(4, 0))
        ttk.Entry(top, textvariable=self.target_lang_var, width=12).grid(row=10, column=3, sticky="w", pady=(4, 0))
        ttk.Label(top, text="VLM模型(逗号分隔)").grid(row=11, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(top, textvariable=self.vlm_models_var, width=86).grid(
            row=11, column=1, columnspan=5, sticky="ew", padx=(6, 4), pady=(4, 0)
        )
        ttk.Label(top, text="当前VLM模型").grid(row=12, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(top, textvariable=self.vlm_model_var, width=40).grid(
            row=12, column=1, columnspan=2, sticky="w", padx=(6, 4), pady=(4, 0)
        )

        top.columnconfigure(1, weight=1)

        info = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        info.pack(fill=tk.X)
        ttk.Label(
            info,
            text=(
                "用法：先选择 ROI 组件，再在画面上按住左键拖动框选。"
                "当前帧匹配分数会自动刷新。"
            ),
        ).pack(anchor="w")
        ttk.Label(info, textvariable=self.status_var, foreground="#1f5f99").pack(anchor="w")

        canvas_frame = ttk.Frame(self.root, padding=8)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="#111111", highlightthickness=1, highlightbackground="#444")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.root.bind("<Control-z>", lambda _e: self._undo())
        self.root.bind("<Control-y>", lambda _e: self._redo())

        profile_box = ttk.LabelFrame(self.root, text="Profile YAML 预览/编辑", padding=8)
        profile_box.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0, 8))
        pb = ttk.Frame(profile_box)
        pb.pack(fill=tk.X)
        ttk.Button(pb, text="从文件加载到编辑框", command=self._load_profile_text_from_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(pb, text="保存编辑框到文件", command=self._save_profile_text_to_file).pack(side=tk.LEFT, padx=2)
        self.profile_text = tk.Text(profile_box, height=10, wrap=tk.NONE, undo=True, font=("Consolas", 10))
        self.profile_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    def _bind_live_updates(self) -> None:
        self.template_paths_var.trace_add("write", lambda *_: self._on_template_paths_changed())
        for var in [
            self.template_selected_var,
            self.shift_mode_var,
            self.vshift_px_var,
            self.vshift_step_var,
            self.hshift_px_var,
            self.hshift_step_var,
        ]:
            var.trace_add("write", lambda *_: self._update_match_score())
        self.vlm_models_var.trace_add("write", lambda *_: self._sync_current_model_from_list())
        self.output_name_var.trace_add("write", lambda *_: self._on_output_name_changed())

    def _pick_video(self) -> None:
        p = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频文件", "*.mp4 *.mkv *.mov *.avi"), ("所有文件", "*.*")],
        )
        if p:
            vp = Path(p).resolve()
            self.video_path_var.set(vp.as_posix())
            self._ensure_output_name_default(vp)
            self._apply_video_to_current_profile(vp)

    def _pick_config(self) -> None:
        p = filedialog.askopenfilename(
            title="选择配置",
            filetypes=[("YAML", "*.yaml *.yml"), ("JSON", "*.json"), ("所有文件", "*.*")],
        )
        if p:
            self.config_path_var.set(p)
            
    def _apply_video_to_current_profile(self, video_path: Path) -> None:
        raw_cfg = self.config_path_var.get().strip()
        if not raw_cfg:
            return
        cfg_path = Path(raw_cfg).resolve()
        if not cfg_path.exists():
            return

        video_cfg = self._to_cfg_path(video_path.resolve())

        data = _load_raw_cfg(cfg_path)
        if not isinstance(data, dict):
            data = {}
        data["video_path"] = video_cfg
        _save_raw_cfg(cfg_path, data)

        self.cfg_raw = data
        if isinstance(self.cfg_merged, dict):
            self.cfg_merged["video_path"] = video_cfg

        self._load_profile_text_from_file()

    def _default_profile_filename(self) -> str:
        raw_video = self.video_path_var.get().strip()
        if raw_video:
            stem = Path(raw_video).stem.strip()
            if stem:
                stem = re.sub(r'[<>:"/\\\\|?*]+', "_", stem)
                return f"{stem}.yaml"
        return "new_game_profile.yaml"

    def _empty_profile_payload(self) -> dict[str, Any]:
        return {
            "extends": "general_config.yaml",
            "game": {
                "name": "",
                "source_language": "ja",
                "target_language": "zh-CN",
            },
            "output_name": "",
            "video_path": "",
            "roi": {},
            "marker": {
                "ocr_anchor_from_end_frames": 3,
            },
            "general": {
                "blank_ignore_under_frames": 1,
            },
        }

    def _create_new_profile(self) -> None:
        default_dir = (ROOT / "config").resolve()
        default_dir.mkdir(parents=True, exist_ok=True)
        save_path = filedialog.asksaveasfilename(
            title="新建配置",
            initialdir=str(default_dir),
            initialfile=self._default_profile_filename(),
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not save_path:
            return

        out = Path(save_path).resolve()
        payload = self._empty_profile_payload()

        raw_video = self.video_path_var.get().strip()
        if raw_video:
            payload["video_path"] = self._to_cfg_path(Path(raw_video).resolve())

        _save_raw_cfg(out, payload)
        self.config_path_var.set(str(out.as_posix()))
        self.status_var.set(f"已新建配置: {out.as_posix()}")
        self._load_config_only()

    def _create_profile_from_existing(self) -> None:
        default_dir = (ROOT / "config").resolve()
        default_dir.mkdir(parents=True, exist_ok=True)
        src_path = filedialog.askopenfilename(
            title="选择要导入的现有配置",
            initialdir=str(default_dir),
            filetypes=[("YAML", "*.yaml *.yml"), ("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not src_path:
            return
        try:
            payload = _load_raw_cfg(Path(src_path).resolve())
            if not isinstance(payload, dict):
                raise RuntimeError("配置格式错误")
        except Exception as exc:
            self.status_var.set(f"导入配置失败: {exc}")
            return
        save_path = filedialog.asksaveasfilename(
            title="另存为新的配置",
            initialdir=str(default_dir),
            initialfile=self._default_profile_filename(),
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not save_path:
            return

        raw_video = self.video_path_var.get().strip()
        if raw_video:
            payload["video_path"] = self._to_cfg_path(Path(raw_video).resolve())

        out = Path(save_path).resolve()
        _save_raw_cfg(out, payload)
        self.config_path_var.set(str(out.as_posix()))
        self.status_var.set(f"已导入并新建配置: {out.as_posix()}")
        self._load_config_only()

    def _load_profile_text_from_file(self) -> None:
        raw_cfg = self.config_path_var.get().strip()
        if not raw_cfg:
            self.status_var.set("未选择配置文件。")
            return
        cfg_path = Path(raw_cfg).resolve()
        if not cfg_path.exists():
            self.status_var.set(f"配置不存在: {cfg_path.as_posix()}")
            return
        try:
            text = cfg_path.read_text(encoding="utf-8")
        except Exception as exc:
            self.status_var.set(f"读取配置失败: {exc}")
            return
        self.profile_text.delete("1.0", tk.END)
        self.profile_text.insert("1.0", text)

    def _save_profile_text_to_file(self) -> None:
        raw_cfg = self.config_path_var.get().strip()
        if not raw_cfg:
            self.status_var.set("未选择配置文件。")
            return
        cfg_path = Path(raw_cfg).resolve()
        text = self.profile_text.get("1.0", tk.END)
        try:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(text, encoding="utf-8")
        except Exception as exc:
            self.status_var.set(f"保存配置失败: {exc}")
            return
        self.status_var.set(f"配置已保存: {cfg_path.as_posix()}")
        self._load_config_only()

    def _parse_model_list(self, text: str) -> list[str]:
        out: list[str] = []
        raw = str(text or "").strip()
        if not raw:
            return out
        for p in raw.replace("\n", ",").replace(";", ",").split(","):
            x = p.strip()
            if x:
                out.append(x)
        return out

    def _sync_current_model_from_list(self) -> None:
        models = self._parse_model_list(self.vlm_models_var.get())
        cur = self.vlm_model_var.get().strip()
        if not models:
            self.vlm_model_var.set("")
            return
        if not cur or cur not in models:
            self.vlm_model_var.set(models[0])

    def _load_config_only(self) -> None:
        raw_cfg = self.config_path_var.get().strip()
        if not raw_cfg:
            self.status_var.set("未选择配置，请先选择或新建")
            return
        cfg_path = Path(raw_cfg).resolve()
        if not cfg_path.exists():
            self.status_var.set(f"配置不存在: {cfg_path.as_posix()}")
            return
        try:
            self.cfg_raw = _load_raw_cfg(cfg_path)
            self.cfg_merged = load_config(cfg_path)
        except Exception as exc:
            self.status_var.set(f"加载配置失败: {exc}")
            return

        roi_cfg = self.cfg_merged.get("roi", {})
        self.rois = {}
        legacy_alias = {
            "subtitle_location": "subtitle_roi",
            "title_translation_location": "title_text_roi",
            "title_info_location": "title_info_roi",
            "title_ocr_roi": "title_roi",
        }
        for k in ROI_KEYS:
            v = roi_cfg.get(k)
            if v is None and k in legacy_alias:
                v = roi_cfg.get(legacy_alias[k])
            if v is None and k == "title_ocr_roi":
                v = roi_cfg.get("dialogue_roi")
            if v is None and k == "title_translation_location":
                v = roi_cfg.get("title_subtitle_roi")
            if v is None and k == "title_info_location":
                v = roi_cfg.get("title_speaker_roi")
            if isinstance(v, list) and len(v) == 4:
                self.rois[k] = [int(v[0]), int(v[1]), int(v[2]), int(v[3])]

        marker_cfg = self.cfg_merged.get("marker", {})
        tpl_list = [str(x) for x in (marker_cfg.get("template_paths") or [])] if isinstance(marker_cfg.get("template_paths"), list) else []
        self.template_paths_var.set(";".join(tpl_list))
        self.shift_mode_var.set("horizontal" if str(marker_cfg.get("shift_mode", "vertical")).lower() == "horizontal" else "vertical")
        self.vshift_px_var.set(str(int(marker_cfg.get("vertical_shift_px", 6))))
        self.vshift_step_var.set(str(int(marker_cfg.get("vertical_shift_step", 1))))
        self.hshift_px_var.set(str(int(marker_cfg.get("horizontal_shift_px", 0))))
        self.hshift_step_var.set(str(int(marker_cfg.get("horizontal_shift_step", 1))))
        self.template_center_width = int(marker_cfg.get("template_center_width")) if marker_cfg.get("template_center_width") is not None else None
        force_thd = marker_cfg.get("force_threshold", None)
        self.force_thd_var.set("" if force_thd is None else str(force_thd))
        self.anchor_from_end_var.set(
            str(max(1, self._int_or_default(str(marker_cfg.get("ocr_anchor_from_end_frames", 3)), 3)))
        )
        general_cfg = self.cfg_merged.get("general", {})
        self.blank_ignore_var.set(
            str(max(0, self._int_or_default(str(general_cfg.get("blank_ignore_under_frames", 1)), 1)))
        )
        game_cfg = self.cfg_merged.get("game", {})
        self.source_lang_var.set(str(game_cfg.get("source_language", "ja")))
        self.target_lang_var.set(str(game_cfg.get("target_language", "zh-CN")))
        tr_cfg = self.cfg_merged.get("translation", {})
        models = tr_cfg.get("vlm_models", [])
        if isinstance(models, list):
            model_text = ", ".join([str(x).strip() for x in models if str(x).strip()])
        else:
            model_text = str(tr_cfg.get("model", "qwen3.6-plus"))
        self.vlm_models_var.set(model_text or "qwen3.6-plus")
        self.vlm_model_var.set(str(tr_cfg.get("model", "") or ""))
        self._sync_current_model_from_list()
        cfg_out_name = str(self.cfg_merged.get("output_name", "") or "").strip()
        if cfg_out_name:
            self._set_output_name_value(cfg_out_name, is_auto=False)
        tools_cfg = self.cfg_merged.get("tools", {})
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

        cfg_video = str(self.cfg_merged.get("video_path", "") or "").strip()
        if cfg_video:
            vv = Path(cfg_video)
            if not vv.is_absolute():
                vv = (ROOT / vv).resolve()
            self.video_path_var.set(vv.as_posix())
            self._ensure_output_name_default(vv)
            if vv.exists():
                self._open_video(vv.as_posix())
        else:
            self._ensure_output_name_default()

        self._refresh_template_selector()
        self._reset_history()
        self.status_var.set(f"配置已加载: {cfg_path.as_posix()}")
        self._load_profile_text_from_file()
        self._refresh_canvas()
        self._update_match_score()

    def _reload_video(self) -> None:
        raw = self.video_path_var.get().strip()
        if not raw:
            self.status_var.set("未选择视频，请先在界面选择。")
            return
        self._open_video(raw)

    def _ask_repair_video_mode(self) -> str | None:
        """Return one of: 'replace', 'saveas', or None (cancel)."""
        win = tk.Toplevel(self.root)
        win.title("修复视频")
        win.geometry("520x210")
        win.resizable(False, False)
        win.transient(self.root)

        choice: dict[str, str | None] = {"mode": None}

        body = ttk.Frame(win, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            body,
            text="请选择输出方式：",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            body,
            text="替换原视频会自动创建备份文件。",
        ).pack(anchor="w", pady=(6, 12))

        btns = ttk.Frame(body)
        btns.pack(fill=tk.X, pady=(4, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        btns.columnconfigure(2, weight=1)

        def _pick(mode: str | None) -> None:
            choice["mode"] = mode
            try:
                win.destroy()
            except Exception:
                pass

        ttk.Button(btns, text="替换原视频", width=14, command=lambda: _pick("replace")).grid(
            row=0, column=0, padx=10, pady=2
        )
        ttk.Button(btns, text="另存为", width=14, command=lambda: _pick("saveas")).grid(
            row=0, column=1, padx=10, pady=2
        )
        ttk.Button(btns, text="取消", width=14, command=lambda: _pick(None)).grid(
            row=0, column=2, padx=10, pady=2
        )

        win.protocol("WM_DELETE_WINDOW", lambda: _pick(None))
        win.bind("<Escape>", lambda _e: _pick(None))
        self.root.update_idletasks()
        win.update_idletasks()
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        win_w = win.winfo_width()
        win_h = win.winfo_height()
        pos_x = root_x + max(0, (root_w - win_w) // 2)
        pos_y = root_y + max(0, (root_h - win_h) // 2)
        win.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        try:
            win.grab_set()
        except Exception:
            pass
        win.focus_set()
        self.root.wait_window(win)
        return choice["mode"]

    def _repair_video_with_ffmpeg(self) -> None:
        raw = self.video_path_var.get().strip()
        if not raw:
            self.status_var.set("未选择视频，请先在界面选择。")
            return
        src = Path(raw).resolve()
        if not src.exists():
            self.status_var.set(f"视频不存在: {src}")
            return
        if not self.ffmpeg_path.exists():
            self.status_var.set(f"ffmpeg 不存在: {self.ffmpeg_path}")
            messagebox.showerror("修复失败", f"ffmpeg 不存在:\n{self.ffmpeg_path}", parent=self.root)
            return

        mode = self._ask_repair_video_mode()
        if mode is None:
            return

        replace_original = mode == "replace"
        if replace_original:
            out = src.with_name(f"{src.stem}.repair_tmp{src.suffix}")
        else:
            save_path = filedialog.asksaveasfilename(
                title="另存为修复后视频",
                initialdir=str(src.parent),
                initialfile=f"{src.stem}_repaired{src.suffix}",
                defaultextension=src.suffix or ".mp4",
                filetypes=[("视频文件", "*.mp4 *.mkv *.mov *.avi"), ("所有文件", "*.*")],
            )
            if not save_path:
                return
            out = Path(save_path).resolve()
            if out == src:
                self.status_var.set("另存为路径不能与原视频相同，请选择替换原视频模式。")
                return

        self.status_var.set("视频修复中...（后台执行）")

        def _run() -> None:
            try:
                ok, detail = self._run_ffmpeg_repair_pipeline(src, out)
                if not ok:
                    raise RuntimeError(detail)

                final_path = out
                backup_path: Path | None = None
                if replace_original:
                    backup_path = src.with_name(
                        f"{src.stem}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}{src.suffix}"
                    )
                    src.replace(backup_path)
                    out.replace(src)
                    final_path = src

                def _on_ok() -> None:
                    self.video_path_var.set(final_path.as_posix())
                    self._ensure_output_name_default(final_path)
                    self._open_video(final_path.as_posix())
                    if backup_path is not None:
                        self.status_var.set(
                            f"视频修复完成并替换原文件。备份: {backup_path.as_posix()}"
                        )
                    else:
                        self.status_var.set(f"视频修复完成: {final_path.as_posix()}")

                self.root.after(0, _on_ok)
            except Exception as exc:
                try:
                    if replace_original and out.exists():
                        out.unlink()
                except Exception:
                    pass
                err_msg = str(exc)

                def _on_err(msg: str = err_msg) -> None:
                    self.status_var.set(f"视频修复失败: {msg}")
                    messagebox.showerror("修复失败", msg, parent=self.root)

                self.root.after(0, _on_err)

        threading.Thread(target=_run, daemon=True).start()

    def _run_ffmpeg_repair_pipeline(self, src: Path, out: Path) -> tuple[bool, str]:
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd_remux = [
            str(self.ffmpeg_path),
            "-y",
            "-v",
            "warning",
            "-err_detect",
            "ignore_err",
            "-fflags",
            "+genpts+discardcorrupt",
            "-i",
            str(src),
            "-map",
            "0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(out),
        ]
        ok, detail = self._run_ffmpeg_cmd(cmd_remux)
        if ok:
            return True, "remux"

        cmd_transcode = [
            str(self.ffmpeg_path),
            "-y",
            "-v",
            "warning",
            "-err_detect",
            "ignore_err",
            "-fflags",
            "+genpts+discardcorrupt",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(out),
        ]
        ok2, detail2 = self._run_ffmpeg_cmd(cmd_transcode)
        if ok2:
            return True, "transcode"
        return False, f"remux失败: {detail}\ntranscode失败: {detail2}"

    def _run_ffmpeg_cmd(self, cmd: list[str]) -> tuple[bool, str]:
        try:
            p = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            out = (p.stdout or "").strip()
            if p.returncode == 0:
                return True, "ok"
            tail = "\n".join(out.splitlines()[-20:])
            return False, f"exit={p.returncode}\n{tail}"
        except Exception as exc:
            return False, str(exc)

    def _sanitize_output_name(self, name: str) -> str:
        s = str(name or "").strip()
        s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
        s = re.sub(r"\s+", "_", s)
        return s.strip("._ ")

    def _set_output_name_value(self, value: str, *, is_auto: bool = False) -> None:
        clean = self._sanitize_output_name(value)
        self._suppress_output_name_trace = True
        try:
            self.output_name_var.set(clean)
        finally:
            self._suppress_output_name_trace = False
        if is_auto:
            self._last_auto_output_name = clean

    def _on_output_name_changed(self) -> None:
        if self._suppress_output_name_trace:
            return
        clean = self._sanitize_output_name(self.output_name_var.get())
        if clean != self.output_name_var.get():
            self._set_output_name_value(clean, is_auto=False)

    def _ensure_output_name_default(self, video_path: Path | None = None) -> None:
        cur = self._sanitize_output_name(self.output_name_var.get())
        vp = video_path
        if vp is None:
            raw = self.video_path_var.get().strip()
            if raw:
                vp = Path(raw)

        if cur and cur != self._last_auto_output_name:
            self._set_output_name_value(cur, is_auto=False)
            return

        if vp is not None:
            stem = self._sanitize_output_name(vp.stem)
            if stem:
                self._set_output_name_value(stem, is_auto=True)
                return

        fallback = cur or "run_output"
        self._set_output_name_value(fallback, is_auto=(not cur or cur == self._last_auto_output_name))

    def _resolve_output_dir(self) -> Path:
        self._ensure_output_name_default()
        out_name = self._sanitize_output_name(self.output_name_var.get())
        if not out_name:
            out_name = "run_output"
        self._set_output_name_value(out_name, is_auto=(out_name == self._last_auto_output_name))
        return (ROOT / "outputs" / out_name).resolve()

    def _validate_before_run(self) -> list[str]:
        errs: list[str] = []
        raw_video = self.video_path_var.get().strip()
        raw_cfg = self.config_path_var.get().strip()
        cfg_for_run: dict[str, Any] | None = None
        if not raw_video:
            errs.append("未选择视频路径。")
        elif not Path(raw_video).resolve().exists():
            errs.append(f"视频不存在：{Path(raw_video).resolve()}")
        if not raw_cfg:
            errs.append("未选择配置文件。")
        elif not Path(raw_cfg).resolve().exists():
            errs.append(f"配置不存在：{Path(raw_cfg).resolve()}")
        else:
            try:
                cfg_for_run = load_config(Path(raw_cfg).resolve())
            except Exception as exc:
                errs.append(f"配置加载失败：{exc}")

        required_rois = ["name_roi", "dialogue_roi", "marker_roi"]
        for key in required_rois:
            rect = self.rois.get(key)
            if not rect or len(rect) != 4:
                errs.append(f"缺少必要区域：{key}")
                continue
            x0, y0, x1, y1 = rect
            if int(x1) - int(x0) < 2 or int(y1) - int(y0) < 2:
                errs.append(f"区域无效（过小）：{key}")

        self._ensure_output_name_default()
        if not self._sanitize_output_name(self.output_name_var.get()):
            errs.append("输出文件夹名无效。")

        # Only require online translation config when translation is enabled.
        if not bool(self.skip_translation_var.get()) and cfg_for_run is not None:
            tr_cfg = cfg_for_run.get("translation", {})
            if not isinstance(tr_cfg, dict):
                tr_cfg = {}

            vlm_api = str(tr_cfg.get("vlm_api", "") or "").strip()
            if not vlm_api:
                errs.append(
                    "未勾选“跳过翻译”时必须配置 translation.vlm_api（请在 config/general_config.yaml 中填写）。"
                )

            api_key_inline = str(tr_cfg.get("api_key", "") or "").strip()
            api_key_file = str(tr_cfg.get("api_key_file", "") or "").strip()
            if not api_key_inline and not api_key_file:
                errs.append(
                    "未勾选“跳过翻译”时必须配置 translation.api_key 或 translation.api_key_file。"
                )
            elif (not api_key_inline) and api_key_file:
                key_path = Path(api_key_file)
                if not key_path.is_absolute():
                    key_path = (ROOT / key_path).resolve()
                if not key_path.exists() or not key_path.is_file():
                    errs.append(
                        f"api_key_file 不存在：{key_path}（请检查 translation.api_key_file）。"
                    )
        return errs

    def _append_run_log(self, text: str) -> None:
        if self._run_log_text is None:
            return
        self._run_log_text.insert(tk.END, text)
        self._run_log_text.see(tk.END)

    def _set_run_active(self, active: bool) -> None:
        self._run_active = bool(active)
        if not self._run_active:
            if self._run_elapsed_after_id is not None:
                try:
                    self.root.after_cancel(self._run_elapsed_after_id)
                except Exception:
                    pass
                self._run_elapsed_after_id = None
            self._run_proc = None

    def _tick_run_elapsed(self) -> None:
        if self._run_started_ts <= 0 or not self._run_active:
            return
        elapsed = int(max(0.0, time.time() - self._run_started_ts))
        mm = elapsed // 60
        ss = elapsed % 60
        self._run_elapsed_var.set(f"{mm:02d}:{ss:02d}")
        if elapsed > 0 and elapsed % 15 == 0 and elapsed != self._run_last_heartbeat_sec:
            self._run_last_heartbeat_sec = elapsed
            # self._append_run_log(f"[INFO] 仍在运行中... 已耗时 {mm:02d}:{ss:02d}\n")
        self._run_elapsed_after_id = self.root.after(500, self._tick_run_elapsed)

    def _open_run_window(self, video: Path, config: Path, output_dir: Path) -> None:
        if self._run_window is not None and self._run_window.winfo_exists():
            try:
                self._run_window.destroy()
            except Exception:
                pass
        win = tk.Toplevel(self.root)
        win.title("运行监控")
        win.geometry("1180x760")
        win.protocol("WM_DELETE_WINDOW", self._on_run_window_close)
        self._run_window = win
        self._run_video_var.set(str(video))
        self._run_config_var.set(str(config))
        self._run_output_var.set(str(output_dir))
        self._run_elapsed_var.set("00:00")

        top = ttk.Frame(win, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="视频：").grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self._run_video_var).grid(row=0, column=1, sticky="w")
        ttk.Label(top, text="配置：").grid(row=1, column=0, sticky="w")
        ttk.Label(top, textvariable=self._run_config_var).grid(row=1, column=1, sticky="w")
        ttk.Label(top, text="输出：").grid(row=2, column=0, sticky="w")
        ttk.Label(top, textvariable=self._run_output_var).grid(row=2, column=1, sticky="w")
        ttk.Label(top, text="运行时长：").grid(row=3, column=0, sticky="w")
        ttk.Label(top, textvariable=self._run_elapsed_var).grid(row=3, column=1, sticky="w")
        top.columnconfigure(1, weight=1)

        log_box = ttk.Frame(win, padding=(8, 0, 8, 8))
        log_box.pack(fill=tk.BOTH, expand=True)
        txt = tk.Text(log_box, wrap=tk.WORD, font=("Consolas", 10))
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log_box, orient=tk.VERTICAL, command=txt.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt.configure(yscrollcommand=sb.set)
        self._run_log_text = txt

    def _terminate_run_process(self) -> None:
        proc = self._run_proc
        if proc is None:
            return
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
            return
        except Exception:
            pass
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                proc.kill()
        except Exception:
            pass

    def _on_run_window_close(self) -> None:
        self._run_cancel_requested = True
        self._append_run_log("[INFO] 用户关闭运行监控窗口，正在停止任务...\n")
        self._terminate_run_process()
        try:
            if self._run_window is not None and self._run_window.winfo_exists():
                self._run_window.destroy()
        except Exception:
            pass
        self._run_window = None
        self._run_log_text = None
        self.status_var.set("任务已停止。")

    def _find_latest_translation_cache(self, output_dir: Path) -> Path | None:
        work_root = output_dir / "work"
        if not work_root.exists():
            return None
        runs = [p for p in work_root.glob("run_*") if p.is_dir()]
        if not runs:
            return None
        latest = max(runs, key=lambda p: p.stat().st_mtime)
        cache = latest / "translation_cache.json"
        return cache if cache.exists() else None

    def _run_subprocess_stream(self, cmd: list[str], run_log_file: Path) -> int:
        run_log_file.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self.root.after(0, lambda: self._append_run_log(f"$ {' '.join(cmd)}\n"))
        with run_log_file.open("a", encoding="utf-8") as wf:
            wf.write(f"$ {' '.join(cmd)}\n")
            wf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            self._run_proc = proc
            self.root.after(0, self._tick_run_elapsed)
            assert proc.stdout is not None
            for line in proc.stdout:
                wf.write(line)
                wf.flush()
                self.root.after(0, lambda s=line: self._append_run_log(s))
            rc = int(proc.wait())
            self._run_proc = None
            return rc

    def _reset_run_log_file(self, run_log_file: Path) -> None:
        """Truncate GUI output run.log before a new run starts."""
        run_log_file.parent.mkdir(parents=True, exist_ok=True)
        run_log_file.write_text("", encoding="utf-8")

    def _launch_cache_review(self, cache_path: Path, video: Path, config: Path) -> None:
        cmd = [
            sys.executable,
            str((ROOT / "app" / "cache_browser_review.py").resolve()),
            "--cache",
            str(cache_path),
            "--video",
            str(video),
            "--config",
            str(config),
        ]
        self._append_run_log(f"[INFO] Launch cache review: {' '.join(cmd)}\n")
        try:
            subprocess.Popen(cmd, cwd=str(ROOT))
        except Exception as exc:
            self._append_run_log(f"[ERROR] Failed to launch cache review: {exc}\n")
            self.status_var.set(f"打开 Review 失败: {exc}")

    def _build_pipeline_cmd(self, video: Path, config: Path, output_dir: Path, extra_args: list[str]) -> list[str]:
        # Use module mode so package-relative imports in pipeline keep working.
        base = [
            sys.executable,
            "-m",
            "src.auto_gamevideo_subtitles.pipeline",
            "--video",
            str(video),
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
        ]
        cmd = base + list(extra_args)
        if bool(self.skip_translation_var.get()):
            cmd.append("--skip-translation")
        if bool(self.debug_mode_var.get()):
            cmd.append("--debug")
        return cmd

    def _open_video(self, video_path: str) -> None:
        p = Path(video_path).resolve()
        if not p.exists():
            self.status_var.set(f"视频不存在: {p}")
            return
        self._last_fast_seek_frame_idx = None
        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            self.status_var.set(f"打开视频失败: {p}")
            return
        self.cap = cap
        self.video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        self.duration_sec = (frame_count / self.fps) if (self.fps > 0 and frame_count > 0) else 0.0
        self.time_scale.configure(to=max(0.0, self.duration_sec))
        self.current_sec = 0.0
        self.time_scale.set(0.0)
        self.time_var.set("0.00")
        self.status_var.set(
            f"视频已加载: {p.name} ({self.video_w}x{self.video_h}, fps={self.fps:.3f}, 时长={self.duration_sec:.2f}s)"
        )
        self._ensure_output_name_default(p)
        self._seek(0.0)

    def _on_close(self) -> None:
        try:
            if self._decode_spinner_after_id is not None:
                try:
                    self.root.after_cancel(self._decode_spinner_after_id)
                except Exception:
                    pass
                self._decode_spinner_after_id = None
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()

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
        # Decode-friendly preroll: seek a bit earlier, then read forward.
        preroll_frames = max(24, min(360, int(round(fps * 3.0))))
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
        # Fallback path.
        self.cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        return self.cap.read()

    def _read_frame_ffmpeg_once(self, sec: float) -> tuple[bool, Any | None]:
        video_raw = self.video_path_var.get().strip()
        if not video_raw or not self.ffmpeg_path.exists():
            return False, None
        video_path = Path(video_raw).resolve()
        if not video_path.exists():
            return False, None
        cmd_primary = self._build_ffmpeg_single_frame_cmd(
            video_path=video_path,
            sec=float(sec),
            two_stage_seek=bool(self.ffmpeg_two_stage_seek),
        )
        if self.ffmpeg_two_stage_seek:
            cmd_fallback = self._build_ffmpeg_single_frame_cmd(
                video_path=video_path,
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
        video_path: Path,
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
                    str(video_path),
                    "-ss",
                    f"{post_seek_sec:.6f}",
                ]
            )
        else:
            cmd.extend(
                [
                    "-i",
                    str(video_path),
                    "-ss",
                    f"{target_sec:.6f}",
                ]
            )
        cmd.extend(
            [
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
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
            arr = np.frombuffer(proc.stdout, dtype=np.uint8)
            if arr.size == 0:
                return False, None
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return False, None
            return True, frame
        except Exception:
            return False, None

    def _snap_to_frame_time(self, sec: float) -> float:
        fps = max(1.0, float(self.fps or 25.0))
        frame_idx = int(round(max(0.0, float(sec)) * fps))
        snapped = frame_idx / fps
        return max(0.0, min(snapped, max(0.0, self.duration_sec)))

    def _seek(self, sec: float, update_scale: bool = True, prefer_fast: bool = False, update_match_score: bool = True,) -> None:
        if self._is_seeking or self.cap is None:
            return
        self._is_seeking = True
        try:
            sec = self._snap_to_frame_time(float(sec))
            fps = max(1.0, float(self.fps or 25.0))
            frame_idx = int(round(sec * fps))
            if (
                prefer_fast
                and self._last_fast_seek_frame_idx == frame_idx
                and self.frame_rgb_full is not None
            ):
                self.current_sec = sec
                self.time_var.set(f"{sec:.2f}")
                return
            self.current_sec = sec
            ok, frame = self._read_frame_safe(sec, prefer_fast=prefer_fast)
            if not ok or frame is None:
                self.status_var.set(f"无法读取该时间点帧: {sec:.2f}s")
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_rgb_full = frame.copy()
            if prefer_fast:
                self._last_fast_seek_frame_idx = frame_idx
            self._set_frame(frame)
            self.time_var.set(f"{sec:.2f}")
            if update_scale:
                self._suppress_scale_callback = True
                try:
                    self.time_scale.set(sec)
                except tk.TclError:
                    pass
                finally:
                    self._suppress_scale_callback = False
            self._refresh_canvas()
            if update_match_score:
                self._update_match_score()
        finally:
            self._is_seeking = False

    def _set_frame(self, frame_rgb: Any) -> None:
        h, w = frame_rgb.shape[:2]
        canvas_w = max(1, int(self.canvas.winfo_width()))
        canvas_h = max(1, int(self.canvas.winfo_height()))
        if canvas_w <= 1:
            canvas_w = 1360
        if canvas_h <= 1:
            canvas_h = 760

        scale = canvas_w / float(max(1, w))
        self.display_w = canvas_w
        self.display_h = max(1, int(round(h * scale)))
        if self.display_h > canvas_h and canvas_h > 1:
            scale = canvas_h / float(max(1, h))
            self.display_h = canvas_h
            self.display_w = max(1, int(round(w * scale)))

        self.scale = self.display_w / float(max(1, w))
        self.scale_x = self.scale
        self.scale_y = self.scale
        self.offset_x = max(0, (canvas_w - self.display_w) // 2)
        self.offset_y = max(0, (canvas_h - self.display_h) // 2)
        resized = cv2.resize(frame_rgb, (self.display_w, self.display_h), interpolation=cv2.INTER_LINEAR)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(resized.copy()))

    def _on_scale_change(self, value: str) -> None:
        if self._suppress_scale_callback:
            return
        try:
            sec = self._snap_to_frame_time(float(value))
        except Exception:
            return

        self._pending_seek_sec = sec
        self.time_var.set(f"{sec:.2f}")
        self._suppress_scale_callback = True
        try:
            self.time_scale.set(sec)
        except tk.TclError:
            pass
        finally:
            self._suppress_scale_callback = False

        now = time.time()
        elapsed_ms = (now - self._last_preview_seek_ts) * 1000.0

        if elapsed_ms >= self._preview_interval_ms:
            self._last_preview_seek_ts = now
            self._seek(sec, update_scale=False, prefer_fast=True, update_match_score=False)
        else:
            if self._preview_seek_after_id is not None:
                try:
                    self.root.after_cancel(self._preview_seek_after_id)
                except Exception:
                    pass
            delay = max(1, int(self._preview_interval_ms - elapsed_ms))
            self._preview_seek_after_id = self.root.after(delay, self._flush_preview_seek)

        if self._seek_after_id is not None:
            try:
                self.root.after_cancel(self._seek_after_id)
            except Exception:
                pass
        self._seek_after_id = self.root.after(180, self._flush_precise_seek)
        
    def _flush_preview_seek(self) -> None:
        self._preview_seek_after_id = None
        sec = self._pending_seek_sec
        if sec is None:
            return
        self._last_preview_seek_ts = time.time()
        self._seek(float(sec), update_scale=False, prefer_fast=True, update_match_score=True)

    def _on_scale_release(self, _e: tk.Event[Any]) -> None:
        if self._preview_seek_after_id is not None:
            try:
                self.root.after_cancel(self._preview_seek_after_id)
            except Exception:
                pass
            self._preview_seek_after_id = None
        self._flush_precise_seek()

    def _cancel_pending_seek(self) -> None:
        self._pending_seek_sec = None
        if self._seek_after_id is not None:
            try:
                self.root.after_cancel(self._seek_after_id)
            except Exception:
                pass
            self._seek_after_id = None
            
    def _start_precise_seek_async(self, sec: float) -> None:
        self._precise_req_id += 1
        req_id = self._precise_req_id
        self._precise_pending_sec = self._snap_to_frame_time(float(sec))
        self._set_decode_overlay(True)

        if self._precise_running:
            return

        self._precise_running = True

        def _worker() -> None:
            while True:
                target = self._precise_pending_sec
                self._precise_pending_sec = None
                if target is None:
                    break

                target = self._snap_to_frame_time(float(target))

                ok, frame = self._read_frame_ffmpeg_once(float(target))
                current_req_id = self._precise_req_id

                def _apply() -> None:
                    if current_req_id != self._precise_req_id:
                        return
                    self._set_decode_overlay(False)
                    if not ok or frame is None:
                        return

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.current_sec = float(target)
                    self.frame_rgb_full = rgb.copy()
                    self._set_frame(rgb)
                    self._refresh_canvas()
                    self._update_match_score()

                self.root.after(0, _apply)

                # 如果在处理过程中又来了新请求，继续下一轮；否则退出
                if self._precise_pending_sec is None:
                    break

            def _done() -> None:
                self._precise_running = False
                if self._precise_pending_sec is not None:
                    self._start_precise_seek_async(self._precise_pending_sec)
                else:
                    self._set_decode_overlay(False)

            self.root.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def _flush_precise_seek(self) -> None:
        self._seek_after_id = None
        sec = self._pending_seek_sec
        if sec is None:
            return
        self._pending_seek_sec = None
        self._start_precise_seek_async(float(sec))

    def _on_canvas_configure(self, _e: tk.Event[Any]) -> None:
        if self.frame_rgb_full is not None:
            self._set_frame(self.frame_rgb_full)
            self._refresh_canvas()

    def _jump_time(self) -> None:
        try:
            sec = float(self.time_var.get().strip())
        except Exception:
            self.status_var.set(f"时间格式错误: {self.time_var.get()}")
            return
        sec = self._snap_to_frame_time(sec)
        self.time_var.set(f"{sec:.2f}")
        self._cancel_pending_seek()
        self._seek(sec)

    def _step_frame(self, delta: int) -> None:
        if self.fps <= 0:
            return
        self._cancel_pending_seek()
        self._seek(self.current_sec + float(delta) / self.fps)

    def _canvas_to_src(self, x: int, y: int) -> tuple[int, int]:
        if self.scale_x <= 0 or self.scale_y <= 0:
            return 0, 0
        rx = int(x) - int(self.offset_x)
        ry = int(y) - int(self.offset_y)
        sx = int(round(rx / self.scale_x))
        sy = int(round(ry / self.scale_y))
        sx = max(0, min(sx, max(0, self.video_w - 1)))
        sy = max(0, min(sy, max(0, self.video_h - 1)))
        return sx, sy

    def _src_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        return int(round(x * self.scale_x)) + int(self.offset_x), int(round(y * self.scale_y)) + int(self.offset_y)

    def _on_mouse_down(self, e: tk.Event[Any]) -> None:
        if self.tk_img is None:
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
        key = self.roi_key_var.get()
        self.rois[key] = [sx0, sy0, sx1, sy1]
        self._record_history()
        self.status_var.set(f"{key} = [{sx0}, {sy0}, {sx1}, {sy1}]")
        self._refresh_canvas()
        self._update_match_score()

    def _parse_template_paths(self) -> list[str]:
        raw = self.template_paths_var.get().strip()
        if not raw:
            return []
        out: list[str] = []
        for part in raw.replace("\n", ";").replace(",", ";").split(";"):
            p = part.strip()
            if p:
                out.append(p)
        return out

    def _on_template_paths_changed(self) -> None:
        self._refresh_template_selector()
        self._update_match_score()

    def _refresh_template_selector(self) -> None:
        vals = self._parse_template_paths()
        self.template_select_combo.configure(values=vals)
        cur = self.template_selected_var.get().strip()
        if vals and cur not in vals:
            self.template_selected_var.set(vals[0])
        if not vals:
            self.template_selected_var.set("")

    def _to_abs_template_path(self, path_text: str) -> Path:
        p = Path(path_text)
        if p.is_absolute():
            return p
        return (ROOT / p).resolve()

    def _pick_templates(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择 marker 模板（可多选）",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")],
        )
        if not paths:
            return
        vals = [self._to_cfg_path(Path(p).resolve()) for p in paths]
        self.template_paths_var.set(";".join(vals))
        self._record_history()
        self.status_var.set(f"已选择模板: {len(vals)}")

    def _capture_marker_template(self) -> None:
        if self.frame_rgb_full is None:
            self.status_var.set("当前没有可用帧。")
            return
        raw_cfg = self.config_path_var.get().strip()
        if not raw_cfg:
            messagebox.showwarning("未打开 Profile", "请先选择或新建 profile，再截取 Marker。", parent=self.root)
            self.status_var.set("未打开 profile。")
            return
        cfg_path = Path(raw_cfg).resolve()
        if not cfg_path.exists():
            messagebox.showwarning("未打开 Profile", "当前 profile 不存在，请先加载有效的 profile。", parent=self.root)
            self.status_var.set("profile 不存在。")
            return
        rect = self.rois.get("marker_roi")
        if not rect:
            self.status_var.set("marker_roi 尚未设置。")
            return
        x0, y0, x1, y1 = rect
        h, w = self.frame_rgb_full.shape[:2]
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            self.status_var.set("marker_roi 无效。")
            return

        crop = self.frame_rgb_full[y0:y1, x0:x1].copy()
        profile_name = re.sub(r'[<>:"/\\\\|?*]+', "_", cfg_path.stem.strip()) or "default_profile"
        default_dir = (ROOT / "config" / profile_name).resolve()
        default_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"marker_template_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        save_path = filedialog.asksaveasfilename(
            title="保存 marker 模板",
            defaultextension=".png",
            initialdir=str(default_dir),
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("所有文件", "*.*")],
        )
        if not save_path:
            return
        out = Path(save_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        paths = self._parse_template_paths()
        rel = self._to_cfg_path(out)
        if rel not in paths:
            paths.append(rel)
            self.template_paths_var.set(";".join(paths))
            self._record_history()
        self.status_var.set(f"模板已保存: {out}")

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "rois": {k: list(v) for k, v in self.rois.items()},
            "template_paths": self.template_paths_var.get(),
            "template_selected": self.template_selected_var.get(),
            "shift_mode": self.shift_mode_var.get(),
            "vshift_px": self.vshift_px_var.get(),
            "vshift_step": self.vshift_step_var.get(),
            "hshift_px": self.hshift_px_var.get(),
            "hshift_step": self.hshift_step_var.get(),
            "force_thd": self.force_thd_var.get(),
        }

    def _apply_state(self, st: dict[str, Any]) -> None:
        self._applying_history = True
        try:
            self.rois = {k: list(v) for k, v in (st.get("rois") or {}).items()}
            self.template_paths_var.set(str(st.get("template_paths", "")))
            self.template_selected_var.set(str(st.get("template_selected", "")))
            self.shift_mode_var.set(str(st.get("shift_mode", "vertical")))
            self.vshift_px_var.set(str(st.get("vshift_px", "6")))
            self.vshift_step_var.set(str(st.get("vshift_step", "1")))
            self.hshift_px_var.set(str(st.get("hshift_px", "0")))
            self.hshift_step_var.set(str(st.get("hshift_step", "1")))
            self.force_thd_var.set(str(st.get("force_thd", "")))
            self._refresh_canvas()
            self._update_match_score()
        finally:
            self._applying_history = False

    def _reset_history(self) -> None:
        self.undo_stack = [self._snapshot_state()]
        self.redo_stack = []

    def _record_history(self) -> None:
        if self._applying_history:
            return
        cur = self._snapshot_state()
        if self.undo_stack and self.undo_stack[-1] == cur:
            return
        self.undo_stack.append(cur)
        self.redo_stack = []

    def _undo(self) -> None:
        if len(self.undo_stack) <= 1:
            return
        cur = self.undo_stack.pop()
        self.redo_stack.append(cur)
        self._apply_state(self.undo_stack[-1])
        self.status_var.set("已撤回 (Ctrl+Z)")

    def _redo(self) -> None:
        if not self.redo_stack:
            return
        st = self.redo_stack.pop()
        self.undo_stack.append(st)
        self._apply_state(st)
        self.status_var.set("已恢复 (Ctrl+Y)")

    def _to_cfg_path(self, path: Path) -> str:
        try:
            return str(path.as_posix())
        except Exception:
          return str(path)

    def _int_or_default(self, v: str, default: int) -> int:
        try:
            return int(str(v).strip())
        except Exception:
            return int(default)

    def _float_or_none(self, v: str) -> float | None:
        s = str(v).strip()
        if not s:
            return None
        try:
            f = float(s)
        except Exception:
            return None
        return max(0.0, min(1.0, f))

    def _compute_match_score(self) -> float | None:
        if self.frame_rgb_full is None:
            return None
        rect = self.rois.get("marker_roi")
        if not rect:
            return None
        selected = self.template_selected_var.get().strip()
        if not selected:
            return None

        x0, y0, x1, y1 = rect
        h, w = self.frame_rgb_full.shape[:2]
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            return None

        roi_rgb = self.frame_rgb_full[y0:y1, x0:x1]
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        tpl = self._to_abs_template_path(selected)
        if not tpl.exists():
            return None

        matcher = MarkerTemplateMatcher(
            template_paths=[tpl],
            center_width=self.template_center_width,
            vertical_shift_px=self._int_or_default(self.vshift_px_var.get(), 0),
            vertical_shift_step=max(1, self._int_or_default(self.vshift_step_var.get(), 1)),
            horizontal_shift_px=self._int_or_default(self.hshift_px_var.get(), 0),
            horizontal_shift_step=max(1, self._int_or_default(self.hshift_step_var.get(), 1)),
            shift_mode=self.shift_mode_var.get(),
        )
        return float(matcher.score(roi_gray))

    def _update_match_score(self) -> None:
        try:
            score = self._compute_match_score()
        except Exception as exc:
            self.match_score_var.set(f"匹配分数: N/A ({exc})")
            return
        if score is None:
            self.match_score_var.set("匹配分数: N/A")
            return
        self.match_score_var.set(f"匹配分数: {score:.4f}")

    def _refresh_canvas(self) -> None:
        self.canvas.delete("all")
        canvas_w = max(1, int(self.canvas.winfo_width()))
        canvas_h = max(1, int(self.canvas.winfo_height()))
        if self.tk_img is None:
            if self._decode_overlay_visible:
                spinner = self._decode_spinner_chars[self._decode_spinner_phase]
                self.canvas.create_text(
                    canvas_w // 2,
                    canvas_h // 2,
                    text=f"{spinner} Decoding...",
                    fill="#ffffff",
                    font=("Segoe UI", 12, "bold"),
                )
            return
        self.canvas.create_image(int(self.offset_x), int(self.offset_y), anchor="nw", image=self.tk_img)
        self.canvas.configure(scrollregion=(0, 0, canvas_w, canvas_h))
        for key in ROI_KEYS:
            rect = self.rois.get(key)
            if not rect:
                continue
            color = ROI_COLORS.get(key, "#ffffff")
            x0, y0 = self._src_to_canvas(rect[0], rect[1])
            x1, y1 = self._src_to_canvas(rect[2], rect[3])
            width = 3 if key == self.roi_key_var.get() else 2
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=width)
            self.canvas.create_text(
                x0 + 5,
                y0 + 5,
                text=key,
                anchor="nw",
                fill=color,
                font=("Segoe UI", 12, "bold"),
            )
        if self.drag_start is not None and self.drag_now is not None:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_now
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ffffff", width=2, dash=(6, 3))
        if self._decode_overlay_visible:
            spinner = self._decode_spinner_chars[self._decode_spinner_phase]
            tip = f"{spinner} Decoding..."
            box_w = 190
            box_h = 34
            cx = self.offset_x + (self.display_w // 2)
            cy = self.offset_y + (self.display_h // 2)
            x1 = cx - (box_w // 2)
            y1 = cy - (box_h // 2)
            x2 = x1 + box_w
            y2 = y1 + box_h
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#000000", outline="#4c9aff", width=1)
            self.canvas.create_text(x1 + 14, y1 + 9, text=tip, anchor="nw", fill="#ffffff", font=("Segoe UI", 10, "bold"))

    def _save_config(self) -> bool:
        raw_cfg = self.config_path_var.get().strip()
        if not raw_cfg:
            self.status_var.set("未选择配置，请先选择或新建 profile。")
            return False
        cfg_path = Path(raw_cfg).resolve()
        if not cfg_path.exists():
            self.status_var.set(f"配置不存在: {cfg_path.as_posix()}")
            return False
        if not self.cfg_raw:
            self.cfg_raw = _load_raw_cfg(cfg_path)
        raw_video = self.video_path_var.get().strip()
        if raw_video:
            self.cfg_raw["video_path"] = self._to_cfg_path(Path(raw_video).resolve())
        else:
            self.cfg_raw.pop("video_path", None)
        self.cfg_raw.setdefault("roi", {})
        # Drop legacy keys when saving to keep profile concise.
        self.cfg_raw["roi"].pop("subtitle_roi", None)
        self.cfg_raw["roi"].pop("title_text_roi", None)
        self.cfg_raw["roi"].pop("title_info_roi", None)
        self.cfg_raw["roi"].pop("title_subtitle_roi", None)
        self.cfg_raw["roi"].pop("title_speaker_roi", None)
        self.cfg_raw["roi"].pop("title_roi", None)
        for k, rect in self.rois.items():
            self.cfg_raw["roi"][k] = [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])]
        if self.video_w > 0 and self.video_h > 0:
            self.cfg_raw["roi"]["resolution_base"] = [int(self.video_w), int(self.video_h)]

        self.cfg_raw.setdefault("marker", {})
        marker = self.cfg_raw["marker"]
        tpl_paths = self._parse_template_paths()
        if tpl_paths:
            marker["template_paths"] = tpl_paths
        else:
            marker.pop("template_paths", None)
        marker.pop("template_path", None)
        marker.pop("high_res_template_path", None)
        marker["shift_mode"] = "horizontal" if self.shift_mode_var.get() == "horizontal" else "vertical"
        marker["vertical_shift_px"] = self._int_or_default(self.vshift_px_var.get(), 0)
        marker["vertical_shift_step"] = max(1, self._int_or_default(self.vshift_step_var.get(), 1))
        marker["horizontal_shift_px"] = self._int_or_default(self.hshift_px_var.get(), 0)
        marker["horizontal_shift_step"] = max(1, self._int_or_default(self.hshift_step_var.get(), 1))
        marker["ocr_anchor_from_end_frames"] = max(1, self._int_or_default(self.anchor_from_end_var.get(), 3))
        force_thd = self._float_or_none(self.force_thd_var.get())
        if force_thd is None:
            marker.pop("force_threshold", None)
        else:
            marker["force_threshold"] = force_thd

        self.cfg_raw.setdefault("general", {})
        self.cfg_raw["general"]["blank_ignore_under_frames"] = max(
            0, self._int_or_default(self.blank_ignore_var.get(), 1)
        )

        self.cfg_raw.setdefault("game", {})
        self.cfg_raw["game"]["source_language"] = str(self.source_lang_var.get().strip() or "ja")
        self.cfg_raw["game"]["target_language"] = str(self.target_lang_var.get().strip() or "zh-CN")
        self.cfg_raw["output_name"] = self._sanitize_output_name(self.output_name_var.get())

        self.cfg_raw.setdefault("translation", {})
        model_list = self._parse_model_list(self.vlm_models_var.get())
        self.cfg_raw["translation"]["vlm_models"] = model_list if model_list else ["qwen3.6-plus"]
        self._sync_current_model_from_list()
        self.cfg_raw["translation"]["model"] = str(
            self.vlm_model_var.get().strip() or self.cfg_raw["translation"]["vlm_models"][0]
        )

        _save_raw_cfg(cfg_path, self.cfg_raw)
        self.status_var.set(f"配置已保存: {cfg_path.as_posix()}")
        self._load_profile_text_from_file()
        return True

    def _run_pipeline_from_gui(self) -> None:
        errs = self._validate_before_run()
        if errs:
            messagebox.showerror("参数未配置完整", "\n".join(errs), parent=self.root)
            return
        if not self._save_config():
            messagebox.showerror("保存失败", "配置保存失败，请检查后重试。", parent=self.root)
            return

        video = Path(self.video_path_var.get().strip()).resolve()
        config = Path(self.config_path_var.get().strip()).resolve()
        output_dir = self._resolve_output_dir()
        self._open_run_window(video, config, output_dir)
        self._run_cancel_requested = False
        self._run_started_ts = time.time()
        self._run_last_heartbeat_sec = -1
        self._set_run_active(True)
        self._tick_run_elapsed()
        # self._append_run_log("[INFO] 已启动任务，等待子进程日志输出...\n")
        self.status_var.set("流程运行中...")

        def _worker() -> None:
            run_log_file = output_dir / "run.log"
            try:
                if self._run_cancel_requested:
                    self.root.after(0, lambda: self.status_var.set("任务已停止。"))
                    return
                try:
                    self._reset_run_log_file(run_log_file)
                except Exception as exc:
                    err_msg = str(exc)
                    self.root.after(0, lambda m=err_msg: self.status_var.set(f"初始化 run.log 失败: {m}"))
                    return
                cmd_cache = self._build_pipeline_cmd(video, config, output_dir, ["--cache-only"])
                rc = self._run_subprocess_stream(cmd_cache, run_log_file)
                if self._run_cancel_requested:
                    self.root.after(0, lambda: self.status_var.set("任务已停止。"))
                    return
                if rc != 0:
                    self.root.after(0, lambda: self.status_var.set(f"运行失败，退出码={rc}"))
                    return

                cache_path = self._find_latest_translation_cache(output_dir)
                if cache_path is None:
                    self.root.after(0, lambda: self.status_var.set("运行完成，但未找到 translation_cache.json"))
                    return

                cmd_sub_template = self._build_pipeline_cmd(
                    video,
                    config,
                    output_dir,
                    ["--subtitles-from-cache", "--translation-cache", str(cache_path)],
                )
                if bool(self.post_action_make_subtitle_var.get()):
                    rc2 = self._run_subprocess_stream(list(cmd_sub_template), run_log_file)
                    if self._run_cancel_requested:
                        self.root.after(0, lambda: self.status_var.set("任务已停止。"))
                        return
                    if rc2 == 0:
                        self.root.after(0, lambda: self.status_var.set("缓存与字幕生成完成。"))
                    else:
                        self.root.after(0, lambda: self.status_var.set(f"字幕生成失败，退出码={rc2}"))
                    return

                if self._run_cancel_requested:
                    self.root.after(0, lambda: self.status_var.set("任务已停止。"))
                    return
                self.root.after(0, lambda: self.status_var.set("缓存生成完成，正在打开 Review 界面..."))
                self.root.after(0, lambda: self._launch_cache_review(cache_path, video, config))
            finally:
                self.root.after(0, lambda: self._set_run_active(False))

        threading.Thread(target=_worker, daemon=True).start()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile编辑器 - IGNITE")
    parser.add_argument("--video", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--output-name", default="")
    args = parser.parse_args()
    app = RoiEditorApp(video_path=args.video, config_path=args.config, output_name=args.output_name)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
