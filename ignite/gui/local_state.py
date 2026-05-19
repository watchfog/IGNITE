from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOCAL_GUI_STATE_PATH = ROOT / "config" / "local_gui_state.json"

CONFIG_DIALOG_DIR_KEYS = (
    "profile.config_open",
    "profile.config_create",
    "profile.config_import_source",
    "profile.config_import_save",
    "review.config",
    "config",
)


def _load_state() -> dict[str, Any]:
    if not LOCAL_GUI_STATE_PATH.exists():
        return {}
    try:
        raw = json.loads(LOCAL_GUI_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_state(state: dict[str, Any]) -> None:
    LOCAL_GUI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_GUI_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _selected_dir(selected_path: str | Path) -> Path | None:
    if not str(selected_path or "").strip():
        return None
    try:
        path = Path(selected_path).expanduser().resolve()
    except Exception:
        return None
    directory = path if path.is_dir() else path.parent
    return directory if directory.exists() else None


def load_dialog_dirs() -> dict[str, Path]:
    state = _load_state()
    raw_dirs = state.get("dialog_dirs")
    if not isinstance(raw_dirs, dict):
        return {}
    dirs: dict[str, Path] = {}
    for key, raw_path in raw_dirs.items():
        if not isinstance(key, str) or not key.strip():
            continue
        directory = _selected_dir(str(raw_path or ""))
        if directory is not None:
            dirs[key] = directory
    return dirs


def related_dialog_dir(dialog_dirs: dict[str, Path], key: str) -> Path | None:
    clean_key = str(key or "").strip()
    if clean_key not in CONFIG_DIALOG_DIR_KEYS:
        return None
    for candidate_key in CONFIG_DIALOG_DIR_KEYS:
        if candidate_key == clean_key:
            continue
        directory = dialog_dirs.get(candidate_key)
        if directory is not None and directory.exists():
            return directory
    return None


def remember_dialog_dir(
    key: str,
    selected_path: str | Path,
    dialog_dirs: dict[str, Path] | None = None,
) -> Path | None:
    clean_key = str(key or "").strip()
    if not clean_key:
        return None
    directory = _selected_dir(selected_path)
    if directory is None:
        return None

    if dialog_dirs is not None and clean_key not in dialog_dirs:
        related = related_dialog_dir(dialog_dirs, clean_key)
        if related is not None and related == directory:
            return None

    state = _load_state()
    state["version"] = 1
    raw_dirs = state.get("dialog_dirs")
    if not isinstance(raw_dirs, dict):
        raw_dirs = {}
        state["dialog_dirs"] = raw_dirs
    raw_dirs[clean_key] = str(directory)
    _save_state(state)
    return directory


def load_window_state(key: str) -> dict[str, Any]:
    clean_key = str(key or "").strip()
    if not clean_key:
        return {}
    state = _load_state()
    windows = state.get("windows")
    if not isinstance(windows, dict):
        return {}
    raw = windows.get(clean_key)
    return dict(raw) if isinstance(raw, dict) else {}


def remember_window_state(
    key: str,
    *,
    geometry: str,
    window_state: str = "",
    layout: dict[str, int] | None = None,
) -> None:
    clean_key = str(key or "").strip()
    clean_geometry = str(geometry or "").strip()
    if not clean_key or not clean_geometry:
        return

    state = _load_state()
    state["version"] = 1
    windows = state.get("windows")
    if not isinstance(windows, dict):
        windows = {}
        state["windows"] = windows

    payload: dict[str, Any] = {"geometry": clean_geometry}
    clean_window_state = str(window_state or "").strip()
    if clean_window_state in {"normal", "zoomed"}:
        payload["state"] = clean_window_state
    if layout:
        payload["layout"] = {str(k): int(v) for k, v in layout.items()}
    windows[clean_key] = payload
    _save_state(state)
