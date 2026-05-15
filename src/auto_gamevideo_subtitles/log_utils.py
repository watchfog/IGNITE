from __future__ import annotations

from datetime import datetime
from pathlib import Path

_LOG_FILE_PATH: Path | None = None


def set_log_file(path: Path) -> None:
    global _LOG_FILE_PATH
    _LOG_FILE_PATH = path


def _log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE_PATH is not None:
        try:
            _LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_FILE_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
