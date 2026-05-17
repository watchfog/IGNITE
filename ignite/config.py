from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _parse_scalar(raw: str) -> Any:
    if raw in {"null", "None"}:
        return None
    if raw in {"true", "True"}:
        return True
    if raw in {"false", "False"}:
        return False
    if raw.startswith("[") and raw.endswith("]"):
        text = raw.replace("'", '"')
        return json.loads(text)
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _simple_yaml_load(text: str) -> dict[str, Any]:
    # Minimal YAML subset parser used when PyYAML is unavailable.
    # It preserves `key:` as None unless nested mapping lines follow.
    entries: list[tuple[int, str, str | None]] = []
    for line in text.splitlines():
        raw = line.rstrip()
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, value = stripped.split(":", 1)
        k = key.strip()
        v = value.strip()
        entries.append((indent, k, v if v != "" else None))

    def _parse_block(start: int, parent_indent: int) -> tuple[dict[str, Any], int]:
        out: dict[str, Any] = {}
        i = start
        while i < len(entries):
            indent, key, value = entries[i]
            if indent <= parent_indent:
                break
            if value is not None:
                out[key] = _parse_scalar(value)
                i += 1
                continue

            has_child = (i + 1) < len(entries) and entries[i + 1][0] > indent
            if has_child:
                child, next_i = _parse_block(i + 1, indent)
                out[key] = child
                i = next_i
            else:
                out[key] = None
                i += 1
        return out, i

    parsed, _ = _parse_block(0, -1)
    return parsed


def _load_single_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    text_strip = text.strip()

    if not text_strip:
        raise ValueError(f"Config file is empty: {path}")
    if text_strip.startswith("{"):
        return json.loads(text_strip)

    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)  # pragma: no cover
    except Exception:
        return _simple_yaml_load(text)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if (
            isinstance(v, dict)
            and isinstance(out.get(k), dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_extends_paths(raw: Any, cfg_path: Path) -> list[Path]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: list[Path] = []
    for item in items:
        p = Path(str(item))
        if not p.is_absolute():
            p = (cfg_path.parent / p).resolve()
        out.append(p)
    return out


def _load_with_extends(path: Path, seen: set[Path]) -> dict[str, Any]:
    rp = path.resolve()
    if rp in seen:
        raise ValueError(f"Config cycle detected: {rp}")
    seen.add(rp)

    cfg = _load_single_config(rp) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {rp}")

    merged: dict[str, Any] = {}
    for base_path in _resolve_extends_paths(cfg.get("extends"), rp):
        base_cfg = _load_with_extends(base_path, seen)
        merged = _deep_merge(merged, base_cfg)

    cfg.pop("extends", None)
    merged = _deep_merge(merged, cfg)
    seen.remove(rp)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_with_extends(Path(path), seen=set())
