from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .archive_manager import archive_project


def _find_caches(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("translation_cache_latest.json") if p.is_file())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Archive IGNITE video/config/cache/subtitle artifacts")
    parser.add_argument("--cache", help="translation_cache_latest.json path")
    parser.add_argument("--cache-root", default="outputs", help="Root to scan in --batch mode")
    parser.add_argument("--dest-root", required=True, help="Archive root directory")
    parser.add_argument("--batch", action="store_true", help="Archive all translation_cache_latest.json under --cache-root")
    parser.add_argument("--video", default="", help="Override source video path (single cache mode)")
    parser.add_argument("--config", default="", help="Override source config path (single cache mode)")
    parser.add_argument("--hard-sub-video", default="", help="Optional hard-subtitle video path (single cache mode)")
    parser.add_argument("--name", default="", help="Archive folder name (single cache mode)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files in target archive folder if it already exists")
    return parser


def _run_one(args: argparse.Namespace, cache_path: Path) -> None:
    result = archive_project(
        cache_path=cache_path,
        dest_root=Path(args.dest_root),
        video_path=Path(args.video).resolve() if str(args.video).strip() else None,
        config_path=Path(args.config).resolve() if str(args.config).strip() else None,
        hard_sub_video=Path(args.hard_sub_video).resolve() if str(args.hard_sub_video).strip() else None,
        name=str(args.name or "").strip() or None,
        overwrite=bool(args.overwrite),
    )
    print(f"Archived: {cache_path} -> {result.archive_dir}")
    if result.missing:
        print("Missing optional artifacts:")
        for item in result.missing:
            print(f"  - {item}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.batch:
        caches = _find_caches(Path(args.cache_root))
        if not caches:
            raise RuntimeError(f"No translation_cache_latest.json found under {args.cache_root}")
        failed = 0
        for cache_path in caches:
            try:
                _run_one(args, cache_path.resolve())
            except Exception as exc:
                failed += 1
                print(f"FAILED: {cache_path}: {exc}", file=sys.stderr)
        return 1 if failed else 0

    if not str(args.cache or "").strip():
        parser.error("--cache is required unless --batch is used")
    _run_one(args, Path(args.cache).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
