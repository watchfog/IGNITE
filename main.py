from __future__ import annotations

import argparse
from pathlib import Path

from app.roi_editor import RoiEditorApp


def main() -> int:
    parser = argparse.ArgumentParser(description="IGNITE GUI 主入口")
    parser.add_argument("--video", default="", help="默认视频路径（可选）")
    parser.add_argument("--config", default="", help="默认配置路径（可选）")
    parser.add_argument(
        "--output-dir",
        default="",
        help="默认输出目录（可选），仅取最后一级目录名作为输出文件夹名。",
    )
    args = parser.parse_args()

    out_name = ""
    raw_out = str(args.output_dir or "").strip()
    if raw_out:
        out_name = Path(raw_out).name.strip()

    app = RoiEditorApp(
        video_path=str(args.video or "").strip(),
        config_path=str(args.config or "").strip(),
        output_name=out_name,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
