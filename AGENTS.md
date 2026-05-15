# AGENTS.md

## 项目概述

IGNITE — 面向游戏剧情视频的字幕流水线 GUI 工具（OCR + VLM 识别/翻译 → 校对 → 导出 SRT/ASS）。纯 Python，Tkinter GUI。

## 常用命令

```bash
pip install -r requirements.txt                   # 安装依赖

python main.py                                    # 启动 Profile 编辑器 GUI
python main.py --video <path> --config <path>      # 带参数启动

python -m src.auto_gamevideo_subtitles.pipeline \  # 直接运行 pipeline（无 GUI）
    --video <path> --config <path> --output-dir <path>

python app/cache_browser_review.py --cache <path>   # 重新打开校对 GUI

python tools/debug_rapidocr_single_image.py \       # OCR 调试
    --image <path> --config <path>
```

**本项目无测试、无 lint、无 CI**。不要尝试运行 `pytest`、`npm run` 等命令。

## 架构要点

### 入口点

| 入口 | 用途 |
|------|------|
| `main.py` | Profile 编辑器 GUI（ROI 框选、marker 配置、启动流程） |
| `src/auto_gamevideo_subtitles/pipeline.py` | 核心流水线：分段 → OCR → VLM 翻译 → 缓存 → 字幕生成 |
| `app/cache_browser_review.py` | 字幕校对 GUI（编辑翻译、复译、生成字幕） |

### 核心模块 (`src/auto_gamevideo_subtitles/`)

| 模块 | 职责 |
|------|------|
| `pipeline.py` | 主流水线（~3800 行），`build_parser()` 定义 CLI 参数 |
| `state_machine.py` | 状态机分段：NO_DIALOGUE → TEXT_APPEARING → TEXT_STABLE → TEXT_CLEARING |
| `event_detect.py` | 帧差异计算、marker 模板匹配 |
| `ocr_engines.py` | OCR 引擎（RapidOCR） |
| `translation_runtime.py` | VLM 翻译客户端（阿里云百炼 DashScope，OpenAI 兼容模式） |
| `subtitle_export.py` | SRT + ASS 字幕导出 |
| `config.py` | YAML 配置加载，支持 `extends` 字段做深层合并 |
| `ffmpeg_utils.py` | FFmpeg/FFprobe 封装 |
| `models.py` | 数据类：`VideoMeta`、`Roi`、`OcrResult`、`DialogueSegment` |

### 配置系统

- `config/general_config.yaml` — 全局默认配置（**唯一被 git 跟踪的 config 文件**）
- Per-video 配置文件通过 `extends` 字段继承 `general_config.yaml`，覆盖 ROI 坐标等参数
- API 密钥：在 `general_config.yaml` 的 `translation.api_key_file` 指向 `config/keys/` 下的文件（不是环境变量）
- VLM 默认模型：Qwen 3.6 Plus，`responses` 模式，须支持视觉识别

## 代码约定

### sys.path 操作（重要）

`app/` 下的文件通过操作 `sys.path` 定位 `src/`，遵循固定模式：

```python
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_gamevideo_subtitles.xxx import yyy  # noqa: E402
```

**新增 `app/` 下文件时，必须遵循此模式**。import 语句放在 `sys.path` 修改之后，并加 `# noqa: E402`。

### 文件头

- 首行 `from __future__ import annotations`（几乎所有文件）
- 部分旧文件有 `# -*- coding: utf-8 -*-`
- 使用延迟类型注解（PEP 563）

### 线程模型

- GUI `threading.Thread` 后台执行 pipeline 等耗时任务
- `concurrent.futures.ThreadPoolExecutor` 做并发 OCR/模板匹配

### 外部依赖

- **FFmpeg/FFprobe**：路径默认为 `tools/ffmpeg/bin/`，可通过配置覆盖
- **Tkinter**：Windows Python 自带，Linux 需 `python3-tk`

## 输出与缓存

- 输出目录：`outputs/<名称>/`
- 翻译缓存：`translation_cache_latest.json`（最新）和 `work/run_cache_<时间>/translation_cache.json`（仅保留 3 次）
- 最终字幕：`subtitles.srt`、`subtitles.ass`、`subtitles_debug.srt`、`subtitles_debug.ass`

## .gitignore 注意

- `config/` 下仅跟踪 `general_config.yaml`，其他 config 文件被忽略
- `tools/`、`outputs/`、`archive/`、`examples/` 仅保留 `.gitkeep`
- 所有视频格式（mp4/avi/mov/mkv/webm）和 API 密钥文件均被忽略
