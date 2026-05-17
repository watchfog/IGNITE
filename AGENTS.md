# AGENTS.md

## 项目概述

IGNITE — 面向游戏剧情视频的字幕流水线 GUI 工具（OCR + VLM 识别/翻译 → 校对 → 导出 SRT/ASS）。纯 Python，Tkinter GUI。

## 常用命令

```bash
pip install -r requirements.txt                   # 安装依赖

python main.py                                    # 启动 Profile 编辑器 GUI
python main.py --video <path> --config <path>      # 带参数启动

python -m ignite.pipeline \  # 直接运行 pipeline（无 GUI）
    --video <path> --config <path> --output-dir <path>

python -m ignite.gui.review --cache <path>   # 重新打开校对 GUI

python tools/debug_rapidocr_single_image.py \       # OCR 调试
    --image <path> --config <path>
```

**本项目无测试、无 lint、无 CI**。不要尝试运行 `pytest`、`npm run` 等命令。

## 架构要点

### 入口点

| 入口 | 用途 |
|------|------|
| `main.py` | Profile 编辑器 GUI（ROI 框选、marker 配置、启动流程） |
| `ignite/pipeline.py` | 核心流水线：分段 → OCR → VLM 翻译 → 缓存 → 字幕生成 |
| `ignite/gui/review.py` | 字幕校对 GUI（编辑翻译、复译、生成字幕） |

### 核心模块 (`ignite/`)

| 模块 | 行数 | 职责 |
|------|------|------|
| `pipeline.py` | 1690 | 主流水线编排：阶段调度、CLI 入口 `build_parser()`、`run_pipeline()` |
| `state_machine.py` | 325 | 状态机分段：NO_DIALOGUE → TEXT_APPEARING → TEXT_STABLE → TEXT_CLEARING |
| `event_detect.py` | 450 | 帧差异计算、marker 模板匹配 (`MarkerTemplateMatcher`)、`score_frame`、`score_batch` |
| `ocr_engines.py` | 156 | OCR 引擎 (`build_ocr_engine`，RapidOCR) |
| `translation_runtime.py` | 1128 | VLM 翻译客户端 (`BailianVlmTranslator`) + `translate_segment_with_retry`、`normalize_quotes_for_subtitle` |
| `subtitle_export.py` | 142 | SRT + ASS 字幕导出 + 多行括号缩进对齐 |
| `config.py` | 114 | YAML 配置加载，支持 `extends` 深合并 |
| `ffmpeg_utils.py` | 270 | FFmpeg/FFprobe 封装（取帧、视频信息）+ 统一四路输出 |
| `datatypes.py` | 50 | 数据类：`VideoMeta`、`Roi`、`OcrResult`、`DialogueSegment` |
| `cache_manager.py` | 226 | 翻译缓存 JSON 读写、索引、命中检查 |
| `name_splitter.py` | 808 | Name OCR 分割 (`_split_segment_by_name_ocr`) + 子段标准化 |
| `marker_ops.py` | 326 | Marker/Marker2 评分、裁剪、分割 (`_split_segment_by_marker2`、`_build_refined_subsegment`) |
| `name_ocr_runner.py` | 218 | `NameOcrRunner` 类（姓名区域 mask/OCR 检测，线程池并发） |
| `debug_utils.py` | 99 | Debug 中间产物导出（marker/name 帧） |
| `review_utils.py` | 78 | Review 元数据工具 (`_merge_review_reasons`、`_fill_short_false_gaps`、`_first_true_run_bounds`) |
| `image_utils.py` | 77 | 图像裁剪/Base64/帧缓存加载 |
| `log_utils.py` | 17 | 共享 `_log` / `set_log_file` |
| `gui/` | — | GUI 子包：`profile.py`（Profile 编辑器）、`review.py`（字幕校对） |

### 模块依赖关系

```
pipeline.py  ← 编排入口，import 所有下层模块
  ├── cache_manager.py      (翻译缓存 CRUD，纯函数)
  ├── review_utils.py       (review 元数据，纯工具)
  ├── image_utils.py        (图像处理，依赖 PIL)
  ├── debug_utils.py        (debug 导出)
  ├── log_utils.py          (共享日志)
  ├── marker_ops.py         (marker 评分+分割，依赖 event_detect + review_utils)
  ├── name_splitter.py      (name OCR 分割，依赖 name_ocr_runner + marker_ops + review_utils)
  ├── name_ocr_runner.py    (姓名 OCR 检测，依赖 ocr_engines + event_detect)
  ├── translation_runtime.py (VLM 客户端，独立)
  ├── subtitle_export.py    (字幕导出，依赖 datatypes)
  ├── state_machine.py      (状态机，独立)
  ├── ffmpeg_utils.py       (FFmpeg 封装，独立)
  ├── config.py             (配置加载，独立)
  └── datatypes.py          (共享数据类)
```

### Pipeline 分段流程

1. **Marker 粗分段**：`MarkerTemplateMatcher.score_batch()` 对全视频做 marker 模板匹配 → 状态机输出 NO_DIALOGUE/TEXT_APPEARING/TEXT_STABLE/TEXT_CLEARING 序列
2. **Name/Marker2 精化**（二选一）：
   - **OCR 模式**：`_split_segment_by_name_ocr` 通过姓名区域 OCR 进一步切分空白→对话
   - **Marker2 模式**（OCR 禁用时）：`_split_segment_by_marker2` 通过第二标记模板匹配切分
3. **Normalize**：`_normalize_name_subsegments_per_marker` 将精化结果按 marker_seg_id 归组标准化

### 分段结构约束

每个 segment 的 `dialogue_type` 只能是以下两种情况：

| 结构 | 说明 |
|------|------|
| `blank_no_name` → `dialogue` | 空白段（无对话）→ 对话段 |
| 纯 `dialogue` | 整段都是对话，无前置空白 |

### Marker2 匹配逻辑

- **True 可靠**：只用第一个 True 连续段判定对话边界，不做 `_blank_confirm` 二次确认
- **两段式匹配**：若配置了 `marker_2_match_roi`（> `marker_2_roi`），帧截取即用大区域。`cv2.matchTemplate` 自动搜索全图，位移偏移 `vertical_shift_px/horizontal_shift_px` 自动设为 0

### 配置系统

- `config/general_config.yaml` — 全局默认配置（**被 git 跟踪**）
- `config/subtitle_style.yaml` — 字幕样式配置（字体、字号、颜色等，**被 git 跟踪**），含 `speaker_styles` 按角色名配置描边颜色，仅 ASS/硬字幕生效
- Per-video 配置文件通过 `extends` 字段继承 `general_config.yaml`，覆盖 ROI 坐标等参数
- API 密钥：在 `general_config.yaml` 的 `translation.api_key_file` 指向 `config/keys/` 下的文件（不是环境变量）
- VLM 默认模型：Qwen 3.6 Plus，`responses` 模式，须支持视觉识别

## 代码约定

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

- `config/` 下仅跟踪 `general_config.yaml` 和 `subtitle_style.yaml`，其他 config 文件被忽略
- `tools/`、`outputs/`、`archive/`、`examples/` 仅保留 `.gitkeep`
- 所有视频格式（mp4/avi/mov/mkv/webm）和 API 密钥文件均被忽略
