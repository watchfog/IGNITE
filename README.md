# IGNITE

> **I**ndexing, reco**G**nition, tra**N**slation, rev**I**ew and subti**T**l**E**ing

IGNITE 是一个面向游戏剧情视频的字幕流水线工具。目标是针对**画面文字驱动**的游戏剧情录屏，使用OCR配合VLM自动化完成时间轴、文本识别、翻译，经人工校对和字幕导出的GUI工具。

## 功能概览

- 基于视频画面自动分段并生成时间轴
- 使用VLM完成识别/翻译
- 生成可编辑的翻译缓存
- 基于缓存进行GUI审查，支持人工复核翻译内容/重译/时间点编辑等功能
- 从缓存生成 ASS 字幕和内嵌硬字幕视频

## 依赖与环境

### 拉取仓库

```bash
git clone https://github.com/watchfog/IGNITE.git
cd IGNITE
```

或从网页端下载。

### Python 依赖安装

```bash
pip install -r requirements.txt
```

依赖包括：

- `numpy`：数值计算
- `opencv-python`：视频读取与图像处理
- `Pillow`：Tk GUI 图像显示
- `PyYAML`：读取配置
- `rapidocr` + `onnxruntime`：OCR 运行时
- `requests`：访问在线模型接口

`Tkinter` 通常随 Windows 官方 Python 一起安装；如果是 Linux，通常需要额外安装系统包，例如 `python3-tk`。

### FFmpeg 配置

本项目依赖 FFmpeg 进行取帧和精确预览。默认配置路径写在 `config/general_config.yaml` 的 `tools` 段中：

- `tools.ffmpeg_path`
- `tools.ffprobe_path`

默认值在相对路径 `tools/ffmpeg/bin/` 下，可指定外部绝对路径。可自行编译或采用编译好的二进制文件，见 [FFmpeg 官方文档](https://ffmpeg.org/download.html)。

### VLM 配置

VLM 相关配置在 `config/general_config.yaml` 的 `translation` 部分。

必须先配置 `translation.api_key_file` 指向包含 API 密钥的文件。

默认使用阿里云百炼的 OpenAI 兼容地址 `https://dashscope.aliyuncs.com/compatible-mode/v1`，默认模型 `Qwen 3.6 Plus`，`responses` 模式。

API 地址通过 `translation.vlm_api` 指定，模型通过 `translation.model` 指定，模型必须在 `translation.vlm_models` 列表中。

注意使用的模型必须带有视觉识别功能且支持 `responses` 模式调用。

## GUI 使用流程

### 0. 配置 `config/general_config.yaml`

除了 VLM 配置外，其他参数见注释，基本不需要更改。

每个视频的配置通过 `extends` 字段继承通用配置，可覆盖 ROI 坐标等参数。示例见 `config/example_profile.yaml`。

### 1. 启动主 GUI

```bash
python main.py --video [VIDEO] --config [PROFILE] --output-dir [NAME]
```

也可以不指定参数，直接运行 `python main.py` 启动 GUI 后再选择视频和配置文件。

文件选择对话框会按用途独立记忆本次运行中的目录，例如视频、配置、Marker 模板和输出视频互不影响。

### 2. 在 Profile 编辑器 GUI 中进行配置

不传入参数时的 GUI 界面如下

![Main GUI](docs/images/profile_gui_empty.png)

首先点击视频右侧的选择，选择目标翻译视频；
然后点击配置右侧的新建配置，或从现有配置导入。

新建完成后，选取 ROI 组件，并在视频预览上拖动框选。注意如果视频 GOP 间隔过大，拖动时的预览可能与实际不符，请等待加载完成。

组件包括 8 种：

- `name_roi` — 人名位置
- `dialogue_roi` — 对话位置
- `marker_roi` — 对话结束标记位置（Marker 1）
  - 区域在完整包裹 marker（包括浮动）的前提下，越小越好。
  - 在预览图稳定后，点击 `截取 Marker`，并保存截图，用于模板匹配。默认会在 `config/` 下新建与视频同名的文件夹。
  - 对于浮动较大的 Marker，可以选择截取多次，或配置位移方向，或同时使用两者。
  - **建议强制阈值 0.6**
- `marker_2_roi` — 第二对话标记位置（Marker 2），用于 OCR 禁用时做空白→对话分割
  - 区域选取和截取方式与 marker_roi 相同。
  - **建议强制阈值 0.2**
- `subtitle_location` — 字幕放置位置（左下角为锚点）
- `title_ocr_roi` — 标题 OCR 区域
- `title_translation_location` — 标题翻译文字放置位置（居中）
- `title_info_location` — 标题信息放置位置（翻译信息如时间轴校对）

![ROI 1](docs/images/roi_1.png)

![ROI 2](docs/images/roi_2.png)

接下来拖动进度条，查看 Marker 在不同位置的匹配分数，根据匹配分数配置强制阈值。

如果在有 Marker 的位置匹配分数仍然很低，建议确认 marker_roi 范围是否合理，以及多次截取 Marker 并保存，选取多个模板用于匹配。

如果拖动视频非常**卡顿**，请点击 `修复视频(ffmpeg)` 尝试自动修复。问题常出现在**流媒体平台**下载的视频中。

![Marker 1](docs/images/profile_gui_marker.png)

完成后，点击 `保存配置`。在底部预览框的 `name` 处填入游戏名字，用于告知 VLM 相关背景信息。较新的游戏受限于 VLM 知识，可能效果不佳，可开启联网搜索。

填入完成后，点击 `保存编辑框到文件`。

然后修改输出文件夹名（如有需要），默认与视频名相同。

根据需要选择参数：

- **直接生成字幕**：缓存后直接生成字幕，不打开 Review
- **Debug**：保存更多中间结果，用于解决分段错误问题
- **跳过翻译**：跳过 VLM 翻译，只打时间轴
- **启用 OCR**：启用姓名 OCR 分割
- **源语言 / 目标语言**：默认为 `ja` / `zh-CN`
- **VLM 模型**：支持多模型逗号分隔切换
- **编辑额外要求**：添加自定义翻译提示词

最后点击 `运行流程`，会弹出监控窗口，开始运行识别和翻译。

![GUI End](docs/images/profile_gui_end.png)

### 3. 等待流程运行完毕

运行阶段如下：

- **粗分段**：根据 Marker 模板匹配结果进行较粗的划分
- **姓名切分**：将对话与说话人同步，两种模式二选一：
  - **OCR 模式**（默认）：通过 OCR 检测姓名区域是否存在文字来切分
  - **Marker 2 模式**（禁用 OCR 时）：利用 Marker 2 模板匹配做空白→对话分割
- **缓存命中检查**：与已有缓存比对，时间段一致的片段直接复用
- **VLM 翻译**：将姓名和对话部分截图传给 VLM 进行识别和翻译，格式化保存至缓存
- **字幕导出**：生成 ASS 字幕文件

流程耗时根据视频分辨率、划分段数、文本量和电脑性能有关，一般为视频长度的 2 倍左右。

token 消耗根据视频分辨率和文本量有所不同，测试视频 720p 大约每分钟不到 25k token，其中输入输出约为 1:1.5，按百炼平台 Qwen 3.6 Plus 的价格大约 0.2 元，使用前请保证 API 余额充足。

![GUI End](docs/images/cmd_run.png)

### 4. 结果校对

运行完成后会弹出字幕校对工具，支持的功能有：

- **标题操作**：插入/更新标题，识别并翻译标题画面
- **编辑字幕**：修改片段起止时间、翻译文本
- **插入/删除段**：在当前段前/后插入新段，或删除当前段；`segment_id` 自动顺延
- **合并段**：将当前段与上/下一段合并，支持非模态弹窗编辑起止时间，选择 speaker/原文/译文/needs_review/style 的来源（保留当前段/保留另一段/合并/清除）
- **撤销/恢复**：支持多步撤回和恢复，覆盖合并、插入、删除操作
- **重译**：重新截图并复译（针对 ROI 未框选准的情况），或用原文直接重译
- **自定义 Prompt**：追加额外提示词到复译请求，支持开启联网搜索
- **ROI 拖拽编辑**：在校对界面中直接调整 name_roi、dialogue_roi、title_ocr_roi
- **JSON 直接编辑**：编辑当前段的原始缓存 JSON
- **存疑标记**：标记/跳转存疑片段；操作条可勾选"插入/合并额外review标记"控制是否追加 `manual_insert`/`manual_merge`
- **内嵌字幕视频生成**：使用 ffmpeg 渲染硬字幕视频，支持 CPU/NVENC/QSV/AMF 编码

![GUI End](docs/images/title_review.png)

对标题功能：通过预览确定标题开始/结束时间，将时间轴拖动到有标题文字的画面，填写翻译信息后点击 `识别并翻译Title` 按钮即可。

点击按钮 `上一段` / `下一段` 可以在不同字幕段之间切换。键盘快捷键 `←` / `→`。
注意这个跳转性能极差，快速点击会导致 GUI 卡死。如果需要快速预览效果，建议先生成字幕在播放器中预览，配合 debug 字幕定位后回来编辑。

![GUI End](docs/images/dialogue_review.png)

复译结果返回后，可选择 `一键替换(人名+原文+译文)` 或 `仅替换译文`，也可以手动编辑。

编辑后点击 `保存全部`（Ctrl+S）保存。

如果不小心关闭此界面，可通过以下命令重新启动：

```bash
python -m ignite.gui.review --cache [CACHE] --video [VIDEO] --config [CONFIG]
```

如果 cache 里记录的视频路径已经失效，校对工具仍可打开并编辑 JSON。需要视频预览、截图复译或重新生成字幕时，可在顶部“视频”栏重新选择/加载视频；保存 cache 时会同步新的有效视频路径。

需要整理已完成项目时，可在校对工具中点击 `归档项目`。归档会自动保存当前 cache，并复制原视频（保留原文件名）、合并后的 `config.yaml`、Marker 模板图片、`translation_cache_latest.json`、`subtitles.ass`、`subtitles_debug.ass` 和可选硬字幕视频到同一个归档目录；归档后的 cache 会把 `video` / `config_path` 改为同目录相对路径，合并后的 config 也会把 marker 模板路径改为归档内相对路径，便于之后整体移动和二次编辑。

缓存位置：
- `outputs/<名称>/translation_cache_latest.json` — 最近一次运行的结果
- `outputs/<名称>/work/run_cache_<时间>/translation_cache.json` — 当次运行的结果

注意仅保留 3 次运行结果，如有需要请及时备份。

### 5. 生成字幕

确认当前结果无误后，在校对工具中点击 `生成字幕（当前Cache）`，即可在指定目录下得到字幕文件。

最终输出包括：

- `subtitles.ass`
- `subtitles_debug.ass`

其中 `subtitles_debug.ass` 会在普通字幕基础上，于视频顶部额外叠加 debug 信息（如 `raw_id` 和 `segment_id`），方便在校对工具中定位。
新生成的 cache entry 会写入原始粗分段编号 `raw_id`，debug overlay 会基于 `raw_id` 动态生成；旧 cache 中已有的 `debug_subtitle` 仍会在从 cache 生成字幕时优先使用。

如需生成内嵌硬字幕视频，点击 `生成内嵌字幕视频`，在弹出的对话框中配置编码选项后生成。

字幕样式（字体、字号、颜色等）可在 `config/subtitle_style.yaml` 中配置，字号基于 `dialogue_roi` 高度自动计算。

支持按 speaker（角色名）使用不同的描边颜色，仅对 ASS 和硬字幕生效。在 `config/subtitle_style.yaml` 的 `speaker_styles` 下按角色名配置 `outline_colour` 即可。模糊匹配可处理 OCR 轻微错字。

每条 cache entry 也可以通过 `subtitle_style` 覆盖单段 ASS 样式，例如 `outline_colour`、`font_size`、`font_size_scale`、`position`、`alignment` 等。未填写的字段会自动使用默认样式。

![字幕结果](docs/images/subtitle_result.png)

## 命令行使用

除 GUI 外，pipeline 也可以作为命令行工具直接运行：

```bash
python -m ignite.pipeline \
    --video <path> --config <path> --output-dir <path> \
    --dialogue-presence-mode <marker2|ocr> \
    --skip-translation
```

主要参数：

| 参数 | 说明 |
|------|------|
| `--video` | 输入视频路径 |
| `--config` | 配置文件路径（默认 `config/example_profile.yaml`） |
| `--output-dir` | 输出目录 |
| `--resume` | 从缓存 JSON 产物恢复 |
| `--skip-translation` | 跳过 VLM 翻译，仅打时间轴 |
| `--cache-only` | 仅生成/更新翻译缓存，不生成字幕 |
| `--render-video` | 渲染硬字幕视频 |
| `--translation-cache` | 指定翻译缓存 JSON 路径 |
| `--subtitles-from-cache` | 仅从缓存生成字幕并退出 |
| `--debug` | 启用 debug 产物 |
| `--dialogue-presence-mode` | 对话存在判定/空白段过滤模式：`marker2` / `ocr`（正常运行必填；`--subtitles-from-cache` 不需要） |

若想在命令行复现 GUI 默认的“禁用 OCR、使用 Marker2 判定对话出现”的流程，请配置 `roi.marker_2_roi`、`marker_2.template_paths`，并传入 `--dialogue-presence-mode marker2`。如需使用姓名 OCR 判定对话出现，则传入 `--dialogue-presence-mode ocr`。

## 辅助工具

```bash
# 单张图片 OCR 调试
python tools/debug_rapidocr_single_image.py --image <path> --config <path>

# 归档单个已处理项目
python -m ignite.archive --cache outputs/<名称>/translation_cache_latest.json --dest-root D:/Archive

# 批量归档 outputs 下所有 latest cache
python -m ignite.archive --batch --cache-root outputs --dest-root D:/Archive
```

## 开源协议

本项目基于 GNU General Public License v3.0 协议开源
