# FunASR-Lite · 本地语音识别 GPU WebUI

> 基于 [FunASR](https://github.com/modelscope/FunASR) 的**轻量、低显存友好本地语音识别 Web 界面**：双击即用、完全离线推理、支持批量转写与字幕导出。

---

## ✨ 特色

- **本地优先 / 隐私安全**：模型在本地 GPU 推理，音频不上传；关闭 Gradio 联网遥测，后台不外联。仅首次需联网下载依赖与模型。
- **开箱即用**：路径无关启动器，首次自动创建虚拟环境并安装依赖，**双击 `webui.bat` 即可**，无需手动配环境。
- **全系 N 卡 + 无 N 卡都能跑**：默认 CUDA 12.8 wheel 覆盖 **GTX 10 系 ~ RTX 50 系**；**3GB 显存实测可跑**（GTX 1060 3GB），切换 / 卸载模型即释放显存。无 N 卡时启动器自动装 CPU 轻量版并回退 CPU 推理。
- **多语种 + 情感标签**：SenseVoiceSmall 支持 中 / 英 / 日 / 韩 / 粤语，并输出**情感与事件标签**。
- **长音频 VAD + 说话人分离**：内置 fsmn-vad 按停顿自动切句；可选 cam++ 区分不同说话人。
- **单文件 + 批量**：支持上传音频或**麦克风录音**；批量模式一次多文件、实时进度，结果**打包下载 SRT+TXT (zip)**。
- **字幕导出**：自动生成带时间戳的 **SRT 字幕** 与 **TXT 文本**，一键下载。
- **ffmpeg 自动配置**：从 imageio-ffmpeg 自动抽取，无需手动安装即可解码 mp3 / m4a / aac / wma。

---

## 🚀 快速开始

> 🟢 **普通用户无需 Git、无需克隆源码**：到 **[Releases](https://github.com/arika-xsf/FunASR-Lite/releases)** 下载 `FunASR-Lite-source-vX.X.X.zip` → 解压 → **双击 `webui.bat`** 即可。只有想研究 / 修改源码才需要 `git clone`。

### 方式一 · 下载 ZIP（推荐）

1. 打开 [Releases](https://github.com/arika-xsf/FunASR-Lite/releases) 页面，下载最新的 `FunASR-Lite-source-vX.X.X.zip`；
2. 解压到任意目录（路径随意，不要求固定位置）；
3. **双击 `webui.bat`**，浏览器自动打开 **http://127.0.0.1:7860**。

### 方式二 · git clone（仅开发者）

```bash
git clone https://github.com/arika-xsf/FunASR-Lite.git FunASR-Lite
cd FunASR-Lite
# 双击 webui.bat
```

> **首次运行会做什么**（仅一次，之后秒开）：启动器先用 `nvidia-smi` 探测显卡 —— **有 N 卡装 GPU 版**（下载约 5–8GB 的 PyTorch CUDA 12.8），**无 N 卡自动装 CPU 轻量版**（约 200MB）；首次识别时再自动下载模型（SenseVoiceSmall 约数百 MB）。之后每次双击 `webui.bat` 即可，不再下载。
>
> 手动指定：`webui.bat cpu` 强制 CPU 轻量版，`webui.bat gpu` 强制 GPU 版（适用于显卡存在但 `nvidia-smi` 不在 PATH 导致误判的情况）。

**可选 · 手动安装（不使用启动器）**

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt     # 无 N 卡改用 requirements-cpu.txt
python webui.py
```

---

## 📋 环境要求 / 硬件配置

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10 / 11（64 位） |
| Python | 3.10 或 3.11，安装时勾选 **Add Python to PATH** |
| 显卡（推荐） | NVIDIA 显卡 + **较新驱动（CUDA 12.8 需 R570 以上）**；显存 ≥ 3GB（GTX 1060 3GB 实测可用）。默认 `torch 2.7.1+cu128` 一个 wheel 覆盖 **sm_61（GTX 10 系）~ sm_120（RTX 50 系）**，老卡新卡通吃；CUDA 运行时已打包，**无需装 CUDA Toolkit** |
| 无 N 卡 | 启动器自动改用 CPU 轻量依赖并回退 **CPU 推理**（速度较慢；依赖下载量从 5–8GB 降到约 200MB） |
| 内存 | 建议 ≥ 8GB |
| 网络 | 首次运行需联网（下载依赖与模型，仅此一次） |

**磁盘占用（首次安装完成后）**

| 内容 | 占用 | 说明 |
|---|---|---|
| `venv/`（GPU 版，含 PyTorch cu128） | 约 **5–8 GB** | 首次自动生成，不入库 |
| `venv/`（CPU 轻量版） | 约 **0.5–1 GB** | 无 N 卡时自动选用，torch CPU wheel 仅约 200MB |
| 模型缓存 `~/.cache/modelscope/` | 约 **900 MB** | 首次识别自动下载，不在项目目录内 |
| 源码 + `bin/ffmpeg.exe` | 数十 MB | 即解压出来的内容 |
| `outputs/` | 视使用量 | 识别结果，不入库 |

> GPU 版合计约需 **6–9 GB** 可用空间；CPU 轻量版约 **1.5 GB**（均含模型缓存）。

---

## 📦 安装说明

普通用户走完「快速开始」即可，本节补充几个前置细节。

### 1. 安装 Python
- 下载 [Python 3.10 / 3.11](https://www.python.org/downloads/windows/)（推荐 3.10）；
- **务必勾选 “Add Python to PATH”**（否则双击 bat 会报 “python 不是内部或外部命令”）；
- 验证：命令行 `python --version` 能显示版本号。

### 2. 获取代码
见「快速开始」——普通用户下 ZIP，开发者 `git clone`。

### 3. NVIDIA 驱动（仅 GPU 用户）
需 **R570 及以上**驱动以支持 CUDA 12.8（`nvidia-smi` 能看到显卡信息即正常）；无需 CUDA Toolkit。**无 N 卡可跳过本节**，启动器会自动装 CPU 轻量版。

### 4. FFmpeg（通常无需手动安装）
本项目**已内置自动配置**：首次启动时会从 `imageio-ffmpeg`（已写入 `requirements.txt`）抽取 ffmpeg 静态二进制，自动落地到项目内 `bin/ffmpeg.exe` 并加入运行环境，mp3 / m4a / aac / wma 直接可解码。

仅当自动抽取失败（完全离线、被安全软件拦截等极少见情况）才需手动处理：

1. 打开 <https://www.gyan.dev/ffmpeg/builds/> ，下载 **Windows builds (release essentials)** 的 ZIP；
2. 解压后把其中 `bin/ffmpeg.exe` 复制到项目目录下的 `bin/` 文件夹（即 `FunASR-Lite/bin/ffmpeg.exe`，`bin/` 不存在就新建）；
3. 或把 `ffmpeg.exe` 所在目录加入系统 **PATH**。

> 重新双击 `webui.bat` 即可；左侧「模型状态」显示 `ffmpeg: 已就绪` 说明正常。

---

## 🖥️ 使用说明

界面左右两栏：

- **左侧**：模型选择 · VAD 开关 · 说话人分离开关 · 加载 / 卸载模型 · 模型状态；
- **右侧两个标签页**：
  - **单个识别**：上传音频或直接用麦克风录音 →【开始识别】→ 识别文本 + SRT 字幕，可分别下载 SRT / TXT；
  - **批量转录**：一次上传多个音频 →【批量识别】→ 实时进度 + 全部结果，一键打包下载 SRT+TXT (zip)，并查看逐文件耗时。

> 默认开启 VAD（长音频自动切句）；需要区分说话人勾选【说话人分离】；显存紧张时点【卸载模型】立即释放显存。

---

## 📁 目录结构

```
FunASR-Lite/
├── webui.py                # 主程序 (Gradio 界面 + 推理逻辑)
├── webui.bat               # 路径无关启动器 (自动探测 GPU/CPU, 首次自动建环境)
├── requirements.txt        # GPU 版依赖 (PyTorch cu128, 覆盖老卡到 RTX 50)
├── requirements-cpu.txt    # CPU 轻量版依赖 (无 N 卡时自动选用)
├── README.md
├── LICENSE                 # MIT 许可证
├── .gitignore
├── venv/                   # 虚拟环境 (首次自动生成, 不入库)
├── bin/                    # ffmpeg 二进制 (自动抽取, 不入库)
└── outputs/                # 识别结果 (自动生成, 不入库)
```

---

## 🔄 更换目录 / 迁移到其它机器

应用层（`webui.py` 用 `SCRIPT_DIR`、`webui.bat` 用 `%~dp0`）**不写死路径**，但有一处例外（见下）。

### 一、同机更换目录

项目里**唯一**写死绝对路径的，是 funasr「源码可编辑安装（editable）」留下的映射文件 `venv/Lib/site-packages/__editable___funasr_1_4_2_finder.py`，其第 9 行形如：

```python
MAPPING = {'funasr': 'D:\\Temp\\FunASR\\FunASR\\funasr'}
```

移动文件夹后旧前缀失效 → `import funasr` 失败。两种解法任选：

- **方法 A（推荐，保留 venv，立即生效）**：移动后，用记事本把该行旧的盘符 / 目录前缀（`D:\Temp\FunASR`）改成新位置（如 `E:\FunASR-Lite`），后面的 `\FunASR\funasr` 不变，再双击 `webui.bat`。
- **方法 B（干净但慢）**：移动后删掉整个 `venv/`，双击 `webui.bat` 自动重建并重装依赖（重下约 5GB PyTorch，需联网）。

> 模型缓存在用户目录 `~/.cache/modelscope/`（约 900MB），**不在项目目录内**；同机移动不影响，无需重下模型。

### 二、迁移到其它机器

📌 **请使用「全新安装」**：在目标机从头搭建环境，**切勿复制本机的 `venv/`、`bin/`、`outputs/` 等运行产物**——虚拟环境不可移植（`pyvenv.cfg` 的 `home` 指向本机 Python，torch CUDA 二进制与系统 DLL 也绑定当前机器）。

1. 把**干净源码包**给对方：Release 里的 `FunASR-Lite-source-vX.X.X.zip`（已排除 `venv/` `bin/` `outputs/` `FunASR/`），或自己用 `git archive` 导出；
2. 对方解压到任意目录，**双击 `webui.bat`**，启动器会在目标机全新创建 `venv` 并安装全部依赖；
3. 目标机需满足上文「环境要求」：Windows 10/11 + N 卡 + 较新驱动 + **预装 Python 3.10/3.11 并加 PATH**。

> **模型**：目标机首次识别会自动联网下载（约 900MB）。若完全离线，把本机 `~/.cache/modelscope/` 整目录拷过去；因路径含用户名，目标机用户名不同时可用环境变量指过去：
> ```bat
> set MODELSCOPE_CACHE=D:\modelscope_cache
> ```

---

## ❓ 常见问题

- **双击没反应 / “python 不是内部或外部命令”**：未装 Python 或未加 PATH，装 3.10/3.11 并勾选 *Add Python to PATH*。
- **首次启动特别慢**：正在装 PyTorch CUDA 版与下载模型，属正常，仅一次。
- **CUDA / 显存报错**：会自动回退 CPU；需要 GPU 请更新 N 卡驱动到 R570 以上。
- **有 N 卡却装成了 CPU 版**：启动器按 `nvidia-smi` 是否在 PATH 判断；删掉 `venv/` 后运行 `webui.bat gpu` 重装即可。
- **端口 7860 被占用**：`python webui.py --port 7861` 换端口。

---

## ⚖️ 与同类项目对比 · 适用场景

FunASR 本身生态里 WebUI 方案不少，本项目**不是"又造一个"**，而是把目标钉在了一个常被忽略的场景：**Windows + 低显存单卡 + 零配置上手**。下面是客观对比，便于你选型。

| 维度 | 官方仓库自带 WebUI | 社区一键封装 (各类 funasr-webui) | **FunASR-Lite** |
|---|---|---|---|
| 维护方 | 阿里巴巴 ModelScope | 个人 / 团队（质量参差） | 本仓库 |
| 目标系统 | **Linux** 脚本 (`bash start_app.sh`) | 多为 **Linux** 脚本 | **Windows 优先**，双击 `.bat` |
| 启动器 | 手动建 venv / 装依赖 | 一键脚本，但常内置大体积 `models/` | **路径无关**，自动识别 GPU/CPU 并建环境装依赖 |
| 低显存适配 | 普遍要求 8GB+ | 视项目而定 | **3GB 显卡实测可跑**，自动释放显存 |
| Windows 实测坑 | 未覆盖 | 未覆盖 | 已趟平（下载权限 / Content-Length / ffmpeg 缺失 / bat 编码 等） |
| 流式识别 | 部分支持 | 多数支持 | **不含**（离线文件 / 批量为主） |
| 热词 / 识别历史 | 官方支持 | 多数支持 | **不含** |
| 离线隐私 | ✅ | ✅ | ✅ |
| 功能面 | 全面 | 较全 | 单文件+麦克风 / 批量 / SRT·TXT 导出 / 说话人分离 |

> 注：官方仓库（`modelscope/FunASR`）还提供一个 `examples/openai_api/gradio_app.py` 的 Gradio 客户端示例，但它是**纯 HTTP 客户端**——自身不加载模型，需先单独起一个 FastAPI 后端服务再连上去，比本项目多一层部署。

### 什么时候选 FunASR-Lite
- ✅ 你用 **Windows**，不想折腾 WSL / Linux 脚本；
- ✅ 显卡显存紧张（如 GTX 1060 3GB / 1050 Ti 4GB 等入门卡）；
- ✅ 想**下载即跑**：双击 `webui.bat`，首次自动装环境，之后秒开；
- ✅ 主要做**录音 / 会议音频的离线转写与字幕**，不需要实时字幕流。

### 什么时候选官方 / 社区方案
- 需要**实时流式识别**或**热词增强**；
- 部署在 **Linux 服务器**做长期服务（官方/社区在 Linux 上更成熟）；
- 需要**识别历史管理**等附加模块。

---

## 🙏 致谢与许可

**本项目代码采用 [MIT License](LICENSE)**（Copyright (c) 2026 arika-xsf）：可自由使用、修改、商用，只需保留版权与许可声明。

| 组成 | 来源 | 许可 |
|---|---|---|
| FunASR 工具包源码 | [modelscope/FunASR](https://github.com/modelscope/FunASR)（阿里巴巴 ModelScope） | **MIT** |
| Web 界面 | [Gradio](https://www.gradio.app/) | Apache-2.0 |
| 模型权重 SenseVoiceSmall / fsmn-vad / cam++ | ModelScope 各模型卡 | **单独授权**（SenseVoiceSmall 标注 Apache-2.0） |
| 其它 Python 依赖 | PyPI | 各自许可 |

> 上游 FunASR 明确声明「源码 MIT，**预训练模型权重单独授权**」。本仓库与 Release ZIP **不分发任何模型文件**，模型由首次运行时自动下载；商用前请自行核对对应模型卡的许可条款。
