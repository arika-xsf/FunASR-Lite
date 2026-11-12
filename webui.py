# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""FunASR 网页调用界面 (Gradio)

模型懒加载: 切换模型时自动卸载旧模型释放显存 (适配 GTX 1060 3GB)
支持: 文件上传 / 麦克风录音 / VAD / 说话人分离(选配)
"""

import gc
import json
import os
import time
import traceback
import zipfile

# --- 固定工作目录到脚本所在目录 ---
# 坑: 从快捷方式或其它目录启动时, 进程 cwd 可能是 C:\Windows\system32。该目录是系统
# 受保护目录, 任何相对路径解析落进去都会 PermissionError (曾导致 Gradio 把下载按钮的
# 空串当路径, 相对到 system32 后 open 报 Permission denied)。这里统一把 cwd 固定住,
# 且必须在任何依赖相对路径的代码之前执行。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# --- ffmpeg 接入 PATH ---
# 关键: 必须在 import torch/funasr 之前。mp3/m4a/aac/wma 等压缩格式 torchaudio 与
# libsndfile 都解不了, FunASR 会兜底 subprocess 拉起 ffmpeg CLI; ffmpeg 不在 PATH 时
# 抛 [WinError 2] 系统找不到指定的文件。imageio-ffmpeg 提供自包含静态二进制。
def _ensure_ffmpeg_on_path() -> str:
    """确保存在名为 ffmpeg(.exe) 的可执行文件并前置其目录到 PATH; 返回路径, 失败返回空串

    坑: imageio-ffmpeg 的二进制名为 ffmpeg-win-x86_64-v7.1.exe, 而 FunASR 用
    subprocess 找字面量 "ffmpeg", shutil.which 匹配不到, 所以必须落地成标准文件名。
    """
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    local = os.path.join(SCRIPT_DIR, "bin", "ffmpeg.exe")
    if os.path.isfile(local):
        os.environ["PATH"] = os.path.dirname(local) + os.pathsep + os.environ.get("PATH", "")
        return local

    try:
        import imageio_ffmpeg

        src = imageio_ffmpeg.get_ffmpeg_exe()
        if src and os.path.isfile(src):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            shutil.copyfile(src, local)
            os.environ["PATH"] = os.path.dirname(local) + os.pathsep + os.environ.get("PATH", "")
            return local
    except Exception:
        pass
    return ""


FFMPEG_EXE = _ensure_ffmpeg_on_path()

# Gradio 环境下 tqdm 进度条刷新 stderr 会 OSError 22 (Windows 已知坑), 全局禁用
os.environ.setdefault("TQDM_DISABLE", "1")

import gradio as gr
import torch

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _ms_cache(repo_id: str) -> str:
    """ModelScope 仓库 ID -> 本机缓存 snapshot 目录; 缓存不存在则退回原 ID (走在线下载)

    注: ModelScope 缓存实际在 <repo>/snapshots/<commit>/ 子目录,
    父目录只有 snapshots/ 文件夹, 没有 configuration.json, 必须指到子目录.
    """
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "modelscope", "models",
        repo_id.replace("/", "--"),
    )
    snap_dir = os.path.join(cache_dir, "snapshots")
    if os.path.isdir(snap_dir):
        subs = [
            d for d in os.listdir(snap_dir)
            if os.path.isdir(os.path.join(snap_dir, d))
        ]
        if subs:
            return os.path.join(snap_dir, subs[0])
    return repo_id


# 模型配置: 名称 -> (model, 附加参数说明)
MODEL_CHOICES = {
    "SenseVoiceSmall (多语种/情感/快)": {
        "model": _ms_cache("iic/SenseVoiceSmall"),
        "vad": _ms_cache("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
        "spk": "cam++",
        "desc": "阿里 SenseVoice 小模型, 支持中/英/日/韩/粤, 带情感与事件标签, 显存占用低",
    },
}

_loaded = {"key": None, "model": None}

# SRT 导出目录
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ms_to_srt_time(ms: int) -> str:
    """毫秒 -> SRT 时间戳 (HH:MM:SS,mmm)"""
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _audio_duration_ms(path: str) -> int:
    """获取音频时长(毫秒), 失败返回 0"""
    try:
        import torchaudio

        info = torchaudio.info(path)
        return int(info.num_frames / info.sample_rate * 1000)
    except Exception:
        return 0


def build_srt(results, model_key: str) -> str:
    """从 generate 结果构建 SRT 字幕文本 (句子级时间戳优先, 否则退化为单条)"""
    lines = []
    idx = 1
    for res in results:
        segs = res.get("sentence_info") or []
        if segs:
            for seg in segs:
                text = (seg.get("sentence") or seg.get("text") or "").strip()
                if model_key.startswith("SenseVoice"):
                    text = rich_transcription_postprocess(text).strip()
                if not text:
                    continue
                start = int(seg.get("start", 0))
                end = int(seg.get("end", start))
                lines.append(f"{idx}\n{ms_to_srt_time(start)} --> {ms_to_srt_time(end)}\n{text}\n")
                idx += 1
        else:
            # 无句子级时间戳 (如 Nano/无 VAD): 用词级 timestamp 或整段时长兜底
            text = res.get("text", "")
            if model_key.startswith("SenseVoice"):
                text = rich_transcription_postprocess(text).strip()
            text = text.strip()
            if not text:
                continue
            ts = res.get("timestamp") or []
            if ts:
                start, end = int(ts[0][0]), int(ts[-1][1])
            else:
                start, end = 0, _audio_duration_ms(res.get("key") or "")
            lines.append(f"{idx}\n{ms_to_srt_time(start)} --> {ms_to_srt_time(end)}\n{text}\n")
            idx += 1
    return "\n".join(lines)


def _out_name(audio_path: str, ext: str) -> str:
    """生成唯一输出文件名: 保留原音频名便于识别, 追加时间戳+随机串避免同名覆盖。

    重要性: gr.DownloadButton 直接服务 outputs/ 下的原始文件 (非内容寻址缓存)。
    若文件名可复用, 连续识别同一音频时第二次写入会覆盖正在被下载服务中的文件,
    导致 stat 大小 < 实际字节 -> h11 'Too much data for declared Content-Length'。
    """
    import random

    base = os.path.splitext(os.path.basename(str(audio_path)))[0] or "output"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rnd = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=4))
    return os.path.join(OUTPUT_DIR, f"{base}__{stamp}_{rnd}{ext}")


def save_srt(srt_text: str, audio_path: str) -> str:
    """SRT 文本写入文件, 返回路径; 空文本返回空串"""
    if not srt_text.strip():
        return ""
    path = _out_name(audio_path, ".srt")
    with open(path, "w", encoding="utf-8-sig") as f:  # BOM: Windows 播放器兼容
        f.write(srt_text)
    return path


def save_txt(text: str, audio_path: str) -> str:
    """纯文本写入文件, 返回路径; 空文本返回空串"""
    if not text.strip():
        return ""
    path = _out_name(audio_path, ".txt")
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(text)
    return path


def _dl(path):
    """下载按钮值: 空路径返回 None (禁用按钮), 避免 Gradio 把空串当成 cwd 目录去哈希缓存 -> PermissionError"""
    return path if path else None


def vram_info() -> str:
    if not torch.cuda.is_available():
        return "CPU 模式 (未检测到 CUDA)"
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024**3
    return f"{torch.cuda.get_device_name(0)} | 显存 {used:.2f} / {total / 1024**3:.2f} GB"


def load_model(model_key: str, use_vad: bool, use_spk: bool):
    """加载模型; 切换时卸载旧模型释放显存"""
    if _loaded["key"] == model_key and _loaded["model"] is not None:
        return f"模型已就绪: {model_key}"

    # 卸载旧模型
    unload_model()

    cfg = MODEL_CHOICES[model_key]
    # 模型可指定 device (如 Nano 用 CPU), 否则跟随全局 DEVICE
    # disable_pbar=True: Gradio 环境下 tqdm 刷新 stderr 会 OSError 22 (Windows 已知坑)
    kwargs = {
        "model": cfg["model"],
        "device": cfg.get("device", DEVICE),
        "disable_pbar": True,  # Gradio 环境下 tqdm 刷新 stderr 会 OSError 22
        "disable_update": True,  # 跳过每次加载时的联网版本检查, 加快加载并避免离线白等
    }
    if use_vad and cfg.get("vad"):
        kwargs["vad_model"] = cfg["vad"]
    if use_spk and cfg.get("spk"):
        kwargs["spk_model"] = cfg["spk"]
    if cfg.get("punc"):
        kwargs["punc_model"] = cfg["punc"]

    try:
        _loaded["model"] = AutoModel(**kwargs)
        _loaded["key"] = model_key
        return f"已加载: {model_key}\n{device_info(cfg)}"
    except Exception as e:
        _loaded["key"] = None
        return f"加载失败: {e}\n{traceback.format_exc()[-500:]}"


def device_info(cfg=None) -> str:
    """显示当前运行设备: CPU 模型显示内存, GPU 模型显示显存"""
    if cfg is not None and cfg.get("device") == "cpu":
        try:
            import psutil

            m = psutil.virtual_memory()
            return f"CPU | 内存 {m.used / 1024**3:.1f} / {m.total / 1024**3:.1f} GB"
        except Exception:
            return "CPU 推理"
    return vram_info()


def unload_model():
    if _loaded["model"] is not None:
        del _loaded["model"]
        _loaded["model"] = None
        _loaded["key"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return "已卸载模型"


def transcribe(audio_path, model_key, use_vad, use_spk):
    if not audio_path:
        return "请先上传或录制音频", "", "", None, None
    if model_key not in MODEL_CHOICES:
        return "请选择模型", "", "", None, None

    # 模型未加载则先加载
    if _loaded["key"] != model_key or _loaded["model"] is None:
        msg = load_model(model_key, use_vad, use_spk)
        if _loaded["model"] is None:
            return msg, "", "", None, None

    t0 = time.time()
    try:
        kwargs = {
            "input": audio_path,
            "batch_size_s": 300,
            "sentence_timestamp": True,  # 逐句时间戳 -> SRT 字幕
        }
        cfg = MODEL_CHOICES[model_key]
        if model_key.startswith("SenseVoice"):
            kwargs["language"] = "auto"
            kwargs["use_itn"] = True
            kwargs["merge_vad"] = bool(use_vad)
            kwargs["merge_length_s"] = 15

        results = _loaded["model"].generate(**kwargs)

        texts = []
        detail_lines = []
        for res in results:
            text = res.get("text", "")
            if model_key.startswith("SenseVoice"):
                text = rich_transcription_postprocess(text)
            texts.append(text)
            info = {k: v for k, v in res.items() if k != "text"}
            if info:
                detail_lines.append(str(info))

        elapsed = time.time() - t0
        result_text = "\n".join(texts)
        srt_text = build_srt(results, model_key)
        srt_path = save_srt(srt_text, audio_path)
        txt_path = save_txt(result_text, audio_path)
        detail = f"耗时 {elapsed:.2f}s | {device_info(cfg)}"
        if srt_path:
            detail += f"\nSRT 已导出: {srt_path}"
        if txt_path:
            detail += f"\nTXT 已导出: {txt_path}"
        if detail_lines:
            detail += "\n" + "\n".join(detail_lines[:10])
        return result_text, srt_text, detail, _dl(srt_path), _dl(txt_path)
    except torch.cuda.OutOfMemoryError:
        unload_model()
        return (
            "显存不足 (OOM)! 已自动卸载模型。建议关闭说话人分离后重试。",
            "",
            vram_info(),
            None,
            None,
        )
    except Exception as e:
        msg = f"识别失败: {e}"
        if isinstance(e, FileNotFoundError) and not FFMPEG_EXE:
            msg += (
                "\n\n原因: 系统未找到 ffmpeg。当前音频为压缩格式 (mp3/m4a/aac/wma), "
                "torchaudio 与 libsndfile 均无法解码。\n"
                "解决: 在 venv 中安装 imageio-ffmpeg (pip install imageio-ffmpeg), "
                "或把音频转成 wav/flac 再识别。"
            )
        return msg, "", traceback.format_exc()[-800:], None, None


# --- 批量转录 ---

ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus"}


def batch_transcribe(file_list, model_key, use_vad, use_spk):
    """批量转录多个音频文件: 逐个推理, 实时推送进度, SRT+TXT 打包 zip 下载 (generator)"""
    if not file_list:
        yield "请先上传音频文件", "", "", None
        return
    if model_key not in MODEL_CHOICES:
        yield "请选择模型", "", "", None
        return

    # 模型未加载则先加载
    if _loaded["key"] != model_key or _loaded["model"] is None:
        msg = load_model(model_key, use_vad, use_spk)
        if _loaded["model"] is None:
            yield msg, "", "", None
            return

    # 过滤音频文件
    audio_files = [
        f for f in file_list
        if os.path.splitext(str(f))[1].lower() in ALLOWED_EXTS
    ]
    if not audio_files:
        yield "未找到支持的音频文件 (wav/mp3/flac/m4a/ogg/aac/wma/opus)", "", "", None
        return

    cfg = MODEL_CHOICES[model_key]
    all_texts = []
    all_details = []
    srt_files = []
    txt_files = []
    total_t0 = time.time()
    total = len(audio_files)

    def progress_line(done: int) -> str:
        """当前进度: 已完成数 / 总数 + 进度条"""
        pct = done / total * 100
        bar_len = 20
        filled = int(bar_len * done / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"进度: {done}/{total} ({pct:.0f}%) {bar}"

    for i, audio_path in enumerate(audio_files):
        fname = os.path.basename(str(audio_path))
        all_texts.append(f"[{i + 1}/{total}] {fname}")

        # 推送「正在处理」状态
        yield (
            "\n".join(all_texts) + f"\n⏳ 正在识别: {fname}",
            progress_line(i),
            "\n".join(all_details) + f"\n⏳ [{i + 1}/{total}] 正在处理: {fname}",
            None,
        )

        t0 = time.time()
        try:
            kwargs = {
                "input": str(audio_path),
                "batch_size_s": 300,
                "sentence_timestamp": True,
            }
            if model_key.startswith("SenseVoice"):
                kwargs["language"] = "auto"
                kwargs["use_itn"] = True
                kwargs["merge_vad"] = bool(use_vad)
                kwargs["merge_length_s"] = 15

            results = _loaded["model"].generate(**kwargs)

            texts = []
            for res in results:
                text = res.get("text", "")
                if model_key.startswith("SenseVoice"):
                    text = rich_transcription_postprocess(text)
                texts.append(text)

            elapsed = time.time() - t0
            file_text = "\n".join(texts)
            all_texts.append(file_text)

            # SRT + TXT
            srt_text = build_srt(results, model_key)
            srt_path = save_srt(srt_text, str(audio_path))
            if srt_path:
                srt_files.append(srt_path)
            txt_path = save_txt(file_text, str(audio_path))
            if txt_path:
                txt_files.append(txt_path)

            all_details.append(f"✅ [{i + 1}/{total}] {fname}: {elapsed:.2f}s OK")
        except torch.cuda.OutOfMemoryError:
            unload_model()
            all_texts.append("显存不足 (OOM)! 已自动卸载模型, 后续文件跳过。")
            all_details.append(f"❌ [{i + 1}/{total}] {fname}: OOM, 已中止")
            # 推送 OOM 状态后终止
            yield (
                "\n".join(all_texts),
                progress_line(i),
                "\n".join(all_details),
                None,
            )
            return
        except Exception as e:
            all_texts.append(f"识别失败: {e}")
            all_details.append(f"❌ [{i + 1}/{total}] {fname}: 失败 - {e}")

        all_texts.append("")  # 空行分隔

        # 每完成一个文件推送一次进度
        yield (
            "\n".join(all_texts),
            progress_line(i + 1),
            "\n".join(all_details),
            None,
        )

    total_elapsed = time.time() - total_t0
    all_details.append(
        f"\n🎉 全部完成: {total} 个文件, 耗时 {total_elapsed:.1f}s | {device_info(cfg)}"
    )

    # SRT + TXT 打包 zip
    download_path = None
    srt_summary = ""
    all_export_files = srt_files + txt_files
    if all_export_files:
        if len(all_export_files) == 1:
            download_path = all_export_files[0]
            srt_summary = f"已生成 {len(srt_files)} 个 SRT + {len(txt_files)} 个 TXT 文件"
        else:
            zip_name = f"batch_results_{int(time.time())}.zip"
            zip_path = os.path.join(OUTPUT_DIR, zip_name)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in all_export_files:
                    zf.write(f, os.path.basename(f))
            download_path = zip_path
            srt_summary = (
                f"已生成 {len(srt_files)} 个 SRT + {len(txt_files)} 个 TXT, 打包为 zip"
            )
    else:
        srt_summary = "无文件生成"

    result_text = "\n".join(all_texts)
    detail_text = "\n".join(all_details)
    yield result_text, srt_summary, detail_text, _dl(download_path)


# (LLM 关键词/摘要分析功能已移除 — 2026-10-02)


def build_app():
    # analytics_enabled=False: 关闭 Gradio 启动时的联网遥测/版本检查 (api.gradio.app),
    # 本地离线工具无需外联, 也省掉每次启动的一次网络往返
    with gr.Blocks(title="FunASR-Lite", analytics_enabled=False) as demo:
        gr.Markdown(f"# FunASR 语音识别 (GPU WebUI)\n设备: **{DEVICE.upper()}** | {vram_info()}")

        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    label="模型",
                    choices=list(MODEL_CHOICES.keys()),
                    value=list(MODEL_CHOICES.keys())[0],
                )
                use_vad = gr.Checkbox(label="启用 VAD (长音频切分)", value=True)
                use_spk = gr.Checkbox(label="说话人分离 (cam++)", value=False)
                load_btn = gr.Button("加载模型")
                unload_btn = gr.Button("卸载模型 (释放显存)")
                model_status = gr.Textbox(label="模型状态", lines=4, interactive=False)
                gr.Markdown(
                    "说明: 首次使用会自动从 ModelScope 下载模型 (SenseVoiceSmall 约 230MB)。\n\n"
                    "GPU 推理 (3GB 显存, 切换/卸载时自动释放)。\n"
                    "支持中/英/日/韩/粤, 带情感与事件标签, 识别后自动生成 SRT 字幕与 TXT 文本。\n"
                    "批量转录: 上传多个音频文件, 逐个识别并打包 SRT+TXT 下载。\n\n"
                    f"ffmpeg: **{'已就绪 (' + FFMPEG_EXE + ')' if FFMPEG_EXE else '未找到'}**"
                    "  (mp3/m4a/aac/wma 等压缩格式需要它解码)"
                )
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("单个识别"):
                        audio_in = gr.Audio(
                            label="音频 (上传或麦克风录音)",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )
                        run_btn = gr.Button("开始识别", variant="primary")
                        text_out = gr.Textbox(label="识别结果", lines=8)
                        srt_out = gr.Textbox(label="SRT 字幕", lines=6, interactive=False)
                        with gr.Row():
                            srt_download = gr.DownloadButton(label="下载 SRT 文件", value=None)
                            txt_download = gr.DownloadButton(label="下载 TXT 文件", value=None)
                        detail_out = gr.Textbox(label="详细信息", lines=4, interactive=False)

                    with gr.Tab("批量转录"):
                        files_in = gr.Files(
                            label="批量上传音频文件",
                            file_types=[".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus"],
                        )
                        batch_btn = gr.Button("批量识别", variant="primary")
                        batch_progress = gr.Textbox(label="处理进度", lines=2, interactive=False)
                        batch_text = gr.Textbox(label="识别结果 (全部文件)", lines=14)
                        with gr.Row():
                            batch_download = gr.DownloadButton(label="下载全部 SRT+TXT (zip)", value=None)
                        batch_detail = gr.Textbox(label="详细信息 (逐文件耗时)", lines=6, interactive=False)

        model_dd.change(
            fn=lambda: "模型未加载, 点【加载/切换模型】或直接识别",
            outputs=model_status,
        )
        load_btn.click(fn=load_model, inputs=[model_dd, use_vad, use_spk], outputs=model_status)
        unload_btn.click(fn=unload_model, outputs=model_status)
        run_btn.click(
            fn=transcribe,
            inputs=[audio_in, model_dd, use_vad, use_spk],
            outputs=[text_out, srt_out, detail_out, srt_download, txt_download],
        )
        batch_btn.click(
            fn=batch_transcribe,
            inputs=[files_in, model_dd, use_vad, use_spk],
            outputs=[batch_text, batch_progress, batch_detail, batch_download],
        )
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, inbrowser=True)
