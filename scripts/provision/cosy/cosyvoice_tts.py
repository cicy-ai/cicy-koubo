#!/usr/bin/env python3
"""CosyVoice2 zero-shot 声音克隆 TTS(在 Colab 上跑)。
用法(用 /content/cosy/env/bin/python 调):
  python cosyvoice_tts.py --ref ref.wav --ref-text "参考音频转写" --text "要合成的文字" --out out.wav
--ref-text 是参考音频的转写(prompt text);留空则退化为跨语种/指令模式效果较差,建议提供。
详细生成日志同时打到 stderr 和 /content/cosy/last_tts.log(方便 tail -f 排查 voice clone 问题)。
"""
import argparse
import sys
import time

REPO = "/content/cosy/CosyVoice"
sys.path.append(REPO)
sys.path.append(REPO + "/third_party/Matcha-TTS")

LOG_PATH = "/content/cosy/last_tts.log"
_logf = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    global _logf
    try:
        if _logf is None:
            _logf = open(LOG_PATH, "w", encoding="utf-8")
        _logf.write(line + "\n")
        _logf.flush()
    except Exception:
        pass


from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402
import torchaudio  # noqa: E402
import base64  # noqa: E402
import re  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="参考音频 wav(传路径,本版本内部自行加载)")
ap.add_argument("--ref-text", default="", help="参考音频的转写文本")
ap.add_argument("--text", default="", help="要合成的目标文本")
ap.add_argument("--ref-text-b64", default="", help="ref-text 的 base64(避免引号/编码问题)")
ap.add_argument("--text-b64", default="", help="text 的 base64")
ap.add_argument("--speed", type=float, default=1.15, help="语速 0.5-1.5")
ap.add_argument("--whole", action="store_true", help="整段生成(不分句,语气连贯;超长文本可能截断)")
ap.add_argument("--out", required=True, help="输出 wav 路径")
a = ap.parse_args()

# base64 优先:彻底规避 shell 嵌套引号破坏中文标点导致分句失效
a.text = base64.b64decode(a.text_b64).decode("utf-8") if a.text_b64 else a.text
a.ref_text = base64.b64decode(a.ref_text_b64).decode("utf-8") if a.ref_text_b64 else a.ref_text
if not a.text:
    raise SystemExit("no --text/--text-b64")

log("==== TTS 开始 ====")
log(f"参考音频: {a.ref}")
log(f"参考转写(ref_text) [{len(a.ref_text)}字]: {a.ref_text!r}")
log(f"目标文案(text) [{len(a.text)}字]: {a.text!r}")
log(f"语速: {a.speed}")

# 参考音频信息
try:
    import soundfile as sf
    info = sf.info(a.ref)
    log(f"参考音频规格: {info.duration:.2f}s / {info.samplerate}Hz / {info.channels}ch")
    if info.duration > 12:
        log(f"⚠️ 参考音频 {info.duration:.1f}s 偏长,可能挤占上下文;建议 <10s")
    if info.duration < 3:
        log(f"⚠️ 参考音频仅 {info.duration:.1f}s 偏短,音色相似度可能不足")
except Exception as e:  # noqa: BLE001
    log(f"读参考音频信息失败: {e}")

if not a.ref_text.strip():
    log("⚠️ ref_text 为空:未提供参考音频转写,克隆相似度会明显下降")

t0 = time.time()
log("加载 CosyVoice2 模型…")
m = CosyVoice2(REPO + "/pretrained_models/CosyVoice2-0.5B",
               load_jit=False, load_trt=False, fp16=False)
log(f"模型就绪(sr={m.sample_rate}),耗时 {time.time() - t0:.1f}s")

# 按句末标点切成短句,逐句合成再拼接:CosyVoice 自回归对长文本会提前截断,分句可规避
# 二次切分:AI 改写的文案常通篇无句号只有空格,超长段落按 空格/逗号 聚合到 ~40 字
MAX_SEG = 60


def _split(text):
    prim = [s.strip() for s in re.split(r"(?<=[。！？；;!?\n])", text) if s.strip()]
    if not prim:
        prim = [text.strip()]
    out = []
    for s in prim:
        if len(s) <= MAX_SEG:
            out.append(s)
            continue
        cur = ""
        for p in re.split(r"[ 　，,、:：]+", s):
            if not p:
                continue
            if cur and len(cur) + len(p) + 1 > 45:
                out.append(cur)
                cur = p
            else:
                cur = (cur + "，" + p) if cur else p
        if cur:
            out.append(cur)
    return out


segments = [a.text.strip()] if a.whole else _split(a.text)
if a.whole:
    log("整段生成模式(--whole):不分句")
log(f"分句结果: {len(segments)} 段")
for i, s in enumerate(segments):
    log(f"  段{i + 1} [{len(s.strip())}字]: {s.strip()!r}")

def _fade(x, sr, ms=8):
    """段首尾短淡入淡出:消除硬拼接接缝处的爆点(啪声)。"""
    n = min(int(sr * ms / 1000), x.shape[1] // 4)
    if n > 0:
        ramp = torch.linspace(0.0, 1.0, n)
        x = x.clone()
        x[:, :n] *= ramp
        x[:, -n:] *= ramp.flip(0)
    return x


pause = torch.zeros(1, int(m.sample_rate * 0.15))  # 句间 0.15s 停顿
chunks = []
for idx, seg in enumerate(segments):
    st = time.time()
    seg_len = 0.0
    n_yield = 0
    for j in m.inference_zero_shot(seg.strip(), a.ref_text, a.ref, stream=False, speed=a.speed):
        chunks.append(_fade(j["tts_speech"], m.sample_rate))
        seg_len += j["tts_speech"].shape[1] / m.sample_rate
        n_yield += 1
    log(f"段{idx + 1} 合成完成: {seg_len:.2f}s 音频 / {n_yield} 次yield / 耗时 {time.time() - st:.1f}s")
    if idx < len(segments) - 1:
        chunks.append(pause)

audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
# 60Hz 高通:滤掉低频嗡声/风噪(参考音频底噪常带低频)
try:
    import torchaudio.functional as AF
    audio = AF.highpass_biquad(audio, m.sample_rate, cutoff_freq=60.0)
    log("已应用 60Hz 高通降噪")
except Exception as e:  # noqa: BLE001
    log(f"高通跳过: {e}")
# 峰值归一到 -1.5dB(0.84):模型输出峰值常超 1.0,直接保存会硬削波(炸音)
peak = float(audio.abs().max())
if peak > 0:
    audio = audio * (0.84 / max(peak, 0.84))
log(f"峰值归一: 原峰值 {peak:.3f} → {'压到 0.84' if peak > 0.84 else '未超,保持'}")
total = audio.shape[1] / m.sample_rate
torchaudio.save(a.out, audio, m.sample_rate)
log(f"==== 完成: 总时长 {total:.2f}s / {len(segments)}段 / 全程 {time.time() - t0:.1f}s ====")
log(f"输出: {a.out}")
print(f"OK out={a.out} sr={m.sample_rate} segs={len(segments)} dur={total:.2f}")
