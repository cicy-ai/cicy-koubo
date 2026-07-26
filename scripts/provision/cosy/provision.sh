#!/bin/bash
# CosyVoice2-0.5B self-provision for a Google Colab GPU runtime (voice clone TTS).
# 与 MuseTalk 独立的 micromamba 环境(torch 版本不同),装在 /content/cosy/。
# 由 Colab 引导 cell 在 MuseTalk 之后后台拉起:
#   curl -fsSL .../cosyvoice-provision.sh > /content/cosy/provision.sh
#   nohup bash /content/cosy/provision.sh > /content/cosy/provision.log 2>&1 &
# 成功后写 /content/cosy/COSY_READY,供编排方轮询。
set -uo pipefail
export LD_LIBRARY_PATH="/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
# 强制无头 matplotlib 后端(从笔记本 cell 启动会继承 Jupyter 的 inline 后端导致崩)
export MPLBACKEND=Agg

WORK=/content/cosy
ENV=$WORK/env
REPO=$WORK/CosyVoice
RAW=https://raw.githubusercontent.com/cicy-ai/cicy-tools/main
export MAMBA_ROOT_PREFIX=$WORK/mamba

log(){ echo "=== [$(date +%H:%M:%S)] $*"; }
die(){ echo "!!! $*" >&2; exit 1; }

rm -f $WORK/COSY_READY
mkdir -p $WORK
nvidia-smi --query-gpu=name --format=csv,noheader || die "no GPU"

log "1/6 micromamba + python 3.10 + pynini(conda-forge)"
MM=/content/mt/bin/micromamba
if [ ! -x "$MM" ]; then
  mkdir -p $WORK/bin
  (cd $WORK && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba) || die "micromamba"
  MM=$WORK/bin/micromamba
fi
[ -x $ENV/bin/python ] || $MM create -y -q -p $ENV -c conda-forge python=3.10 pip "pynini==2.1.5" || die "env create"
PIP=$ENV/bin/pip; PY=$ENV/bin/python

log "2/6 torch 2.3.1 + cu121"
$PY -c 'import torch' 2>/dev/null || \
  $PIP install -q torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 || die "torch"

log "3/6 CosyVoice repo + requirements"
[ -d $REPO/.git ] || git clone --recursive https://github.com/FunAudioLLM/CosyVoice $REPO || die "clone"
# numpy 必须先装:否则一旦 requirements 中断,后续 modelscope 因缺 numpy 崩
# setuptools<80:新版移除了 pkg_resources,lightning(matcha 依赖)加载时需要它
$PIP install -q "numpy<2" scipy "setuptools<80" || die "numpy/setuptools"
# 剔掉编译易崩且推理不需要的 openai-whisper / deepspeed,避免中断整批安装
grep -v -iE 'openai-whisper|deepspeed' $REPO/requirements.txt > /tmp/cosy-req.txt
$PIP install -q -r /tmp/cosy-req.txt 2>&1 | tail -3 || log "requirements 部分失败,兜底补装"
# zero-shot 推理关键依赖兜底
$PIP install -q modelscope onnxruntime librosa soundfile hyperpyyaml omegaconf \
  conformer inflect gdown "diffusers==0.29.0" transformers || die "关键依赖补装失败"
# openai-whisper:CosyVoice2 加载模型时 import whisper;它在构建隔离下会崩,numpy 就绪后免隔离装
$PY -c "import whisper" 2>/dev/null || \
  $PIP install -q --no-build-isolation openai-whisper || \
  $PIP install -q openai-whisper || die "openai-whisper 安装失败"
$PY -c "import numpy, modelscope, whisper" || die "关键依赖仍不可用"

log "4/6 下载 CosyVoice2-0.5B 权重(真权重,约 2GB)"
[ -f $REPO/pretrained_models/CosyVoice2-0.5B/llm.pt ] && \
  [ $(stat -c%s $REPO/pretrained_models/CosyVoice2-0.5B/llm.pt) -gt 1000000 ] || \
  $PY -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='$REPO/pretrained_models/CosyVoice2-0.5B')" || die "model download"

log "5/6 TTS 封装脚本"
curl -fsSL $RAW/cosyvoice_tts.py -o $WORK/cosyvoice_tts.py || die "tts.py download"

log "6/6 冒烟:加载模型"
$PY -c "
import sys; sys.path.append('$REPO'); sys.path.append('$REPO/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
m=CosyVoice2('$REPO/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
print('model load ok, sr=', m.sample_rate)
" || die "model load failed"

echo "ready_at=$(date -u +%FT%TZ)" > $WORK/COSY_READY
log "DONE — COSY_READY written"
