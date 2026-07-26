#!/bin/bash
# HeyGem(Linux Python 版,Holasyb918/HeyGem-Linux-Python-Hack)self-provision — EXPERIMENTAL。
# 显存要求高(≥16GB,T4 不够,A100/L4 可),与 MuseTalk/CosyVoice 独立环境,装在 /content/hg/。
#   curl -fsSL .../heygem-provision.sh > /content/hg/provision.sh
#   nohup bash /content/hg/provision.sh > /content/hg/provision.log 2>&1 &
# 成功后写 /content/hg/HG_READY。
set -uo pipefail
export LD_LIBRARY_PATH="/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg

WORK=/content/hg
ENV=$WORK/env
REPO=$WORK/HeyGem-Linux-Python-Hack
RAW=https://raw.githubusercontent.com/cicy-ai/cicy-tools/main
export MAMBA_ROOT_PREFIX=$WORK/mamba

log(){ echo "=== [$(date +%H:%M:%S)] $*"; }
die(){ echo "!!! $*" >&2; exit 1; }

rm -f $WORK/HG_READY
mkdir -p $WORK
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) || die "no GPU"
log "0/5 GPU ${GPU_MB}MiB"
[ "$GPU_MB" -ge 15000 ] || echo "WARN: 显存 <15GB,HeyGem 可能 OOM"

log "1/5 micromamba + python 3.8"
MM=/content/mt/bin/micromamba
[ -x "$MM" ] || MM=/content/cosy/bin/micromamba
if [ ! -x "$MM" ]; then
  mkdir -p $WORK/bin
  (cd $WORK && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba) || die "micromamba"
  MM=$WORK/bin/micromamba
fi
[ -x $ENV/bin/python ] || $MM create -y -q -p $ENV -c conda-forge python=3.8 pip || die "env create"
PIP=$ENV/bin/pip; PY=$ENV/bin/python

log "2/5 HeyGem repo"
[ -d $REPO/.git ] || git clone -q --depth 1 https://github.com/Holasyb918/HeyGem-Linux-Python-Hack $REPO || die "clone"

log "3/5 requirements(cuda11.8 + onnxruntime-gpu 1.16)"
cd $REPO
# requirements 钉死的 onnxruntime-gpu==1.9.0 已下架,装到那行整批中断 → 剔掉后再装
grep -viE "onnxruntime" requirements.txt > /tmp/hg_req.txt
$PIP install -q -r /tmp/hg_req.txt 2>&1 | tail -3 || echo "requirements 部分失败,继续"
$PIP install -q "onnxruntime-gpu==1.16.0" typeguard opencv-python-headless \
  librosa soundfile tqdm flask 2>&1 | tail -1 || true

log "4/5 模型权重(download.sh)"
bash download.sh 2>&1 | tail -5 || die "model download"
# 多人脸检测兜底模型
if [ ! -f face_detect_utils/resources/.scrfd10g ]; then
  curl -fsSL -o /tmp/scrfd_10g_kps.onnx \
    https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/scrfd_10g_kps.onnx \
    && cp -f /tmp/scrfd_10g_kps.onnx face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx \
    && touch face_detect_utils/resources/.scrfd10g || echo "WARN: scrfd 替换失败(多人脸场景可能报错)"
fi

log "5/5 onnx/cuda 自检 + 合成封装"
$PY check_env/check_onnx_cuda.py 2>&1 | tail -2 || echo "WARN: onnx cuda 自检未过,首次合成时再定位"
curl -fsSL $RAW/heygem-synthesize.sh -o $WORK/synthesize.sh && chmod +x $WORK/synthesize.sh || die "synthesize wrapper"

cat > $WORK/HG_READY <<EOF
provisioned_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
gpu_mb=$GPU_MB
python=3.8 onnxruntime-gpu=1.16.0
note=experimental
EOF
log "DONE — HG_READY written"
