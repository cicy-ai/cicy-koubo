#!/bin/bash
# MuseTalk 1.5 self-provision for a Google Colab GPU runtime.
# Curled + launched by the Colab bootstrap cell right after the frp tunnel:
#   curl -fsSL https://raw.githubusercontent.com/cicy-ai/cicy-tools/main/musetalk-provision.sh \
#     > /content/mt/provision.sh && nohup bash /content/mt/provision.sh > /content/mt/provision.log 2>&1 &
# Idempotent — re-run any time to repair or pick up script updates.
# Writes /content/mt/READY (gpu + versions + smoke-test timing) only after the
# official-sample smoke test passes, so orchestrators can poll for readiness.
set -uo pipefail

# Colab 容器的 NVIDIA 驱动库不在默认搜索路径;SSH 会话没有 notebook 的 env,必须显式补上
export LD_LIBRARY_PATH="/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
# 从笔记本 cell 启动会继承 Jupyter 的 MPLBACKEND=inline,mmpose 导入 matplotlib 会崩;强制无头后端
export MPLBACKEND=Agg

WORK=/content/mt
ENV=$WORK/env
REPO=$WORK/MuseTalk
RAW=https://raw.githubusercontent.com/cicy-ai/cicy-tools/main
export MAMBA_ROOT_PREFIX=$WORK/mamba

log() { echo "=== [$(date +%H:%M:%S)] $*"; }
die() { echo "!!! $*" >&2; exit 1; }

rm -f $WORK/READY
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) || die "no GPU on this runtime"
log "GPU: $GPU"

log "1/7 micromamba + python 3.10 env"
mkdir -p $WORK
if [ ! -x $WORK/bin/micromamba ]; then
  (cd $WORK && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba) || die "micromamba download failed"
fi
if [ ! -x $ENV/bin/python ]; then
  $WORK/bin/micromamba create -y -q -p $ENV -c conda-forge python=3.10 pip || die "env create failed"
fi
PIP="$ENV/bin/pip"
PY="$ENV/bin/python"

log "2/7 torch 2.0.1 + cu118"
$PY -c 'import torch; assert torch.__version__.startswith("2.0.1")' 2>/dev/null || \
  $PIP install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118 || die "torch install failed"

log "3/7 MuseTalk repo + requirements"
[ -d $REPO/.git ] || git clone --depth 1 https://github.com/TMElyralab/MuseTalk $REPO || die "clone failed"
$PIP install -q -r $REPO/requirements.txt || die "requirements failed"
$PIP install -q -U openmim || die "openmim failed"

log "4/7 mmlab stack (mmcv 2.0.1 / mmdet 3.1.0 / mmpose 1.1.0)"
# chumpy(mmpose 传递依赖)的 setup.py 在新 pip 的隔离构建下崩;用环境内 numpy 免隔离预装
$PY -c 'import chumpy' 2>/dev/null || {
  $PIP install -q "setuptools<70" wheel cython
  $PIP install -q --no-build-isolation chumpy==0.70 || die "chumpy failed"
}
$PY -c 'import mmpose' 2>/dev/null || \
  $ENV/bin/mim install -q mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0" || die "mmlab failed"

log "5/7 model weights (direct HuggingFace)"
# hub 必须锁 0.30.2:1.x 移除了 huggingface-cli,且 transformers 4.39 要求 hub<1.0
$PIP install -q "huggingface_hub[cli]==0.30.2" gdown
M=$REPO/models
mkdir -p $M/musetalkV15 $M/syncnet $M/dwpose $M/face-parse-bisent $M/sd-vae $M/whisper
HF=$ENV/bin/huggingface-cli
[ -f $M/musetalkV15/unet.pth ] || $HF download TMElyralab/MuseTalk --local-dir $M \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth" || die "musetalk weights failed"
[ -f $M/sd-vae/diffusion_pytorch_model.bin ] || $HF download stabilityai/sd-vae-ft-mse --local-dir $M/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin" || die "sd-vae failed"
[ -f $M/whisper/pytorch_model.bin ] || $HF download openai/whisper-tiny --local-dir $M/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json" || die "whisper failed"
[ -f $M/dwpose/dw-ll_ucoco_384.pth ] || $HF download yzd-v/DWPose --local-dir $M/dwpose \
  --include "dw-ll_ucoco_384.pth" || die "dwpose failed"
[ -f $M/syncnet/latentsync_syncnet.pt ] || $HF download ByteDance/LatentSync --local-dir $M/syncnet \
  --include "latentsync_syncnet.pt" || die "syncnet failed"
[ -f $M/face-parse-bisent/79999_iter.pth ] || $ENV/bin/gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
  -O $M/face-parse-bisent/79999_iter.pth || die "face-parse failed"
[ -f $M/face-parse-bisent/resnet18-5c106cde.pth ] || curl -sL https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $M/face-parse-bisent/resnet18-5c106cde.pth || die "resnet18 failed"

log "6/7 ffmpeg + synthesize wrapper (latest from repo)"
command -v ffmpeg >/dev/null || (apt-get -qq update && apt-get -qq install -y ffmpeg)
curl -fsSL $RAW/musetalk-synthesize.sh > $WORK/synthesize.sh || die "synthesize.sh download failed"
chmod +x $WORK/synthesize.sh
$PY -c "import torch, mmpose, diffusers; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())" \
  || die "sanity import failed"

log "7/7 smoke test (official sample)"
START=$(date +%s)
bash $WORK/synthesize.sh > $WORK/smoke.log 2>&1 || { tail -20 $WORK/smoke.log; die "smoke test failed (see $WORK/smoke.log)"; }
SMOKE=$(( $(date +%s) - START ))

{
  echo "gpu=$GPU"
  echo "provisioned_at=$(date -u +%FT%TZ)"
  echo "smoke_wall_s=$SMOKE"
  $PY -c "import torch; print('torch='+torch.__version__)"
} > $WORK/READY
log "DONE — READY written. smoke=${SMOKE}s"
