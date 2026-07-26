#!/bin/bash
# MuseTalk provision script for Colab/GPU
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/provision.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== MuseTalk provision started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "$DIR"

# Create venv
if [ ! -d "$DIR/env" ]; then
    echo "[mt] creating python 3.10 venv..."
    python3 -m venv "$DIR/env" --without-pip 2>/dev/null || python3 -m venv "$DIR/env"
fi
source "$DIR/env/bin/activate"

# Upgrade pip
pip install -q --upgrade pip

# Clone MuseTalk
if [ ! -d "$DIR/MuseTalk" ]; then
    echo "[mt] cloning MuseTalk..."
    git clone https://github.com/TMElyralab/MuseTalk.git "$DIR/MuseTalk"
fi
cd "$DIR/MuseTalk"

# Install PyTorch 2.0.1 + CUDA 11.8
echo "[mt] installing PyTorch..."
pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install deps
echo "[mt] installing dependencies..."
pip install -q -r requirements.txt

# Install MMLab
echo "[mt] installing mm lab packages..."
pip install -q --no-cache-dir -U openmim
mim install -q mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"

# Download weights
echo "[mt] downloading model weights..."
mkdir -p models
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('TMElyralab/MuseTalk', local_dir='models', local_dir_use_symlinks=False)
" 2>&1

echo "[mt] provision done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "musetalk_version=1.5" > "$DIR/READY"
echo "musetalk_installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DIR/READY"
echo "OK"
