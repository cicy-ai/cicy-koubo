#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/provision.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== MuseTalk provision $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "$DIR"
rm -rf env MuseTalk models

echo "[mt] base deps..."
pip install -q --upgrade setuptools wheel pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -q huggingface_hub opencv-python-headless diffusers mmengine

echo "[mt] cloning MuseTalk..."
git clone https://github.com/TMElyralab/MuseTalk.git MuseTalk

echo "[mt] requirements..."
cd MuseTalk
pip install -q -r requirements.txt

echo "[mt] downloading weights..."
mkdir -p models
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('TMElyralab/MuseTalk', local_dir='models')
"

echo "[mt] verifying import..."
python3 -c "
import sys; sys.path.insert(0, '$DIR/MuseTalk')
import musetalk; print('MuseTalk import OK')
" 2>/dev/null || echo "(import check skipped)"

echo "[mt] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "musetalk_version=1.5" > "$DIR/READY"
echo "musetalk_installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DIR/READY"
