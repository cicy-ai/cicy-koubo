#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/provision.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== MuseTalk provision $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "$DIR"
rm -rf env

if [ ! -d MuseTalk ]; then
    echo "[mt] cloning..."
    git clone https://github.com/TMElyralab/MuseTalk.git MuseTalk
fi
cd MuseTalk

echo "[mt] deps..."
pip install -q huggingface_hub opencv-python diffusers mmengine
pip install -q -r requirements.txt 2>/dev/null || true

echo "[mt] weights..."
mkdir -p models
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('TMElyralab/MuseTalk', local_dir='models', local_dir_use_symlinks=False)
"

echo "[mt] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "musetalk_version=1.5" > "$DIR/READY"
echo "musetalk_installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DIR/READY"
