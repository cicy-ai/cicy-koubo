#!/bin/bash
# CosyVoice 3.0 provision script for Colab/GPU
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/provision.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== CosyVoice provision started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Create venv if not exists
if [ ! -d "$DIR/env" ]; then
    echo "[cosy] creating python venv..."
    python3 -m venv "$DIR/env"
fi
source "$DIR/env/bin/activate"

# Clone CosyVoice if not exists
if [ ! -d "$DIR/CosyVoice" ]; then
    echo "[cosy] cloning CosyVoice..."
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$DIR/CosyVoice"
fi

# Install deps
echo "[cosy] installing dependencies..."
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121
cd "$DIR/CosyVoice"
pip install -q huggingface_hub modelscope onnxruntime soundfile
pip install -q -r requirements.txt 2>/dev/null || true
apt-get install -qq -y sox libsox-dev 2>/dev/null || true

# Download model
echo "[cosy] downloading CosyVoice3-0.5B model..."
MODEL_DIR="$DIR/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
if [ ! -f "$MODEL_DIR/llm.pt" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$MODEL_DIR', local_dir_use_symlinks=False)
" 2>&1
fi

# Write cosyvoice_tts.py wrapper
cat > "$DIR/cosyvoice_tts.py" << 'PYEOF'
#!/usr/bin/env python3
"""Simple TTS wrapper for CosyVoice — zero-shot voice cloning."""
import sys, os, io, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CosyVoice'))
os.environ['MODELSCOPE_CACHE'] = os.path.join(os.path.dirname(__file__), 'CosyVoice', 'pretrained_models')

parser = argparse.ArgumentParser()
parser.add_argument('--ref', required=True)
parser.add_argument('--ref-text-b64', default='')
parser.add_argument('--text-b64', required=True)
parser.add_argument('--speed', type=float, default=1.0)
parser.add_argument('--whole', action='store_true')
parser.add_argument('--out', required=True)
args = parser.parse_args()

import base64
ref_text = base64.b64decode(args.ref_text_b64).decode()
text = base64.b64decode(args.text_b64).decode()

from cosyvoice.cli.cosyvoice import AutoModel
ckpt = os.path.join(os.path.dirname(__file__), 'CosyVoice', 'pretrained_models', 'Fun-CosyVoice3-0.5B')
model = AutoModel(model_dir=ckpt)

outputs = []
for i, result in enumerate(model.inference_zero_shot(text, ref_text, args.ref)):
    tts_speech = result['tts_speech']
    outputs.append(tts_speech)
    if not args.whole:
        break

import numpy as np
final = np.concatenate(outputs)
import soundfile as sf
sf.write(args.out, final, 24000)
print(f"OK: {args.out}")
PYEOF
chmod +x "$DIR/cosyvoice_tts.py"

# Mark ready
echo "cosy_version=3.0" > "$DIR/COSY_READY"
echo "cosy_model=Fun-CosyVoice3-0.5B" >> "$DIR/COSY_READY"
echo "cosy_installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DIR/COSY_READY"

echo "=== CosyVoice provision done at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
