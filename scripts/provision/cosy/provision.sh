#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/provision.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== CosyVoice provision $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "$DIR"
rm -rf env

if [ ! -d CosyVoice ]; then
    echo "[cosy] cloning..."
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git CosyVoice
fi

echo "[cosy] deps..."
pip install -q huggingface_hub modelscope onnxruntime soundfile
cd CosyVoice
pip install -q -r requirements.txt 2>/dev/null || true

echo "[cosy] model..."
MODEL_DIR="$DIR/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
if [ ! -f "$MODEL_DIR/llm.pt" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$MODEL_DIR')
" 2>&1
fi

# Write TTS wrapper
cat > "$DIR/cosyvoice_tts.py" << 'PYEOF'
#!/usr/bin/env python3
"""CosyVoice zero-shot TTS wrapper."""
import sys, os, argparse, base64, numpy as np, soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CosyVoice'))
os.environ['MODELSCOPE_CACHE'] = os.path.join(os.path.dirname(__file__), 'CosyVoice', 'pretrained_models')

parser = argparse.ArgumentParser()
parser.add_argument('--ref', required=True)
parser.add_argument('--ref-text-b64', default='')
parser.add_argument('--text-b64', required=True)
parser.add_argument('--speed', type=float, default=1.0)
parser.add_argument('--out', required=True)
args = parser.parse_args()

ref_text = base64.b64decode(args.ref_text_b64).decode()
text = base64.b64decode(args.text_b64).decode()

from cosyvoice.cli.cosyvoice import AutoModel
ckpt = os.path.join(os.path.dirname(__file__), 'CosyVoice', 'pretrained_models', 'Fun-CosyVoice3-0.5B')
model = AutoModel(model_dir=ckpt)
result = list(model.inference_zero_shot(text, ref_text, args.ref))[0]
sf.write(args.out, result['tts_speech'], 24000)
print(f"OK: {args.out}")
PYEOF
chmod +x "$DIR/cosyvoice_tts.py"

echo "[cosy] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cosy_version=3.0" > "$DIR/COSY_READY"
echo "cosy_model=Fun-CosyVoice3-0.5B" >> "$DIR/COSY_READY"
echo "cosy_installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DIR/COSY_READY"
